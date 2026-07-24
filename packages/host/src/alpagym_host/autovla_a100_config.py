# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure copied AutoVLA run artifacts from a validated A100 profile."""

from __future__ import annotations

import argparse
import shutil
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import tomli_w
import yaml

_PROFILE_KEYS = {
    "name",
    "cuda_visible_devices",
    "mode",
    "transport",
    "policy_replicas",
    "rollout_replicas",
    "n_generation",
    "rollout_batch_size",
    "train_batch_per_replica",
    "mini_batch",
    "max_inference_batch_size",
    "max_num_steps",
    "num_epochs",
    "learning_rate",
    "warmup_steps",
    "ratio_clip",
    "allowed_outdated_steps",
    "save_freq",
    "grad_norm_clip",
}


@dataclass(frozen=True)
class A100TrainGeometry:
    """Coupled rollout and policy settings for an A100 process layout."""

    policy_replicas: int
    rollout_replicas: int
    n_generation: int
    rollout_batch_size: int
    train_batch_per_replica: int
    mini_batch: int
    max_inference_batch_size: int
    max_num_steps: int
    num_epochs: int
    learning_rate: float
    warmup_steps: int
    ratio_clip: float
    allowed_outdated_steps: int
    save_freq: int
    grad_norm_clip: float
    experiment_name: str

    def validate(self) -> None:
        """Reject geometry that would produce partial or uneven policy batches."""
        positive_ints = {
            "policy_replicas": self.policy_replicas,
            "rollout_replicas": self.rollout_replicas,
            "n_generation": self.n_generation,
            "rollout_batch_size": self.rollout_batch_size,
            "train_batch_per_replica": self.train_batch_per_replica,
            "mini_batch": self.mini_batch,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_num_steps": self.max_num_steps,
            "num_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "save_freq": self.save_freq,
        }
        invalid = {name: value for name, value in positive_ints.items() if value <= 0}
        if invalid:
            raise ValueError(f"A100 training values must be positive: {invalid}")
        if self.rollout_replicas != 1:
            raise ValueError(
                "The current 5012 SSH tunnel exposes one Egodriver port, so this "
                "profile requires rollout_replicas=1. Add one driver port/tunnel per "
                "rollout replica before increasing it."
            )
        if self.train_batch_per_replica % self.mini_batch != 0:
            raise ValueError(
                "train_batch_per_replica must be divisible by mini_batch; got "
                f"{self.train_batch_per_replica} and {self.mini_batch}"
            )
        produced = self.rollout_replicas * self.rollout_batch_size * self.n_generation
        consumed = self.policy_replicas * self.train_batch_per_replica
        if produced != consumed:
            raise ValueError(
                "Global rollout and policy batch geometry must match: "
                f"rollout_replicas({self.rollout_replicas}) * "
                f"rollout_batch_size({self.rollout_batch_size}) * "
                f"n_generation({self.n_generation}) = {produced}, but "
                f"policy_replicas({self.policy_replicas}) * "
                f"train_batch_per_replica({self.train_batch_per_replica}) = {consumed}"
            )
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if not 0.0 < self.ratio_clip < 1.0:
            raise ValueError(f"ratio_clip must be in (0, 1), got {self.ratio_clip}")
        if self.allowed_outdated_steps < 0:
            raise ValueError(
                "allowed_outdated_steps must be non-negative, got "
                f"{self.allowed_outdated_steps}"
            )
        if self.grad_norm_clip <= 0.0:
            raise ValueError(
                f"grad_norm_clip must be positive, got {self.grad_norm_clip}"
            )


@dataclass(frozen=True)
class A100LaunchProfile:
    """Validated GPU layout plus the training geometry written into a run."""

    name: str
    cuda_visible_devices: str
    mode: str
    transport: str
    geometry: A100TrainGeometry

    @property
    def gpu_ids(self) -> tuple[str, ...]:
        """Return the CUDA device identifiers selected by this profile."""
        return tuple(
            device.strip()
            for device in self.cuda_visible_devices.split(",")
            if device.strip()
        )

    def validate(self) -> None:
        """Validate process placement, transport, and training geometry together."""
        self.geometry.validate()
        if not self.name.strip():
            raise ValueError("A100 profile name must not be empty")
        if self.mode not in {"colocated", "disaggregated"}:
            raise ValueError(
                "A100 profile mode must be 'colocated' or 'disaggregated', got "
                f"{self.mode!r}"
            )
        if self.transport not in {"disk", "nccl"}:
            raise ValueError(
                "A100 profile transport must be 'disk' or 'nccl', got "
                f"{self.transport!r}"
            )
        if self.transport == "nccl" and self.mode != "disaggregated":
            raise ValueError("transport=nccl requires mode=disaggregated")
        if not self.gpu_ids:
            raise ValueError("cuda_visible_devices must select at least one GPU")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError(
                f"cuda_visible_devices contains duplicate IDs: {self.gpu_ids}"
            )

        if self.mode == "disaggregated":
            expected_gpu_count = (
                self.geometry.policy_replicas + self.geometry.rollout_replicas
            )
        else:
            if (
                self.geometry.policy_replicas != 1
                or self.geometry.rollout_replicas != 1
            ):
                raise ValueError(
                    "The supported colocated profile requires one policy and one "
                    "rollout replica sharing one GPU"
                )
            expected_gpu_count = 1
        if len(self.gpu_ids) != expected_gpu_count:
            raise ValueError(
                f"mode={self.mode} with policy={self.geometry.policy_replicas} and "
                f"rollout={self.geometry.rollout_replicas} requires "
                f"{expected_gpu_count} visible GPU(s), but cuda_visible_devices="
                f"{self.cuda_visible_devices!r} selects {len(self.gpu_ids)}"
            )


def load_a100_profile(path: Path) -> A100LaunchProfile:
    """Load a complete A100 launch profile and reject missing or unknown fields."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    missing = sorted(_PROFILE_KEYS - raw.keys())
    unknown = sorted(raw.keys() - _PROFILE_KEYS)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(f"Invalid A100 profile {path}: {', '.join(details)}")

    profile = A100LaunchProfile(
        name=_require_str(raw, "name"),
        cuda_visible_devices=_require_str(raw, "cuda_visible_devices"),
        mode=_require_str(raw, "mode"),
        transport=_require_str(raw, "transport"),
        geometry=A100TrainGeometry(
            policy_replicas=_require_int(raw, "policy_replicas"),
            rollout_replicas=_require_int(raw, "rollout_replicas"),
            n_generation=_require_int(raw, "n_generation"),
            rollout_batch_size=_require_int(raw, "rollout_batch_size"),
            train_batch_per_replica=_require_int(raw, "train_batch_per_replica"),
            mini_batch=_require_int(raw, "mini_batch"),
            max_inference_batch_size=_require_int(raw, "max_inference_batch_size"),
            max_num_steps=_require_int(raw, "max_num_steps"),
            num_epochs=_require_int(raw, "num_epochs"),
            learning_rate=_require_number(raw, "learning_rate"),
            warmup_steps=_require_int(raw, "warmup_steps"),
            ratio_clip=_require_number(raw, "ratio_clip"),
            allowed_outdated_steps=_require_int(raw, "allowed_outdated_steps"),
            save_freq=_require_int(raw, "save_freq"),
            grad_norm_clip=_require_number(raw, "grad_norm_clip"),
            experiment_name=_require_str(raw, "name"),
        ),
    )
    profile.validate()
    return profile


def configure_autovla_a100_run(run_dir: Path, profile: A100LaunchProfile) -> None:
    """Patch both config sources consumed by Cosmos and the AlpaGym backend."""
    profile.validate()
    resolved_path = run_dir / "resolved_config.yaml"
    cosmos_path = run_dir / "cosmos_config.toml"
    if not resolved_path.is_file() or not cosmos_path.is_file():
        raise FileNotFoundError(
            f"Expected resolved_config.yaml and cosmos_config.toml under {run_dir}"
        )

    _backup_once(resolved_path)
    _backup_once(cosmos_path)

    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(resolved, dict):
        raise TypeError(f"Expected a YAML mapping in {resolved_path}")
    cosmos_config = tomllib.loads(cosmos_path.read_text(encoding="utf-8"))

    _update_resolved_config(resolved, profile)
    _update_cosmos_config(cosmos_config, profile)

    _atomic_write(
        resolved_path,
        yaml.safe_dump(resolved, sort_keys=False, default_flow_style=False),
    )
    _atomic_write(cosmos_path, tomli_w.dumps(cosmos_config))


def _update_resolved_config(config: dict[str, Any], profile: A100LaunchProfile) -> None:
    """Update values read by AlpaGym rollout, packer, and transport code."""
    geometry = profile.geometry
    transport = _mapping(config, "transport")
    transport.update(
        {
            "kind": profile.transport,
            "nccl_env": _nccl_env() if profile.transport == "nccl" else {},
            "nccl_read_device": "cpu",
        }
    )

    # Always use progress_safety reward for closed-loop GRPO: rewards progress,
    # penalizes collision/offroad, plus a small GT-deviation term.
    config["reward"] = {
        "terms": [
            {"kind": "metric", "metric_name": "progress", "scale": 1.0},
            {"kind": "metric", "metric_name": "collision_any", "scale": -10.0},
            {"kind": "metric", "metric_name": "offroad", "scale": -5.0},
            {"kind": "distance_to_gt", "scale": -0.01},
        ]
    }

    # Use the latest SFT checkpoint as GRPO starting point (better than original AutoVLA_PDMS_89)
    model_cfg = _mapping(config, "model")
    model_cfg["sft_model_path"] = "/tmp/model/AutoVLA/autovla_sft_step10000.ckpt"


    cosmos = _mapping(config, "cosmos")
    cosmos["mode"] = profile.mode

    launch = _mapping(cosmos, "launch")
    launch["policy_replicas"] = geometry.policy_replicas
    launch["rollout_replicas"] = geometry.rollout_replicas

    policy_parallelism = _mapping(_mapping(cosmos, "policy"), "parallelism")
    policy_parallelism["dp_shard_size"] = 1

    train = _mapping(cosmos, "train")
    train.update(
        {
            "train_batch_per_replica": geometry.train_batch_per_replica,
            "max_num_steps": geometry.max_num_steps,
            "num_epochs": geometry.num_epochs,
            "optm_lr": geometry.learning_rate,
            "optm_warmup_steps": geometry.warmup_steps,
            "optm_decay_type": "cosine",
            "optm_decay_ratio": 1.0,
            "optm_min_lr_factor": 0.1,
        }
    )
    checkpoint = _mapping(train, "ckpt")
    checkpoint.update(
        {
            "enable_checkpoint": True,
            "save_freq": geometry.save_freq,
            "export_safetensors": False,
            "max_keep": 3,
        }
    )
    train_policy = _mapping(train, "train_policy")
    train_policy.update(
        {
            "allowed_outdated_steps": geometry.allowed_outdated_steps,
            "on_policy": False,
            "mini_batch": geometry.mini_batch,
            "grpo_ratio_clip_low": geometry.ratio_clip,
            "grpo_ratio_clip_high": geometry.ratio_clip,
            "grpo_optimization_iterations": 1,
            "kl_beta": 0.0,
            "reference_reset_interval": 0,
        }
    )

    rollout = _mapping(cosmos, "rollout")
    rollout.update(
        {
            "n_generation": geometry.n_generation,
            "batch_size": geometry.rollout_batch_size,
            "prefetch_rollout": False,
        }
    )
    _mapping(cosmos, "logging")["experiment_name"] = geometry.experiment_name

    inference = _mapping(_mapping(config, "policy"), "inference")
    inference["max_batch_size"] = geometry.max_inference_batch_size


def _update_cosmos_config(config: dict[str, Any], profile: A100LaunchProfile) -> None:
    """Update values parsed directly by Cosmos-RL workers."""
    geometry = profile.geometry
    config["mode"] = profile.mode
    train = _mapping(config, "train")
    train.update(
        {
            "train_batch_per_replica": geometry.train_batch_per_replica,
            "max_num_steps": geometry.max_num_steps,
            "epoch": geometry.num_epochs,
            "optm_lr": geometry.learning_rate,
            "optm_warmup_steps": geometry.warmup_steps,
            "optm_decay_type": "cosine",
            "optm_decay_ratio": 1.0,
            "optm_min_lr_factor": 0.1,
            "optm_grad_norm_clip": geometry.grad_norm_clip,
        }
    )
    checkpoint = _mapping(train, "ckpt")
    checkpoint.update(
        {
            "enable_checkpoint": True,
            "save_freq": geometry.save_freq,
            "export_safetensors": False,
            "max_keep": 3,
        }
    )
    train_policy = _mapping(train, "train_policy")
    train_policy.update(
        {
            "allowed_outdated_steps": geometry.allowed_outdated_steps,
            "on_policy": False,
            "mini_batch": geometry.mini_batch,
            "epsilon_low": geometry.ratio_clip,
            "epsilon_high": geometry.ratio_clip,
            "mu_iterations": 1,
            "kl_beta": 0.0,
            "reference_reset_interval": 0,
        }
    )

    policy_parallelism = _mapping(_mapping(config, "policy"), "parallelism")
    policy_parallelism["dp_shard_size"] = 1
    rollout = _mapping(config, "rollout")
    rollout.update(
        {
            "n_generation": geometry.n_generation,
            "batch_size": geometry.rollout_batch_size,
            "prefetch_rollout": False,
        }
    )
    _mapping(config, "logging")["experiment_name"] = geometry.experiment_name


def _nccl_env() -> dict[str, str]:
    return {
        "NCCL_SHM_DISABLE": "0",
        "NCCL_DEBUG": "WARN",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_IFNAME": "^lo,^docker",
        "NCCL_TIMEOUT": "1800",
        "NCCL_P2P_DISABLE": "0",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }


def _require_int(config: dict[str, Any], key: str) -> int:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"A100 profile field {key!r} must be an integer, got {value!r}")
    return value


def _require_str(config: dict[str, Any], key: str) -> str:
    value = config[key]
    if not isinstance(value, str):
        raise TypeError(f"A100 profile field {key!r} must be a string, got {value!r}")
    return value


def _require_number(config: dict[str, Any], key: str) -> float:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"A100 profile field {key!r} must be numeric, got {value!r}")
    return float(value)


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise TypeError(
            f"Expected config field {key!r} to be a mapping, got {type(value).__name__}"
        )
    return value


def _backup_once(path: Path) -> None:
    backup_path = path.with_name(f"{path.name}.pre_autovla_a100")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def _atomic_write(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--print-launch-fields",
        action="store_true",
        help="Print tab-separated launcher fields without modifying a run",
    )
    parser.add_argument("--max-num-steps", type=int)
    parser.add_argument("--save-freq", type=int)
    args = parser.parse_args(argv)
    if args.run_dir is None and not args.print_launch_fields:
        parser.error("--run-dir is required unless --print-launch-fields is used")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    profile = load_a100_profile(args.profile.resolve())
    geometry = profile.geometry
    if args.max_num_steps is not None:
        geometry = replace(geometry, max_num_steps=args.max_num_steps)
    if args.save_freq is not None:
        geometry = replace(geometry, save_freq=args.save_freq)
    profile = replace(profile, geometry=geometry)
    profile.validate()

    if args.print_launch_fields:
        print(
            "\t".join(
                (
                    profile.cuda_visible_devices,
                    str(geometry.policy_replicas),
                    str(geometry.rollout_replicas),
                    profile.mode,
                    profile.transport,
                    profile.name,
                )
            )
        )
        if args.run_dir is None:
            return

    configure_autovla_a100_run(args.run_dir.resolve(), profile)
    produced = (
        geometry.rollout_replicas * geometry.rollout_batch_size * geometry.n_generation
    )
    print(
        f"Configured AutoVLA A100 profile {profile.name!r}: "
        f"mode={profile.mode}, transport={profile.transport}, "
        f"policy={geometry.policy_replicas}, rollout={geometry.rollout_replicas}, "
        f"global_episodes={produced}, n_generation={geometry.n_generation}, "
        f"train_batch_per_replica={geometry.train_batch_per_replica}, "
        f"mini_batch={geometry.mini_batch}, lr={geometry.learning_rate:g}"
    )


if __name__ == "__main__":
    main()
