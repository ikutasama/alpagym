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
    "dp_shard_size",
    "alpasim_topology",
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
    dp_shard_size: int
    alpasim_topology: str
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
            # Colocated: policy+rollout share one FSDP model.  When
            # dp_shard_size=1 the model lives on a single GPU; when >1
            # FSDP shards it across that many GPUs, so the GPU count
            # must equal dp_shard_size.
            if (
                self.geometry.policy_replicas != 1
                or self.geometry.rollout_replicas != 1
            ):
                raise ValueError(
                    "The supported colocated profile requires one policy and one "
                    "rollout replica sharing the model"
                )
            expected_gpu_count = self.dp_shard_size
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
        dp_shard_size=_require_int(raw, "dp_shard_size"),
        alpasim_topology=_require_str(raw, "alpasim_topology"),
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
            {"kind": "metric", "metric_name": "progress", "scale": 5.0},
            {"kind": "metric", "metric_name": "collision_any", "scale": -2.0},
            {"kind": "metric", "metric_name": "offroad", "scale": -1.0},
            {"kind": "distance_to_gt", "scale": -0.01},
        ]
    }


    cosmos = _mapping(config, "cosmos")
    cosmos["mode"] = profile.mode

    launch = _mapping(cosmos, "launch")
    launch["policy_replicas"] = geometry.policy_replicas
    launch["rollout_replicas"] = geometry.rollout_replicas

    policy_parallelism = _mapping(_mapping(cosmos, "policy"), "parallelism")
    policy_parallelism["dp_shard_size"] = profile.dp_shard_size

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
            "prefetch_rollout": profile.mode == "disaggregated",
        }
    )
    _mapping(cosmos, "logging")["experiment_name"] = geometry.experiment_name

    inference = _mapping(_mapping(config, "policy"), "inference")
    inference["max_batch_size"] = geometry.max_inference_batch_size

    sampling = _mapping(inference, "sampling")
    sampling["temperature"] = 0.5

    # Fix step_dt_us to match AutoVLA's 0.5s trajectory interval (10 poses × 0.5s = 5s).
    # The wizard default is 100000 (100ms) which compresses 5s trajectory into 1s,
    # causing 5x speed and gRPC timeouts.
    policy_model = _mapping(_mapping(config, "policy"), "model")
    policy_model["step_dt_us"] = 500000

    # Override SFT checkpoint path to use warmup checkpoint (with history
    # waypoints). The launch script's sed replacement may not match all
    # path formats, so we set it directly here.
    bundle_config = _mapping(policy_model, "bundle_config")
    bundle_config["checkpoint_path"] = "/tmp/model/AutoVLA/AutoVLA_PDMS_89.ckpt"
    bundle_config["use_cot"] = True

    # Ego history must be collected at 0.5s intervals (interval_length) to
    # match AutoVLA's SFT training. pose_reporting_interval_us=500000 in
    # extra_overrides ensures the sim reports one pose per 0.5s.
    # control_timestep=100ms (divides evenly into 500ms), force_gt=8.0s
    # (16 warmup poses × 0.5s), expected_valid_steps=24 →
    # n_sim_steps = 24 + 80 = 104, total = 104 × 0.1s = 10.4s.
    alpasim = _mapping(config, "alpasim")
    alpasim["simulation_timeout_s"] = 3600.0
    alpasim["repo_path"] = "/data/mnt_m62/10_personal/z59900495/workspace/alpasim"
    alpasim["repo_url"] = None
    alpasim["repo_ref"] = None
    wizard = _mapping(alpasim, "wizard_args")
    wizard["topology"] = profile.alpasim_topology
    wizard["control_timestep_us"] = 100000
    wizard["force_gt_duration_us"] = 8000000
    wizard["n_sim_steps"] = 104

    # Match expected_valid_steps to n_sim_steps - warmup_steps.
    # warmup_steps = force_gt_duration_us / control_timestep_us = 80.
    config["expected_valid_steps"] = 24
    wizard["extra_overrides"] = (
        "+cameras=3cam_1080"
        " runtime.simulation_config.pose_reporting_interval_us=500000"
        " scenes.local_usdz_dir=/data/mnt_m62/10_personal/z59900495/workspace/DownloadTool-master/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/sample_set/26.02_release"
    )


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
    policy_parallelism["dp_shard_size"] = profile.dp_shard_size
    rollout_parallelism = _mapping(_mapping(config, "rollout"), "parallelism")
    rollout_parallelism["dp_shard_size"] = profile.dp_shard_size

    # Gradient checkpointing trades compute for memory. In colocated mode
    # the DAgger SFT forward adds a second forward on top of GRPO's, so
    # both sets of activations need checkpointing to fit in 80GB alongside
    # vLLM. In disaggregated mode the policy GPU has full 80GB to itself,
    # so checkpointing is unnecessary and only slows training.
    policy = _mapping(config, "policy")
    # Gradient checkpointing trades ~30% step time for 3-5x smaller
    # activation memory. Disaggregated DAgger OOM'd without it: the
    # uncheckpointed training forward + reference forward spiked +34GB
    # on top of the 45.5GB steady state (VRAM-GUARD evidence), blowing
    # the 79GB card. Host-RAM offload is not viable on this shared box
    # (3 replicas need ~150GB host RAM; only ~43GB available).
    policy["model_gradient_checkpointing"] = True

    train_config = _mapping(config, "train")
    train_config["fsdp_reshard_after_forward"] = "never"
    # FSDP CPU offload DISABLED: moving optimizer states to host RAM caused
    # 3 replicas x ~87GB = ~261GB CPU RAM, triggering kernel OOM kills.
    # The 3B model (~7.6GB bf16) + AdamW states (~45GB total) fits in 80GB
    # GPU VRAM without offload. Gradient checkpointing bounds activation spikes.
    train_config["fsdp_offload"] = False

    rollout = _mapping(config, "rollout")
    is_colocated = profile.mode == "colocated"
    rollout.update(
        {
            "n_generation": geometry.n_generation,
            "batch_size": geometry.rollout_batch_size,
            "prefetch_rollout": profile.mode == "disaggregated",
            # Colocated: vLLM shares the GPU with FSDP training, so it must
            # leave room for training activations (~44GB).
            # Disaggregated: vLLM has a dedicated GPU (possibly sharing with
            # the expert service at ~22GB), so 0.5 gives ~40GB—plenty for
            # the 3B model + KV cache while leaving room for the expert.
            "gpu_memory_utilization": 0.45 if is_colocated else 0.5,
        }
    )
    rollout["sampling_config"] = {
        "temperature": 0.5,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
    }
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
                    str(profile.dp_shard_size),
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
