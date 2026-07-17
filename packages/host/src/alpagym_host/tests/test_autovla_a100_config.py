# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the profile-driven AutoVLA A100 copied-run configurator."""

import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
import tomli_w
import yaml
from alpagym_host.autovla_a100_config import (
    A100LaunchProfile,
    configure_autovla_a100_run,
    load_a100_profile,
    main,
)

PROFILE_DIR = (
    Path(__file__).resolve().parents[5]
    / "packages/policies/autovla/src/alpagym_autovla/configs/a100"
)


def _load_profile(gpu_count: int) -> A100LaunchProfile:
    return load_a100_profile(PROFILE_DIR / f"autovla_a100_{gpu_count}gpu.yaml")


def _write_run_configs(run_dir: Path) -> None:
    resolved = {
        "policy": {"inference": {"max_batch_size": 1}},
        "transport": {"kind": "disk"},
        "cosmos": {
            "mode": "colocated",
            "launch": {"policy_replicas": 1, "rollout_replicas": 1},
            "policy": {"parallelism": {"dp_shard_size": 1}},
            "train": {
                "train_batch_per_replica": 1,
                "max_num_steps": 1,
                "num_epochs": 1,
                "optm_lr": 1.0e-5,
                "optm_warmup_steps": 1,
                "ckpt": {},
                "train_policy": {},
            },
            "rollout": {"n_generation": 2, "batch_size": 1},
            "logging": {"experiment_name": "smoke"},
        },
    }
    cosmos = {
        "mode": "colocated",
        "train": {"ckpt": {}, "train_policy": {}},
        "policy": {"parallelism": {"dp_shard_size": 1}},
        "rollout": {"n_generation": 2, "batch_size": 1},
        "logging": {"experiment_name": "smoke"},
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved), encoding="utf-8"
    )
    (run_dir / "cosmos_config.toml").write_text(tomli_w.dumps(cosmos), encoding="utf-8")


def test_load_four_gpu_profile_has_matched_global_episode_count() -> None:
    profile = _load_profile(4)

    assert profile.mode == "disaggregated"
    assert profile.transport == "nccl"
    assert profile.gpu_ids == ("0", "1", "2", "3")
    assert profile.geometry.policy_replicas == 3
    assert profile.geometry.rollout_batch_size == 3
    assert profile.geometry.n_generation == 8


def test_load_one_gpu_profile_keeps_complete_grpo_group() -> None:
    profile = _load_profile(1)

    assert profile.mode == "colocated"
    assert profile.transport == "disk"
    assert profile.gpu_ids == ("0",)
    assert profile.geometry.policy_replicas == 1
    assert profile.geometry.rollout_batch_size == 1
    assert profile.geometry.n_generation == 8
    assert profile.geometry.train_batch_per_replica == 8
    assert profile.geometry.mini_batch == 1


def test_print_launch_fields_is_stable_for_shell_launcher(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "--profile",
            str(PROFILE_DIR / "autovla_a100_1gpu.yaml"),
            "--print-launch-fields",
        ]
    )

    assert capsys.readouterr().out == (
        "0\t1\t1\tcolocated\tdisk\tautovla_a100_1gpu_grpo\n"
    )


def test_cli_allows_only_run_length_overrides(tmp_path: Path) -> None:
    _write_run_configs(tmp_path)

    main(
        [
            "--profile",
            str(PROFILE_DIR / "autovla_a100_1gpu.yaml"),
            "--run-dir",
            str(tmp_path),
            "--max-num-steps",
            "10",
            "--save-freq",
            "5",
        ]
    )

    updated_resolved = yaml.safe_load(
        (tmp_path / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    assert updated_resolved["cosmos"]["train"]["max_num_steps"] == 10
    assert updated_resolved["cosmos"]["train"]["ckpt"]["save_freq"] == 5
    assert updated_resolved["cosmos"]["rollout"]["n_generation"] == 8


@pytest.mark.parametrize(
    ("gpu_count", "mode", "transport", "policy_replicas", "rollout_batch"),
    [
        (1, "colocated", "disk", 1, 1),
        (4, "disaggregated", "nccl", 3, 3),
    ],
)
def test_configure_autovla_a100_run_updates_both_config_sources(
    tmp_path: Path,
    gpu_count: int,
    mode: str,
    transport: str,
    policy_replicas: int,
    rollout_batch: int,
) -> None:
    """The backend YAML and Cosmos TOML must agree for each profile."""
    _write_run_configs(tmp_path)

    configure_autovla_a100_run(tmp_path, _load_profile(gpu_count))

    updated_resolved = yaml.safe_load(
        (tmp_path / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    updated_cosmos = tomllib.loads(
        (tmp_path / "cosmos_config.toml").read_text(encoding="utf-8")
    )
    assert updated_resolved["cosmos"]["mode"] == mode
    assert updated_cosmos["mode"] == mode
    assert updated_resolved["cosmos"]["launch"] == {
        "policy_replicas": policy_replicas,
        "rollout_replicas": 1,
    }
    assert updated_resolved["cosmos"]["rollout"]["n_generation"] == 8
    assert updated_resolved["cosmos"]["rollout"]["batch_size"] == rollout_batch
    assert updated_cosmos["rollout"]["n_generation"] == 8
    assert updated_cosmos["rollout"]["batch_size"] == rollout_batch
    assert updated_resolved["transport"]["kind"] == transport
    if transport == "nccl":
        assert updated_resolved["transport"]["nccl_env"]["NCCL_TIMEOUT"] == "1800"
    else:
        assert updated_resolved["transport"]["nccl_env"] == {}
    assert updated_cosmos["train"]["train_policy"]["epsilon_low"] == pytest.approx(0.1)
    assert (tmp_path / "resolved_config.yaml.pre_autovla_a100").is_file()
    assert (tmp_path / "cosmos_config.toml.pre_autovla_a100").is_file()


def test_a100_geometry_rejects_unmatched_global_episode_count() -> None:
    geometry = replace(_load_profile(4).geometry, rollout_batch_size=1)

    with pytest.raises(ValueError, match="Global rollout and policy batch geometry"):
        geometry.validate()


def test_switching_to_one_gpu_clears_four_gpu_nccl_fields(tmp_path: Path) -> None:
    """Reusing latest/ for 1GPU must not retain the previous NCCL transport."""
    _write_run_configs(tmp_path)
    configure_autovla_a100_run(tmp_path, _load_profile(4))

    configure_autovla_a100_run(tmp_path, _load_profile(1))

    updated_resolved = yaml.safe_load(
        (tmp_path / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    updated_cosmos = tomllib.loads(
        (tmp_path / "cosmos_config.toml").read_text(encoding="utf-8")
    )
    assert updated_resolved["transport"]["kind"] == "disk"
    assert updated_resolved["transport"]["nccl_env"] == {}
    assert updated_resolved["cosmos"]["mode"] == "colocated"
    assert updated_cosmos["mode"] == "colocated"


def test_a100_geometry_rejects_multiple_rollouts_on_single_driver_tunnel() -> None:
    geometry = replace(
        _load_profile(4).geometry,
        policy_replicas=2,
        rollout_replicas=2,
        rollout_batch_size=2,
    )

    with pytest.raises(ValueError, match="5012 SSH tunnel"):
        geometry.validate()


def test_a100_profile_rejects_nccl_colocated_mode() -> None:
    profile = replace(
        _load_profile(1),
        transport="nccl",
    )

    with pytest.raises(ValueError, match="transport=nccl requires"):
        profile.validate()


def test_a100_profile_rejects_wrong_visible_gpu_count() -> None:
    profile = replace(_load_profile(4), cuda_visible_devices="0,1")

    with pytest.raises(ValueError, match="requires 4 visible GPU"):
        profile.validate()
