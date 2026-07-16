# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the four-A100 AutoVLA copied-run configurator."""

import tomllib
from pathlib import Path

import pytest
import tomli_w
import yaml
from alpagym_host.autovla_a100_config import (
    A100TrainGeometry,
    configure_autovla_a100_run,
)


def test_configure_autovla_a100_run_updates_both_config_sources(tmp_path: Path) -> None:
    """The backend YAML and Cosmos TOML must describe identical rollout geometry."""
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
    resolved_path = tmp_path / "resolved_config.yaml"
    cosmos_path = tmp_path / "cosmos_config.toml"
    resolved_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
    cosmos_path.write_text(tomli_w.dumps(cosmos), encoding="utf-8")

    configure_autovla_a100_run(tmp_path, A100TrainGeometry())

    updated_resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    updated_cosmos = tomllib.loads(cosmos_path.read_text(encoding="utf-8"))
    assert updated_resolved["cosmos"]["launch"] == {
        "policy_replicas": 3,
        "rollout_replicas": 1,
    }
    assert updated_resolved["cosmos"]["rollout"]["n_generation"] == 8
    assert updated_resolved["cosmos"]["rollout"]["batch_size"] == 3
    assert updated_resolved["cosmos"]["train"]["train_batch_per_replica"] == 8
    assert updated_resolved["policy"]["inference"]["max_batch_size"] == 3
    assert updated_resolved["transport"]["kind"] == "nccl"
    assert updated_resolved["transport"]["nccl_env"]["NCCL_TIMEOUT"] == "1800"
    assert updated_cosmos["rollout"]["n_generation"] == 8
    assert updated_cosmos["rollout"]["batch_size"] == 3
    assert updated_cosmos["train"]["train_batch_per_replica"] == 8
    assert updated_cosmos["train"]["train_policy"]["epsilon_low"] == pytest.approx(0.1)
    assert (tmp_path / "resolved_config.yaml.pre_autovla_4gpu").is_file()
    assert (tmp_path / "cosmos_config.toml.pre_autovla_4gpu").is_file()


def test_a100_geometry_rejects_unmatched_global_episode_count() -> None:
    """Rollout production must equal the three policy replicas' global demand."""
    geometry = A100TrainGeometry(rollout_batch_size=1)

    with pytest.raises(ValueError, match="Global rollout and policy batch geometry"):
        geometry.validate()


def test_a100_geometry_rejects_multiple_rollouts_on_single_driver_tunnel() -> None:
    """One fixed 5012 port cannot host multiple rollout Egodriver servers."""
    geometry = A100TrainGeometry(
        policy_replicas=2,
        rollout_replicas=2,
        rollout_batch_size=2,
        train_batch_per_replica=8,
    )

    with pytest.raises(ValueError, match="5012 SSH tunnel"):
        geometry.validate()
