# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for transport selection through the host's Hydra config."""

from alpagym_host.config import TransportKind, register_config_schema
from hydra import compose, initialize_config_module


def test_transport_nccl_override_loads_env_vars() -> None:
    """``transport=nccl`` selects the NCCL transport and exposes its env vars."""
    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=["deploy=local", "transport=nccl", "topology=local_colocated_1gpu"],
        )
    assert cfg.transport.kind == TransportKind.nccl
    assert cfg.transport.nccl_env["NCCL_DEBUG"] == "WARN"
    assert cfg.transport.nccl_env["NCCL_SOCKET_IFNAME"] == "^lo,^docker"
    assert cfg.transport.nccl_env["NCCL_TIMEOUT"] == "1800"


def test_transport_nccl_individual_env_override() -> None:
    """One-off env knobs survive a CLI override on top of the nccl preset."""
    register_config_schema()
    with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
        cfg = compose(
            config_name="default",
            overrides=[
                "deploy=local",
                "transport=nccl",
                "topology=local_colocated_1gpu",
                "transport.nccl_env.NCCL_DEBUG=INFO",
            ],
        )
    assert cfg.transport.nccl_env["NCCL_DEBUG"] == "INFO"
    # The dot-key override merges into the preset; sibling keys must survive.
    assert "NCCL_TIMEOUT" in cfg.transport.nccl_env
