# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the host-side env-var writers that propagate transport settings.

The host writes a handful of env vars before launching cosmos workers so the
subprocess inherits the values; cosmos workers read them at startup. These
tests pin that writing contract without exercising the heavier hydra path.
"""

import os

import pytest
from alpagym_host.config import TransportConfig, TransportKind
from alpagym_host.transport_env import apply_transport_env_vars


@pytest.fixture(autouse=True)
def _clean_env() -> None:
    """Strip any test-set env vars between cases."""
    yield
    for key in (
        "NCCL_DEBUG",
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "NCCL_TIMEOUT",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "ALPAGYM_RESOLVED_CONFIG_PATH",
    ):
        os.environ.pop(key, None)


def test_apply_transport_env_vars_disk_is_noop() -> None:
    """A disk transport leaves a pre-existing env untouched (neither writes nor cleans)."""
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    apply_transport_env_vars(TransportConfig(kind=TransportKind.disk))
    assert os.environ["NCCL_DEBUG"] == "INFO"
    assert os.environ["NCCL_SOCKET_IFNAME"] == "eth0"


def test_apply_transport_env_vars_nccl_writes_each_entry() -> None:
    """An NCCL transport writes every entry in ``nccl_env`` to ``os.environ``."""
    apply_transport_env_vars(
        TransportConfig(
            kind=TransportKind.nccl,
            nccl_env={"NCCL_DEBUG": "INFO", "NCCL_TIMEOUT": "1800"},
        )
    )
    assert os.environ["NCCL_DEBUG"] == "INFO"
    assert os.environ["NCCL_TIMEOUT"] == "1800"


def test_apply_transport_env_vars_cleans_inherited_nccl_env() -> None:
    """The resolved transport config is the authoritative NCCL env surface."""
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    apply_transport_env_vars(
        TransportConfig(
            kind=TransportKind.nccl,
            nccl_env={
                "NCCL_TIMEOUT": "1800",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            },
        )
    )
    assert "NCCL_DEBUG" not in os.environ
    assert "NCCL_SOCKET_IFNAME" not in os.environ
    assert os.environ["NCCL_TIMEOUT"] == "1800"
    assert os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"
