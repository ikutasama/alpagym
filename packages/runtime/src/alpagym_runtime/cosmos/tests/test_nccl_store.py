# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Controller-owned NCCL rendezvous store.

The store binds all interfaces and advertises this node's routable hostname so
disaggregated Policy and Rollout workers on other Slurm nodes can reach it. A
loopback-only node name fails fast instead of silently publishing an
unreachable endpoint.
"""

import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from alpagym_host.endpoint_registry import FileTopologyRegistry
from alpagym_runtime.cosmos.nccl_store import start_nccl_store_master


def _resolved_config(tmp_path: Path) -> SimpleNamespace:
    """Build a minimal resolved-config stand-in the store helpers read."""
    return SimpleNamespace(
        artifact_paths=SimpleNamespace(topology_registry_dir=tmp_path / "topology"),
        transport=SimpleNamespace(nccl_env={"NCCL_TIMEOUT": "5"}),
    )


def test_master_publishes_routable_host_and_reader_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The master advertises this node's routable hostname, not loopback."""
    resolved_config = _resolved_config(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: "routable-host")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.0.0.5", 0))
        ],
    )

    store = start_nccl_store_master(resolved_config)
    try:
        registry = FileTopologyRegistry(resolved_config.artifact_paths.topology_registry_dir)
        host, port = registry.read_nccl_master(
            float(resolved_config.transport.nccl_env["NCCL_TIMEOUT"])
        )
        assert host == "routable-host"
        assert port > 0
    finally:
        del store


def test_master_rejects_loopback_hostname(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loopback-only node name fails fast: cross-node workers cannot reach it."""
    resolved_config = _resolved_config(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: "localhost")

    with pytest.raises(RuntimeError, match="loopback"):
        start_nccl_store_master(resolved_config)


def test_master_rejects_unresolvable_hostname(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node name that does not resolve fails fast with a clear error."""
    resolved_config = _resolved_config(tmp_path)
    monkeypatch.setattr(socket, "gethostname", lambda: "ghost-node")

    def _raise_gaierror(*args: object, **kwargs: object) -> list[object]:
        raise socket.gaierror("name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", _raise_gaierror)

    with pytest.raises(RuntimeError, match="does not resolve"):
        start_nccl_store_master(resolved_config)
