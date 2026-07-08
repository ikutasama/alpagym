# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import time
from pathlib import Path

import pytest
from alpagym_host.endpoint_registry import FileTopologyRegistry, TopologyEndpoint


def test_file_topology_registry_lists_and_assigns_runtime_endpoints(tmp_path: Path) -> None:
    """Registry returns all runtimes and balances driver assignments."""
    registry = FileTopologyRegistry(tmp_path / "topology")
    registry.publish_alpasim_runtime(TopologyEndpoint("alpasim-runtime-0", "node-a", 30051, 100))
    registry.publish_alpasim_runtime(TopologyEndpoint("alpasim-runtime-1", "node-b", 30051, 1))

    endpoints = registry.list_alpasim_runtimes()

    assert [endpoint.id for endpoint in endpoints] == [
        "alpasim-runtime-0",
        "alpasim-runtime-1",
    ]
    assert registry.acquire_alpasim_runtime(driver_id="driver-a").host == "node-a"
    assert registry.acquire_alpasim_runtime(driver_id="driver-b").host == "node-b"
    assert registry.acquire_alpasim_runtime(driver_id="driver-c").host == "node-a"
    assert registry.acquire_alpasim_runtime(driver_id="driver-a").host == "node-a"


def test_reset_alpasim_topology_clears_runtimes_and_assignments(tmp_path: Path) -> None:
    """Reset lets a requeued attempt republish endpoints and re-balance from zero."""
    registry = FileTopologyRegistry(tmp_path / "topology")
    registry.publish_alpasim_runtime(TopologyEndpoint("alpasim-runtime-0", "node-a", 30051, 100))
    registry.acquire_alpasim_runtime(driver_id="driver-pid-1")

    # Without a reset, the exclusive-create publish raises on the stale file.
    with pytest.raises(FileExistsError):
        registry.publish_alpasim_runtime(TopologyEndpoint("alpasim-runtime-0", "node-b", 30051, 5))

    registry.reset_alpasim_topology()
    registry.publish_alpasim_runtime(TopologyEndpoint("alpasim-runtime-0", "node-b", 30051, 5))

    endpoints = registry.list_alpasim_runtimes()
    assert [(ep.id, ep.host) for ep in endpoints] == [("alpasim-runtime-0", "node-b")]
    # Prior-attempt assignments are gone, so balancing starts from zero.
    assert not list((tmp_path / "topology" / "alpasim_assignments").glob("*.yaml"))


def test_nccl_master_round_trips(tmp_path: Path) -> None:
    """The published NCCL master endpoint reads back as (host, port)."""
    registry = FileTopologyRegistry(tmp_path / "topology")
    registry.publish_nccl_master(host="controller-node", port=29501)
    assert registry.read_nccl_master(timeout_s=5.0) == ("controller-node", 29501)


def test_read_nccl_master_times_out_when_unpublished(tmp_path: Path) -> None:
    """Reading the NCCL master raises once the timeout elapses with no endpoint."""
    registry = FileTopologyRegistry(tmp_path / "topology")
    with pytest.raises(RuntimeError, match="not published"):
        registry.read_nccl_master(timeout_s=0.2)


def test_read_nccl_master_polls_until_published(tmp_path: Path) -> None:
    """The reader waits through an initial absence and returns once the master appears."""
    registry = FileTopologyRegistry(tmp_path / "topology")

    def _publish_after_delay() -> None:
        time.sleep(0.3)
        registry.publish_nccl_master(host="controller-node", port=29501)

    publisher = threading.Thread(target=_publish_after_delay)
    publisher.start()
    try:
        # The endpoint is absent when the read begins, so a non-polling reader would
        # fail; returning the published value proves the poll loop ran past the gap.
        assert registry.read_nccl_master(timeout_s=5.0) == ("controller-node", 29501)
    finally:
        publisher.join()


@pytest.mark.parametrize(
    "endpoint_yaml",
    [
        "host: null\nport: 29501\n",
        "host: controller-node\nport: true\n",
    ],
)
def test_read_nccl_master_rejects_malformed_endpoint(
    tmp_path: Path,
    endpoint_yaml: str,
) -> None:
    """A malformed endpoint fails at the registry boundary instead of coercing."""
    topology = tmp_path / "topology"
    topology.mkdir(parents=True)
    (topology / "nccl_master.yaml").write_text(endpoint_yaml, encoding="utf-8")
    registry = FileTopologyRegistry(topology)
    with pytest.raises(TypeError):
        registry.read_nccl_master(timeout_s=1.0)
