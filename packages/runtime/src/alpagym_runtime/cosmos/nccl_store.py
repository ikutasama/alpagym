# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller-owned TCPStore master for the NCCL transport.

NCCL communicators bootstrap from a torch ``TCPStore``: one process opens it
as master, others connect as clients. The Controller owns the master because
it runs for the whole job. It binds an ephemeral port on all interfaces and
publishes its hostname as ``{host, port}`` through the run's topology registry
(``<run_dir>/topology/nccl_master.yaml``). Policy and Rollout workers, including
those on other Slurm nodes, read that endpoint to find the master.
"""

import ipaddress
import logging
import socket

from alpagym_host.config import RunConfig
from alpagym_host.endpoint_registry import FileTopologyRegistry
from torch.distributed import TCPStore

logger = logging.getLogger(__name__)


def start_nccl_store_master(run_config: RunConfig) -> TCPStore:
    """Start the Controller-owned TCPStore and publish its endpoint.

    Binds the master on all interfaces and advertises this node's hostname
    (``socket.gethostname()``) through the topology registry, so Policy and
    Rollout workers on other Slurm nodes can reach it. Fails fast if that
    hostname is loopback-only or does not resolve; it does not otherwise
    guarantee the name is reachable from peer nodes.
    """
    # Bind all interfaces but advertise the routable node name: disaggregated
    # runs place Policy and Rollout workers on other Slurm nodes, which cannot
    # reach a loopback master. Fail fast if the node name is loopback-only.
    published_hostname = socket.gethostname()
    try:
        addresses = socket.getaddrinfo(published_hostname, None, proto=socket.IPPROTO_TCP)
        host_is_loopback = all(ipaddress.ip_address(info[4][0]).is_loopback for info in addresses)
    except socket.gaierror as error:
        raise RuntimeError(
            f"NCCL TCPStore master hostname {published_hostname!r} does not resolve; "
            "cross-node Policy/Rollout workers cannot reach it."
        ) from error
    if host_is_loopback:
        raise RuntimeError(
            f"NCCL TCPStore master resolved a loopback-only hostname {published_hostname!r}; "
            "cross-node Policy/Rollout workers cannot reach it."
        )
    registry = FileTopologyRegistry(run_config.artifact_paths.topology_registry_dir)
    # A Slurm requeue reuses run_dir, so a previous attempt's endpoint can linger.
    # Drop it before binding so a worker polling during startup waits for the fresh
    # publish below instead of connecting to the previous attempt's dead TCPStore.
    registry.clear_nccl_master()
    attempted_ports: list[int] = []
    last_error: OSError | None = None
    for _ in range(10):
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", 0))
            port = int(probe.getsockname()[1])
        attempted_ports.append(port)
        try:
            store = TCPStore(
                host_name="0.0.0.0",
                port=port,
                world_size=1,
                is_master=True,
                wait_for_workers=False,
            )
        except OSError as error:
            last_error = error
            logger.debug("NCCL TCPStore master failed to bind port %d: %s", port, error)
            continue
        registry.publish_nccl_master(host=published_hostname, port=port)
        return store
    raise RuntimeError(
        f"Failed to bind NCCL TCPStore master after {len(attempted_ports)} attempts; "
        f"tried ports {attempted_ports}"
    ) from last_error
