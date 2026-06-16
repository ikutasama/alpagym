# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from alpagym_host.config import (
    AllInOneSlurmTopologyConfig,
    ExecutionBackend,
    SeparateNodesSlurmTopologyConfig,
    SlurmLayout,
    SlurmTopologyConfig,
)


@dataclass(frozen=True)
class RunHostPlan:
    """Workload role and GPU allocation for one logical run host."""

    hostname: str
    host_index: int
    runs_cosmos: bool
    runs_alpasim: bool
    cosmos_gpus: int
    alpasim_gpus: int

    @property
    def cosmos_gpu_count(self) -> int:
        """Return the number of GPUs assigned to Cosmos work."""
        return self.cosmos_gpus

    @property
    def cosmos_gpu_ids(self) -> tuple[int, ...]:
        """Return contiguous GPU ids assigned to Cosmos work."""
        return tuple(range(self.cosmos_gpu_count))

    @property
    def alpasim_gpu_ids(self) -> tuple[int, ...]:
        """Return contiguous GPU ids assigned to AlpaSim work."""
        alpasim_start = self.cosmos_gpu_count if self.runs_cosmos else 0
        return tuple(range(alpasim_start, alpasim_start + self.alpasim_gpus))


@dataclass(frozen=True)
class RunTopologyPlan:
    """Expanded host topology for a run backend."""

    hosts: tuple[RunHostPlan, ...]

    @property
    def cosmos_host_plans(self) -> tuple[RunHostPlan, ...]:
        """Return host plans that run Cosmos work."""
        return tuple(host for host in self.hosts if host.runs_cosmos)

    @property
    def alpasim_host_plans(self) -> tuple[RunHostPlan, ...]:
        """Return host plans that run AlpaSim work."""
        return tuple(host for host in self.hosts if host.runs_alpasim)

    @property
    def cosmos_hosts(self) -> tuple[str, ...]:
        """Return hostnames that run Cosmos work."""
        return tuple(host.hostname for host in self.cosmos_host_plans)

    @property
    def alpasim_hosts(self) -> tuple[str, ...]:
        """Return hostnames that run AlpaSim work."""
        return tuple(host.hostname for host in self.alpasim_host_plans)


def build_local_topology() -> RunTopologyPlan:
    """Return the single logical host used by local execution."""
    return RunTopologyPlan(
        hosts=(
            RunHostPlan(
                hostname="localhost",
                host_index=0,
                runs_cosmos=True,
                runs_alpasim=True,
                # Local subprocesses do not bind GPUs through the topology plan.
                cosmos_gpus=0,
                alpasim_gpus=0,
            ),
        )
    )


def build_slurm_topology(
    backend: ExecutionBackend | str,
    hostnames: list[str],
    gpus_per_node: int,
    topology: SlurmTopologyConfig,
) -> RunTopologyPlan:
    """Expand Slurm layout settings into per-host run topology.

    Args:
        backend: Slurm execution backend.
        hostnames: Slurm hostnames in allocation order.
        gpus_per_node: Number of GPUs available on each Slurm node.
        topology: Slurm host topology settings.

    Returns:
        Per-host topology with Cosmos GPUs before AlpaSim GPUs on each host.

    Raises:
        ValueError: The backend, layout, host count, or GPU counts are invalid.
    """
    execution_backend = ExecutionBackend(backend)
    if gpus_per_node < 1:
        raise ValueError("gpus_per_node must be at least 1")
    if execution_backend is not ExecutionBackend.slurm:
        raise ValueError("backend must be a Slurm backend")

    match SlurmLayout(topology.kind):
        case SlurmLayout.all_in_one:
            if not isinstance(topology, AllInOneSlurmTopologyConfig):
                raise TypeError(type(topology))
            return _build_all_in_one_slurm_topology(
                hostnames=hostnames,
                gpus_per_node=gpus_per_node,
                alpasim_gpus=topology.alpasim_gpus,
            )
        case SlurmLayout.separate_nodes:
            if not isinstance(topology, SeparateNodesSlurmTopologyConfig):
                raise TypeError(type(topology))
            return _build_separate_nodes_slurm_topology(
                hostnames=hostnames,
                gpus_per_node=gpus_per_node,
                cosmos_nodes=topology.cosmos_nodes,
                alpasim_nodes=topology.alpasim_nodes,
            )


def _build_all_in_one_slurm_topology(
    hostnames: list[str],
    gpus_per_node: int,
    alpasim_gpus: int,
) -> RunTopologyPlan:
    """Build a one-node topology that colocates Cosmos and AlpaSim."""
    if len(hostnames) != 1:
        raise ValueError("all_in_one requires exactly one hostname")
    if alpasim_gpus < 1:
        raise ValueError("all_in_one requires at least one AlpaSim GPU")
    if alpasim_gpus >= gpus_per_node:
        raise ValueError("all_in_one requires alpasim_gpus to leave at least one Cosmos GPU")

    return RunTopologyPlan(
        hosts=(
            RunHostPlan(
                hostname=hostnames[0],
                host_index=0,
                runs_cosmos=True,
                runs_alpasim=True,
                cosmos_gpus=gpus_per_node - alpasim_gpus,
                alpasim_gpus=alpasim_gpus,
            ),
        )
    )


def _build_separate_nodes_slurm_topology(
    hostnames: list[str],
    gpus_per_node: int,
    cosmos_nodes: int,
    alpasim_nodes: int,
) -> RunTopologyPlan:
    """Build a topology with disjoint full-node Cosmos and AlpaSim hosts."""
    if cosmos_nodes < 1:
        raise ValueError("separate_nodes requires at least one Cosmos node")
    if alpasim_nodes < 1:
        raise ValueError("separate_nodes requires at least one AlpaSim node")
    if len(hostnames) != cosmos_nodes + alpasim_nodes:
        raise ValueError("separate_nodes host count must match cosmos_nodes + alpasim_nodes")

    hosts: list[RunHostPlan] = []
    for host_index, hostname in enumerate(hostnames[:cosmos_nodes]):
        hosts.append(
            RunHostPlan(
                hostname=hostname,
                host_index=host_index,
                runs_cosmos=True,
                runs_alpasim=False,
                cosmos_gpus=gpus_per_node,
                alpasim_gpus=0,
            )
        )
    for host_index, hostname in enumerate(
        hostnames[cosmos_nodes:],
        start=cosmos_nodes,
    ):
        hosts.append(
            RunHostPlan(
                hostname=hostname,
                host_index=host_index,
                runs_cosmos=False,
                runs_alpasim=True,
                cosmos_gpus=0,
                alpasim_gpus=gpus_per_node,
            )
        )
    return RunTopologyPlan(hosts=tuple(hosts))
