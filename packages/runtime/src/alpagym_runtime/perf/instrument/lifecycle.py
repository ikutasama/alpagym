# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`initialize_perf` / `shutdown_perf` and the `WorkerIdentity` resolver."""

from __future__ import annotations

import atexit
import logging
import os
import socket
from pathlib import Path

from alpagym_host.config import PerfConfig, RunConfig

from alpagym_runtime.perf.instrument.marker import record_perf_marker
from alpagym_runtime.perf.instrument.monitor import ResourceMonitor
from alpagym_runtime.perf.instrument.store import (
    PerfStore,
    ResourceRuntime,
    TimingStore,
    WorkerIdentity,
    perf_lifecycle_lock,
    set_perf_store,
    teardown_perf_store,
    try_get_perf_store,
)

logger = logging.getLogger(__name__)


def initialize_perf(resolved_config: RunConfig) -> None:
    """Build the process-wide `PerfStore` and start any periodic monitors.

    Reads `resolved_config.perf` for behavior and
    `resolved_config.artifact_paths.perf_dir` for output. Resolves
    `WorkerIdentity` once from Cosmos-RL launch env vars plus local system
    calls. No-op when `PerfConfig.enabled` is `False`.
    """
    cfg = resolved_config.perf
    if not cfg.enabled:
        return
    with perf_lifecycle_lock():
        # Cosmos colocated mode hosts several roles (policy + rollout) in one process,
        # so this runs more than once per process; the first call installs the store
        # and later calls are a no-op. Disaggregated runs call it once per process.
        if try_get_perf_store() is not None:
            return
        identity = _resolve_worker_identity(Path(resolved_config.artifact_paths.run_dir))
        perf_dir = Path(resolved_config.artifact_paths.perf_dir)
        output_path = perf_dir / f"alpagym_perf_{identity.role}_{identity.rank}_{identity.pid}.json"
        store = PerfStore(
            identity=identity,
            config=cfg,
            output_path=output_path,
            timer=TimingStore(
                sample_every_n=cfg.sample_every_n,
                max_samples_per_series=cfg.max_samples_per_series,
            ),
        )
        if cfg.collect_cpu:
            store.cpu = _build_cpu_runtime(cfg)
        if cfg.collect_gpu:
            store.gpu = _build_gpu_runtime(cfg, identity)
        try:
            set_perf_store(store)
        except Exception:
            if store.gpu is not None:
                import pynvml

                pynvml.nvmlShutdown()
            raise
        atexit.register(shutdown_perf)
        record_perf_marker("worker/perf_ready", cpu_snapshot=True, gpu_snapshot=True)
        logger.info(
            "perf: initialized for role=%s rank=%s device=%s -> %s",
            identity.role,
            identity.rank,
            identity.device,
            output_path,
        )


def shutdown_perf() -> None:
    """Stop monitors, flush the final JSON artifact, and shut down NVML.

    NVML is only shut down once `teardown_perf_store()` reports it cleared the
    store: a wedged flush thread leaves the GPU monitor bound, and shutting NVML
    down under a live reader would spin it on `NVMLError` every poll.
    """
    with perf_lifecycle_lock():
        store = try_get_perf_store()
        if store is None:
            return
        had_gpu = store.gpu is not None
        cleared = teardown_perf_store()
        if had_gpu and cleared:
            import pynvml

            pynvml.nvmlShutdown()


def _resolve_worker_identity(run_dir: Path) -> WorkerIdentity:
    """Resolve the per-process identity stamped on the JSON artifact."""
    local_rank = int(os.environ["LOCAL_RANK"])
    return WorkerIdentity(
        run_id=run_dir.name,
        # COSMOS_ROLE is title case (e.g. "Rollout"/"Policy"); lowercase it so the
        # artifact role, filename, and `--role` filter match the documented form.
        role=os.environ["COSMOS_ROLE"].lower(),
        rank=int(os.environ["RANK"]),
        local_rank=local_rank,
        world_size=int(os.environ["WORLD_SIZE"]),
        hostname=socket.gethostname(),
        pid=os.getpid(),
        device=f"cuda:{local_rank}",
    )


def _build_cpu_runtime(cfg: PerfConfig) -> ResourceRuntime:
    """Build the CPU `ResourceRuntime`: store, reader, optional periodic monitor."""
    from alpagym_runtime.perf.instrument.cpu import CpuMetricsStore, CpuReader

    store = CpuMetricsStore(max_samples_per_series=cfg.max_samples_per_series)
    # Separate readers so the periodic monitor and the scope-boundary checkpoints
    # keep independent CPU-time baselines and never reset each other's measurement
    # window (see `CpuReader`). GPU readers need no such split — NVML utilization is
    # a recent-window value, not a per-reader delta.
    checkpoint_reader = CpuReader()
    monitor: ResourceMonitor | None = None
    if cfg.resource_sample_interval_s is not None:
        monitor = ResourceMonitor(
            sample_interval_s=cfg.resource_sample_interval_s,
            reader=CpuReader(),
            store=store,
            name="alpagym-perf-cpu",
        )
    return ResourceRuntime(store=store, reader=checkpoint_reader, monitor=monitor)


def _build_gpu_runtime(cfg: PerfConfig, identity: WorkerIdentity) -> ResourceRuntime:
    """Build the GPU `ResourceRuntime`: store, reader, optional periodic monitor."""
    import pynvml

    from alpagym_runtime.perf.instrument.gpu import GpuMetricsStore, GpuReader

    pynvml.nvmlInit()
    # If reader/store construction fails (e.g. no matching NVML UUID), shut NVML
    # back down here: the store is not yet registered, so the atexit-driven
    # `shutdown_perf` path that would otherwise call `nvmlShutdown` never runs.
    try:
        reader = GpuReader(device_index=identity.local_rank)
        store = GpuMetricsStore(
            max_samples_per_series=cfg.max_samples_per_series,
            device_index=reader.device_index,
            device_uuid=reader.device_uuid,
            device_name=reader.device_name,
        )
    except Exception:
        pynvml.nvmlShutdown()
        raise
    monitor: ResourceMonitor | None = None
    if cfg.resource_sample_interval_s is not None:
        monitor = ResourceMonitor(
            sample_interval_s=cfg.resource_sample_interval_s,
            reader=reader,
            store=store,
            name="alpagym-perf-gpu",
        )
    return ResourceRuntime(store=store, reader=reader, monitor=monitor)
