# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU resource reader and metrics store backed by `psutil`."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import psutil

from alpagym_runtime.perf.instrument.sampling import ReservoirSampler, percentile
from alpagym_runtime.perf.instrument.snapshot import (
    ResourceSnapshotKey,
    SnapshotCapture,
    SnapshotPhase,
)


@dataclass(frozen=True)
class SystemCpuMetrics:
    """Node-level CPU and memory readings."""

    cpu_util_percent: float
    memory_used_mb: float
    memory_available_mb: float
    memory_total_mb: float


@dataclass(frozen=True)
class ProcessCpuMetrics:
    """Per-process CPU and memory readings attributed to this worker.

    Memory fields use psutil's standard names: `rss` is the resident set size
    (physical RAM held by the process), `vms` is the virtual memory size (total
    address space mapped), and `uss` is the unique set size (memory private to
    the process — the closest estimate of what would be freed if it exited).
    """

    cpu_util_percent: float
    rss_mb: float
    vms_mb: float
    uss_mb: float
    num_threads: int


@dataclass(frozen=True)
class CpuSnapshot:
    """One CPU snapshot keyed by capture/name/phase/path."""

    name: str
    path: tuple[str, ...]
    capture: SnapshotCapture
    phase: SnapshotPhase
    system: SystemCpuMetrics
    process: ProcessCpuMetrics


class CpuReader:
    """Read normalized CPU snapshots from `psutil`.

    Each reader keeps its own cumulative-time baselines (system `cpu_times`, the
    process's `cpu_times`, and a monotonic wall clock), so the utilization it
    reports covers the window since *this* reader's previous read. The periodic
    monitor and the scope-boundary checkpoints therefore use separate readers and
    never reset each other's measurement window — unlike the stateful
    `cpu_percent(interval=None)`, whose baseline is shared per process. Reads are
    lock-guarded so concurrent scope boundaries on different threads stay correct.
    The first read after construction covers the (tiny) window since `__init__`.
    """

    def __init__(self) -> None:
        """Build the reader and capture the initial CPU-time baselines."""
        self._process = psutil.Process()
        self._lock = threading.Lock()
        self._last_wall_s = time.monotonic()
        self._last_system_times = psutil.cpu_times()
        self._last_process_cpu_s = self._process_cpu_seconds()

    def _process_cpu_seconds(self) -> float:
        """Return the process's cumulative user + system CPU seconds."""
        times = self._process.cpu_times()
        return times.user + times.system

    def read_snapshot(self, key: ResourceSnapshotKey) -> CpuSnapshot:
        """Read one CPU snapshot identified by `key`."""
        return CpuSnapshot(
            name=key.name,
            path=key.path,
            capture=key.capture,
            phase=key.phase,
            system=self._read_system(),
            process=self._read_process(),
        )

    def _read_system(self) -> SystemCpuMetrics:
        """Read node-level CPU and memory state."""
        vm = psutil.virtual_memory()
        return SystemCpuMetrics(
            cpu_util_percent=self._system_cpu_percent(),
            memory_used_mb=vm.used / (1024.0 * 1024.0),
            memory_available_mb=vm.available / (1024.0 * 1024.0),
            memory_total_mb=vm.total / (1024.0 * 1024.0),
        )

    def _read_process(self) -> ProcessCpuMetrics:
        """Read worker-process CPU and memory state, including USS."""
        memory = self._process.memory_full_info()
        return ProcessCpuMetrics(
            cpu_util_percent=self._process_cpu_percent(),
            rss_mb=memory.rss / (1024.0 * 1024.0),
            vms_mb=memory.vms / (1024.0 * 1024.0),
            uss_mb=memory.uss / (1024.0 * 1024.0),
            num_threads=self._process.num_threads(),
        )

    def _system_cpu_percent(self) -> float:
        """Whole-node CPU utilization (0-100) since this reader's previous read."""
        with self._lock:
            current = psutil.cpu_times()
            previous = self._last_system_times
            self._last_system_times = current
        # Standard /proc-style utilization: busy = total - idle - iowait. `guest`
        # and `guest_nice` are already counted under `user`/`nice`, so exclude
        # them from the total to avoid double counting.
        total_delta = (sum(current) - current.guest - current.guest_nice) - (
            sum(previous) - previous.guest - previous.guest_nice
        )
        if total_delta <= 0.0:
            return 0.0
        idle_delta = (current.idle + current.iowait) - (previous.idle + previous.iowait)
        return max(0.0, min(100.0, (total_delta - idle_delta) / total_delta * 100.0))

    def _process_cpu_percent(self) -> float:
        """Process CPU utilization since the previous read; may exceed 100% on multiple cores."""
        with self._lock:
            now_s = time.monotonic()
            process_cpu_s = self._process_cpu_seconds()
            wall_delta = now_s - self._last_wall_s
            cpu_delta = process_cpu_s - self._last_process_cpu_s
            self._last_wall_s = now_s
            self._last_process_cpu_s = process_cpu_s
        if wall_delta <= 0.0:
            return 0.0
        return max(0.0, cpu_delta / wall_delta * 100.0)


@dataclass
class _SystemAggregate:
    """Reservoir-backed aggregate for the system-view CPU metrics."""

    cpu_util_percent: ReservoirSampler
    memory_used_mb: ReservoirSampler
    memory_available_mb: ReservoirSampler
    cpu_util_max: float = float("-inf")
    memory_used_max: float = float("-inf")
    memory_available_max: float = float("-inf")
    memory_total_mb: float = 0.0


@dataclass
class _ProcessAggregate:
    """Reservoir-backed aggregate for the process-view CPU metrics."""

    cpu_util_percent: ReservoirSampler
    rss_mb: ReservoirSampler
    vms_mb: ReservoirSampler
    uss_mb: ReservoirSampler
    num_threads: ReservoirSampler
    cpu_util_max: float = float("-inf")
    rss_max: float = float("-inf")
    vms_max: float = float("-inf")
    uss_max: float = float("-inf")
    num_threads_max: int = 0


@dataclass
class _CpuRowAggregate:
    """One row of the CPU JSON output: aggregate stats grouped by snapshot key."""

    name: str
    path: tuple[str, ...]
    capture: SnapshotCapture
    phase: SnapshotPhase
    system: _SystemAggregate
    process: _ProcessAggregate
    count: int = 0


class CpuMetricsStore:
    """Aggregate CPU snapshots into one row per `ResourceSnapshotKey`."""

    def __init__(self, max_samples_per_series: int) -> None:
        """Build an empty store with the given per-series sample cap."""
        self._max_samples = max_samples_per_series
        self._rows: dict[ResourceSnapshotKey, _CpuRowAggregate] = {}
        self._lock = threading.Lock()
        self._dirty: int = 0

    def record_snapshot(self, snapshot: CpuSnapshot) -> None:
        """Fold one snapshot into its aggregated row."""
        key = ResourceSnapshotKey(
            name=snapshot.name,
            path=snapshot.path,
            capture=snapshot.capture,
            phase=snapshot.phase,
        )
        with self._lock:
            row = self._rows.get(key)
            if row is None:
                row = _CpuRowAggregate(
                    name=snapshot.name,
                    path=snapshot.path,
                    capture=snapshot.capture,
                    phase=snapshot.phase,
                    system=_SystemAggregate(
                        cpu_util_percent=ReservoirSampler(self._max_samples),
                        memory_used_mb=ReservoirSampler(self._max_samples),
                        memory_available_mb=ReservoirSampler(self._max_samples),
                    ),
                    process=_ProcessAggregate(
                        cpu_util_percent=ReservoirSampler(self._max_samples),
                        rss_mb=ReservoirSampler(self._max_samples),
                        vms_mb=ReservoirSampler(self._max_samples),
                        uss_mb=ReservoirSampler(self._max_samples),
                        num_threads=ReservoirSampler(self._max_samples),
                    ),
                )
                self._rows[key] = row
            _accumulate_system(row.system, snapshot.system)
            _accumulate_process(row.process, snapshot.process)
            row.count += 1
            self._dirty += 1

    def dirty_count(self) -> int:
        """Return the number of snapshots recorded since the last `reset_dirty`."""
        with self._lock:
            return self._dirty

    def reset_dirty(self) -> None:
        """Zero the dirty-snapshot counter once a flush has persisted the data."""
        with self._lock:
            self._dirty = 0

    def to_json(self) -> dict[str, Any]:
        """Render the store as the `cpu` block of the per-process JSON artifact."""
        with self._lock:
            periodic = [
                _row_to_json(row)
                for row in self._rows.values()
                if row.capture is SnapshotCapture.PERIODIC
            ]
            checkpoints = [
                _row_to_json(row)
                for row in self._rows.values()
                if row.capture is SnapshotCapture.CHECKPOINT
            ]
        return {
            "source": "psutil",
            "periodic": periodic,
            "checkpoints": checkpoints,
        }


def _accumulate_system(aggregate: _SystemAggregate, metrics: SystemCpuMetrics) -> None:
    """Fold one `SystemCpuMetrics` reading into the running aggregate."""
    aggregate.cpu_util_percent.add(metrics.cpu_util_percent)
    aggregate.memory_used_mb.add(metrics.memory_used_mb)
    aggregate.memory_available_mb.add(metrics.memory_available_mb)
    aggregate.cpu_util_max = max(aggregate.cpu_util_max, metrics.cpu_util_percent)
    aggregate.memory_used_max = max(aggregate.memory_used_max, metrics.memory_used_mb)
    aggregate.memory_available_max = max(
        aggregate.memory_available_max, metrics.memory_available_mb
    )
    aggregate.memory_total_mb = metrics.memory_total_mb


def _accumulate_process(aggregate: _ProcessAggregate, metrics: ProcessCpuMetrics) -> None:
    """Fold one `ProcessCpuMetrics` reading into the running aggregate."""
    aggregate.cpu_util_percent.add(metrics.cpu_util_percent)
    aggregate.rss_mb.add(metrics.rss_mb)
    aggregate.vms_mb.add(metrics.vms_mb)
    aggregate.uss_mb.add(metrics.uss_mb)
    aggregate.num_threads.add(float(metrics.num_threads))
    aggregate.cpu_util_max = max(aggregate.cpu_util_max, metrics.cpu_util_percent)
    aggregate.rss_max = max(aggregate.rss_max, metrics.rss_mb)
    aggregate.vms_max = max(aggregate.vms_max, metrics.vms_mb)
    aggregate.uss_max = max(aggregate.uss_max, metrics.uss_mb)
    aggregate.num_threads_max = max(aggregate.num_threads_max, metrics.num_threads)


def _row_to_json(row: _CpuRowAggregate) -> dict[str, Any]:
    """Render one aggregated row as a JSON dict."""
    payload: dict[str, Any] = {
        "capture": row.capture.value,
        "name": row.name,
        "phase": row.phase.value,
        "count": row.count,
        "system": {
            "cpu_util_percent": _stats(row.system.cpu_util_percent, row.system.cpu_util_max),
            "memory_used_mb": _stats(row.system.memory_used_mb, row.system.memory_used_max),
            "memory_available_mb": _stats(
                row.system.memory_available_mb, row.system.memory_available_max
            ),
            "memory_total_mb": row.system.memory_total_mb,
        },
        "process": {
            "cpu_util_percent": _stats(row.process.cpu_util_percent, row.process.cpu_util_max),
            "rss_mb": _stats(row.process.rss_mb, row.process.rss_max),
            "vms_mb": _stats(row.process.vms_mb, row.process.vms_max),
            "uss_mb": _stats(row.process.uss_mb, row.process.uss_max),
            "num_threads": _stats(row.process.num_threads, float(row.process.num_threads_max)),
        },
    }
    if row.capture is SnapshotCapture.CHECKPOINT:
        payload["path"] = list(row.path)
    return payload


def _stats(reservoir: ReservoirSampler, max_value: float) -> dict[str, float]:
    """Return `{mean, p95, max}` for one reservoir-backed quantity.

    `mean` is exact over every reading; the bounded reservoir backs only the
    percentile, which stays representative once the sample cap is reached.
    """
    samples = reservoir.samples()
    if not samples:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": reservoir.mean,
        "p95": percentile(samples, 95.0),
        "max": float(max_value),
    }
