# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local performance store and write-once accessor."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from alpagym_host.config import PerfConfig

from alpagym_runtime.perf.instrument.sampling import ReservoirSampler, percentile

if TYPE_CHECKING:
    from alpagym_runtime.perf.instrument.cpu import CpuMetricsStore, CpuReader
    from alpagym_runtime.perf.instrument.gpu import GpuMetricsStore, GpuReader
    from alpagym_runtime.perf.instrument.monitor import ResourceMonitor

logger = logging.getLogger(__name__)

_SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class WorkerIdentity:
    """Per-process identity stamped on every JSON artifact."""

    run_id: str
    role: str
    rank: int
    local_rank: int
    world_size: int
    hostname: str
    pid: int
    device: str


@dataclass
class _TimingEntry:
    """One aggregated timing scope keyed by full runtime path."""

    name: str
    category: str
    reservoir: ReservoirSampler
    count: int = 0
    total_ns: int = 0
    max_ns: int = 0


class TimingStore:
    """Bounded timing aggregate keyed by runtime scope path."""

    def __init__(self, sample_every_n: int, max_samples_per_series: int) -> None:
        """Build an empty timing store.

        `sample_every_n` controls how often a recorded duration is offered to
        the bounded sample reservoir; `count`, `total_ns`, and `max_ns` stay
        exact across every invocation.
        """
        if sample_every_n <= 0:
            raise ValueError(f"sample_every_n must be > 0, got {sample_every_n}")
        if max_samples_per_series <= 0:
            raise ValueError(f"max_samples_per_series must be > 0, got {max_samples_per_series}")
        self._sample_every_n = sample_every_n
        self._max_samples = max_samples_per_series
        self._entries: dict[tuple[str, ...], _TimingEntry] = {}
        self._lock = threading.Lock()
        self._dirty: int = 0

    def record(
        self,
        *,
        path: tuple[str, ...],
        name: str,
        category: str,
        duration_ns: int,
    ) -> None:
        """Record one timed scope completion under the given runtime path."""
        with self._lock:
            entry = self._entries.get(path)
            if entry is None:
                entry = _TimingEntry(
                    name=name,
                    category=category,
                    reservoir=ReservoirSampler(self._max_samples),
                )
                self._entries[path] = entry
            entry.count += 1
            entry.total_ns += duration_ns
            entry.max_ns = max(entry.max_ns, duration_ns)
            if entry.count == 1 or entry.count % self._sample_every_n == 0:
                entry.reservoir.add(float(duration_ns))
            self._dirty += 1

    def dirty_count(self) -> int:
        """Return the number of updates recorded since the last `reset_dirty`."""
        with self._lock:
            return self._dirty

    def reset_dirty(self) -> None:
        """Zero the dirty-update counter once a flush has persisted the data.

        Called before the flush's `to_json` snapshot: records already recorded are
        captured by that snapshot and persisted, so clearing their flush credit
        here loses nothing. A record that lands between this reset and the snapshot
        is both persisted and counted again — at worst one extra, early flush,
        never a lost one. Zeroing is idempotent, so concurrent flushes cannot drive
        the counter wrong.
        """
        with self._lock:
            self._dirty = 0

    def to_json(self) -> dict[str, Any]:
        """Render the timing store as the `timing` block of the artifact."""
        with self._lock:
            scopes = [_timing_entry_to_json(path, entry) for path, entry in self._entries.items()]
        return {
            "clock": "time.perf_counter_ns",
            "scopes": scopes,
        }


def _timing_entry_to_json(
    path: tuple[str, ...],
    entry: _TimingEntry,
) -> dict[str, Any]:
    """Render one `_TimingEntry` into its JSON row."""
    samples = entry.reservoir.samples()
    mean_ns = entry.total_ns / entry.count if entry.count else 0.0
    return {
        "path": list(path),
        "name": entry.name,
        "category": entry.category,
        "count": entry.count,
        "total_ms": entry.total_ns / 1e6,
        "mean_ms": mean_ns / 1e6,
        "p50_ms": percentile(samples, 50.0) / 1e6,
        "p95_ms": percentile(samples, 95.0) / 1e6,
        "max_ms": entry.max_ns / 1e6,
    }


@dataclass
class ResourceRuntime:
    """Live handles for one resource family: store + reader + optional monitor."""

    store: CpuMetricsStore | GpuMetricsStore
    reader: CpuReader | GpuReader
    monitor: ResourceMonitor | None


@dataclass
class PerfStore:
    """Process-local root store and JSON-artifact owner."""

    identity: WorkerIdentity
    config: PerfConfig
    output_path: Path
    timer: TimingStore
    cpu: ResourceRuntime | None = None
    gpu: ResourceRuntime | None = None
    _write_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last_flush_ns: int = field(default_factory=time.perf_counter_ns, repr=False)

    def to_json(self) -> dict[str, Any]:
        """Render the full JSON artifact for this worker process."""
        payload: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "project": "alpagym",
            **asdict(self.identity),
            "config": asdict(self.config),
            "timing": self.timer.to_json(),
        }
        if self.cpu is not None:
            payload["cpu"] = self.cpu.store.to_json()
        if self.gpu is not None:
            payload["gpu"] = self.gpu.store.to_json()
        return payload

    def write_atomic(self) -> None:
        """Write the per-process JSON artifact atomically via tmp+rename."""
        self.timer.reset_dirty()
        if self.cpu is not None:
            self.cpu.store.reset_dirty()
        if self.gpu is not None:
            self.gpu.store.reset_dirty()
        payload = self.to_json()
        encoded = json.dumps(payload, indent=2, sort_keys=False)
        tmp_path = self.output_path.with_name(f"{self.output_path.name}.{uuid4().hex}.tmp")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock:
            tmp_path.write_text(encoded, encoding="utf-8")
            os.replace(tmp_path, self.output_path)
            self._last_flush_ns = time.perf_counter_ns()

    def should_flush(self) -> bool:
        """Return whether the count- or time-trigger says it's time to flush."""
        dirty = self.timer.dirty_count()
        if self.cpu is not None:
            dirty += self.cpu.store.dirty_count()
        if self.gpu is not None:
            dirty += self.gpu.store.dirty_count()
        if dirty >= self.config.flush_every_n_updates:
            return True
        elapsed_s = (time.perf_counter_ns() - self._last_flush_ns) / 1e9
        return elapsed_s >= self.config.flush_interval_s

    def start_monitors(self) -> None:
        """Start periodic resource monitors after the store is installed."""
        for runtime in (self.cpu, self.gpu):
            if runtime is not None and runtime.monitor is not None:
                runtime.monitor.start()

    def stop_monitors(self) -> bool:
        """Stop the periodic resource monitors and wait for them to exit.

        Joined before the final `write_atomic()` so the durable artifact
        captures every sample a monitor had already recorded. A monitor that
        does not exit within the timeout is logged and left as a daemon; the
        caller must keep process-wide resources such as NVML bound in that case.
        """
        monitors = [
            runtime.monitor
            for runtime in (self.cpu, self.gpu)
            if runtime is not None and runtime.monitor is not None
        ]
        for monitor in monitors:
            monitor.stop()
        for monitor in monitors:
            if isinstance(monitor, threading.Thread) and monitor.ident is None:
                continue
            monitor.join(timeout=5.0)
            if monitor.is_alive():
                logger.warning("perf: resource monitor %s did not exit within 5s", monitor.name)
                return False
        return True

    def cleanup_stale_tmp(self) -> None:
        """Remove any leftover `.tmp` siblings of `output_path` from prior runs."""
        directory = self.output_path.parent
        if not directory.exists():
            return
        prefix = f"{self.output_path.name}."
        for entry in directory.iterdir():
            if entry.name.startswith(prefix) and entry.name.endswith(".tmp"):
                try:
                    entry.unlink()
                except FileNotFoundError:
                    continue


_perf_store: PerfStore | None = None
_flush_thread: _FlushThread | None = None
_lifecycle_lock = threading.RLock()


class _FlushThread(threading.Thread):
    """Background daemon that periodically writes the perf artifact."""

    def __init__(self, store: PerfStore) -> None:
        """Build the flush thread and capture its target store."""
        super().__init__(name="alpagym-perf-flush", daemon=True)
        self._store = store
        self._stop_event = threading.Event()
        # Half the configured interval keeps wakeups close to the requested cadence.
        self._poll_interval_s = max(0.5, store.config.flush_interval_s / 2.0)

    def run(self) -> None:
        """Periodically check flush triggers and write the JSON artifact."""
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._poll_interval_s):
                break
            if self._store.should_flush():
                try:
                    self._store.write_atomic()
                except Exception:
                    logger.exception("perf: periodic flush failed")

    def stop(self) -> None:
        """Signal the loop to exit at the next wakeup."""
        self._stop_event.set()


def set_perf_store(store: PerfStore) -> None:
    """Install the process-wide perf store; raise if one is already installed."""
    global _perf_store, _flush_thread
    with _lifecycle_lock:
        if _perf_store is not None:
            raise RuntimeError("initialize_perf() called twice")
        store.cleanup_stale_tmp()
        _perf_store = store
        try:
            store.start_monitors()
            _flush_thread = _FlushThread(store)
            _flush_thread.start()
        except Exception:
            store.stop_monitors()
            _perf_store = None
            _flush_thread = None
            raise


def try_get_perf_store() -> PerfStore | None:
    """Return the active perf store, or `None` when instrumentation is disabled."""
    with _lifecycle_lock:
        return _perf_store


def perf_lifecycle_lock() -> threading.RLock:
    """Return the process-wide lock guarding perf install/teardown."""
    return _lifecycle_lock


def teardown_perf_store() -> bool:
    """Stop monitors, flush the final artifact, and clear the global store.

    Returns whether the store was fully torn down and its globals cleared.
    A wedged flush thread leaves the store bound and returns `False`, so the
    caller knows process-wide resources (e.g. NVML) must not yet be released.
    """
    global _perf_store, _flush_thread
    with _lifecycle_lock:
        if _perf_store is None:
            return True
        store = _perf_store
        if _flush_thread is not None:
            _flush_thread.stop()
            _flush_thread.join(timeout=5.0)
            if _flush_thread.is_alive():
                # A wedged daemon must not be silently dropped. Quiesce the resource
                # monitors so their readers stop touching NVML/psutil, but leave the
                # globals bound so a later initialize_perf() fails loudly instead of
                # racing a live writer, and skip the final flush, which would block on
                # the write lock the stuck thread may still hold.
                store.stop_monitors()
                logger.warning(
                    "perf: flush thread did not exit within 5s; skipping final flush "
                    "and leaving the store bound"
                )
                return False
        if not store.stop_monitors():
            try:
                store.write_atomic()
            except Exception:
                logger.exception("perf: final flush failed after monitor-stop timeout")
            logger.warning("perf: leaving store bound because a resource monitor is still live")
            return False
        try:
            store.write_atomic()
        except Exception:
            logger.exception("perf: final flush failed")
        _perf_store = None
        _flush_thread = None
        return True
