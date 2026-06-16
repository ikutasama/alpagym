# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU resource reader and metrics store backed by NVML + `torch.cuda`."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

import pynvml
import torch

from alpagym_runtime.perf.instrument.sampling import ReservoirSampler, percentile
from alpagym_runtime.perf.instrument.snapshot import (
    ResourceSnapshotKey,
    SnapshotCapture,
    SnapshotPhase,
)


@dataclass(frozen=True)
class DeviceGpuMetrics:
    """Selected-GPU utilization and memory state (whole-device view)."""

    gpu_util_percent: float
    memory_util_percent: float
    memory_used_mb: float
    memory_total_mb: float


@dataclass(frozen=True)
class ProcessGpuMetrics:
    """Process-attributed GPU memory state.

    `driver_memory_mb` is `None` when the NVML process query returns no entry
    for the current pid (e.g. CUDA context not yet established or driver-side
    query is unsupported in the environment).
    """

    torch_allocated_mb: float
    torch_reserved_mb: float
    driver_memory_mb: float | None


@dataclass(frozen=True)
class GpuSnapshot:
    """One GPU snapshot keyed by capture/name/phase/path."""

    name: str
    path: tuple[str, ...]
    capture: SnapshotCapture
    phase: SnapshotPhase
    device: DeviceGpuMetrics
    process: ProcessGpuMetrics


class GpuReader:
    """Read normalized GPU snapshots from NVML and the PyTorch CUDA allocator.

    The NVML handle is bound to `device_index` for the lifetime of the reader.
    Checkpoint reads do not synchronize CUDA streams; device-level utilization
    is a recent-window NVML value, not line-level attribution.
    """

    def __init__(self, device_index: int) -> None:
        """Bind the NVML handle for the GPU behind `torch.cuda` `device_index`.

        `device_index` is a `torch.cuda` index, which enumerates the
        `CUDA_VISIBLE_DEVICES` subset, while NVML enumerates every physical GPU.
        The handle is resolved by UUID so NVML and `torch.cuda` point at the same
        device even when the launcher masks GPUs per replica.
        """
        self._device_index = device_index
        self._handle = _nvml_handle_for_torch_device(device_index)
        self._uuid = _maybe_decode(pynvml.nvmlDeviceGetUUID(self._handle))
        self._device_name = _maybe_decode(pynvml.nvmlDeviceGetName(self._handle))
        self._pid = os.getpid()

    @property
    def device_index(self) -> int:
        """Return the `torch.cuda` visible-device index, not the NVML physical index.

        The NVML handle is resolved by UUID, so this index must not be passed
        back to NVML as a physical-device index.
        """
        return self._device_index

    @property
    def device_uuid(self) -> str:
        """Return the NVML UUID of the bound device."""
        return self._uuid

    @property
    def device_name(self) -> str:
        """Return the NVML name of the bound device."""
        return self._device_name

    def read_snapshot(self, key: ResourceSnapshotKey) -> GpuSnapshot:
        """Read one GPU snapshot identified by `key`."""
        return GpuSnapshot(
            name=key.name,
            path=key.path,
            capture=key.capture,
            phase=key.phase,
            device=self._read_device(),
            process=self._read_process(),
        )

    def _read_device(self) -> DeviceGpuMetrics:
        """Read whole-device utilization and memory state."""
        util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return DeviceGpuMetrics(
            gpu_util_percent=float(util.gpu),
            memory_util_percent=float(util.memory),
            memory_used_mb=memory.used / (1024.0 * 1024.0),
            memory_total_mb=memory.total / (1024.0 * 1024.0),
        )

    def _read_process(self) -> ProcessGpuMetrics:
        """Read process-attributed GPU memory from torch + NVML process query."""
        torch_allocated_mb = torch.cuda.memory_allocated(self._device_index) / (1024.0 * 1024.0)
        torch_reserved_mb = torch.cuda.memory_reserved(self._device_index) / (1024.0 * 1024.0)
        driver_memory_mb = _read_driver_process_memory_mb(self._handle, self._pid)
        return ProcessGpuMetrics(
            torch_allocated_mb=torch_allocated_mb,
            torch_reserved_mb=torch_reserved_mb,
            driver_memory_mb=driver_memory_mb,
        )


def _nvml_handle_for_torch_device(device_index: int) -> Any:
    """Return the NVML handle for the physical GPU behind `torch.cuda` `device_index`.

    NVML indexes every physical GPU while `torch.cuda` indexes the
    `CUDA_VISIBLE_DEVICES` subset, so the two indices diverge under per-replica
    GPU masks. Matching on normalized UUID binds the handle to the same GPU whose
    `torch.cuda` memory this reader records, regardless of masking.
    """
    torch_uuid = _normalize_gpu_uuid(str(torch.cuda.get_device_properties(device_index).uuid))
    for nvml_index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
        if _normalize_gpu_uuid(pynvml.nvmlDeviceGetUUID(handle)) == torch_uuid:
            return handle
    raise RuntimeError(
        f"perf GPU mismatch: no NVML device matches torch[{device_index}] "
        f"UUID {torch_uuid}; cannot bind GPU metrics to the policy device."
    )


def _normalize_gpu_uuid(value: str | bytes) -> str:
    """Normalize torch/NVML GPU UUID spellings for exact comparison."""
    uuid = _maybe_decode(value).lower()
    return uuid.removeprefix("gpu-")


def _maybe_decode(value: str | bytes) -> str:
    """Return `value` as a `str`, decoding from utf-8 when NVML returns bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_driver_process_memory_mb(handle: Any, pid: int) -> float | None:
    """Return this process's NVML-reported GPU memory, or `None` when unknown.

    On systems where the driver does not expose per-process attribution (some
    VM/sandboxed environments) NVML returns an empty list, raises `NVMLError`,
    or reports `usedGpuMemory` as `None`. In all cases the metric is dropped
    from the snapshot rather than reported as 0.
    """
    try:
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except pynvml.NVMLError:
        return None
    for proc in processes:
        if int(proc.pid) == pid:
            # NVML reports unavailable per-process memory as None (already mapped
            # from the NVML_VALUE_NOT_AVAILABLE sentinel by pynvml); keep the
            # documented None-not-zero contract instead of raising on float(None).
            if proc.usedGpuMemory is None:
                return None
            return float(proc.usedGpuMemory) / (1024.0 * 1024.0)
    return None


@dataclass
class _DeviceAggregate:
    """Reservoir-backed aggregate for the whole-device GPU metrics."""

    gpu_util_percent: ReservoirSampler
    memory_util_percent: ReservoirSampler
    memory_used_mb: ReservoirSampler
    gpu_util_max: float = float("-inf")
    memory_util_max: float = float("-inf")
    memory_used_max: float = float("-inf")
    memory_total_mb: float = 0.0


@dataclass
class _ProcessAggregate:
    """Reservoir-backed aggregate for process-attributed GPU memory."""

    torch_allocated_mb: ReservoirSampler
    torch_reserved_mb: ReservoirSampler
    driver_memory_mb: ReservoirSampler
    torch_allocated_max: float = float("-inf")
    torch_reserved_max: float = float("-inf")
    driver_memory_max: float = float("-inf")
    driver_memory_seen: bool = False


@dataclass
class _GpuRowAggregate:
    """One row of the GPU JSON output: aggregate stats by snapshot key."""

    name: str
    path: tuple[str, ...]
    capture: SnapshotCapture
    phase: SnapshotPhase
    device: _DeviceAggregate
    process: _ProcessAggregate
    count: int = 0


class GpuMetricsStore:
    """Aggregate GPU snapshots into one row per `ResourceSnapshotKey`."""

    def __init__(
        self,
        max_samples_per_series: int,
        device_index: int,
        device_uuid: str,
        device_name: str,
    ) -> None:
        """Build an empty store tagged with the bound NVML device identity."""
        self._max_samples = max_samples_per_series
        self._device_index = device_index
        self._device_uuid = device_uuid
        self._device_name = device_name
        self._rows: dict[ResourceSnapshotKey, _GpuRowAggregate] = {}
        self._lock = threading.Lock()
        self._dirty: int = 0

    def record_snapshot(self, snapshot: GpuSnapshot) -> None:
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
                row = _GpuRowAggregate(
                    name=snapshot.name,
                    path=snapshot.path,
                    capture=snapshot.capture,
                    phase=snapshot.phase,
                    device=_DeviceAggregate(
                        gpu_util_percent=ReservoirSampler(self._max_samples),
                        memory_util_percent=ReservoirSampler(self._max_samples),
                        memory_used_mb=ReservoirSampler(self._max_samples),
                    ),
                    process=_ProcessAggregate(
                        torch_allocated_mb=ReservoirSampler(self._max_samples),
                        torch_reserved_mb=ReservoirSampler(self._max_samples),
                        driver_memory_mb=ReservoirSampler(self._max_samples),
                    ),
                )
                self._rows[key] = row
            _accumulate_device(row.device, snapshot.device)
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
        """Render the store as the `gpu` block of the per-process JSON artifact."""
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
            "sources": {
                "device": "nvml",
                "process": ["torch.cuda", "nvml_process_query"],
            },
            "device_index": self._device_index,
            "device_uuid": self._device_uuid,
            "device_name": self._device_name,
            "periodic": periodic,
            "checkpoints": checkpoints,
        }


def _accumulate_device(aggregate: _DeviceAggregate, metrics: DeviceGpuMetrics) -> None:
    """Fold one `DeviceGpuMetrics` reading into the running aggregate."""
    aggregate.gpu_util_percent.add(metrics.gpu_util_percent)
    aggregate.memory_util_percent.add(metrics.memory_util_percent)
    aggregate.memory_used_mb.add(metrics.memory_used_mb)
    aggregate.gpu_util_max = max(aggregate.gpu_util_max, metrics.gpu_util_percent)
    aggregate.memory_util_max = max(aggregate.memory_util_max, metrics.memory_util_percent)
    aggregate.memory_used_max = max(aggregate.memory_used_max, metrics.memory_used_mb)
    aggregate.memory_total_mb = metrics.memory_total_mb


def _accumulate_process(aggregate: _ProcessAggregate, metrics: ProcessGpuMetrics) -> None:
    """Fold one `ProcessGpuMetrics` reading into the running aggregate."""
    aggregate.torch_allocated_mb.add(metrics.torch_allocated_mb)
    aggregate.torch_reserved_mb.add(metrics.torch_reserved_mb)
    aggregate.torch_allocated_max = max(aggregate.torch_allocated_max, metrics.torch_allocated_mb)
    aggregate.torch_reserved_max = max(aggregate.torch_reserved_max, metrics.torch_reserved_mb)
    if metrics.driver_memory_mb is not None:
        aggregate.driver_memory_mb.add(metrics.driver_memory_mb)
        aggregate.driver_memory_max = max(aggregate.driver_memory_max, metrics.driver_memory_mb)
        aggregate.driver_memory_seen = True


def _row_to_json(row: _GpuRowAggregate) -> dict[str, Any]:
    """Render one aggregated row as a JSON dict."""
    process_payload: dict[str, Any] = {
        "torch_allocated_mb": _stats(
            row.process.torch_allocated_mb, row.process.torch_allocated_max
        ),
        "torch_reserved_mb": _stats(row.process.torch_reserved_mb, row.process.torch_reserved_max),
    }
    if row.process.driver_memory_seen:
        process_payload["driver_memory_mb"] = _stats(
            row.process.driver_memory_mb, row.process.driver_memory_max
        )
    payload: dict[str, Any] = {
        "capture": row.capture.value,
        "name": row.name,
        "phase": row.phase.value,
        "count": row.count,
        "device": {
            "gpu_util_percent": _stats(row.device.gpu_util_percent, row.device.gpu_util_max),
            "memory_util_percent": _stats(
                row.device.memory_util_percent, row.device.memory_util_max
            ),
            "memory_used_mb": _stats(row.device.memory_used_mb, row.device.memory_used_max),
            "memory_total_mb": row.device.memory_total_mb,
        },
        "process": process_payload,
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
