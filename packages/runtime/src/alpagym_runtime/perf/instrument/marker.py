# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`record_perf_marker` API for named instants without a measured duration."""

from __future__ import annotations

import logging

from alpagym_runtime.perf.instrument.scope import current_scope_path
from alpagym_runtime.perf.instrument.snapshot import ResourceSnapshotKey
from alpagym_runtime.perf.instrument.store import try_get_perf_store

logger = logging.getLogger(__name__)


def record_perf_marker(
    name: str,
    *,
    cpu_snapshot: bool = False,
    gpu_snapshot: bool = False,
) -> None:
    """Record CPU/GPU snapshots for a named instant marker.

    The marker is attached under the active-scope path (or becomes a root
    when no scope is active). Has no effect when `PerfConfig.enabled` is
    `False` or when neither resource flag is set. Reader failures (psutil/NVML)
    are logged and dropped, so a telemetry hiccup never breaks the caller —
    markers fire on worker-init paths (e.g. `worker/perf_ready`).
    """
    store = try_get_perf_store()
    if store is None:
        return
    marker_path = current_scope_path() + (name,)
    if cpu_snapshot and store.cpu is not None:
        key = ResourceSnapshotKey.for_marker(marker_path)
        try:
            store.cpu.store.record_snapshot(store.cpu.reader.read_snapshot(key))
        except Exception:
            logger.exception("perf: CPU marker snapshot failed at %s", marker_path)
    if gpu_snapshot and store.gpu is not None:
        key = ResourceSnapshotKey.for_marker(marker_path)
        try:
            store.gpu.store.record_snapshot(store.gpu.reader.read_snapshot(key))
        except Exception:
            logger.exception("perf: GPU marker snapshot failed at %s", marker_path)
