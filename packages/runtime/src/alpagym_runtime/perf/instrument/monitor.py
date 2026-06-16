# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single periodic-sampling thread shared by CPU and GPU resource families."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from alpagym_runtime.perf.instrument.snapshot import ResourceSnapshotKey

if TYPE_CHECKING:
    from alpagym_runtime.perf.instrument.cpu import CpuMetricsStore, CpuReader
    from alpagym_runtime.perf.instrument.gpu import GpuMetricsStore, GpuReader

logger = logging.getLogger(__name__)


class ResourceMonitor(threading.Thread):
    """Background daemon that periodically samples a reader into a store."""

    def __init__(
        self,
        sample_interval_s: float,
        reader: CpuReader | GpuReader,
        store: CpuMetricsStore | GpuMetricsStore,
        name: str,
    ) -> None:
        """Build the monitor and capture its target reader + store.

        Args:
            sample_interval_s: Cadence between two consecutive samples.
            reader: Source of resource snapshots.
            store: Destination for the recorded snapshots.
            name: Thread display name (`alpagym-perf-cpu` or `-gpu`).
        """
        if sample_interval_s <= 0.0:
            raise ValueError(f"sample_interval_s must be > 0, got {sample_interval_s}")
        super().__init__(name=name, daemon=True)
        self._sample_interval_s = sample_interval_s
        self._reader = reader
        self._store = store
        self._stop_event = threading.Event()
        self._key = ResourceSnapshotKey.periodic_sample()

    def run(self) -> None:
        """Sample the reader at the configured cadence until stopped."""
        while not self._stop_event.is_set():
            try:
                snapshot = self._reader.read_snapshot(self._key)
                self._store.record_snapshot(snapshot)
            except Exception:
                logger.exception("perf: periodic %s sample failed", self.name)
            if self._stop_event.wait(self._sample_interval_s):
                break

    def stop(self) -> None:
        """Signal the loop to exit at the next wakeup."""
        self._stop_event.set()
