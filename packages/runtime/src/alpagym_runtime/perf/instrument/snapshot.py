# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared snapshot key + enums used by every CPU/GPU reader and store."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SnapshotCapture(StrEnum):
    """How a snapshot was triggered."""

    PERIODIC = "periodic"
    CHECKPOINT = "checkpoint"


class SnapshotPhase(StrEnum):
    """Phase a snapshot represents within its scope or marker."""

    INSTANT = "instant"
    START = "start"
    END = "end"


@dataclass(frozen=True)
class ResourceSnapshotKey:
    """Aggregation identity for one resource snapshot.

    `path` is the active-scope path at capture time (empty for periodic samples).
    `name` is the last path element for checkpoint snapshots, or the constant
    `"sample"` for periodic samples.
    """

    name: str
    path: tuple[str, ...]
    capture: SnapshotCapture
    phase: SnapshotPhase

    @classmethod
    def periodic_sample(cls) -> ResourceSnapshotKey:
        """Return the constant key used for every periodic monitor sample."""
        return cls(
            name="sample",
            path=(),
            capture=SnapshotCapture.PERIODIC,
            phase=SnapshotPhase.INSTANT,
        )

    @classmethod
    def for_scope_boundary(
        cls,
        path: tuple[str, ...],
        phase: SnapshotPhase,
    ) -> ResourceSnapshotKey:
        """Build a key for a timed scope's `start` or `end` checkpoint.

        Args:
            path: Active-scope path the boundary belongs to. Must be non-empty.
            phase: `SnapshotPhase.START` or `SnapshotPhase.END`.

        Raises:
            ValueError: If `phase` is `INSTANT` or `path` is empty.
        """
        if phase is SnapshotPhase.INSTANT:
            raise ValueError("for_scope_boundary requires START or END phase")
        if not path:
            raise ValueError("for_scope_boundary requires a non-empty path")
        return cls(
            name=path[-1],
            path=path,
            capture=SnapshotCapture.CHECKPOINT,
            phase=phase,
        )

    @classmethod
    def for_marker(cls, path: tuple[str, ...]) -> ResourceSnapshotKey:
        """Build a key for a named instant marker.

        Args:
            path: Marker path produced by appending the marker name to the
                active-scope path.

        Raises:
            ValueError: If `path` is empty.
        """
        if not path:
            raise ValueError("for_marker requires a non-empty path")
        return cls(
            name=path[-1],
            path=path,
            capture=SnapshotCapture.CHECKPOINT,
            phase=SnapshotPhase.INSTANT,
        )
