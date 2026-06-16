# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rollout freshness filtering for the AlpaGym Cosmos trainer."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)


def filter_trainable_rollouts(
    rollouts: list[Any],
    current_step: int,
    train_batch_per_replica: int,
    allowed_outdated_steps: int,
) -> list[Any]:
    """Validate rollouts and keep the freshest ``train_batch_per_replica``.

    Cosmos-RL computes GRPO advantages before the trainer sees the data and
    dispatches rollouts to data-parallel ranks individually, so a rank may
    receive only part of a prompt's generation group. Filtering is therefore
    per-rollout, not per-group: rollouts are sorted freshest-first by
    ``weight_version``, the top ``train_batch_per_replica`` are kept, and the
    completion artifacts of the dropped lower-version excess are unlinked so
    workers don't accumulate stale files on disk. Kept rollouts older than
    ``allowed_outdated_steps`` are warned about (the staleness check), not
    dropped.

    Args:
        rollouts: Cosmos-RL rollouts for this rank; each carries
            ``completion`` (artifact path) and ``weight_version``.
        current_step: Current trainer policy step. ``weight_version`` is the
            policy step whose weights generated the rollout, so
            ``current_step - weight_version`` is the rollout's policy lag.
        train_batch_per_replica: Number of rollouts to keep per replica.
        allowed_outdated_steps: Staleness tolerance in policy steps. A kept
            rollout whose ``weight_version`` is below
            ``current_step - allowed_outdated_steps`` triggers a warning.

    Returns:
        Up to ``train_batch_per_replica`` rollouts in fresh-first order.
    """
    if not rollouts:
        return rollouts

    # Per-rollout completion paths must be unique; sharing one means the unlink
    # below would race with the other rollout that still references the file.
    # Empty completions are skipped by that unlink loop, so ignore them here too.
    paths = [str(rollout.completion) for rollout in rollouts if rollout.completion]
    if len(set(paths)) != len(paths):
        raise ValueError(f"Rollouts from Cosmos-RL have duplicate completion paths: {paths}")

    ordered = sorted(rollouts, key=lambda rollout: rollout.weight_version, reverse=True)
    keep = ordered[:train_batch_per_replica]
    drop = ordered[train_batch_per_replica:]

    # weight_version is the policy step whose weights generated the rollout;
    # current_step is the trainer's policy step (same unit), so
    # current_step - weight_version is the rollout's policy lag in steps.
    min_allowed = current_step - allowed_outdated_steps
    stale_kept = sum(1 for rollout in keep if rollout.weight_version < min_allowed)
    if stale_kept:
        logger.warning(
            "Using %d/%d stale rollouts (min_allowed_version=%d, kept_versions=%s)",
            stale_kept,
            len(keep),
            min_allowed,
            [rollout.weight_version for rollout in keep],
        )

    for rollout in drop:
        path = str(rollout.completion)
        if not path:
            continue
        try:
            pathlib.Path(path).unlink()
        except FileNotFoundError:
            pass  # already reaped by another worker

    return keep
