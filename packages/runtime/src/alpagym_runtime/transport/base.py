# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rollout-side egress protocol for completed rollout artifacts.

The rollout-to-trainer data flow is one-directional: the rollout worker
produces episodes and writes them through an :class:`EpisodeWriter`, returning an
opaque handle. The trainer reads that handle back through the data packer
(``get_policy_input``): the NCCL trainer resolves it over its receiver, the disk
trainer reads the JSON artifact.
"""

from typing import Protocol, runtime_checkable

import redis

from alpagym_runtime.types import EpisodeOutput


@runtime_checkable
class EpisodeWriter(Protocol):
    """Rollout-side egress: persist produced episodes and free discarded ones."""

    def write(self, episode: EpisodeOutput) -> str:
        """Persist ``episode`` and return its opaque wire-format handle."""

    def release(self, handle: str, reason: str) -> None:
        """Discard a handle that will not be read.

        Idempotent: releasing an unknown or already-released handle is a no-op.
        ``reason`` is for logs and metrics only.
        """

    def start_cleanup(self, redis_client: redis.Redis) -> None:
        """Start any discard-cleanup subscriber this writer needs.

        Disk has no out-of-band cleanup, so its implementation is a no-op; the
        NCCL writer subscribes to the controller's discard channel.
        """

    def flush_pending_sends(self) -> None:
        """Block until in-flight egress drains, before a weight sync.

        Disk writes synchronously so its implementation is a no-op; the NCCL
        writer waits for already-started sender transfers. Held transfers that
        have not started may stay registered and ship after the weight sync.
        """

    def close(self) -> None:
        """Release writer resources (background threads, NCCL comms, etc.)."""
