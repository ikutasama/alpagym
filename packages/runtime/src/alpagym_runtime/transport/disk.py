# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import uuid
from pathlib import Path
from typing import Any

import redis
import torch

from alpagym_runtime.types import EpisodeOutput


def write_episode(path: Path, episode: EpisodeOutput) -> None:
    """Write ``episode`` as a torch artifact at ``path``, creating parent dirs.

    Uses an atomic tmp-then-rename write so a preemption mid-write never leaves
    a partial file at the final path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(episode, tmp_path)
    os.replace(tmp_path, path)


def read_episode(handle: str | Path) -> EpisodeOutput:
    """Read a rollout episode result from a disk artifact handle."""
    return torch.load(Path(handle), weights_only=False, mmap=False)


class DiskEpisodeWriter:
    """Rollout-side disk egress: writes each episode as a torch artifact."""

    def __init__(self, artifacts_dir: Path):
        """Create a writer that writes artifacts under ``artifacts_dir``."""
        self._artifacts_dir = Path(artifacts_dir).resolve()

    def write(self, episode: EpisodeOutput) -> str:
        """Persist ``episode`` and return its file path as the handle.

        The handle carries a fresh ``uuid4`` suffix so two episodes that share a
        ``(scene_id, session_uuid)`` cannot overwrite each other's artifact.
        """
        filename = f"{episode.scene_id}_{episode.session_uuid}_{uuid.uuid4().hex}.pt"
        path = self._artifacts_dir / filename
        write_episode(path, episode)
        return str(path)

    def release(self, handle: str, reason: str) -> None:
        """Discard an artifact that will not be read."""
        del reason
        Path(handle).unlink(missing_ok=True)

    def start_cleanup(self, redis_client: redis.Redis) -> None:
        """No-op: the disk writer has no out-of-band discard channel."""
        del redis_client

    def flush_pending_sends(self) -> None:
        """No-op: disk writes are synchronous, so nothing is ever pending."""

    def close(self) -> None:
        """No-op: disk writer holds no live resources."""
