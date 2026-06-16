# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport boundary for completed rollout artifacts.

A *transport* is the channel that carries episodes from the rollout process to
the trainer; a *transfer* is one episode crossing it. This package holds the
abstraction and its disk and NCCL implementations.

The rollout-to-trainer flow is one-directional. The rollout side's
:class:`EpisodeWriter` ``write`` returns an opaque string handle; the trainer
reads that handle back through the data packer (NCCL receiver or disk JSON).
The handle's format is owned by the implementation; callers must not parse it.
"""

from alpagym_runtime.transport.base import EpisodeWriter
from alpagym_runtime.transport.disk import DiskEpisodeWriter

__all__ = [
    "DiskEpisodeWriter",
    "EpisodeWriter",
]
