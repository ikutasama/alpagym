# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused unit tests for `CameraFrameBuffer`."""

import torch
from alpagym_runtime.policies.alpamayo.buffers import CameraFrameBuffer


def _frame(value: int, shape: tuple[int, int, int] = (3, 4, 5)) -> torch.Tensor:
    """Return a CHW uint8 tensor filled with `value` (mod 256) for identity checks."""
    return torch.full(shape, value % 256, dtype=torch.uint8)


def test_wrap_around_rotates_ring_in_chronological_order() -> None:
    """Wrapping past `capacity` drops the oldest frame and rotates remaining ones."""
    ring = CameraFrameBuffer(
        capacity=3,
        frame_shape=(3, 2, 2),
        camera_name="cam",
        frame_device=torch.device("cpu"),
    )
    for i in range(4):
        ring.add(_frame(i + 1, shape=(3, 2, 2)), 100 * (i + 1))
    assert len(ring) == 3
    view = ring.frames_ordered
    assert view[0, 0, 0, 0].item() == 2
    assert view[1, 0, 0, 0].item() == 3
    assert view[2, 0, 0, 0].item() == 4
    assert ring.tstamps_ordered.tolist() == [200, 300, 400]

    ring.add(_frame(5, shape=(3, 2, 2)), 500)
    assert ring.tstamps_ordered.tolist() == [300, 400, 500]
