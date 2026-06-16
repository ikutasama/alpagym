# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the recursive tensor device mover."""

import torch
from alpagym_runtime.tensor_utils import to_device_recursive


def test_to_device_recursive_preserves_structure_and_tuple_type() -> None:
    """to_device_recursive walks dicts and tuples without changing the shape."""
    nested = {
        "tensor": torch.randn(2, 2),
        "tuple_values": (torch.randn(1), "x"),
    }
    moved = to_device_recursive(nested, torch.device("cpu"))
    assert moved["tensor"].device.type == "cpu"
    assert isinstance(moved["tuple_values"], tuple)
    assert moved["tuple_values"][0].device.type == "cpu"
    assert moved["tuple_values"][1] == "x"
