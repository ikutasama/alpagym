# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Move nested tensor structures across devices."""

from typing import Any

import torch


def to_device_recursive(obj: Any, device: torch.device) -> Any:
    """Recursively move every tensor in ``obj`` to ``device``, preserving structure.

    Walks dicts, lists, and tuples; a tuple input returns a tuple and a list
    input returns a list. Non-tensor leaves pass through unchanged.

    Args:
        obj: A tensor, or an arbitrarily nested dict/list/tuple of tensors and
            other leaves.
        device: Target device for every tensor leaf.

    Returns:
        ``obj`` with identical structure and every tensor moved to ``device``.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: to_device_recursive(value, device) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [to_device_recursive(value, device) for value in obj]
        return tuple(moved) if isinstance(obj, tuple) else moved
    return obj
