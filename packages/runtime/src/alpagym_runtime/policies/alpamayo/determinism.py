# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic batched-inference helpers for Alpamayo adapters."""

import os
import random
from typing import Any

import numpy as np
import torch

from alpagym_runtime.inference.types import BatchedModelOutput


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic(seed: int = 42) -> None:
    """Configure deterministic PyTorch runtime behavior."""
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed_everything(seed)


def seeded_diffusion_kwargs(seed: int, device: torch.device) -> dict[str, Any]:
    """Seed global RNGs and return diffusion kwargs carrying a per-device generator."""
    seed_everything(seed)
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    return {"generator": generator}


def merge_batched_outputs(outputs: list[BatchedModelOutput]) -> BatchedModelOutput:
    """Concatenate B=1 adapter outputs back into the public batched shape."""
    if not outputs:
        raise ValueError("merge_batched_outputs requires at least one output")
    logprob = None
    if all(output.logprob is not None for output in outputs):
        logprob = torch.cat(
            [output.logprob for output in outputs if output.logprob is not None],
            dim=0,
        )
    elif any(output.logprob is not None for output in outputs):
        raise ValueError("deterministic adapter outputs mixed present/missing logprob")
    return BatchedModelOutput(
        pred_xyz=torch.cat([output.pred_xyz for output in outputs], dim=0),
        pred_rot=torch.cat([output.pred_rot for output in outputs], dim=0),
        logprob=logprob,
        extra=_merge_extra([output.extra for output in outputs]),
    )


def _merge_extra(extras: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-row extras when a B-leading representation is obvious."""
    if not any(extras):
        return {}
    merged: dict[str, Any] = {}
    for key in set().union(*(extra.keys() for extra in extras)):
        values = [extra[key] for extra in extras if key in extra]
        if len(values) != len(extras):
            merged[key] = values
        elif all(value is None for value in values):
            merged[key] = None
        elif all(isinstance(value, torch.Tensor) for value in values):
            first = values[0]
            if first.ndim > 0 and all(value.shape[0] == 1 for value in values):
                merged[key] = torch.cat(values, dim=0)
            else:
                merged[key] = values
        elif all(isinstance(value, list) for value in values):
            merged[key] = [item for value in values for item in value]
        elif all(isinstance(value, dict) for value in values):
            merged[key] = _merge_extra(values)
        else:
            merged[key] = values
    return merged
