# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic-inference helpers shared across AlpaGym policies."""

import os
import random

import numpy as np
import torch


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
    # cuBLAS requires a fixed workspace size for reproducible GEMMs under
    # ``use_deterministic_algorithms``; ":4096:8" is the value documented for
    # deterministic cuBLAS (see torch.use_deterministic_algorithms).
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed_everything(seed)
