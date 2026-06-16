# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for typed I/O helpers in `alpagym_runtime.inference.types`."""

from typing import Any

import numpy as np
import pytest
import torch
from alpagym_runtime.inference.types import BatchedModelOutput


def _zero_pred_xyz(batch_size: int) -> torch.Tensor:
    """Synthetic `pred_xyz` shaped `[B, 1, 1, 1, 3]` for `unbind` tests."""
    return torch.zeros((batch_size, 1, 1, 1, 3), dtype=torch.float32)


def _identity_pred_rot(batch_size: int) -> torch.Tensor:
    """Synthetic `pred_rot` shaped `[B, 1, 1, 1, 3, 3]` for `unbind` tests."""
    return torch.eye(3, dtype=torch.float32).expand(batch_size, 1, 1, 1, 3, 3).clone()


def test_batched_model_output_unbind_slices_extra_per_model_output() -> None:
    """Batch-shaped `extra` leaves split with the corresponding prediction."""
    batch_size = 3
    token_log_probs = torch.arange(batch_size * 4, dtype=torch.float32).reshape(batch_size, 4)
    cot = ["cot-0", "cot-1", "cot-2"]
    model_outputs = BatchedModelOutput(
        pred_xyz=_zero_pred_xyz(batch_size),
        pred_rot=_identity_pred_rot(batch_size),
        extra={"token_log_probs": token_log_probs, "cot": cot},
    ).unbind()

    for index, model_output in enumerate(model_outputs):
        assert torch.equal(model_output.extra["token_log_probs"], token_log_probs[index])
        assert model_output.extra["cot"] == cot[index]


def test_batched_model_output_unbind_slices_extra_ndarray_per_model_output() -> None:
    """Per-batch `ndarray[B, ns, nj]` text leaves slice into `ndarray[ns, nj]`.

    Mirrors the shape released VLA/AR1/1.5 models produce via
    `np.array(extract_text_tokens(...)).reshape([B, ns, nj])`.
    """
    batch_size = 3
    cot = np.array(
        [
            [["b0-s0-j0", "b0-s0-j1"]],
            [["b1-s0-j0", "b1-s0-j1"]],
            [["b2-s0-j0", "b2-s0-j1"]],
        ],
        dtype=object,
    )
    assert cot.shape == (batch_size, 1, 2)

    model_outputs = BatchedModelOutput(
        pred_xyz=_zero_pred_xyz(batch_size),
        pred_rot=_identity_pred_rot(batch_size),
        extra={"cot": cot},
    ).unbind()

    for index, model_output in enumerate(model_outputs):
        sliced = model_output.extra["cot"]
        assert isinstance(sliced, np.ndarray)
        assert sliced.shape == (1, 2)
        np.testing.assert_array_equal(sliced, cot[index])


@pytest.mark.parametrize(
    "bad_extra",
    [
        pytest.param(
            {"k": torch.zeros((2, 4), dtype=torch.float32)}, id="tensor-wrong-leading-dim"
        ),
        pytest.param({"k": np.zeros((2, 4), dtype=np.float32)}, id="ndarray-wrong-leading-dim"),
        pytest.param({"k": torch.tensor(1.0)}, id="tensor-zero-dim"),
        pytest.param({"k": np.array(1.0, dtype=np.float32)}, id="ndarray-zero-dim"),
        pytest.param({"k": ["a", "b"]}, id="list-wrong-len"),
        pytest.param({"k": ("a", "b")}, id="tuple-wrong-len"),
    ],
)
def test_batched_model_output_unbind_rejects_non_per_output_extra(
    bad_extra: dict[str, Any],
) -> None:
    """`unbind()` fails fast on any `extra` leaf that does not equal `batch_size` along axis 0."""
    batch_size = 3
    batched = BatchedModelOutput(
        pred_xyz=_zero_pred_xyz(batch_size),
        pred_rot=_identity_pred_rot(batch_size),
        extra=bad_extra,
    )
    with pytest.raises(ValueError, match=r"extra\['k'\]"):
        batched.unbind()


def test_batched_model_output_unbind_rejects_unsupported_extra_leaf_type() -> None:
    """`unbind()` rejects leaf types outside Tensor / ndarray / list / tuple."""
    batch_size = 3
    batched = BatchedModelOutput(
        pred_xyz=_zero_pred_xyz(batch_size),
        pred_rot=_identity_pred_rot(batch_size),
        extra={"k": 42},
    )
    with pytest.raises(TypeError, match=r"extra\['k'\]"):
        batched.unbind()
