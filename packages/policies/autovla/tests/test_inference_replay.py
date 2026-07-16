# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay-payload contract tests for the AutoVLA inference adapter."""

from __future__ import annotations

import torch
from alpagym_autovla.inference_model import AutoVLAInferenceModel
from alpagym_runtime.inference.types import ModelInput, ModelOutput
from alpagym_runtime.replay import ActionSelection


def _model_input() -> ModelInput:
    """Build the minimum typed input needed by replay packing."""
    return ModelInput(
        ego_history_xyz=torch.zeros((4, 3), dtype=torch.float32),
        ego_history_rot=torch.eye(3, dtype=torch.float32).repeat(4, 1, 1),
        camera_frames=torch.zeros((12, 3, 4, 4), dtype=torch.uint8),
        camera_indices=torch.zeros((12,), dtype=torch.int64),
        relative_timestamps=torch.zeros((12,), dtype=torch.int64),
        route_xy=torch.zeros((20, 2), dtype=torch.float32),
    )


def test_build_policy_replay_data_keeps_full_completion_ids() -> None:
    """Per-output completion_ids are already [T] and must not be indexed to comp[0]."""
    adapter = object.__new__(AutoVLAInferenceModel)
    completion_ids = torch.tensor([101, 102, 103, 104], dtype=torch.int64)
    model_output = ModelOutput(
        pred_xyz=torch.zeros((1, 1, 10, 3), dtype=torch.float32),
        pred_rot=torch.eye(3, dtype=torch.float32).reshape(1, 1, 1, 3, 3).repeat(1, 1, 10, 1, 1),
        logprob=torch.tensor([[-3.5]], dtype=torch.float32),
        extra={
            "action_token_ids": torch.tensor([[[151665, 151666]]], dtype=torch.int64),
            "completion_ids": completion_ids,
        },
    )

    replay = adapter.build_policy_replay_data(
        model_input=_model_input(),
        model_output=model_output,
        action_selection=ActionSelection(set_ix=0, sample_ix=0),
    )

    assert torch.equal(replay.payload["completion_ids"], completion_ids)
    assert replay.old_logprob is not None
    torch.testing.assert_close(replay.old_logprob, torch.tensor(-3.5))
