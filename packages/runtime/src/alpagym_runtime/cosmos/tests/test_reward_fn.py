# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Cosmos-RL reward callback reading reward off the completion."""

from typing import cast

import pytest
import torch
from alpagym_runtime.cosmos.reward_fn import episode_reward_from_artifact
from alpagym_runtime.types import EpisodeMetrics, EpisodeOutput, PolicyOutput, RewardResult
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.policy.config import Config as CosmosConfig
from transformers import PreTrainedTokenizer


def _episode(reward: RewardResult | None) -> EpisodeOutput:
    """Build a single-step episode with the given (optional) reward."""
    return EpisodeOutput(
        scene_id="scene",
        session_uuid="session",
        num_steps=1,
        policy_outputs=(
            PolicyOutput(
                chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
                chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                chosen_dt_us=torch.tensor([0], dtype=torch.int64),
            ),
        ),
        reward=reward,
        metrics=EpisodeMetrics(aggregated={"progress": 0.5}),
    )


def _call(completion: EpisodeOutput) -> float:
    """Invoke the reward callback with throwaway cosmos arguments."""
    return episode_reward_from_artifact(
        completion,
        reference="unused-reference",
        prompt="unused-prompt",
        data_packer=cast(BaseDataPacker, object()),
        config=cast(CosmosConfig, object()),
        tokenizer=cast(PreTrainedTokenizer, object()),
    )


def test_episode_reward_raises_when_reward_missing() -> None:
    """A completion without reward fails loudly on attribute access."""
    with pytest.raises(AttributeError, match="total"):
        _call(_episode(reward=None))


def test_episode_reward_reads_precomputed_total() -> None:
    """A completion carrying a reward returns its total as a float."""
    assert _call(_episode(reward=RewardResult(total=2.5))) == 2.5
