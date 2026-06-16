# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos-RL reward callback that reads each episode's reward on the rollout worker."""

from typing import Any

from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.policy.config import Config as CosmosConfig
from transformers import PreTrainedTokenizer

from alpagym_runtime.types import EpisodeOutput


def episode_reward_from_artifact(
    episode_output: EpisodeOutput,
    reference: str | None,
    prompt: str | list[Any],
    data_packer: BaseDataPacker,
    config: CosmosConfig,
    tokenizer: PreTrainedTokenizer,
) -> float:
    """Return the precomputed reward read straight off the in-memory episode.

    Under ``train.non_text=True`` the rollout dispatcher passes the live
    ``EpisodeOutput``, which already carries its computed reward, as the
    completion (the first positional argument of the cosmos reward callback).
    """
    del reference, prompt, data_packer, config, tokenizer
    return float(episode_output.reward.total)
