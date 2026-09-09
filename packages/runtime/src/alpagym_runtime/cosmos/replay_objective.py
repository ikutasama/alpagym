# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay objective helpers for the AlpaGym Cosmos trainer."""

from __future__ import annotations

import logging
import torch

logger = logging.getLogger(__name__)


def assert_replay_shapes(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    kl_div: torch.Tensor | None,
) -> None:
    """Validate shapes and sanitize non-finite values in-place.

    Shape mismatches still raise ``ValueError`` (a real bug). Non-finite
    values in ``old_logprobs`` or ``new_logprobs`` are replaced with zeros
    and logged as warnings, so a single bad rollout does not crash the
    entire training process. Downstream PPO/KL code treats zero-logprob
    rows neutrally when they are also marked as padding.
    """
    if new_logprobs.shape != old_logprobs.shape:
        raise ValueError(
            f"new log_probs shape {tuple(new_logprobs.shape)} != old_logprobs "
            f"shape {tuple(old_logprobs.shape)}"
        )
    if advantages.shape != old_logprobs.shape:
        raise ValueError(
            f"advantages shape {tuple(advantages.shape)} != old_logprobs "
            f"shape {tuple(old_logprobs.shape)}"
        )
    if kl_div is not None and kl_div.shape != old_logprobs.shape:
        raise ValueError(
            f"kl_div shape {tuple(kl_div.shape)} != old_logprobs shape {tuple(old_logprobs.shape)}"
        )
    if not torch.isfinite(new_logprobs).all():
        n_bad = int((~torch.isfinite(new_logprobs)).sum().item())
        logger.warning(
            "Sanitizing %d non-finite new_logprobs values to 0.0", n_bad
        )
        new_logprobs[~torch.isfinite(new_logprobs)] = 0.0
    if not torch.isfinite(old_logprobs).all():
        n_bad = int((~torch.isfinite(old_logprobs)).sum().item())
        logger.warning(
            "Sanitizing %d non-finite old_logprobs values to 0.0", n_bad
        )
        old_logprobs[~torch.isfinite(old_logprobs)] = 0.0
    if kl_div is not None and not torch.isfinite(kl_div).all():
        n_bad = int((~torch.isfinite(kl_div)).sum().item())
        logger.warning(
            "Sanitizing %d non-finite kl_div values to 0.0", n_bad
        )
        kl_div[~torch.isfinite(kl_div)] = 0.0


def compute_ppo_surrogate(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    ratio_clip_low: float,
    ratio_clip_high: float,
    is_padding: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the PPO clipped surrogate loss and importance ratio.

    The loss is averaged over valid (non-padding) rows, matching how
    ``compute_kl_penalty`` reduces, so the per-sample gradient scale and the
    policy-vs-KL balance do not depend on how many padding rows a shuffled
    minibatch happens to contain. An all-padding minibatch yields a
    graph-connected zero so every DP worker still backprops in lockstep.
    """
    # Bound the exponent input for numeric stability; PPO ratio clipping is below.
    log_ratio = (new_logprobs - old_logprobs).clamp(min=-5.0, max=5.0)
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - ratio_clip_low, 1.0 + ratio_clip_high) * advantages
    per_row = -torch.min(surr1, surr2)
    valid = (~is_padding).to(per_row.dtype)
    return (per_row * valid).sum() / valid.sum().clamp_min(1.0), ratio


def compute_kl_penalty(
    kl_div: torch.Tensor | None,
    is_padding: torch.Tensor,
    kl_beta: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute KL penalty over valid rows, returning zero when KL is disabled."""
    if kl_div is None or kl_beta <= 0.0:
        return torch.tensor(0.0, device=device)
    valid_kl = kl_div[~is_padding]
    if valid_kl.numel() == 0:
        return torch.tensor(0.0, device=device)
    return valid_kl.mean() * kl_beta
