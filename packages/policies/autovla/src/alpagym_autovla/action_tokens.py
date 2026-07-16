"""AutoVLA action-token layout checks shared by rollout and trainer."""

from __future__ import annotations

from typing import Any

import torch


def action_token_strings(n_bins: int) -> list[str]:
    """Return the canonical AutoVLA action-token strings."""
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    return [f"<action_{index}>" for index in range(n_bins)]


def ensure_action_token_layout(
    tokenizer: Any,
    *,
    action_start_id: int,
    n_bins: int,
    source: str,
) -> torch.Tensor:
    """Ensure tokenizer ids and return the exact action-token id tensor.

    AutoVLA checkpoints are trained with a fixed discrete action-token block:
    ``<action_0>`` must map to ``action_start_id``, ``<action_1>`` to
    ``action_start_id + 1``, and so on. A raw Qwen tokenizer is accepted only
    when its current length is exactly ``action_start_id``; in that case appending
    AutoVLA tokens produces the trained row layout. Any partial or shifted layout
    is rejected before rollout/training starts.
    """
    expected_tokens = action_token_strings(n_bins)
    missing_before = [
        token for token in expected_tokens if _token_to_id(tokenizer, token) is None
    ]
    if missing_before:
        _add_missing_action_tokens(
            tokenizer,
            missing_before,
            action_start_id=action_start_id,
            n_bins=n_bins,
            source=source,
        )

    expected = list(range(action_start_id, action_start_id + n_bins))
    actual = [_token_to_id(tokenizer, token) for token in expected_tokens]

    missing = [
        token
        for token, token_id in zip(expected_tokens, actual, strict=True)
        if token_id is None
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{source}: tokenizer is missing AutoVLA action tokens ({preview}). "
            "Use an AutoVLA model bundle/tokenizer that already contains the "
            "trained action-token vocabulary; do not point model.path at a raw "
            "Qwen2.5-VL checkpoint."
        )

    actual_ids = [int(token_id) for token_id in actual if token_id is not None]
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError(f"{source}: AutoVLA action token ids contain duplicates")
    if actual_ids != expected:
        raise ValueError(
            f"{source}: AutoVLA action token ids do not match configured layout. "
            f"Expected <action_0>..<action_{n_bins - 1}> to occupy "
            f"[{expected[0]}, {expected[-1]}], but got first/last ids "
            f"{actual_ids[0]}/{actual_ids[-1]}. This usually means model.path "
            "points at a raw Qwen tokenizer or an incompatible AutoVLA checkpoint."
        )
    return torch.tensor(actual_ids, dtype=torch.int64)


def action_token_mask(token_ids: torch.Tensor, action_token_ids: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask selecting exact AutoVLA action-token ids."""
    if action_token_ids.device != token_ids.device:
        action_token_ids = action_token_ids.to(token_ids.device)
    return torch.isin(token_ids, action_token_ids)


def _token_to_id(tokenizer: Any, token: str) -> int | None:
    """Return one token id, treating unknown-token aliases as missing."""
    token_id = None
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None and hasattr(tokenizer, "get_vocab"):
        token_id = tokenizer.get_vocab().get(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unk_id:
        return None
    return int(token_id)


def _add_missing_action_tokens(
    tokenizer: Any,
    missing_tokens: list[str],
    *,
    action_start_id: int,
    n_bins: int,
    source: str,
) -> None:
    """Append a full missing action-token block when the base vocab is aligned."""
    all_tokens = action_token_strings(n_bins)
    if missing_tokens != all_tokens:
        preview = ", ".join(missing_tokens[:5])
        raise ValueError(
            f"{source}: tokenizer has a partial AutoVLA action-token block "
            f"({preview}); refusing to infer a shifted action vocabulary."
        )
    tokenizer_len = _tokenizer_len(tokenizer)
    if tokenizer_len != action_start_id:
        raise ValueError(
            f"{source}: tokenizer is missing AutoVLA action tokens but has length "
            f"{tokenizer_len}, expected base length {action_start_id}. Use the "
            "matching AutoVLA tokenizer/checkpoint pair."
        )
    if not hasattr(tokenizer, "add_tokens"):
        raise TypeError(f"{source}: tokenizer cannot add missing AutoVLA action tokens")
    added = int(tokenizer.add_tokens(all_tokens, special_tokens=False))
    if added != n_bins:
        raise ValueError(
            f"{source}: tokenizer.add_tokens added {added} action tokens, expected {n_bins}"
        )


def _tokenizer_len(tokenizer: Any) -> int:
    """Return tokenizer length with a vocab fallback for lightweight tests."""
    try:
        return int(len(tokenizer))
    except TypeError:
        if hasattr(tokenizer, "get_vocab"):
            return len(tokenizer.get_vocab())
        raise
