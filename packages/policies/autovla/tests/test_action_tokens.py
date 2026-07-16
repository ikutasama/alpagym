# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AutoVLA action-token vocabulary handling."""

from __future__ import annotations

import pytest
import torch
from alpagym_autovla.action_tokens import action_token_mask, ensure_action_token_layout


class FakeTokenizer:
    """Small tokenizer stub with the methods AutoVLA uses."""

    unk_token_id = -1

    def __init__(self, size: int, existing: dict[str, int] | None = None) -> None:
        self.vocab = {f"tok_{i}": i for i in range(size)}
        if existing:
            self.vocab.update(existing)

    def __len__(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def add_tokens(self, tokens: list[str], special_tokens: bool = False) -> int:
        del special_tokens
        start = len(self.vocab)
        for offset, token in enumerate(tokens):
            self.vocab[token] = start + offset
        return len(tokens)


def test_ensure_action_token_layout_adds_raw_qwen_block_at_start_id() -> None:
    """A raw base tokenizer is safe only when appending starts at action_start_id."""
    tokenizer = FakeTokenizer(size=10)

    ids = ensure_action_token_layout(
        tokenizer,
        action_start_id=10,
        n_bins=4,
        source="test",
    )

    torch.testing.assert_close(ids, torch.tensor([10, 11, 12, 13]))
    assert tokenizer.convert_tokens_to_ids("<action_3>") == 13


def test_ensure_action_token_layout_rejects_shifted_raw_tokenizer() -> None:
    """Appending action tokens after any unexpected vocab growth would misalign weights."""
    tokenizer = FakeTokenizer(size=11)

    with pytest.raises(ValueError, match="expected base length 10"):
        ensure_action_token_layout(
            tokenizer,
            action_start_id=10,
            n_bins=4,
            source="test",
        )


def test_ensure_action_token_layout_rejects_partial_action_block() -> None:
    """A partially present action block is ambiguous and should fail fast."""
    tokenizer = FakeTokenizer(size=10, existing={"<action_0>": 10})

    with pytest.raises(ValueError, match="partial AutoVLA action-token block"):
        ensure_action_token_layout(
            tokenizer,
            action_start_id=10,
            n_bins=4,
            source="test",
        )


def test_action_token_mask_matches_exact_ids_only() -> None:
    """The mask must not treat every id above action_start_id as a valid action."""
    token_ids = torch.tensor([[9, 10, 12, 14]])
    action_ids = torch.tensor([10, 11, 12, 13])

    mask = action_token_mask(token_ids, action_ids)

    torch.testing.assert_close(mask, torch.tensor([[False, True, True, False]]))
