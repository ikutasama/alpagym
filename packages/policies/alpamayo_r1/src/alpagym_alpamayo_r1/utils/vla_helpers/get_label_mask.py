# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from transformers import AutoTokenizer


def get_role_mask(
    tokenizer: AutoTokenizer,
    tokens: torch.Tensor | list[int],
    bos_token: str = "<|im_start|>",
    eos_token: str = "<|im_end|>",
    role: str = "assistant",
) -> torch.Tensor:
    """Generate a boolean mask indicating which tokens correspond to the assistant's response.

    Args:
        tokenizer (AutoTokenizer): The tokenizer used to convert tokens to IDs.
        tokens (torch.Tensor | list[int]): The sequence of token IDs.
        bos_token (str, optional): The beginning-of-sequence token. Defaults to "<|im_start|>".
        eos_token (str, optional): The end-of-sequence token. Defaults to "<|im_end|>".
        role (str, optional): The assistant role string. Defaults to "assistant".

    Returns:
        torch.Tensor: A boolean mask with True for assistant tokens, False otherwise.

    Reference:
        Adapted from an InternVL3.5 processor's assistant-token masking.
    """
    # Offsets: skip the bos + "assistant\n" (always 3 tokens) and include the eos (+1)
    # for supervision
    START_OFFSET = 3
    END_OFFSET = 1

    np_tokens = tokens.cpu().numpy() if isinstance(tokens, torch.Tensor) else np.array(tokens)

    # Retrieve token IDs for the markers and the role.
    bos_token_id = tokenizer.convert_tokens_to_ids(bos_token)
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    role_id = tokenizer.convert_tokens_to_ids(role)

    # Locate all positions where the start and end markers appear.
    start_indices = np.where(np_tokens == bos_token_id)[0]
    end_indices = np.where(np_tokens == eos_token_id)[0]

    # Initialize the mask with False values.
    masks = np.zeros_like(np_tokens, dtype=bool)
    assert len(start_indices) == len(end_indices), (
        f"Number of bos ({len(start_indices)}) does not match eos ({len(end_indices)})"
    )
    # For each pair of bos/eos, check if the role is 'assistant'
    # and apply the mask accordingly.
    for start, end in zip(start_indices, end_indices):
        if np_tokens[start + 1] == role_id:
            # Mask tokens from after the assistant header (start+3) to include the end
            # marker (end+1)
            masks[start + START_OFFSET : end + END_OFFSET] = True

    assert masks.shape == np_tokens.shape
    if isinstance(tokens, torch.Tensor):
        return torch.from_numpy(masks)
    else:
        return masks.tolist()
