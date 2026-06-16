# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Literal, Mapping

import torch

_logger = logging.getLogger(__name__)


def recursive_collate_fn(batch: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively collate a batch of hierarchical dictionaries, lists, or tensors."""
    if isinstance(batch[0], torch.Tensor):
        # Stack tensors directly
        return torch.stack(batch, dim=0)
    elif isinstance(batch[0], dict):
        # Recursively collate dictionaries
        return {key: recursive_collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], list):
        # Recursively collate lists
        return [recursive_collate_fn(items) for items in zip(*batch)]
    else:
        # Default case for other data types (e.g., scalars, strings)
        return batch


# Sequence-dim keys inside ``tokenized_data`` that must stay aligned with each
# other when truncating a sample (``pixel_values`` / ``image_grid_thw`` are
# image-axis, not sequence-axis, and must NOT be truncated here).
_TOKENIZED_SEQ_KEYS = ("input_ids", "attention_mask", "labels")


def _right_truncate_sample(sample: dict[str, Any], max_len: int) -> dict[str, Any]:
    """Right-truncate seq-dim tensors of ``sample`` to at most ``max_len``.

    Returns the sample unchanged when no truncation is needed; otherwise returns
    a shallow-copied sample (input dict not mutated) with the relevant tensors
    sliced. Truncation is always from the right so the prefix (system prompt +
    image-placeholder tokens) is preserved and stays aligned with image
    features.
    """
    tokenized = sample.get("tokenized_data")
    if tokenized is None or tokenized["input_ids"].shape[0] <= max_len:
        return sample
    orig_len = tokenized["input_ids"].shape[0]
    _logger.warning(
        "Right-truncating sample from seq_len=%d to max_len=%d (dropped %d tokens). "
        "This can desync image-placeholder tokens from image features if the "
        "truncated tail contained image tokens — increase model_max_length or "
        "filter long samples upstream.",
        orig_len,
        max_len,
        orig_len - max_len,
    )
    new_tokenized = dict(tokenized)
    for k in _TOKENIZED_SEQ_KEYS:
        if k in new_tokenized:
            new_tokenized[k] = new_tokenized[k][:max_len]
    new_sample = dict(sample)
    new_sample["tokenized_data"] = new_tokenized
    if "labels_mask" in new_sample:
        new_sample["labels_mask"] = new_sample["labels_mask"][:max_len]
    return new_sample


def qwen_collate_processed_samples(
    samples: list[Any],
    pad_token_id: int,
    ignore_label_id: int,
    padding_side: Literal["left", "right"],
    max_len: int | None = None,
) -> dict[str, Any]:
    """Collate the VLM processed samples into a batch.

    Args:
        samples (list[Any]): The VLM processed samples.
        pad_token_id (int): The pad token id used to pad when seq length is different.
        ignore_label_id (int): The ignore label id for padded tokens.
        padding_side (str): The padding side for padding tensors.
        max_len (int | None): If provided, force all sequence-length tensors
            (``input_ids``, ``attention_mask``, ``labels``, ``labels_mask``) to
            exactly this length. Samples longer than ``max_len`` are
            right-truncated per sample BEFORE padding (regardless of
            ``padding_side``) so the prefix is preserved; shorter rows are then
            padded up on ``padding_side``. If None, fall back to padding to the
            longest sequence in the batch (variable-shape).

    Returns:
        dict[str, Any]: The collated batch.
    """
    assert padding_side in ("left", "right"), f"Padding side {padding_side} is not supported"
    if max_len is not None:
        samples = [_right_truncate_sample(s, max_len) for s in samples]
    batch = {"tokenized_data": {}}

    # collate tokenized data if it exists
    if "tokenized_data" in samples[0]:
        batch["tokenized_data"]["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            [sample["tokenized_data"]["input_ids"] for sample in samples],
            batch_first=True,
            padding_value=pad_token_id,
            padding_side=padding_side,
        )
        batch["tokenized_data"]["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            [sample["tokenized_data"]["attention_mask"] for sample in samples],
            batch_first=True,
            padding_value=False,
            padding_side=padding_side,
        )
        if "labels" in samples[0]["tokenized_data"]:
            batch["tokenized_data"]["labels"] = torch.nn.utils.rnn.pad_sequence(
                [sample["tokenized_data"]["labels"] for sample in samples],
                batch_first=True,
                padding_value=ignore_label_id,
                padding_side=padding_side,
            )
        # shape: (num_img, d)
        if "pixel_values" in samples[0]["tokenized_data"]:
            batch["tokenized_data"]["pixel_values"] = torch.cat(
                [sample["tokenized_data"]["pixel_values"] for sample in samples], dim=0
            )
        # shape: (num_img, 3)
        if "image_grid_thw" in samples[0]["tokenized_data"]:
            batch["tokenized_data"]["image_grid_thw"] = torch.cat(
                [sample["tokenized_data"]["image_grid_thw"] for sample in samples], dim=0
            )

    if "labels_mask" in samples[0]:
        batch["labels_mask"] = torch.nn.utils.rnn.pad_sequence(
            [sample["labels_mask"] for sample in samples],
            batch_first=True,
            padding_value=False,
            padding_side=padding_side,
        )

    # Keys that depend on camera count and can vary across samples
    # (e.g. with camera_subsample_weights), so they must be kept as lists.
    for k in ["image_frames", "camera_indices", "absolute_timestamps", "relative_timestamps"]:
        if k in samples[0]:
            batch[k] = [sample[k] for sample in samples]

    # for camera model dict, we don't stack them, we just keep them as a list
    if "camera_model_dict" in samples[0]:
        batch["camera_model_dict"] = [sample["camera_model_dict"] for sample in samples]

    # use standard collation fn to collate the rest of the keys
    keys = [k for k in samples[0].keys() if k not in batch.keys()]
    others = recursive_collate_fn([{k: sample[k] for k in keys} for sample in samples])
    batch.update(others)

    if max_len is not None:
        tokenized = batch["tokenized_data"]
        pad_specs = [
            (tokenized, "input_ids", pad_token_id),
            (tokenized, "attention_mask", 0),
            (tokenized, "labels", ignore_label_id),
            (batch, "labels_mask", 0),
        ]
        for container, key, pad_value in pad_specs:
            if key not in container:
                continue
            tensor = container[key]
            pad_len = max_len - tensor.shape[1]
            if pad_len == 0:
                continue
            pad = (pad_len, 0) if padding_side == "left" else (0, pad_len)
            container[key] = torch.nn.functional.pad(tensor, pad, value=pad_value)

    return batch
