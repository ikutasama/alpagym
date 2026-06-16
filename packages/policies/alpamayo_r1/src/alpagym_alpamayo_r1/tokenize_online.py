# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Online tokenization for the AlpaGym Alpamayo R1 / 1.5 ExpertModel.

Runs the AlpaGym data pipeline (``AlpaQwenPacker`` + ``build_conversation`` +
``sort_images_by_camera_ids``) on raw model input to produce the
``tokenized_data`` dict (``input_ids``, ``attention_mask``, ``pixel_values``,
``image_grid_thw``, optional ``labels``) the ExpertModel forward expects.
The packer is lazily built from ``model.config`` and cached on the model
instance so it is shared across rollout and trainer-side calls.
"""

from __future__ import annotations

from typing import Any

from alpagym_alpamayo_r1.data.chat_template.conversation import build_conversation
from alpagym_alpamayo_r1.data.cosmos.datapacker.alpa_qwen_sft import AlpaQwenPacker
from alpagym_alpamayo_r1.data.data_utils import sort_images_by_camera_ids


def _build_packer(config: Any) -> AlpaQwenPacker:
    """Build an ``AlpaQwenPacker`` aligned with the data pipeline's tokenizer extension."""
    packer = AlpaQwenPacker()
    packer._initialize_from_params(
        vlm_name_or_path=config.vlm_name_or_path,
        traj_vocab_size=config.traj_vocab_size,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    packer._padding_side = config.padding_side
    return packer


def _get_packer(model: Any) -> AlpaQwenPacker:
    """Lazy-create and cache an ``AlpaQwenPacker`` on the model instance."""
    if not hasattr(model, "_alpagym_packer"):
        model._alpagym_packer = _build_packer(model.config)
    return model._alpagym_packer


def tokenize_for_generation(
    model: Any, data: dict[str, Any], last_component: str = "traj_future"
) -> dict[str, Any]:
    """Tokenize raw model input into generation-mode ``tokenized_data``.

    Builds the AlpaGym chat template, runs ``AlpaQwenPacker.sft_process_sample``
    per element, then collates into the batched ``tokenized_data`` dict the
    ExpertModel forward consumes. Output tensors land on the same device as
    ``data['image_frames']``.

    Args:
        model: ExpertModel / ExpertModelRL instance. ``model.config`` provides
            the packer params and the AlpaGym tokenization knobs
            (``tokens_per_history_traj``, ``tokens_per_future_traj``,
            ``include_camera_ids``, ``include_frame_nums``,
            ``legacy_inference_image_input_format``). The packer is cached as
            ``model._alpagym_packer`` after the first call.
        data: Batched input with at least ``image_frames`` and ``camera_indices``.
            When ``legacy_inference_image_input_format`` is set on the config,
            ``image_frames`` is treated as ``[-1, 1]`` and rescaled to
            ``[0, 1]`` in-place before tokenization.
        last_component: ``'traj_future'`` to stop at the future-trajectory turn,
            or ``'cot'`` to keep the CoT turn as the last component (both
            future and CoT remain in the prompt).

    Returns:
        ``tokenized_data`` dict with ``input_ids``, ``attention_mask``,
        ``pixel_values``, ``image_grid_thw`` (plus ``labels`` when the packer
        emits one), on the input device.
    """
    if model.config.legacy_inference_image_input_format:
        data["image_frames"] = (data["image_frames"] + 1.0) / 2.0

    if last_component == "traj_future":
        components_order = ["image", "traj_history", "prompt", "traj_future"]
        components_prompt = ["traj_future"]
    elif last_component == "cot":
        components_order = ["image", "traj_history", "prompt", "cot"]
        components_prompt = ["cot", "traj_future"]
    else:
        raise ValueError(f"Unsupported last_component: {last_component}")

    packer = _get_packer(model)
    batch_size = data["image_frames"].shape[0]
    processed_samples = []
    for i in range(batch_size):
        sample: dict[str, Any] = {
            "image_frames": data["image_frames"][i],
            "camera_indices": data["camera_indices"][i],
        }
        image_keys = ("image_frames", "camera_indices")
        sample.update(sort_images_by_camera_ids({k: sample[k] for k in image_keys if k in sample}))
        sample["messages"] = build_conversation(
            data=sample,
            num_tokens_per_history_traj=model.config.tokens_per_history_traj,
            num_tokens_per_future_traj=model.config.tokens_per_future_traj,
            components_order=components_order,
            components_prompt=components_prompt,
            generation_mode=True,
            include_camera_ids=model.config.include_camera_ids,
            frame_label="frame_num" if model.config.include_frame_nums else "none",
        )
        sample["generation_mode"] = True
        processed_samples.append(packer.sft_process_sample(sample))

    device = data["image_frames"].device
    tokenized = packer.sft_collate_fn(processed_samples)["tokenized_data"]
    return {k: v.to(device, non_blocking=(device.type != "cpu")) for k, v in tokenized.items()}
