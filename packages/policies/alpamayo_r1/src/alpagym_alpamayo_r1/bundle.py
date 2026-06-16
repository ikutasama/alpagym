# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Alpamayo R1 / 1.5 policy bundle entry point."""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Callable

from alpagym_runtime.policies.registry import PolicyBundle

logger = logging.getLogger(__name__)


def install_alpamayo_r1_runtime_bridge() -> None:
    """Wire R1 into cosmos: register the HF model type, the cosmos wrapper, and the BT patch."""
    # Importing the recipe cosmos wrapper runs ``AutoConfig.register`` /
    # ``AutoModel.register`` for the ``alpamayo_reasoning_vla_expert`` HF
    # model_type and exposes ``ExpertModelCosmos``.
    import alpamayo1_x_rl.models.expert_model.cosmos_wrapper  # noqa: F401

    _patch_expert_model_cosmos_bt_flatten()


def _patch_expert_model_cosmos_bt_flatten() -> None:
    """Adapt ``ExpertModelCosmos.forward`` to the alpagym trainer contract.

    Two adjustments:

    1. Rebuild ``tokenized_data`` from raw ``image_frames`` /
       ``camera_indices`` at trainer time. The rollout-side packer
       persists only those raw inputs (artifact stays ~25x smaller than
       caching the Qwen processor output), so the trainer reruns
       ``_get_generation_mode_tokenized_data_online`` on the BT-batched
       inputs to recover the same ``pixel_values`` / ``image_grid_thw``
       / ``input_ids`` the rollout used — bitwise-equivalent because
       the processor is deterministic given the same images.
    2. Alias the wrapper's ``logits`` → ``log_probs`` and squeeze ``[B, 1]``
       to the ``[BT]`` the trainer expects. ``ExpertModelCosmos`` returns the
       per-sample CFM SDE log-probability under a ``logits`` key (a misnomer
       in that wrapper); it is already a normalized log-density, so there is
       no ``log_softmax`` / ``- logsumexp`` step — just a rename and reshape.
    """
    import torch
    from alpamayo1_x_rl.models.expert_model.cosmos_wrapper import ExpertModelCosmos

    from alpagym_alpamayo_r1.tokenize_online import tokenize_for_generation

    original = ExpertModelCosmos.forward
    if getattr(original, "_alpagym_bt_flatten_patch", False):
        return

    def patched_forward(
        self: Any,
        image_frames: torch.Tensor,
        camera_indices: torch.Tensor,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
        samples_list: torch.Tensor,
        timesteps: torch.Tensor,
        teacher_model: Any = None,
        noise_level: torch.Tensor | None = None,
        vlm_generated_ids: torch.Tensor | None = None,
        return_log_prob: bool = True,
    ) -> Any:
        # ``AlpagymDataPacker`` always injects ``return_log_prob=True`` (cosmos/
        # packer.py:178); ``cfm_logprob_sde`` is the only path here and always
        # returns the log_prob, so accept and ignore the flag.
        del return_log_prob
        # CoT replay needs a different ``last_component`` and pads / masks the
        # generated tokens to align across collated steps. Neither is in place
        # yet, so reject the CoT path until the rollout-side ``last_component``
        # is persisted and the packer supports ragged ``vlm_generated_ids``.
        if vlm_generated_ids is not None:
            raise NotImplementedError(
                "Alpamayo R1 CoT replay is not supported yet — bundle rebuilds "
                "tokenized_data with last_component='traj_future' and the packer "
                "torch.stacks generated ids without length padding. Reject until "
                "both paths are wired through."
            )
        # Trainer-side mirror of ``load_inference_model``: feed [-1, 1] frames
        # into ``tokenize_for_generation`` and rely on the model's internal
        # [-1, 1] -> [0, 1] rescale, so the BT-batched ``pixel_values`` here
        # match the rollout's bit-for-bit. Without this force, ckpts whose
        # ``legacy_inference_image_input_format=False`` would skip the rescale
        # on the trainer path while the rollout path forces it on, yielding
        # different ``pixel_values`` per the docstring's "bitwise-equivalent"
        # claim.
        self.expert_model.config.legacy_inference_image_input_format = True
        tokenized_data = tokenize_for_generation(
            self.expert_model,
            {
                "image_frames": image_frames.to(torch.float32) / 127.5 - 1.0,
                "camera_indices": camera_indices,
            },
            last_component="traj_future",
        )
        forward_kwargs: dict[str, Any] = {
            "tokenized_data": tokenized_data,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "samples_list": samples_list,
            "timesteps": timesteps,
        }
        if noise_level is not None:
            forward_kwargs["noise_level"] = noise_level
        if vlm_generated_ids is not None:
            forward_kwargs["vlm_generated_ids"] = vlm_generated_ids
        result = original(self, teacher_model=teacher_model, **forward_kwargs)
        # ``logits`` [B, 1] is the wrapper's misnamed CFM SDE log-prob (already
        # normalized); rename to log_probs and squeeze to [BT].
        return {**result, "log_probs": result["logits"].squeeze(-1)}

    patched_forward._alpagym_bt_flatten_patch = True  # type: ignore[attr-defined]
    ExpertModelCosmos.forward = patched_forward


def setup_tokenizer(config: Any) -> Any | None:
    """No tokenizer override; just install the bridge before super-init."""
    del config
    install_alpamayo_r1_runtime_bridge()
    return None


def build_data_packer(run_config: Any, cosmos_role: str | None) -> Any:
    """Build the R1 replay data packer."""
    from alpagym_runtime.cosmos.packer import build_alpagym_data_packer

    install_alpamayo_r1_runtime_bridge()
    return build_alpagym_data_packer(
        run_config=run_config,
        cosmos_role=cosmos_role,
        build_model_inputs=build_model_inputs(run_config),
    )


def load_inference_model(
    run_config: Any,
    device: Any,
    dtype: Any,
) -> Any:
    """Load an Alpamayo R1 / 1.5 bundle and return its inference adapter."""
    import torch
    from alpagym_runtime.policies.model_bundle import resolve_model_bundle_path

    from alpagym_alpamayo_r1.inference_model import AlpamayoR1InferenceModel

    model_config = run_config.policy.model
    if dtype != torch.bfloat16:
        raise ValueError(f"Alpamayo R1 expects dtype=bfloat16; got {dtype!r}.")

    from alpamayo1_x_rl.models.expert_model.model import ExpertModelRL

    bundle_dir = resolve_model_bundle_path(Path(model_config.path))
    model = ExpertModelRL.from_pretrained(
        bundle_dir,
        dtype="bfloat16",
        device_map=str(device),
        attn_implementation="sdpa",
    )
    expected_model_type = "alpamayo_reasoning_vla_expert"
    if model.config.model_type != expected_model_type:
        raise ValueError(
            f"Expected model_type {expected_model_type!r} at {bundle_dir!r}; "
            f"got {model.config.model_type!r}."
        )
    if not getattr(model.config, "legacy_inference_image_input_format", False):
        logger.warning(
            "Forcing model.config.legacy_inference_image_input_format=True "
            "(was %r); the AlpamayoR1InferenceModel adapter feeds frames in "
            "[-1, 1] and relies on the model's internal rescale.",
            getattr(model.config, "legacy_inference_image_input_format", "<unset>"),
        )
        model.config.legacy_inference_image_input_format = True
    return AlpamayoR1InferenceModel(
        model,
        num_context_frames=model_config.num_context_frames,
    )


def build_model_inputs(
    run_config: Any,
) -> Callable[[Any], tuple[dict[str, Any], Any]]:
    """Return the R1 trainer-side replay input builder."""
    from alpagym_alpamayo_r1.inference_model import AlpamayoR1InferenceModel

    return functools.partial(
        AlpamayoR1InferenceModel.build_trainer_model_inputs,
        num_context_frames=run_config.policy.model.num_context_frames,
    )


def get_bundle() -> PolicyBundle:
    """Return the Alpamayo R1 runtime hooks."""
    return PolicyBundle(
        setup_tokenizer=setup_tokenizer,
        build_data_packer=build_data_packer,
        install_runtime_bridge=install_alpamayo_r1_runtime_bridge,
        load_inference_model=load_inference_model,
        build_model_inputs=build_model_inputs,
    )
