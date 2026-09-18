# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos-RL BaseModel wrapper for Qwen-Drive-1.0 Planning Expert.

Registers the ``qwen_drive`` model type with Cosmos-RL's ``ModelRegistry``
so the GRPO trainer can load and train the model.  The wrapper:

1. Registers ``QwenDriveConfig`` with ``AutoConfig`` and
   ``QwenDriveForPlanning`` with ``AutoModel``.
2. Provides ``from_pretrained`` / ``load_hf_weights`` so Cosmos-RL can
   instantiate the model on meta device and then load weights.
3. Provides ``forward`` for the GRPO training step (Phase 4: real
   flow-matching logprob; Phase 3 smoke test: stub).

The planner (Planning Expert) checkpoint lives in a separate directory
whose path is injected via :func:`set_planner_path` from the policy
bundle's ``install_runtime_bridge`` hook.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

import torch
from safetensors.torch import load_file
from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.policy.model.hf_models.weight_mapper import HFModelWeightMapper
from cosmos_rl.utils.logging import logger as cosmos_logger
from transformers import AutoConfig, AutoModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom weight mapper for Qwen-Drive's composite config
# ---------------------------------------------------------------------------


class QwenDriveWeightMapper(HFModelWeightMapper):
    """Weight mapper that extracts attention config from ``vlm_config.text_config``.

    ``QwenDriveConfig`` is a composite config with ``vlm_config`` (Qwen3.5 VLM)
    and ``expert_config`` (Planning Expert).  The VLM's attention parameters
    live in ``vlm_config.text_config``, which the default
    ``HFModelWeightMapper`` cannot find (it only checks top-level
    ``text_config`` / ``llm_config``).

    Additionally, Qwen3.5 has an explicit ``head_dim`` field (256) that
    differs from ``hidden_size // num_attention_heads`` (160), and uses
    ``attn_output_gate=True`` which makes vLLM store Q+gate separately
    from K+V.  This mapper corrects the head_dim and handles the
    Q-only QKV weight layout.
    """

    def __init__(self, hf_config: AutoConfig) -> None:
        # Temporarily expose text_config at the top level so the parent
        # __init__ can find the attention parameters.
        vlm_config = getattr(hf_config, "vlm_config", None)
        if vlm_config is not None and not hasattr(hf_config, "text_config"):
            text_config = getattr(vlm_config, "text_config", None)
            if text_config is not None:
                hf_config.text_config = text_config
        super().__init__(hf_config)

        # Qwen3.5 has an explicit head_dim that differs from
        # hidden_size // num_attention_heads.  Use it if available.
        if self.text_config is not None:
            explicit_head_dim = getattr(self.text_config, "head_dim", None)
            if explicit_head_dim is not None and explicit_head_dim > 0:
                self.head_dim = explicit_head_dim

    def rollout_split_local_key_n_param_to_hf_key_n_param(
        self, param_name: str, param: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Handle Qwen3.5's Q-only QKV weight layout.

        When ``attn_output_gate=True``, vLLM may store the Q+gate weight
        separately from K+V.  In that case the ``qkv_proj`` weight has
        ``dim_0 = total_q * head_dim`` (Q+gate only) instead of the
        expected ``(total_q + 2*n_kv) * head_dim`` (Q+K+V).
        We detect this and return only the ``q_proj`` mapping, letting
        K and V be handled by their own weights.
        """
        compatible_key = self.rollout_map_local_key_to_hf_key(param_name)

        is_qkv = ("qkv_proj" in compatible_key) or (
            "qkv" in compatible_key and not self.is_vlm
        )
        if is_qkv:
            tc = self.text_config if self.text_config is not None else self.config
            dim_0 = param.shape[0]
            total_q = tc.num_attention_heads * (1 + int(self.attn_output_gate))
            expected_full = (total_q + 2 * tc.num_key_value_heads) * self.head_dim
            q_only = total_q * self.head_dim

            if dim_0 == q_only and dim_0 != expected_full:
                # Q-only weight (Q+gate, no K+V fused)
                rule = "qkv_proj" if "qkv_proj" in compatible_key else "qkv"
                q_proj_key = compatible_key.replace(rule, "q_proj")
                cosmos_logger.debug(
                    "[QwenDriveWeightMapper] Q-only QKV detected: "
                    f"{compatible_key} dim_0={dim_0} (Q+gate only)"
                )
                return [(q_proj_key, param)]

        return super().rollout_split_local_key_n_param_to_hf_key_n_param(
            param_name, param
        )

# ---------------------------------------------------------------------------
# Module-level planner path (set by bundle.install_runtime_bridge)
# ---------------------------------------------------------------------------

_PLANNER_PATH: str | None = None


def set_planner_path(path: str | None) -> None:
    """Store the Planning Expert checkpoint path for ``load_hf_weights``."""
    global _PLANNER_PATH
    _PLANNER_PATH = path
    if path:
        logger.info("QwenDriveCosmos: planner path set to %s", path)


# ---------------------------------------------------------------------------
# AutoConfig / AutoModel registration
# ---------------------------------------------------------------------------

def _register_autoconfig_and_model() -> None:
    """Register QwenDriveConfig and QwenDriveForPlanning with transformers."""
    try:
        from qwen_drive.configuration_qwen_drive import (
            QwenDriveConfig,
            QwenDrivePlanningExpertConfig,
        )
        from qwen_drive.modeling_qwen_drive import QwenDriveForPlanning
    except ImportError:
        logger.debug("qwen_drive not importable yet, skipping AutoConfig registration")
        return

    for cfg_cls in (QwenDriveConfig, QwenDrivePlanningExpertConfig):
        mt = cfg_cls.model_type
        try:
            AutoConfig.register(mt, cfg_cls, exist_ok=True)
            logger.info("Registered AutoConfig for model_type=%s", mt)
        except (ValueError, TypeError):
            try:
                AutoConfig.register(mt, cfg_cls)
            except ValueError:
                pass

    try:
        AutoModel.register(QwenDriveConfig, QwenDriveForPlanning, exist_ok=True)
        logger.info("Registered AutoModel for QwenDriveConfig")
    except (ValueError, TypeError):
        try:
            AutoModel.register(QwenDriveConfig, QwenDriveForPlanning)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Cosmos-RL BaseModel wrapper
# ---------------------------------------------------------------------------


class QwenDriveCosmos(BaseModel):
    """Cosmos-RL BaseModel wrapper for Qwen-Drive-1.0.

    The wrapper holds a :class:`QwenDriveForPlanning` model.  Cosmos-RL
    instantiates it on meta device (via ``from_pretrained``), materialises
    the tensors, then calls ``load_hf_weights`` to load the actual
    safetensors checkpoints (VLM + Planning Expert).
    """

    def __init__(self, hf_config: AutoConfig) -> None:
        super().__init__(hf_config)
        self.hf_config = hf_config

        # Build the model from config (on meta device when called inside
        # ``init_on_device("meta")`` context).
        from qwen_drive.modeling_qwen_drive import QwenDriveForPlanning

        self.model = QwenDriveForPlanning(hf_config)

        # Freeze the VLM 鈥?only the Planning Expert is trained.
        for param in self.model.vlm.parameters():
            param.requires_grad = False
        cosmos_logger.info(
            "[QwenDriveCosmos] Froze VLM params, trainable Planning Expert params: %d",
            sum(1 for p in self.model.planning_expert.parameters() if p.requires_grad),
        )

    @staticmethod
    def supported_model_types() -> list[str]:
        """Return the HF model types this wrapper handles."""
        return ["qwen_drive"]

    # -- construction -------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str | None = None,
        max_position_embeddings: int | None = None,
        **kwargs: Any,
    ) -> "QwenDriveCosmos":
        """Construct a QwenDriveCosmos from its HF config (meta device)."""
        del model_name_or_path, max_position_embeddings  # unused
        return cls(hf_config)

    # -- weight loading -----------------------------------------------------

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: Any = None,
        device: Any = None,
        revision: str | None = None,
    ) -> None:
        """Load VLM and Planning Expert weights from safetensors checkpoints.

        ``model_name_or_path`` points at the VLM directory.  The Planning
        Expert checkpoint path is read from the module-level
        ``_PLANNER_PATH`` (set by ``bundle.install_runtime_bridge``).
        """
        from safetensors.torch import load_file

        model_path = model_name_or_path

        # --- Load VLM weights ---
        vlm_state = self._load_safetensors_dir(model_path, prefix_strip=None)
        if vlm_state:
            # QwenDriveForPlanning stores the VLM under self.model.vlm
            # and the top-level model under "vlm.*" in the checkpoint.
            # Map "vlm." prefix -> "" for direct loading into self.model.vlm
            vlm_prefixed = {}
            for k, v in vlm_state.items():
                if k.startswith("vlm."):
                    vlm_prefixed[k[len("vlm.") :]] = v
                elif k.startswith("model.vlm."):
                    vlm_prefixed[k[len("model.vlm.") :]] = v
                else:
                    # Keep non-VLM keys for the expert (shouldn't happen here)
                    pass
            if vlm_prefixed:
                missing, unexpected = self.model.vlm.load_state_dict(
                    vlm_prefixed, strict=False
                )
                if missing:
                    cosmos_logger.warning(
                        "[QwenDriveCosmos] VLM missing keys: %d", len(missing)
                    )
                if unexpected:
                    cosmos_logger.warning(
                        "[QwenDriveCosmos] VLM unexpected keys: %d", len(unexpected)
                    )
            cosmos_logger.info(
                "[QwenDriveCosmos] Loaded VLM weights from %s (%d tensors)",
                model_path,
                len(vlm_state),
            )

        # --- Load Planning Expert weights ---
        planner_path = _PLANNER_PATH
        if planner_path is None:
            # Fallback: look for planner-sft subdirectory
            planner_path = os.path.join(model_path, "planner-sft")
        if os.path.isdir(planner_path):
            expert_state = self._load_safetensors_dir(planner_path, prefix_strip="planning_expert.")
            if expert_state:
                target_dtype = next(self.model.planning_expert.parameters()).dtype
                expert_state = {k: v.to(target_dtype) for k, v in expert_state.items()}
                missing, unexpected = self.model.planning_expert.load_state_dict(
                    expert_state, strict=False
                )
                if missing:
                    cosmos_logger.warning(
                        "[QwenDriveCosmos] Planning Expert missing keys: %d",
                        len(missing),
                    )
                if unexpected:
                    cosmos_logger.warning(
                        "[QwenDriveCosmos] Planning Expert unexpected keys: %d",
                        len(unexpected),
                    )
                cosmos_logger.info(
                    "[QwenDriveCosmos] Loaded Planning Expert weights from %s (%d tensors)",
                    planner_path,
                    len(expert_state),
                )
            else:
                cosmos_logger.warning(
                    "[QwenDriveCosmos] No safetensors found at %s", planner_path
                )

        # --- Move to device if specified ---
        if device is not None:
            self.model = self.model.to(device)
            cosmos_logger.info("[QwenDriveCosmos] Moved model to %s", device)

    @staticmethod
    def _load_safetensors_dir(
        dir_path: str, prefix_strip: str | None = None
    ) -> dict[str, torch.Tensor]:
        """Load all safetensors shards from a directory."""
        state: dict[str, torch.Tensor] = {}

        # Check for index file (sharded checkpoint)
        index_path = os.path.join(dir_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            for shard in shard_files:
                shard_path = os.path.join(dir_path, shard)
                if os.path.exists(shard_path):
                    state.update(load_file(shard_path))
        else:
            # Single file
            single_path = os.path.join(dir_path, "model.safetensors")
            if os.path.exists(single_path):
                state.update(load_file(single_path))

        if prefix_strip and state:
            state = {
                k[len(prefix_strip) :] if k.startswith(prefix_strip) else k: v
                for k, v in state.items()
            }
        return state

    # -- position ids (required by BaseModel ABC) --------------------------

    def get_position_ids(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return position ids, input ids, and sequence dimension index.

        Required by Cosmos-RL's ``BaseModel`` ABC for context-parallelism
        support.  Qwen-Drive does not use context parallelism, so we
        return a simple sequential position id tensor.
        """
        if "input_ids" in kwargs and kwargs["input_ids"] is not None:
            inputs = kwargs["input_ids"]
        else:
            inputs_embeds = kwargs.get("inputs_embeds", None)
            if inputs_embeds is None:
                # No text inputs 鈥?return a dummy 1-token position
                device = next(self.parameters()).device
                inputs = torch.zeros(1, 1, dtype=torch.long, device=device)
            else:
                seq_len = inputs_embeds.size(1)
                batch = inputs_embeds.size(0)
                inputs = torch.zeros(
                    batch, seq_len, dtype=torch.long, device=inputs_embeds.device
                )

        position_ids = (
            torch.arange(inputs.size(-1), dtype=torch.long, device=inputs.device)
            .unsqueeze(0)
            .expand_as(inputs)
        )
        return position_ids, inputs, 1  # seq_dim_idx = 1

    # -- forward (GRPO training) --------------------------------------------

    def forward(self, teacher_model: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Forward pass for GRPO training.

        Phase 3 (smoke test): returns dummy log_probs so the training
        step completes without errors.

        Phase 4: will compute the flow-matching log-probability of the
        recorded trajectory (samples_list) under the current policy.
        """
        is_padding = kwargs.get("is_padding")
        if is_padding is not None:
            batch_size = is_padding.shape[0]
            device = is_padding.device
        else:
            # Try to infer batch size from other inputs
            for key in ("ego_history_xyz", "camera_frames", "samples_list"):
                val = kwargs.get(key)
                if val is not None and hasattr(val, "shape"):
                    batch_size = val.shape[0]
                    device = val.device
                    break
            else:
                batch_size = 1
                device = next(self.parameters()).device

        log_probs = torch.zeros(batch_size, device=device, dtype=torch.float32)
        # Connect log_probs to model parameters so loss.backward() has a grad_fn.
        # The 0.0 multiplier ensures values don't change; with lr=0.0 the optimizer
        # step is a no-op. Phase 4 will replace this with real flow-matching logprobs.
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if trainable_params:
            log_probs = log_probs + 0.0 * trainable_params[0].sum()
        kl_div = None
        if teacher_model is not None:
            kl_div = torch.zeros(batch_size, device=device, dtype=torch.float32)
            if trainable_params:
                kl_div = kl_div + 0.0 * trainable_params[0].sum()

        return {"log_probs": log_probs, "kl_div": kl_div}

    # -- parallelism / FSDP -------------------------------------------------

    @property
    def parallelize_fn(self) -> tuple[Callable, "QwenDriveCosmos"]:
        """Return ``(parallelize_fn, self)`` for single-GPU / DDP."""

        def parallelize(
            model: torch.nn.Module,
            parallel_dims: Any,
            config: Any,
            pp_loss_fn: Callable | None = None,
        ):
            # For single-GPU or DDP replicate
            if (
                hasattr(parallel_dims, "dp_replicate_enabled")
                and parallel_dims.dp_replicate_enabled
            ):
                from torch.distributed._composable.replicate import replicate

                replicate(model, device_mesh=parallel_dims.mesh, bucket_cap_mb=100)
                cosmos_logger.info("[QwenDriveCosmos] Applied DDP (replicate)")
            return None, None

        return parallelize, self

    def _apply_fsdp2(self, dp_mesh: Any, fsdp_config: dict, reshard_fn: Any) -> None:
        """Apply FSDP2 sharding (no-op for single GPU)."""
        from torch.distributed.fsdp import fully_shard

        # Shard the planning expert layers
        expert = self.model.planning_expert
        if hasattr(expert, "layers"):
            items = list(enumerate(expert.layers))
            for idx, blk in items:
                fully_shard(
                    blk,
                    **fsdp_config,
                    reshard_after_forward=reshard_fn(idx, len(items)),
                )
            fully_shard(expert, **fsdp_config, reshard_after_forward=True)
            cosmos_logger.info(
                "[QwenDriveCosmos] Sharded %d planning expert layers", len(items)
            )

        fully_shard(self, **fsdp_config, reshard_after_forward=True)
        cosmos_logger.info("[QwenDriveCosmos] Applied FSDP2 to full model")

    def post_to_empty_hook(self, cosmos_config: Any) -> None:
        """Warm up after meta->device materialisation.

        No-op for Qwen-Drive; the VLM and expert are already materialised
        by ``load_hf_weights``.
        """
        pass

    def separate_model_parts(self) -> list[torch.nn.Module]:
        """Return model sub-modules for per-part parallelization."""
        return [self]

    @classmethod
    def get_nparams_and_flops(cls, seq_len: int) -> tuple[int, int]:
        """Return estimated parameter count and FLOPs."""
        return 0, 0

    def apply_pipeline_split(self, pp_rank: int, pp_size: int) -> None:
        """Pipeline parallel is not supported."""
        raise NotImplementedError("Pipeline parallel is not supported for Qwen-Drive.")

    # -- parameter delegation -----------------------------------------------

    def named_parameters(self, *args: Any, **kwargs: Any):
        return self.model.named_parameters(*args, **kwargs)

    def named_modules(self, *args: Any, **kwargs: Any):
        return self.model.named_modules(*args, **kwargs)

    def parameters(self, *args: Any, **kwargs: Any):
        return self.model.parameters(*args, **kwargs)

    def modules(self, *args: Any, **kwargs: Any):
        return self.model.modules(*args, **kwargs)


# ---------------------------------------------------------------------------
# ModelRegistry registration
# ---------------------------------------------------------------------------

_register_autoconfig_and_model()

if "qwen_drive" not in ModelRegistry._MODEL_REGISTRY:
    ModelRegistry.register(QwenDriveWeightMapper)(QwenDriveCosmos)
    cosmos_logger.info("[QwenDriveCosmos] Registered with ModelRegistry")
