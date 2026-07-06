"""Cosmos-RL integration for AutoVLA (Qwen2.5-VL).

Provides a WeightMapper and BaseModel wrapper so that Cosmos-RL can
synchronise *all* model parameters between Policy and Rollout workers —
including the vision encoder (visual.*), which the default
HFModelWeightMapper silently skips.
"""

from __future__ import annotations

import torch
from cosmos_rl.policy.model.base import BaseModel
from cosmos_rl.policy.model.hf_models.weight_mapper import HFModelWeightMapper
from cosmos_rl.utils import util
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Weight mapper
# ---------------------------------------------------------------------------

class Qwen2_5_VLWeightMapper(HFModelWeightMapper):
    """Weight-name mapper for Qwen2.5-VL.

    The default ``HFModelWeightMapper`` is initialised with the LLM sub-config
    only and therefore cannot route ``visual.*`` parameters, producing the
    flood of "No send instructions generated for parameter visual.blocks.*"
    warnings that block the Cosmos-RL parameter sync.

    This subclass passes ``visual.*`` keys through as-is (they are already in
    HF checkpoint format) before delegating to the base class for ``model.*``
    and ``lm_head.*`` keys.
    """

    def __init__(self, hf_config: AutoConfig):
        try:
            llm_config = hf_config.get_llm_config()
        except (AttributeError, TypeError):
            llm_config = hf_config
        super().__init__(llm_config)

    # -- Policy side --------------------------------------------------------

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        """Map policy-side parameter names to HF checkpoint key-space."""
        name = util.clear_weight_name(name)
        # visual.* (vision encoder, merger, patch_embed) are already canonical
        if name.startswith("visual."):
            return name
        # model.*, lm_head.*  — delegate to base class
        return super().policy_map_local_key_to_hf_key(name)

    # -- Rollout side -------------------------------------------------------

    def rollout_map_local_key_to_hf_key(self, rollout_weight_name: str) -> str:
        """Map rollout-side parameter names to HF checkpoint key-space."""
        name = rollout_weight_name
        # Strip common Cosmos/vLLM prefixes
        if name.startswith("model.vlm."):
            name = name.replace("model.vlm.", "", 1)
        elif name.startswith("vlm."):
            name = name[len("vlm."):]
        if name.startswith("llm.model."):
            name = name.replace("llm.model.", "model.", 1)
        elif name.startswith("llm.lm_head."):
            name = name.replace("llm.lm_head.", "lm_head.", 1)
        # visual.* pass-through
        if name.startswith("visual."):
            return name
        return self.policy_map_local_key_to_hf_key(name)

    def rollout_split_local_key_n_param_to_hf_key_n_param(
        self, param_name: str, param: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Rollout-side mapping/splitting with vocab-padding trimming."""
        group = super().rollout_split_local_key_n_param_to_hf_key_n_param(
            param_name, param
        )
        vocab_size = getattr(self.config, "vocab_size", None)
        if vocab_size is None:
            return group
        trimmed: list[tuple[str, torch.Tensor]] = []
        for nm, t in group:
            if (
                nm in ("model.embed_tokens.weight", "lm_head.weight")
                and isinstance(t, torch.Tensor)
                and t.ndim == 2
                and t.shape[0] > vocab_size
                and t.shape[1] > 0
            ):
                trimmed.append((nm, t[:vocab_size]))
            else:
                trimmed.append((nm, t))
        return trimmed


# ---------------------------------------------------------------------------
# BaseModel wrapper
# ---------------------------------------------------------------------------

class Qwen2_5_VLBaseModel(BaseModel):
    """Minimal Cosmos-RL BaseModel wrapper around Qwen2_5_VLForConditionalGeneration.

    On a single GPU (smoke test, dp_shard_size=1) FSDP2 is unnecessary, so
    ``parallelize_fn`` is a no-op.  For multi-GPU training this should be
    replaced with proper FSDP2 sharding (shard visual tower + LM layers).
    """

    def __init__(self, hf_config: AutoConfig):
        super().__init__(hf_config)
        self.hf_config = hf_config
        self.model = Qwen2_5_VLForConditionalGeneration(hf_config)

    @staticmethod
    def supported_model_types():
        return ["qwen2_5_vl"]

    # -- Parallelism --------------------------------------------------------

    @property
    def parallelize_fn(self):
        """No-op parallelisation for single-GPU smoke tests."""
        def _noop(model, parallel_dims, config, pp_loss_fn=None):
            return None, None
        return _noop, self

    def post_to_empty_hook(self, cosmos_config):
        """No-op; load_hf_weights handles all weight restoration."""
        pass

    # -- Weight loading -----------------------------------------------------

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims=None,
        device: torch.device | None = None,
        revision: str | None = None,
    ) -> None:
        """Load checkpoint weights into self.model."""
        ckpt = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, trust_remote_code=True
        ).to("cpu")
        state = ckpt.state_dict()
        del ckpt
        self.model.load_state_dict(state, strict=True)
        if device is not None:
            self.model = self.model.to(device)

    # -- Forward ------------------------------------------------------------

    def get_position_ids(self, **kwargs):
        if "input_ids" in kwargs and kwargs["input_ids"] is not None:
            inputs = kwargs["input_ids"]
        else:
            inputs_embeds = kwargs.get("inputs_embeds")
            if inputs_embeds is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
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
        return position_ids, inputs, 1

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        """Bridge Cosmos-RL inputs to Qwen2.5-VL forward pass."""
        out = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            **kwargs,
        )
        return out

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: int | None = None,
    ) -> "Qwen2_5_VLBaseModel":
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if max_position_embeddings is not None:
            config.max_position_embeddings = max_position_embeddings
        config._name_or_path = model_name_or_path
        return cls(config)
