"""Cosmos-RL integration for AutoVLA (Qwen2.5-VL).

Provides a WeightMapper and BaseModel wrapper so that Cosmos-RL can
synchronise *all* model parameters between Policy and Rollout workers —
including the vision encoder (visual.*), which the default
HFModelWeightMapper silently skips.
"""

from __future__ import annotations

import logging
import torch
from cosmos_rl.policy.model.base import BaseModel
from cosmos_rl.policy.model.hf_models.weight_mapper import HFModelWeightMapper
from cosmos_rl.utils import util
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

logger = logging.getLogger(__name__)


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
        if name.startswith("visual."):
            return name
        return super().policy_map_local_key_to_hf_key(name)

    # -- Rollout side -------------------------------------------------------

    def rollout_map_local_key_to_hf_key(self, rollout_weight_name: str) -> str:
        """Map rollout-side parameter names to HF checkpoint key-space."""
        name = rollout_weight_name
        if name.startswith("model.vlm."):
            name = name.replace("model.vlm.", "", 1)
        elif name.startswith("vlm."):
            name = name[len("vlm."):]
        if name.startswith("llm.model."):
            name = name.replace("llm.model.", "model.", 1)
        elif name.startswith("llm.lm_head."):
            name = name.replace("llm.lm_head.", "lm_head.", 1)
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
# BaseModel wrapper (fallback if built-in registration fails)
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

    @property
    def parallelize_fn(self):
        """No-op parallelisation for single-GPU smoke tests."""
        def _noop(model, parallel_dims, config, pp_loss_fn=None):
            return None, None
        return _noop, self

    def post_to_empty_hook(self, cosmos_config):
        pass

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
        if hasattr(ckpt, "lm_head") and hasattr(ckpt.model, "embed_tokens"):
            ckpt.lm_head.weight = torch.nn.Parameter(
                ckpt.model.embed_tokens.weight.data.clone()
            )
        state = ckpt.state_dict()
        del ckpt
        self.model.load_state_dict(state, strict=True)
        if device is not None:
            self.model = self.model.to(device)

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


# ---------------------------------------------------------------------------
# Built-in model patching for AutoVLA action tokens
# ---------------------------------------------------------------------------

def patch_cosmos_qwen_model_for_autovla(sft_checkpoint_path: str | None) -> None:
    """Patch the built-in Cosmos-RL Qwen2.5-VL model to support AutoVLA action tokens.

    The built-in ``Qwen2_5_VLConditionalModel`` does not resize its vocab for
    AutoVLA's 2048 extra action tokens (IDs 151665-153712).  This patch wraps
    ``load_hf_weights`` to resize embeddings and load the SFT checkpoint after
    the base HF weights are loaded.
    """
    try:
        from cosmos_rl.policy.model.qwen2_5_vl import Qwen2_5_VLConditionalModel
    except ImportError:
        return

    if getattr(Qwen2_5_VLConditionalModel, "_autovla_patched", False):
        if sft_checkpoint_path is not None:
            Qwen2_5_VLConditionalModel._autovla_sft_checkpoint_path = sft_checkpoint_path
        return

    Qwen2_5_VLConditionalModel._autovla_sft_checkpoint_path = sft_checkpoint_path
    original_load_hf_weights = Qwen2_5_VLConditionalModel.load_hf_weights

    def patched_load_hf_weights(self, model_name_or_path, parallel_dims, device, revision=None):
        original_load_hf_weights(self, model_name_or_path, parallel_dims, device, revision=revision)

        action_start_id = 151665
        n_bins = 2048
        needed_vocab = action_start_id + n_bins
        current_vocab = self.model.config.vocab_size
        if current_vocab < needed_vocab:
            old_embed = self.model.embed_tokens.weight.data
            self.model.embed_tokens = torch.nn.Embedding(
                needed_vocab, self.model.config.dim,
                device=old_embed.device, dtype=old_embed.dtype,
            )
            self.model.embed_tokens.weight.data[:current_vocab] = old_embed
            self.model.config.vocab_size = needed_vocab
            self.vocab_size = needed_vocab
            if hasattr(self, "hf_config"):
                self.hf_config.vocab_size = needed_vocab
            if hasattr(self, "weight_mapper") and hasattr(self.weight_mapper, "config"):
                self.weight_mapper.config.vocab_size = needed_vocab
            if hasattr(self.model, "lm_head") and not self.model.tie_embed_tokens:
                old_lm_head = self.model.lm_head.weight.data
                self.model.lm_head = torch.nn.Linear(
                    self.model.config.dim, needed_vocab, bias=False,
                    device=old_lm_head.device, dtype=old_lm_head.dtype,
                )
                self.model.lm_head.weight.data[:current_vocab] = old_lm_head
            logger.info(
                "Resized Qwen2.5-VL policy model vocab: %d -> %d for AutoVLA action tokens",
                current_vocab, needed_vocab,
            )

        ckpt_path = self.__class__._autovla_sft_checkpoint_path
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            raw = ckpt["state_dict"]
            new_state: dict[str, torch.Tensor] = {}
            for k, v in raw.items():
                if k.startswith("autovla."):
                    k = k[len("autovla."):]
                if k.startswith("vlm."):
                    k = k[len("vlm."):]
                if k == "lm_head.weight":
                    k = "model.lm_head.weight"
                new_state[k] = v

            full_state = self.state_dict()

            loadable: dict[str, torch.Tensor] = {}
            skipped = 0
            for k, v in new_state.items():
                if k in full_state and full_state[k].shape == v.shape:
                    loadable[k] = v
                else:
                    skipped += 1

            if loadable:
                self.load_state_dict(loadable, strict=False)
            logger.info(
                "Policy model loaded AutoVLA SFT checkpoint from %s "
                "(loaded=%d keys, skipped=%d)",
                ckpt_path, len(loadable), skipped,
            )
            self.__class__._autovla_sft_checkpoint_path = None

    patched_load_hf_weights._autovla_patched = True  # type: ignore[attr-defined]
    Qwen2_5_VLConditionalModel.load_hf_weights = patched_load_hf_weights