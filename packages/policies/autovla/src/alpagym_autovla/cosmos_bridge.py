"""Cosmos-RL integration for AutoVLA (Qwen2.5-VL).

Provides a WeightMapper and BaseModel wrapper so that Cosmos-RL can
synchronise *all* model parameters between Policy and Rollout workers —
including the vision encoder (visual.*), which the default
HFModelWeightMapper silently skips.
"""

from __future__ import annotations

import logging
import torch
from collections.abc import Callable
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

_AUTOVLA_ACTION_START_ID = 151665
_AUTOVLA_N_BINS = 2048
_AUTOVLA_NEEDED_VOCAB = _AUTOVLA_ACTION_START_ID + _AUTOVLA_N_BINS

_VOCAB_PAD_KEYS = {
    "model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "lm_head.weight",
    "model.lm_head.weight",
}


def _load_sft_weights_into_hfmodel(hf_model: "HFModel", ckpt_path: str) -> None:
    """Load AutoVLA SFT checkpoint weights into an HFModel (FSDP-safe).

    Uses the same ``local_view.data.copy_()`` pattern as
    ``HFModel.load_hf_weights_from_safetensors`` so that FSDP-sharded
    parameters are handled correctly.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["state_dict"]
    new_state: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("autovla."):
            k = k[len("autovla."):]
        if k.startswith("vlm."):
            k = k[len("vlm."):]
        new_state[k] = v
    del ckpt, raw

    model_state = hf_model.model.state_dict()
    model_state = {util.clear_weight_name(k): v for k, v in model_state.items()}

    loaded = 0
    skipped = 0
    for k, v in new_state.items():
        if k not in model_state:
            skipped += 1
            continue
        target_tensor = model_state[k]
        is_dist_tensor = isinstance(target_tensor, torch.distributed.tensor.DTensor)
        local_view = target_tensor.to_local() if is_dist_tensor else target_tensor
        if local_view.shape != v.shape:
            skipped += 1
            continue
        with torch.no_grad():
            local_view.data.copy_(v.to(local_view.device))
        loaded += 1

    logger.info(
        "Policy model loaded AutoVLA SFT checkpoint from %s "
        "(loaded=%d keys, skipped=%d)",
        ckpt_path, loaded, skipped,
    )


def _autovla_load_hf_weights_from_safetensors(
    self, model_name_or_path, parallel_dims, device, revision=None
):
    """Load base Qwen2.5-VL safetensors with vocab-size padding for AutoVLA.

    When the model was built with ``vocab_size=153713`` (AutoVLA action
    tokens) but the base checkpoint only has ``vocab_size=151936``, the
    standard ``load_hf_weights_from_safetensors`` raises an assertion
    error because ``embed_tokens.weight`` / ``lm_head.weight`` shapes
    don't match.

    This replacement loads the base safetensors, zero-pads embedding
    tensors that are smaller than the model, then copies into the model
    using the same FSDP-safe ``local_view.data.copy_()`` pattern.
    """
    from cosmos_rl.policy.model.hf_models.weight_converter import convert_weight_from_hf
    from cosmos_rl.utils.multi_rank_weight_loader import MultiRankWeightLoader
    import os

    loader = MultiRankWeightLoader(parallel_dims)
    model_type = self.hf_config.model_type
    model_path = util.resolve_model_path(model_name_or_path, revision=revision)
    safetensors_files = sorted(
        [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    )

    self_state_dict = self.model.state_dict()
    self_state_dict = {util.clear_weight_name(k): v for k, v in self_state_dict.items()}

    lm_head_weight_key = None
    embed_tokens_weight_key = None
    for k in self_state_dict.keys():
        if "embed_tokens" in k or "embeddings" in k:
            embed_tokens_weight_key = k
            if lm_head_weight_key is not None:
                break
        if "lm_head" in k:
            lm_head_weight_key = k
            if embed_tokens_weight_key is not None:
                break
    assert lm_head_weight_key is not None and embed_tokens_weight_key is not None, (
        "lm_head and embed_tokens weight keys not found in the state dict"
    )

    hf_checkpoint_conversion_mapping = getattr(
        self.model, "_checkpoint_conversion_mapping", None
    )
    from cosmos_rl.utils.transformers_utils import is_transformers_v5
    use_v5_suffix_lookup = (
        is_transformers_v5() and not hf_checkpoint_conversion_mapping
    )
    model_key_by_tail: dict[str, str] = {}
    if use_v5_suffix_lookup:
        _prefixes = (
            "model.language_model.model.",
            "model.language_model.",
            "language_model.model.",
            "language_model.",
            "model.",
            "",
        )
        for model_key in self_state_dict:
            for pfx in _prefixes:
                if model_key.startswith(pfx):
                    tail = model_key[len(pfx):]
                    if tail and tail not in model_key_by_tail:
                        model_key_by_tail[tail] = model_key
                        break

    import re as _re

    def name_converter(name: str) -> str:
        if hf_checkpoint_conversion_mapping:
            for pattern, replacement in hf_checkpoint_conversion_mapping.items():
                if _re.search(pattern, name):
                    return _re.sub(pattern, replacement, name)
        elif use_v5_suffix_lookup:
            if name in self_state_dict:
                return name
            for pfx in _prefixes:
                if name.startswith(pfx):
                    tail = name[len(pfx):]
                    if tail and tail in model_key_by_tail:
                        return model_key_by_tail[tail]
            return name
        return name

    rank_tensors, rank_tensor_metadata, weights_of_ckpt_names = (
        loader.load_files_parallel(
            model_path, device, safetensors_files,
            name_converter=name_converter,
        )
    )
    all_tensor_names, tensor_to_rank_map = (
        loader.gather_tensor_names_and_build_mapping(
            weights_of_ckpt_names, rank_tensors
        )
    )

    reserved = {}
    for name, tensor in loader.iterate_tensors(
        all_tensor_names, tensor_to_rank_map, rank_tensors,
        rank_tensor_metadata, device,
    ):
        if self._should_skip_tensor(name):
            continue
        if name == embed_tokens_weight_key:
            reserved[name] = tensor.clone()

        tp_slice_dim = None
        if self.tp_slice_dim_map is not None:
            tp_slice_dim = self.tp_slice_dim_map.get(name, None)
        dest_name, sharded_weight = convert_weight_from_hf(
            tensor, name, model_type, parallel_dims,
            tp_slice_dim=tp_slice_dim, hf_config=self.model.config,
        )
        if dest_name is None and sharded_weight is None:
            continue
        elif isinstance(dest_name, Callable):
            target_tensor = dest_name(self_state_dict)
        else:
            target_tensor = self_state_dict[dest_name]

        is_dist_tensor = isinstance(target_tensor, torch.distributed.tensor.DTensor)
        local_view = target_tensor.to_local() if is_dist_tensor else target_tensor

        if local_view.shape != sharded_weight.shape:
            if (
                dest_name in _VOCAB_PAD_KEYS
                and sharded_weight.ndim == 2
                and local_view.ndim == 2
                and sharded_weight.shape[1] == local_view.shape[1]
                and sharded_weight.shape[0] < local_view.shape[0]
            ):
                ckpt_rows = sharded_weight.shape[0]
                padded = torch.zeros(
                    local_view.shape, dtype=sharded_weight.dtype,
                    device=sharded_weight.device,
                )
                padded[:ckpt_rows] = sharded_weight
                sharded_weight = padded
                logger.info(
                    "AutoVLA: zero-padded %s %d -> %d rows for action tokens",
                    dest_name, ckpt_rows, local_view.shape[0],
                )
            else:
                raise AssertionError(
                    f"Shape mismatch: {local_view.shape} != {sharded_weight.shape} for {dest_name}"
                )

        with torch.no_grad():
            local_view.data.copy_(sharded_weight)

    if (
        lm_head_weight_key not in all_tensor_names
        and embed_tokens_weight_key in all_tensor_names
    ):
        name = lm_head_weight_key
        assert embed_tokens_weight_key in reserved
        tensor = reserved[embed_tokens_weight_key]

        tp_slice_dim = None
        if self.tp_slice_dim_map is not None:
            tp_slice_dim = self.tp_slice_dim_map.get(name, None)
        dest_name, sharded_weight = convert_weight_from_hf(
            tensor, name, model_type, parallel_dims,
            tp_slice_dim=tp_slice_dim, hf_config=self.model.config,
        )
        if dest_name in self_state_dict:
            target_tensor = self_state_dict[dest_name]
            is_dist_tensor = isinstance(target_tensor, torch.distributed.tensor.DTensor)
            local_view = target_tensor.to_local() if is_dist_tensor else target_tensor

            if local_view.shape != sharded_weight.shape:
                if (
                    dest_name in _VOCAB_PAD_KEYS
                    and sharded_weight.ndim == 2
                    and local_view.ndim == 2
                    and sharded_weight.shape[1] == local_view.shape[1]
                    and sharded_weight.shape[0] < local_view.shape[0]
                ):
                    padded = torch.zeros(
                        local_view.shape, dtype=sharded_weight.dtype,
                        device=sharded_weight.device,
                    )
                    padded[:sharded_weight.shape[0]] = sharded_weight
                    sharded_weight = padded

            assert local_view.shape == sharded_weight.shape, (
                f"Shape mismatch: {local_view.shape} != {sharded_weight.shape} for {dest_name}"
            )
            with torch.no_grad():
                local_view.data.copy_(sharded_weight.to(device))


def patch_cosmos_qwen_model_for_autovla(sft_checkpoint_path: str | None) -> None:
    """Patch Cosmos-RL model loading to support AutoVLA action tokens.

    Three patches are applied:

    1. **AutoConfig vocab resize**: ``AutoConfig.from_pretrained`` is patched
       so that ``qwen2_5_vl`` configs return ``vocab_size=153713``.  This
       ensures the model is constructed with the correct embedding size
       before FSDP wrapping (which makes resizing impossible).

    2. **Safetensors vocab padding**: ``HFModel.load_hf_weights_from_safetensors``
       is replaced with a version that zero-pads embedding/lm_head tensors
       when the checkpoint has fewer vocab rows than the model, instead of
       raising an assertion error.

    3. **SFT weight loading**: ``HFModel.load_hf_weights`` is wrapped so
       that after the base Qwen2.5-VL safetensors are loaded, the AutoVLA
       SFT checkpoint is loaded on top using the same FSDP-safe
       ``local_view.data.copy_()`` pattern.
    """
    from cosmos_rl.policy.model.hf_models import HFModel

    if getattr(HFModel, "_autovla_patched", False):
        if sft_checkpoint_path is not None:
            HFModel._autovla_sft_checkpoint_path = sft_checkpoint_path
        return

    HFModel._autovla_sft_checkpoint_path = sft_checkpoint_path

    # --- Patch 1: AutoConfig.from_pretrained vocab resize ------------------
    _original_autoconfig_from_pretrained = AutoConfig.from_pretrained

    def _patched_autoconfig_from_pretrained(model_name_or_path, *args, **kwargs):
        cfg = _original_autoconfig_from_pretrained(model_name_or_path, *args, **kwargs)
        if (
            getattr(cfg, "model_type", None) == "qwen2_5_vl"
            and getattr(cfg, "vocab_size", 0) < _AUTOVLA_NEEDED_VOCAB
        ):
            old_vocab = cfg.vocab_size
            cfg.vocab_size = _AUTOVLA_NEEDED_VOCAB
            if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "vocab_size"):
                cfg.text_config.vocab_size = _AUTOVLA_NEEDED_VOCAB
            logger.info(
                "AutoConfig.from_pretrained: resized qwen2_5_vl vocab_size "
                "%d -> %d for AutoVLA action tokens",
                old_vocab, _AUTOVLA_NEEDED_VOCAB,
            )
        return cfg

    AutoConfig.from_pretrained = _patched_autoconfig_from_pretrained

    # --- Patch 2: HFModel.load_hf_weights_from_safetensors vocab padding ---
    HFModel.load_hf_weights_from_safetensors = _autovla_load_hf_weights_from_safetensors

    # --- Patch 3: HFModel.load_hf_weights SFT loading ----------------------
    original_load_hf_weights = HFModel.load_hf_weights

    def patched_load_hf_weights(self, model_name_or_path, parallel_dims, device, revision=None):
        original_load_hf_weights(self, model_name_or_path, parallel_dims, device, revision=revision)

        ckpt_path = self.__class__._autovla_sft_checkpoint_path
        if ckpt_path is not None:
            _load_sft_weights_into_hfmodel(self, ckpt_path)
            self.__class__._autovla_sft_checkpoint_path = None

    HFModel._autovla_patched = True
    HFModel.load_hf_weights = patched_load_hf_weights