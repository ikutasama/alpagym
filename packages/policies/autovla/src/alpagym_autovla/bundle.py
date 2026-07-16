"""AutoVLA policy bundle for AlpaGym closed-loop RL.

Adapter between AlpaGym's typed I/O (BatchedModelInput/BatchedModelOutput)
and AutoVLA's Qwen2.5-VL-3B + discrete action token pipeline.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Callable

from alpagym_runtime.policies.registry import PolicyBundle

logger = logging.getLogger(__name__)


def install_autovla_runtime_bridge() -> None:
    """Register AutoVLA's Qwen2.5-VL model and weight mapper in Cosmos-RL.

    Cosmos-RL auto-discovers and registers a built-in
    ``Qwen2_5_VLConditionalModel`` for model_type ``qwen2_5_vl`` at import
    time.  Attempting to register our custom classes for the same
    model_type raises ``ValueError``.

    We try to register our custom ``Qwen2_5_VLWeightMapper`` (which routes
    ``visual.*`` keys that the default mapper skips).  If the built-in is
    already registered we fall back to it — sufficient for single-GPU
    smoke tests where no cross-process weight sync is needed.

    We also patch ``AlpagymGRPOTrainer._forward_with_reference`` so the
    trainer can handle AutoVLA's raw replay inputs (camera_frames,
    action_token_ids) by rebuilding Qwen2.5-VL processor inputs and
    computing per-token log_probs — the same pattern Alpamayo R1 uses.
    """
    from cosmos_rl.policy.model.base import ModelRegistry
    from alpagym_autovla.cosmos_bridge import Qwen2_5_VLBaseModel, Qwen2_5_VLWeightMapper

    try:
        ModelRegistry.register_model(
            Qwen2_5_VLBaseModel,
            Qwen2_5_VLWeightMapper,
        )
    except ValueError:
        logger.debug(
            "qwen2_5_vl already registered by Cosmos-RL auto-discovery; "
            "using built-in model."
        )

    from alpagym_autovla.autovla_trainer_forward import patch_trainer_forward
    patch_trainer_forward()

    from alpagym_autovla.cosmos_bridge import patch_cosmos_qwen_model_for_autovla
    patch_cosmos_qwen_model_for_autovla(sft_checkpoint_path=None)


def setup_tokenizer(config: Any) -> Any | None:
    """Install the runtime bridge before super-init.  No tokenizer override.

    The AlpaGym RunConfig is not available here (config is the Cosmos-RL
    Config).  Trainer config (model path, action_start_id) is set later
    in ``build_data_packer`` which receives the full RunConfig.
    """
    install_autovla_runtime_bridge()
    return None


def build_data_packer(run_config: Any, cosmos_role: str | None) -> Any:
    """Build the AutoVLA replay data packer."""
    from alpagym_runtime.cosmos.packer import build_alpagym_data_packer

    install_autovla_runtime_bridge()

    from alpagym_autovla.autovla_trainer_forward import set_trainer_config, _trainer_config
    if not _trainer_config:
        from pathlib import Path
        bc = run_config.policy.model.bundle_config
        bundle_dir = Path(run_config.policy.model.path)
        vlm_path = bundle_dir / "vlm"
        if not vlm_path.is_dir():
            vlm_path = bundle_dir
        set_trainer_config(
            model_path=str(vlm_path),
            action_start_id=bc.get("action_start_id", 151665),
            action_token_count=bc.get("action_token_count", 2048),
            num_poses=bc.get("trajectory", {}).get("num_poses", 10),
            use_cot=bc.get("use_cot", False),
        )

    # Set SFT checkpoint path so the policy model can load it
    from alpagym_autovla.cosmos_bridge import patch_cosmos_qwen_model_for_autovla
    bc = run_config.policy.model.bundle_config
    ckpt_path = bc.get("checkpoint_path")
    if ckpt_path:
        patch_cosmos_qwen_model_for_autovla(sft_checkpoint_path=ckpt_path)

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
    """Load AutoVLA checkpoint and return its inference adapter."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    from alpagym_autovla.inference_model import AutoVLAInferenceModel

    model_config = run_config.policy.model
    if dtype != torch.bfloat16:
        logger.warning("AutoVLA expects dtype=bfloat16; got %r. Forcing bf16.", dtype)
        dtype = torch.bfloat16

    bundle_dir = Path(model_config.path)

    # VLM path: AutoVLA bundles have a vlm/ subdir; raw Qwen checkpoints don't.
    vlm_path = bundle_dir / "vlm"
    if not vlm_path.is_dir():
        vlm_path = bundle_dir

    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(vlm_path),
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(str(vlm_path))

    # Action tokenizer codebook: try bundle dir, then AUTOVLA_REPO_PATH.
    import os
    import pickle
    codebook_path = bundle_dir / "codebook_cache" / "agent_vocab.pkl"
    if not codebook_path.exists():
        repo = Path(os.environ.get("AUTOVLA_REPO_PATH", ""))
        if repo.is_dir():
            codebook_path = repo / "codebook_cache" / "agent_vocab.pkl"
    if not codebook_path.exists():
        raise FileNotFoundError(
            f"agent_vocab.pkl not found. Checked:\n"
            f"  {bundle_dir}/codebook_cache/agent_vocab.pkl\n"
            f"  $AUTOVLA_REPO_PATH/codebook_cache/agent_vocab.pkl\n"
            f"Set model.path to an AutoVLA bundle, or set AUTOVLA_REPO_PATH."
        )
    with open(codebook_path, "rb") as f:
        codebook_data = pickle.load(f)

    bc = model_config.bundle_config

    model = AutoVLAInferenceModel(
        vlm=vlm,
        processor=processor,
        codebook=codebook_data,
        action_start_id=bc.get("action_start_id", 151665),
        num_poses=bc.get("trajectory", {}).get("num_poses", 10),
        interval_length=bc.get("trajectory", {}).get("interval_length", 0.5),
        device=device,
        use_cot=bc.get("use_cot", False),
    )

    # Load SFT checkpoint (.ckpt) if provided.  AutoVLA releases a
    # PyTorch-Lightning checkpoint whose state_dict keys are prefixed
    # 'autovla.vlm.' — strip both to match Qwen2_5_VLForConditionalGeneration.
    ckpt_path = bc.get("checkpoint_path")
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw = ckpt["state_dict"]
        new_state: dict[str, torch.Tensor] = {}
        for k, v in raw.items():
            if k.startswith("autovla."):
                k = k[len("autovla."):]
            if k.startswith("vlm."):
                k = k[len("vlm."):]
            # SFT checkpoint uses model.layers.*, model.embed_tokens.*, visual.*
            # but Qwen2_5_VLForConditionalGeneration uses
            # model.language_model.layers.*, model.language_model.embed_tokens.*,
            # model.visual.*
            if k.startswith("model.layers."):
                k = "model.language_model." + k[len("model."):]
            elif k.startswith("model.embed_tokens") or k.startswith("model.norm"):
                k = "model.language_model." + k[len("model."):]
            elif k.startswith("visual."):
                k = "model." + k
            new_state[k] = v
        # SFT checkpoint has extended vocab (153713); resize before loading
        sft_embed = new_state.get("model.language_model.embed_tokens.weight")
        if sft_embed is not None and sft_embed.shape[0] > model._vlm.config.vocab_size:
            model._vlm.resize_token_embeddings(sft_embed.shape[0])
        missing, unexpected = model._vlm.load_state_dict(new_state, strict=False)
        logger.info(
            "Loaded AutoVLA SFT checkpoint from %s "
            "(missing=%d, unexpected=%d)",
            ckpt_path, len(missing), len(unexpected),
        )

    return model


def build_model_inputs(
    run_config: Any,
) -> Callable[[Any], tuple[dict[str, Any], Any]]:
    """Return the trainer-side replay input builder."""
    from alpagym_autovla.inference_model import AutoVLAInferenceModel

    return functools.partial(
        AutoVLAInferenceModel.build_trainer_model_inputs,
        action_start_id=run_config.policy.model.bundle_config.get("action_start_id", 151665),
    )


def get_bundle() -> PolicyBundle:
    """Return the AutoVLA runtime hooks."""
    return PolicyBundle(
        setup_tokenizer=setup_tokenizer,
        build_data_packer=build_data_packer,
        install_runtime_bridge=install_autovla_runtime_bridge,
        load_inference_model=load_inference_model,
        build_model_inputs=build_model_inputs,
    )
