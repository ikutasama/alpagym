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


def setup_tokenizer(config: Any) -> Any | None:
    """Install the runtime bridge before super-init.  No tokenizer override."""
    del config
    install_autovla_runtime_bridge()
    return None


def build_data_packer(run_config: Any, cosmos_role: str | None) -> Any:
    """Build the AutoVLA replay data packer."""
    from alpagym_runtime.cosmos.packer import build_alpagym_data_packer

    install_autovla_runtime_bridge()
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

    # Load Qwen2.5-VL backbone
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(bundle_dir / "vlm"),
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(str(bundle_dir / "vlm"))

    # Load action tokenizer with codebook
    import pickle
    codebook_path = bundle_dir / "codebook_cache" / "agent_vocab.pkl"
    with open(codebook_path, "rb") as f:
        codebook_data = pickle.load(f)

    bc = model_config.bundle_config

    return AutoVLAInferenceModel(
        vlm=vlm,
        processor=processor,
        codebook=codebook_data,
        action_start_id=bc.get("action_start_id", 151665),
        num_poses=bc.get("trajectory", {}).get("num_poses", 10),
        interval_length=bc.get("trajectory", {}).get("interval_length", 0.5),
        device=device,
        use_cot=bc.get("use_cot", False),
    )


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
