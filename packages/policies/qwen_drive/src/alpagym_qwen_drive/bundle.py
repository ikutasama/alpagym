"""Qwen-Drive-1.0 Planning Expert policy bundle for AlpaGym closed-loop RL.

Adapter between AlpaGym's typed I/O (BatchedModelInput/BatchedModelOutput)
and Qwen-Drive's flow-matching Planning Expert pipeline.
"""

from __future__ import annotations

import functools
import logging
import sys
from pathlib import Path
from typing import Any, Callable

from alpagym_runtime.policies.registry import PolicyBundle

logger = logging.getLogger(__name__)


def _ensure_qwen_drive_on_path(model_path: str | Path) -> None:
    """Ensure the Qwen-Drive source is importable.

    The Qwen-Drive-1.0 model code (qwen_drive package) lives in the upstream
    repo's src/ directory. We add it to sys.path if not already present.
    """
    model_path = Path(model_path)
    for candidate in [
        model_path / "src",
        model_path.parent / "src",
        model_path,
    ]:
        if (candidate / "qwen_drive").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            logger.info("Added Qwen-Drive source to sys.path: %s", candidate)
            return


def _get_upstream_path(config: Any) -> str | None:
    """Extract the upstream Qwen-Drive source path from config if available."""
    try:
        return config.policy.model.bundle_config.get("upstream_path")
    except (AttributeError, KeyError, TypeError):
        return None


def setup_tokenizer(config: Any) -> Any | None:
    """Install the runtime bridge before super-init.  No tokenizer override.

    The AlpaGym RunConfig is not available here (config is the Cosmos-RL
    Config).  ``build_data_packer`` (called earlier in the entrypoint) has
    already added the upstream path to sys.path and registered AutoConfig.
    This call is a safety net in case the call order changes.
    """
    install_runtime_bridge()
    return None


def install_runtime_bridge(upstream_path: str | None = None) -> None:
    """Register the Qwen-Drive model type with transformers and Cosmos-RL.

    Cosmos-RL's ``load_model_config`` calls ``AutoConfig.from_pretrained``
    which fails for the custom ``qwen_drive`` model_type unless we register
    it first.  We also register the ``qwen_drive_planning_expert`` sub-config.

    Importing ``cosmos_wrapper`` additionally registers a ``QwenDriveCosmos``
    ``BaseModel`` subclass with Cosmos-RL's ``ModelRegistry`` so the GRPO
    trainer uses our wrapper instead of the default ``HFModel`` /
    ``AutoModelForCausalLM`` path.

    This must run before the Cosmos-RL framework tries to load the model
    config.  It is called from ``setup_tokenizer`` and ``build_data_packer``,
    both of which execute before model loading.
    """
    # Ensure Qwen-Drive source is importable so we can register its config
    if upstream_path:
        _ensure_qwen_drive_on_path(upstream_path)

    # Import cosmos_wrapper 鈥?this registers AutoConfig, AutoModel, and
    # ModelRegistry for the qwen_drive model type as a side effect.
    try:
        import alpagym_qwen_drive.cosmos_wrapper  # noqa: F401
    except ImportError as e:
        logger.warning("Failed to import cosmos_wrapper: %s", e)


def build_data_packer(run_config: Any, cosmos_role: str | None) -> Any:
    """Build the Qwen-Drive replay data packer."""
    from alpagym_runtime.cosmos.packer import build_alpagym_data_packer

    upstream = _get_upstream_path(run_config)
    install_runtime_bridge(upstream_path=upstream)

    # Inject the planner path so QwenDriveCosmos.load_hf_weights can find
    # the Planning Expert checkpoint.
    try:
        from alpagym_qwen_drive.cosmos_wrapper import set_planner_path

        planner_path = run_config.policy.model.bundle_config.get("planner_path")
        set_planner_path(planner_path)
    except (ImportError, AttributeError, KeyError):
        pass

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
    """Load the Qwen-Drive model and build the inference adapter.

    Reads the model path and planner path from the run config, loads the
    Qwen-Drive model with the SFT-trained Planning Expert, and wraps it
    in a QwenDriveInferenceModel.
    """
    model_cfg = run_config.policy.model
    model_path = Path(model_cfg.path)

    # The planner (Planning Expert) is in a separate checkpoint directory.
    planner_path = model_cfg.bundle_config.get("planner_path")
    if planner_path:
        planner_path = Path(planner_path)
    else:
        # Default: look for planner-sft in the model directory
        planner_path = model_path / "planner-sft"

    # Ensure Qwen-Drive source code is importable
    upstream_path = model_cfg.bundle_config.get("upstream_path")
    if upstream_path:
        _ensure_qwen_drive_on_path(upstream_path)
    else:
        _ensure_qwen_drive_on_path(model_path)

    # Import Qwen-Drive model class
    from qwen_drive.modeling_qwen_drive import QwenDriveForPlanning

    logger.info(
        "Loading Qwen-Drive model from %s with planner from %s",
        model_path,
        planner_path,
    )

    # Load model with the trained planner
    model = QwenDriveForPlanning.from_pretrained(
        str(model_path),
        planner=str(planner_path),
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model = model.to(device)
    model.eval()

    # The processor is a lazy property on the model 鈥?access it to trigger loading
    processor = model.processor

    # Build inference adapter
    from alpagym_qwen_drive.inference_model import QwenDriveInferenceModel

    bc = model_cfg.bundle_config
    return QwenDriveInferenceModel(
        model=model,
        processor=processor,
        device=str(device),
        dtype=dtype,
        num_future_waypoints=model_cfg.num_future_waypoints,
        step_dt_us=model_cfg.step_dt_us,
        num_inference_steps=bc.get("num_inference_steps", 10),
    )


def build_model_inputs(
    run_config: Any,
) -> Callable[[Any], tuple[dict[str, Any], Any]]:
    """Return the trainer-side replay input builder.

    Phase 3: not used (no training).
    Phase 4 will build proper flow-matching replay inputs.
    """
    from alpagym_qwen_drive.inference_model import QwenDriveInferenceModel

    return functools.partial(
        QwenDriveInferenceModel.build_trainer_model_inputs,
    )


def get_bundle() -> PolicyBundle:
    """Return the Qwen-Drive runtime hooks."""
    return PolicyBundle(
        setup_tokenizer=setup_tokenizer,
        build_data_packer=build_data_packer,
        install_runtime_bridge=install_runtime_bridge,
        load_inference_model=load_inference_model,
        build_model_inputs=build_model_inputs,
    )
