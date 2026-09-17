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


def setup_tokenizer(config: Any) -> Any | None:
    """No tokenizer override needed for Qwen-Drive."""
    return None


def install_runtime_bridge() -> None:
    """Phase 3 (inference baseline): no Cosmos-RL bridge needed."""
    pass


def build_data_packer(run_config: Any, cosmos_role: str | None) -> Any:
    """Build the Qwen-Drive replay data packer. Phase 3: not used."""
    from alpagym_runtime.cosmos.packer import build_alpagym_data_packer

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
    """Load the Qwen-Drive model and build the inference adapter."""
    model_cfg = run_config.policy.model
    model_path = Path(model_cfg.path)

    planner_path = model_cfg.bundle_config.get("planner_path")
    if planner_path:
        planner_path = Path(planner_path)
    else:
        planner_path = model_path / "planner-sft"

    upstream_path = model_cfg.bundle_config.get("upstream_path")
    if upstream_path:
        _ensure_qwen_drive_on_path(upstream_path)
    else:
        _ensure_qwen_drive_on_path(model_path)

    from qwen_drive.modeling_qwen_drive import QwenDriveForPlanning

    logger.info(
        "Loading Qwen-Drive model from %s with planner from %s",
        model_path,
        planner_path,
    )

    model = QwenDriveForPlanning.from_pretrained(
        str(model_path),
        planner=str(planner_path),
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model = model.to(device)
    model.eval()

    processor = model.processor

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
    """Return the trainer-side replay input builder. Phase 3: not used."""
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
