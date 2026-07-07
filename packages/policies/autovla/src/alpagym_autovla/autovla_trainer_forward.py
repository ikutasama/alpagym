"""AutoVLA training forward adapter.

Patches ``AlpagymGRPOTrainer._forward_with_reference`` to handle AutoVLA's
raw replay inputs (camera_frames, ego_history, action_token_ids) by rebuilding
Qwen2.5-VL processor inputs and computing per-token log_probs for the
recorded action tokens — the same logic the rollout-side
``AutoVLAInferenceModel._compute_logprob`` uses, but with gradients enabled
and no dependency on the inference adapter.

This mirrors the Alpamayo R1 pattern (``_patch_expert_model_cosmos_bt_flatten``)
where the rollout persists raw inputs and the trainer-side forward rebuilds
the full Qwen view.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Module-level config set by setup_tokenizer / install_autovla_runtime_bridge
_trainer_config: dict[str, Any] = {}


def set_trainer_config(
    model_path: str,
    action_start_id: int = 151665,
    num_poses: int = 10,
) -> None:
    """Store config needed by the patched training forward."""
    _trainer_config["model_path"] = Path(model_path)
    _trainer_config["action_start_id"] = action_start_id
    _trainer_config["num_poses"] = num_poses
    _trainer_config["_processor"] = None  # lazy init


def _get_processor():
    """Lazily create and cache the Qwen AutoProcessor."""
    if _trainer_config.get("_processor") is None:
        from transformers import AutoProcessor
        _trainer_config["_processor"] = AutoProcessor.from_pretrained(
            str(_trainer_config["model_path"])
        )
    return _trainer_config["_processor"]


def _find_qwen_model(cosmos_model: Any) -> Any:
    """Find the underlying Qwen2_5_VLForConditionalGeneration from a Cosmos-RL wrapper."""
    # Qwen2_5_VLBaseModel has .model
    if hasattr(cosmos_model, "model") and hasattr(cosmos_model.model, "forward"):
        return cosmos_model.model
    # HFModel might have .hf_model or .model
    for attr in ("hf_model", "model", "_model"):
        candidate = getattr(cosmos_model, attr, None)
        if candidate is not None and hasattr(candidate, "forward") and hasattr(candidate, "parameters"):
            return candidate
    return cosmos_model


def _build_qwen_inputs_for_training(
    camera_frames: torch.Tensor,
    ego_history_xyz: torch.Tensor,
    route_xy: torch.Tensor | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Rebuild Qwen2.5-VL processor inputs from raw camera frames.

    This is the trainer-side mirror of
    ``AutoVLAInferenceModel._build_qwen_inputs`` — same chat template,
    same video grouping, same pixel constraints — so the rebuilt
    ``pixel_values`` / ``input_ids`` match what the rollout used.
    """
    from PIL import Image
    import numpy as np

    processor = _get_processor()

    num_cams = camera_frames.shape[0]
    pil_images = []
    for i in range(num_cams):
        frame = camera_frames[i].cpu().numpy()
        # AlPaGym camera frames are CHW [C, H, W]; PIL expects HWC [H, W, C].
        if frame.ndim == 3 and frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))
        pil_images.append(Image.fromarray(frame))

    # Velocity from ego history
    if ego_history_xyz.shape[0] >= 2:
        diff = ego_history_xyz[1:] - ego_history_xyz[:-1]
        velocity = float(torch.norm(diff[-1][:2]).item())
    else:
        velocity = 0.0

    # Instruction from route
    if route_xy is not None and route_xy.shape[0] > 0:
        first_wp = route_xy[0]
        if abs(float(first_wp[0])) > abs(float(first_wp[1])):
            instruction = "turn left" if float(first_wp[0]) < 0 else "turn right"
        else:
            instruction = "move forward"
    else:
        instruction = "move forward"

    # Build user content (same structure as inference)
    min_pixels = 28 * 28 * 128
    max_pixels = 28 * 28 * 128
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "The autonomous vehicle is equipped with cameras enabling perception of the surrounding environment."},
    ]

    frames_per_cam = max(1, num_cams // 3) if num_cams >= 3 else num_cams
    cam_names = ["front", "left", "right"]

    if num_cams >= 3:
        for cam_idx in range(3):
            start = cam_idx * frames_per_cam
            cam_frames = pil_images[start:start + frames_per_cam]
            content.append({
                "type": "text",
                "text": f"Video {cam_idx+1}: {cam_names[cam_idx]} view, {frames_per_cam} frames at 2 Hz.",
            })
            content.append({
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": cam_frames,
            })
    elif num_cams >= 4:
        content.append({
            "type": "text",
            "text": f"Front view video, {num_cams} frames at 2 Hz.",
        })
        content.append({
            "type": "video",
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "video": pil_images,
        })
    else:
        for img in pil_images:
            content.append({
                "type": "image",
                "image": img,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            })

    content.append({
        "type": "text",
        "text": (
            f"The current velocity of the vehicle is {velocity:.3f} m/s, "
            f"the acceleration is 0.000 m/s^2. "
            f"The driving instruction is: {instruction}. "
            f"Please predict the future trajectory."
        ),
    })

    system_text = (
        "You are an Advanced Driver Assistance and Full Self-Driving System. "
        "You will receive visual observations from the ego vehicle's cameras "
        "and dynamic information about the vehicle's current state. "
        "Your task is to predict the optimal driving action for the next five seconds."
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_vision_id=True,
    )

    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        processor_kwargs = {
            "text": [text],
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
    except Exception:
        processor_kwargs = {
            "text": [text],
            "images": pil_images,
            "padding": True,
            "return_tensors": "pt",
        }

    inputs = processor(**processor_kwargs)
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()}


def _compute_action_logprobs(
    qwen_model: Any,
    model_inputs: dict[str, torch.Tensor],
    action_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Forward prompt + action tokens and compute sum log-prob for the action part."""
    prompt_ids = model_inputs["input_ids"]  # [1, prompt_len]
    prompt_length = prompt_ids.shape[1]

    # Append action tokens to prompt
    act_ids = action_token_ids.to(prompt_ids.dtype).to(prompt_ids.device)
    prompt_completion_ids = torch.cat([prompt_ids, act_ids.unsqueeze(0)], dim=1)

    forward_kwargs = {
        k: v for k, v in model_inputs.items()
        if k not in ("input_ids", "attention_mask")
    }
    outputs = qwen_model(
        input_ids=prompt_completion_ids,
        attention_mask=torch.ones_like(prompt_completion_ids),
        **forward_kwargs,
    )
    logits = outputs.logits  # [1, L, V]

    # Shift for next-token prediction
    logits = logits[:, :-1, :]  # [1, L-1, V]
    target_ids = prompt_completion_ids[:, 1:]  # [1, L-1]

    log_probs = torch.log_softmax(logits, dim=-1)  # [1, L-1, V]
    per_token_logps = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # [1, L-1]

    # Sum only action token logprobs (completion part)
    completion_logps = per_token_logps[:, prompt_length - 1:]
    logprob = completion_logps.sum(dim=-1)  # [1]
    return logprob.squeeze(0)  # scalar


def autovla_training_forward(
    trainer: Any,
    model_inputs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """AutoVLA training forward: rebuild Qwen inputs, compute action log_probs.

    Returns ``(log_probs, None)`` where ``log_probs`` is ``[B]``.
    """
    camera_frames = model_inputs["camera_frames"]
    ego_history_xyz = model_inputs["ego_history_xyz"]
    route_xy = model_inputs.get("route_xy")
    action_token_ids = model_inputs.get("action_token_ids")

    logger.info(
        "AutoVLA training forward shapes: camera_frames=%s dtype=%s, "
        "ego_history_xyz=%s, route_xy=%s, action_token_ids=%s",
        tuple(camera_frames.shape), camera_frames.dtype,
        tuple(ego_history_xyz.shape),
        tuple(route_xy.shape) if route_xy is not None else None,
        tuple(action_token_ids.shape) if action_token_ids is not None else None,
    )

    # Ensure batch dim
    if camera_frames.dim() == 4:
        camera_frames = camera_frames.unsqueeze(0)
    if ego_history_xyz.dim() == 2:
        ego_history_xyz = ego_history_xyz.unsqueeze(0)
    if route_xy is not None and route_xy.dim() == 2:
        route_xy = route_xy.unsqueeze(0)
    if action_token_ids is not None and action_token_ids.dim() == 1:
        action_token_ids = action_token_ids.unsqueeze(0)

    batch_size = camera_frames.shape[0]
    qwen_model = _find_qwen_model(trainer.model)
    device = next(qwen_model.parameters()).device

    all_logprobs: list[torch.Tensor] = []
    for b in range(batch_size):
        qwen_inputs = _build_qwen_inputs_for_training(
            camera_frames[b],
            ego_history_xyz[b],
            route_xy[b] if route_xy is not None else None,
            device,
        )
        if action_token_ids is not None:
            logprob = _compute_action_logprobs(
                qwen_model, qwen_inputs, action_token_ids[b],
            )
        else:
            # No action tokens — shouldn't happen in RL training
            logprob = torch.tensor(0.0, device=device)
        all_logprobs.append(logprob)

    log_probs = torch.stack(all_logprobs)  # [B]
    return log_probs, None


def patch_trainer_forward() -> None:
    """Patch ``AlpagymGRPOTrainer._forward_with_reference`` for AutoVLA.

    Only intercepts when ``camera_frames`` is present in model_inputs
    (AutoVLA-specific).  All other policies fall through to the original
    method unchanged.
    """
    from alpagym_runtime.cosmos.trainer import AlpagymGRPOTrainer
    from alpagym_runtime.tensor_utils import to_device_recursive

    original = AlpagymGRPOTrainer._forward_with_reference
    if getattr(original, "_autovla_patch", False):
        return

    def patched_forward_with_reference(
        self,
        model_inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # AutoVLA path: camera_frames + action_token_ids present
        if isinstance(model_inputs, dict) and "camera_frames" in model_inputs:
            return autovla_training_forward(self, model_inputs)
        # Original path for all other policies
        forward_kwargs = to_device_recursive(model_inputs, self.device)
        if self._reference_model is not None:
            forward_kwargs["teacher_model"] = self._reference_model
        result = self.model(**forward_kwargs)
        return result["log_probs"], result.get("kl_div")

    patched_forward_with_reference._autovla_patch = True  # type: ignore[attr-defined]
    AlpagymGRPOTrainer._forward_with_reference = patched_forward_with_reference
    logger.info("Patched AlpagymGRPOTrainer._forward_with_reference for AutoVLA")
