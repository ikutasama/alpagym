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

from alpagym_autovla.action_tokens import action_token_mask, ensure_action_token_layout

logger = logging.getLogger(__name__)

# Module-level config set by setup_tokenizer / install_autovla_runtime_bridge
_trainer_config: dict[str, Any] = {}


def set_trainer_config(
    model_path: str,
    action_start_id: int = 151665,
    action_token_count: int = 2048,
    num_poses: int = 10,
    use_cot: bool = False,
) -> None:
    """Store config needed by the patched training forward."""
    _trainer_config["model_path"] = Path(model_path)
    _trainer_config["action_start_id"] = action_start_id
    _trainer_config["action_token_count"] = action_token_count
    _trainer_config["num_poses"] = num_poses
    _trainer_config["use_cot"] = use_cot
    _trainer_config["_processor"] = None  # lazy init
    _trainer_config["_action_token_ids"] = None  # lazy validation


def _get_processor():
    """Lazily create and cache the Qwen AutoProcessor."""
    if _trainer_config.get("_processor") is None:
        from transformers import AutoProcessor
        _trainer_config["_processor"] = AutoProcessor.from_pretrained(
            str(_trainer_config["model_path"])
        )
    return _trainer_config["_processor"]


def _get_action_token_ids(device: torch.device) -> torch.Tensor:
    """Return validated AutoVLA action-token ids on ``device``."""
    if _trainer_config.get("_action_token_ids") is None:
        _trainer_config["_action_token_ids"] = ensure_action_token_layout(
            _get_processor().tokenizer,
            action_start_id=int(_trainer_config.get("action_start_id", 151665)),
            n_bins=int(_trainer_config.get("action_token_count", 2048)),
            source="AutoVLA trainer tokenizer",
        )
    return _trainer_config["_action_token_ids"].to(device)


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
    ``AutoVLAInferenceModel._build_qwen_inputs`` + ``_build_user_content``
    — same chat template, same video grouping, same pixel constraints,
    same system text (including CoT variant) — so the rebuilt
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

    # Velocity/acceleration from ego history (same as inference)
    if ego_history_xyz.shape[0] >= 2:
        diff = ego_history_xyz[1:] - ego_history_xyz[:-1]
        velocity = float(torch.norm(diff[-1][:2]).item())
        if diff.shape[0] >= 2:
            acceleration = float(torch.norm(diff[-1][:2] - diff[-2][:2]).item())
        else:
            acceleration = 0.0
    else:
        velocity = 0.0
        acceleration = 0.0

    # Instruction from route (same as inference)
    if route_xy is not None and route_xy.shape[0] > 0:
        first_wp = route_xy[0]
        if abs(float(first_wp[0])) > abs(float(first_wp[1])):
            instruction = "turn left" if float(first_wp[0]) < 0 else "turn right"
        else:
            instruction = "move forward"
    else:
        instruction = "move forward"

    # Build user content — mirrors _build_user_content exactly
    min_pixels = 28 * 28 * 128
    max_pixels = 28 * 28 * 128
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "The autonomous vehicle is equipped with cameras enabling perception of the surrounding environment."},
    ]

    num_images = len(pil_images)
    if num_images >= 12:
        cam_names = ["front", "front-left", "front-right"]
        frames_per_cam = num_images // 3
        for cam_idx in range(3):
            start = cam_idx * frames_per_cam
            cam_frames = pil_images[start:start + frames_per_cam]
            content.append({"type": "text", "text": f"Video {cam_idx+1}: {cam_names[cam_idx]} view, {frames_per_cam} frames at 2 Hz."})
            content.append({
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": cam_frames,
            })
    elif num_images >= 4:
        content.append({"type": "text", "text": f"Front view video, {num_images} frames at 2 Hz."})
        content.append({
            "type": "video",
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "video": pil_images,
        })
    else:
        for img in pil_images:
            content.append({"type": "image", "image": img, "min_pixels": min_pixels, "max_pixels": max_pixels})

    content.append({
        "type": "text",
        "text": (
            f"The current velocity of the vehicle is {velocity:.3f} m/s, "
            f"and the current acceleration is {acceleration:.3f} m/s^2. "
            f"The driving instruction is: {instruction}. "
            f"Based on this information, plan the action trajectory for the "
            f"autonomous vehicle over the next five seconds."
        ),
    })

    # System text — matches inference model's CoT / non-CoT variants
    use_cot = _trainer_config.get("use_cot", False)
    if use_cot:
        system_text = (
            "You are an Advanced Driver Assistance and Full Self-Driving System. "
            "You will be provided with video observations from the ego vehicle's "
            "surrounding cameras, along with the vehicle's current dynamic states. "
            "Your task is to predict the most appropriate driving action for the "
            "next five seconds."
        )
    else:
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
        processor_kwargs: dict[str, Any] = {
            "text": [text],
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
    except Exception as exc:
        logger.warning("Falling back to image-only Qwen processor path: %s", exc)
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
    completion_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward prompt + completion and compute sum log-prob for the action part.

    If ``completion_ids`` is provided (the full generated completion including
    assistant prefix and any intermediate text), it is appended to the prompt
    so the model sees the same context as during rollout.  Only tokens with
    exact AutoVLA action-token ids contribute to the sum, matching the
    rollout-side ``_compute_logprob``.

    If ``completion_ids`` is not available, falls back to appending only
    ``action_token_ids`` (less accurate for CoT models).
    """
    prompt_ids = model_inputs["input_ids"]  # [1, prompt_len]
    prompt_length = prompt_ids.shape[1]

    if completion_ids is not None:
        comp_ids = completion_ids.to(prompt_ids.dtype).to(prompt_ids.device)
        prompt_completion_ids = torch.cat([prompt_ids, comp_ids.unsqueeze(0)], dim=1)
    else:
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

    logits = logits[:, :-1, :]  # [1, L-1, V]
    target_ids = prompt_completion_ids[:, 1:]  # [1, L-1]

    log_probs = torch.log_softmax(logits, dim=-1)  # [1, L-1, V]
    per_token_logps = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # [1, L-1]

    completion_part_ids = target_ids[:, prompt_length - 1:]
    completion_logps = per_token_logps[:, prompt_length - 1:]

    action_ids = _get_action_token_ids(completion_part_ids.device)
    action_mask = action_token_mask(completion_part_ids, action_ids)
    if action_mask.any():
        logprob = completion_logps[action_mask].sum()
    else:
        logprob = completion_logps.new_zeros(())

    return logprob.squeeze() if logprob.dim() > 0 else logprob


def _compute_action_logprobs_from_qwen_inputs(
    qwen_model: Any,
    qwen_inputs: dict[str, torch.Tensor],
    prompt_length: int,
    action_token_ids: torch.Tensor,
    completion_ids: torch.Tensor | None = None,
    teacher_model: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute action log-probs using the exact Qwen inputs from rollout.

    Uses the persisted ``qwen_inputs`` (input_ids, pixel_values,
    image_grid_thw, etc.) and ``prompt_length`` so the trainer scores
    the same token alignment the rollout used — no rebuilding.

    When ``teacher_model`` is provided, also computes per-sample KL
    divergence between the current and reference policy over the action
    tokens.
    """
    prompt_ids = qwen_inputs["input_ids"]  # [1, prompt_len]

    if completion_ids is not None:
        comp_ids = completion_ids.to(prompt_ids.dtype).to(prompt_ids.device)
        if comp_ids.dim() == 1:
            comp_ids = comp_ids.unsqueeze(0)
        prompt_completion_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    else:
        act_ids = action_token_ids.to(prompt_ids.dtype).to(prompt_ids.device)
        if act_ids.dim() == 1:
            act_ids = act_ids.unsqueeze(0)
        prompt_completion_ids = torch.cat([prompt_ids, act_ids], dim=1)

    forward_kwargs = {
        k: v for k, v in qwen_inputs.items()
        if k not in ("input_ids", "attention_mask", "prompt_length")
    }
    outputs = qwen_model(
        input_ids=prompt_completion_ids,
        attention_mask=torch.ones_like(prompt_completion_ids),
        **forward_kwargs,
    )
    logits = outputs.logits  # [1, L, V]

    logits = logits[:, :-1, :]  # [1, L-1, V]
    target_ids = prompt_completion_ids[:, 1:]  # [1, L-1]

    log_probs = torch.log_softmax(logits, dim=-1)  # [1, L-1, V]
    per_token_logps = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # [1, L-1]

    completion_part_ids = target_ids[:, prompt_length - 1:]
    completion_logps = per_token_logps[:, prompt_length - 1:]

    action_ids = _get_action_token_ids(completion_part_ids.device)
    action_mask = action_token_mask(completion_part_ids, action_ids)
    if action_mask.any():
        logprob = completion_logps[action_mask].sum()
    else:
        logprob = completion_logps.new_zeros(())

    logprob_out = logprob.squeeze() if logprob.dim() > 0 else logprob

    kl_div = None
    if teacher_model is not None:
        with torch.no_grad():
            ref_outputs = teacher_model(
                input_ids=prompt_completion_ids,
                attention_mask=torch.ones_like(prompt_completion_ids),
                **forward_kwargs,
            )
            ref_logits = ref_outputs.logits[:, :-1, :]
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            ref_per_token_logps = ref_log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            ref_completion_logps = ref_per_token_logps[:, prompt_length - 1:]

            if action_mask.any():
                cur_lp = completion_logps[action_mask]
                ref_lp = ref_completion_logps[action_mask]
                kl_div = (cur_lp.exp() * (cur_lp - ref_lp)).sum()
            else:
                kl_div = completion_logps.new_zeros(())
            kl_div = kl_div.squeeze() if kl_div.dim() > 0 else kl_div

    return logprob_out, kl_div


def autovla_training_forward(
    trainer: Any,
    model_inputs: dict[str, Any],
    teacher_model: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """AutoVLA training forward: compute action log_probs using persisted Qwen inputs.

    When the rollout persisted its Qwen processor inputs (``qwen_inputs``),
    those are used directly — guaranteeing token-level alignment with the
    rollout-side logprob.  Otherwise, falls back to rebuilding from raw
    camera frames (legacy path, less accurate).

    When ``teacher_model`` is provided, also computes per-sample KL
    divergence between the current and reference policy over the action
    tokens.

    Returns ``(log_probs, kl_divs)`` where ``log_probs`` is ``[B]`` and
    ``kl_divs`` is ``[B]`` or ``None``.
    """
    qwen_inputs_list = model_inputs.get("qwen_inputs")
    action_token_ids = model_inputs.get("action_token_ids")
    completion_ids = model_inputs.get("completion_ids")

    qwen_model = _find_qwen_model(trainer.model)
    device = next(qwen_model.parameters()).device
    ref_qwen = _find_qwen_model(teacher_model) if teacher_model is not None else None

    if qwen_inputs_list is not None:
        if isinstance(qwen_inputs_list, dict):
            stacked_qi = qwen_inputs_list
            first_tensor = next(
                (v for v in stacked_qi.values() if isinstance(v, torch.Tensor)),
                None,
            )
            batch_size = first_tensor.shape[0] if first_tensor is not None else 1
            qwen_inputs_per_sample = []
            for b in range(batch_size):
                sample_qi: dict[str, Any] = {}
                for k, v in stacked_qi.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == batch_size:
                        sample_qi[k] = v[b]
                    elif isinstance(v, torch.Tensor):
                        sample_qi[k] = v
                    else:
                        sample_qi[k] = v
                qwen_inputs_per_sample.append(sample_qi)
            qwen_inputs_list = qwen_inputs_per_sample
        if not isinstance(qwen_inputs_list, list):
            qwen_inputs_list = [qwen_inputs_list]
        batch_size = len(qwen_inputs_list)
        logger.info(
            "AutoVLA training forward: using persisted qwen_inputs, batch=%d, "
            "action_token_ids=%s, completion_ids=%s, has_teacher=%s",
            batch_size,
            tuple(action_token_ids.shape) if action_token_ids is not None else None,
            tuple(completion_ids.shape) if completion_ids is not None else None,
            ref_qwen is not None,
        )
        all_logprobs: list[torch.Tensor] = []
        all_kl_divs: list[torch.Tensor] = []
        for b in range(batch_size):
            qwen_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in qwen_inputs_list[b].items()
            }
            prompt_length = int(qwen_inputs.pop("prompt_length"))
            if action_token_ids is not None:
                act_ids = action_token_ids
                if act_ids.dim() == 1:
                    act_ids = act_ids.unsqueeze(0)
                comp_ids = completion_ids
                if comp_ids is not None and comp_ids.dim() == 1:
                    comp_ids = comp_ids.unsqueeze(0)
                logprob, kl_div = _compute_action_logprobs_from_qwen_inputs(
                    qwen_model, qwen_inputs, prompt_length,
                    act_ids[b] if act_ids.dim() > 1 else act_ids[0],
                    comp_ids[b] if comp_ids is not None and comp_ids.dim() > 1 else comp_ids,
                    teacher_model=ref_qwen,
                )
            else:
                logprob = torch.tensor(0.0, device=device)
                kl_div = None
            all_logprobs.append(logprob)
            if kl_div is not None:
                all_kl_divs.append(kl_div)
        log_probs = torch.stack(all_logprobs)
        kl_divs = torch.stack(all_kl_divs) if all_kl_divs else None
        return log_probs, kl_divs

    camera_frames = model_inputs["camera_frames"]
    ego_history_xyz = model_inputs["ego_history_xyz"]
    route_xy = model_inputs.get("route_xy")

    logger.info(
        "AutoVLA training forward (legacy rebuild): camera_frames=%s dtype=%s, "
        "ego_history_xyz=%s, route_xy=%s, action_token_ids=%s, completion_ids=%s",
        tuple(camera_frames.shape), camera_frames.dtype,
        tuple(ego_history_xyz.shape),
        tuple(route_xy.shape) if route_xy is not None else None,
        tuple(action_token_ids.shape) if action_token_ids is not None else None,
        tuple(completion_ids.shape) if completion_ids is not None else None,
    )

    if camera_frames.dim() == 4:
        camera_frames = camera_frames.unsqueeze(0)
    if ego_history_xyz.dim() == 2:
        ego_history_xyz = ego_history_xyz.unsqueeze(0)
    if route_xy is not None and route_xy.dim() == 2:
        route_xy = route_xy.unsqueeze(0)
    if action_token_ids is not None and action_token_ids.dim() == 1:
        action_token_ids = action_token_ids.unsqueeze(0)
    if completion_ids is not None and completion_ids.dim() == 1:
        completion_ids = completion_ids.unsqueeze(0)

    batch_size = camera_frames.shape[0]

    all_logprobs = []
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
                completion_ids[b] if completion_ids is not None else None,
            )
        else:
            logprob = torch.tensor(0.0, device=device)
        all_logprobs.append(logprob)

    log_probs = torch.stack(all_logprobs)
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
        # AutoVLA path: camera_frames or qwen_inputs present
        if isinstance(model_inputs, dict) and (
            "camera_frames" in model_inputs or "qwen_inputs" in model_inputs
        ):
            return autovla_training_forward(
                self, model_inputs,
                teacher_model=self._reference_model,
            )
        # Original path for all other policies
        forward_kwargs = to_device_recursive(model_inputs, self.device)
        if self._reference_model is not None:
            forward_kwargs["teacher_model"] = self._reference_model
        result = self.model(**forward_kwargs)
        return result["log_probs"], result.get("kl_div")

    patched_forward_with_reference._autovla_patch = True  # type: ignore[attr-defined]
    AlpagymGRPOTrainer._forward_with_reference = patched_forward_with_reference
    logger.info("Patched AlpagymGRPOTrainer._forward_with_reference for AutoVLA")
