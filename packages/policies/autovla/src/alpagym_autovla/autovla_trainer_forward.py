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
import os
from pathlib import Path
from typing import Any

import torch

from alpagym_autovla.action_tokens import action_token_mask, ensure_action_token_layout, sanitize_completion_vision_tokens

logger = logging.getLogger(__name__)

# Module-level config set by setup_tokenizer / install_autovla_runtime_bridge
_trainer_config: dict[str, Any] = {}


def set_trainer_config(
    model_path: str,
    action_start_id: int = 151665,
    action_token_count: int = 2048,
    num_poses: int = 10,
    use_cot: bool = False,
    interval_length: float = 0.5,
) -> None:
    """Store config needed by the patched training forward."""
    _trainer_config["model_path"] = Path(model_path)
    _trainer_config["action_start_id"] = action_start_id
    _trainer_config["action_token_count"] = action_token_count
    _trainer_config["num_poses"] = num_poses
    _trainer_config["use_cot"] = use_cot
    _trainer_config["interval_length"] = interval_length
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
        dt = _trainer_config.get("interval_length", 0.5)
        velocity = float(torch.norm(diff[-1][:2]).item()) / dt
        if diff.shape[0] >= 2:
            acceleration = float(torch.norm(diff[-1][:2] - diff[-2][:2]).item()) / (dt * dt)
        else:
            acceleration = 0.0
    else:
        velocity = 0.0
        acceleration = 0.0

    # History waypoints for prompt: last 4 positions (past 2s at 0.5s).
    # Fixed 2 decimal places + always 4 points for consistent token length.
    ego_hist = ego_history_xyz
    if ego_hist.dim() > 2:
        ego_hist = ego_hist.squeeze(0)
    if ego_hist.shape[0] >= 4:
        hist = ego_hist[-4:, :2]
    elif ego_hist.shape[0] >= 1:
        hist = ego_hist[:, :2]
    else:
        hist = torch.zeros(1, 2, device=ego_hist.device)
    if hist.shape[0] < 4:
        pad = torch.zeros(4 - hist.shape[0], 2, device=hist.device)
        hist = torch.cat([pad, hist], dim=0)
    history_xy = [[f"{float(hist[i, 0]):7.2f}", f"{float(hist[i, 1]):7.2f}"]
                   for i in range(4)]

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
            f"The recent trajectory of the ego vehicle (x, y) in ego frame "
            f"over the past 2 seconds at 0.5s intervals is: {history_xy}. "
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

    prompt_completion_ids = sanitize_completion_vision_tokens(
        prompt_completion_ids, prompt_length, qwen_model.config,
    )

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
    target_ids = prompt_completion_ids[:, 1:].to(logits.device)  # [1, L-1]

    # Memory-efficient logprob: avoid materializing full [1, L-1, V] float32.
    target_logits = logits.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    max_logits = logits.max(dim=-1).values
    shifted = logits - max_logits.unsqueeze(-1)
    exp_sum = shifted.exp().sum(dim=-1)
    lse = max_logits.float() + torch.log(exp_sum.float())
    per_token_logps = target_logits.float() - lse
    del shifted, exp_sum, max_logits, target_logits, lse

    completion_part_ids = target_ids[:, prompt_length - 1:]
    completion_logps = per_token_logps[:, prompt_length - 1:]

    action_ids = _get_action_token_ids(completion_part_ids.device)
    action_mask = action_token_mask(completion_part_ids, action_ids)
    if action_mask.any():
        logprob = completion_logps[action_mask].sum()
    else:
        # No action tokens in completion — use mean of all completion logps
        # to keep the tensor in the autograd graph (avoids "does not require
        # grad" crash on backward when rollouts produce 0 action tokens).
        logprob = completion_logps.mean() * 0.0

    return logprob.squeeze() if logprob.dim() > 0 else logprob


def _compute_action_logprobs_from_qwen_inputs(
    qwen_model: Any,
    qwen_inputs: dict[str, torch.Tensor],
    prompt_length: int,
    action_token_ids: torch.Tensor,
    completion_ids: torch.Tensor | None = None,
    teacher_model: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any] | None]:
    """Compute action log-probs using the exact Qwen inputs from rollout.

    Uses the persisted ``qwen_inputs`` (input_ids, pixel_values,
    image_grid_thw, etc.) and ``prompt_length`` so the trainer scores
    the same token alignment the rollout used — no rebuilding.

    When ``teacher_model`` is provided, also computes per-sample KL
    divergence between the current and reference policy over the action
    tokens.

    Returns ``(logprob, kl_div, aux)`` where ``aux`` is a dict with
    ``completion_logits``, ``action_mask``, and ``prompt_length`` that
    can be used to compute an SFT loss from the same forward pass.
    """
    prompt_ids = qwen_inputs["input_ids"]  # [1, prompt_len]

    if completion_ids is not None:
        comp_ids = completion_ids.to(prompt_ids.dtype).to(prompt_ids.device)
        if comp_ids.dim() == 1:
            comp_ids = comp_ids.unsqueeze(0)
        # Fix 13: EOS append removed — it causes a shape mismatch in
        # Qwen2.5-VL get_rope_index because vision inputs (pixel_values,
        # image_grid_thw) are tied to the original input_ids length.
        # Appending EOS makes input_ids 1 token longer than the vision
        # inputs expect, crashing with IndexError.
        prompt_completion_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    else:
        act_ids = action_token_ids.to(prompt_ids.dtype).to(prompt_ids.device)
        if act_ids.dim() == 1:
            act_ids = act_ids.unsqueeze(0)
        # Fix 13: EOS append removed (same rationale as above).
        prompt_completion_ids = torch.cat([prompt_ids, act_ids], dim=1)

    prompt_completion_ids = sanitize_completion_vision_tokens(
        prompt_completion_ids, prompt_length, qwen_model.config,
    )

    forward_kwargs = {
        k: v for k, v in qwen_inputs.items()
        if k not in ("input_ids", "attention_mask", "prompt_length")
    }
    # Fix 16: Pad mm_token_type_ids and input_token_type to match prompt_completion_ids length.
    # These tensors have length = prompt_length (vision token types), but
    # prompt_completion_ids has length = prompt_length + completion_length.
    # Pad with zeros (text/non-vision tokens) for the completion part.
    _target_len = prompt_completion_ids.shape[-1]
    for _tt_key in ("mm_token_type_ids", "input_token_type"):
        if _tt_key in forward_kwargs:
            _tt = forward_kwargs[_tt_key]
            if hasattr(_tt, "shape") and len(_tt.shape) >= 1:
                _curr_len = _tt.shape[-1]
                if _curr_len < _target_len:
                    _pad_len = _target_len - _curr_len
                    _pad = torch.zeros(
                        *_tt.shape[:-1], _pad_len,
                        dtype=_tt.dtype, device=_tt.device,
                    )
                    forward_kwargs[_tt_key] = torch.cat([_tt, _pad], dim=-1)
    outputs = qwen_model(
        input_ids=prompt_completion_ids,
        attention_mask=torch.ones_like(prompt_completion_ids),
        **forward_kwargs,
    )
    logits = outputs.logits  # [1, L, V]

    logits = logits[:, :-1, :]  # [1, L-1, V]
    target_ids = prompt_completion_ids[:, 1:].to(logits.device)  # [1, L-1]

    # Memory-efficient logprob: avoid materializing full [1, L-1, V] float32.
    target_logits = logits.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    max_logits = logits.max(dim=-1).values
    shifted = logits - max_logits.unsqueeze(-1)
    exp_sum = shifted.exp().sum(dim=-1)
    lse = max_logits.float() + torch.log(exp_sum.float())
    per_token_logps = target_logits.float() - lse
    del shifted, exp_sum, max_logits, target_logits, lse

    completion_part_ids = target_ids[:, prompt_length - 1:]
    completion_logps = per_token_logps[:, prompt_length - 1:]

    action_ids = _get_action_token_ids(completion_part_ids.device)
    action_mask = action_token_mask(completion_part_ids, action_ids)
    if action_mask.any():
        logprob = completion_logps[action_mask].sum()
    else:
        # No action tokens — keep grad_fn alive to avoid backward crash.
        logprob = completion_logps.mean() * 0.0

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
            ref_log_probs = torch.log_softmax(ref_logits.float(), dim=-1)
            ref_per_token_logps = ref_log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            ref_completion_logps = ref_per_token_logps[:, prompt_length - 1:]

            if action_mask.any():
                cur_lp = completion_logps[action_mask]
                ref_lp = ref_completion_logps[action_mask]
                kl_div = (cur_lp.exp() * (cur_lp - ref_lp)).sum()
            else:
                kl_div = completion_logps.new_zeros(())
            kl_div = kl_div.squeeze() if kl_div.dim() > 0 else kl_div

    completion_logits_out = logits[:, prompt_length - 1:, :]
    aux = {
        "completion_logits": completion_logits_out,
        "action_mask": action_mask,
        "prompt_length": prompt_length,
    }
    return logprob_out, kl_div, aux


def autovla_training_forward(
    trainer: Any,
    model_inputs: dict[str, Any],
    teacher_model: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, list[dict[str, Any] | None]]:
    """AutoVLA training forward: compute action log_probs using persisted Qwen inputs.

    When the rollout persisted its Qwen processor inputs (``qwen_inputs``),
    those are used directly — guaranteeing token-level alignment with the
    rollout-side logprob.  Otherwise, falls back to rebuilding from raw
    camera frames (legacy path, less accurate).

    When ``teacher_model`` is provided, also computes per-sample KL
    divergence between the current and reference policy over the action
    tokens.

    Returns ``(log_probs, kl_divs, aux_list)`` where ``log_probs`` is
    ``[B]``, ``kl_divs`` is ``[B]`` or ``None``, and ``aux_list`` is a
    per-sample list of dicts with completion logits and action mask for
    SFT loss computation.
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
        all_aux: list[dict[str, Any]] = []
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
                logprob, kl_div, aux = _compute_action_logprobs_from_qwen_inputs(
                    qwen_model, qwen_inputs, prompt_length,
                    act_ids[b] if act_ids.dim() > 1 else act_ids[0],
                    comp_ids[b] if comp_ids is not None and comp_ids.dim() > 1 else comp_ids,
                    teacher_model=ref_qwen,
                )
                all_aux.append(aux)
            else:
                logprob = torch.tensor(0.0, device=device)
                kl_div = None
                all_aux.append(None)
            all_logprobs.append(logprob)
            if kl_div is not None:
                all_kl_divs.append(kl_div)
        log_probs = torch.stack(all_logprobs)
        kl_divs = torch.stack(all_kl_divs) if all_kl_divs else None
        return log_probs, kl_divs, all_aux

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
    return log_probs, None, []


def _compute_expert_sft_loss(
    trainer: Any,
    model_inputs: dict[str, Any],
    log_probs: torch.Tensor,
) -> torch.Tensor | None:
    """Compute SFT cross-entropy loss on expert action tokens (DAgger).

    When ``expert_action_tokens`` is present in ``model_inputs``, this
    computes the cross-entropy loss between the policy's action-token
    logits and the expert's action-token labels, averaged over non-padding
    rows.  Returns ``None`` when no expert tokens are available.
    """
    expert_tokens = model_inputs.get("expert_action_tokens")
    if expert_tokens is None:
        return None

    qwen_model = _find_qwen_model(trainer.model)
    device = next(qwen_model.parameters()).device
    expert_tokens = expert_tokens.to(device)
    if expert_tokens.dim() == 1:
        expert_tokens = expert_tokens.unsqueeze(0)

    qwen_inputs_list = model_inputs.get("qwen_inputs")
    if qwen_inputs_list is None:
        logger.warning("expert_action_tokens present but no qwen_inputs — skipping SFT loss")
        return None

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
    elif isinstance(qwen_inputs_list, list):
        qwen_inputs_per_sample = qwen_inputs_list
    else:
        qwen_inputs_per_sample = [qwen_inputs_list]

    batch_size = len(qwen_inputs_per_sample)
    is_padding = model_inputs.get("is_padding")
    if is_padding is not None:
        is_padding = is_padding.to(device)
    else:
        is_padding = torch.zeros(batch_size, dtype=torch.bool, device=device)

    action_start = _trainer_config.get("action_start_id", 151665)
    action_count = _trainer_config.get("action_token_count", 2048)

    total_loss = torch.tensor(0.0, device=device)
    total_valid = 0

    for b in range(batch_size):
        if bool(is_padding[b].item()):
            continue
        if b >= expert_tokens.shape[0]:
            break
        qi = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in qwen_inputs_per_sample[b].items()}
        prompt_length = int(qi.pop("prompt_length"))

        expert_ids = expert_tokens[b].to(device)
        prompt_ids = qi["input_ids"]
        prompt_expert_ids = torch.cat([prompt_ids, expert_ids.unsqueeze(0)], dim=1)

        forward_kwargs = {
            k: v for k, v in qi.items()
            if k not in ("input_ids", "attention_mask", "prompt_length")
        }
        outputs = qwen_model(
            input_ids=prompt_expert_ids,
            attention_mask=torch.ones_like(prompt_expert_ids),
            use_cache=False,
            **forward_kwargs,
        )
        logits = outputs.logits[:, :-1, :]  # [1, L-1, V]
        target_ids = prompt_expert_ids[:, 1:].to(logits.device)

        completion_part_ids = target_ids[:, prompt_length - 1:]
        completion_logits = logits[:, prompt_length - 1:, :]

        action_ids = _get_action_token_ids(completion_part_ids.device)
        action_mask = action_token_mask(completion_part_ids, action_ids)

        if action_mask.any():
            masked_logits = completion_logits[action_mask.unsqueeze(-1).expand_as(completion_logits)]
            masked_logits = masked_logits.view(-1, completion_logits.shape[-1])
            masked_targets = completion_part_ids[action_mask]
            loss = torch.nn.functional.cross_entropy(
                masked_logits.float(), masked_targets,
            )
            total_loss = total_loss + loss
            total_valid += 1

        del outputs, logits, target_ids, prompt_expert_ids

    if total_valid == 0:
        return None

    avg_loss = total_loss / total_valid
    logger.info("DAgger SFT loss: %.4f (valid_samples=%d)", avg_loss.item(), total_valid)
    return avg_loss


def _compute_expert_sft_loss_from_aux(
    model_inputs: dict[str, Any],
    aux_list: list[dict[str, Any] | None],
    expert_tokens: torch.Tensor | None,
) -> torch.Tensor | None:
    """Compute SFT cross-entropy loss from the main forward pass's logits.

    Uses the ``completion_logits`` and ``action_mask`` from the GRPO
    forward pass (passed via ``aux_list``) to compute the cross-entropy
    loss with the expert's action tokens as targets.  This avoids a
    second forward pass through the FSDP-wrapped model, which would
    cause DTensor mixing errors during ``backward()``.

    BUG-1 fix (2026-09-10): Previously took ``completion_logits[0, :n_loss]``
    — the first N positions of the completion — as the logits to fit expert
    action tokens.  But the completion starts with a text prefix
    (``<answer>\\nThe final output action is: …``) before the action tokens
    appear, so the CE was aligning "prefix logits" with "expert action tokens"
    — actively teaching the model to emit action tokens at the wrong positions.
    Now uses ``action_mask`` to select only the action-token positions.
    """
    if expert_tokens is None:
        logger.debug(
            "DAgger SFT loss skipped: expert_action_tokens is None "
            "(expert query failed, fallback payload, or beta=0 sampling)"
        )
        return None
    if aux_list is None or all(a is None for a in aux_list):
        return None

    # BUG-2 fix: respect is_padding — but only skip rows that are padding AND
    # have no expert tokens.  Expert-takeover steps (BUG-3 fix) have their
    # old_logprob set to NaN, which causes _prepare_training_data to mark them
    # is_padding=True for GRPO exclusion.  However, these steps DO have
    # expert_action_tokens and SHOULD contribute to the DAgger SFT loss — that's
    # the whole point of DAgger: learn from states where the expert intervened.
    is_padding = model_inputs.get("is_padding")
    batch_size = len(aux_list)

    total_loss = None
    total_valid = 0

    for b, aux in enumerate(aux_list):
        if aux is None:
            continue
        if b >= expert_tokens.shape[0]:
            break

        # Skip padding rows ONLY if they have no expert tokens (true padding =
        # cloned template with no expert labels).  Expert-takeover rows are
        # is_padding=True (for GRPO) but have expert tokens (for DAgger SFT).
        if is_padding is not None:
            pad_b = is_padding[b] if b < len(is_padding) else False
            if isinstance(pad_b, torch.Tensor):
                pad_b = pad_b.item()
            if bool(pad_b):
                # Check if this row has valid expert tokens — if so, it's an
                # expert-takeover step that should still get SFT loss.
                expert_b = expert_tokens[b]
                if expert_b is not None and expert_b.numel() > 0:
                    has_real_tokens = bool((expert_b != 0).any().item())
                    if not has_real_tokens:
                        continue  # true padding, skip
                    # expert-takeover step: fall through to compute SFT loss
                else:
                    continue  # no expert tokens, skip

        completion_logits = aux["completion_logits"]   # [1, T, V]
        action_mask = aux["action_mask"]               # [1, T] bool

        expert_ids = expert_tokens[b].to(completion_logits.device)
        if expert_ids.dim() == 0:
            expert_ids = expert_ids.unsqueeze(0)

        # BUG-1 fix: use action_mask to find action-token positions, not the
        # first N positions.  action_mask marks True wherever the completion
        # token id falls in the action-token codebook range.
        masked_positions = action_mask[0]  # [T] bool
        n_action = int(masked_positions.sum().item())
        n_expert = expert_ids.shape[0]

        if n_action == 0:
            logger.warning(
                "DAgger SFT loss: sample %d has 0 action tokens in completion, skipping", b
            )
            continue

        n_loss = min(n_expert, n_action)

        # Select logits at action-token positions only
        loss_logits = completion_logits[0][masked_positions][:n_loss]  # [n_loss, V]
        loss_targets = expert_ids[:n_loss]                              # [n_loss]

        loss = torch.nn.functional.cross_entropy(
            loss_logits.float(), loss_targets,
        )

        # EOS loss: teach the model to predict EOS after the action tokens.
        # The forward pass (_compute_action_logprobs_from_qwen_inputs)
        # appends EOS to the completion, so the last logit position predicts
        # EOS.  Without this term the model never learns when to stop
        # generating and degenerates into emitting max_new_tokens every time.
        # Controlled by ALPAGYM_DAGGER_EOS_BETA (default 1.0).
        eos_weight = float(os.environ.get("ALPAGYM_DAGGER_EOS_BETA", "1.0"))
        if eos_weight > 0.0 and not bool(masked_positions[-1].item()):
            eos_id = _get_processor().tokenizer.eos_token_id
            eos_logit = completion_logits[0, -1, :].unsqueeze(0)  # [1, V]
            eos_target = torch.tensor([eos_id], device=eos_logit.device)
            eos_loss = torch.nn.functional.cross_entropy(
                eos_logit.float(), eos_target,
            )
            loss = loss + eos_weight * eos_loss
            logger.debug(
                "DAgger EOS loss: %.4f (weight=%.1f, sample=%d, eos_id=%d)",
                eos_loss.item(), eos_weight, b, eos_id,
            )

        total_loss = loss if total_loss is None else total_loss + loss
        total_valid += 1

    if total_valid == 0:
        logger.warning(
            "DAgger SFT loss: 0 valid samples after filtering (batch_size=%d)", batch_size
        )
        return None

    return total_loss / total_valid


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
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # AutoVLA path: camera_frames or qwen_inputs present
        if isinstance(model_inputs, dict) and (
            "camera_frames" in model_inputs or "qwen_inputs" in model_inputs
        ):
            log_probs, kl_divs, aux_list = autovla_training_forward(
                self, model_inputs,
                teacher_model=self._reference_model,
            )
            torch.cuda.empty_cache()
            eat = model_inputs.get("expert_action_tokens")
            sft_loss = _compute_expert_sft_loss_from_aux(
                model_inputs, aux_list, eat,
            )
            if sft_loss is not None:
                logger.info("DAgger SFT loss: %.4f", sft_loss.item())
            return log_probs, kl_divs, sft_loss
        # Original path for all other policies
        forward_kwargs = to_device_recursive(model_inputs, self.device)
        if self._reference_model is not None:
            forward_kwargs["teacher_model"] = self._reference_model
        result = self.model(**forward_kwargs)
        return result["log_probs"], result.get("kl_div"), None

    patched_forward_with_reference._autovla_patch = True  # type: ignore[attr-defined]
    AlpagymGRPOTrainer._forward_with_reference = patched_forward_with_reference
    logger.info("Patched AlpagymGRPOTrainer._forward_with_reference for AutoVLA + DAgger")
