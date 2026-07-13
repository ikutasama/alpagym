"""AutoVLA inference model adapter for AlpaGym.

Bridges AlpaGym's BatchedModelInput/BatchedModelOutput to AutoVLA's
Qwen2.5-VL-3B + discrete action token pipeline.

Key differences from Alpamayo R1:
- AutoVLA uses discrete action tokens (LLM softmax), not flow-matching
- logprob = sum of per-token logprobs (simple case, no CFM SDE)
- AutoVLA takes 3 cameras x 4 frames = 12 images as "video" inputs
- Trajectory output is (x, y, heading) in ego frame, 10 poses, 0.5s interval
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from alpagym_host.config import SamplingParamsConfig
from alpagym_runtime.inference.types import (
    BatchedModelInput,
    BatchedModelOutput,
    ModelInput,
    ModelOutput,
)
from alpagym_runtime.replay import ActionSelection, PolicyReplayData, require_payload_keys

logger = logging.getLogger(__name__)


def heading_to_rotation_matrix(heading: torch.Tensor) -> torch.Tensor:
    """Convert yaw heading (radians) to SO(3) rotation matrices.

    AlPaGym selectors expect ``pred_rot`` shaped ``[..., 3, 3]``.  AutoVLA
    predicts planar yaw only, so roll/pitch are identity.
    """
    cos = torch.cos(heading)
    sin = torch.sin(heading)
    rot = torch.zeros(*heading.shape, 3, 3, dtype=torch.float32, device=heading.device)
    rot[..., 0, 0] = cos
    rot[..., 0, 1] = -sin
    rot[..., 1, 0] = sin
    rot[..., 1, 1] = cos
    rot[..., 2, 2] = 1.0
    return rot


class AutoVLAInferenceModel:
    """Adapter between AlpaGym typed I/O and an AutoVLA model."""

    def __init__(
        self,
        vlm: Any,
        processor: Any,
        codebook: dict,
        action_start_id: int,
        num_poses: int = 10,
        interval_length: float = 0.5,
        device: str = "cuda",
        use_cot: bool = False,
    ) -> None:
        self._vlm = vlm
        self._processor = processor
        self._action_start_id = action_start_id
        self._num_poses = num_poses
        self._interval_length = interval_length
        self._device = device
        self._use_cot = use_cot

        # Extract codebook for action token decoding
        # codebook format: {"token_all": {"veh": array(n_bins, 6, 4, 2)}}
        if isinstance(codebook, dict) and "token_all" in codebook:
            cb = codebook["token_all"]["veh"]
            self._code_book = torch.tensor(cb)  # (n_bins, 6, 4, 2)
        else:
            self._code_book = torch.tensor(codebook)

        # Add action tokens to tokenizer if not already present
        n_bins = self._code_book.shape[0]
        existing_tokens = set(self._processor.tokenizer.get_vocab().keys())
        new_tokens = [f"<action_{i}>" for i in range(n_bins)
                      if f"<action_{i}>" not in existing_tokens]
        if new_tokens:
            self._processor.tokenizer.add_tokens(new_tokens, special_tokens=False)
            self._vlm.resize_token_embeddings(len(self._processor.tokenizer))

    def get_model(self) -> torch.nn.Module:
        return self._vlm

    def set_model(self, model: torch.nn.Module) -> None:
        """Replace the inference model (for Cosmos weight sync)."""
        if hasattr(model, "_get_fsdp_state"):
            from torch.distributed.fsdp import FSDPModule
            model._get_fsdp_state()._lazy_init()
            for submodule in model.modules():
                if isinstance(submodule, FSDPModule):
                    submodule.unshard()
        self._vlm = model

    def sample_trajectories_from_data(
        self,
        model_input: BatchedModelInput,
        sampling: SamplingParamsConfig,
        return_trace_for_rl: bool = False,
    ) -> BatchedModelOutput:
        """Run AutoVLA inference and return trajectory + logprob."""
        batch_size = model_input.camera_frames.shape[0]
        all_pred_xyz = []
        all_pred_rot = []
        all_logprobs = []
        all_action_token_ids = []

        for batch_idx in range(batch_size):
            pred_xyz, pred_rot, logprob, action_token_ids = self._infer_single(
                model_input, batch_idx, sampling, return_trace_for_rl
            )
            all_pred_xyz.append(pred_xyz)
            all_pred_rot.append(pred_rot)
            if logprob is not None:
                all_logprobs.append(logprob)
            if action_token_ids is not None:
                all_action_token_ids.append(action_token_ids)

        pred_xyz = torch.stack(all_pred_xyz, dim=0)  # [B, S, K, T, 3]
        pred_rot = torch.stack(all_pred_rot, dim=0)  # [B, S, K, T, 3, 3]
        logprob = torch.stack(all_logprobs, dim=0) if all_logprobs else None  # [B, S, K]
        extra: dict[str, Any] = {}
        if all_action_token_ids:
            extra["action_token_ids"] = torch.stack(all_action_token_ids, dim=0)  # [B, S, K, T]

        return BatchedModelOutput(
            pred_xyz=pred_xyz,
            pred_rot=pred_rot,
            logprob=logprob,
            extra=extra,
        )

    def _infer_single(
        self,
        model_input: BatchedModelInput,
        batch_idx: int,
        sampling: SamplingParamsConfig,
        return_trace_for_rl: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Process one batch row: build prompt, generate, decode trajectory.

        Returns AlPaGym candidate axes for one batch row:
          pred_xyz ``[num_traj_sets, num_traj_samples, T, 3]``
          pred_rot ``[num_traj_sets, num_traj_samples, T, 3, 3]``
          logprob ``[num_traj_sets, num_traj_samples]`` when requested
        """
        if sampling.num_traj_sets != 1 or sampling.num_traj_samples != 1:
            raise ValueError(
                "AutoVLA adapter currently supports exactly one candidate per tick; "
                f"got num_traj_sets={sampling.num_traj_sets}, "
                f"num_traj_samples={sampling.num_traj_samples}."
            )

        # 1. Convert camera frames to PIL images and build Qwen processor inputs
        model_inputs = self._build_qwen_inputs(model_input, batch_idx)

        # 2. Generate action tokens
        gen_kwargs = {
            "do_sample": True,
            "max_length": 2048,
            "temperature": sampling.temperature if sampling.temperature else 0.01,
            "top_k": sampling.top_k if sampling.top_k else 0,
            "top_p": sampling.top_p if sampling.top_p else 1.0,
        }

        with torch.no_grad():
            prompt_completion_ids = self._vlm.generate(
                **model_inputs,
                **gen_kwargs,
            )

        prompt_length = model_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # 3. Extract action tokens
        action_tokens = completion_ids[0][completion_ids[0] >= self._action_start_id]
        if len(action_tokens) > self._num_poses:
            action_tokens = action_tokens[:self._num_poses]
        elif len(action_tokens) < self._num_poses:
            pad = torch.full(
                (self._num_poses - len(action_tokens),),
                self._action_start_id,
                dtype=action_tokens.dtype,
                device=action_tokens.device,
            )
            action_tokens = torch.cat([action_tokens, pad])

        # 4. Decode tokens to trajectory (x, y, heading)
        trajectory = self._decode_tokens_to_trajectory(action_tokens.cpu())
        # trajectory shape: (T, 3) — (x, y, heading) in ego frame
        # Skip first point (origin), take remaining
        traj = trajectory[1:]  # (num_poses, 3)

        # 5. Convert to AlpaGym format
        # pred_xyz: [T, 3] — (x, y, 0)
        pred_xyz = torch.zeros(traj.shape[0], 3, dtype=torch.float32)
        pred_xyz[:, 0] = traj[:, 0]  # x
        pred_xyz[:, 1] = traj[:, 1]  # y
        pred_xyz[:, 2] = 0.0         # z = 0 (2D planning)

        # pred_rot: [T, 3, 3] — yaw-only rotation matrix
        pred_rot = heading_to_rotation_matrix(traj[:, 2])

        # Add AlPaGym candidate axes: [S=1, K=1, T, ...]
        pred_xyz = pred_xyz.unsqueeze(0).unsqueeze(0)
        pred_rot = pred_rot.unsqueeze(0).unsqueeze(0)

        # 6. Compute logprob if needed for RL
        logprob = None
        action_token_ids_out = None
        if return_trace_for_rl:
            logprob = self._compute_logprob(
                model_inputs, prompt_completion_ids, prompt_length
            ).reshape(1, 1)
            action_token_ids_out = action_tokens.reshape(1, 1, -1).to(torch.int64).cpu()

        return pred_xyz, pred_rot, logprob, action_token_ids_out

    def _build_qwen_inputs(
        self,
        model_input: BatchedModelInput,
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Build Qwen2.5-VL processor inputs from AlPaGym camera frames.

        AlPaGym provides camera_frames [B, num_cams, H, W, C] uint8.
        AutoVLA expects 3 cameras x 4 frames as "video" inputs.
        We convert uint8 tensors to PIL images and pass to the Qwen processor.
        """
        camera_frames = model_input.camera_frames[batch_idx]  # [num_cams, H, W, C]
        camera_indices = model_input.camera_indices[batch_idx]  # [num_cams]
        ego_history = model_input.ego_history_xyz[batch_idx]  # [T_hist, 3]
        route_xy = model_input.route_xy[batch_idx]  # [N, 2]

        num_cams = camera_frames.shape[0]
        # Convert frames to PIL images grouped by camera
        # Assume first num_cams/4 cameras are different views, each with 4 frames
        # or all frames belong to the same camera sequence
        frames_per_cam = max(1, num_cams // 3) if num_cams >= 3 else num_cams

        pil_images = []
        for i in range(num_cams):
            frame = camera_frames[i].cpu().numpy()
            if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
                pil_images.append(Image.fromarray(frame.transpose(1, 2, 0)))
            elif frame.ndim <= 1 or (frame.ndim >= 2 and min(frame.shape[:2]) <= 1):
                pil_images.append(Image.open(io.BytesIO(frame.tobytes())))
            else:
                pil_images.append(Image.fromarray(frame))

        # Build velocity/acceleration from ego history
        if ego_history.shape[0] >= 2:
            diff = ego_history[1:] - ego_history[:-1]
            velocity = float(torch.norm(diff[-1][:2]).item())
            acceleration = float(torch.norm(diff[-1][:2] - diff[-2][:2]).item()
                                 if diff.shape[0] >= 2 else 0.0)
        else:
            velocity = 0.0
            acceleration = 0.0

        # Build driving instruction from route
        if route_xy.shape[0] > 0:
            first_wp = route_xy[0]
            if abs(first_wp[0]) > abs(first_wp[1]):
                instruction = "turn left" if first_wp[0] < 0 else "turn right"
            else:
                instruction = "move forward"
        else:
            instruction = "move forward"

        # Build chat messages
        user_content = self._build_user_content(pil_images, velocity, acceleration, instruction)

        if self._use_cot:
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
            {"role": "user", "content": user_content},
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )

        processor_kwargs: dict[str, Any]
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
        except Exception as exc:  # pragma: no cover - fallback for lightweight smoke envs
            logger.warning("Falling back to image-only Qwen processor path: %s", exc)
            processor_kwargs = {
                "text": [text],
                "images": pil_images,
                "padding": True,
                "return_tensors": "pt",
            }

        inputs = self._processor(**processor_kwargs)

        # Move to device
        return {k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()}

    def _build_user_content(
        self,
        pil_images: list,
        velocity: float,
        acceleration: float,
        instruction: str,
    ) -> list:
        """Build user content list for the chat template.

        Groups images into video segments (3 cameras x 4 frames).
        Falls back to single-camera if fewer images available.
        """
        num_images = len(pil_images)
        min_pixels = 28 * 28 * 128
        max_pixels = 28 * 28 * 128

        content = [
            {"type": "text", "text": "The autonomous vehicle is equipped with cameras enabling perception of the surrounding environment."},
        ]

        # Group images: try 3 cameras x 4 frames, fallback to single sequence
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
                    "video": cam_frames,  # PIL images directly
                })
        elif num_images >= 4:
            # Single camera with multiple frames
            content.append({"type": "text", "text": f"Front view video, {num_images} frames at 2 Hz."})
            content.append({
                "type": "video",
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "video": pil_images,
            })
        else:
            # Single frames
            for i, img in enumerate(pil_images):
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

        return content

    def _decode_tokens_to_trajectory(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Decode action token IDs to trajectory via codebook rollout.

        Replicates AutoVLA's ActionTokenizer.decode_token_ids_to_trajectory +
        rollout logic.
        """
        # Decode token IDs to codebook indices
        action_indices = []
        for tid in token_ids:
            if tid < self._action_start_id:
                action_indices.append(0)
            else:
                # Parse action index from token ID
                # Token ID = action_start_id + index
                action_indices.append(int(tid.item() - self._action_start_id))

        action_indices = torch.tensor(action_indices)
        action_tokens = self._code_book[action_indices]  # (T, 6, 4, 2)

        # Rollout trajectory
        pos_a = torch.tensor([[[0.0, 0.0]]])  # [1, 1, 2]
        head_a = torch.tensor([[0.0]])  # [1, 1]

        for t in range(action_tokens.shape[0]):
            next_token_traj = action_tokens[None, t]  # [1, 6, 4, 2]
            # Flatten for transform_to_global
            pos_local = next_token_traj.flatten(1, 2)  # [1, 6*4, 2]
            pos_now = pos_a[:, t]  # [1, 2]
            head_now = head_a[:, t]  # [1]

            # Transform to global
            cos, sin = head_now.cos(), head_now.sin()
            rot_mat = torch.zeros((1, 2, 2))
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            pos_global = torch.bmm(pos_local, rot_mat) + pos_now.unsqueeze(1)
            pos_global = pos_global.view(*next_token_traj.shape)

            # Next state
            pos_a_next = pos_global[:, -1].mean(dim=1)
            diff_xy = pos_global[:, -1, 0] - pos_global[:, -1, 3]
            head_a_next = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])

            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)

        trajectory = torch.cat([pos_a, head_a.unsqueeze(-1)], dim=-1)  # [1, T+1, 3]
        return trajectory[0]  # [T+1, 3]

    def _compute_logprob(
        self,
        model_inputs: dict[str, torch.Tensor],
        prompt_completion_ids: torch.Tensor,
        prompt_length: int,
    ) -> torch.Tensor:
        """Compute sum of per-token logprobs for the generated action tokens.

        Simple case: AutoVLA uses discrete LLM softmax, so logprob =
        sum of per-token log-probs from the model's logits.
        """
        with torch.no_grad():
            # Forward pass to get logits.  Reuse all processor-produced vision
            # tensors (image/video pixel values + grid metadata) and replace only
            # the token sequence with the full prompt+completion IDs.
            forward_kwargs = {
                key: value
                for key, value in model_inputs.items()
                if key not in ("input_ids", "attention_mask")
            }
            outputs = self._vlm(
                input_ids=prompt_completion_ids,
                attention_mask=torch.ones_like(prompt_completion_ids),
                **forward_kwargs,
            )
            logits = outputs.logits  # (1, L, V)

        # Shift for next-token prediction
        logits = logits[:, :-1, :]  # (1, L-1, V)
        target_ids = prompt_completion_ids[:, 1:]  # (1, L-1)

        # Compute log probs
        log_probs = torch.log_softmax(logits, dim=-1)  # (1, L-1, V)
        per_token_logps = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # (1, L-1)

        # Sum only action token logprobs (completion part)
        completion_logps = per_token_logps[:, prompt_length - 1:]
        logprob = completion_logps.sum(dim=-1)  # (1,)

        return logprob.squeeze(0)  # scalar

    def build_policy_replay_data(
        self,
        model_input: ModelInput,
        model_output: ModelOutput,
        action_selection: ActionSelection,
    ) -> PolicyReplayData:
        """Pack selected-only AutoVLA replay data.

        Persist the raw AlPaGym input plus the selected discrete action-token
        sequence so trainer-side scoring can reconstruct the exact rollout action.
        """
        if model_output.logprob is None:
            raise ValueError("AutoVLA replay requires rollout logprob")
        if "action_token_ids" not in model_output.extra:
            raise ValueError("AutoVLA replay requires action_token_ids in model_output.extra")
        old_logprob = model_output.logprob[action_selection.set_ix, action_selection.sample_ix]
        selected_action_token_ids = model_output.extra["action_token_ids"][
            action_selection.set_ix, action_selection.sample_ix
        ]
        return PolicyReplayData(
            replay_schema_version=1,
            payload_schema="autovla.action_tokens.v1",
            payload_schema_version=1,
            model_family="autovla",
            action_selection=action_selection,
            old_logprob=torch.as_tensor(old_logprob, dtype=torch.float32).reshape(()),
            payload={
                "model_input": asdict(model_input),
                "action_token_ids": torch.as_tensor(selected_action_token_ids, dtype=torch.int64),
            },
        )

    @staticmethod
    def build_trainer_model_inputs(
        replay_data: PolicyReplayData,
        action_start_id: int = 151665,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Build trainer-side model inputs from replay payload.

        This preserves the raw AlPaGym model input plus the selected action-token
        IDs.  The policy-side model wrapper can then rebuild Qwen inputs and
        score exactly the action sequence sampled during rollout.
        """
        del action_start_id
        if replay_data.payload_schema != "autovla.action_tokens.v1":
            raise ValueError(
                f"{replay_data.model_family} replay payload_schema "
                f"{replay_data.payload_schema!r} != 'autovla.action_tokens.v1'"
            )
        if replay_data.payload_schema_version != 1:
            raise ValueError(
                f"{replay_data.model_family} replay payload_schema_version "
                f"{replay_data.payload_schema_version} != 1"
            )
        payload = replay_data.payload
        require_payload_keys(
            replay_data.model_family,
            payload,
            ("model_input", "action_token_ids"),
        )
        model_input_payload = payload["model_input"]
        if not isinstance(model_input_payload, Mapping):
            raise TypeError(
                "autovla replay model_input must be a mapping, "
                f"got {type(model_input_payload).__name__}"
            )
        model_input = ModelInput.from_payload(model_input_payload)
        model_inputs = {
            "camera_frames": model_input.camera_frames,
            "ego_history_xyz": model_input.ego_history_xyz,
            "ego_history_rot": model_input.ego_history_rot,
            "camera_indices": model_input.camera_indices,
            "relative_timestamps": model_input.relative_timestamps,
            "route_xy": model_input.route_xy,
            "action_token_ids": torch.as_tensor(payload["action_token_ids"], dtype=torch.int64),
        }
        if replay_data.old_logprob is None:
            raise ValueError("autovla replay requires old_logprob")
        old_logprob = torch.as_tensor(replay_data.old_logprob, dtype=torch.float32).reshape(())
        return model_inputs, old_logprob
