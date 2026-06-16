# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alpamayo policy glue between buffers, inference, and trajectory selection."""

import logging
from bisect import bisect_left

import torch
from alpagym_host.config import AlpamayoPolicyConfig

from alpagym_runtime.alpasim.cameras import CAMERA_NAMES_TO_INDICES
from alpagym_runtime.inference.inference_engine import InferenceEngine
from alpagym_runtime.inference.types import NUM_ROUTE_WAYPOINTS, ModelInput, ModelOutput
from alpagym_runtime.replay import ActionSelection, PolicyReplayData
from alpagym_runtime.types import ChosenTrajectory, EgoPose, PolicyInput, PolicyOutput

from .buffers import ObservationBuffers
from .geometry_utils import (
    pred_rot_to_quat,
    quat_tensor_to_rotation_matrix,
    quaternion_to_rotation_matrix,
)
from .output_trajectory import build_policy_output_trajectory
from .selectors import build_selector

logger = logging.getLogger(__name__)


class AlpamayoPolicy:
    """Per-session Alpamayo policy implementing the `Policy` Protocol."""

    def __init__(
        self,
        inference_engine: InferenceEngine,
        session_uuid: str,
        config: AlpamayoPolicyConfig,
        device: torch.device,
        dtype: torch.dtype,
        seed: int | None = None,
    ) -> None:
        """Build one per-session policy bound to a shared inference engine."""
        self._session_uuid = session_uuid
        self._inference_engine = inference_engine
        self._config = config
        self._device = device
        self._dtype = dtype
        self._session_seed = seed
        self._inference_step_counter = 0
        self._valid_output_count = 0
        self._buffers = ObservationBuffers(
            frame_buffer_capacity=config.model.num_context_frames,
            ego_history_capacity=config.model.num_historical_waypoints + 1,
            use_cameras=tuple(config.model.use_cameras),
            frame_size=(config.model.input_size[0], config.model.input_size[1]),
            frame_device=self._device,
        )
        self._selector = build_selector(config.trajectory_selector)

    def step(self, policy_input: PolicyInput) -> PolicyOutput:
        """Run one drive tick: preprocess, infer, postprocess."""
        seed: torch.Tensor | None = None
        if self._config.inference.sampling.force_determinism:
            session_seed = int(self._session_seed)
            seed = torch.tensor(
                session_seed + self._inference_step_counter,
                dtype=torch.int64,
                device=self._device,
            )
        model_input = self._preprocess(policy_input, seed=seed)
        if seed is not None:
            self._inference_step_counter += 1
        future = self._inference_engine.infer(model_input)
        model_output = future.result()
        return self._postprocess(
            model_output=model_output,
            model_input=model_input,
            policy_input=policy_input,
        )

    def close(self) -> None:
        """Release per-session resources."""

    def _preprocess(
        self,
        policy_input: PolicyInput,
        seed: torch.Tensor | None = None,
    ) -> ModelInput:
        """Ingest the tick and pack a typed :class:`ModelInput`."""
        self._buffers.ingest(policy_input)
        camera_frames, camera_indices, relative_timestamps = self._extract_camera_context()
        ego_history_xyz, ego_history_rot = self._extract_historical_motion()
        route_xy = self._convert_route()
        return ModelInput(
            ego_history_xyz=ego_history_xyz,
            ego_history_rot=ego_history_rot,
            camera_frames=camera_frames,
            camera_indices=camera_indices,
            relative_timestamps=relative_timestamps,
            route_xy=route_xy,
            seed=seed,
        )

    def _extract_camera_context(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw uint8 camera frames, camera indices, and t0-relative timestamps.

        Per-model normalization (e.g. uint8 → float32 [-1, 1]) lives in the
        inference adapter so this method stays model-agnostic.
        """
        if not self._config.model.use_cameras:
            empty_frames = torch.zeros((0, 3, 1, 1), dtype=torch.uint8, device=self._device)
            empty_indices = torch.zeros((0,), dtype=torch.int64, device=self._device)
            empty_tstamps = torch.zeros((0,), dtype=torch.int64, device=self._device)
            return empty_frames, empty_indices, empty_tstamps

        cameras = self._buffers.cameras
        latest_per_camera: list[int] = []
        for cam_id in self._config.model.use_cameras:
            ring = cameras.get(cam_id)
            if ring is None or len(ring) == 0:
                raise ValueError(f"No frames buffered for camera {cam_id!r}")
            latest_per_camera.append(int(ring.tstamps_ordered[-1].item()))
        t0 = max(latest_per_camera)

        num_context_frames = self._config.model.num_context_frames
        frame_chunks: list[torch.Tensor] = []
        camera_index_chunks: list[torch.Tensor] = []
        relative_timestamp_chunks: list[torch.Tensor] = []
        for cam_id in self._config.model.use_cameras:
            ring = cameras[cam_id]
            sample_frames = ring.frames_ordered
            sample_tstamps = ring.tstamps_ordered
            available = sample_frames.shape[0]
            if available != num_context_frames:
                raise ValueError(
                    f"Camera {cam_id!r} has {available} frames buffered; "
                    f"expected {num_context_frames}."
                )
            if sample_frames.dtype != torch.uint8:
                raise ValueError(
                    f"Camera {cam_id!r} frame dtype is {sample_frames.dtype}, "
                    f"expected torch.uint8 (CHW)."
                )
            if sample_frames.ndim != 4:
                raise ValueError(
                    f"Camera {cam_id!r} frames ndim is {sample_frames.ndim}, "
                    f"expected 4 ([T, C, H, W])."
                )
            cam_idx = CAMERA_NAMES_TO_INDICES[cam_id]
            num_frames = sample_frames.shape[0]
            frame_chunks.append(sample_frames)
            camera_index_chunks.append(
                torch.full((num_frames,), cam_idx, dtype=torch.int64, device=self._device)
            )
            relative_timestamp_chunks.append(
                (sample_tstamps - t0).to(dtype=torch.int64, device=self._device)
            )

        camera_frames = torch.cat(frame_chunks, dim=0)
        camera_indices = torch.cat(camera_index_chunks, dim=0)
        relative_timestamps = torch.cat(relative_timestamp_chunks, dim=0)
        return camera_frames, camera_indices, relative_timestamps

    def _extract_historical_motion(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the rig-frame xyz and rotation history tensors.

        Raises if the buffer holds fewer than ``num_historical_waypoints``
        poses; alpasim warmup (``force_gt_duration_us``) is responsible for
        supplying a long enough history before the first inference tick.
        """
        num_historical_waypoints = self._config.model.num_historical_waypoints
        ego_history = self._buffers.ego_history
        if len(ego_history) < num_historical_waypoints:
            raise ValueError(
                f"AlpamayoPolicy has only {len(ego_history)} ego poses; "
                f"need at least {num_historical_waypoints}. Increase the "
                f"alpasim ``force_gt_duration_us`` warmup."
            )
        history_world_xyz = torch.zeros((num_historical_waypoints, 3), dtype=torch.float32)
        history_world_rot = (
            torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(num_historical_waypoints, 1, 1)
        )
        recent_poses = ego_history[-num_historical_waypoints:]
        for history_index, ego_pose in enumerate(recent_poses):
            history_world_xyz[history_index] = _vec3_to_tensor(ego_pose)
            history_world_rot[history_index] = quaternion_to_rotation_matrix(ego_pose.pose.quat)

        # Normalize world-frame history into the rig-t0 frame:
        # last row maps to the origin with identity rotation.
        latest_world_xyz = history_world_xyz[-1]
        latest_world_rot = history_world_rot[-1]
        history_rig_xyz = (
            latest_world_rot.T @ (history_world_xyz - latest_world_xyz).unsqueeze(-1)
        ).squeeze(-1)
        history_rig_rot = latest_world_rot.T @ history_world_rot
        # Keep ego history fp32; the action_space CPU cholesky has no bf16
        # kernel.
        return (
            history_rig_xyz.unsqueeze(0).to(device=self._device, dtype=torch.float32),
            history_rig_rot.unsqueeze(0).to(device=self._device, dtype=torch.float32),
        )

    def _convert_route(self) -> torch.Tensor:
        """Build the `[NUM_ROUTE_WAYPOINTS, 2]` route tensor in the rig-t0 frame.

        AlpaSim's `RouteGenerator.prepare_for_policy` guarantees exactly
        `NUM_ROUTE_WAYPOINTS` waypoints per submission -- already NaN-padded on
        the AlpaSim side when fewer live waypoints exist -- so any received
        route is required to match that length exactly.
        """
        route_state = self._buffers.route
        if len(route_state.waypoints) != NUM_ROUTE_WAYPOINTS:
            raise ValueError(
                f"AlpamayoPolicy received {len(route_state.waypoints)} route "
                f"waypoints from AlpaSim; the fixed contract is "
                f"{NUM_ROUTE_WAYPOINTS}."
            )

        ego_history = self._buffers.ego_history
        route_timestamp_us = route_state.timestamp_us
        first_history_timestamp_us = ego_history[0].timestamp_us
        last_history_timestamp_us = ego_history[-1].timestamp_us
        if (
            route_timestamp_us < first_history_timestamp_us
            or route_timestamp_us > last_history_timestamp_us
        ):
            raise ValueError(
                f"Route timestamp {route_timestamp_us} outside buffered ego window "
                f"[{first_history_timestamp_us}, {last_history_timestamp_us}]; an "
                f"out-of-window route timestamp "
                f"from AlpaSim is an invariant violation."
            )

        history_timestamps_us = [ego_pose.timestamp_us for ego_pose in ego_history]
        next_history_index = bisect_left(history_timestamps_us, route_timestamp_us)
        previous_history_index = max(0, next_history_index - 1)
        previous_history_pose = ego_history[previous_history_index]
        next_history_pose = ego_history[next_history_index]
        history_span_us = next_history_pose.timestamp_us - previous_history_pose.timestamp_us
        route_alpha = (
            0.0
            if history_span_us == 0
            else (route_timestamp_us - previous_history_pose.timestamp_us) / history_span_us
        )

        route_frame_world_xyz = (
            _vec3_to_tensor(previous_history_pose) * (1.0 - route_alpha)
            + _vec3_to_tensor(next_history_pose) * route_alpha
        )

        # Flip sign so quaternion lerp takes the short path.
        previous_quat = previous_history_pose.pose.quat
        next_quat = next_history_pose.pose.quat
        quat_dot = (
            previous_quat.w * next_quat.w
            + previous_quat.x * next_quat.x
            + previous_quat.y * next_quat.y
            + previous_quat.z * next_quat.z
        )
        next_quat_sign = -1.0 if quat_dot < 0.0 else 1.0
        previous_quat_tensor = torch.tensor(
            [previous_quat.w, previous_quat.x, previous_quat.y, previous_quat.z],
            dtype=torch.float32,
        )
        next_quat_tensor = torch.tensor(
            [next_quat.w, next_quat.x, next_quat.y, next_quat.z],
            dtype=torch.float32,
        )
        route_frame_world_quat = previous_quat_tensor * (1.0 - route_alpha) + next_quat_tensor * (
            next_quat_sign * route_alpha
        )
        route_frame_world_quat = route_frame_world_quat / torch.linalg.vector_norm(
            route_frame_world_quat
        ).clamp_min(1e-12)
        route_frame_world_rot = quat_tensor_to_rotation_matrix(route_frame_world_quat)

        latest_pose = ego_history[-1]
        latest_world_xyz = _vec3_to_tensor(latest_pose)
        latest_world_rot = quaternion_to_rotation_matrix(latest_pose.pose.quat)

        route_frame_to_rig_rot = latest_world_rot.T @ route_frame_world_rot
        route_frame_to_rig_xyz = latest_world_rot.T @ (route_frame_world_xyz - latest_world_xyz)

        route_waypoints_xyz = torch.tensor(
            [[wp.x, wp.y, wp.z] for wp in route_state.waypoints],
            dtype=torch.float32,
        )
        route_waypoints_rig_xyz = (
            route_frame_to_rig_rot @ route_waypoints_xyz.unsqueeze(-1)
        ).squeeze(-1) + route_frame_to_rig_xyz

        # Keep route fp32; bf16 loses visible precision over meter-scale waypoints.
        return route_waypoints_rig_xyz[:, :2].to(device=self._device, dtype=torch.float32)

    # ----- postprocess: ModelOutput -> PolicyOutput -----

    def _postprocess(
        self,
        model_output: ModelOutput,
        model_input: ModelInput,
        policy_input: PolicyInput,
    ) -> PolicyOutput:
        """Run the selector and pack the `PolicyOutput`."""
        if not policy_input.ego_trajectory.poses:
            raise ValueError("AlpamayoPolicy requires at least one ego pose")
        ego_pose_at_choice = policy_input.ego_trajectory.poses[-1].pose

        per_traj_logprob: torch.Tensor | None = model_output.logprob

        model_cfg = self._config.model
        future_dt_us = (
            torch.arange(1, model_cfg.num_future_waypoints + 1, dtype=torch.int64)
            * model_cfg.step_dt_us
        )
        horizon = model_output.pred_xyz.shape[2]
        if horizon != model_cfg.num_future_waypoints:
            raise ValueError(
                f"Model produced trajectory horizon {horizon} but PolicyConfig "
                f"declares num_future_waypoints={model_cfg.num_future_waypoints}; "
                "the model and policy config disagree."
            )

        chosen_traj = self._selector.select(
            model_output=model_output,
            previous_traj=self._buffers.last_chosen_traj,
            time_query_us=policy_input.time_query_us,
            time_now_us=policy_input.time_now_us,
            ego_pose_at_choice=ego_pose_at_choice,
            future_dt_us=future_dt_us,
        )
        self._buffers.update_last_chosen_traj(chosen_traj)
        output_xyz, output_quat, output_dt_us = build_policy_output_trajectory(
            ego_pose_now=ego_pose_at_choice,
            future_xyz_ego=chosen_traj.xyz,
            future_rot_ego=chosen_traj.rot,
            future_dt_us=future_dt_us,
        )

        chosen_logprob: torch.Tensor | None = None
        replay_data: PolicyReplayData | None = None
        if per_traj_logprob is not None:
            action_selection = ActionSelection(
                set_ix=int(chosen_traj.set_ix),
                sample_ix=int(chosen_traj.sample_ix),
            )
            chosen_logprob = per_traj_logprob[
                action_selection.set_ix,
                action_selection.sample_ix,
            ].view(1)
            replay_data = self._inference_engine.build_policy_replay_data(
                model_input=model_input,
                model_output=model_output,
                action_selection=action_selection,
            )
            self._log_valid_model_output(chosen_traj, chosen_logprob, model_output)

        all_pred_xyz: torch.Tensor | None = None
        all_pred_quat: torch.Tensor | None = None
        if per_traj_logprob is not None:
            all_pred_xyz = model_output.pred_xyz
            all_pred_quat = pred_rot_to_quat(model_output.pred_rot)

        return PolicyOutput(
            chosen_xyz=output_xyz,
            chosen_quat=output_quat,
            chosen_dt_us=output_dt_us,
            chosen_logprob=chosen_logprob,
            replay_data=replay_data,
            all_pred_xyz=all_pred_xyz,
            all_pred_quat=all_pred_quat,
        )

    def _log_valid_model_output(
        self,
        chosen_traj: ChosenTrajectory,
        chosen_logprob: torch.Tensor | None,
        model_output: ModelOutput,
    ) -> None:
        """Log a bounded stream proving the rollout policy emitted trajectories."""
        self._valid_output_count += 1
        if self._valid_output_count > 3 and self._valid_output_count % 10 != 0:
            return
        logprob_summary = "none"
        if chosen_logprob is not None and chosen_logprob.numel() > 0:
            logprob_summary = f"{float(chosen_logprob.reshape(-1)[0].item()):.4f}"
        logger.info(
            "AlpamayoPolicy session=%s valid_output=%d pred_xyz_shape=%s "
            "selected=(%d,%d) chosen_logprob=%s",
            self._session_uuid,
            self._valid_output_count,
            tuple(model_output.pred_xyz.shape),
            chosen_traj.set_ix,
            chosen_traj.sample_ix,
            logprob_summary,
        )


def _vec3_to_tensor(ego_pose: EgoPose) -> torch.Tensor:
    """Pack a `Pose.vec` into a `[3]` float32 tensor (xyz)."""
    v = ego_pose.pose.vec
    return torch.tensor([v.x, v.y, v.z], dtype=torch.float32)
