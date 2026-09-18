"""Qwen-Drive-1.0 inference model adapter for AlpaGym.

Bridges AlpaGym's BatchedModelInput/BatchedModelOutput to Qwen-Drive's
flow-matching Planning Expert pipeline.

Key differences from AutoVLA:
- Qwen-Drive uses flow-matching (continuous trajectory), not discrete action tokens
- Qwen-Drive outputs 50 waypoints @ 10Hz (0.1s interval), 5s horizon
- Trajectory output is (x, y, heading) in ego frame
- No token-level logprob; flow-matching density is not computed at inference time
- logprob is None for Phase 3 (inference baseline); Phase 4 (GRPO) will add it
"""


import logging
import math
from typing import Any

import numpy as np
import torch
from alpagym_host.config import SamplingParamsConfig
from alpagym_runtime.inference.types import (
    BatchedModelInput,
    BatchedModelOutput,
    ModelInput,
    ModelOutput,
)
from alpagym_runtime.replay import (
    ActionSelection,
    PolicyReplayData,
    require_payload_keys,
)
from PIL import Image

from alpagym_qwen_drive.stochastic_sampler import (
    stochastic_sample,
)

logger = logging.getLogger(__name__)

# Camera name mapping: AlpaGym logical IDs -> Qwen-Drive view names
# AlpaGym uses camera_front_wide_120fov, camera_cross_left_120fov, camera_cross_right_120fov
# Qwen-Drive uses <FRONT VIEW>, <FRONT LEFT VIEW>, <FRONT RIGHT VIEW>
CAMERA_ORDER_ALPAGYM = [
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
]
QWEN_DRIVE_VIEWS = ("<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>")


def heading_to_rotation_matrix(heading: torch.Tensor) -> torch.Tensor:
    """Convert yaw heading (radians) to SO(3) rotation matrices.

    Qwen-Drive predicts planar yaw only, so roll/pitch are identity.
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


def rotation_matrix_to_heading(rot: torch.Tensor) -> torch.Tensor:
    """Extract yaw heading from a 3x3 rotation matrix.

    heading = atan2(rot[1,0], rot[0,0])
    """
    return torch.atan2(rot[..., 1, 0], rot[..., 0, 0])


def route_to_nav_command(route_xy: torch.Tensor) -> int:
    """Convert route waypoints to a discrete navigation command.

    Qwen-Drive uses: 0=GO STRAIGHT, 1=TURN LEFT, 2=TURN RIGHT

    Heuristic: look at the lateral displacement of route waypoints
    relative to the forward (x) direction. If the route curves
    significantly left (positive y), return 1. If right (negative y),
    return 2. Otherwise 0 (straight).
    """
    # route_xy is [N, 2] in ego frame (x forward, y left)
    # Use waypoints beyond immediate vicinity for direction
    if route_xy.shape[0] == 0:
        return 0

    # Filter out NaN-padded waypoints
    valid = ~torch.isnan(route_xy[:, 0])
    route = route_xy[valid]
    if route.shape[0] < 2:
        return 0

    # Look at waypoints 5-15 meters ahead
    ahead_mask = (route[:, 0] > 3.0) & (route[:, 0] < 30.0)
    ahead = route[ahead_mask]
    if ahead.shape[0] == 0:
        ahead = route

    # Compute average lateral displacement
    avg_y = float(torch.mean(ahead[:, 1]).item())
    avg_x = float(torch.mean(ahead[:, 0]).item())

    # Compute heading angle of the route
    route_angle = math.atan2(avg_y, avg_x)

    # Threshold: 15 degrees
    if route_angle > 0.26:
        return 1  # TURN LEFT
    elif route_angle < -0.26:
        return 2  # TURN RIGHT
    else:
        return 0  # GO STRAIGHT


def compute_history_velocity_acceleration(
    history_xyz: np.ndarray, dt: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-point velocity and acceleration from history positions.

    Args:
        history_xyz: [N, 3] array of (x, y, heading) at 10Hz
        dt: time step between consecutive history points (0.1s for 10Hz)

    Returns:
        history_velocity: [N, 2] (vx, vy)
        history_acceleration: [N, 2] (ax, ay)
    """
    n = len(history_xyz)
    velocity = np.zeros((n, 2), dtype=np.float32)
    acceleration = np.zeros((n, 2), dtype=np.float32)

    if n >= 2:
        # Central differences for interior points, forward/backward for edges
        for i in range(n):
            if i == 0:
                velocity[i] = (history_xyz[i + 1, :2] - history_xyz[i, :2]) / dt
            elif i == n - 1:
                velocity[i] = (history_xyz[i, :2] - history_xyz[i - 1, :2]) / dt
            else:
                velocity[i] = (history_xyz[i + 1, :2] - history_xyz[i - 1, :2]) / (2 * dt)

    if n >= 3:
        for i in range(n):
            if i == 0:
                acceleration[i] = (velocity[i + 1] - velocity[i]) / dt
            elif i == n - 1:
                acceleration[i] = (velocity[i] - velocity[i - 1]) / dt
            else:
                acceleration[i] = (velocity[i + 1] - velocity[i - 1]) / (2 * dt)

    return velocity, acceleration


class QwenDriveInferenceModel:
    """Adapter between AlpaGym typed I/O and Qwen-Drive-1.0 Planning Expert."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_future_waypoints: int = 50,
        step_dt_us: int = 100000,
        num_inference_steps: int = 10,
        rl_sampling_config: Any = None,
    ) -> None:
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype = dtype
        self._num_future_waypoints = num_future_waypoints
        self._step_dt_us = step_dt_us
        self._num_inference_steps = num_inference_steps
        self._rl_sampling_config = rl_sampling_config
        self._step_counter = 0

    def _get_inner_model(self) -> Any:
        """Unwrap the actual QwenDriveForPlanning model.

        Cosmos-RL may replace ``self._model`` with a ``QwenDriveCosmos``
        wrapper (its ``BaseModel`` subclass) via ``set_model``.  The
        ``generate_trajectory`` method lives on the inner
        ``QwenDriveForPlanning`` which is at ``wrapper.model``.
        """
        m = self._model
        # QwenDriveCosmos has .model = QwenDriveForPlanning
        if hasattr(m, "model") and hasattr(m.model, "generate_trajectory"):
            return m.model
        return m

    def get_model(self) -> torch.nn.Module:
        return self._model

    def set_model(self, model: torch.nn.Module) -> None:
        self._model = model

    def sample_trajectories_from_data(
        self,
        model_input: BatchedModelInput,
        sampling: SamplingParamsConfig,
        return_trace_for_rl: bool = False,
    ) -> BatchedModelOutput:
        """Run Qwen-Drive flow-matching inference and return trajectories.

        Output shapes:
            pred_xyz: [B, S, K, T, 3]  (B=batch, S=1 set, K=samples, T=50 waypoints)
            pred_rot: [B, S, K, T, 3, 3]
            logprob: None (Phase 3; flow-matching density not computed at inference)
        """
        batch_size = model_input.camera_frames.shape[0]
        num_samples = max(1, sampling.num_traj_samples)

        all_pred_xyz = []
        all_pred_rot = []
        all_rl_traces: list[list[dict[str, Any] | None]] = []

        for batch_idx in range(batch_size):
            pred_xyz, pred_rot, rl_trace = self._infer_single(
                model_input, batch_idx, num_samples
            )
            all_pred_xyz.append(pred_xyz)
            all_pred_rot.append(pred_rot)
            all_rl_traces.append(rl_trace)

        pred_xyz = torch.stack(all_pred_xyz, dim=0)  # [B, S, K, T, 3]
        pred_rot = torch.stack(all_pred_rot, dim=0)  # [B, S, K, T, 3, 3]

        if all_rl_traces and all_rl_traces[0] is not None:
            # Stack each trace key across the batch axis so the leaves carry
            # the leading dim ``unbind`` requires. Trace tensors already hold
            # the sample axis K inside them ([K, ...]).
            extra = {
                key: torch.stack(
                    [trace[key] for trace in all_rl_traces], dim=0
                )
                for key in all_rl_traces[0]
            }
            # logprob: [B] rows of [K] trajectory-level log-densities.
            logprob = extra.pop("old_logprob").view(batch_size, 1, num_samples)
        else:
            logprob = None
            extra = {}

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
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
        """Process one batch row: build DrivingScene, run inference, decode trajectory.

        Returns:
            pred_xyz: [1, num_samples, T, 3]
            pred_rot: [1, num_samples, T, 3, 3]
            rl_trace: per-sample RL trace dict (``None`` when deterministic
                planning is used). Each value stacks the K samples on dim 0.
        """
        # 1. Extract camera frames and convert to DrivingScene views
        views = self._build_camera_views(model_input, batch_idx)

        # 2. Extract ego history and compute heading/velocity/acceleration
        history, history_velocity, history_acceleration, ego_velocity, ego_acceleration = \
            self._extract_ego_history(model_input, batch_idx)

        # 3. Convert route to navigation command
        route_xy = model_input.route_xy[batch_idx]  # [N, 2]
        nav_command = route_to_nav_command(route_xy)

        # 4. Build driving command (one-hot: 4 classes 鈥?[straight, left, right, unknown])
        # Qwen-Drive's PlanningExpert expects ego_status_dim=8:
        # velocity[2] + acceleration[2] + driving_command[4] = 8
        driving_command = np.zeros(4, dtype=np.float32)
        driving_command[nav_command] = 1.0

        # 5. Build DrivingScene
        from qwen_drive.scene import DrivingScene

        scene = DrivingScene(
            views=views,
            history=history,
            history_velocity=history_velocity,
            history_acceleration=history_acceleration,
            ego_velocity=ego_velocity,
            ego_acceleration=ego_acceleration,
            driving_command=driving_command,
            nav_command=nav_command,
            token=f"alpagym_step_{self._step_counter}",
        )
        self._step_counter += 1

        # 6. Run Qwen-Drive inference. With RL sampling enabled, use the
        # stochastic sampler (paper Eq. 9-12) so the emitted action has a
        # differentiable transition likelihood; otherwise deterministic
        # direct planning.
        inner_model = self._get_inner_model()
        if self._rl_sampling_config is not None:
            trajectories, rl_trace = self._stochastic_infer(
                inner_model, scene, num_samples
            )
        else:
            result = inner_model.generate_trajectory(
                scene,
                mode="direct_planning",
                num_samples=num_samples,
                num_steps=self._num_inference_steps,
                seed=42,
            )
            trajectories = result.trajectories  # [num_samples, 50, 3] numpy
            rl_trace = None

        # 7. Convert to AlpaGym format
        traj_tensor = torch.as_tensor(trajectories, dtype=torch.float32)  # [K, T, 3]

        # pred_xyz: [x, y, 0.0] (z=0 for planar driving)
        pred_xyz = torch.zeros(num_samples, self._num_future_waypoints, 3, dtype=torch.float32)
        pred_xyz[:, :, 0] = traj_tensor[:, :, 0]  # x
        pred_xyz[:, :, 1] = traj_tensor[:, :, 1]  # y
        # z = 0 (planar)

        # pred_rot: heading -> rotation matrix
        heading = traj_tensor[:, :, 2]  # [K, T]
        pred_rot = heading_to_rotation_matrix(heading)  # [K, T, 3, 3]

        # Add set dimension: [1, K, T, 3] and [1, K, T, 3, 3]
        pred_xyz = pred_xyz.unsqueeze(0)  # [1, K, T, 3]
        pred_rot = pred_rot.unsqueeze(0)  # [1, K, T, 3, 3]

        if self._step_counter <= 3 or self._step_counter % 20 == 0:
            logger.info(
                "QwenDriveInferenceModel step=%d nav_cmd=%d pred_xyz_first3=%s "
                "pred_heading_first3=%s",
                self._step_counter,
                nav_command,
                pred_xyz[0, 0, :3].tolist(),
                heading[0, :3].tolist(),
            )

        return pred_xyz, pred_rot, rl_trace

    def _stochastic_infer(
        self,
        inner_model: Any,
        scene: Any,
        num_samples: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sample with the stochastic flow sampler and record the RL trace.

        Builds the processor inputs from ``scene``, runs the VLM prefill
        exactly like ``_plan_from_cache``, then integrates the stochastic
        sampler (paper Eq. 9-12). The trace persists everything the trainer
        needs to re-evaluate the likelihood under new parameters: the
        normalized conditioning tensors, the shared initial noise, and the
        injected subspace coefficients.
        """
        from qwen_drive.trajectory import denormalize_trajectory, normalize_history

        processor_inputs = inner_model.processor(scene, with_reasoning=False)
        inputs = {
            key: value.to(inner_model.device) if torch.is_tensor(value) else value
            for key, value in processor_inputs.items()
        }
        scene_cache, anchor = inner_model._prefill(inputs)
        scale = inner_model.trajectory_scale(inner_model.device)
        history = normalize_history(inputs["history"].float(), scale)

        seed = int(self._step_counter)
        generator = torch.Generator(device=inner_model.device).manual_seed(seed)
        noise = inner_model.config.noise_init_std * torch.randn(
            num_samples,
            inner_model.config.num_future_points,
            inner_model.config.trajectory_point_dim,
            generator=generator,
            device=inner_model.device,
            dtype=torch.float32,
        )
        normalized, z, states, old_logprob = stochastic_sample(
            inner_model.planning_expert,
            scene_cache=scene_cache,
            position_anchor=anchor,
            history=history,
            history_velocity=inputs["history_velocity"].float(),
            history_acceleration=inputs["history_acceleration"].float(),
            nav_command=inputs["nav_command"],
            ego_status=inputs["ego_status"].float(),
            noise=noise,
            config=self._rl_sampling_config,
            generator=generator,
        )
        trajectories = denormalize_trajectory(normalized, scale).cpu().numpy()
        # Condition the trace tensors to the sample axis so every value
        # carries K on dim 0 (the caller stacks them across the batch axis).
        trace = {
            "history": history.expand(num_samples, -1, -1).cpu(),
            "history_velocity": inputs["history_velocity"]
            .float()
            .expand(num_samples, -1, -1)
            .cpu(),
            "history_acceleration": inputs["history_acceleration"]
            .float()
            .expand(num_samples, -1, -1)
            .cpu(),
            "nav_command": inputs["nav_command"].expand(num_samples).cpu(),
            "ego_status": inputs["ego_status"].float().expand(num_samples, -1).cpu(),
            "initial_noise": noise.cpu(),
            "z": z.cpu(),
            "states": states.cpu(),
            "old_logprob": old_logprob.cpu(),
        }
        return trajectories, trace

    def _build_camera_views(
        self,
        model_input: BatchedModelInput,
        batch_idx: int,
    ) -> dict:
        """Convert AlPaGym camera frames to Qwen-Drive DrivingScene views.

        AlPaGym provides camera_frames [B, C*T, 3, H, W] uint8 (CHW format).
        Frames are ordered: all T frames of camera 0, then camera 1, then camera 2.

        Returns: dict mapping Qwen-Drive view names to lists of CameraFrame objects.
        """
        from qwen_drive.scene import CameraFrame

        camera_frames = model_input.camera_frames[batch_idx]  # [C*T, 3, H, W]
        camera_indices = model_input.camera_indices[batch_idx]  # [C*T]


        # Determine number of cameras and frames per camera
        unique_cams = torch.unique(camera_indices)

        # Group frames by camera index
        views = {}
        for cam_idx_pos, cam_id_val in enumerate(unique_cams.tolist()):
            cam_mask = camera_indices == cam_id_val
            cam_frames = camera_frames[cam_mask]  # [T, 3, H, W]

            # Map camera index to Qwen-Drive view name
            if cam_idx_pos < len(QWEN_DRIVE_VIEWS):
                view_name = QWEN_DRIVE_VIEWS[cam_idx_pos]
            else:
                view_name = QWEN_DRIVE_VIEWS[0]  # fallback

            # Convert uint8 CHW tensors to PIL Images, wrap in CameraFrame
            frame_list = []
            for t in range(cam_frames.shape[0]):
                frame_uint8 = cam_frames[t]  # [3, H, W]
                # Convert CHW -> HWC for PIL
                frame_hwc = frame_uint8.permute(1, 2, 0).cpu().numpy()
                if frame_hwc.shape[2] == 1:
                    frame_hwc = np.repeat(frame_hwc, 3, axis=2)
                pil_img = Image.fromarray(frame_hwc.astype(np.uint8))
                frame_list.append(CameraFrame(image=pil_img))

            views[view_name] = frame_list

        # Ensure all expected views are present
        for view_name in QWEN_DRIVE_VIEWS:
            if view_name not in views:
                # Duplicate front view if missing
                front = views.get(QWEN_DRIVE_VIEWS[0], [])
                views[view_name] = front

        return views

    def _extract_ego_history(
        self,
        model_input: BatchedModelInput,
        batch_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract ego history and derive heading, velocity, acceleration.

        AlPaGym provides:
            ego_history_xyz [B, H, 3] - positions in ego frame (last pose = origin)
            ego_history_rot [B, H, 3, 3] - rotation matrices in ego frame

        Returns:
            history: [H, 3] (x, y, heading) at 10Hz
            history_velocity: [H, 2] (vx, vy)
            history_acceleration: [H, 2] (ax, ay)
            ego_velocity: [2] (vx, vy) at current time
            ego_acceleration: [2] (ax, ay) at current time
        """
        ego_history_xyz = model_input.ego_history_xyz[batch_idx]  # expected [H, 3] but may be [S, H, 3]
        ego_history_rot = model_input.ego_history_rot[batch_idx]  # expected [H, 3, 3] but may be [S, H, 3, 3]

        # Handle extra "set" dimension: model_input is [B, S, H, 3] / [B, S, H, 3, 3]
        # After [batch_idx], we get [S, H, 3] / [S, H, 3, 3] where S=1
        if ego_history_xyz.dim() == 3:
            ego_history_xyz = ego_history_xyz[0]  # [S, H, 3] 鈫?[H, 3]
        if ego_history_rot.dim() == 4:
            ego_history_rot = ego_history_rot[0]  # [S, H, 3, 3] 鈫?[H, 3, 3]

        n = ego_history_xyz.shape[0]

        # Extract heading from rotation matrices
        headings = rotation_matrix_to_heading(ego_history_rot)  # [H]

        # Build history array [H, 3] = (x, y, heading)
        history = np.zeros((n, 3), dtype=np.float64)
        history[:, 0] = ego_history_xyz[:, 0].cpu().numpy()
        history[:, 1] = ego_history_xyz[:, 1].cpu().numpy()
        history[:, 2] = headings.cpu().numpy()

        # Compute velocity and acceleration at 10Hz (dt=0.1s)
        # AlPaGym history is at the simulator's reporting rate.
        # The AlpamayoPolicy buffers ego poses at the simulator's control rate.
        # For Qwen-Drive, history should be at 10Hz (100ms intervals).
        # If the simulator runs at 10Hz (control_timestep_us=100000), this is aligned.
        dt = self._step_dt_us / 1e6  # Convert microseconds to seconds
        history_velocity, history_acceleration = compute_history_velocity_acceleration(
            history, dt=dt
        )

        # Current ego velocity and acceleration (last history point)
        ego_velocity = history_velocity[-1].astype(np.float32)
        ego_acceleration = history_acceleration[-1].astype(np.float32)

        return (
            history,
            history_velocity.astype(np.float32),
            history_acceleration.astype(np.float32),
            ego_velocity,
            ego_acceleration,
        )

    def build_policy_replay_data(
        self,
        model_input: ModelInput,
        model_output: ModelOutput,
        action_selection: ActionSelection,
    ) -> PolicyReplayData:
        """Pack selected-only Qwen-Drive replay data for the GRPO trainer.

        Persists the six raw ``ModelInput`` fields (the trainer rebuilds the
        ``DrivingScene`` from them, mirroring the rollout-side adapter) plus
        the stochastic-sampling trace of the *selected* sample: initial noise,
        injected subspace coefficients, and the rollout-time old log-prob.
        """
        if model_output.logprob is None or "z" not in model_output.extra:
            raise ValueError(
                "qwen_drive RL replay requires the stochastic sampling trace; "
                "enable RL sampling (bundle_config.rl_sampling) for training runs"
            )
        old_logprob = model_output.logprob[
            action_selection.set_ix, action_selection.sample_ix
        ].view(()).cpu()

        payload: dict[str, Any] = {
            "ego_history_xyz": model_input.ego_history_xyz.cpu().numpy().tolist(),
            "ego_history_rot": model_input.ego_history_rot.cpu().numpy().tolist(),
            "camera_frames": model_input.camera_frames.cpu().numpy().tolist(),
            "camera_indices": model_input.camera_indices.cpu().numpy().tolist(),
            "relative_timestamps": model_input.relative_timestamps.cpu().numpy().tolist(),
            "route_xy": model_input.route_xy.cpu().numpy().tolist(),
            "selected_states": model_output.extra["states"][
                action_selection.sample_ix
            ].cpu(),
            "selected_z": model_output.extra["z"][
                action_selection.sample_ix
            ].cpu(),
        }
        return PolicyReplayData(
            replay_schema_version=1,
            payload_schema="qwen_drive.trajectory.v1",
            payload_schema_version=2,
            model_family="qwen_drive",
            action_selection=action_selection,
            old_logprob=old_logprob,
            payload=payload,
        )

    @classmethod
    def build_trainer_model_inputs(
        cls,
        replay_data: PolicyReplayData,
        **kwargs,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Build trainer-side forward kwargs from replay data (Phase 4: GRPO).

        The trainer re-runs the same ``DrivingScene`` construction the rollout
        used (from the persisted raw inputs), re-runs the VLM prefill to get
        the scene cache, and forwards the recorded initial noise and subspace
        coefficients for the differentiable likelihood re-evaluation. The
        forward kwargs keep the rollout's single-sample layout (no batch
        axis); the packer stacks them across steps.
        """
        if replay_data.payload_schema != "qwen_drive.trajectory.v1":
            raise ValueError(
                f"qwen_drive replay payload_schema "
                f"{replay_data.payload_schema!r} != 'qwen_drive.trajectory.v1'"
            )
        if replay_data.payload_schema_version != 2:
            raise ValueError(
                "qwen_drive replay payload_schema_version "
                f"{replay_data.payload_schema_version} != 2 (stochastic trace); "
                "re-run the rollout with the Phase 4 sampler"
            )
        payload = replay_data.payload
        require_payload_keys(
            replay_data.model_family,
            payload,
            (
                "ego_history_xyz",
                "ego_history_rot",
                "camera_frames",
                "camera_indices",
                "relative_timestamps",
                "route_xy",
                "selected_states",
                "selected_z",
            ),
        )

        model_inputs: dict[str, Any] = {
            key: torch.as_tensor(payload[key], dtype=torch.float32)
            for key in (
                "ego_history_xyz",
                "ego_history_rot",
                "camera_frames",
                "route_xy",
            )
        }
        model_inputs["camera_frames"] = torch.as_tensor(
            payload["camera_frames"], dtype=torch.uint8
        )
        model_inputs["camera_indices"] = torch.as_tensor(
            payload["camera_indices"], dtype=torch.int64
        )
        model_inputs["relative_timestamps"] = torch.as_tensor(
            payload["relative_timestamps"], dtype=torch.int64
        )
        model_inputs["selected_states"] = torch.as_tensor(
            payload["selected_states"], dtype=torch.float32
        )
        model_inputs["selected_z"] = torch.as_tensor(
            payload["selected_z"], dtype=torch.float32
        )

        if replay_data.old_logprob is None:
            raise ValueError("qwen_drive replay requires old_logprob")
        old_logprob = torch.as_tensor(
            replay_data.old_logprob, dtype=torch.float32
        ).reshape(())
        return model_inputs, old_logprob
