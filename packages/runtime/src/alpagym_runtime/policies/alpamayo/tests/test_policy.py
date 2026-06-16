# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for `AlpamayoPolicy`."""

import io
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import pytest
import torch
from alpagym_host.config import (
    AlpamayoPolicyConfig,
    InferenceConfig,
    ModelConfig,
    SamplingParamsConfig,
    TrajectorySelectorKind,
)
from alpagym_runtime.inference.inference_engine import InferenceEngine
from alpagym_runtime.inference.testing import driven_inference_engine
from alpagym_runtime.inference.types import (
    NUM_ROUTE_WAYPOINTS,
    BatchedModelInput,
    BatchedModelOutput,
    ModelInput,
    ModelOutput,
)
from alpagym_runtime.policies import factory as factory_mod
from alpagym_runtime.policies.alpamayo.geometry_utils import rotation_matrix_to_quat
from alpagym_runtime.policies.alpamayo.policy import AlpamayoPolicy
from alpagym_runtime.replay import ActionSelection, PolicyReplayData
from alpagym_runtime.types import (
    CameraCalibration,
    CameraImage,
    CameraIntrinsics,
    ChosenTrajectory,
    EgoPose,
    PolicyInput,
    Pose,
    Quaternion,
    RolloutCalibration,
    RouteWaypoint,
    Trajectory,
    Vec3,
)
from PIL import Image

CAMERA_ID = "camera_front_wide_120fov"


@dataclass
class _FakeOutput:
    """Preset tensors and capture slot for a synthetic InferenceModel."""

    pred_xyz: torch.Tensor  # [num_traj_sets, num_traj_samples, T, 3]
    pred_rot: torch.Tensor  # [num_traj_sets, num_traj_samples, T, 3, 3]
    logprob: torch.Tensor | None  # [num_traj_sets, num_traj_samples] or None
    last_input: BatchedModelInput | None = None


class _FakeInferenceModel:
    """Minimal :class:`InferenceModel` returning preset batched tensors per call."""

    def __init__(self, output: _FakeOutput) -> None:
        """Bind the inference model to the preset output it should emit."""
        self._output = output

    def sample_trajectories_from_data(
        self,
        model_input: BatchedModelInput,
        sampling: SamplingParamsConfig,
        return_trace_for_rl: bool = False,
    ) -> BatchedModelOutput:
        """Record `model_input` and return the preset tensors batched."""
        del sampling
        self._output.last_input = model_input
        batch_size = model_input.ego_history_xyz.shape[0]
        pred_xyz = self._output.pred_xyz.unsqueeze(0).expand(
            batch_size, *self._output.pred_xyz.shape
        )
        pred_rot = self._output.pred_rot.unsqueeze(0).expand(
            batch_size, *self._output.pred_rot.shape
        )
        logprob: torch.Tensor | None = None
        if return_trace_for_rl and self._output.logprob is not None:
            logprob = self._output.logprob.unsqueeze(0).expand(
                batch_size, *self._output.logprob.shape
            )
        return BatchedModelOutput(
            pred_xyz=pred_xyz.contiguous(),
            pred_rot=pred_rot.contiguous(),
            logprob=logprob.contiguous() if logprob is not None else None,
        )

    def build_policy_replay_data(
        self,
        model_input: ModelInput,
        model_output: ModelOutput,
        action_selection: ActionSelection,
    ) -> PolicyReplayData:
        """Build a minimal replay envelope for policy tests."""
        if model_output.logprob is None:
            raise ValueError("test replay requires logprob")
        return PolicyReplayData(
            replay_schema_version=1,
            payload_schema="alpamayo_r1.trajectory.v1",
            payload_schema_version=1,
            model_family="alpamayo_r1",
            action_selection=action_selection,
            old_logprob=(
                model_output.logprob[action_selection.set_ix, action_selection.sample_ix]
                .to(dtype=torch.float32)
                .reshape(())
            ),
            payload={
                "model_input": asdict(model_input),
                "tokenized_data": {
                    "input_ids": torch.tensor([1, 2, 3], dtype=torch.int64),
                    "attention_mask": torch.ones(3, dtype=torch.bool),
                    "position_ids": None,
                    "attention_mask_4d": None,
                    "labels_mask": torch.tensor([False, True, True]),
                },
                "ego_future_xyz": model_output.pred_xyz[
                    action_selection.set_ix, action_selection.sample_ix
                ].unsqueeze(0),
                "ego_future_rot": model_output.pred_rot[
                    action_selection.set_ix, action_selection.sample_ix
                ].unsqueeze(0),
                "token_logprob_count": torch.tensor(2, dtype=torch.int64),
            },
        )


def _make_jpeg(width: int = 8, height: int = 6) -> bytes:
    """Encode a single-color RGB image as JPEG bytes for the policy tests."""
    image = Image.new("RGB", (width, height), color=(123, 45, 67))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def _calibration_for(camera_id: str) -> RolloutCalibration:
    """Synthetic single-camera calibration for the policy tests."""
    return (
        CameraCalibration(
            name=camera_id,
            logical_id=camera_id,
            intrinsics=CameraIntrinsics(fx=400.0, fy=410.0, cx=320.0, cy=240.0),
            extrinsic_pose=Pose(
                vec=Vec3(x=1.0, y=2.0, z=3.0),
                quat=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
            ),
        ),
    )


def _policy_config(
    selector_name: TrajectorySelectorKind,
    num_traj_samples: int,
    num_traj_sets: int,
    num_future_waypoints: int,
    num_context_frames: int = 1,
    model_dtype: str = "float32",
    return_trace_for_rl: bool = True,
    force_determinism: bool = False,
) -> AlpamayoPolicyConfig:
    """Build an AlpamayoPolicyConfig for policy tests."""
    sampling = SamplingParamsConfig(
        top_p=0.98,
        top_k=None,
        temperature=0.6,
        num_traj_samples=num_traj_samples,
        num_traj_sets=num_traj_sets,
        max_generation_length=None,
        force_determinism=force_determinism,
    )
    return AlpamayoPolicyConfig(
        kind="alpamayo",
        model=ModelConfig(
            kind="alpamayo_r1",
            path="unused-by-tests",
            device="cpu",
            dtype=model_dtype,
            use_cameras=[CAMERA_ID],
            num_context_frames=num_context_frames,
            num_historical_waypoints=2,
            num_future_waypoints=num_future_waypoints,
            step_dt_us=1_000_000 // num_future_waypoints,
            input_size=[6, 8],
        ),
        inference=InferenceConfig(
            max_batch_size=1,
            return_trace_for_rl=return_trace_for_rl,
            sampling=sampling,
        ),
        trajectory_selector=selector_name,
    )


def _pad_route_to_contract(
    waypoints: tuple[RouteWaypoint, ...],
) -> tuple[RouteWaypoint, ...]:
    """Pad waypoints to `NUM_ROUTE_WAYPOINTS` with NaN tails (mimics AlpaSim's prep)."""
    nan = float("nan")
    padding_count = NUM_ROUTE_WAYPOINTS - len(waypoints)
    if padding_count < 0:
        raise ValueError(
            f"_pad_route_to_contract received {len(waypoints)} waypoints, more "
            f"than the contract size {NUM_ROUTE_WAYPOINTS}."
        )
    return waypoints + tuple(RouteWaypoint(x=nan, y=nan, z=nan) for _ in range(padding_count))


def _policy_input(
    time_now_us: int,
    time_query_us: int | None = None,
    ego_poses: tuple[EgoPose, ...] | None = None,
    route_waypoints: tuple[RouteWaypoint, ...] | None = None,
    route_timestamp_us: int | None = None,
) -> PolicyInput:
    """Build a PolicyInput with one freshly-encoded JPEG for `CAMERA_ID`."""
    if ego_poses is None:
        ego_poses = (
            EgoPose(timestamp_us=time_now_us - 100_000, pose=Pose()),
            EgoPose(timestamp_us=time_now_us, pose=Pose()),
        )
    if route_waypoints is None:
        route_waypoints = _pad_route_to_contract((RouteWaypoint(x=5.0, y=0.0),))
    if route_timestamp_us is None:
        route_timestamp_us = time_now_us
    return PolicyInput(
        step_index=0,
        time_now_us=time_now_us,
        time_query_us=time_query_us if time_query_us is not None else time_now_us + 200_000,
        camera_images=(
            CameraImage(
                logical_id=CAMERA_ID,
                image_bytes=_make_jpeg(),
                frame_end_us=time_now_us,
            ),
        ),
        ego_trajectory=Trajectory(poses=ego_poses),
        route_waypoints=route_waypoints,
        route_timestamp_us=route_timestamp_us,
        calibration=_calibration_for(CAMERA_ID),
    )


def _run_policy_once(
    config: AlpamayoPolicyConfig,
    policy_input: PolicyInput,
) -> BatchedModelInput:
    """Run one policy step and return the model input seen by inference."""
    horizon = config.model.num_future_waypoints
    pred_xyz = torch.zeros((1, 1, horizon, 3), dtype=torch.float32)
    pred_rot = torch.eye(3).expand(1, 1, horizon, 3, 3).clone()
    output = _FakeOutput(pred_xyz=pred_xyz, pred_rot=pred_rot, logprob=None)
    inference_model = _FakeInferenceModel(output)
    with driven_inference_engine(
        inference_model=inference_model,
        sampling=config.inference.sampling,
        return_trace_for_rl=config.inference.return_trace_for_rl,
    ) as inference_engine:
        policy = AlpamayoPolicy(
            inference_engine=inference_engine,
            session_uuid="test-session",
            config=config,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        policy.step(policy_input)
    assert output.last_input is not None
    return output.last_input


def test_alpamayo_policy_identity_selector_packs_full_output() -> None:
    """Identity selector picks (set=0, sample=0); logprob and trace fields populate."""
    horizon = 4
    num_traj_sets = 1
    num_traj_samples = 2

    pred_xyz = torch.arange(
        num_traj_sets * num_traj_samples * horizon * 3, dtype=torch.float32
    ).reshape(num_traj_sets, num_traj_samples, horizon, 3)
    pred_rot = torch.eye(3).expand(num_traj_sets, num_traj_samples, horizon, 3, 3).clone()
    logprob = torch.linspace(
        -1.0, 1.0, num_traj_sets * num_traj_samples, dtype=torch.float32
    ).reshape(num_traj_sets, num_traj_samples)
    output = _FakeOutput(pred_xyz=pred_xyz, pred_rot=pred_rot, logprob=logprob)
    inference_model = _FakeInferenceModel(output)
    sampling = SamplingParamsConfig(
        top_p=0.98,
        top_k=None,
        temperature=0.6,
        num_traj_samples=num_traj_samples,
        num_traj_sets=num_traj_sets,
        max_generation_length=None,
    )
    config = _policy_config(
        selector_name=TrajectorySelectorKind.identity,
        num_traj_samples=num_traj_samples,
        num_traj_sets=num_traj_sets,
        num_future_waypoints=horizon,
    )
    with driven_inference_engine(
        inference_model=inference_model, sampling=sampling, return_trace_for_rl=True
    ) as inference_engine:
        policy = AlpamayoPolicy(
            inference_engine=inference_engine,
            session_uuid="test-session",
            config=config,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        result = policy.step(_policy_input(time_now_us=1_000_000))

    assert result.chosen_xyz.shape == (horizon + 1, 3)
    assert torch.equal(
        result.chosen_xyz,
        torch.cat([torch.zeros((1, 3), dtype=torch.float32), pred_xyz[0, 0]], dim=0),
    )

    expected_quat = rotation_matrix_to_quat(pred_rot[0, 0])
    assert result.chosen_quat.shape == (horizon + 1, 4)
    assert torch.allclose(
        result.chosen_quat,
        torch.cat(
            [torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32), expected_quat],
            dim=0,
        ),
        atol=1e-6,
    )

    expected_dt_us = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64),
            torch.arange(1, horizon + 1, dtype=torch.int64) * config.model.step_dt_us,
        ],
        dim=0,
    )
    assert result.chosen_dt_us.dtype == torch.int64
    assert result.chosen_dt_us.shape == (horizon + 1,)
    assert torch.equal(result.chosen_dt_us, expected_dt_us)

    expected_per_traj_logprob = logprob
    assert result.chosen_logprob is not None
    assert result.chosen_logprob.shape == (1,)
    assert torch.allclose(result.chosen_logprob, expected_per_traj_logprob[0, 0].view(1), atol=1e-6)

    assert result.replay_data is not None
    assert isinstance(result.replay_data, PolicyReplayData)
    assert result.replay_data.model_family == "alpamayo_r1"
    assert result.replay_data.action_selection == ActionSelection(
        set_ix=0,
        sample_ix=0,
    )
    payload = result.replay_data.payload
    assert torch.equal(payload["ego_future_xyz"], pred_xyz[0, 0].unsqueeze(0))
    assert torch.equal(payload["ego_future_rot"], pred_rot[0, 0].unsqueeze(0))
    assert "old_logprob" not in payload
    assert result.replay_data.old_logprob is not None
    assert torch.allclose(result.replay_data.old_logprob, expected_per_traj_logprob[0, 0])
    assert payload["tokenized_data"]["input_ids"].dtype == torch.int64
    assert payload["model_input"]["ego_history_xyz"].shape == (1, 2, 3)

    assert result.all_pred_xyz is not None
    assert result.all_pred_quat is not None
    assert result.all_pred_xyz.shape == (num_traj_sets, num_traj_samples, horizon, 3)
    assert result.all_pred_quat.shape == (num_traj_sets, num_traj_samples, horizon, 4)
    assert torch.equal(result.all_pred_xyz, pred_xyz)


def test_alpamayo_policy_keeps_history_and_route_fp32_with_bfloat16_policy() -> None:
    """bf16 policy execution still sends trajectory-sensitive tensors as fp32."""
    horizon = 4
    num_traj_sets = 1
    num_traj_samples = 1
    pred_xyz = torch.zeros((num_traj_sets, num_traj_samples, horizon, 3), dtype=torch.float32)
    pred_rot = torch.eye(3).expand(num_traj_sets, num_traj_samples, horizon, 3, 3).clone()
    output = _FakeOutput(pred_xyz=pred_xyz, pred_rot=pred_rot, logprob=None)
    inference_model = _FakeInferenceModel(output)
    sampling = SamplingParamsConfig(
        top_p=0.98,
        top_k=None,
        temperature=0.6,
        num_traj_samples=num_traj_samples,
        num_traj_sets=num_traj_sets,
        max_generation_length=None,
    )
    config = _policy_config(
        selector_name=TrajectorySelectorKind.identity,
        num_traj_samples=num_traj_samples,
        num_traj_sets=num_traj_sets,
        num_future_waypoints=horizon,
        model_dtype="bfloat16",
    )
    with driven_inference_engine(
        inference_model=inference_model, sampling=sampling, return_trace_for_rl=False
    ) as inference_engine:
        policy = AlpamayoPolicy(
            inference_engine=inference_engine,
            session_uuid="test-session",
            config=config,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        policy.step(_policy_input(time_now_us=1_000_000))

    assert output.last_input is not None
    assert output.last_input.ego_history_xyz.dtype == torch.float32
    assert output.last_input.ego_history_rot.dtype == torch.float32
    assert output.last_input.route_xy.dtype == torch.float32


def test_alpamayo_policy_attaches_incrementing_deterministic_seed() -> None:
    """Policy preprocessing emits per-step seeds only when deterministic mode is enabled."""
    horizon = 4
    pred_xyz = torch.zeros((1, 1, horizon, 3), dtype=torch.float32)
    pred_rot = torch.eye(3, dtype=torch.float32).expand(1, 1, horizon, 3, 3).clone()
    output = _FakeOutput(pred_xyz=pred_xyz, pred_rot=pred_rot, logprob=None)
    inference_model = _FakeInferenceModel(output)
    sampling = SamplingParamsConfig(
        top_p=0.98,
        top_k=None,
        temperature=0.6,
        num_traj_samples=1,
        num_traj_sets=1,
        max_generation_length=None,
        force_determinism=True,
    )
    config = _policy_config(
        selector_name=TrajectorySelectorKind.identity,
        num_traj_samples=1,
        num_traj_sets=1,
        return_trace_for_rl=False,
        num_future_waypoints=horizon,
        force_determinism=True,
    )

    with driven_inference_engine(
        inference_model=inference_model, sampling=sampling, return_trace_for_rl=False
    ) as inference_engine:
        policy = AlpamayoPolicy(
            inference_engine=inference_engine,
            session_uuid="seeded-session",
            config=config,
            device=torch.device("cpu"),
            dtype=torch.float32,
            seed=41,
        )
        policy.step(_policy_input(time_now_us=1_000_000))
        first_input = output.last_input
        policy.step(_policy_input(time_now_us=2_000_000))
        second_input = output.last_input

    assert first_input is not None
    assert second_input is not None
    assert first_input.seed is not None
    assert second_input.seed is not None
    assert first_input.seed.tolist() == [41]
    assert second_input.seed.tolist() == [42]


def test_build_policy_factory_passes_random_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Driver policy factory passes AlpaSim's per-session random seed to the policy."""
    config = _policy_config(
        selector_name=TrajectorySelectorKind.identity,
        num_traj_samples=1,
        num_traj_sets=1,
        return_trace_for_rl=False,
        num_future_waypoints=4,
    )
    captured: dict[str, object] = {}

    class _PolicyStub:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(factory_mod, "AlpamayoPolicy", _PolicyStub)
    policy_factory = factory_mod.build_policy_factory(
        run_config=SimpleNamespace(policy=config),
        inference_engine=object(),
    )

    policy_factory("session-id", calibration=(), random_seed=77)

    assert captured["session_uuid"] == "session-id"
    assert captured["seed"] == 77


def test_alpamayo_policy_closest_to_previous_selector_picks_nearest_candidate() -> None:
    """`closest_to_previous` argmins mean L2-xyz vs. the reprojected previous trajectory.

    The previous tick's xyz forms a regular 8-step diagonal in the prev ego
    frame. Both ego poses are identity, so re-projection is a no-op and the
    interpolated previous trajectory over the overlap window equals
    ``prev.xyz[3:8]``. Sample 1's first 5 candidate waypoints match those
    values exactly while sample 0's are far away, so the selector must pick
    ``(set_ix=0, sample_ix=1)``.
    """
    horizon = 8
    num_traj_sets = 1
    num_traj_samples = 2
    step_dt_us = 1_000_000 // horizon

    pred_xyz = torch.zeros((num_traj_sets, num_traj_samples, horizon, 3), dtype=torch.float32)
    interp_overlap = torch.stack(
        [torch.tensor([float(k + 1), float(k + 1), 0.0]) for k in range(3, horizon)]
    )
    pred_xyz[0, 0, : interp_overlap.shape[0]] = torch.full((interp_overlap.shape[0], 3), 50.0)
    pred_xyz[0, 1, : interp_overlap.shape[0]] = interp_overlap
    pred_rot = torch.eye(3).expand(num_traj_sets, num_traj_samples, horizon, 3, 3).clone()
    logprob = torch.zeros((num_traj_sets, num_traj_samples), dtype=torch.float32)
    output = _FakeOutput(pred_xyz=pred_xyz, pred_rot=pred_rot, logprob=logprob)
    inference_model = _FakeInferenceModel(output)
    sampling = SamplingParamsConfig(
        top_p=0.98,
        top_k=None,
        temperature=0.6,
        num_traj_samples=num_traj_samples,
        num_traj_sets=num_traj_sets,
        max_generation_length=None,
    )
    config = _policy_config(
        selector_name=TrajectorySelectorKind.closest_to_previous,
        num_traj_samples=num_traj_samples,
        num_traj_sets=num_traj_sets,
        num_future_waypoints=horizon,
    )
    previous_traj = ChosenTrajectory(
        set_ix=0,
        sample_ix=0,
        xyz=torch.stack([torch.tensor([float(k + 1), float(k + 1), 0.0]) for k in range(horizon)]),
        rot=torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(horizon, 3, 3).clone(),
        dt_us=torch.arange(1, horizon + 1, dtype=torch.int64) * step_dt_us,
        time_now_us=1_000_000,
        ego_pose_at_choice=Pose(),
    )
    with driven_inference_engine(
        inference_model=inference_model, sampling=sampling, return_trace_for_rl=True
    ) as inference_engine:
        policy = AlpamayoPolicy(
            inference_engine=inference_engine,
            session_uuid="test-session",
            config=config,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        policy._buffers.update_last_chosen_traj(previous_traj)

        # Current tick lands 3 steps after the previous tick, so 5 current
        # waypoints overlap the previous horizon and the interpolated previous
        # trajectory at those timestamps equals ``previous_traj.xyz[3:8]``.
        result = policy.step(
            _policy_input(
                time_now_us=1_000_000 + 3 * step_dt_us,
                time_query_us=1_000_000 + 3 * step_dt_us,
            )
        )

    assert result.replay_data is not None
    assert result.replay_data.action_selection.set_ix == 0
    assert result.replay_data.action_selection.sample_ix == 1
    assert torch.equal(
        result.chosen_xyz,
        torch.cat([torch.zeros((1, 3), dtype=torch.float32), pred_xyz[0, 1]], dim=0),
    )


def _build_policy_for_frame_tests() -> AlpamayoPolicy:
    """Build a minimal AlpamayoPolicy for direct `_buffers` ingest in frame-conversion tests."""
    horizon = 1
    num_traj_sets = 1
    num_traj_samples = 1
    pred_xyz = torch.zeros((num_traj_sets, num_traj_samples, horizon, 3), dtype=torch.float32)
    pred_rot = torch.eye(3).expand(num_traj_sets, num_traj_samples, horizon, 3, 3).clone()
    output = _FakeOutput(pred_xyz=pred_xyz, pred_rot=pred_rot, logprob=None)
    inference_model = _FakeInferenceModel(output)
    sampling = SamplingParamsConfig(
        top_p=0.98,
        top_k=None,
        temperature=0.6,
        num_traj_samples=num_traj_samples,
        num_traj_sets=num_traj_sets,
        max_generation_length=None,
    )
    inference_engine = InferenceEngine(
        inference_model=inference_model,
        sampling=sampling,
        return_trace_for_rl=False,
        max_batch_size=1,
    )
    config = _policy_config(
        selector_name=TrajectorySelectorKind.identity,
        num_traj_samples=num_traj_samples,
        num_traj_sets=num_traj_sets,
        num_future_waypoints=horizon,
    )
    return AlpamayoPolicy(
        inference_engine=inference_engine,
        session_uuid="test-session",
        config=config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _frame_test_policy_input(
    time_now_us: int,
    ego_poses: tuple[EgoPose, ...],
    route_waypoints: tuple[RouteWaypoint, ...] | None = None,
    route_timestamp_us: int | None = None,
) -> PolicyInput:
    """Build a camera-free `PolicyInput` for direct frame-conversion tests."""
    if route_waypoints is None:
        route_waypoints = _pad_route_to_contract((RouteWaypoint(x=0.0, y=0.0),))
    if route_timestamp_us is None:
        route_timestamp_us = time_now_us
    return PolicyInput(
        step_index=0,
        time_now_us=time_now_us,
        time_query_us=time_now_us + 200_000,
        camera_images=(),
        ego_trajectory=Trajectory(poses=ego_poses),
        route_waypoints=route_waypoints,
        route_timestamp_us=route_timestamp_us,
        calibration=_calibration_for(CAMERA_ID),
    )


def test_extract_historical_motion_normalizes_to_ego_t0_frame() -> None:
    """The latest history pose maps to (xyz=0, rot=I) after frame normalization."""
    policy = _build_policy_for_frame_tests()
    yaw_quat = Quaternion(w=0.7071068, x=0.0, y=0.0, z=0.7071068)  # 90 deg around +z
    policy._buffers.ingest(
        _frame_test_policy_input(
            time_now_us=2_000_000,
            ego_poses=(
                EgoPose(timestamp_us=1_000_000, pose=Pose(vec=Vec3(x=5.0))),
                EgoPose(timestamp_us=2_000_000, pose=Pose(vec=Vec3(x=10.0), quat=yaw_quat)),
            ),
        )
    )

    history_rig_xyz, history_rig_rot = policy._extract_historical_motion()

    assert torch.allclose(history_rig_xyz[0, -1], torch.zeros(3, dtype=torch.float32), atol=1e-5)
    assert torch.allclose(history_rig_rot[0, -1], torch.eye(3, dtype=torch.float32), atol=1e-5)


def test_extract_historical_motion_rejects_short_ego_history() -> None:
    """Buffer shorter than ``num_historical_waypoints`` fails fast.

    A short warmup must raise rather than fabricate fake poses to fill
    the buffer; the caller is expected to size ``force_gt_duration_us``
    so the buffer is long enough at the first inference tick.
    """
    policy = _build_policy_for_frame_tests()
    policy._buffers.ingest(
        _frame_test_policy_input(
            time_now_us=1_000_000,
            ego_poses=(EgoPose(timestamp_us=1_000_000, pose=Pose()),),
        )
    )

    with pytest.raises(ValueError, match="need at least"):
        policy._extract_historical_motion()


def test_convert_route_reprojects_waypoints_into_rig_t0_frame() -> None:
    """Rig-frame waypoints at `route_ts` re-project through a non-identity rel pose.

    The first waypoint becomes `(0, 9)` in the rig-t0 frame.
    """
    policy = _build_policy_for_frame_tests()
    yaw_quat = Quaternion(w=0.7071068, x=0.0, y=0.0, z=0.7071068)  # 90 deg around +z
    route_ts = 500_000
    rig_t0_ts = 1_000_000
    policy._buffers.ingest(
        _frame_test_policy_input(
            time_now_us=rig_t0_ts,
            ego_poses=(
                EgoPose(timestamp_us=route_ts, pose=Pose()),
                EgoPose(
                    timestamp_us=rig_t0_ts,
                    pose=Pose(vec=Vec3(x=10.0), quat=yaw_quat),
                ),
            ),
            route_waypoints=_pad_route_to_contract(
                (RouteWaypoint(x=1.0, y=0.0), RouteWaypoint(x=2.0, y=3.0))
            ),
            route_timestamp_us=route_ts,
        )
    )

    route_xy = policy._convert_route()

    assert route_xy.shape == (NUM_ROUTE_WAYPOINTS, 2)
    # World y → rig-t0 x (after the -90deg yaw inverse); world -x → rig-t0 y.
    # Waypoint (1, 0) at rig-route_ts is world (1, 0); relative to rig-t0
    # (10, 0) it's world (-9, 0), which the inverse yaw maps to (0, 9).
    assert torch.allclose(route_xy[0], torch.tensor([0.0, 9.0]), atol=1e-5)
    # Waypoint (2, 3) → world (2, 3) → world relative (-8, 3) → (3, 8).
    assert torch.allclose(route_xy[1], torch.tensor([3.0, 8.0]), atol=1e-5)
    assert torch.isnan(route_xy[2:]).all()


def test_convert_route_interpolates_pose_at_route_timestamp() -> None:
    """Route conversion uses the ego pose interpolated at the route timestamp."""
    policy = _build_policy_for_frame_tests()
    policy._buffers.ingest(
        _frame_test_policy_input(
            time_now_us=2_000_000,
            ego_poses=(
                EgoPose(timestamp_us=0, pose=Pose(vec=Vec3(x=0.0))),
                EgoPose(timestamp_us=1_000_000, pose=Pose(vec=Vec3(x=10.0))),
                EgoPose(timestamp_us=2_000_000, pose=Pose(vec=Vec3(x=20.0))),
            ),
            route_waypoints=_pad_route_to_contract((RouteWaypoint(x=1.0, y=2.0),)),
            route_timestamp_us=500_000,
        )
    )

    route_xy = policy._convert_route()

    assert route_xy.shape == (NUM_ROUTE_WAYPOINTS, 2)
    assert torch.allclose(route_xy[0], torch.tensor([-14.0, 2.0]), atol=1e-5)
    assert torch.isnan(route_xy[1:]).all()


def test_convert_route_rejects_unexpected_waypoint_count() -> None:
    """AlpaSim must always submit exactly `NUM_ROUTE_WAYPOINTS` waypoints."""
    policy = _build_policy_for_frame_tests()
    policy._buffers.ingest(
        _frame_test_policy_input(
            time_now_us=1_000_000,
            ego_poses=(
                EgoPose(timestamp_us=500_000, pose=Pose()),
                EgoPose(timestamp_us=1_000_000, pose=Pose()),
            ),
            route_waypoints=(RouteWaypoint(x=1.0, y=2.0),),
            route_timestamp_us=1_000_000,
        )
    )

    with pytest.raises(ValueError, match="route waypoints from AlpaSim"):
        policy._convert_route()
