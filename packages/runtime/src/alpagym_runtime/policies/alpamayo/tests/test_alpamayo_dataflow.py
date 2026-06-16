# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Alpamayo observation buffering."""

import io

import pytest
import torch
from alpagym_host.config import (
    AlpamayoPolicyConfig,
    InferenceConfig,
    ModelConfig,
    SamplingParamsConfig,
    TrajectorySelectorKind,
)
from alpagym_runtime.alpasim.cameras import CAMERA_NAMES_TO_INDICES
from alpagym_runtime.inference.inference_engine import InferenceEngine
from alpagym_runtime.policies.alpamayo.buffers import ObservationBuffers
from alpagym_runtime.policies.alpamayo.policy import AlpamayoPolicy
from alpagym_runtime.types import (
    CameraImage,
    EgoPose,
    PolicyInput,
    Pose,
    RouteWaypoint,
    Trajectory,
    Vec3,
)
from PIL import Image


def _make_jpeg(width: int = 8, height: int = 6, color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    """Encode a single-color RGB image as JPEG bytes for the buffer tests."""
    image = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def _ingest_camera_tick(
    buffers: ObservationBuffers,
    camera_id: str,
    frame_end_us: int,
    color: tuple[int, int, int],
) -> None:
    """Ingest one camera frame into `buffers`."""
    buffers.ingest(
        PolicyInput(
            step_index=0,
            time_now_us=frame_end_us,
            time_query_us=frame_end_us,
            camera_images=(
                CameraImage(
                    logical_id=camera_id,
                    image_bytes=_make_jpeg(color=color),
                    frame_end_us=frame_end_us,
                ),
            ),
            ego_trajectory=Trajectory(),
            route_waypoints=(RouteWaypoint(x=0.0, y=0.0),),
            route_timestamp_us=frame_end_us,
            calibration=(),
        )
    )


def _build_policy_for_camera_context_tests(use_cameras: tuple[str, ...]) -> AlpamayoPolicy:
    """Build a policy shell for testing camera-context extraction."""
    sampling = SamplingParamsConfig(
        top_p=1.0,
        top_k=None,
        temperature=1.0,
        num_traj_samples=1,
        num_traj_sets=1,
        max_generation_length=None,
    )
    config = AlpamayoPolicyConfig(
        kind="alpamayo",
        model=ModelConfig(
            kind="alpamayo_r1",
            path="unused-by-tests",
            device="cpu",
            dtype="float32",
            use_cameras=list(use_cameras),
            num_context_frames=2,
            num_historical_waypoints=0,
            num_future_waypoints=1,
            step_dt_us=1_000_000,
            input_size=[6, 8],
        ),
        inference=InferenceConfig(
            max_batch_size=1,
            return_trace_for_rl=True,
            sampling=sampling,
        ),
        trajectory_selector=TrajectorySelectorKind.identity,
    )
    return AlpamayoPolicy(
        inference_engine=InferenceEngine.__new__(InferenceEngine),
        session_uuid="camera-context-test",
        config=config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_camera_context_is_cam_major_with_t0_relative_timestamps() -> None:
    """Camera context preserves cam-major order with t0-relative timestamps."""
    cameras = ("camera_front_wide_120fov", "camera_front_tele_30fov")
    policy = _build_policy_for_camera_context_tests(use_cameras=cameras)
    buffers = policy._buffers

    _ingest_camera_tick(buffers, camera_id=cameras[0], frame_end_us=100, color=(10, 0, 0))
    _ingest_camera_tick(buffers, camera_id=cameras[0], frame_end_us=200, color=(20, 0, 0))
    _ingest_camera_tick(buffers, camera_id=cameras[1], frame_end_us=150, color=(0, 30, 0))
    _ingest_camera_tick(buffers, camera_id=cameras[1], frame_end_us=250, color=(0, 40, 0))

    frames, indices, tstamps = policy._extract_camera_context()

    assert indices.tolist() == [
        CAMERA_NAMES_TO_INDICES[cameras[0]],
        CAMERA_NAMES_TO_INDICES[cameras[0]],
        CAMERA_NAMES_TO_INDICES[cameras[1]],
        CAMERA_NAMES_TO_INDICES[cameras[1]],
    ]
    assert tstamps.tolist() == [-150, -50, -100, 0]
    assert frames.shape[0] == len(indices)


def test_observation_buffers_accumulate_ego_history_across_ticks() -> None:
    """Ego observations append across drive ticks and retain the latest window."""
    buffers = ObservationBuffers(
        frame_buffer_capacity=1,
        ego_history_capacity=3,
        use_cameras=(),
        frame_size=(6, 8),
    )

    for timestamp_us in (100, 200, 300, 400):
        buffers.ingest(
            PolicyInput(
                step_index=0,
                time_now_us=timestamp_us,
                time_query_us=timestamp_us,
                camera_images=(),
                ego_trajectory=Trajectory(
                    poses=(
                        EgoPose(
                            timestamp_us=timestamp_us,
                            pose=Pose(vec=Vec3(x=float(timestamp_us))),
                        ),
                    )
                ),
                route_waypoints=(RouteWaypoint(x=0.0, y=0.0),),
                route_timestamp_us=timestamp_us,
                calibration=(),
            )
        )

    assert [pose.timestamp_us for pose in buffers.ego_history] == [200, 300, 400]


def test_observation_buffers_reject_out_of_order_ego_history() -> None:
    """Ego history must arrive in strictly increasing timestamp order."""
    buffers = ObservationBuffers(
        frame_buffer_capacity=1,
        ego_history_capacity=3,
        use_cameras=(),
        frame_size=(6, 8),
    )

    buffers.ingest(
        PolicyInput(
            step_index=0,
            time_now_us=300,
            time_query_us=300,
            camera_images=(),
            ego_trajectory=Trajectory(
                poses=(
                    EgoPose(timestamp_us=100, pose=Pose(vec=Vec3(x=1.0))),
                    EgoPose(timestamp_us=300, pose=Pose(vec=Vec3(x=3.0))),
                )
            ),
            route_waypoints=(RouteWaypoint(x=0.0, y=0.0),),
            route_timestamp_us=300,
            calibration=(),
        )
    )

    with pytest.raises(ValueError, match="Ego pose timestamps must increase"):
        buffers.ingest(
            PolicyInput(
                step_index=1,
                time_now_us=400,
                time_query_us=400,
                camera_images=(),
                ego_trajectory=Trajectory(
                    poses=(
                        EgoPose(timestamp_us=400, pose=Pose(vec=Vec3(x=4.0))),
                        EgoPose(timestamp_us=200, pose=Pose(vec=Vec3(x=2.0))),
                    )
                ),
                route_waypoints=(RouteWaypoint(x=0.0, y=0.0),),
                route_timestamp_us=400,
                calibration=(),
            )
        )
