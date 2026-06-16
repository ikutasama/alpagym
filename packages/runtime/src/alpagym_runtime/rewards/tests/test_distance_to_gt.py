# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for `compute_distance_to_gt_reward`."""

import math

import numpy as np
import pytest
from alpagym_runtime.rewards.distance_to_gt import compute_distance_to_gt_reward
from alpagym_runtime.types import (
    EgoPose,
    EpisodeOutput,
    GroundTruth,
    Pose,
    Quaternion,
    Trajectory,
    Vec3,
)


def _yaw_quat(yaw: float) -> Quaternion:
    """Build a yaw-only quaternion (ENU Z-rotation)."""
    return Quaternion(w=math.cos(yaw / 2.0), x=0.0, y=0.0, z=math.sin(yaw / 2.0))


def _euler_quat(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Build a quaternion from roll, pitch, and yaw."""
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    return Quaternion(
        w=cr * cp * cy + sr * sp * sy,
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
    )


def _ego_pose(timestamp_us: int, x: float, y: float, yaw: float = 0.0) -> EgoPose:
    """Build an `EgoPose` at `timestamp_us` with the given XY position and yaw."""
    return EgoPose(
        timestamp_us=timestamp_us,
        pose=Pose(vec=Vec3(x=x, y=y, z=0.0), quat=_yaw_quat(yaw)),
    )


def _inputs(
    executed_local_poses: tuple[EgoPose, ...],
    gt_rig_poses: tuple[EgoPose, ...] | None,
    gt_anchor_us: int = 0,
) -> tuple[EpisodeOutput, GroundTruth | None]:
    """Build a minimal `(EpisodeOutput, GroundTruth | None)` pair."""
    ground_truth = (
        None
        if gt_rig_poses is None
        else GroundTruth(
            ego_trajectory=Trajectory(poses=gt_rig_poses),
            timestamp_us=gt_anchor_us,
        )
    )
    episode = EpisodeOutput(
        scene_id="scene",
        session_uuid="session",
        num_steps=0,
        policy_outputs=(),
        executed_ego_trajectory=Trajectory(poses=executed_local_poses),
    )
    return episode, ground_truth


def _rig_to_local(
    anchor_xy: tuple[float, float], anchor_yaw: float, rig_xy: tuple[float, float]
) -> tuple[float, float]:
    """Active `local` <- `rig@anchor` transform for one XY point."""
    cos_y = math.cos(anchor_yaw)
    sin_y = math.sin(anchor_yaw)
    rx, ry = rig_xy
    return (cos_y * rx - sin_y * ry + anchor_xy[0], sin_y * rx + cos_y * ry + anchor_xy[1])


def test_perfect_match_in_local_yields_zero_rmse() -> None:
    """An executed trajectory that exactly matches GT (after frame transform) scores 0."""
    gt_anchor_us = 0
    anchor_xy = (10.0, -3.0)
    anchor_yaw = 0.4
    gt_rig = tuple(_ego_pose(t, x=float(t), y=0.0) for t in (0, 100, 200, 300))
    executed = tuple(
        EgoPose(
            timestamp_us=pose.timestamp_us,
            pose=Pose(
                vec=Vec3(
                    *_rig_to_local(
                        anchor_xy=anchor_xy,
                        anchor_yaw=anchor_yaw,
                        rig_xy=(pose.pose.vec.x, pose.pose.vec.y),
                    ),
                    0.0,
                ),
                quat=_yaw_quat(anchor_yaw),
            ),
        )
        for pose in gt_rig
    )

    episode, ground_truth = _inputs(executed, gt_rig, gt_anchor_us=gt_anchor_us)
    result = compute_distance_to_gt_reward(episode, ground_truth, distance_scale=-2.0)

    assert result.report_metrics["gt_rmse_xy"] == pytest.approx(0.0, abs=1e-9)
    assert result.report_metrics["gt_num_paired"] == pytest.approx(4.0)
    assert result.report_metrics["gt_anchor_us"] == pytest.approx(float(gt_anchor_us))
    assert result.total == pytest.approx(0.0, abs=1e-9)


def test_coarse_vs_fine_interpolation_aligns() -> None:
    """A finely sampled executed trajectory along the GT path scores ~0 via interpolation."""
    gt_rig = tuple(_ego_pose(t * 1_000_000, x=float(t), y=0.0) for t in range(6))
    executed = tuple(
        _ego_pose(t_us, x=float(t_us) / 1_000_000.0, y=0.0) for t_us in range(0, 5_000_001, 100_000)
    )

    episode, ground_truth = _inputs(executed, gt_rig, gt_anchor_us=0)
    result = compute_distance_to_gt_reward(episode, ground_truth, distance_scale=1.0)

    assert result.report_metrics["gt_rmse_xy"] == pytest.approx(0.0, abs=1e-9)
    assert result.report_metrics["gt_num_paired"] == pytest.approx(float(len(executed)))


def test_yaw_wrap_interpolation_uses_cos_sin() -> None:
    """A yaw spanning ±π interpolates through the short way."""
    eps = 1e-2
    executed = (
        _ego_pose(0, x=0.0, y=0.0, yaw=math.pi - eps),
        _ego_pose(2_000_000, x=0.0, y=0.0, yaw=-math.pi + eps),
    )
    gt_rig = (
        _ego_pose(0, x=1.0, y=0.0),
        _ego_pose(1_000_000, x=1.0, y=0.0),
        _ego_pose(2_000_000, x=1.0, y=0.0),
    )

    episode, ground_truth = _inputs(executed, gt_rig, gt_anchor_us=1_000_000)

    cos_a = math.cos(math.pi - eps)
    sin_a = math.sin(math.pi - eps)
    cos_b = math.cos(-math.pi + eps)
    sin_b = math.sin(-math.pi + eps)
    cos_interp = 0.5 * (cos_a + cos_b)
    sin_interp = 0.5 * (sin_a + sin_b)
    yaw_anchor = math.atan2(sin_interp, cos_interp)
    expected_gt_local_xy = np.array(
        [_rig_to_local(anchor_xy=(0.0, 0.0), anchor_yaw=yaw_anchor, rig_xy=(1.0, 0.0))]
    )
    # The transformed GT should be ~ (-1, 0) (yaw is close to ±π).
    assert expected_gt_local_xy[0, 0] < 0.0

    result = compute_distance_to_gt_reward(episode, ground_truth, distance_scale=1.0)

    expected_rmse = math.sqrt(
        float(expected_gt_local_xy[0, 0] ** 2 + expected_gt_local_xy[0, 1] ** 2)
    )
    assert result.report_metrics["gt_rmse_xy"] == pytest.approx(expected_rmse, abs=1e-6)


def test_executed_points_outside_gt_span_are_dropped() -> None:
    """Executed poses outside the GT timestamp range do not contribute to RMSE."""
    gt_rig = (
        _ego_pose(100, x=0.0, y=0.0),
        _ego_pose(200, x=0.0, y=0.0),
    )
    executed = (
        _ego_pose(50, x=-3.0, y=-4.0),
        _ego_pose(150, x=3.0, y=4.0),
        _ego_pose(250, x=999.0, y=999.0),
    )

    episode, ground_truth = _inputs(executed, gt_rig, gt_anchor_us=100)
    result = compute_distance_to_gt_reward(episode, ground_truth, distance_scale=-0.01)

    assert result.report_metrics["gt_num_paired"] == pytest.approx(1.0)
    assert result.report_metrics["gt_rmse_xy"] == pytest.approx(5.0)


def test_tilted_executed_orientation_uses_yaw() -> None:
    """Tilted simulator poses still provide usable yaw for the planar reward."""
    yaw = 0.4
    executed = (
        EgoPose(
            timestamp_us=0,
            pose=Pose(
                vec=Vec3(x=10.0, y=-3.0, z=0.0),
                quat=_euler_quat(roll=-0.017, pitch=-0.054, yaw=yaw),
            ),
        ),
        EgoPose(
            timestamp_us=100,
            pose=Pose(
                vec=Vec3(
                    *_rig_to_local((10.0, -3.0), yaw, (1.0, 0.0)),
                    0.0,
                ),
                quat=_euler_quat(roll=-0.017, pitch=-0.054, yaw=yaw),
            ),
        ),
    )
    gt_rig = (_ego_pose(0, x=0.0, y=0.0), _ego_pose(100, x=1.0, y=0.0))
    episode, ground_truth = _inputs(executed, gt_rig, gt_anchor_us=0)
    result = compute_distance_to_gt_reward(episode, ground_truth, distance_scale=-1.0)

    assert result.report_metrics["gt_rmse_xy"] == pytest.approx(0.0, abs=1e-9)
