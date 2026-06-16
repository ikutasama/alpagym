# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reward comparing executed rollout poses to ground truth."""

import math

import numpy as np
from alpagym_runtime.types import EgoPose, EpisodeOutput, GroundTruth, RewardResult


def compute_distance_to_gt_reward(
    episode: EpisodeOutput,
    ground_truth: GroundTruth | None,
    distance_scale: float,
) -> RewardResult:
    """Compute XY RMSE between executed poses and ground truth."""
    if ground_truth is None:
        raise ValueError("compute_distance_to_gt_reward requires ground_truth")

    executed_poses = episode.executed_ego_trajectory.poses
    gt_poses = ground_truth.ego_trajectory.poses
    if len(executed_poses) < 2:
        raise ValueError("compute_distance_to_gt_reward requires at least two executed ego poses")
    if len(gt_poses) < 2:
        raise ValueError(
            "compute_distance_to_gt_reward requires at least two ground-truth ego poses"
        )

    gt_anchor_us = int(ground_truth.timestamp_us)

    exec_ts, exec_xy, exec_yaw = _executed_arrays(executed_poses)
    gt_ts, gt_xy_rig = _gt_arrays(gt_poses)

    anchor_xy_local, anchor_yaw_local = _interp_pose_at(
        query_us=gt_anchor_us,
        ts=exec_ts,
        xy=exec_xy,
        yaw=exec_yaw,
    )

    cos_yaw = math.cos(anchor_yaw_local)
    sin_yaw = math.sin(anchor_yaw_local)
    rotated = np.column_stack(
        [
            cos_yaw * gt_xy_rig[:, 0] - sin_yaw * gt_xy_rig[:, 1],
            sin_yaw * gt_xy_rig[:, 0] + cos_yaw * gt_xy_rig[:, 1],
        ]
    )
    gt_xy_in_local = rotated + np.asarray(anchor_xy_local, dtype=np.float64)

    in_span = (exec_ts >= gt_ts[0]) & (exec_ts <= gt_ts[-1])
    if not in_span.any():
        raise ValueError(
            "compute_distance_to_gt_reward found no executed points inside the GT span"
        )
    query_ts = exec_ts[in_span]
    paired_exec_xy = exec_xy[in_span]
    gt_xy_at_exec = np.column_stack(
        [np.interp(query_ts, gt_ts, gt_xy_in_local[:, axis]) for axis in range(2)]
    )

    squared_errors = np.sum((paired_exec_xy - gt_xy_at_exec) ** 2, axis=1)
    rmse = float(math.sqrt(squared_errors.mean()))
    return RewardResult(
        total=distance_scale * rmse,
        report_metrics={
            "gt_rmse_xy": rmse,
            "gt_num_paired": float(squared_errors.size),
            "gt_anchor_us": float(gt_anchor_us),
        },
    )


def _executed_arrays(
    executed_poses: tuple[EgoPose, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return strictly-monotone `(timestamps_us, xy, yaw)` arrays."""
    timestamps = np.array([pose.timestamp_us for pose in executed_poses], dtype=np.int64)
    xy = np.array(
        [(pose.pose.vec.x, pose.pose.vec.y) for pose in executed_poses],
        dtype=np.float64,
    )
    yaw = np.array([_yaw_from_pose(pose) for pose in executed_poses], dtype=np.float64)

    if not np.all(np.diff(timestamps) > 0):
        raise ValueError(
            "compute_distance_to_gt_reward requires strictly increasing executed timestamps; "
            f"got {timestamps.tolist()!r}."
        )
    return timestamps, xy, yaw


def _gt_arrays(gt_poses: tuple[EgoPose, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return `(timestamps_us, xy)` arrays sorted by timestamp."""
    timestamps = np.array([pose.timestamp_us for pose in gt_poses], dtype=np.int64)
    xy = np.array(
        [(pose.pose.vec.x, pose.pose.vec.y) for pose in gt_poses],
        dtype=np.float64,
    )
    order = np.argsort(timestamps, kind="stable")
    return timestamps[order], xy[order]


def _interp_pose_at(
    query_us: int,
    ts: np.ndarray,
    xy: np.ndarray,
    yaw: np.ndarray,
) -> tuple[tuple[float, float], float]:
    """Interpolate `xy` and yaw at `query_us`."""
    query_float = float(query_us)
    if query_float < ts[0] or query_float > ts[-1]:
        raise ValueError(
            f"GT anchor timestamp {query_us} is outside the executed trajectory range "
            f"[{int(ts[0])}, {int(ts[-1])}]; cannot interpolate."
        )
    x = float(np.interp(query_float, ts, xy[:, 0]))
    y = float(np.interp(query_float, ts, xy[:, 1]))
    cos_interp = float(np.interp(query_float, ts, np.cos(yaw)))
    sin_interp = float(np.interp(query_float, ts, np.sin(yaw)))
    yaw_interp = math.atan2(sin_interp, cos_interp)
    return (x, y), yaw_interp


def _yaw_from_pose(pose: EgoPose) -> float:
    """Return the ENU yaw encoded in `pose.pose.quat`."""
    quat = pose.pose.quat
    w = float(quat.w)
    x = float(quat.x)
    y = float(quat.y)
    z = float(quat.z)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
