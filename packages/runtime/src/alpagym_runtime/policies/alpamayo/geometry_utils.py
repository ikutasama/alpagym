# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Quaternion, rotation, and ego-frame coordinate helpers."""

import torch

from alpagym_runtime.types import Pose, Quaternion


def quaternion_to_rotation_matrix(
    quat: Quaternion,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert a wxyz unit quaternion to a `[3, 3]` rotation matrix."""
    w, x, y, z = quat.w, quat.x, quat.y, quat.z
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=dtype,
        device=device,
    )


def quat_tensor_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert a wxyz unit-quaternion tensor to a `[..., 3, 3]` rotation matrix."""
    w, x, y, z = quat.unbind(dim=-1)
    row0 = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
        ]
    )
    row1 = torch.stack(
        [
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
        ]
    )
    row2 = torch.stack(
        [
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ]
    )
    return torch.stack([row0, row1, row2], dim=0)


def rotation_matrix_to_quat(rot: torch.Tensor) -> torch.Tensor:
    """Convert `[T, 3, 3]` rotation matrices into `[T, 4]` wxyz quaternions."""
    horizon = rot.shape[0]
    quat = torch.zeros((horizon, 4), dtype=rot.dtype, device=rot.device)
    for i in range(horizon):
        m = rot[i]
        m00, m01, m02 = m[0, 0], m[0, 1], m[0, 2]
        m10, m11, m12 = m[1, 0], m[1, 1], m[1, 2]
        m20, m21, m22 = m[2, 0], m[2, 1], m[2, 2]
        trace = m00 + m11 + m22
        if trace > 0:
            s = 2.0 * torch.sqrt(trace + 1.0)
            quat[i, 0] = 0.25 * s
            quat[i, 1] = (m21 - m12) / s
            quat[i, 2] = (m02 - m20) / s
            quat[i, 3] = (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            s = 2.0 * torch.sqrt(1.0 + m00 - m11 - m22)
            quat[i, 0] = (m21 - m12) / s
            quat[i, 1] = 0.25 * s
            quat[i, 2] = (m01 + m10) / s
            quat[i, 3] = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * torch.sqrt(1.0 + m11 - m00 - m22)
            quat[i, 0] = (m02 - m20) / s
            quat[i, 1] = (m01 + m10) / s
            quat[i, 2] = 0.25 * s
            quat[i, 3] = (m12 + m21) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + m22 - m00 - m11)
            quat[i, 0] = (m10 - m01) / s
            quat[i, 1] = (m02 + m20) / s
            quat[i, 2] = (m12 + m21) / s
            quat[i, 3] = 0.25 * s
        if quat[i, 0] < 0:
            quat[i] = -quat[i]
    return quat


def pred_rot_to_quat(pred_rot: torch.Tensor) -> torch.Tensor:
    """Convert predicted rotation matrices to wxyz quaternions."""
    num_sets, num_samples, horizon, _, _ = pred_rot.shape
    flat = pred_rot.reshape(num_sets * num_samples * horizon, 3, 3)
    quat_flat = rotation_matrix_to_quat(flat)
    return quat_flat.reshape(num_sets, num_samples, horizon, 4)


def local_to_world(
    points: torch.Tensor,
    pose: Pose,
) -> torch.Tensor:
    """Map points from `pose`'s local frame into the world frame.

    Args:
        points: `[..., 3]` points in `pose`'s local frame.
        pose: Pose whose rotation maps local-frame vectors into world; its
            translation is ``pose.vec``.

    Returns:
        `[..., 3]` world-frame points equal to ``points @ R.T + t``, where
        ``R`` is ``pose.quat``'s rotation matrix and ``t`` is ``pose.vec``.
        Dtype and device match `points`.
    """
    rot = quaternion_to_rotation_matrix(pose.quat, dtype=points.dtype, device=points.device)
    t = _pose_translation_tensor(pose, dtype=points.dtype, device=points.device)
    return points @ rot.T + t


def world_to_local(
    points: torch.Tensor,
    pose: Pose,
) -> torch.Tensor:
    """Inverse of `local_to_world`.

    Args:
        points: `[..., 3]` points in the world frame.
        pose: Pose defining the target local frame.

    Returns:
        `[..., 3]` points in `pose`'s local frame; dtype and device match
        `points`.
    """
    rot = quaternion_to_rotation_matrix(pose.quat, dtype=points.dtype, device=points.device)
    t = _pose_translation_tensor(pose, dtype=points.dtype, device=points.device)
    return (points - t) @ rot


def _pose_translation_tensor(
    pose: Pose,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a `[3]` translation tensor from ``pose.vec``."""
    return torch.tensor(
        [pose.vec.x, pose.vec.y, pose.vec.z],
        dtype=dtype,
        device=device,
    )


def linear_interp_xyz(
    src_tstamps: torch.Tensor,
    src_xyz: torch.Tensor,
    query_tstamps: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate `src_xyz` at `query_tstamps`.

    Args:
        src_tstamps: Strictly-increasing `[N]` source timeline.
        src_xyz: `[N, 3]` source values aligned with `src_tstamps`.
        query_tstamps: `[M]` timestamps to interpolate at. Queries outside
            `[src_tstamps[0], src_tstamps[-1]]` are extrapolated from the
            boundary bracket; callers that need exact coverage must filter
            first.

    Returns:
        `[M, 3]` interpolated values at `query_tstamps`.
    """
    n = int(src_tstamps.shape[0])
    if n < 2:
        raise ValueError(f"need at least 2 source points to interpolate; got {n}")
    idx_right = torch.searchsorted(src_tstamps, query_tstamps).clamp(min=1, max=n - 1)
    idx_left = idx_right - 1
    t_left = src_tstamps[idx_left].to(dtype=torch.float64)
    t_right = src_tstamps[idx_right].to(dtype=torch.float64)
    span = (t_right - t_left).clamp_min(1.0)
    alpha = ((query_tstamps.to(dtype=torch.float64) - t_left) / span).to(dtype=src_xyz.dtype)
    return src_xyz[idx_left] + (src_xyz[idx_right] - src_xyz[idx_left]) * alpha.unsqueeze(-1)
