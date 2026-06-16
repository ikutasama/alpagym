# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build AlpaSim-ready policy trajectories."""

import torch

from alpagym_runtime.types import Pose

from .geometry_utils import quat_tensor_to_rotation_matrix, rotation_matrix_to_quat


def build_policy_output_trajectory(
    ego_pose_now: Pose,
    future_xyz_ego: torch.Tensor,
    future_rot_ego: torch.Tensor,
    future_dt_us: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepend the current pose and convert future rows to world frame."""
    device = future_xyz_ego.device

    base_xyz_64 = torch.tensor(
        [ego_pose_now.vec.x, ego_pose_now.vec.y, ego_pose_now.vec.z],
        dtype=torch.float64,
        device=device,
    )
    base_quat_64 = torch.tensor(
        [
            ego_pose_now.quat.w,
            ego_pose_now.quat.x,
            ego_pose_now.quat.y,
            ego_pose_now.quat.z,
        ],
        dtype=torch.float64,
        device=device,
    )
    base_quat_64 = base_quat_64 / torch.linalg.vector_norm(base_quat_64).clamp_min(1e-12)
    base_rot_64 = quat_tensor_to_rotation_matrix(base_quat_64)

    future_xyz_64 = future_xyz_ego.to(dtype=torch.float64)
    future_rot_64 = future_rot_ego.to(dtype=torch.float64)

    future_xyz_world_64 = future_xyz_64 @ base_rot_64.T + base_xyz_64
    future_rot_world_64 = base_rot_64 @ future_rot_64

    future_quat_world_64 = rotation_matrix_to_quat(future_rot_world_64)
    future_quat_world_64 = future_quat_world_64 / torch.linalg.vector_norm(
        future_quat_world_64, dim=-1, keepdim=True
    ).clamp_min(1e-12)

    chosen_xyz = torch.cat([base_xyz_64.unsqueeze(0), future_xyz_world_64], dim=0).to(
        dtype=torch.float32
    )
    chosen_quat = torch.cat([base_quat_64.unsqueeze(0), future_quat_world_64], dim=0).to(
        dtype=torch.float32
    )
    chosen_dt_us = torch.cat(
        [
            torch.zeros(1, dtype=future_dt_us.dtype, device=future_dt_us.device),
            future_dt_us,
        ],
        dim=0,
    )
    return chosen_xyz, chosen_quat, chosen_dt_us
