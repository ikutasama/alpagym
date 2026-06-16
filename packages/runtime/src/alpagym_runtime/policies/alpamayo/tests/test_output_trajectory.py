# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Alpamayo policy-output trajectory construction."""

import torch
from alpagym_runtime.policies.alpamayo.output_trajectory import build_policy_output_trajectory
from alpagym_runtime.types import Pose, Quaternion, Vec3


def test_build_policy_output_trajectory_prepends_current_pose_and_composes_future_rows() -> None:
    """Policy output starts at now and converts rig-frame future rows to world frame."""
    chosen_xyz, chosen_quat, chosen_dt_us = build_policy_output_trajectory(
        ego_pose_now=Pose(
            vec=Vec3(x=10.0, y=20.0, z=30.0),
            quat=Quaternion(w=0.70710678, x=0.0, y=0.0, z=0.70710678),
        ),
        future_xyz_ego=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        future_rot_ego=torch.eye(3, dtype=torch.float32).unsqueeze(0),
        future_dt_us=torch.tensor([100_000], dtype=torch.int64),
    )

    torch.testing.assert_close(
        chosen_xyz,
        torch.tensor([[10.0, 20.0, 30.0], [10.0, 21.0, 30.0]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        chosen_quat,
        torch.tensor(
            [
                [0.70710678, 0.0, 0.0, 0.70710678],
                [0.70710678, 0.0, 0.0, 0.70710678],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(chosen_dt_us, torch.tensor([0, 100_000], dtype=torch.int64))


def test_build_policy_output_trajectory_emits_unit_norm_quaternions_from_noisy_rotations() -> None:
    """Slightly non-orthonormal model rotations still yield unit-norm quats."""
    horizon = 6
    base_rot = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(horizon, 3, 3).clone()
    # Inject a small non-orthogonal perturbation typical of model output in fp32.
    base_rot += torch.full_like(base_rot, 5e-5)
    _, chosen_quat, _ = build_policy_output_trajectory(
        ego_pose_now=Pose(
            vec=Vec3(x=0.0, y=0.0, z=0.0),
            quat=Quaternion(w=0.70710678, x=0.0, y=0.0, z=0.70710678),
        ),
        future_xyz_ego=torch.zeros((horizon, 3), dtype=torch.float32),
        future_rot_ego=base_rot,
        future_dt_us=torch.arange(1, horizon + 1, dtype=torch.int64) * 100_000,
    )
    lengths = torch.linalg.vector_norm(chosen_quat.to(torch.float64), dim=-1)
    assert torch.all((lengths - 1.0).abs() < 1e-6), (
        f"chosen_quat not unit-norm: max |q|-1 = {(lengths - 1.0).abs().max().item():.3e}"
    )
