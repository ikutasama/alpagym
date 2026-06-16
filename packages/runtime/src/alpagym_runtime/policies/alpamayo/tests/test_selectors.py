# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Alpamayo trajectory selectors."""

from typing import Any

import pytest
import torch
from alpagym_runtime.inference.types import ModelOutput
from alpagym_runtime.policies.alpamayo.selectors import ClosestToPreviousSelector, IdentitySelector
from alpagym_runtime.types import ChosenTrajectory, Pose


def _trivial_model_output(
    num_traj_sets: int = 1, num_traj_samples: int = 1, horizon: int = 3
) -> ModelOutput:
    """Build a ModelOutput with deterministic xyz and identity rotations."""
    pred_xyz = torch.zeros((num_traj_sets, num_traj_samples, horizon, 3))
    for set_ix in range(num_traj_sets):
        for sample_ix in range(num_traj_samples):
            for waypoint_ix in range(horizon):
                value = float(set_ix * 10 + sample_ix * 5 + waypoint_ix)
                pred_xyz[set_ix, sample_ix, waypoint_ix, 0] = value
                pred_xyz[set_ix, sample_ix, waypoint_ix, 1] = value * 0.5
    pred_rot = torch.eye(3).expand(num_traj_sets, num_traj_samples, horizon, 3, 3).clone()
    return ModelOutput(pred_xyz=pred_xyz, pred_rot=pred_rot)


def _selector_kwargs(
    model_output: ModelOutput,
    previous_traj: ChosenTrajectory | None = None,
    time_now_us: int = 5_000_000,
    time_query_us: int = 5_400_000,
    future_dt_us: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Pack default keyword arguments for selector tests."""
    if future_dt_us is None:
        horizon = model_output.pred_xyz.shape[2]
        future_dt_us = torch.arange(1, horizon + 1, dtype=torch.int64) * 100_000
    return {
        "model_output": model_output,
        "previous_traj": previous_traj,
        "time_query_us": time_query_us,
        "time_now_us": time_now_us,
        "ego_pose_at_choice": Pose(),
        "future_dt_us": future_dt_us,
    }


def _aligned_previous_traj(horizon: int, step_dt_us: int, time_now_us: int) -> ChosenTrajectory:
    """Build a previous pick whose xyz is `[k+1, k+1, 0]`."""
    xyz = torch.stack(
        [torch.tensor([k + 1.0, k + 1.0, 0.0], dtype=torch.float32) for k in range(horizon)]
    )
    return ChosenTrajectory(
        set_ix=0,
        sample_ix=0,
        xyz=xyz,
        rot=torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(horizon, 3, 3).clone(),
        dt_us=torch.arange(1, horizon + 1, dtype=torch.int64) * step_dt_us,
        time_now_us=time_now_us,
        ego_pose_at_choice=Pose(),
    )


def test_identity_selector_picks_first_set_first_sample() -> None:
    """`IdentitySelector.select` returns `(0, 0)` with the matching tensors."""
    output = _trivial_model_output(num_traj_sets=2, num_traj_samples=2, horizon=3)
    kwargs = _selector_kwargs(model_output=output)
    chosen = IdentitySelector().select(**kwargs)

    assert chosen.set_ix == 0
    assert chosen.sample_ix == 0
    assert torch.equal(chosen.xyz, output.pred_xyz[0, 0])
    assert torch.allclose(chosen.rot[0], torch.eye(3), atol=1e-6)
    assert chosen.time_now_us == kwargs["time_now_us"]
    assert chosen.ego_pose_at_choice == kwargs["ego_pose_at_choice"]
    assert torch.equal(chosen.dt_us, kwargs["future_dt_us"])


def test_closest_to_previous_falls_back_to_identity_when_previous_traj_none() -> None:
    """No previous pick means the selector behaves like `IdentitySelector`."""
    output = _trivial_model_output(num_traj_sets=2, num_traj_samples=2, horizon=3)

    chosen = ClosestToPreviousSelector().select(**_selector_kwargs(model_output=output))

    assert chosen.set_ix == 0
    assert chosen.sample_ix == 0


def test_closest_to_previous_picks_smallest_mean_distance_over_overlap() -> None:
    """Argmin candidate is the one closest to the reprojected previous trajectory."""
    horizon = 8
    step_dt_us = 100_000
    previous_traj = _aligned_previous_traj(
        horizon=horizon, step_dt_us=step_dt_us, time_now_us=1_000_000
    )
    current_time_now_us = 1_300_000
    future_dt_us = torch.arange(1, horizon + 1, dtype=torch.int64) * step_dt_us
    expected_overlap_xyz = torch.stack(
        [torch.tensor([float(k), float(k), 0.0]) for k in range(4, 9)]
    )

    pred_xyz = torch.zeros((1, 2, horizon, 3))
    pred_xyz[0, 0] = torch.full((horizon, 3), 99.0)
    pred_xyz[0, 1, :5] = expected_overlap_xyz
    pred_xyz[0, 1, 5:] = torch.tensor([[0.0, 0.0, 0.0]] * (horizon - 5))
    pred_rot = torch.eye(3).expand(1, 2, horizon, 3, 3).clone()
    output = ModelOutput(pred_xyz=pred_xyz, pred_rot=pred_rot)

    chosen = ClosestToPreviousSelector().select(
        **_selector_kwargs(
            model_output=output,
            previous_traj=previous_traj,
            time_now_us=current_time_now_us,
            time_query_us=current_time_now_us,
            future_dt_us=future_dt_us,
        )
    )

    assert chosen.set_ix == 0
    assert chosen.sample_ix == 1
    assert torch.equal(chosen.xyz, output.pred_xyz[0, 1])
    assert torch.equal(chosen.dt_us, future_dt_us)


def test_closest_to_previous_interpolates_previous_between_waypoints() -> None:
    """Half-step offset forces interpolation; the candidate matching the lerp wins."""
    horizon = 8
    step_dt_us = 100_000
    previous_traj = _aligned_previous_traj(
        horizon=horizon, step_dt_us=step_dt_us, time_now_us=1_000_000
    )
    current_time_now_us = 1_050_000
    future_dt_us = torch.arange(1, horizon + 1, dtype=torch.int64) * step_dt_us
    expected_overlap_xyz = torch.stack([torch.tensor([k + 1.5, k + 1.5, 0.0]) for k in range(7)])

    pred_xyz = torch.zeros((1, 2, horizon, 3))
    pred_xyz[0, 0] = torch.full((horizon, 3), 99.0)
    pred_xyz[0, 1, :7] = expected_overlap_xyz
    pred_xyz[0, 1, 7:] = torch.tensor([[0.0, 0.0, 0.0]])
    pred_rot = torch.eye(3).expand(1, 2, horizon, 3, 3).clone()
    output = ModelOutput(pred_xyz=pred_xyz, pred_rot=pred_rot)

    chosen = ClosestToPreviousSelector().select(
        **_selector_kwargs(
            model_output=output,
            previous_traj=previous_traj,
            time_now_us=current_time_now_us,
            time_query_us=current_time_now_us,
            future_dt_us=future_dt_us,
        )
    )

    assert chosen.sample_ix == 1


def test_closest_to_previous_raises_when_overlap_too_small() -> None:
    """The selector refuses to score on `<= 3` overlapping waypoints."""
    horizon = 4
    step_dt_us = 100_000
    previous_traj = _aligned_previous_traj(
        horizon=horizon, step_dt_us=step_dt_us, time_now_us=1_000_000
    )
    current_time_now_us = 1_500_000
    future_dt_us = torch.arange(1, horizon + 1, dtype=torch.int64) * step_dt_us

    output = ModelOutput(
        pred_xyz=torch.zeros((1, 1, horizon, 3)),
        pred_rot=torch.eye(3).expand(1, 1, horizon, 3, 3).clone(),
    )

    with pytest.raises(ValueError, match="overlapping waypoints"):
        ClosestToPreviousSelector().select(
            **_selector_kwargs(
                model_output=output,
                previous_traj=previous_traj,
                time_now_us=current_time_now_us,
                time_query_us=current_time_now_us,
                future_dt_us=future_dt_us,
            )
        )
