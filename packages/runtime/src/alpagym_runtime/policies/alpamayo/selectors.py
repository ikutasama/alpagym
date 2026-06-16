# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trajectory selectors."""

from typing import Protocol, runtime_checkable

import torch
from alpagym_host.config import TrajectorySelectorKind

from alpagym_runtime.inference.types import ModelOutput
from alpagym_runtime.types import ChosenTrajectory, Pose

from .geometry_utils import linear_interp_xyz, local_to_world, world_to_local


@runtime_checkable
class TrajectorySelector(Protocol):
    """Pick one trajectory out of a `ModelOutput` each drive tick."""

    def select(
        self,
        model_output: ModelOutput,
        previous_traj: ChosenTrajectory | None,
        time_query_us: int,
        time_now_us: int,
        ego_pose_at_choice: Pose,
        future_dt_us: torch.Tensor,
    ) -> ChosenTrajectory:
        """Return the chosen trajectory for this tick.

        Args:
            model_output: Model predictions for this tick to choose a candidate from.
            previous_traj: Trajectory chosen on the prior tick, or ``None`` at
                episode start.
            time_query_us: Microsecond time at which the policy is being
                queried.
            time_now_us: Microsecond time of the current drive tick.
            ego_pose_at_choice: Ego pose at the moment of selection; stamped
                onto the returned `ChosenTrajectory`.
            future_dt_us: `[horizon]` int64 tensor of time offsets from
                ``time_now_us`` to each waypoint in ``model_output.pred_xyz``.
        """


class IdentitySelector:
    """Always picks set_ix=0, sample_ix=0."""

    def select(
        self,
        model_output: ModelOutput,
        previous_traj: ChosenTrajectory | None,
        time_query_us: int,
        time_now_us: int,
        ego_pose_at_choice: Pose,
        future_dt_us: torch.Tensor,
    ) -> ChosenTrajectory:
        """Return the first set / first sample candidate, stamped with this tick's pose."""
        del previous_traj, time_query_us
        return _build_chosen_traj(
            model_output=model_output,
            set_ix=0,
            sample_ix=0,
            time_now_us=time_now_us,
            ego_pose_at_choice=ego_pose_at_choice,
            future_dt_us=future_dt_us,
        )


class ClosestToPreviousSelector:
    """Pick the candidate whose mean L2-xyz to the reprojected previous pick is smallest.

    The previous trajectory is re-projected into the current ego frame, then
    linearly interpolated onto the current candidates' timestamps over the
    overlap window (current waypoints whose absolute time falls within the
    previous trajectory's horizon). Each candidate's score is the mean L2
    distance to that interpolated previous trajectory over the overlap window.
    """

    def select(
        self,
        model_output: ModelOutput,
        previous_traj: ChosenTrajectory | None,
        time_query_us: int,
        time_now_us: int,
        ego_pose_at_choice: Pose,
        future_dt_us: torch.Tensor,
    ) -> ChosenTrajectory:
        """Return argmin candidate over mean L2-xyz to the reprojected previous pick."""
        del time_query_us
        if previous_traj is None:
            return _build_chosen_traj(
                model_output=model_output,
                set_ix=0,
                sample_ix=0,
                time_now_us=time_now_us,
                ego_pose_at_choice=ego_pose_at_choice,
                future_dt_us=future_dt_us,
            )

        prev_tstamps = (
            torch.tensor(previous_traj.time_now_us, dtype=torch.int64) + previous_traj.dt_us
        )
        curr_tstamps = torch.tensor(time_now_us, dtype=torch.int64) + future_dt_us

        overlap_mask = curr_tstamps <= prev_tstamps[-1]
        overlap_count = int(overlap_mask.sum().item())
        if overlap_count <= 3:
            raise ValueError(
                "ClosestToPreviousSelector needs more than 3 overlapping waypoints with "
                f"the previous trajectory; got {overlap_count}. Drive ticks may be spaced "
                "too far apart relative to the prediction horizon."
            )

        prev_world_xyz = local_to_world(
            points=previous_traj.xyz, pose=previous_traj.ego_pose_at_choice
        )
        reprojected_xyz = world_to_local(points=prev_world_xyz, pose=ego_pose_at_choice)
        prev_interp_xyz = linear_interp_xyz(
            src_tstamps=prev_tstamps,
            src_xyz=reprojected_xyz,
            query_tstamps=curr_tstamps[overlap_mask],
        )

        candidate_xyz = model_output.pred_xyz[:, :, overlap_mask, :]
        distances = torch.linalg.norm(
            candidate_xyz - prev_interp_xyz.to(dtype=candidate_xyz.dtype),
            dim=-1,
        )
        if not bool(torch.all(torch.isfinite(distances)).item()):
            raise ValueError("ClosestToPreviousSelector produced non-finite distances")
        mean_distances = distances.mean(dim=-1)

        num_samples = mean_distances.shape[1]
        flat_argmin = int(torch.argmin(mean_distances).item())
        set_ix = flat_argmin // num_samples
        sample_ix = flat_argmin % num_samples
        return _build_chosen_traj(
            model_output=model_output,
            set_ix=set_ix,
            sample_ix=sample_ix,
            time_now_us=time_now_us,
            ego_pose_at_choice=ego_pose_at_choice,
            future_dt_us=future_dt_us,
        )


def build_selector(kind: TrajectorySelectorKind) -> TrajectorySelector:
    """Instantiate the selector named by `kind`."""
    match kind:
        case TrajectorySelectorKind.identity:
            return IdentitySelector()
        case TrajectorySelectorKind.closest_to_previous:
            return ClosestToPreviousSelector()
        case _:
            raise ValueError(f"Unknown selector kind: {kind!r}")


def _build_chosen_traj(
    model_output: ModelOutput,
    set_ix: int,
    sample_ix: int,
    time_now_us: int,
    ego_pose_at_choice: Pose,
    future_dt_us: torch.Tensor,
) -> ChosenTrajectory:
    """Pull the selected candidate out and pack a `ChosenTrajectory`."""
    xyz = model_output.pred_xyz[set_ix, sample_ix]
    rot = model_output.pred_rot[set_ix, sample_ix]
    return ChosenTrajectory(
        set_ix=set_ix,
        sample_ix=sample_ix,
        xyz=xyz,
        rot=rot,
        dt_us=future_dt_us,
        time_now_us=time_now_us,
        ego_pose_at_choice=ego_pose_at_choice,
    )
