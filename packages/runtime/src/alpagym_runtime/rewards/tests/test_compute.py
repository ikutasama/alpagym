# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the top-level `compute_reward` dispatcher."""

import pytest
from alpagym_host.config import RewardConfig, RewardTermConfig
from alpagym_runtime.rewards.compute import compute_reward
from alpagym_runtime.types import (
    EgoPose,
    EpisodeMetrics,
    EpisodeOutput,
    GroundTruth,
    Pose,
    Trajectory,
    Vec3,
)


def _ego_pose(timestamp_us: int, x: float, y: float) -> EgoPose:
    """Build an `EgoPose` at `timestamp_us` with the given XY position."""
    return EgoPose(timestamp_us=timestamp_us, pose=Pose(vec=Vec3(x=x, y=y, z=0.0)))


def _ground_truth(
    gt_poses: tuple[EgoPose, ...] | None,
    gt_anchor_us: int = 0,
) -> GroundTruth | None:
    """Build a minimal `GroundTruth` carrying `gt_poses`, or `None`."""
    if gt_poses is None:
        return None
    return GroundTruth(
        ego_trajectory=Trajectory(poses=gt_poses),
        timestamp_us=gt_anchor_us,
    )


def _episode(
    metrics: EpisodeMetrics | None,
    executed_poses: tuple[EgoPose, ...] = (),
) -> EpisodeOutput:
    """Build a minimal `EpisodeOutput` carrying the given metrics and executed trajectory."""
    return EpisodeOutput(
        scene_id="scene",
        session_uuid="session",
        num_steps=0,
        policy_outputs=(),
        executed_ego_trajectory=Trajectory(poses=executed_poses),
        metrics=metrics,
    )


def test_single_distance_to_gt_term_sums_to_scaled_rmse() -> None:
    """A lone `distance_to_gt` term reproduces the GT distance reward."""
    gt_poses = (_ego_pose(0, 0.0, 1.0), _ego_pose(100, 1.0, 1.0))
    executed_poses = (_ego_pose(0, 0.0, 0.0), _ego_pose(100, 1.0, 0.0))
    config = RewardConfig(terms=[RewardTermConfig(kind="distance_to_gt", scale=-1.0)])

    result = compute_reward(
        _episode(metrics=None, executed_poses=executed_poses),
        _ground_truth(gt_poses=gt_poses, gt_anchor_us=0),
        config,
    )

    assert result.total == pytest.approx(-1.0)
    assert "gt_rmse_xy" in result.report_metrics
    assert "gt_num_paired" in result.report_metrics
    assert result.report_metrics["gt_anchor_us"] == pytest.approx(0.0)


def test_single_metric_term_sums_to_scaled_value() -> None:
    """A lone `metric` term reads aggregated, scales, and reports the raw value."""
    config = RewardConfig(
        terms=[RewardTermConfig(kind="metric", metric_name="progress", scale=2.0)]
    )
    metrics = EpisodeMetrics(aggregated={"progress": 0.4, "noise": 99.0})

    result = compute_reward(
        _episode(metrics=metrics),
        _ground_truth(gt_poses=None),
        config,
    )

    assert result.total == pytest.approx(0.8)
    assert result.report_metrics == {"progress": 0.4}


def test_combined_terms_sum_contributions_and_merge_reports() -> None:
    """Mixed term kinds add up and the report carries both kinds of keys."""
    gt_poses = (_ego_pose(0, 0.0, 1.0), _ego_pose(100, 1.0, 1.0))
    executed_poses = (_ego_pose(0, 0.0, 0.0), _ego_pose(100, 1.0, 0.0))
    metrics = EpisodeMetrics(aggregated={"progress": 0.5, "collision_any": 1.0})
    config = RewardConfig(
        terms=[
            RewardTermConfig(kind="distance_to_gt", scale=-1.0),
            RewardTermConfig(kind="metric", metric_name="progress", scale=1.0),
            RewardTermConfig(kind="metric", metric_name="collision_any", scale=-5.0),
        ]
    )

    result = compute_reward(
        _episode(metrics=metrics, executed_poses=executed_poses),
        _ground_truth(gt_poses=gt_poses, gt_anchor_us=0),
        config,
    )

    assert result.total == pytest.approx(-1.0 + 0.5 - 5.0)
    assert result.report_metrics["gt_rmse_xy"] == pytest.approx(1.0)
    assert result.report_metrics["progress"] == pytest.approx(0.5)
    assert result.report_metrics["collision_any"] == pytest.approx(1.0)


def test_missing_aggregated_metric_raises() -> None:
    """A metric term naming an absent aggregated key fails fast."""
    config = RewardConfig(terms=[RewardTermConfig(kind="metric", metric_name="missing", scale=1.0)])
    metrics = EpisodeMetrics(aggregated={"progress": 0.5})

    with pytest.raises(KeyError, match="missing"):
        compute_reward(
            _episode(metrics=metrics),
            _ground_truth(gt_poses=None),
            config,
        )


def test_metric_term_requires_episode_metrics() -> None:
    """A metric term fails fast when the simulator surfaced no aggregated metrics."""
    config = RewardConfig(
        terms=[RewardTermConfig(kind="metric", metric_name="progress", scale=1.0)]
    )

    with pytest.raises(ValueError, match="episode.metrics"):
        compute_reward(
            _episode(metrics=None),
            _ground_truth(gt_poses=None),
            config,
        )


def test_empty_terms_rejected_at_construction() -> None:
    """A `RewardConfig` without any term is rejected before reaching the dispatcher."""
    with pytest.raises(ValueError, match="at least one term"):
        RewardConfig(terms=[])


def test_metric_term_requires_metric_name() -> None:
    """`kind='metric'` without `metric_name` is rejected at construction."""
    with pytest.raises(ValueError, match="metric_name"):
        RewardTermConfig(kind="metric", scale=1.0)


def test_distance_to_gt_term_rejects_metric_name() -> None:
    """`kind='distance_to_gt'` with `metric_name` is rejected at construction."""
    with pytest.raises(ValueError, match="must not set metric_name"):
        RewardTermConfig(kind="distance_to_gt", scale=-0.01, metric_name="progress")


def test_unknown_term_kind_rejected_at_construction() -> None:
    """Unknown term kinds are rejected before they reach the dispatcher."""
    with pytest.raises(ValueError, match="Unknown RewardTermConfig.kind"):
        RewardTermConfig(kind="bogus", scale=1.0)
