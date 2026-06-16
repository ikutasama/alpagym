# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level reward dispatcher for `EpisodeOutput` rewards."""

from alpagym_host.config import RewardConfig, RewardTermConfig
from alpagym_runtime.rewards.distance_to_gt import compute_distance_to_gt_reward
from alpagym_runtime.types import EpisodeOutput, GroundTruth, RewardResult


def compute_reward(
    episode: EpisodeOutput,
    ground_truth: GroundTruth | None,
    config: RewardConfig,
) -> RewardResult:
    """Sum every term's contribution into one episode-level reward."""
    total = 0.0
    report: dict[str, float] = {}
    for term in config.terms:
        contribution, term_report = _compute_term(term, episode, ground_truth)
        total += contribution
        report.update(term_report)
    return RewardResult(total=total, report_metrics=report)


def _compute_term(
    term: RewardTermConfig,
    episode: EpisodeOutput,
    ground_truth: GroundTruth | None,
) -> tuple[float, dict[str, float]]:
    """Compute one term's scaled contribution and report entries."""
    if term.kind == "metric":
        if episode.metrics is None:
            raise ValueError("metric reward term requires episode.metrics to be populated")
        name = term.metric_name
        if name is None:
            raise ValueError("metric reward term requires RewardTermConfig.metric_name to be set")
        value = float(episode.metrics.aggregated[name])
        return term.scale * value, {name: value}
    if term.kind == "distance_to_gt":
        result = compute_distance_to_gt_reward(episode, ground_truth, distance_scale=term.scale)
        return result.total, dict(result.report_metrics)
    raise ValueError(f"Unknown RewardTermConfig.kind: {term.kind!r}")
