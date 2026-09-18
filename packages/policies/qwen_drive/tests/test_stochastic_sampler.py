# SPDX-License-Identifier: Apache-2.0
"""Numerical tests for the stochastic sampler (paper Eq. 9-14).

Uses a tiny real ``PlanningExpert`` from the upstream repo so the math is
exercised end-to-end without GPU or the 4B VLM. Requires ``qwen_drive`` on
sys.path (``QWEN_DRIVE_PATH`` or a sibling checkout); skipped otherwise.
"""

import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_QD_CANDIDATES = [
    Path("/tmp/qwen-drive/src"),
    Path(os.environ["QWEN_DRIVE_PATH"]) / "src"
    if "QWEN_DRIVE_PATH" in os.environ
    else None,
]
for _c in _QD_CANDIDATES:
    if _c and (_c / "qwen_drive").is_dir():
        sys.path.insert(0, str(_c))
        break

qwen_drive = pytest.importorskip("qwen_drive")

from alpagym_qwen_drive.stochastic_sampler import (
    StochasticSamplingConfig,
    cosine_basis,
    stochastic_logprob,
    stochastic_sample,
    transition_std,
)


def tiny_expert(hidden_size: int = 32):
    from qwen_drive.planning_expert import PlanningExpert, PlanningExpertConfig

    config = PlanningExpertConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layers_per_kv=1,
        time_embed_dim=8,
        fourier_num_features=4,
        partial_rotary_factor=0.5,
        mrope_section=(1, 1, 2),
    )
    expert = PlanningExpert(
        config,
        num_future_points=50,
        num_history_points=16,
        trajectory_point_dim=3,
    )
    expert.eval()
    return expert


def fake_scene_cache(expert, batch=1, length=8):
    """A KV cache stand-in with the shape ``predict_endpoint`` expects."""
    device = next(expert.parameters()).device
    kv_heads = expert.config.num_key_value_heads
    head_dim = expert.config.head_dim
    # One cache entry per layers_per_kv group the expert consumes.
    num_caches = expert.config.num_hidden_layers // expert.config.layers_per_kv
    return [
        (
            torch.randn(batch, length, kv_heads, head_dim, device=device),
            torch.randn(batch, length, kv_heads, head_dim, device=device),
        )
        for _ in range(num_caches)
    ]


@pytest.fixture
def conditioning():
    torch.manual_seed(0)
    H = 16
    return {
        "history": torch.randn(1, H - 1, 3),
        "history_velocity": torch.randn(1, H, 2),
        "history_acceleration": torch.randn(1, H, 2),
        "nav_command": torch.tensor([0]),
        "ego_status": torch.randn(1, 8),
    }


def test_cosine_basis_orthonormal():
    basis = cosine_basis(50, 8, torch.device("cpu"))
    gram = basis.T @ basis
    assert torch.allclose(gram, torch.eye(8), atol=1e-5)


def test_sample_shapes_and_finiteness(conditioning):
    expert = tiny_expert()
    scene_cache = fake_scene_cache(expert, batch=2)
    anchor = torch.zeros(3, 1, dtype=torch.long)
    config = StochasticSamplingConfig()
    noise = torch.randn(2, 50, 3)
    trajectory, z, states, old_logprob = stochastic_sample(
        expert,
        scene_cache=scene_cache,
        position_anchor=anchor,
        noise=noise,
        config=config,
        **conditioning,
    )
    assert trajectory.shape == (2, 50, 3)
    assert z.shape == (2, 3, 8)  # K=3 stochastic steps, m=8 modes
    assert states.shape == (2, 11, 50, 3)  # N=10 steps + initial noise
    assert old_logprob.shape == (2,)
    assert torch.isfinite(trajectory).all()
    assert torch.isfinite(old_logprob).all()


def test_logprob_matches_rollout_before_update(conditioning):
    """Single on-policy check: re-evaluated logprob == recorded old_logprob."""
    expert = tiny_expert()
    scene_cache = fake_scene_cache(expert, batch=2)
    anchor = torch.zeros(3, 1, dtype=torch.long)
    config = StochasticSamplingConfig()
    noise = torch.randn(2, 50, 3)
    _, z, states, old_logprob = stochastic_sample(
        expert,
        scene_cache=scene_cache,
        position_anchor=anchor,
        noise=noise,
        config=config,
        **conditioning,
    )
    new_logprob = stochastic_logprob(
        expert,
        scene_cache=scene_cache,
        position_anchor=anchor,
        states=states,
        z=z,
        config=config,
        **conditioning,
    )
    assert torch.allclose(new_logprob.detach(), old_logprob, atol=1e-4), (
        f"{new_logprob} != {old_logprob}"
    )


def test_gradient_flows_through_expert(conditioning):
    """Perturbing the expert changes the logprob and yields finite grads."""
    expert = tiny_expert()
    for p in expert.parameters():
        p.requires_grad_(True)
    scene_cache = fake_scene_cache(expert, batch=2)
    anchor = torch.zeros(3, 1, dtype=torch.long)
    config = StochasticSamplingConfig()
    noise = torch.randn(2, 50, 3)
    _, z, states, _ = stochastic_sample(
        expert,
        scene_cache=scene_cache,
        position_anchor=anchor,
        noise=noise,
        config=config,
        **conditioning,
    )

    logprob_before = stochastic_logprob(
        expert,
        scene_cache=scene_cache,
        position_anchor=anchor,
        states=states,
        z=z,
        config=config,
        **conditioning,
    ).detach().clone()

    with torch.no_grad():
        for p in expert.parameters():
            p.add_(0.01 * torch.randn_like(p))

    logprob = stochastic_logprob(
        expert,
        scene_cache=scene_cache,
        position_anchor=anchor,
        states=states,
        z=z,
        config=config,
        **conditioning,
    )
    assert not torch.allclose(logprob.detach(), logprob_before), (
        "logprob did not change after a parameter perturbation — "
        "the policy gradient would be zero"
    )
    logprob.sum().backward()
    grads = [p.grad for p in expert.parameters() if p.grad is not None]
    assert grads, "no gradients reached the expert"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_transition_std_matches_paper():
    config = StochasticSamplingConfig(epsilon=0.05, num_inference_steps=10)
    assert abs(transition_std(config) - 0.05 * 0.1**0.5) < 1e-9
