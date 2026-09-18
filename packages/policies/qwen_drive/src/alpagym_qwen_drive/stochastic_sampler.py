# SPDX-License-Identifier: Apache-2.0
"""Stochastic flow-matching sampler for Qwen-Drive RL (paper Eq. 9-14).

The deterministic Euler integration in ``PlanningExpert.sample`` defines no
transition probabilities, so a policy gradient has nothing to differentiate.
Following the Qwen-Drive paper, the deterministic flow is turned into a
stochastic policy by injecting noise over the final integration steps only
(``k in {7, 8, 9}`` of the 10-step solver, zero-indexed), restricted to a
low-frequency cosine subspace.

Mathematical contract
---------------------
At stochastic step ``k`` with flow time ``t = k / num_steps`` and the state
``x_k`` (normalized coordinates):

- The endpoint prediction ``x1_hat = expert.predict_endpoint(x_k, t, ...)``
  induces the flow velocity ``v = (x1_hat - x_k) / max(1 - t, 0.1)``.
- The restoring score (Eq. 9, predicted endpoint substituted for the unknown
  clean path) is ``s = (t * x1_hat - x_k) / (1 - t)^2``.
- The transition mean (Eq. 10) is ``mu(x_k, theta) = x_k + (v - sigma^2 * s) * dt``
  with ``dt = 1 / num_steps`` and ``sigma = epsilon * sqrt(dt)``.
- The sampled transition (Eq. 11) is ``x_{k+1} = mu + sigma * (z_k @ B^T)``
  broadcast over the point dimension, with ``z_k ~ N(0, I_m)`` in the space
  of the first ``m`` orthonormal DCT-II cosine modes ``B`` over the waypoints.

The transition places noise only in the ``m``-dimensional mode subspace, so
the likelihood surrogate (Eq. 12) is a Gaussian in the corresponding basis
coordinates of the *recorded next state*:

    r_k(theta) = B^T mean_d( (x_{k+1} - mu(x_k, theta)) / sigma )
    log pi_theta(x_{k+1}) = -0.5 * mean_m(r_k^2) - log(sqrt(2 pi) sigma)

summed over the stochastic steps. At rollout time ``theta == theta_rollout``
so ``r_k == z_k`` exactly. The trainer replays the *recorded* states as
constants and recomputes only ``mu`` under the current parameters (paper:
"the sampled states and advantages are treated as constants, while the
transition means are recomputed"), so the log-density — and hence the GRPO
ratio — moves with the parameters. The squared residual is averaged (not
summed) over modes, matching the paper's constant rescaling absorbed into
the learning rate.

All trajectory quantities are in the expert's normalized coordinates
(x/165, y/25, heading/(pi/2)) and clipped to ``[-clip_bound, clip_bound]``
before each velocity/score evaluation, as the paper prescribes.
"""

import math
from dataclasses import dataclass

import torch


def cosine_basis(num_waypoints: int, num_modes: int, device: torch.device) -> torch.Tensor:
    """First ``num_modes`` orthonormal cosine modes over ``num_waypoints`` points.

    Uses the DCT-II kernel ``cos(pi * k * (2n + 1) / (2N))``, which is
    orthonormal under the plain inner product with the usual sqrt scaling
    (the DCT-I kernel is only orthogonal under endpoint-weighted inner
    products). Returns ``[T, m]`` with orthonormal columns, so a coefficient
    vector ``z`` maps to the per-waypoint perturbation ``basis @ z`` and unit
    coefficients give unit RMS displacement across waypoints.
    """
    n = torch.arange(num_waypoints, dtype=torch.float32, device=device)
    k = torch.arange(num_modes, dtype=torch.float32, device=device)
    modes = torch.cos(torch.pi * torch.outer(k, 2.0 * n + 1.0) / (2.0 * num_waypoints))
    scale = torch.full((num_modes,), (2.0 / num_waypoints) ** 0.5, device=device)
    scale[0] = (1.0 / num_waypoints) ** 0.5
    return modes.transpose(0, 1) * scale


@dataclass(frozen=True)
class StochasticSamplingConfig:
    """Knobs of the stochastic sampler. Defaults follow the paper's setup.

    ``stochastic_steps``: the paper indexes the 10 Euler steps from zero and
    perturbs the final three transitions, i.e. the steps entering ``k in
    {7, 8, 9}`` (the transition *out of* step ``k``).

    ``epsilon``: continuous diffusion coefficient; the discrete transition
    standard deviation is ``sigma = epsilon * sqrt(1 / num_inference_steps)``.
    """

    num_inference_steps: int = 10
    epsilon: float = 0.05
    num_modes: int = 8
    stochastic_steps: tuple = (7, 8, 9)
    min_one_minus_t: float = 0.1
    sigma_min: float = 1e-3
    clip_bound: float = 5.0


def transition_std(config: StochasticSamplingConfig) -> float:
    """Discrete transition std ``sigma = epsilon * sqrt(dt)`` (paper Eq. 10)."""
    return max(
        config.epsilon * (1.0 / config.num_inference_steps) ** 0.5, config.sigma_min
    )


def _mean_drift(
    expert,
    waypoints,
    flow_time,
    history_queries,
    scene_cache,
    position_anchor,
    nav_command,
    ego_status,
    config,
):
    """Score-corrected transition increment ``(v - sigma^2 * s) * dt``.

    Evaluates ``predict_endpoint`` at the clipped state ``waypoints`` and
    combines the induced flow velocity with the Eq. 9 restoring score.
    Differentiable w.r.t. the expert parameters.
    """
    step = 1.0 / config.num_inference_steps
    remaining = max(1.0 - flow_time, config.min_one_minus_t)
    batch = waypoints.shape[0]
    flow_time_t = torch.full(
        (batch,), flow_time, device=waypoints.device, dtype=torch.float32
    )
    endpoint = expert.predict_endpoint(
        waypoints,
        flow_time_t,
        history_queries,
        scene_cache,
        position_anchor,
        nav_command,
        ego_status,
    )
    velocity = (endpoint - waypoints) / remaining
    score = (flow_time * endpoint - waypoints) / max(1.0 - flow_time, 1e-6) ** 2
    sigma2 = transition_std(config) ** 2
    return (velocity - sigma2 * score) * step


def _tile_conditioning(nav_command, ego_status, position_anchor, batch):
    """Expand the single-sample conditioning tensors to the sample axis."""
    return (
        nav_command.expand(batch),
        ego_status.expand(batch, -1),
        position_anchor.expand(-1, batch),
    )


def stochastic_sample(
    expert,
    scene_cache,
    position_anchor,
    history,
    history_velocity,
    history_acceleration,
    nav_command,
    ego_status,
    noise,
    config,
    generator=None,
):
    """Run the stochastic sampler.

    Returns ``(trajectory, z, states, old_logprob)``:

    - trajectory: ``[B, T, D]`` final normalized trajectory.
    - z: injected subspace coefficients ``[B, K, m]`` (K stochastic steps).
    - states: the visited states ``[B, N+1, T, D]`` including the initial
      noise as ``states[:, 0]`` — the constants the trainer replays.
    - old_logprob: rollout-time log-density ``[B]`` (Eq. 12). Because the
      sampling mean and the scoring mean coincide, the residual is exactly
      ``z_k``, so ``old_logprob`` is computed from ``z`` directly.

    See the module docstring for the conditioning-tensor shapes; they mirror
    ``PlanningExpert.sample`` (single sample, no batch axis) and are tiled
    internally to the ``B`` samples.
    """
    batch = noise.shape[0]
    nav_command, ego_status, position_anchor = _tile_conditioning(
        nav_command, ego_status, position_anchor, batch
    )
    history_queries = expert.encode_history(
        history.expand(batch, -1, -1),
        nav_command,
        history_velocity.expand(batch, -1, -1),
        history_acceleration.expand(batch, -1, -1),
    )
    sigma = transition_std(config)
    log_norm = math.log(math.sqrt(2.0 * math.pi) * sigma)

    waypoints = noise.float().clone()
    states = [waypoints.clone()]
    z_list = []
    old_logprob = torch.zeros(batch, device=noise.device, dtype=torch.float32)

    with torch.no_grad():
        for k in range(config.num_inference_steps):
            clipped = torch.clamp(waypoints, -config.clip_bound, config.clip_bound)
            mean_drift = _mean_drift(
                expert,
                clipped,
                k / config.num_inference_steps,
                history_queries,
                scene_cache,
                position_anchor,
                nav_command,
                ego_status,
                config,
            )
            if k in config.stochastic_steps:
                z_k = torch.randn(
                    batch,
                    config.num_modes,
                    generator=generator,
                    device=noise.device,
                    dtype=torch.float32,
                )
                basis = cosine_basis(
                    noise.shape[1], config.num_modes, noise.device
                )
                perturbation = sigma * (z_k @ basis.transpose(0, 1)).unsqueeze(-1)
                waypoints = waypoints + mean_drift + perturbation
                z_list.append(z_k)
            else:
                waypoints = waypoints + mean_drift
            states.append(waypoints.clone())

    # Rollout-time density of the coefficient displacement u = sigma * z
    # (u ~ N(0, sigma^2 I_m) in coefficient space): log N(u; 0, sigma^2 I)
    # averaged over modes, matching the trainer-side evaluation.
    old_logprob = torch.zeros(batch, device=noise.device, dtype=torch.float32)
    for z_k in z_list:
        old_logprob = old_logprob - 0.5 * z_k.pow(2).mean(dim=-1) - log_norm

    z = torch.stack(z_list, dim=1)  # [B, K, m]
    states = torch.stack(states, dim=1)  # [B, N+1, T, D]
    return waypoints, z, states, old_logprob


def stochastic_logprob(
    expert,
    scene_cache,
    position_anchor,
    history,
    history_velocity,
    history_acceleration,
    nav_command,
    ego_status,
    states,
    z,
    config,
):
    """Differentiable log-density of the recorded transitions (Eq. 12/14).

    The recorded state trajectory ``states`` ``[B, N+1, T, D]`` and the
    injected coefficients ``z`` ``[B, K, m]`` are constants. At each
    stochastic step the transition mean is recomputed from the recorded
    state under the *current* expert parameters, and the residual

        r_k(theta) = B^T mean_d((x_{k+1} - mu(x_k, theta)) / sigma)

    carries the parameter gradient through ``predict_endpoint``. At rollout
    parameters ``r_k == z_k``; after any update it differs, so the GRPO
    ratio moves.

    Returns log-probs ``[B]``: Eq. 12 summed over the stochastic steps.
    """
    batch = z.shape[0]
    nav_command, ego_status, position_anchor = _tile_conditioning(
        nav_command, ego_status, position_anchor, batch
    )
    history_queries = expert.encode_history(
        history.expand(batch, -1, -1),
        nav_command,
        history_velocity.expand(batch, -1, -1),
        history_acceleration.expand(batch, -1, -1),
    )
    basis = cosine_basis(states.shape[2], config.num_modes, z.device)  # [T, m]
    sigma = transition_std(config)
    log_norm = math.log(math.sqrt(2.0 * math.pi) * sigma)

    logprob = torch.zeros(batch, device=z.device, dtype=torch.float32)
    stochastic_index = 0
    for k in range(config.num_inference_steps):
        if k not in config.stochastic_steps:
            continue
        x_k = torch.clamp(
            states[:, k].detach(), -config.clip_bound, config.clip_bound
        )
        x_next = states[:, k + 1].detach()
        mean_drift = _mean_drift(
            expert,
            x_k,
            k / config.num_inference_steps,
            history_queries,
            scene_cache,
            position_anchor,
            nav_command,
            ego_status,
            config,
        )
        mu = x_k + mean_drift
        # Coefficient residual: project the recorded displacement onto the
        # cosine basis and normalize by sigma. The perturbation is identical
        # across the D point dimensions, so mean over d recovers it exactly.
        residual = torch.einsum(
            "btd,tm->bm", (x_next - mu) / sigma, basis
        ) / states.shape[-1]
        logprob = logprob - 0.5 * residual.pow(2).mean(dim=-1) - log_norm
        stochastic_index += 1

    if stochastic_index != z.shape[1]:
        raise ValueError(
            f"recorded z has {z.shape[1]} stochastic steps but config has "
            f"{len(config.stochastic_steps)}"
        )
    return logprob
