# Phase 4: Closed-loop GRPO for Qwen-Drive — Design & Fix Log

Status: implementation complete, numerically tested on CPU (5/5 unit tests),
pending first real run on A100.

## 1. Why closed-loop RL would not start (environment root causes)

Ranked by likelihood, all found while reading the deployed setup:

1. **`packages/policies/qwen_drive` was not a uv workspace member.**
   The root `pyproject.toml` workspace list ended at `autovla`; after any
   venv rebuild, `uv sync --all-packages` silently skipped the qwen_drive
   package, so the `alpagym_qwen_drive` entry points / imports vanished and
   every launch failed with import errors that looked like "environment
   problems". Fixed: added to `[tool.uv.workspace] members` and the
   `ty.environment` root list.

2. **Everything lived in `/tmp`.** venv `/tmp/alpagym_venv`, model
   `/tmp/qd_model/`, checkpoints `/tmp/qd_runs/` — any container restart or
   `/tmp` cleaner wipes them. Move to persistent paths, e.g.
   `/root/alpagym_venv`, `/root/models/qd_model`, `/root/runs/qd_runs`.

3. **transformers version conflict.** Upstream Qwen-Drive requires
   `transformers==5.14.1`; the alpagym env may pin a different line. The SFT
   venv solved this by isolation; for closed-loop, patch `qwen_drive` into
   the same env (see §4 step 1) instead of maintaining two envs.

## 2. Phase 4 math (paper arXiv:2609.00111, Eq. 9-14)

The official PlanningExpert only ships a deterministic no-grad Euler
sampler, so no likelihood exists. We randomize the last 3 integration steps
(k = 7, 8, 9 of 10):

- Perturbation lives in the orthonormal **DCT-II cosine subspace** (first
  `num_modes=8` modes over T=50 waypoints). DCT-I is NOT orthonormal under
  the plain inner product — verified numerically; DCT-II is.
- Transition: `x_{k+1} = mu_k(x_k) + sigma_k * (B z_k)` with
  `sigma_k = epsilon * sqrt(delta_t)`, plus the paper's restoring-score
  correction pulling each sample toward the flow mean (keeps mode-collapse
  from degenerating the spread).
- Density (Eq. 12): `log pi = sum_k log N(x_{k+1}; mu_k, sigma_k^2 B B^T)`,
  evaluated in the m-dim coefficient space: `-0.5 ||z||^2/m - log sqrt(2 pi sigma^2)`.
- Gradient path: **states recorded at rollout are detached**; θ enters only
  through `mu_k` (via `predict_endpoint`). This is the crucial fix — replaying
  with non-detached states makes the residual identically z_k and the
  gradient exactly zero.
- Advantage: GRPO group-relative (Eq. 13) with gamma-discounted returns
  (Eq. 14), handled by the existing `cosmos/trainer.py` — no changes needed
  there because log_probs now come from our `forward()`.

## 3. What was implemented

| File | Change |
|---|---|
| `stochastic_sampler.py` (new) | `StochasticSamplingConfig`, `cosine_basis`, `stochastic_sample`, `stochastic_logprob` |
| `inference_model.py` | RL branch uses the stochastic sampler; records full trace (states/z/old_logprob/conditioning) into `ModelOutput.extra`; real `build_policy_replay_data` (payload: selected_states `[B,N+1,T,D]`, selected_z `[B,K,m]`, camera/history/route) and `build_trainer_model_inputs` |
| `cosmos_wrapper.py` | real `forward()`: rebuilds DrivingScene per row, frozen-VLM prefill, differentiable `stochastic_logprob`; padding rows forward 0 logprob |
| `bundle.py` | passes `rl_sampling` config from YAML into both rollout adapter and trainer wrapper (`set_rl_sampling_config`) |
| `configs/policy/qwen_drive.yaml` | `rl_sampling:` block (epsilon, num_modes, stochastic_steps) |
| `configs/experiment/qwen_drive_a100_1gpu_grpo.yaml` (new) | training experiment with GRPO + stochastic sampling on |
| `tests/test_stochastic_sampler.py` (new) | 5 numerical tests on a tiny real PlanningExpert |

Verified on CPU with a tiny real upstream `PlanningExpert`:
- cosine basis orthonormal
- sample shapes/finite
- **replayed logprob == recorded old_logprob** (on-policy consistency)
- **perturbing expert params changes logprob; backward gives finite,
  non-zero grads** (this test caught and killed two real bugs: the DCT-I
  basis and the non-detached replay states)

## 4. A100 bring-up order

1. `cd alpagym && git pull && uv sync --all-packages` (qwen_drive is now a
   workspace member). Then `uv pip install -e /path/to/Qwen-Drive-1.0 --no-deps`
   plus its pinned `transformers==5.14.1` — verify no conflict with the
   runtime's own pins first: `uv pip check`.
2. Run the unit tests in the A100 env (they need the upstream repo on
   `QWEN_DRIVE_PATH`): `pytest packages/policies/qwen_drive/tests -v`.
3. Smoke the inference path with `qwen_drive_a100_1gpu_inference.yaml`
   (unchanged behavior — RL sampling off).
4. First GRPO run with `qwen_drive_a100_1gpu_grpo.yaml`, 1 GPU, tiny rollout
   count. Watch for `log_probs` being finite and old-vs-new logprob ratio
   near 1 in step 0.
5. Scale to the 4gpu_dagger topology (GPU6 rollout / GPU2 expert /
   GPU3-5 policy).

Hyperparameters the paper leaves open (epsilon, mode count M): start with
`epsilon: 0.3`, `num_modes: 8`, last-3 steps; sweep epsilon in
{0.1, 0.3, 1.0} on the smoke config.
