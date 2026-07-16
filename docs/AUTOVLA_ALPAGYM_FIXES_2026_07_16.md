# AutoVLA AlpaGym Fixes - 2026-07-16

This note summarizes the code changes made for the AutoVLA-on-AlpaGym closed-loop
RL path. It is written for the A100 server-side agent/operator that will pull
this repository and continue training/debugging.

## Context

The repository integrates AutoVLA as an AlpaGym policy bundle:

- rollout inference: `packages/policies/autovla/src/alpagym_autovla/inference_model.py`
- trainer forward patch: `packages/policies/autovla/src/alpagym_autovla/autovla_trainer_forward.py`
- bundle registration: `packages/policies/autovla/src/alpagym_autovla/bundle.py`
- policy config: `packages/policies/autovla/src/alpagym_autovla/configs/policy/autovla.yaml`

The immediate goal is to make the closed-loop GRPO path train the same discrete
AutoVLA action-token sequence that was generated during rollout.

## What Changed

### 1. Action-token layout is now checked and repaired safely

New file:

```text
packages/policies/autovla/src/alpagym_autovla/action_tokens.py
```

AutoVLA assumes a fixed token block:

```text
<action_0>   -> action_start_id
<action_1>   -> action_start_id + 1
...
<action_2047>
```

The code now accepts two safe tokenizer states:

1. the tokenizer already has the full AutoVLA action-token block at the expected
   ids;
2. the tokenizer is a raw Qwen tokenizer whose length is exactly
   `action_start_id`, so appending all `<action_i>` tokens creates the correct
   id layout.

It rejects shifted, partial, or incompatible tokenizers before rollout/training
starts. This prevents the model from silently decoding or optimizing the wrong
codebook rows.

### 2. Rollout and trainer now use exact action-token ids

The old logic treated every token id `>= action_start_id` as an action token and
decoded the codebook index as `token_id - action_start_id`.

The new logic validates the exact `<action_i>` ids once and uses membership in
that id set for:

- extracting action tokens from generated completions;
- computing rollout-time action-token logprob;
- recomputing trainer-side action-token logprob.

If a completion contains no valid action token, the logprob contribution is now
zero instead of summing unrelated text tokens.

### 3. Short action-token generations are visible

When AutoVLA generates fewer than `num_poses` action tokens, rollout still pads
the decode with codebook index 0 so the simulator can receive a trajectory, but
it now logs a warning. Repeated warnings mean the tokenizer/checkpoint or prompt
format should be inspected before trusting training curves.

### 4. `completion_ids` replay truncation stays fixed and is now tested

The latest upstream fix in this branch kept full `completion_ids` instead of
taking only the first token after `BatchedModelOutput.unbind()`. A new unit test
locks that behavior:

```text
packages/policies/autovla/tests/test_inference_replay.py
```

### 5. AutoVLA package is included in normal checks

`pyproject.toml` now includes AutoVLA in:

- `tool.ty.environment.root`
- `tool.pytest.ini_options.testpaths`
- `tool.uv.sources`

`uv.lock` was updated to include the editable `alpagym-autovla` workspace
package. On the A100 server, run `uv lock --check`; if the local `uv` version
wants to rewrite the lock, inspect the diff before committing.

### 6. Smoke script experiment name is fixed

`scripts/run_autovla_smoke.sh` now defaults to the actual config name:

```text
autovla_local_1gpu_smoke
```

The old default `autovla_local_smoke` did not match any experiment YAML.

## New Tests

```text
packages/policies/autovla/tests/test_action_tokens.py
packages/policies/autovla/tests/test_inference_replay.py
```

They verify:

- raw Qwen tokenizer can be extended only when the base vocab length matches
  `action_start_id`;
- shifted raw tokenizers fail fast;
- partial action-token blocks fail fast;
- exact action-token masks do not include unrelated ids above the start id;
- replay payload keeps full `completion_ids`.

## Recommended A100 Validation

After pulling this commit on the server:

```bash
cd /path/to/alpagym
uv lock --check
uv run --no-sync --all-packages pytest packages/policies/autovla
PYTHONPYCACHEPREFIX=/tmp/alpagym-pycache \
  uv run --no-sync --all-packages python -m py_compile \
  packages/policies/autovla/src/alpagym_autovla/*.py
```

Then run a short AutoVLA smoke:

```bash
EXPERIMENT=autovla_local_1gpu_smoke scripts/run_autovla_smoke.sh
```

Watch logs for:

```text
AutoVLA decode: completion_len=... n_action_tokens=...
AutoVLA training forward shapes: ... completion_ids=(...,)
AlpaGym trainer minibatch ... grad_norm=...
```

Healthy signs:

- `n_action_tokens` is normally close to `num_poses` (default 10);
- `completion_ids` length is a sequence, not `(1,)`;
- `new_logprob_mean` is not stuck at exactly 0.0;
- `grad_norm` is nonzero on non-padding minibatches;
- PPO ratio is not a fixed huge constant across every minibatch.

If startup fails with an action-token layout error, check that:

- `policy.model.bundle_config.action_start_id` matches the AutoVLA checkpoint;
- `MODEL_PATH` points to the Qwen/AutoVLA tokenizer that matches the SFT
  checkpoint;
- if using raw Qwen tokenizer, its base vocab length must equal
  `action_start_id` so the runtime append lands at the trained rows.

## Remaining Research/Engineering Work

The current fix makes the discrete action-token RL path safer. It does not yet
solve:

- KL/reference scoring for AutoVLA (`kl_beta` still effectively returns no KL);
- step-level reward decomposition beyond episode-level GRPO advantage;
- invalid/short action-token penalty in reward;
- multi-GPU policy-to-rollout checksum validation;
- formal Hydra presets for the full A100 run instead of `sed`-editing generated
  configs in `scripts/launch_cosmos_a100.sh`.

These are the next items to address before treating training gains as paper-grade
evidence.
