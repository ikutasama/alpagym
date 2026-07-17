# AutoVLA Four-A100 GRPO Run - 2026-07-16

This note is for the A100-side operator or coding agent after pulling the latest
`ikutasama/alpagym` `main` branch. The RTX 5090 AlpaSim runtime and the existing
SSH tunnels on ports 5011/5012 must remain running; this procedure does not
restart or modify them.

## Why the old launch produced too few rollouts

The old `scripts/launch_cosmos_a100.sh` changed `n_generation` only in
`cosmos_config.toml`. AlpaGym's streaming rollout backend reads the same value
from `resolved_config.yaml`, where it remained at the smoke-test value of 2.
That allowed Cosmos-RL and the gRPC rollout backend to disagree about GRPO group
size. The same mismatch existed for replica counts when changing launcher flags
without changing the resolved config.

The new configurator updates both files together and validates global batch
geometry before any GPU process starts.

## Four-GPU topology with the current tunnels

The existing 5012 tunnel exposes one A100 Egodriver port to the 5090. Each
rollout replica starts its own Egodriver server, so three rollout replicas would
all attempt to bind port 5012. The safe four-card allocation is therefore:

```text
GPU 0: policy replica 0
GPU 1: policy replica 1
GPU 2: policy replica 2
GPU 3: rollout replica 0 / Egodriver port 5012
```

The matched global batch is:

```text
rollout: 1 replica * batch_size 3 * n_generation 8 = 24 episodes
policy:  3 replicas * train_batch_per_replica 8       = 24 episodes
```

GRPO advantage normalization uses eight candidates per prompt instead of the
previous two. Three prompt groups are produced per rollout cycle. The policy
side performs synchronous data-parallel training with one full AutoVLA model on
each of three A100s; `dp_shard_size=1` avoids FSDP all-gather overhead.

The A100-side replay transport is switched from disk JSON to NCCL. AutoVLA
episode artifacts contain large multi-camera tensors, so materializing 24 JSON
artifacts per cycle would waste substantial disk bandwidth and space. This NCCL
channel is local to the A100 processes; the 5090 connection remains the existing
gRPC tunnel on ports 5011/5012.

## Stability settings

The launch defaults are deliberately conservative:

```text
n_generation=8
train_batch_per_replica=8
mini_batch=2
learning_rate=1e-6
warmup_steps=20
PPO ratio clip=0.1
allowed_outdated_steps=1
GRPO optimization iterations=1
gradient norm clip=1.0
prefetch_rollout=false
replay transport=NCCL (A100-local)
cosine decay to 0.1 * initial LR
```

`mini_batch=2` is the first setting to reduce to 1 if an 80 GB A100 runs out of
memory. Do not reduce `n_generation` below 8 for the main experiment; that would
return to the high-variance group estimate.

## Start on the A100 server

Do not restart the 5090 process. Confirm its tunnel loop is still running, then:

```bash
cd /data/mnt_m62/10_personal/z59900495/workspace/alpagym
git pull origin main
bash scripts/launch_cosmos_a100.sh 4gpu
```

The script checks that A100 localhost port 5011 is reachable and that local port
5012 is free for the rollout worker. It then updates the copied run artifacts in
`/data/mnt_m62/10_personal/z59900495/workspace/latest` and launches three policy
workers plus one rollout worker on GPUs 0,1,2,3.

For a short first run without changing code:

```bash
MAX_NUM_STEPS=10 SAVE_FREQ=5 bash scripts/launch_cosmos_a100.sh 4gpu
```

For a lower-memory fallback, copy the four-GPU profile, change `mini_batch` to
1, and launch the copied profile explicitly:

```bash
cp packages/policies/autovla/src/alpagym_autovla/configs/a100/autovla_a100_4gpu.yaml /tmp/autovla_a100_4gpu_mb1.yaml
sed -i 's/mini_batch: 2/mini_batch: 1/' /tmp/autovla_a100_4gpu_mb1.yaml
bash scripts/launch_cosmos_a100.sh /tmp/autovla_a100_4gpu_mb1.yaml
```

If this specific A100 host cannot use GPU P2P, retry with:

```bash
NCCL_P2P_DISABLE=1 bash scripts/launch_cosmos_a100.sh 4gpu
```

## What to verify in logs

Before trusting a training curve, verify all of the following:

```text
launcher: --policy 3 --rollout 1
Configured AutoVLA A100 profile 'autovla_a100_4gpu_grpo': ... global_episodes=24 ...
NCCL writer/receiver setup completes for 3 policy ranks and 1 rollout rank
rollout_generation: 3 payloads
each RolloutResult has 8 completions (no Partial rollout warning)
trainer received_rollouts is stable across all three policy replicas
ratio_min/ratio_max stay near 1 early in training
clip_fraction is not pinned near 1.0
grad_norm is finite and nonzero
```

Loss magnitude alone is not a reliable success metric. A jump such as 0.5 to 80
usually accompanies a large old/new log-probability mismatch or a negative
advantage multiplied by a very large importance ratio. Stop and inspect the
ratio, clip fraction, action-token count, and reward dispersion if this repeats
after the action-token and full-group fixes.

For paper-grade comparison, save the same fixed evaluation scene list and random
seeds for baseline, GRPO-8, and each reward ablation. Report mean and bootstrap
confidence intervals across scenes/seeds rather than selecting only the best
checkpoint or a single favorable rollout.
