# AutoVLA A100 GRPO Profiles - 2026-07-17

This is the handoff note for running AutoVLA GRPO after pulling the latest
`ikutasama/alpagym` `main` branch on the A100 server. The RTX 5090 AlpaSim
runtime and the existing SSH tunnels on ports 5011/5012 must stay running. The
commands below only replace the A100-side rollout/policy launch.

## Available profiles

Training geometry now lives in versioned YAML instead of shell defaults:

```text
packages/policies/autovla/src/alpagym_autovla/configs/a100/
  autovla_a100_1gpu.yaml
  autovla_a100_4gpu.yaml
```

The launcher reads the selected profile, validates it, writes matching values
to both `latest/resolved_config.yaml` and `latest/cosmos_config.toml`, and then
passes the same replica counts to Cosmos. It exits before allocating a model if
the GPU count, GRPO group size, or rollout/policy global batch does not match.

| Profile | Placement | GRPO episodes per update | Replay |
| --- | --- | ---: | --- |
| `1gpu` | policy + rollout colocated on GPU 0 | `1 * 1 * 8 = 8` | disk |
| `4gpu` | 3 policy on GPUs 0-2 + 1 rollout on GPU 3 | `1 * 3 * 8 = 24` | NCCL |

Both profiles use `n_generation=8`. The 1GPU version is slower and has only one
prompt group per update, but it does not fall back to the old two- or four-sample
GRPO estimate. Its conservative `mini_batch=1` is intended to fit the colocated
policy and rollout path on one 80 GB A100.

The 1GPU profile uses `mode=colocated` and disk replay because the current
AlpaGym/Cosmos packer supports NCCL replay only between disaggregated rollout
and policy processes. The 4GPU profile uses `mode=disaggregated` and A100-local
NCCL replay. Neither choice changes gRPC communication with the 5090.

## Start a run

On the A100 server:

```bash
cd /data/mnt_m62/10_personal/z59900495/workspace/alpagym
git pull origin main
```

Start the single-GPU profile:

```bash
bash scripts/launch_cosmos_a100.sh 1gpu
```

Start the four-GPU profile:

```bash
bash scripts/launch_cosmos_a100.sh 4gpu
```

Omitting the argument still selects `4gpu` for backward compatibility. A short
configuration/checkpoint smoke run can override only the run length:

```bash
MAX_NUM_STEPS=10 SAVE_FREQ=5 bash scripts/launch_cosmos_a100.sh 1gpu
MAX_NUM_STEPS=10 SAVE_FREQ=5 bash scripts/launch_cosmos_a100.sh 4gpu
```

Core training geometry is intentionally not overridden through environment
variables. To run an ablation, copy one YAML profile, change its values, commit
the profile with the experiment, and pass its path to the launcher:

```bash
bash scripts/launch_cosmos_a100.sh /absolute/path/to/ablation.yaml
```

The profile validator requires:

```text
rollout_replicas * rollout_batch_size * n_generation
  == policy_replicas * train_batch_per_replica
```

It also keeps `rollout_replicas=1` while only one reverse-accessible Egodriver
port (5012) exists. Increasing rollout replicas requires one independent driver
port and SSH tunnel per replica, plus corresponding runtime support; changing
the YAML alone is deliberately rejected.

## First-run checks

For `1gpu`, logs should report:

```text
profile='autovla_a100_1gpu_grpo'
mode=colocated, transport=disk
policy=1, rollout=1, global_episodes=8, n_generation=8
launcher flags: --policy 1 --rollout 1
```

For `4gpu`, logs should report:

```text
profile='autovla_a100_4gpu_grpo'
mode=disaggregated, transport=nccl
policy=3, rollout=1, global_episodes=24, n_generation=8
launcher flags: --policy 3 --rollout 1
```

In both runs, verify every `RolloutResult` contains eight completions and there
are no partial-group warnings. Track reward mean/std, valid action-token count,
importance-ratio min/max, clip fraction, KL, gradient norm, and evaluation
trajectory metrics. Loss magnitude by itself is not a reliable GRPO progress
signal.

The configurator creates one untouched backup of each copied configuration as
`resolved_config.yaml.pre_autovla_a100` and
`cosmos_config.toml.pre_autovla_a100`. Full CUDA, NCCL, and AlpaSim execution
must be verified on the A100/5090 hosts; local tests cover profile loading,
cross-field validation, and synchronized YAML/TOML generation.
