# AlpaGym runtime performance instrumentation

Lightweight, opt-in timing and CPU/GPU resource attribution for AlpaGym workers.
Each worker process writes one JSON artifact; the `analysis` CLI summarizes them.

## Table of Contents

- [Enabling](#enabling)
- [Public API](#public-api)
- [Terminology](#terminology)
- [Categories](#categories)
- [Scope naming](#scope-naming)
- [JSON artifact contract](#json-artifact-contract)
- [Summary CLI](#summary-cli)

The package is split so runtime hooks never pull in offline tooling:

- `instrument/` — the runtime hot path: scopes, markers, resource readers, stores,
  periodic monitors, and the worker lifecycle.
- `analysis/` — offline consumers: `cli` summarizes the AlpaGym JSON artifacts in one
  view. It also discovers AlpaSim's Prometheus telemetry under the run directory and
  renders it (via `alpasim_telemetry`) as the "Simulator internals" section, so the whole
  run reads as a single output.

## Enabling

Instrumentation is off by default. Turn it on in the run config (`perf` block in
`conf/default.yaml`):

```yaml
perf:
  enabled: true
  collect_cpu: true
  collect_gpu: true
  resource_sample_interval_s: 5.0   # null disables the periodic monitor
  sample_every_n: 1                 # reservoir sampling stride for timing
  max_samples_per_series: 1000      # per-series reservoir cap (percentiles)
  flush_every_n_updates: 1000       # count trigger for a background flush
  flush_interval_s: 60.0            # time trigger for a background flush
```

When `enabled` is `false` every public API short-circuits with a single store lookup,
and `pynvml`/`psutil` are never imported.

## Public API

Import from the submodules directly (the package `__init__` exposes no re-exports):

| API                                                                   | Module                 | Purpose                                                                            |
| --------------------------------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------- |
| `initialize_perf(resolved_config)`                                    | `instrument.lifecycle` | Build the process store + readers + monitors.                                      |
| `shutdown_perf()`                                                     | `instrument.lifecycle` | Stop monitors, write the final artifact, shut down NVML.                           |
| `measure_perf(name, *, category, ...)`                                | `instrument.scope`     | Decorator timing a whole function as one scope.                                    |
| `timed_scope(name, *, category, ...)`                                 | `instrument.scope`     | Context manager timing a block inside a function.                                  |
| `record_perf_marker(name, *, cpu_snapshot=False, gpu_snapshot=False)` | `instrument.marker`    | Snapshot CPU/GPU at a named instant; set at least one flag, or it records nothing. |

```python
from alpagym_runtime.perf.instrument.scope import measure_perf, timed_scope

@measure_perf("driver/drive", category="external_rpc")
def drive(self, request, context): ...

with timed_scope("driver/policy_step", category="compute_gpu_wall", gpu_snapshot=True):
    output = policy.step(policy_input)
```

Use a decorator when the scope is exactly one owned function's body and the scope name
is that function's role at every call site; use `timed_scope` when timing only part of a
function, a call into a shared/boundary helper, or an interface method, or when the scope
name is caller-context (e.g. `driver/...`, `rollout/...`).

## Terminology

| Term              | Meaning                                                                                                                                                           |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scope             | Named duration interval (timing, plus optional boundary resource snapshots).                                                                                      |
| Marker            | Named instant with no duration, for lifecycle/resource milestones.                                                                                                |
| Resource snapshot | One CPU or GPU reading produced by a reader.                                                                                                                      |
| Scope path        | Runtime nesting path, e.g. `("rollout/generate", "rollout/sim_request_build")`. A `/` inside a name is naming convention only — never split a path element on it. |
| Capture           | `periodic` (monitor cadence) or `checkpoint` (scope boundary or marker).                                                                                          |
| Phase             | `start`, `end`, or `instant`.                                                                                                                                     |

## Categories

The `category` string groups scopes in the summary; pick the closest:

| Category               | For                                                                          |
| ---------------------- | ---------------------------------------------------------------------------- |
| `orchestration`        | Coordination/glue CPU work between heavier steps.                            |
| `external_rpc`         | Blocking wait on an out-of-process call (e.g. the AlpaSim gRPC tick).        |
| `compute_cpu`          | CPU-bound compute.                                                           |
| `compute_gpu_wall`     | Wall-clock around a GPU op (queue + launch + execute, not pure device time). |
| `io`                   | Disk / artifact I/O.                                                         |
| `synchronization_wait` | Blocking on a barrier, lock, or queue.                                       |

## Scope naming

Scope and marker names follow `<stage>/<operation>[/<sub_op>]` in `lower_snake`, where `/` is a
namespace separator — a path element is never split on it (see Terminology).

- `<stage>` is the owning subsystem: `worker` (process lifecycle), `rollout` (rollout-worker
  experience generation), `driver` (the AlpaSim per-tick policy callback), or `trainer` (the
  policy/optimizer worker).
- `<operation>` names the step with a consistent verb across subsystems: builders end `_build`;
  artifact I/O uses `save`/`load` at the artifact level and `write`/`read` at the disk level;
  readiness markers end `_ready`; blocking RPC waits end `_rpc`.

Names live inline at the call site (one per site); there is no central name registry.

## JSON artifact contract

Each worker writes `{run_dir}/perf/alpagym_perf_{role}_{rank}_{pid}.json` (`schema_version: 1`).
Top level: worker identity (`run_id`, `role`, `rank`, `local_rank`, `world_size`, `hostname`,
`pid`, `device`), the resolved `config`, and `timing`, plus `cpu`/`gpu` when collected.

```json
{
  "schema_version": 1,
  "project": "alpagym",
  "role": "rollout", "rank": 0, "local_rank": 0, "device": "cuda:0",
  "config": { "enabled": true, "collect_cpu": true, "collect_gpu": true, "...": "..." },
  "timing": {
    "clock": "time.perf_counter_ns",
    "scopes": [
      {
        "path": ["rollout/generate", "rollout/sim_request_build"],
        "name": "rollout/sim_request_build", "category": "orchestration",
        "count": 100, "total_ms": 2500.0,
        "mean_ms": 25.0, "p50_ms": 24.0, "p95_ms": 40.0, "max_ms": 55.0
      }
    ]
  },
  "cpu": {
    "source": "psutil",
    "periodic": [ { "capture": "periodic", "name": "sample", "phase": "instant", "count": 24,
                    "system": { "cpu_util_percent": { "mean": 61.2, "p95": 88.0, "max": 94.0 }, "...": "..." },
                    "process": { "rss_mb": { "mean": 12400.0, "p95": 15100.0, "max": 15700.0 }, "...": "..." } } ],
    "checkpoints": [ { "capture": "checkpoint", "path": ["rollout/generate"],
                       "name": "rollout/generate", "phase": "start", "count": 100, "...": "..." } ]
  },
  "gpu": {
    "sources": { "device": "nvml", "process": ["torch.cuda", "nvml_process_query"] },
    "device_index": 0, "device_uuid": "GPU-...", "device_name": "NVIDIA H100",
    "periodic": [ { "capture": "periodic", "name": "sample", "phase": "instant", "count": 24,
                    "device": { "gpu_util_percent": { "mean": 72.1, "p95": 98.0, "max": 100.0 }, "...": "..." },
                    "process": { "torch_allocated_mb": { "max": 27100.0 }, "...": "..." } } ],
    "checkpoints": []
  }
}
```

Notes:

- `mean` is exact over every reading; `p50`/`p95` come from a bounded reservoir, so they stay
  representative once `max_samples_per_series` is reached.
- GPU device identity (`device_index`/`device_uuid`/`device_name`) is emitted once at the `gpu`
  block, not per row. `device_index` is the `torch.cuda` visible index, resolved to the NVML
  handle by UUID — do not pass it back to NVML as a physical index.
- The driver per-tick scopes (`driver/drive`, `driver/input_build`,
  `driver/output_build`) are timing-only: per-tick `psutil` snapshots would dominate their own
  durations, so driver CPU trends come from the periodic monitor and the once-per-session
  `driver/session_start` scope instead.

## Summary CLI

```bash
python -m alpagym_runtime.perf.analysis.cli --perf-dir {run_dir}/perf
# rank 0 only by default.  --all-ranks  --role rollout  --rank 1  --category external_rpc  --capture periodic
```

The CLI processes only rank 0 by default, so the output stays bounded on multi-rank runs; pass
`--all-ranks` to include every rank. When more than one worker is merged (several processes on a
rank, or `--all-ranks`), the TOTAL and percentages sum total_ms across those workers rather than a
single timeline, and the CLI prints a note; narrow with `--role <role>` / `--rank <n>`.

The GPU/CPU resource tables pool workers per `(role, rank)` so a rank with many processes stays one
row: `mean` is the pooled mean, `max` is exact, `p95` is the max of the per-worker p95s, and the
`Procs` column shows how many workers merged. Within each `(role, rank)` block the periodic `sample`
comes first, then the scope checkpoints in the order each scope first runs, with each scope's `start`
immediately before its `end`.

To attribute time inside the AlpaSim simulator itself (render / physics / driver / controller),
the CLI auto-discovers AlpaSim's Prometheus telemetry (`metrics_worker_*.prom`, written under the
run directory on graceful shutdown) and prints it as the "Simulator internals" block within the
Timing section, since it decomposes the `sim_step_rpc` scope. No separate command is needed; the
block is skipped when no `.prom` files exist.
Its render/physics/drive RPC sums overlap across concurrent rollouts and include queue-wait, so
they do not partition `sim_step_rpc`'s wall-time — the section prints that caveat inline.
