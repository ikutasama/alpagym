# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse AlpaSim's built-in Prometheus telemetry into a per-rollout phase table.

AlpaSim writes one `metrics_worker_<id>.prom` per sim worker on graceful shutdown.
The per-method `rpc_duration_seconds{service,method,tag}` histogram already decomposes
a rollout: sensorsim/render_* = render(NRE), physics/ground_intersection = physics,
driver/drive = the per-tick policy callback, controller = controller. This sums the
histogram sum/count across workers, normalizes by total rollout count, and renders a
render/physics/drive breakdown by phase.

The summary CLI (`analysis.cli`) discovers the `.prom` files under the run directory and
renders this as the "Simulator internals" section, so the whole run reads as one view.
These functions are that section's implementation; there is no separate entrypoint.

stdlib only. Caveat: rpc_duration is wall-clock-across-await, so it INCLUDES queue-wait
under concurrent rollouts and does NOT cleanly partition the rollout -- `rpc_blocking`
(scheduler delay) and the idle% are printed alongside to net that out.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

from alpagym_runtime.perf.analysis import _format

# One sample line: name{labels} value  |  name value   (trailing timestamp optional)
# Value regex matches finite decimals only; +Inf/-Inf/NaN are not produced by the
# metrics this tool consumes (_sum/_count histogram fields), so they are not handled.
_LINE = re.compile(r"^([a-zA-Z_:][\w:]*)\{(.*)\}\s+([0-9eE.+-]+)\s*\S*$")
_BARE = re.compile(r"^([a-zA-Z_:][\w:]*)\s+([0-9eE.+-]+)\s*\S*$")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def discover_prom_files(run_dir: Path) -> list[Path]:
    """Return the `metrics_worker_*.prom` files for a run, or `[]` when none exist.

    Accepts the run directory, a telemetry directory, or a single `.prom` file.
    Directory arguments return both direct and nested `metrics_worker_*.prom`
    files with de-duplication. AlpaSim writes these only on graceful shutdown,
    so a killed run legitimately yields `[]`.
    """
    if run_dir.is_file():
        return [run_dir]
    if run_dir.is_dir():
        return sorted(
            set(run_dir.glob("metrics_worker_*.prom")) | set(run_dir.rglob("metrics_worker_*.prom"))
        )
    return []


def _parse_prom(text: str) -> list[tuple[str, dict[str, str], float]]:
    """Parse a Prometheus textfile into (name, labels, value) samples."""
    samples: list[tuple[str, dict[str, str], float]] = []
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        match = _LINE.match(line)
        if match:
            labels = dict(_LABEL.findall(match.group(2)))
            try:
                samples.append((match.group(1), labels, float(match.group(3))))
            except ValueError:
                print(f"warning: skipping unparsable value in line: {line!r}", file=sys.stderr)
            continue
        match = _BARE.match(line)
        if match:
            try:
                samples.append((match.group(1), {}, float(match.group(2))))
            except ValueError:
                print(f"warning: skipping unparsable value in line: {line!r}", file=sys.stderr)
    return samples


def _phase_of(service: str, method: str) -> str:
    """Map a (service, method) RPC to a coarse phase bucket."""
    if service == "sensorsim":
        if method in {"render_rgb", "render_aggregated"}:
            return "render (NRE)"
        return "sensorsim setup"  # get_available_cameras / ego_masks (once per session)
    if service == "physics":
        return "physics"
    if service == "driver":
        if method == "drive":
            return "policy callback (drive)"
        if method.startswith("submit"):
            return "obs push -> policy"
        return "session"  # start/close_session
    if service == "controller":
        if method == "run_controller_and_vehicle":
            return "controller"
        return "session"
    if service in ("traffic", "trafficsim"):
        return "traffic"
    return f"{service}/{method}"


def _print_table(
    header: list[str], rows: list[list[str]], left_cols: set[int], color: bool
) -> None:
    """Print a 2-space-indented table with each column sized to its widest cell.

    Columns in `left_cols` are left-justified (text labels); the rest are right-justified
    (numbers + units), so the values line up vertically down each column. The header row is
    dimmed to match the column headers in the other tables.
    """
    columns = list(zip(header, *rows)) if rows else [(name,) for name in header]
    widths = [max(len(cell) for cell in column) for column in columns]

    def format_row(cells: list[str]) -> str:
        return "  ".join(
            f"{cell:<{widths[i]}}" if i in left_cols else f"{cell:>{widths[i]}}"
            for i, cell in enumerate(cells)
        ).rstrip()

    print("  " + _format.colheader(format_row(header), color))
    for row in rows:
        print("  " + format_row(row))


def render_telemetry(files: list[Path], color: bool = False) -> None:
    """Aggregate the telemetry across `files` and print the Simulator-internals section.

    `color` enables ANSI styling for the sub-header and column headers. The caller
    discovers `files` (see `discover_prom_files`) and skips the section when there are none.
    """
    # Aggregate histogram sum/count by (service, method, tag), and counters per worker.
    rpc_seconds: dict[tuple[str, str, str], float] = defaultdict(float)
    rpc_calls: dict[tuple[str, str, str], float] = defaultdict(float)
    blocking_seconds: dict[str, float] = defaultdict(float)
    blocking_calls: dict[str, float] = defaultdict(float)
    queue_depth_sum: dict[str, float] = defaultdict(float)
    queue_depth_count: dict[str, float] = defaultdict(float)
    # whole-step / whole-rollout histograms (label-less)
    hist_seconds: dict[str, float] = defaultdict(float)
    hist_count: dict[str, float] = defaultdict(float)
    event_loop_counters: dict[str, list[float]] = defaultdict(list)
    total_rollouts = 0.0

    for prom_file in files:
        for name, labels, value in _parse_prom(prom_file.read_text(encoding="utf-8")):
            base = name.rsplit("_", 1)[0]
            suffix = name.rsplit("_", 1)[1] if "_" in name else ""
            if base == "rpc_duration_seconds":
                key = (labels["service"], labels["method"], labels.get("tag", ""))
                if suffix == "sum":
                    rpc_seconds[key] += value
                elif suffix == "count":
                    rpc_calls[key] += value
            elif base == "rpc_blocking_seconds":
                service = labels["service"]
                if suffix == "sum":
                    blocking_seconds[service] += value
                elif suffix == "count":
                    blocking_calls[service] += value
            elif base == "rpc_queue_depth_at_start":
                service = labels["service"]
                if suffix == "sum":
                    queue_depth_sum[service] += value
                elif suffix == "count":
                    queue_depth_count[service] += value
            elif base in (
                "step_duration_seconds",
                "rollout_duration_seconds",
                "dispatch_wait_seconds",
            ):
                if suffix == "sum":
                    hist_seconds[base] += value
                elif suffix == "count":
                    hist_count[base] += value
            elif name == "simulation_rollout_count":
                total_rollouts += value
            elif name in (
                "event_loop_idle_seconds_total",
                "event_loop_poll_seconds_total",
                "event_loop_work_seconds_total",
            ):
                # Sum across workers regardless of labels: merged .prom files add
                # per-worker labels to these counters, and summing monotonic
                # counters is the correct cross-worker total.
                event_loop_counters[name].append(value)

    if total_rollouts == 0:
        print(
            "Warning: simulation_rollout_count not found; /roll and s/roll columns show "
            "totals across all workers, not per-rollout averages.",
            file=sys.stderr,
        )
    per_roll_denom = total_rollouts if total_rollouts > 0 else 1.0

    print(
        _format.subheader("Simulator internals", color)
        + f"   sim_step_rpc breakdown; {len(files)} worker file(s), "
        + f"{int(total_rollouts)} rollouts"
    )
    # anchors
    idle_seconds = sum(event_loop_counters.get("event_loop_idle_seconds_total", []))
    work_seconds = sum(event_loop_counters.get("event_loop_work_seconds_total", []))
    poll_seconds = sum(event_loop_counters.get("event_loop_poll_seconds_total", []))
    event_loop_total = idle_seconds + work_seconds + poll_seconds
    if hist_count.get("rollout_duration_seconds"):
        rollout_samples = int(hist_count["rollout_duration_seconds"])
        mean_rollout_s = (
            hist_seconds["rollout_duration_seconds"] / hist_count["rollout_duration_seconds"]
        )
        print(
            f"  mean rollout_duration : {_format.duration_str(mean_rollout_s * 1000)}  "
            f"(n={rollout_samples})"
        )
        if total_rollouts and hist_count["rollout_duration_seconds"] != total_rollouts:
            print(f"  per-roll denominator  : simulation_rollout_count={int(total_rollouts)}")
            print(f"  completed rollouts    : rollout_duration samples={rollout_samples}")
            print(
                "  "
                + _format.warning(
                    "WARNING: per-roll columns use simulation_rollout_count, "
                    "not completed rollout count.",
                    color,
                )
            )
    if hist_count.get("step_duration_seconds"):
        mean_step_s = hist_seconds["step_duration_seconds"] / hist_count["step_duration_seconds"]
        step_samples = int(hist_count["step_duration_seconds"])
        print(
            f"  mean step_duration    : {_format.duration_str(mean_step_s * 1000)}  "
            f"(n={step_samples} steps, {step_samples / per_roll_denom:.0f}/rollout)"
        )
    if event_loop_total:
        idle_pct = idle_seconds / event_loop_total * 100
        work_pct = work_seconds / event_loop_total * 100
        poll_pct = poll_seconds / event_loop_total * 100
        print(
            f"  event-loop split      : idle {idle_pct:.0f}% / work {work_pct:.0f}% / "
            f"poll {poll_pct:.1f}%  (idle = awaiting remote RPCs)"
        )
    if hist_count.get("dispatch_wait_seconds"):
        # dispatch_wait_seconds splits the simulate()-vs-rollout gap.
        # FRONT (slot-wait + IPC + per-job prep) = this value; BACK (result
        # collection + gRPC return) = (policy-side sim_step_rpc - rollout_duration)
        # - dispatch_wait. Requires dispatch_wait._count == rollout_duration._count.
        mean_dispatch_wait_s = (
            hist_seconds["dispatch_wait_seconds"] / hist_count["dispatch_wait_seconds"]
        )
        print(
            f"  mean dispatch_wait    : {_format.duration_str(mean_dispatch_wait_s * 1000)}  "
            f"(n={int(hist_count['dispatch_wait_seconds'])}) -- FRONT half of the "
            f"simulate()-vs-rollout gap (slot-wait + prep)"
        )

    # ---- per (service, method, tag) detail, by total time ----
    print()
    print("  Per-RPC, summed across workers (sum = wall-time-across-await, includes queue-wait):")
    rpc_keys = sorted(rpc_seconds, key=lambda key: -rpc_seconds[key])
    call_counts = [rpc_calls.get(key, 0.0) for key in rpc_keys]
    total_seconds = [rpc_seconds[key] for key in rpc_keys]
    mean_cells = _format.duration_column(
        [
            (total / count * 1000.0) if count else 0.0
            for total, count in zip(total_seconds, call_counts)
        ]
    )
    total_cells = _format.duration_column([total * 1000.0 for total in total_seconds])
    rpc_per_roll_cells = _format.duration_column(
        [total / per_roll_denom * 1000.0 for total in total_seconds]
    )
    display_methods = {
        "get_available_cameras": "get_cameras",
        "get_available_ego_masks": "get_ego_masks",
        "run_controller_and_vehicle": "run_ctrl_vehicle",
        "submit_egomotion_observation": "submit_egomotion",
        "submit_image_observation": "submit_image_obs",
        "submit_recording_ground_truth": "submit_recording_gt",
    }
    rpc_rows = [
        [
            display_methods.get(method, method),
            service,
            tag,
            str(int(count)),
            f"{count / per_roll_denom:.1f}",
            mean_cells[i],
            total_cells[i],
            rpc_per_roll_cells[i],
        ]
        for i, ((service, method, tag), count) in enumerate(zip(rpc_keys, call_counts))
    ]
    _print_table(
        ["method", "service", "tag", "calls", "calls/roll", "mean", "total*", "s/roll"],
        rpc_rows,
        left_cols={0, 1, 2},
        color=color,
    )

    # ---- phase rollup: steady-state (tag != warmup) ----
    phase_seconds: dict[str, float] = defaultdict(float)
    phase_calls: dict[str, float] = defaultdict(float)
    warmup_seconds = 0.0
    for (service, method, tag), seconds in rpc_seconds.items():
        if tag == "warmup":
            warmup_seconds += seconds
            continue
        phase = _phase_of(service, method)
        phase_seconds[phase] += seconds
        phase_calls[phase] += rpc_calls.get((service, method, tag), 0.0)
    print()
    print("  Phase rollup (steady-state, tag!=warmup), per rollout:")
    phases = sorted(phase_seconds, key=lambda phase: -phase_seconds[phase])
    phase_per_roll_ms = [phase_seconds[phase] / per_roll_denom * 1000.0 for phase in phases]
    if warmup_seconds:
        phase_per_roll_ms.append(warmup_seconds / per_roll_denom * 1000.0)
    phase_per_roll_cells = _format.duration_column(phase_per_roll_ms)
    phase_rows = [
        [phase, f"{phase_calls[phase] / per_roll_denom:.1f}", phase_per_roll_cells[i]]
        for i, phase in enumerate(phases)
    ]
    if warmup_seconds:
        phase_rows.append(["(warmup, one-time)", "-", phase_per_roll_cells[-1]])
    _print_table(["phase", "calls/roll", "s/roll"], phase_rows, left_cols={0}, color=color)
    if warmup_seconds:
        print("  (warmup is one-time; excluded from the phases above, amortizes over a long run)")

    # ---- queue-wait signal ----
    if blocking_seconds:
        print()
        print("  Queue-wait (rpc_blocking = event-loop scheduler delay after I/O completes):")
        for service in sorted(blocking_seconds, key=lambda service: -blocking_seconds[service]):
            count = blocking_calls.get(service, 0.0)
            mean_ms = (blocking_seconds[service] / count * 1000) if count else 0.0
            queue_depth = (
                queue_depth_sum.get(service, 0.0) / queue_depth_count[service]
                if queue_depth_count.get(service)
                else 0.0
            )
            blocking_total = _format.duration_str(blocking_seconds[service] * 1000)
            print(
                f"  {service:<14} blocking total {blocking_total:>10} "
                f"({mean_ms:.1f} ms/call)  mean queue-depth {queue_depth:.1f}"
            )

    print()
    print("  total*: per-RPC totals sum wall-time-across-await over concurrent rollouts (incl.")
    print("          queue-wait), so they OVERLAP and do NOT partition sim_step_rpc's wall-time.")
    print(
        "          Anchor on step/rollout duration; use idle% + rpc_blocking to split compute/wait."
    )
