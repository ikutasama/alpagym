# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summary CLI for alpagym per-process performance artifacts.

Reads `{run_dir}/perf/alpagym_perf_*.json` (or the directory passed via
`--perf-dir`), merges timing scopes by path across files, and prints one
text summary covering timing, CPU, and GPU. Runtime hot paths should not
import this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, NamedTuple

from alpagym_runtime.perf.analysis import _format, alpasim_telemetry

_SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({1})


@dataclass
class _LoadedArtifact:
    """One parsed `alpagym_perf_*.json` file."""

    path: Path
    payload: dict[str, Any]

    @property
    def role(self) -> str:
        """Return the worker role recorded in the artifact."""
        return str(self.payload["role"])

    @property
    def rank(self) -> int:
        """Return the global rank recorded in the artifact."""
        return int(self.payload["rank"])

    @property
    def local_rank(self) -> int:
        """Return the local rank recorded in the artifact."""
        return int(self.payload["local_rank"])

    @property
    def hostname(self) -> str:
        """Return the hostname recorded in the artifact."""
        return str(self.payload["hostname"])

    @property
    def pid(self) -> int:
        """Return the worker pid recorded in the artifact."""
        return int(self.payload["pid"])

    @property
    def device(self) -> str:
        """Return the worker device string recorded in the artifact."""
        return str(self.payload["device"])

    @property
    def run_id(self) -> str:
        """Return the run id recorded in the artifact."""
        return str(self.payload["run_id"])


@dataclass
class _MergedTimingScope:
    """Per-path timing aggregate merged across worker files."""

    name: str
    category: str
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    children: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        """Return total_ms / count, or 0.0 when no samples recorded."""
        return self.total_ms / self.count if self.count else 0.0


class _TimingRow(NamedTuple):
    """One timing-table row with unformatted numerics so the caller can size columns.

    `excl_pct` is `None` when merged child scopes exceed their parent, i.e. the row is a
    pooled per-scope total rather than part of a strict tree.
    """

    component: str
    category: str
    count: int
    mean_ms: float
    total_ms: float
    incl_pct: float
    excl_pct: float | None


def main(argv: list[str] | None = None) -> int:
    """Entry point for `python -m alpagym_runtime.perf.analysis.cli`."""
    args = _parse_args(argv)
    perf_dir = _resolve_perf_dir(args.perf_dir)
    artifacts = _load_artifacts(perf_dir)
    if not artifacts:
        print(f"No alpagym_perf_*.json files found under {perf_dir}", file=sys.stderr)
        return 1
    rank = None if args.all_ranks else args.rank
    filtered = _apply_filters(
        artifacts,
        role=args.role,
        rank=rank,
        hostname=args.hostname,
        device=args.device,
    )
    if not filtered:
        print("No files match the requested filters.", file=sys.stderr)
        return 1
    color = _format.color_enabled()
    _print_header(filtered, color=color, all_ranks=args.all_ranks)
    _format.print_section("Timing", color)
    _print_timing(filtered, category_filter=args.category)
    _print_alpasim_internals(
        perf_dir,
        color,
        artifact_filter_active=len(filtered) != len(artifacts),
    )
    _format.print_section("GPU", color)
    _print_gpu(filtered, capture_filter=args.capture)
    _format.print_section("CPU", color)
    _print_cpu(filtered, capture_filter=args.capture)
    if len(filtered) > len({(artifact.role, artifact.rank) for artifact in filtered}):
        print(
            "(GPU/CPU rows pool workers per role/rank: mean is the pooled mean, max is exact, "
            "p95 ~ max of per-worker p95s; Procs = workers merged)"
        )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the CLI's argument vector."""
    parser = argparse.ArgumentParser(
        prog="python -m alpagym_runtime.perf.analysis.cli",
        description="Print one summary across alpagym performance JSON artifacts.",
    )
    parser.add_argument(
        "--perf-dir",
        type=Path,
        required=True,
        help="Directory containing alpagym_perf_*.json files, or a run directory with perf/.",
    )
    parser.add_argument("--role", help="Filter to one worker role (e.g. rollout, trainer).")
    rank_group = parser.add_mutually_exclusive_group()
    rank_group.add_argument(
        "--rank", type=int, default=0, help="Process one global rank (default: 0)."
    )
    rank_group.add_argument(
        "--all-ranks",
        action="store_true",
        help="Process every rank; resource rows are still aggregated by role/rank/device.",
    )
    parser.add_argument("--hostname", help="Filter to one hostname.")
    parser.add_argument("--device", help="Filter to one device string (e.g. cuda:0).")
    parser.add_argument("--category", help="Filter the timing table to one category.")
    parser.add_argument(
        "--capture",
        choices=("periodic", "checkpoint"),
        help="Filter CPU/GPU tables to one capture mode.",
    )
    return parser.parse_args(argv)


def _resolve_perf_dir(arg: Path) -> Path:
    """Allow `--perf-dir` to point at the run dir; fall back to `<dir>/perf/`."""
    if arg.is_dir() and arg.name == "perf":
        return arg
    nested = arg / "perf"
    if nested.is_dir():
        return nested
    if not arg.is_dir():
        raise SystemExit(f"--perf-dir {arg} is not a directory")
    return arg


def _load_artifacts(perf_dir: Path) -> list[_LoadedArtifact]:
    """Load every `alpagym_perf_*.json` under `perf_dir`."""
    artifacts: list[_LoadedArtifact] = []
    for path in sorted(perf_dir.glob("alpagym_perf_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = int(payload.get("schema_version", -1))
        if version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise SystemExit(
                f"Unsupported schema_version {version} in {path}; expected "
                f"{sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
            )
        artifacts.append(_LoadedArtifact(path=path, payload=payload))
    return artifacts


def _apply_filters(
    artifacts: list[_LoadedArtifact],
    role: str | None,
    rank: int | None,
    hostname: str | None,
    device: str | None,
) -> list[_LoadedArtifact]:
    """Restrict the artifact set by the optional identity filters."""
    result = artifacts
    if role is not None:
        result = [a for a in result if a.role == role]
    if rank is not None:
        result = [a for a in result if a.rank == rank]
    if hostname is not None:
        result = [a for a in result if a.hostname == hostname]
    if device is not None:
        result = [a for a in result if a.device == device]
    return result


def _print_header(artifacts: list[_LoadedArtifact], color: bool, all_ranks: bool) -> None:
    """Print the run-level header summarizing the included artifact set."""
    roles_count: dict[str, int] = {}
    ranks: list[int] = []
    for artifact in artifacts:
        roles_count[artifact.role] = roles_count.get(artifact.role, 0) + 1
        ranks.append(artifact.rank)
    run_id = artifacts[0].run_id
    roles_text = ",".join(f"{role}:{count}" for role, count in sorted(roles_count.items()))
    rank_span = f"{min(ranks)}-{max(ranks)}" if ranks else "n/a"
    print(_format.bold("AlpaGym Performance Summary", color=color))
    print(
        f"run_id={run_id} worker_files={len(artifacts)} "
        f"workers_by_role={roles_text} ranks={rank_span}"
    )
    if not all_ranks:
        print("(one rank only — pass --all-ranks to include every rank)")
    print()


def _print_timing(
    artifacts: list[_LoadedArtifact],
    category_filter: str | None,
) -> None:
    """Print the timing table with inclusive and exclusive percentages."""
    color = _format.color_enabled()
    scopes = _merge_timing(artifacts)
    if not scopes:
        print("(no scopes recorded)")
        print()
        return
    # With more than one worker merged, root scopes are summed by path across
    # workers (data-parallel ranks and/or concurrent roles), so this denominator
    # double-counts wall-clock rather than reflecting one timeline; the notes flag it.
    root_total_ms = sum(scope.total_ms for path, scope in scopes.items() if len(path) == 1)
    grouped_rows = list(_collect_timing_groups(scopes, root_total_ms, category_filter))
    rows = [row for _group, group_rows in grouped_rows for row in group_rows]
    if not rows:
        print(f"(no scopes in category {category_filter!r})")
        print()
        return
    summary_count = sum(scope.count for path, scope in scopes.items() if len(path) == 1)

    # Format Mean and Total as aligned `number unit` cells. The TOTAL row's cell is the
    # last `total` entry, so it lines up with the Total column above it.
    mean_cells = _format.duration_column([row.mean_ms for row in rows])
    total_cells = _format.duration_column([row.total_ms for row in rows] + [root_total_ms])

    counts = [f"{row.count:,}" for row in rows] + [f"{summary_count:,}"]
    excl_cells = ["n/a" if row.excl_pct is None else f"{row.excl_pct:.1f}%" for row in rows]
    percents = [f"{row.incl_pct:.1f}%" for row in rows] + excl_cells
    w_component = max(len("Component"), len("TOTAL*"), *(len(row.component) for row in rows))
    w_category = max(len("Category"), *(len(row.category) for row in rows))
    w_count = max(len("Count"), *(len(count) for count in counts))
    w_mean = max(len("Mean"), *(len(cell) for cell in mean_cells))
    w_total = max(len("Total*"), *(len(cell) for cell in total_cells))
    w_percent = max(len("Incl %"), len("Excl %"), len("100.0%"), *(len(p) for p in percents))

    def render(
        component: str, category: str, count: str, mean: str, total: str, incl: str, excl: str
    ) -> str:
        return (
            f"{component:<{w_component}}  {category:<{w_category}}  {count:>{w_count}}  "
            f"{mean:>{w_mean}}  {total:>{w_total}}  {incl:>{w_percent}}  {excl:>{w_percent}}"
        )

    header = render("Component", "Category", "Count", "Mean", "Total*", "Incl %", "Excl %")

    def render_separator(label: str) -> str:
        prefix = f"- {label} "
        if len(prefix) >= len(header):
            return f"- {label}"
        fill_width = len(header) - len(prefix)
        fill = "".join("-" if i % 2 == 0 else " " for i in range(fill_width))
        return prefix + fill.rstrip()

    print(_format.colheader(header, color=color))
    row_index = 0
    for group, group_rows in grouped_rows:
        print(render_separator(group))
        for row in group_rows:
            print(
                render(
                    row.component,
                    row.category,
                    f"{row.count:,}",
                    mean_cells[row_index],
                    total_cells[row_index],
                    f"{row.incl_pct:.1f}%",
                    excl_cells[row_index],
                )
            )
            row_index += 1
    total_line = render("TOTAL*", "", f"{summary_count:,}", "", total_cells[-1], "100.0%", "")
    print(_format.bold(total_line, color=color))
    _print_timing_notes(
        artifacts,
        color=color,
        has_merged_tree_caveat=any(row.excl_pct is None for row in rows),
    )
    print()


def _print_timing_notes(
    artifacts: list[_LoadedArtifact],
    color: bool,
    has_merged_tree_caveat: bool,
) -> None:
    """Print the column legend and any merge caveat below the timing table."""
    print()
    print(_format.subheader("Notes", color=color))
    print("  Indent   child scopes are nested under their parent; the indent shows the tree.")
    print("  Count    number of timed calls, summed across the merged worker files.")
    print("  Mean     average wall time per call (Total* / Count).")
    print(
        "  Total*   wall-time summed over all calls; concurrent scopes overlap in real time\n"
        "           (driver/drive runs INSIDE sim_step_rpc), so it is busy/occupancy, not\n"
        "           wall-clock elapsed -- the lanes are not additive."
    )
    print("  Groups   root scopes are grouped by execution lane; sibling rows sort by Total*.")
    print("  Incl %   this scope AND its children, as a % of the summed root total (TOTAL*).")
    print("  Excl %   this scope's own time only (excludes children), as a % of that total.")
    caveats = []
    if has_merged_tree_caveat:
        caveats.append(
            "    Merged child scopes can exceed their parent; those rows are pooled\n"
            "    per-scope totals, not a strict tree (Excl % = n/a)."
        )
    if len(artifacts) > 1:
        caveats.append(
            "    TOTAL* and percentages sum across merged workers, not one timeline.\n"
            "    Default output covers rank 0; use --all-ranks, --role, or --rank to narrow."
        )
    if caveats:
        print("  Caveats")
        for caveat in caveats:
            print(caveat)


def _merge_timing(artifacts: list[_LoadedArtifact]) -> dict[tuple[str, ...], _MergedTimingScope]:
    """Sum per-path timing aggregates across artifacts."""
    merged: dict[tuple[str, ...], _MergedTimingScope] = {}
    for artifact in artifacts:
        timing = artifact.payload.get("timing")
        if timing is None:
            continue
        for entry in timing.get("scopes", []):
            path = tuple(entry["path"])
            scope = merged.get(path)
            if scope is None:
                scope = _MergedTimingScope(name=entry["name"], category=entry["category"])
                merged[path] = scope
            scope.count += int(entry["count"])
            scope.total_ms += float(entry["total_ms"])
            scope.max_ms = max(scope.max_ms, float(entry["max_ms"]))
    for path, entry in merged.items():
        for candidate in merged:
            if len(candidate) == len(path) + 1 and candidate[: len(path)] == path:
                entry.children.append(candidate)
    return merged


def _collect_timing_groups(
    scopes: dict[tuple[str, ...], _MergedTimingScope],
    root_total_ms: float,
    category_filter: str | None,
) -> Iterable[tuple[str, list[_TimingRow]]]:
    """Yield timing rows grouped by execution lane.

    Each row is `(component, category, count, mean_ms, total_ms, incl_pct, excl_pct)`
    with the numeric values unformatted so the caller can size the columns. When
    `category_filter` is set only scopes in that category are emitted, but the full tree
    is still walked so percentages stay relative to the real root totals and matching
    descendants of non-matching ancestors are not dropped.
    """
    root_groups: dict[str, list[tuple[str, ...]]] = {}
    for path, scope in scopes.items():
        if len(path) == 1:
            group = _timing_group_for_root(scope.name)
            root_groups.setdefault(group, []).append(path)

    group_order = [
        "Rollout episode",
        "Rollout generation / inference",
        "Policy callback / driver",
        "Training",
        "Other",
    ]
    for group in group_order:
        roots = sorted(root_groups.get(group, []), key=lambda p: -scopes[p].total_ms)
        rows: list[_TimingRow] = []
        for root in roots:
            rows.extend(
                _collect_subtree(
                    root, scopes, root_total_ms, depth=1, category_filter=category_filter
                )
            )
        if rows:
            yield group, rows


def _timing_group_for_root(name: str) -> str:
    """Return the display group for a root timing scope."""
    if name == "rollout/episode":
        return "Rollout episode"
    if name.startswith("rollout/"):
        return "Rollout generation / inference"
    if name.startswith("driver/"):
        return "Policy callback / driver"
    if name.startswith("trainer/"):
        return "Training"
    return "Other"


def _collect_subtree(
    path: tuple[str, ...],
    scopes: dict[tuple[str, ...], _MergedTimingScope],
    root_total_ms: float,
    depth: int,
    category_filter: str | None,
) -> Iterable[_TimingRow]:
    """Yield rows for one scope and its descendants by largest-total first."""
    scope = scopes[path]
    children_total = sum(scopes[child].total_ms for child in scope.children)
    if root_total_ms > 0:
        incl_pct = scope.total_ms / root_total_ms * 100.0
        if children_total > scope.total_ms:
            excl_pct = None
        else:
            excl_pct = (scope.total_ms - children_total) / root_total_ms * 100.0
    else:
        incl_pct = 0.0
        excl_pct = None if children_total > scope.total_ms else 0.0
    if category_filter is None or scope.category == category_filter:
        yield _TimingRow(
            component=_format_component(scope.name, depth=depth),
            category=scope.category,
            count=scope.count,
            mean_ms=scope.mean_ms,
            total_ms=scope.total_ms,
            incl_pct=incl_pct,
            excl_pct=excl_pct,
        )
    children_sorted = sorted(scope.children, key=lambda p: -scopes[p].total_ms)
    for child in children_sorted:
        yield from _collect_subtree(
            child, scopes, root_total_ms, depth=depth + 1, category_filter=category_filter
        )


def _format_component(name: str, depth: int) -> str:
    """Format the `Component` column with hierarchical indent based on depth."""
    if depth <= 1:
        return name
    indent = "  " * (depth - 1)
    return f"{indent}{name.rsplit('/', 1)[-1]}"


def _print_cpu(
    artifacts: list[_LoadedArtifact],
    capture_filter: str | None,
) -> None:
    """Print the CPU System and CPU Process tables."""
    rows = _collect_resource_rows(artifacts, family="cpu")
    if capture_filter is not None:
        rows = [row for row in rows if row["capture"] == capture_filter]
    if not rows:
        return
    color = _format.color_enabled()

    sys_cpu = _format_aligned_stat_cells(
        [row["system"]["cpu_util_percent"] for row in rows], ("mean", "p95", "max"), 1.0, 1, "%"
    )
    sys_ram = _format_aligned_stat_cells(
        [row["system"]["memory_used_mb"] for row in rows], ("mean", "max"), 1 / 1024.0, 1, " GB"
    )
    print(_format.subheader("System", color=color))
    _format.render_table(
        [
            "Capture",
            "Name",
            "Phase",
            "Role",
            "Rank",
            "Node",
            "Procs",
            "Count",
            "CPU mean|p95|max",
            "RAM used mean|max",
        ],
        [
            [
                row["capture"],
                row["display_name"],
                row["phase"],
                row["role"],
                str(row["rank"]),
                row["hostname"],
                str(row["procs"]),
                str(row["count"]),
                sys_cpu[i],
                sys_ram[i],
            ]
            for i, row in enumerate(rows)
        ],
        color,
    )
    print()

    proc_cpu = _format_aligned_stat_cells(
        [row["process"]["cpu_util_percent"] for row in rows], ("mean", "p95", "max"), 1.0, 1, "%"
    )
    proc_rss = _format_aligned_stat_cells(
        [row["process"]["rss_mb"] for row in rows], ("mean", "max"), 1 / 1024.0, 1, " GB"
    )
    proc_uss = _format_aligned_stat_cells(
        [row["process"]["uss_mb"] for row in rows], ("max",), 1 / 1024.0, 1, " GB"
    )
    print(_format.subheader("Process", color=color))
    _format.render_table(
        [
            "Capture",
            "Name",
            "Phase",
            "Role",
            "Rank",
            "Procs",
            "Count",
            "CPU mean|p95|max",
            "RSS mean|max",
            "USS max",
            "Threads max",
        ],
        [
            [
                row["capture"],
                row["display_name"],
                row["phase"],
                row["role"],
                str(row["rank"]),
                str(row["procs"]),
                str(row["count"]),
                proc_cpu[i],
                proc_rss[i],
                proc_uss[i],
                str(int(row["process"]["num_threads"]["max"])),
            ]
            for i, row in enumerate(rows)
        ],
        color,
    )
    print()


def _print_gpu(
    artifacts: list[_LoadedArtifact],
    capture_filter: str | None,
) -> None:
    """Print the GPU Device and GPU Process tables."""
    rows = _collect_resource_rows(artifacts, family="gpu")
    if capture_filter is not None:
        rows = [row for row in rows if row["capture"] == capture_filter]
    if not rows:
        return
    color = _format.color_enabled()

    dev_util = _format_aligned_stat_cells(
        [row["device"]["gpu_util_percent"] for row in rows], ("mean", "p95", "max"), 1.0, 1, "%"
    )
    dev_mem = _format_aligned_stat_cells(
        [row["device"]["memory_used_mb"] for row in rows], ("mean", "max"), 1 / 1024.0, 1, " GB"
    )
    print(_format.subheader("Device", color=color))
    _format.render_table(
        [
            "Capture",
            "Name",
            "Phase",
            "Role",
            "Rank",
            "Device",
            "Procs",
            "Count",
            "Util mean|p95|max",
            "Mem used mean|max",
        ],
        [
            [
                row["capture"],
                row["display_name"],
                row["phase"],
                row["role"],
                str(row["rank"]),
                row["device_str"],
                str(row["procs"]),
                str(row["count"]),
                dev_util[i],
                dev_mem[i],
            ]
            for i, row in enumerate(rows)
        ],
        color,
    )
    print()

    torch_alloc = _format_aligned_stat_cells(
        [row["process"].get("torch_allocated_mb") for row in rows], ("max",), 1 / 1024.0, 1, " GB"
    )
    torch_reserved = _format_aligned_stat_cells(
        [row["process"].get("torch_reserved_mb") for row in rows], ("max",), 1 / 1024.0, 1, " GB"
    )
    driver_mem = _format_aligned_stat_cells(
        [row["process"].get("driver_memory_mb") for row in rows], ("max",), 1 / 1024.0, 1, " GB"
    )
    print(_format.subheader("Process", color=color))
    _format.render_table(
        [
            "Capture",
            "Name",
            "Phase",
            "Role",
            "Rank",
            "Device",
            "Procs",
            "Count",
            "Torch alloc max",
            "Torch reserved max",
            "Driver mem max",
        ],
        [
            [
                row["capture"],
                row["display_name"],
                row["phase"],
                row["role"],
                str(row["rank"]),
                row["device_str"],
                str(row["procs"]),
                str(row["count"]),
                torch_alloc[i],
                torch_reserved[i],
                driver_mem[i],
            ]
            for i, row in enumerate(rows)
        ],
        color,
    )
    print()


def _print_alpasim_internals(
    perf_dir: Path,
    color: bool,
    artifact_filter_active: bool,
) -> None:
    """Print the AlpaSim-internal RPC breakdown when telemetry `.prom` files exist.

    Rendered inside the Timing section because it decomposes the `sim_step_rpc` scope.
    The `.prom` files live under the run directory (the parent of `perf/`), written by
    AlpaSim on graceful shutdown. When none are found (e.g. a killed run) the block is
    skipped. The breakdown is sim-internal and currently run-wide; when artifact
    filters are active it is labeled instead of silently pretending to be narrowed.
    """
    run_dir = perf_dir.parent if perf_dir.name == "perf" else perf_dir
    prom_files = alpasim_telemetry.discover_prom_files(run_dir)
    if not prom_files:
        return
    alpasim_telemetry.render_telemetry(prom_files, color=color)
    if artifact_filter_active:
        print(
            "  "
            + _format.warning(
                "WARNING: simulator telemetry is run-wide; "
                "artifact filters do not narrow this block.",
                color,
            )
        )
    print()


def _collect_resource_rows(
    artifacts: list[_LoadedArtifact],
    family: str,
) -> list[dict[str, Any]]:
    """Build CPU or GPU display rows, aggregated to one row per (role, rank).

    Each worker process writes its own artifact, so a (role, rank) can span many
    PIDs. Per-process rows are grouped by (capture, name, phase, role, rank,
    device) and their stats pooled (see `_aggregate_resource_rows`), so the tables
    stay one row per (role, rank) rather than one per process.
    """
    per_file: list[dict[str, Any]] = []
    for artifact in artifacts:
        block = artifact.payload.get(family)
        if block is None:
            continue
        device_str = (
            f"cuda:{block.get('device_index', artifact.local_rank)}"
            if family == "gpu"
            else artifact.hostname
        )
        for entry in block.get("periodic", []) + block.get("checkpoints", []):
            per_file.append(
                {
                    **entry,
                    "role": artifact.role,
                    "rank": artifact.rank,
                    "hostname": artifact.hostname,
                    "device_str": device_str if family == "gpu" else "",
                    "display_name": _display_name_for_resource_row(entry),
                }
            )
    return _aggregate_resource_rows(per_file, family)


def _aggregate_resource_rows(
    rows: list[dict[str, Any]],
    family: str,
) -> list[dict[str, Any]]:
    """Pool per-process rows into one row per (capture, name, phase, role, rank, device).

    Rows are then ordered for reading: grouped by (role, rank, device), the
    periodic `sample` heads each block, then the scope checkpoints follow in the
    order each scope first appears at runtime, with `start` immediately before its
    `end`. `rows` arrives in recorded order, so a scope's first-seen index here is
    its runtime position.
    """
    metric_groups = ("device", "process") if family == "gpu" else ("system", "process")
    first_seen: dict[tuple[Any, ...], int] = {}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        scope_key = (row["role"], row["rank"], row["device_str"], row["display_name"])
        first_seen.setdefault(scope_key, index)
        key = (
            row["capture"],
            row["display_name"],
            row["phase"],
            row["role"],
            row["rank"],
            row["device_str"],
        )
        groups.setdefault(key, []).append(row)
    merged: list[dict[str, Any]] = []
    for group in groups.values():
        counts = [int(row["count"]) for row in group]
        out: dict[str, Any] = {
            "capture": group[0]["capture"],
            "display_name": group[0]["display_name"],
            "phase": group[0]["phase"],
            "role": group[0]["role"],
            "rank": group[0]["rank"],
            "hostname": group[0]["hostname"],
            "device_str": group[0]["device_str"],
            "count": sum(counts),
            "procs": len(group),
        }
        for metric_group in metric_groups:
            metrics = sorted({metric for row in group for metric in row.get(metric_group, {})})
            out[metric_group] = {
                metric: _merge_metric(
                    [row.get(metric_group, {}).get(metric) for row in group],
                    counts,
                )
                for metric in metrics
            }
        merged.append(out)
    phase_order = {"start": 0, "instant": 1, "end": 2}
    merged.sort(
        key=lambda r: (
            r["role"],
            r["rank"],
            r["device_str"],
            0 if r["capture"] == "periodic" else 1,
            first_seen[(r["role"], r["rank"], r["device_str"], r["display_name"])],
            phase_order[r["phase"]],
        )
    )
    return merged


def _merge_metric(values: list[Any], counts: list[int]) -> Any:
    """Pool one metric across worker files; `None` when every worker lacks it.

    A stat dict (`mean`/`p95`/`max`) merges per key: `mean` is the exact pooled
    mean (count-weighted), `max` is exact, and `p95` is approximated as the max of
    the per-worker p95s -- the true pooled p95 is not recoverable from summaries.
    A scalar (e.g. `memory_total_mb`) takes the max.
    """
    present = [(value, count) for value, count in zip(values, counts) if value is not None]
    if not present:
        return None
    first = present[0][0]
    if not isinstance(first, dict):
        return max(value for value, _ in present)
    total = sum(count for _, count in present) or 1
    return {
        key: (
            sum(value[key] * count for value, count in present) / total
            if key == "mean"
            else max(value[key] for value, _ in present)
        )
        for key in first
    }


def _display_name_for_resource_row(entry: dict[str, Any]) -> str:
    """Return the row label that goes in the `Name` column."""
    if entry.get("capture") == "periodic":
        return "sample"
    path = entry.get("path") or [entry.get("name", "")]
    return "/".join(path)


def _format_aligned_stat_cells(
    stats: list[dict[str, float] | None],
    keys: tuple[str, ...],
    scale: float,
    precision: int,
    suffix: str,
) -> list[str]:
    """Format a column of stat dicts so each term lines up vertically.

    Each cell joins the selected `keys` with `|`, right-justifying every term to
    the widest value at that position across the whole column, so the numbers and
    the `|` separators line up down the column (e.g. `mean|p95|max`). `scale`
    converts the raw value (e.g. MB to GB); a `None` stat (an unavailable metric)
    renders as `n/a`.
    """
    numbers = [
        None if stat is None else [f"{stat[key] * scale:.{precision}f}" for key in keys]
        for stat in stats
    ]
    widths = [
        max((len(cell[pos]) for cell in numbers if cell is not None), default=0)
        for pos in range(len(keys))
    ]
    cells: list[str] = []
    for cell in numbers:
        if cell is None:
            cells.append("n/a")
            continue
        joined = "|".join(f"{cell[pos]:>{widths[pos]}}" for pos in range(len(keys)))
        cells.append(f"{joined}{suffix}")
    return cells


if __name__ == "__main__":
    sys.exit(main())
