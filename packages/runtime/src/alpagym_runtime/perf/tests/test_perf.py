# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for the `alpagym_runtime.perf` package."""

from __future__ import annotations

import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
from alpagym_host.config import ArtifactPaths, PerfConfig

from alpagym_runtime.perf.analysis import cli
from alpagym_runtime.perf.analysis.alpasim_telemetry import discover_prom_files
from alpagym_runtime.perf.instrument import lifecycle as lifecycle_module
from alpagym_runtime.perf.instrument.lifecycle import initialize_perf, shutdown_perf
from alpagym_runtime.perf.instrument.marker import record_perf_marker
from alpagym_runtime.perf.instrument.monitor import ResourceMonitor
from alpagym_runtime.perf.instrument.sampling import ReservoirSampler, percentile
from alpagym_runtime.perf.instrument.scope import current_scope_path, timed_scope
from alpagym_runtime.perf.instrument.store import (
    PerfStore,
    ResourceRuntime,
    TimingStore,
    WorkerIdentity,
    set_perf_store,
    teardown_perf_store,
    try_get_perf_store,
)


@dataclass
class _FakeResolved:
    """Stand-in `RunConfig` used by the lifecycle tests."""

    perf: PerfConfig
    artifact_paths: ArtifactPaths


def _build_artifact_paths(tmp_path: Path) -> ArtifactPaths:
    """Build a minimal `ArtifactPaths` rooted at `tmp_path`."""
    perf_dir = tmp_path / "perf"
    perf_dir.mkdir()
    return ArtifactPaths(
        run_dir=tmp_path,
        artifacts_dir=tmp_path / "a",
        policy_model_bundle_dir=tmp_path / "b",
        resolved_config_path=tmp_path / "c.yaml",
        cosmos_config_path=tmp_path / "c.toml",
        submit_script_path=tmp_path / "s.sbatch",
        log_dir=tmp_path / "logs",
        topology_registry_dir=tmp_path / "topo",
        alpasim_log_dir=tmp_path / "alpasim",
        alpasim_scene_ids_path=tmp_path / "alpasim_scene_ids.yaml",
        perf_dir=perf_dir,
    )


@pytest.fixture
def perf_identity_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject the Cosmos-RL launch env vars `WorkerIdentity` requires."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    # Cosmos sets COSMOS_ROLE in title case; assert the artifact lowercases it.
    monkeypatch.setenv("COSMOS_ROLE", "Rollout")


@pytest.fixture
def perf_lifecycle(tmp_path: Path, perf_identity_env: None) -> _FakeResolved:
    """Initialize perf against a tmp artifact root and tear it down at the end."""
    cfg = PerfConfig(
        enabled=True,
        sample_every_n=1,
        resource_sample_interval_s=None,
        max_samples_per_series=64,
        flush_every_n_updates=100,
        flush_interval_s=5.0,
        collect_cpu=True,
        collect_gpu=False,
    )
    resolved = _FakeResolved(
        perf=cfg,
        artifact_paths=_build_artifact_paths(tmp_path),
    )
    initialize_perf(resolved)
    try:
        yield resolved
    finally:
        shutdown_perf()


def test_timing_api_imports_without_nvml_or_psutil() -> None:
    """Importing the timing API must require neither NVML nor psutil.

    Disabled, GPU-only, and CPU-only call sites import the lifecycle and scope
    APIs but only build CPU/GPU collection behind config flags, so missing
    `pynvml` or `psutil` must not break the import. The subprocess blocks both
    and imports the modules; a regression that pulls either at module load
    (e.g. a top-level `import psutil` reached through `lifecycle`) fails here.
    """
    script = (
        "import sys\n"
        "sys.modules['pynvml'] = None\n"
        "sys.modules['psutil'] = None\n"
        "from alpagym_runtime.perf.instrument import lifecycle, scope\n"
        "assert lifecycle.initialize_perf is not None\n"
        "assert scope.timed_scope is not None\n"
        "assert scope.measure_perf is not None\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_reservoir_sampler_bounded_under_load() -> None:
    """`ReservoirSampler` keeps at most `capacity` samples regardless of input size."""
    sampler = ReservoirSampler(capacity=16, rng=random.Random(0))
    for value in range(10_000):
        sampler.add(float(value))
    assert len(sampler.samples()) == 16
    assert sampler.total_seen == 10_000


def test_percentile_interpolates_between_order_statistics() -> None:
    """`percentile` produces NumPy-style linear interpolation; empty input is 0.0."""
    assert percentile([], 50.0) == 0.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 95.0) == pytest.approx(3.85)


def test_timing_store_keys_by_full_path(tmp_path: Path) -> None:
    """The same scope name under different parents is recorded as two entries."""
    store = TimingStore(sample_every_n=1, max_samples_per_series=8)
    store.record(path=("rollout/x",), name="rollout/x", category="orchestration", duration_ns=10)
    store.record(
        path=("rollout/y", "rollout/x"),
        name="rollout/x",
        category="orchestration",
        duration_ns=20,
    )
    payload = store.to_json()
    paths = [tuple(scope["path"]) for scope in payload["scopes"]]
    assert ("rollout/x",) in paths
    assert ("rollout/y", "rollout/x") in paths
    assert len(paths) == 2


def test_timing_store_percentiles_include_first_observation() -> None:
    """Low-frequency scopes should not emit zero percentiles under sparse sampling."""
    store = TimingStore(sample_every_n=10, max_samples_per_series=8)
    store.record(path=("once",), name="once", category="orchestration", duration_ns=1_000_000)
    [scope] = store.to_json()["scopes"]
    assert scope["count"] == 1
    assert scope["total_ms"] == pytest.approx(1.0)
    assert scope["p50_ms"] == pytest.approx(1.0)
    assert scope["p95_ms"] == pytest.approx(1.0)


def test_should_flush_count_trigger_accumulates_across_polls(tmp_path: Path) -> None:
    """The count trigger fires once total updates cross the threshold.

    `should_flush` must not zero the dirty counter on a poll that returns
    False, or a workload spread just under the threshold per poll never
    triggers a count-based flush.
    """
    timer = TimingStore(sample_every_n=1, max_samples_per_series=8)
    identity = WorkerIdentity(
        run_id="r",
        role="rollout",
        rank=0,
        local_rank=0,
        world_size=1,
        hostname="h",
        pid=1,
        device="cpu",
    )
    config = PerfConfig(
        enabled=True,
        sample_every_n=1,
        resource_sample_interval_s=None,
        max_samples_per_series=8,
        flush_every_n_updates=1000,
        flush_interval_s=3600.0,
        collect_cpu=False,
        collect_gpu=False,
    )
    store = PerfStore(
        identity=identity, config=config, output_path=tmp_path / "perf.json", timer=timer
    )
    for _ in range(999):
        timer.record(path=("a",), name="a", category="orchestration", duration_ns=1)
    assert store.should_flush() is False
    for _ in range(2):
        timer.record(path=("a",), name="a", category="orchestration", duration_ns=1)
    assert store.should_flush() is True


def test_resource_snapshots_trigger_count_flush(
    tmp_path: Path,
    perf_identity_env: None,
) -> None:
    """CPU/GPU snapshots count toward `flush_every_n_updates`, not just timed scopes."""
    artifact_paths = _build_artifact_paths(tmp_path)
    cfg = PerfConfig(
        enabled=True,
        sample_every_n=1,
        resource_sample_interval_s=None,
        max_samples_per_series=8,
        flush_every_n_updates=1,
        flush_interval_s=3600.0,
        collect_cpu=True,
        collect_gpu=False,
    )
    resolved = _FakeResolved(perf=cfg, artifact_paths=artifact_paths)
    try:
        initialize_perf(resolved)
        store = try_get_perf_store()
        assert store is not None
        assert store.should_flush() is True
        store.write_atomic()
        assert store.should_flush() is False
    finally:
        shutdown_perf()


def test_initialize_perf_does_not_start_cpu_monitor_before_gpu_runtime(
    tmp_path: Path,
    perf_identity_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GPU init failure must not leak a CPU monitor with no registered store."""
    artifact_paths = _build_artifact_paths(tmp_path)
    cfg = PerfConfig(
        enabled=True,
        sample_every_n=1,
        resource_sample_interval_s=1.0,
        max_samples_per_series=8,
        flush_every_n_updates=10,
        flush_interval_s=3600.0,
        collect_cpu=True,
        collect_gpu=True,
    )
    resolved = _FakeResolved(perf=cfg, artifact_paths=artifact_paths)
    started: list[str] = []

    def fail_gpu_runtime(cfg: PerfConfig, identity: WorkerIdentity) -> ResourceRuntime:
        raise RuntimeError("gpu init failed")

    monkeypatch.setattr(lifecycle_module, "_build_gpu_runtime", fail_gpu_runtime)
    monkeypatch.setattr(ResourceMonitor, "start", lambda self: started.append(self.name))

    with pytest.raises(RuntimeError, match="gpu init failed"):
        initialize_perf(resolved)

    assert started == []
    assert try_get_perf_store() is None


def test_teardown_leaves_store_bound_when_monitor_stays_alive(tmp_path: Path) -> None:
    """A stuck resource monitor prevents clearing the store and shutting NVML down."""

    class _FakeResourceStore:
        def dirty_count(self) -> int:
            return 0

        def reset_dirty(self) -> None:
            pass

        def to_json(self) -> dict[str, list[dict[str, object]]]:
            return {"periodic": [], "checkpoints": []}

    class _StuckMonitor:
        name = "stuck-monitor"

        def __init__(self) -> None:
            self.started = False
            self.alive = True

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            pass

        def is_alive(self) -> bool:
            return self.alive

    monitor = _StuckMonitor()
    identity = WorkerIdentity(
        run_id="r",
        role="rollout",
        rank=0,
        local_rank=0,
        world_size=1,
        hostname="h",
        pid=1,
        device="cpu",
    )
    config = PerfConfig(
        enabled=True,
        sample_every_n=1,
        resource_sample_interval_s=1.0,
        max_samples_per_series=8,
        flush_every_n_updates=10,
        flush_interval_s=3600.0,
        collect_cpu=True,
        collect_gpu=False,
    )
    store = PerfStore(
        identity=identity,
        config=config,
        output_path=tmp_path / "perf.json",
        timer=TimingStore(sample_every_n=1, max_samples_per_series=8),
        cpu=ResourceRuntime(store=_FakeResourceStore(), reader=object(), monitor=monitor),
    )
    try:
        set_perf_store(store)
        assert monitor.started is True
        assert teardown_perf_store() is False
        assert try_get_perf_store() is store
    finally:
        monitor.alive = False
        teardown_perf_store()


def test_scope_nesting_records_full_runtime_path(perf_lifecycle: _FakeResolved) -> None:
    """Nested `timed_scope` calls build a path tuple from parent to child."""
    with timed_scope("parent", category="orchestration"):
        with timed_scope("child", category="orchestration"):
            assert current_scope_path() == ("parent", "child")
    store = try_get_perf_store()
    assert store is not None
    paths = {tuple(scope["path"]) for scope in store.timer.to_json()["scopes"]}
    assert ("parent",) in paths
    assert ("parent", "child") in paths


def test_marker_attaches_to_active_scope_path(perf_lifecycle: _FakeResolved) -> None:
    """`record_perf_marker` records its snapshot under the active-scope path."""
    with timed_scope("parent", category="orchestration"):
        assert current_scope_path() == ("parent",)
        record_perf_marker("checkpoint", cpu_snapshot=True)
    record_perf_marker("standalone", cpu_snapshot=True)
    store = try_get_perf_store()
    assert store is not None and store.cpu is not None
    checkpoint_paths = {tuple(row["path"]) for row in store.cpu.store.to_json()["checkpoints"]}
    assert ("parent", "checkpoint") in checkpoint_paths
    assert ("standalone",) in checkpoint_paths
    # Markers never push a timed scope, so timing still carries only `parent`.
    timing_paths = {tuple(scope["path"]) for scope in store.timer.to_json()["scopes"]}
    assert timing_paths == {("parent",)}


def test_atomic_write_replaces_final_file(perf_lifecycle: _FakeResolved) -> None:
    """`PerfStore.write_atomic` leaves only the final JSON, no `.tmp` siblings."""
    store = try_get_perf_store()
    assert store is not None
    with timed_scope("foo", category="orchestration"):
        pass
    store.write_atomic()
    perf_dir = perf_lifecycle.artifact_paths.perf_dir
    json_files = list(perf_dir.glob("alpagym_perf_*.json"))
    tmp_files = list(perf_dir.glob("*.tmp"))
    assert len(json_files) == 1
    assert not tmp_files
    payload = json.loads(json_files[0].read_text())
    assert payload["schema_version"] == 1
    assert payload["project"] == "alpagym"
    assert payload["role"] == "rollout"


def test_cleanup_stale_tmp_runs_at_init(
    tmp_path: Path,
    perf_identity_env: None,
) -> None:
    """`initialize_perf(...)` removes leftover `{final}.tmp` siblings owned by this pid."""
    artifact_paths = _build_artifact_paths(tmp_path)
    perf_dir = artifact_paths.perf_dir
    pid = os.getpid()
    own_stale = perf_dir / f"alpagym_perf_rollout_0_{pid}.json.deadbeef.tmp"
    own_stale.write_text("stale")
    cfg = PerfConfig(
        enabled=True,
        sample_every_n=1,
        resource_sample_interval_s=None,
        max_samples_per_series=8,
        flush_every_n_updates=10,
        flush_interval_s=5.0,
        collect_cpu=False,
        collect_gpu=False,
    )
    resolved = _FakeResolved(perf=cfg, artifact_paths=artifact_paths)
    try:
        initialize_perf(resolved)
        assert not own_stale.exists()
    finally:
        shutdown_perf()


def test_disabled_config_is_a_full_noop(tmp_path: Path) -> None:
    """`enabled=False` skips identity resolution and leaves no perf store installed."""
    artifact_paths = _build_artifact_paths(tmp_path)
    cfg = PerfConfig(
        enabled=False,
        sample_every_n=1,
        resource_sample_interval_s=None,
        max_samples_per_series=8,
        flush_every_n_updates=10,
        flush_interval_s=5.0,
        collect_cpu=False,
        collect_gpu=False,
    )
    resolved = _FakeResolved(perf=cfg, artifact_paths=artifact_paths)
    # No env vars set — disabled mode must not touch os.environ.
    initialize_perf(resolved)
    assert try_get_perf_store() is None
    with timed_scope("foo", category="orchestration"):
        pass
    record_perf_marker("bar")
    # Nothing should have been written.
    assert not list(artifact_paths.perf_dir.iterdir())


def test_initialize_perf_is_idempotent_within_process(perf_lifecycle: _FakeResolved) -> None:
    """A second `initialize_perf(...)` in the same process is a no-op.

    Cosmos colocated mode builds the policy and rollout in one process, so both
    the trainer and the rollout backend call `initialize_perf(...)`. The second
    call must not raise or replace the store the first call installed.
    """
    store = try_get_perf_store()
    assert store is not None
    initialize_perf(perf_lifecycle)
    assert try_get_perf_store() is store


def test_cli_renders_hierarchy(
    perf_lifecycle: _FakeResolved,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI prints the timing tree with indent that reflects path depth."""
    with timed_scope("rollout/generate", category="orchestration"):
        with timed_scope("rollout/sim_request_build", category="orchestration"):
            time.sleep(0.001)
        with timed_scope("rollout/simulation_batch", category="orchestration"):
            with timed_scope("rollout/sim_step_rpc", category="external_rpc"):
                time.sleep(0.001)
    store = try_get_perf_store()
    assert store is not None
    store.write_atomic()
    exit_code = cli.main(
        [
            "--perf-dir",
            str(perf_lifecycle.artifact_paths.perf_dir),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "rollout/generate" in captured.out
    assert "  simulation_batch" in captured.out  # depth=2 indent
    assert "    sim_step_rpc" in captured.out  # depth=3 indent
    assert "  sim_request_build" in captured.out
    assert "TOTAL" in captured.out


def test_cli_groups_timing_roots_by_execution_lane(
    perf_lifecycle: _FakeResolved,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Timing roots render by execution lane, with totals sorted inside each group."""
    store = try_get_perf_store()
    assert store is not None
    store.timer.record(
        path=("driver/drive",),
        name="driver/drive",
        category="external_rpc",
        duration_ns=4_000_000,
    )
    store.timer.record(
        path=("rollout/episode",),
        name="rollout/episode",
        category="orchestration",
        duration_ns=1_000_000,
    )
    store.timer.record(
        path=("rollout/inference_forward",),
        name="rollout/inference_forward",
        category="compute_gpu_wall",
        duration_ns=3_000_000,
    )
    store.timer.record(
        path=("rollout/generate",),
        name="rollout/generate",
        category="orchestration",
        duration_ns=2_000_000,
    )
    store.timer.record(
        path=("trainer/step",),
        name="trainer/step",
        category="compute_gpu_wall",
        duration_ns=5_000_000,
    )
    store.write_atomic()

    assert cli.main(["--perf-dir", str(perf_lifecycle.artifact_paths.perf_dir)]) == 0
    out = capsys.readouterr().out
    assert out.index("- Rollout episode ") < out.index("- Rollout generation / inference ")
    assert out.index("- Rollout generation / inference ") < out.index("- Policy callback / driver ")
    assert out.index("- Policy callback / driver ") < out.index("- Training ")
    assert out.index("rollout/inference_forward") < out.index("rollout/generate")
    assert out.index("trainer/step") > out.index("driver/drive")


def test_cli_category_filter_keeps_nested_matches(
    perf_lifecycle: _FakeResolved,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--category` keeps a matching child even when its ancestor differs."""
    with timed_scope("rollout/generate", category="orchestration"):
        with timed_scope("rollout/sim_step_rpc", category="external_rpc"):
            time.sleep(0.001)
    store = try_get_perf_store()
    assert store is not None
    store.write_atomic()
    exit_code = cli.main(
        [
            "--perf-dir",
            str(perf_lifecycle.artifact_paths.perf_dir),
            "--category",
            "external_rpc",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    # The external_rpc child shows even though its root is orchestration.
    assert "sim_step_rpc" in captured.out
    # The non-matching orchestration root is not emitted as a row.
    assert "rollout/generate" not in captured.out


def test_cli_marks_exclusive_percent_na_when_merged_child_exceeds_parent(
    perf_lifecycle: _FakeResolved,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Merged worker subsets can make child totals exceed parent totals."""
    store = try_get_perf_store()
    assert store is not None
    store.timer.record(
        path=("trainer/step",),
        name="trainer/step",
        category="compute_gpu_wall",
        duration_ns=1_000_000,
    )
    store.timer.record(
        path=("trainer/step", "trainer/artifact_load"),
        name="trainer/artifact_load",
        category="orchestration",
        duration_ns=2_000_000,
    )
    store.write_atomic()

    assert cli.main(["--perf-dir", str(perf_lifecycle.artifact_paths.perf_dir)]) == 0
    out = capsys.readouterr().out
    assert "trainer/step" in out
    assert "n/a" in out
    assert "  Caveats" in out
    assert "Merged child scopes can exceed their parent" in out
    assert "not a strict tree (Excl % = n/a)" in out


def test_cli_defaults_to_rank0_and_all_ranks_flags_inflation(
    perf_lifecycle: _FakeResolved,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI processes only rank 0 by default; `--all-ranks` merges every rank
    and warns that the TOTAL sums total_ms across the merged workers.
    """
    with timed_scope("rollout/generate", category="orchestration"):
        time.sleep(0.001)
    store = try_get_perf_store()
    assert store is not None
    store.write_atomic()
    # A data-parallel run writes one artifact per rank; add a second rank.
    perf_dir = perf_lifecycle.artifact_paths.perf_dir
    [first] = list(perf_dir.glob("alpagym_perf_*.json"))
    second = json.loads(first.read_text())
    second["rank"] = 1
    (perf_dir / "alpagym_perf_rollout_1_1.json").write_text(json.dumps(second))

    # Default: rank 0 only -> a single worker here -> no multi-worker caveat.
    assert cli.main(["--perf-dir", str(perf_dir)]) == 0
    assert "merged workers" not in capsys.readouterr().out

    # --all-ranks merges rank 0 + rank 1 -> the caveat fires.
    assert cli.main(["--perf-dir", str(perf_dir), "--all-ranks"]) == 0
    assert "merged workers" in capsys.readouterr().out


def test_aggregate_resource_rows_pools_workers_per_role_rank() -> None:
    """Same-(role,rank) workers collapse into one row: counts sum, mean is the
    pooled (count-weighted) mean, max is exact, p95 is the max of per-worker p95s,
    and a scalar metric (memory_total_mb) is kept.
    """
    rows = [
        {
            "capture": "periodic",
            "display_name": "sample",
            "phase": "instant",
            "role": "rollout",
            "rank": 0,
            "hostname": "h",
            "device_str": "",
            "count": 10,
            "system": {
                "cpu_util_percent": {"mean": 10.0, "p95": 20.0, "max": 30.0},
                "memory_total_mb": 100.0,
            },
            "process": {"cpu_util_percent": {"mean": 1.0, "p95": 2.0, "max": 3.0}},
        },
        {
            "capture": "periodic",
            "display_name": "sample",
            "phase": "instant",
            "role": "rollout",
            "rank": 0,
            "hostname": "h",
            "device_str": "",
            "count": 30,
            "system": {
                "cpu_util_percent": {"mean": 50.0, "p95": 60.0, "max": 70.0},
                "memory_total_mb": 100.0,
            },
            "process": {"cpu_util_percent": {"mean": 5.0, "p95": 6.0, "max": 7.0}},
        },
    ]
    [merged] = cli._aggregate_resource_rows(rows, "cpu")
    assert merged["procs"] == 2
    assert merged["count"] == 40
    cpu = merged["system"]["cpu_util_percent"]
    assert cpu["mean"] == pytest.approx((10.0 * 10 + 50.0 * 30) / 40)  # pooled mean = 40.0
    assert cpu["max"] == 70.0  # exact max
    assert cpu["p95"] == 60.0  # max of per-worker p95s
    assert merged["system"]["memory_total_mb"] == 100.0  # scalar kept


def test_aggregate_resource_rows_unions_optional_metric_keys() -> None:
    """Optional metrics present only on later workers must survive aggregation."""
    rows = [
        {
            "capture": "periodic",
            "display_name": "sample",
            "phase": "instant",
            "role": "rollout",
            "rank": 0,
            "hostname": "h",
            "device_str": "cuda:0",
            "count": 10,
            "device": {"gpu_util_percent": {"mean": 10.0, "p95": 20.0, "max": 30.0}},
            "process": {
                "torch_allocated_mb": {"mean": 1.0, "p95": 2.0, "max": 3.0},
            },
        },
        {
            "capture": "periodic",
            "display_name": "sample",
            "phase": "instant",
            "role": "rollout",
            "rank": 0,
            "hostname": "h",
            "device_str": "cuda:0",
            "count": 5,
            "device": {"gpu_util_percent": {"mean": 40.0, "p95": 50.0, "max": 60.0}},
            "process": {
                "torch_allocated_mb": {"mean": 4.0, "p95": 5.0, "max": 6.0},
                "driver_memory_mb": {"mean": 7.0, "p95": 8.0, "max": 9.0},
            },
        },
    ]
    [merged] = cli._aggregate_resource_rows(rows, "gpu")
    assert merged["process"]["driver_memory_mb"]["mean"] == pytest.approx(7.0)
    assert merged["process"]["driver_memory_mb"]["p95"] == 8.0
    assert merged["process"]["driver_memory_mb"]["max"] == 9.0


def test_aggregate_resource_rows_orders_sample_first_then_runtime_flow() -> None:
    """Rows render in reading order: the periodic `sample` heads the block, then
    scopes in first-seen (runtime) order with each scope's `start` immediately
    before its `end` -- even when the raw recorded order nests and separates them.
    """

    def cpu_row(capture: str, name: str, phase: str):
        return {
            "capture": capture,
            "display_name": name,
            "phase": phase,
            "role": "rollout",
            "rank": 0,
            "hostname": "h",
            "device_str": "",
            "count": 1,
            "system": {"cpu_util_percent": {"mean": 1.0, "p95": 1.0, "max": 1.0}},
        }

    # scope_a opens, scope_b opens+closes nested inside it, scope_a closes, and the
    # periodic sample is recorded last.
    rows = [
        cpu_row("checkpoint", "scope_a", "start"),
        cpu_row("checkpoint", "scope_b", "start"),
        cpu_row("checkpoint", "scope_b", "end"),
        cpu_row("checkpoint", "scope_a", "end"),
        cpu_row("periodic", "sample", "instant"),
    ]
    ordered = [(r["display_name"], r["phase"]) for r in cli._aggregate_resource_rows(rows, "cpu")]
    assert ordered == [
        ("sample", "instant"),
        ("scope_a", "start"),
        ("scope_a", "end"),
        ("scope_b", "start"),
        ("scope_b", "end"),
    ]


def test_cli_resource_columns_align_under_long_names(
    perf_lifecycle: _FakeResolved,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A long checkpoint name widens the Name column instead of shifting a row.

    Earlier fixed-width formatting let a name longer than the column overflow
    and push that row's later columns out of alignment. Every resource-table
    row must keep the Phase column under its header regardless of name length.
    """
    record_perf_marker("init", cpu_snapshot=True)
    record_perf_marker("rollout/engine_init/rollout/model_ready", cpu_snapshot=True)
    store = try_get_perf_store()
    assert store is not None
    store.write_atomic()
    exit_code = cli.main(["--perf-dir", str(perf_lifecycle.artifact_paths.perf_dir)])
    captured = capsys.readouterr()
    assert exit_code == 0
    lines = captured.out.splitlines()
    header = lines[lines.index("System") + 1]
    phase_col = header.index("Phase")
    data_rows = list(itertools.takewhile(bool, lines[lines.index("System") + 2 :]))
    assert any("rollout/engine_init/rollout/model_ready" in row for row in data_rows)
    assert all(row.index("instant") == phase_col for row in data_rows)


def test_cli_renders_alpasim_internals_when_prom_present(
    perf_lifecycle: _FakeResolved,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI renders the AlpaSim Internals section from telemetry `.prom` files.

    AlpaSim writes `metrics_worker_*.prom` under the run dir (the parent of `perf/`); the
    CLI discovers them and folds the render/physics/drive breakdown into one view.
    """
    with timed_scope("rollout/episode", category="orchestration"):
        time.sleep(0.001)
    store = try_get_perf_store()
    assert store is not None
    store.write_atomic()
    perf_dir = perf_lifecycle.artifact_paths.perf_dir
    telemetry = perf_dir.parent / "alpasim" / "telemetry"
    telemetry.mkdir(parents=True)
    (telemetry / "metrics_worker_0.prom").write_text(
        'rpc_duration_seconds_sum{service="sensorsim",method="render_rgb",tag="default"} 12.0\n'
        'rpc_duration_seconds_count{service="sensorsim",method="render_rgb",tag="default"} 100\n'
        'rpc_duration_seconds_sum{service="driver",method="drive",tag="default"} 40.0\n'
        'rpc_duration_seconds_count{service="driver",method="drive",tag="default"} 200\n'
        'rpc_duration_seconds_sum{service="controller",'
        'method="run_controller_and_vehicle",tag="default"} 6.0\n'
        'rpc_duration_seconds_count{service="controller",'
        'method="run_controller_and_vehicle",tag="default"} 10\n'
        'rpc_duration_seconds_sum{service="driver",'
        'method="submit_recording_ground_truth",tag="default"} 3.0\n'
        'rpc_duration_seconds_count{service="driver",'
        'method="submit_recording_ground_truth",tag="default"} 10\n'
        "rollout_duration_seconds_sum 9.0\n"
        "rollout_duration_seconds_count 3\n"
        "simulation_rollout_count 5\n"
    )
    assert cli.main(["--perf-dir", str(perf_dir)]) == 0
    out = capsys.readouterr().out
    assert "Simulator internals" in out
    assert "per-roll denominator  : simulation_rollout_count=5" in out
    assert "completed rollouts    : rollout_duration samples=3" in out
    assert "WARNING: per-roll columns use simulation_rollout_count" in out
    assert "calls/roll" in out
    assert re.search(
        r"render_rgb\s+sensorsim\s+default\s+100\s+20\.0\s+120\.00 ms\s+12\.00 s\s+2\.40 s",
        out,
    )
    assert re.search(r"render \(NRE\)\s+20\.0\s+2\.40 s", out)
    assert "render (NRE)" in out  # phase rollup bucket for sensorsim/render_rgb
    assert "policy callback (drive)" in out  # phase rollup bucket for driver/drive
    assert "run_ctrl_vehicle" in out
    assert "submit_recording_gt" in out
    assert "run_controller_and_vehicle" not in out
    assert "submit_recording_ground_truth" not in out


def test_discover_prom_files_merges_direct_and_nested_files(tmp_path: Path) -> None:
    """Directory discovery includes both direct and nested AlpaSim telemetry files."""
    direct = tmp_path / "metrics_worker_0.prom"
    nested_dir = tmp_path / "alpasim" / "telemetry"
    nested_dir.mkdir(parents=True)
    nested = nested_dir / "metrics_worker_1.prom"
    direct.write_text("", encoding="utf-8")
    nested.write_text("", encoding="utf-8")
    assert set(discover_prom_files(tmp_path)) == {direct, nested}


def test_cli_labels_alpasim_internals_as_run_wide_when_artifacts_filtered(
    perf_lifecycle: _FakeResolved,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Simulator telemetry is not narrowed by artifact filters, so label filtered output."""
    with timed_scope("rollout/episode", category="orchestration"):
        pass
    store = try_get_perf_store()
    assert store is not None
    store.write_atomic()
    perf_dir = perf_lifecycle.artifact_paths.perf_dir
    [first] = list(perf_dir.glob("alpagym_perf_*.json"))
    second = json.loads(first.read_text())
    second["rank"] = 1
    (perf_dir / "alpagym_perf_rollout_1_1.json").write_text(json.dumps(second))
    telemetry = perf_dir.parent / "alpasim" / "telemetry"
    telemetry.mkdir(parents=True)
    (telemetry / "metrics_worker_0.prom").write_text("simulation_rollout_count 1\n")

    assert cli.main(["--perf-dir", str(perf_dir)]) == 0
    assert "simulator telemetry is run-wide" in capsys.readouterr().out


def test_cli_skips_alpasim_internals_without_prom(
    perf_lifecycle: _FakeResolved,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With no telemetry `.prom` files, the AlpaSim section is absent and the CLI succeeds."""
    with timed_scope("rollout/episode", category="orchestration"):
        time.sleep(0.001)
    store = try_get_perf_store()
    assert store is not None
    store.write_atomic()
    assert cli.main(["--perf-dir", str(perf_lifecycle.artifact_paths.perf_dir)]) == 0
    assert "Simulator internals" not in capsys.readouterr().out
