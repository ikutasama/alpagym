# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Active-scope stack, `@measure_perf` decorator, and `timed_scope` CM."""

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator, TypeVar

from alpagym_runtime.perf.instrument.snapshot import ResourceSnapshotKey, SnapshotPhase
from alpagym_runtime.perf.instrument.store import PerfStore, try_get_perf_store

_F = TypeVar("_F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActivePerfScope:
    """One frame on the active-scope stack."""

    name: str
    path: tuple[str, ...]
    category: str
    start_ns: int


_PERF_SCOPE_STACK: ContextVar[tuple[ActivePerfScope, ...]] = ContextVar(
    "alpagym_perf_scope_stack",
    default=(),
)


def current_scope_path() -> tuple[str, ...]:
    """Return the active-scope path at the call site, or `()` when no scope is active."""
    stack = _PERF_SCOPE_STACK.get()
    return stack[-1].path if stack else ()


def measure_perf(
    name: str,
    *,
    category: str,
    cpu_snapshot: bool = False,
    gpu_snapshot: bool = False,
) -> Callable[[_F], _F]:
    """Time the decorated function and optionally snapshot CPU/GPU at boundaries.

    Args:
        name: Stable scope name. Must be explicit; use the caller-context label
            (e.g. ``"driver/drive"``) rather than the function name so the
            attribution stays correct across call sites.
        category: Bottleneck category. See the Categories table in this package's
            README.md.
        cpu_snapshot: Capture CPU start/end snapshots when CPU collection is enabled.
        gpu_snapshot: Capture GPU start/end snapshots when GPU collection is enabled.

    Returns:
        Decorator that wraps the function with timing and optional resource snapshots.
        The decorator always installs the wrapper; when no perf store is installed
        the wrapper calls through to the original after a single store lookup and
        `None` check per invocation.
    """

    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            store = try_get_perf_store()
            if store is None:
                return func(*args, **kwargs)
            with _enter_scope(
                store,
                name,
                category,
                cpu_snapshot=cpu_snapshot,
                gpu_snapshot=gpu_snapshot,
            ):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def timed_scope(
    name: str,
    *,
    category: str,
    cpu_snapshot: bool = False,
    gpu_snapshot: bool = False,
) -> Iterator[None]:
    """Time the enclosed block and optionally snapshot CPU/GPU at its boundaries."""
    store = try_get_perf_store()
    if store is None:
        yield
        return
    with _enter_scope(
        store,
        name,
        category,
        cpu_snapshot=cpu_snapshot,
        gpu_snapshot=gpu_snapshot,
    ):
        yield


@contextmanager
def _enter_scope(
    store: PerfStore,
    name: str,
    category: str,
    cpu_snapshot: bool,
    gpu_snapshot: bool,
) -> Iterator[None]:
    """Push a frame on the active-scope stack, record timing and boundary snapshots."""
    stack = _PERF_SCOPE_STACK.get()
    parent_path = stack[-1].path if stack else ()
    path = parent_path + (name,)
    # Snapshot before starting the clock so a scope never times its own boundary
    # snapshots; the END snapshot below is likewise taken after the duration is
    # computed. Snapshot reads (psutil/NVML) cost real time and would otherwise
    # dominate small scopes.
    _maybe_boundary_snapshot(store, path, SnapshotPhase.START, cpu_snapshot, gpu_snapshot)
    frame = ActivePerfScope(
        name=name,
        path=path,
        category=category,
        start_ns=time.perf_counter_ns(),
    )
    token = _PERF_SCOPE_STACK.set(stack + (frame,))
    try:
        yield
    finally:
        duration_ns = time.perf_counter_ns() - frame.start_ns
        _PERF_SCOPE_STACK.reset(token)
        store.timer.record(
            path=path,
            name=name,
            category=category,
            duration_ns=duration_ns,
        )
        _maybe_boundary_snapshot(store, path, SnapshotPhase.END, cpu_snapshot, gpu_snapshot)


def _maybe_boundary_snapshot(
    store: PerfStore,
    path: tuple[str, ...],
    phase: SnapshotPhase,
    cpu_snapshot: bool,
    gpu_snapshot: bool,
) -> None:
    """Capture CPU/GPU snapshots at a scope boundary when requested and enabled.

    Reader failures (psutil/NVML) are logged and dropped rather than propagated,
    so a telemetry hiccup never turns into a rollout or training failure on the
    timed path. The periodic monitor swallows the same failures in its own loop.
    """
    if cpu_snapshot and store.cpu is not None:
        key = ResourceSnapshotKey.for_scope_boundary(path=path, phase=phase)
        try:
            store.cpu.store.record_snapshot(store.cpu.reader.read_snapshot(key))
        except Exception:
            logger.exception("perf: CPU boundary snapshot failed at %s", path)
    if gpu_snapshot and store.gpu is not None:
        key = ResourceSnapshotKey.for_scope_boundary(path=path, phase=phase)
        try:
            store.gpu.store.record_snapshot(store.gpu.reader.read_snapshot(key))
        except Exception:
            logger.exception("perf: GPU boundary snapshot failed at %s", path)
