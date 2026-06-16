# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group Cosmos per-role logs into role folders via symlinks.

Cosmos-RL's launcher writes ``controller.log``, ``policy_<i>.log``, and
``rollout_<i>.log`` flat into a timestamped ``logs_<timestamp>/`` directory --
one such directory per Cosmos worker node. That layout sorts logs by which node
produced them. This module layers a stable, role-grouped view on top so a log
can be found by what it is: ``controller.log`` at the top of the log directory,
plus ``policy/`` and ``rollout/`` folders for the trainer and generator logs.

It also tees those per-role logs to stdout for local runs, where Cosmos would
otherwise write them only to files and the launching terminal would stay silent.
"""

import threading
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# ANSI foreground codes cycled per role log so each prefix gets a stable color.
_PREFIX_COLORS = (36, 32, 33, 35, 34, 96, 92, 93, 95, 94)


@contextmanager
def organize_role_logs(log_dir: Path) -> Iterator[None]:
    """Symlink Cosmos logs into role folders while a run is in progress.

    Starts a background thread that polls ``log_dir`` for Cosmos' timestamped
    log directories and links each per-role log into the role-grouped layout, so
    the links appear as workers come up and can be tailed live. On exit it stops
    the thread and makes a final pass to catch logs written just before
    shutdown.

    Args:
        log_dir: Directory passed to the Cosmos launcher as ``--log-dir``.
    """
    stop = threading.Event()
    worker = threading.Thread(target=_watch_and_link, args=(log_dir, stop), daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=10.0)


def link_role_logs(log_dir: Path) -> None:
    """Create role-grouped symlinks for every Cosmos timestamped log directory.

    Scans ``log_dir`` for ``logs_<timestamp>/`` directories and links their
    ``controller.log`` to ``log_dir/controller.log`` and their ``policy_*.log``
    and ``rollout_*.log`` into ``log_dir/policy/`` and ``log_dir/rollout/``.
    Links are relative so the run directory stays movable, and existing links
    are left untouched.

    Args:
        log_dir: Directory passed to the Cosmos launcher as ``--log-dir``.
    """
    for ts_dir in sorted(log_dir.glob("logs_[0-9]*")):
        if not ts_dir.is_dir():
            continue
        if (ts_dir / "controller.log").is_file():
            _link(log_dir / "controller.log", Path(ts_dir.name) / "controller.log")
        for role in ("policy", "rollout"):
            for src in sorted(ts_dir.glob(f"{role}_*.log")):
                _link(log_dir / role / src.name, Path("..") / ts_dir.name / src.name)


def _watch_and_link(log_dir: Path, stop: threading.Event) -> None:
    """Link role logs every few seconds until ``stop`` is set, then once more."""
    while not stop.is_set():
        link_role_logs(log_dir)
        stop.wait(5.0)
    link_role_logs(log_dir)


def _link(link_path: Path, target: Path) -> None:
    """Create ``link_path`` as a relative symlink to ``target`` if it is absent."""
    if link_path.is_symlink() or link_path.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target)


@contextmanager
def tee_role_logs(log_dir: Path) -> Iterator[None]:
    """Mirror new Cosmos role-log lines to stdout while a run is in progress.

    Cosmos writes ``controller.log``, ``policy_<i>.log``, and ``rollout_<i>.log``
    to files under ``log_dir`` rather than the launcher's stdout, so a local run's
    terminal shows nothing while replicas load checkpoints and sync weights. This
    starts a background thread that follows those files and echoes appended lines
    to this process's stdout, each prefixed with the log name in a distinct color
    (e.g. ``policy_0``).

    Intended for local runs where a human watches the terminal. Slurm runs route
    replica output to their own job log files, so this is not used there.

    Args:
        log_dir: Directory passed to the Cosmos launcher as ``--log-dir``.
    """
    stop = threading.Event()
    worker = threading.Thread(target=_tee_loop, args=(log_dir, stop), daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join(timeout=10.0)


def _tee_loop(log_dir: Path, stop: threading.Event) -> None:
    """Echo newly appended role-log lines to stdout every second until stopped.

    Follows each ``logs_<timestamp>/*.log`` file by byte offset and prints
    complete lines prefixed with the log name in a per-name color, the way docker
    compose colors service prefixes. Emits once more after ``stop`` is set so
    lines written just before shutdown are not lost.
    """
    offsets: dict[Path, int] = {}
    while True:
        for ts_dir in sorted(log_dir.glob("logs_[0-9]*")):
            if not ts_dir.is_dir():
                continue
            for path in sorted(ts_dir.glob("*.log")):
                start = offsets.get(path, 0)
                try:
                    with path.open("rb") as handle:
                        handle.seek(start)
                        chunk = handle.read()
                except FileNotFoundError:
                    continue
                # Hold back any trailing partial line until its newline is written.
                newline = chunk.rfind(b"\n")
                if newline == -1:
                    continue
                offsets[path] = start + newline + 1
                code = _PREFIX_COLORS[zlib.crc32(path.stem.encode()) % len(_PREFIX_COLORS)]
                label = f"\033[{code}m{path.stem:<11} |\033[0m"
                text = chunk[: newline + 1].decode("utf-8", errors="replace")
                for line in text.splitlines():
                    print(f"{label} {line}", flush=True)
        if stop.is_set():
            return
        stop.wait(1.0)
