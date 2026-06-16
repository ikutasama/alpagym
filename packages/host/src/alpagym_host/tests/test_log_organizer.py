# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from alpagym_host.log_organizer import link_role_logs


def _write(path: Path, text: str = "") -> None:
    """Write ``text`` to ``path``, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_link_role_logs_groups_logs_by_role(tmp_path: Path) -> None:
    """Per-role Cosmos logs are linked into controller.log plus policy/ and rollout/."""
    log_dir = tmp_path / "logs"
    # Cosmos writes one timestamped directory per worker node: the policy node
    # carries the controller, the rollout node carries the rollout workers.
    _write(log_dir / "logs_20260608-220809" / "controller.log", "ctrl")
    _write(log_dir / "logs_20260608-220809" / "policy_0.log", "p0")
    _write(log_dir / "logs_20260608-220815" / "rollout_0.log", "r0")
    _write(log_dir / "logs_20260608-220815" / "rollout_1.log", "r1")

    link_role_logs(log_dir)

    controller = log_dir / "controller.log"
    assert controller.is_symlink()
    assert controller.read_text() == "ctrl"
    assert (log_dir / "policy" / "policy_0.log").read_text() == "p0"
    assert (log_dir / "rollout" / "rollout_0.log").read_text() == "r0"
    assert (log_dir / "rollout" / "rollout_1.log").read_text() == "r1"
    # Links are relative so the run directory can be moved.
    assert not (log_dir / "rollout" / "rollout_0.log").readlink().is_absolute()


def test_link_role_logs_ignores_logs_latest_symlink(tmp_path: Path) -> None:
    """The logs_latest convenience symlink is not treated as a source directory."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    # A directory holding role logs, reachable only through logs_latest and named so it
    # does not match the logs_<timestamp> glob. If the organizer wrongly followed
    # logs_latest as a source, these logs would be linked into log_dir.
    cosmos_output = tmp_path / "cosmos_output"
    _write(cosmos_output / "controller.log", "ctrl")
    _write(cosmos_output / "rollout_0.log", "r0")
    (log_dir / "logs_latest").symlink_to(cosmos_output)

    link_role_logs(log_dir)

    assert not (log_dir / "controller.log").exists()
    assert not (log_dir / "rollout").exists()


def test_link_role_logs_is_idempotent(tmp_path: Path) -> None:
    """A second pass leaves existing links untouched and adds newly appeared logs."""
    log_dir = tmp_path / "logs"
    _write(log_dir / "logs_20260608-220809" / "policy_0.log", "p0")
    link_role_logs(log_dir)

    # A late rollout worker shows up between passes.
    _write(log_dir / "logs_20260608-220809" / "rollout_0.log", "r0")
    link_role_logs(log_dir)

    assert (log_dir / "policy" / "policy_0.log").read_text() == "p0"
    assert (log_dir / "rollout" / "rollout_0.log").read_text() == "r0"
