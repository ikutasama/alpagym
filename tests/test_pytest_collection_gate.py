# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_conftest():
    """Load the project conftest module as a normal Python module."""
    conftest_path = Path(__file__).parents[1] / "conftest.py"
    spec = importlib.util.spec_from_file_location("alpagym_project_conftest", conftest_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(conftest_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config_with_args(*args: str) -> SimpleNamespace:
    """Build the small pytest config shape used by the collection gate."""
    return SimpleNamespace(invocation_params=SimpleNamespace(args=args))


@pytest.mark.parametrize(
    ("run_alpagym_tests", "expected_ignored"),
    [
        (None, True),
        ("1", False),
    ],
)
def test_ci_collection_gate_requires_alpagym_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    run_alpagym_tests: str | None,
    expected_ignored: bool,
) -> None:
    """Skips AlpaGym collection in generic CI root pytest runs."""
    conftest = _load_conftest()
    monkeypatch.setenv("CI", "1")
    if run_alpagym_tests is None:
        monkeypatch.delenv("RUN_ALPAGYM_TESTS", raising=False)
    else:
        monkeypatch.setenv("RUN_ALPAGYM_TESTS", run_alpagym_tests)

    ignored = conftest.pytest_ignore_collect(
        Path("projects/alpagym/tests/test_cosmos_launcher.py"),
        _config_with_args(),
    )

    assert ignored is expected_ignored


def test_ci_collection_gate_allows_explicit_alpagym_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allows direct AlpaGym pytest runs in CI without an env-var opt-in."""
    conftest = _load_conftest()
    monkeypatch.setenv("CI", "1")
    monkeypatch.delenv("RUN_ALPAGYM_TESTS", raising=False)

    ignored = conftest.pytest_ignore_collect(
        Path("projects/alpagym/tests/test_cosmos_launcher.py"),
        _config_with_args("projects/alpagym/tests/test_cosmos_launcher.py::test_launch"),
    )

    assert ignored is False


def test_ci_collection_gate_does_not_ignore_non_alpagym_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keeps the CI collection gate scoped to AlpaGym paths."""
    conftest = _load_conftest()
    monkeypatch.setenv("CI", "1")
    monkeypatch.delenv("RUN_ALPAGYM_TESTS", raising=False)

    ignored = conftest.pytest_ignore_collect(
        Path("projects/other/tests/test_example.py"),
        _config_with_args(),
    )

    assert ignored is False
