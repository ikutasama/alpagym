# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """Skip alpagym in generic CI root pytest runs."""
    if not os.environ.get("CI"):
        return False
    if os.environ.get("RUN_ALPAGYM_TESTS") == "1":
        return False
    if not _path_points_to_alpagym(collection_path):
        return False
    return not _has_explicit_alpagym_request(config)


def _has_explicit_alpagym_request(config: pytest.Config) -> bool:
    """Return whether pytest was explicitly pointed at an alpagym path."""
    for raw_arg in config.invocation_params.args:
        if not isinstance(raw_arg, str):
            continue
        if raw_arg.startswith("-"):
            continue
        if _path_points_to_alpagym(raw_arg.split("::", 1)[0]):
            return True
    return False


def _path_points_to_alpagym(path: str | Path) -> bool:
    """Return whether a path is under projects/alpagym."""
    parts = Path(str(path).replace("\\", "/")).parts
    for idx, part in enumerate(parts):
        if part == "alpagym" and idx > 0 and parts[idx - 1] == "projects":
            return True
    return False
