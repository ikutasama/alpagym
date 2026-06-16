# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers for `alpagym_host` tests."""

import tomllib
from pathlib import Path


def pinned_alpasim_repo_ref() -> str:
    """Return the production-pinned AlpaSim ref from the workspace."""
    workspace_pyproject = Path(__file__).resolve().parents[5] / "pyproject.toml"
    workspace_project = tomllib.loads(workspace_pyproject.read_text(encoding="utf-8"))
    return workspace_project["tool"]["uv"]["sources"]["alpasim-grpc"]["rev"]
