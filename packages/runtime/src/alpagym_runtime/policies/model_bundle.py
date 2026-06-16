# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic model bundle path helpers for policy packages."""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import yaml


def resolve_model_bundle_path(model_path: Path) -> Path:
    """Return an HF bundle directory for ``model_path``.

    ``model_path`` may already be an HF directory containing ``config.json``
    or an ``hf_ckpt: true`` alpackage tarball.
    """
    if model_path.is_dir() and (model_path / "config.json").is_file():
        return model_path
    if model_path.is_file() and tarfile.is_tarfile(model_path):
        return _extract_alpackage(model_path)
    raise RuntimeError(
        f"model_path is neither an HF bundle directory containing config.json "
        f"nor a tarball: model_path={model_path}"
    )


def _extract_alpackage(tarball_path: Path) -> Path:
    """Extract an ``hf_ckpt: true`` alpackage into a temp dir."""
    with tarfile.open(tarball_path, "r:*") as tar:
        config_member = tar.getmember("config.yaml")
        config_handle = tar.extractfile(config_member)
        if config_handle is None:
            raise RuntimeError(
                f"alpackage tarball config.yaml is not a regular file: tarball_path={tarball_path}"
            )
        config = yaml.safe_load(config_handle.read())
        if not isinstance(config, dict) or not config.get("hf_ckpt"):
            raise RuntimeError(
                f"alpackage tarball config.yaml does not declare hf_ckpt: true: "
                f"tarball_path={tarball_path}"
            )
        working_dir = Path(tempfile.mkdtemp(prefix="alpagym-alpackage-"))
        tar.extractall(working_dir, filter="data")

    return working_dir
