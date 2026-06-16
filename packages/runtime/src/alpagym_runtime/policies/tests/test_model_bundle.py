# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for generic model bundle path helpers."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import yaml
from alpagym_runtime.policies.model_bundle import resolve_model_bundle_path


def test_resolve_model_bundle_path_accepts_hf_directory(tmp_path: Path) -> None:
    """HF bundle directories are returned unchanged."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "config.json").write_text("{}", encoding="utf-8")

    assert resolve_model_bundle_path(bundle_dir) == bundle_dir


def test_resolve_model_bundle_path_extracts_hf_alpackage(tmp_path: Path) -> None:
    """HF alpackage tarballs are extracted to a temporary bundle directory."""
    tarball_path = tmp_path / "bundle.tar"
    with tarfile.open(tarball_path, "w") as tar:
        _add_tar_member(tar, "config.yaml", yaml.safe_dump({"hf_ckpt": True}))
        _add_tar_member(tar, "config.json", "{}")
        _add_tar_member(tar, "model.safetensors", "weights")

    bundle_dir = resolve_model_bundle_path(tarball_path)

    assert (bundle_dir / "config.json").read_text(encoding="utf-8") == "{}"
    assert (bundle_dir / "model.safetensors").read_text(encoding="utf-8") == "weights"


def test_resolve_model_bundle_path_rejects_non_hf_alpackage(tmp_path: Path) -> None:
    """Tarballs must opt into HF checkpoint extraction."""
    tarball_path = tmp_path / "bundle.tar"
    with tarfile.open(tarball_path, "w") as tar:
        _add_tar_member(tar, "config.yaml", yaml.safe_dump({"hf_ckpt": False}))

    with pytest.raises(RuntimeError, match="hf_ckpt"):
        resolve_model_bundle_path(tarball_path)


def test_resolve_model_bundle_path_rejects_invalid_path(tmp_path: Path) -> None:
    """Paths that are neither HF directories nor tarballs raise RuntimeError."""
    plain_file = tmp_path / "not_a_bundle.txt"
    plain_file.write_text("not a bundle", encoding="utf-8")

    with pytest.raises(RuntimeError, match="model_path"):
        resolve_model_bundle_path(plain_file)


def _add_tar_member(tar: tarfile.TarFile, name: str, content: str) -> None:
    """Add a UTF-8 text member to ``tar``."""
    data = content.encode("utf-8")
    member = tarfile.TarInfo(name)
    member.size = len(data)
    tar.addfile(member, io.BytesIO(data))
