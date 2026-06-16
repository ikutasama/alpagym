# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for the entry-point policy bundle registry."""

from __future__ import annotations

import pytest
from alpagym_plugins.plugins import PluginNotFoundError
from alpagym_runtime.policies import registry
from alpagym_runtime.policies.registry import get_policy_bundle


def test_get_policy_bundle_raises_for_uninstalled_kind() -> None:
    """Unknown policy kinds raise PluginNotFoundError that names the missing kind."""
    with pytest.raises(PluginNotFoundError, match="not_installed"):
        get_policy_bundle("not_installed")


def test_policy_bundle_registry_returns_sorted_names() -> None:
    """Available kinds is a list; empty when no policy packages are installed."""
    kinds = registry.policy_bundles.get_names()
    assert isinstance(kinds, list)
    assert kinds == sorted(kinds)


def test_get_policy_bundle_raises_type_error_for_non_policy_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A factory that returns something other than PolicyBundle raises TypeError."""
    kind = "bad_factory"

    def bad_factory() -> object:
        return object()

    def fake_get(_kind: str) -> object:
        return bad_factory

    monkeypatch.setattr(registry.policy_bundles, "get", fake_get)

    with pytest.raises(TypeError, match=kind):
        get_policy_bundle(kind)
