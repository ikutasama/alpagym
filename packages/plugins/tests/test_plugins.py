# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from importlib import metadata

import pytest
from alpagym_plugins.plugins import PluginNotFoundError, PluginRegistry


def test_package_init_is_doc_only() -> None:
    """The package root does not re-export the plugin registry API."""
    import alpagym_plugins

    assert not hasattr(alpagym_plugins, "PluginRegistry")
    assert not hasattr(alpagym_plugins, "PluginNotFoundError")
    assert not hasattr(alpagym_plugins, "get_plugin_info")


def test_get_missing_raises_with_available_names(monkeypatch) -> None:
    """PluginNotFoundError names the missing plugin and lists the installed ones."""
    known = [
        metadata.EntryPoint(name="known_bundle", value="pkg:get", group="alpagym.policy_bundles")
    ]
    monkeypatch.setattr("alpagym_plugins.plugins.metadata.entry_points", lambda group: known)
    reg = PluginRegistry("alpagym.policy_bundles")
    with pytest.raises(PluginNotFoundError) as exc:
        reg.get("does_not_exist")
    message = str(exc.value)
    assert "does_not_exist" in message
    assert "known_bundle" in message


def test_get_names_is_sorted() -> None:
    """get_names() returns names in lexicographic order."""
    names = PluginRegistry("alpagym.configs").get_names()
    assert names == sorted(names)


def test_duplicate_entry_points_raise(monkeypatch) -> None:
    """get_entry_points() raises ValueError naming the duplicated entry point."""
    dupes = [
        metadata.EntryPoint(name="dup", value="a:x", group="alpagym.policy_bundles"),
        metadata.EntryPoint(name="dup", value="b:y", group="alpagym.policy_bundles"),
    ]
    monkeypatch.setattr("alpagym_plugins.plugins.metadata.entry_points", lambda group: dupes)
    reg = PluginRegistry("alpagym.policy_bundles")
    with pytest.raises(ValueError, match="dup"):
        reg.get_entry_points()
