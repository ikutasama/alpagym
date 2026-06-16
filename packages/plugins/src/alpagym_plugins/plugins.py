# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry-point plugin registry for AlpaGym extensibility."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from importlib import metadata
from typing import Any

logger = logging.getLogger(__name__)


class PluginNotFoundError(ValueError):
    """Raised when a requested plugin is not registered."""


class PluginRegistry:
    """Discover installed plugins from one Python entry-point group."""

    def __init__(self, group: str) -> None:
        """Initialize a registry for ``group``."""
        self._group = group
        self._cache: dict[str, Any] = {}

    def get_entry_points(self) -> list[metadata.EntryPoint]:
        """Return sorted entry points after rejecting duplicate names."""
        entry_points = list(metadata.entry_points(group=self._group))
        duplicates = _duplicate_names(entry_points)
        if duplicates:
            raise ValueError(
                f"Duplicate entry points in {self._group}: {', '.join(sorted(duplicates))}"
            )
        return sorted(entry_points, key=lambda entry_point: entry_point.name)

    def get_names(self) -> list[str]:
        """Return sorted installed plugin names."""
        return [entry_point.name for entry_point in self.get_entry_points()]

    def get(self, name: str) -> Any:
        """Return the loaded object for plugin ``name``."""
        if name in self._cache:
            return self._cache[name]
        matches = [ep for ep in self.get_entry_points() if ep.name == name]
        if not matches:
            raise PluginNotFoundError(
                f"Plugin {name!r} not found in {self._group}. "
                f"Available: {self.get_names()}. Install a package that provides this plugin."
            )
        self._cache[name] = matches[0].load()
        return self._cache[name]


def get_plugin_info() -> dict[str, list[str]]:
    """Return installed AlpaGym plugin names by entry-point group."""
    return {
        "alpagym.policy_bundles": PluginRegistry("alpagym.policy_bundles").get_names(),
        "alpagym.configs": PluginRegistry("alpagym.configs").get_names(),
    }


def _duplicate_names(entry_points: Iterable[metadata.EntryPoint]) -> set[str]:
    """Return entry-point names that appear more than once."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for entry_point in entry_points:
        if entry_point.name in seen:
            duplicates.add(entry_point.name)
        seen.add(entry_point.name)
    return duplicates
