# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hydra SearchPathPlugin that auto-discovers config directories from installed alpagym plugins.

Any installed package can register its config directory by adding an entry point
in the ``alpagym.configs`` group. For example, in ``pyproject.toml``::

    [project.entry-points."alpagym.configs"]
    my_plugin = "my_plugin.configs"

When Hydra initialises, this plugin discovers all such entry points and adds
``pkg://<entry_point_value>`` to Hydra's config search path. Plugin configs
become available for composition without manual ``hydra.searchpath`` overrides.

After registering paths the plugin walks the config directories AlpaGym itself
registered -- the primary ``alpagym_host.conf`` and any ``alpagym.configs``
plugin root -- and raises ``ValueError`` if two of them ship a YAML file at the
same relative path (e.g. both ``alpagym_host`` and a site plugin ship
``deploy/local.yaml``). Configs shipped by Hydra itself (``pkg://hydra.conf``)
or by unrelated SearchPathPlugins are out of scope and never trigger the check.
"""

import importlib.util
import logging
from importlib.metadata import entry_points
from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

logger = logging.getLogger(__name__)


def find_duplicate_configs(
    config_dirs: dict[str, Path],
) -> dict[str, list[tuple[str, Path]]]:
    """Find YAML config files that exist in more than one provider directory.

    Args:
        config_dirs: Mapping from provider name to its config root directory.

    Returns:
        Mapping from relative YAML path to a list of ``(provider, absolute_path)``
        tuples, only for paths that appear in two or more providers.
    """
    seen: dict[str, list[tuple[str, Path]]] = {}
    for provider, root in config_dirs.items():
        if not root.is_dir():
            continue
        for yaml_file in sorted(root.rglob("*.yaml")):
            rel = str(yaml_file.relative_to(root))
            seen.setdefault(rel, []).append((provider, yaml_file))
    return {rel: providers for rel, providers in seen.items() if len(providers) > 1}


def _resolve_search_path_element(path: str) -> Path | None:
    """Resolve a Hydra search-path string (``file://`` or ``pkg://``) to a directory."""
    if path.startswith("file://"):
        return Path(path.removeprefix("file://"))
    if path.startswith("pkg://"):
        module_path = path.removeprefix("pkg://")
        spec = importlib.util.find_spec(module_path)
        if spec and spec.submodule_search_locations:
            return Path(spec.submodule_search_locations[0])
    return None


class AlpagymConfigDiscoveryPlugin(SearchPathPlugin):
    """Discover and register config search paths from ``alpagym.configs`` entry points."""

    provider = "alpagym-config-discovery"

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Append config roots from every ``alpagym.configs`` entry point."""
        eps = entry_points(group="alpagym.configs")

        for ep in eps:
            path = f"pkg://{ep.value}"
            logger.debug("Auto-registering config search path: %s (from %s)", path, ep.name)
            search_path.append(
                provider=f"alpagym-plugin-{ep.name}",
                path=path,
            )

        config_dirs: dict[str, Path] = {}
        for element in search_path.get_path():
            # Limit the scan to providers AlpaGym owns: "main" (the primary
            # alpagym_host.conf) and the "alpagym-plugin-" entries appended above.
            # Configs from Hydra or unrelated plugins are out of scope.
            if element.provider != "main" and not element.provider.startswith("alpagym-plugin-"):
                continue
            resolved = _resolve_search_path_element(element.path)
            if resolved is None or not resolved.is_dir():
                continue
            if element.provider in config_dirs:
                existing = config_dirs[element.provider]
                if resolved != existing:
                    raise ValueError(
                        f"Provider name collision: {element.provider!r} maps to "
                        f"both {existing} and {resolved}. Each alpagym.configs "
                        f"entry point must use a unique name."
                    )
            config_dirs[element.provider] = resolved

        duplicates = find_duplicate_configs(config_dirs)
        if duplicates:
            lines = []
            for rel, providers in sorted(duplicates.items()):
                providers_str = ", ".join(f"{name} ({fpath})" for name, fpath in providers)
                lines.append(f"  {rel}: {providers_str}")
            raise ValueError(
                "Duplicate Hydra config files found across config providers. "
                "Each config file path must be unique across all providers:\n" + "\n".join(lines)
            )
