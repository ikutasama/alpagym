# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the duplicate config detection in AlpagymConfigDiscoveryPlugin."""

from pathlib import Path
from unittest.mock import patch

import pytest
from hydra_plugins.alpagym_config_discovery.search_path_plugin import (
    AlpagymConfigDiscoveryPlugin,
    find_duplicate_configs,
)


def _write_yaml(path: Path) -> None:
    """Create a minimal YAML file (and parent dirs) at the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# placeholder\n")


class TestFindDuplicateConfigs:
    """Unit tests for the duplicate-config detector."""

    def test_no_duplicates(self, tmp_path: Path) -> None:
        """Disjoint config trees produce no duplicates."""
        host = tmp_path / "host"
        plugin = tmp_path / "plugin"
        _write_yaml(host / "deploy" / "local.yaml")
        _write_yaml(plugin / "deploy" / "cluster.yaml")

        duplicates = find_duplicate_configs({"host": host, "plugin": plugin})

        assert duplicates == {}

    def test_detects_duplicate(self, tmp_path: Path) -> None:
        """Two providers shipping the same relative path are flagged."""
        host = tmp_path / "host"
        plugin = tmp_path / "plugin"
        _write_yaml(host / "deploy" / "local.yaml")
        _write_yaml(plugin / "deploy" / "local.yaml")

        duplicates = find_duplicate_configs({"host": host, "plugin": plugin})

        assert "deploy/local.yaml" in duplicates
        providers = {name for name, _ in duplicates["deploy/local.yaml"]}
        assert providers == {"host", "plugin"}

    def test_missing_directory_is_skipped(self, tmp_path: Path) -> None:
        """A provider whose directory does not exist is ignored."""
        host = tmp_path / "host"
        _write_yaml(host / "deploy" / "local.yaml")
        nonexistent = tmp_path / "does_not_exist"

        duplicates = find_duplicate_configs({"host": host, "missing": nonexistent})

        assert duplicates == {}


class TestProviderCollision:
    """Two search-path entries with the same provider but different dirs must error."""

    def test_same_provider_different_paths_raises(self, tmp_path: Path) -> None:
        """Provider-name collision across distinct directories raises ValueError."""
        from hydra._internal.config_search_path_impl import ConfigSearchPathImpl

        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        # Use an AlpaGym-owned provider name; the duplicate scan only considers
        # providers AlpaGym registered (see TestDuplicateScanScope).
        search_path = ConfigSearchPathImpl()
        search_path.append(provider="alpagym-plugin-dup", path=f"file://{dir_a}")
        search_path.append(provider="alpagym-plugin-dup", path=f"file://{dir_b}")

        plugin = AlpagymConfigDiscoveryPlugin()
        with patch(
            "hydra_plugins.alpagym_config_discovery.search_path_plugin.entry_points",
            return_value=[],
        ):
            with pytest.raises(ValueError, match="Provider name collision"):
                plugin.manipulate_search_path(search_path)


class TestDuplicateScanScope:
    """The duplicate scan is limited to AlpaGym-owned providers."""

    def _run_plugin(self, search_path) -> None:
        """Run the plugin with no real alpagym.configs entry points registered."""
        plugin = AlpagymConfigDiscoveryPlugin()
        with patch(
            "hydra_plugins.alpagym_config_discovery.search_path_plugin.entry_points",
            return_value=[],
        ):
            plugin.manipulate_search_path(search_path)

    def test_unowned_provider_duplicate_is_ignored(self, tmp_path: Path) -> None:
        """A non-AlpaGym provider (e.g. pkg://hydra.conf) sharing a path does not raise."""
        from hydra._internal.config_search_path_impl import ConfigSearchPathImpl

        host_dir = tmp_path / "host"
        hydra_dir = tmp_path / "hydra"
        _write_yaml(host_dir / "deploy" / "local.yaml")
        _write_yaml(hydra_dir / "deploy" / "local.yaml")

        search_path = ConfigSearchPathImpl()
        search_path.append(provider="main", path=f"file://{host_dir}")
        search_path.append(provider="hydra", path=f"file://{hydra_dir}")

        self._run_plugin(search_path)

    def test_owned_provider_duplicate_still_raises(self, tmp_path: Path) -> None:
        """Two AlpaGym-owned providers shipping the same relative path still raise."""
        from hydra._internal.config_search_path_impl import ConfigSearchPathImpl

        host_dir = tmp_path / "host"
        plugin_dir = tmp_path / "plugin"
        _write_yaml(host_dir / "deploy" / "cluster.yaml")
        _write_yaml(plugin_dir / "deploy" / "cluster.yaml")

        search_path = ConfigSearchPathImpl()
        search_path.append(provider="main", path=f"file://{host_dir}")
        search_path.append(provider="alpagym-plugin-extra", path=f"file://{plugin_dir}")

        with pytest.raises(ValueError, match="Duplicate Hydra config files"):
            self._run_plugin(search_path)


class TestPublicComposition:
    """The public install path composes with no extra config-providing plugins."""

    def test_deploy_local_composes_without_alpagym_configs_entry_points(
        self, tmp_path: Path
    ) -> None:
        """deploy=local composes when no alpagym.configs entry points are installed.

        Simulates an install where no deploy-preset plugin is present: only the
        bundled deploy/local.yaml is discoverable, so deploy=local composes and any
        plugin-provided deploy preset is unavailable.
        """
        from alpagym_host.config import register_config_schema
        from hydra import compose, initialize_config_module

        register_config_schema()
        with patch(
            "hydra_plugins.alpagym_config_discovery.search_path_plugin.entry_points",
            return_value=[],
        ):
            with initialize_config_module(version_base=None, config_module="alpagym_host.conf"):
                cfg = compose(
                    config_name="default",
                    overrides=[
                        f"run_root={tmp_path.as_posix()}",
                        "deploy=local",
                        "topology=local_colocated_1gpu",
                    ],
                )
                assert cfg.execution.backend == "local_process"

                with pytest.raises(Exception) as exc_info:
                    compose(
                        config_name="default",
                        overrides=[
                            f"run_root={tmp_path.as_posix()}",
                            "deploy=cluster",
                            "topology=local_colocated_1gpu",
                        ],
                    )
                assert "cluster" in str(exc_info.value)
