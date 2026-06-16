# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tomllib

from alpagym_host.config import alpagym_project_root


def test_alpasim_config_plugin_registers_wizard_config_entrypoint() -> None:
    """The AlpaGym AlpaSim config package is discoverable by Wizard Hydra."""
    plugin_root = alpagym_project_root() / "packages" / "alpasim_configs"
    pyproject = tomllib.loads((plugin_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "alpagym-alpasim-configs"
    assert pyproject["project"]["entry-points"]["alpasim.configs"] == {
        "alpagym": "alpagym_alpasim_configs.configs",
    }
    assert "packages/alpasim_configs" in pyproject_root_members()


def pyproject_root_members() -> list[str]:
    """Return the AlpaGym uv workspace members from the root project file."""
    pyproject_path = alpagym_project_root() / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return list(pyproject["tool"]["uv"]["workspace"]["members"])
