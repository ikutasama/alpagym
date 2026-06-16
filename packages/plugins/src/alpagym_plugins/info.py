# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI to print installed AlpaGym plugins."""

from __future__ import annotations

from alpagym_plugins.plugins import get_plugin_info


def main() -> None:
    """Print a summary of installed AlpaGym plugins."""
    info = get_plugin_info()
    labels = {
        "alpagym.policy_bundles": "Policy bundles",
        "alpagym.configs": "Configs",
    }
    for group, names in info.items():
        label = labels[group]
        line = f"{label}: {', '.join(names)}" if names else f"{label}: (none)"
        print(line)


if __name__ == "__main__":
    main()
