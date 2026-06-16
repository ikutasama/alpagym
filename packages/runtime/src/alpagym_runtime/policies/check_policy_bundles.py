# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime check for installed AlpaGym policy bundle entry points."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from alpagym_runtime.policies.registry import get_policy_bundle, policy_bundles


def check_policy_bundles() -> list[str]:
    """Load all installed policy bundles and validate their runtime hook shape."""
    installed_kinds = sorted(entry_point.name for entry_point in policy_bundles.get_entry_points())
    if not installed_kinds:
        raise ValueError("No installed policy bundles found")

    # Loading each bundle runs the PolicyBundle hook validation; no raise means success.
    for kind in installed_kinds:
        get_policy_bundle(kind)
    return installed_kinds


def main(argv: Sequence[str] | None = None) -> None:
    """Run the policy bundle check from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    checked = check_policy_bundles()
    for kind in checked:
        print(f"Validated policy bundle {kind}")


if __name__ == "__main__":
    main()
