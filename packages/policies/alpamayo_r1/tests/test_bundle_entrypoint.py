# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Alpamayo R1 is discoverable as an installed alpagym.policy_bundles plugin."""

from __future__ import annotations

import inspect

from alpagym_runtime.policies.registry import PolicyBundle, get_policy_bundle, policy_bundles


def test_alpamayo_r1_bundle_is_discoverable() -> None:
    """The installed R1 package registers an alpagym.policy_bundles entry point."""
    assert "alpamayo_r1" in policy_bundles.get_names()


def test_alpamayo_r1_get_policy_bundle_returns_hooks() -> None:
    """get_policy_bundle resolves the R1 entry point to a callable PolicyBundle."""
    bundle = get_policy_bundle("alpamayo_r1")
    assert isinstance(bundle, PolicyBundle)
    for hook in (
        bundle.setup_tokenizer,
        bundle.build_data_packer,
        bundle.install_runtime_bridge,
        bundle.load_inference_model,
        bundle.build_model_inputs,
    ):
        assert callable(hook)
    assert list(inspect.signature(bundle.build_data_packer).parameters) == [
        "run_config",
        "cosmos_role",
    ]
