# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The workspace-pinned public AlpaSim commit must be checkout-able."""

from pathlib import Path

import pytest
from alpagym_host.alpasim_dependency import _fetch_commit, _validate_alpasim_layout
from alpagym_host.config import _resolve_alpasim_grpc_repo_ref


@pytest.mark.alpasim_e2e
def test_pinned_alpasim_ref_is_checkout_able(tmp_path: Path) -> None:
    """Fetch the pinned commit and confirm the Wizard layout is present.

    A pin to a pull-request head is unreachable by ``git clone`` and used to
    break Wizard startup; fetching by SHA over the network exercises the real
    provisioning path so a bad bump fails here instead of at run time. Marked
    ``alpasim_e2e`` (deselected by default) since it reaches GitHub.
    """
    checkout = tmp_path / "alpasim"
    _fetch_commit(
        "https://github.com/NVlabs/alpasim.git", _resolve_alpasim_grpc_repo_ref(), checkout
    )
    _validate_alpasim_layout(checkout)
