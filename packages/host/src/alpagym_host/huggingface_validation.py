# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import urllib.error
import urllib.request
from pathlib import Path


def validate_huggingface_access() -> None:
    """Validate HuggingFace access to the gated dataset that hosts AlpaSim scenes.

    AlpaSim serves its NuRec scenes from a gated HuggingFace dataset. This is a
    fail-fast credential check, run before the expensive Wizard and model
    startup. It catches the two failures that otherwise surface as opaque errors
    deep inside AlpaSim's scene download: no token at all, and a token without
    access to the dataset.

    The check is scene-agnostic. AlpaSim owns the scene catalog and the mapping
    from scene-sets to scenes, so AlpaGym validates only dataset-level access.
    That stays decoupled from how AlpaSim hosts scenes and works the same for
    individual scenes and scene-sets.

    This dataset is the only gated HuggingFace dependency, so dataset-level access
    is the complete token check: policy weights load from a local bundle and the
    tokenizer is either local or a public repo, neither of which needs a token. If
    AlpaSim ever adds a second gated source, extend this check to cover it.
    """
    token = _read_huggingface_token()
    if token is None:
        raise RuntimeError(
            "AlpaSim scenes live in the gated HuggingFace dataset "
            "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec, but no token was found. Run "
            "`hf auth login` or set HF_TOKEN to a token from an account with dataset access."
        )
    _check_huggingface_dataset_access(token)


def _read_huggingface_token() -> str | None:
    """Read the HuggingFace token the same way `huggingface_hub` does."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and token.strip():
        return token.strip()

    token_path = os.environ.get("HF_TOKEN_PATH")
    if token_path is None:
        hf_home_env = os.environ.get("HF_HOME")
        hf_home = Path(hf_home_env) if hf_home_env else Path.home() / ".cache" / "huggingface"
        token_path = str(hf_home / "token")

    try:
        token = Path(token_path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not token:
        return None
    return token


def _check_huggingface_dataset_access(token: str) -> None:
    """Confirm the token has access via HuggingFace's dataset auth-check endpoint.

    This is the endpoint `huggingface_hub.HfApi.auth_check` uses: a gated
    dataset returns 401/403 for a token without access and 200 otherwise.
    """
    url = (
        "https://huggingface.co/api/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/auth-check"
    )
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10):
            return
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError(
                "HuggingFace token cannot access the gated dataset "
                f"nvidia/PhysicalAI-Autonomous-Vehicles-NuRec (HTTP {exc.code}). Request or "
                "accept dataset access at "
                "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec, "
                "then rerun `hf auth login` or export HF_TOKEN with an approved token."
            ) from exc
        raise RuntimeError(
            "Could not validate HuggingFace access to "
            f"nvidia/PhysicalAI-Autonomous-Vehicles-NuRec (HTTP {exc.code})."
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Could not reach HuggingFace while validating access to "
            f"nvidia/PhysicalAI-Autonomous-Vehicles-NuRec: {exc}"
        ) from exc
