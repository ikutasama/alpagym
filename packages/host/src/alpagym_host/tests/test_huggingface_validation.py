# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import urllib.error
import urllib.request
from pathlib import Path

import pytest
from alpagym_host import huggingface_validation
from alpagym_host.huggingface_validation import validate_huggingface_access


def test_validate_huggingface_access_probes_dataset_with_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token the auth-check endpoint accepts passes, and the request is authenticated."""
    monkeypatch.setenv("HF_TOKEN", "approved-token")

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    captured: dict[str, str | None] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: int) -> _Response:
        del timeout
        captured["url"] = request.full_url
        captured["authorization"] = request.get_header("Authorization")
        return _Response()

    monkeypatch.setattr(huggingface_validation.urllib.request, "urlopen", fake_urlopen)

    validate_huggingface_access()

    assert captured["url"].endswith(
        "/api/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec/auth-check"
    )
    assert captured["authorization"] == "Bearer approved-token"


def test_validate_huggingface_access_requires_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation fails with an actionable message when no token is found."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    with pytest.raises(RuntimeError, match="no token was found"):
        validate_huggingface_access()


def test_validate_huggingface_access_explains_gated_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 403 from the auth-check endpoint becomes an actionable gated-dataset error."""
    monkeypatch.setenv("HF_TOKEN", "present-but-unapproved")

    def raise_forbidden(request: urllib.request.Request, timeout: int) -> None:
        del timeout
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=403,
            msg="Forbidden",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

    monkeypatch.setattr(huggingface_validation.urllib.request, "urlopen", raise_forbidden)

    with pytest.raises(RuntimeError, match="Request or accept dataset access"):
        validate_huggingface_access()


def test_validate_huggingface_access_wraps_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A socket timeout becomes the actionable reachability error, not a raw traceback.

    urlopen raises TimeoutError (an OSError, not a URLError) on a header/read
    timeout, so the handler must catch it alongside URLError.
    """
    monkeypatch.setenv("HF_TOKEN", "present-token")

    def raise_timeout(request: urllib.request.Request, timeout: int) -> None:
        del request, timeout
        raise TimeoutError("timed out")

    monkeypatch.setattr(huggingface_validation.urllib.request, "urlopen", raise_timeout)

    with pytest.raises(RuntimeError, match="Could not reach HuggingFace"):
        validate_huggingface_access()
