# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Alpamayo R1 bundle's ``load_inference_model`` entry point.

Reached through ``get_policy_bundle('alpamayo_r1')`` and pinning the four
R1-only boundary knobs: ``attn_implementation='sdpa'``, ``dtype='bfloat16'``,
the ``model_type`` assertion, and the post-load
``legacy_inference_image_input_format=True`` patch.
``alpamayo1_x_rl.models.expert_model.model.ExpertModelRL`` is stubbed so no
real expert weights load.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import alpamayo1_x_rl.models.expert_model.model
import pytest
import torch
from alpagym_alpamayo_r1.inference_model import AlpamayoR1InferenceModel
from alpagym_runtime.policies.registry import get_policy_bundle

_EXPERT_MODEL_TYPE = "alpamayo_reasoning_vla_expert"


def _write_bundle(bundle_dir: Path, model_type: str = _EXPERT_MODEL_TYPE) -> None:
    """Materialize a minimal HF bundle directory so ``resolve_model_bundle_path`` accepts it."""
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "config.json").write_text(
        json.dumps({"model_type": model_type}), encoding="utf-8"
    )


def _patch_expert_model_rl(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_type: str = _EXPERT_MODEL_TYPE,
    legacy_inference_image_input_format: bool = False,
) -> dict[str, Any]:
    """Stub the recipe ``ExpertModelRL.from_pretrained`` and capture call args.

    Patches ``alpamayo1_x_rl.models.expert_model.model.ExpertModelRL``.
    """
    captured: dict[str, Any] = {}

    class _FakeExpertModelRL:
        @classmethod
        def from_pretrained(cls, path: Any, **kwargs: Any) -> Any:
            """Record the call and return a fake model with a configurable ``.config``."""
            captured["model_path"] = path
            captured["model_kwargs"] = kwargs
            config = SimpleNamespace(
                model_type=model_type,
                legacy_inference_image_input_format=legacy_inference_image_input_format,
            )
            return SimpleNamespace(config=config)

    monkeypatch.setattr(
        alpamayo1_x_rl.models.expert_model.model, "ExpertModelRL", _FakeExpertModelRL
    )
    return captured


def _make_run_config(bundle: Path, num_context_frames: int) -> SimpleNamespace:
    """Build a minimal ``run_config`` exposing ``policy.model.{path,num_context_frames}``."""
    return SimpleNamespace(
        policy=SimpleNamespace(
            model=SimpleNamespace(
                path=str(bundle),
                num_context_frames=num_context_frames,
            )
        )
    )


def test_load_alpamayo_r1_contract_builds_r1_inference_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The R1 bundle entry point builds an :class:`AlpamayoR1InferenceModel`.

    Pins three R1-only knobs at the boundary: ``attn_implementation='sdpa'``,
    ``dtype='bfloat16'``, and ``device_map`` derived from the device.
    """
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, model_type=_EXPERT_MODEL_TYPE)
    captured = _patch_expert_model_rl(monkeypatch, legacy_inference_image_input_format=False)

    result = get_policy_bundle("alpamayo_r1").load_inference_model(
        _make_run_config(bundle, num_context_frames=4),
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert isinstance(result, AlpamayoR1InferenceModel)
    assert Path(captured["model_path"]) == bundle
    assert captured["model_kwargs"]["dtype"] == "bfloat16"
    assert captured["model_kwargs"]["attn_implementation"] == "sdpa"
    assert str(captured["model_kwargs"]["device_map"]) == "cpu"
    assert result._num_context_frames == 4


def test_load_alpamayo_r1_forces_legacy_inference_image_input_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bundles shipping ``legacy_inference_image_input_format=False`` are patched to True."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, model_type=_EXPERT_MODEL_TYPE)
    _patch_expert_model_rl(monkeypatch, legacy_inference_image_input_format=False)

    result = get_policy_bundle("alpamayo_r1").load_inference_model(
        _make_run_config(bundle, num_context_frames=4),
        torch.device("cpu"),
        torch.bfloat16,
    )

    assert isinstance(result, AlpamayoR1InferenceModel)
    # The adapter feeds frames in [-1, 1] and relies on the model's internal
    # ``(x + 1) / 2`` rescale; this flag must be True at inference time.
    assert result._model.config.legacy_inference_image_input_format is True


def test_load_alpamayo_r1_rejects_non_bfloat16_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The R1 bundle refuses anything but bfloat16."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, model_type=_EXPERT_MODEL_TYPE)
    _patch_expert_model_rl(monkeypatch)

    with pytest.raises(ValueError, match="dtype=bfloat16"):
        get_policy_bundle("alpamayo_r1").load_inference_model(
            _make_run_config(bundle, num_context_frames=4),
            torch.device("cpu"),
            torch.float32,
        )


def test_load_alpamayo_r1_rejects_wrong_model_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The R1 bundle refuses bundles whose ``model_type`` does not match."""
    bundle = tmp_path / "bundle"
    _write_bundle(bundle, model_type="some_other_model")
    _patch_expert_model_rl(monkeypatch, model_type="some_other_model")

    with pytest.raises(ValueError, match=_EXPERT_MODEL_TYPE):
        get_policy_bundle("alpamayo_r1").load_inference_model(
            _make_run_config(bundle, num_context_frames=4),
            torch.device("cpu"),
            torch.bfloat16,
        )
