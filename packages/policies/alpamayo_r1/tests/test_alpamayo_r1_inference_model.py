# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `AlpamayoR1InferenceModel` contract.

The adapter wraps `ExpertModel.sample_trajectories_from_data` and the
module-level `tokenize_for_generation`. These tests pin the contract:
- prompt tokenization replaces ``data["tokenized_data"]`` before sampling;
- ``with_vlm_rollout`` flips with ``last_component``;
- ``return_extra`` follows the trace flag (the SDE branch returns a
  4-tuple regardless, but the call site stays semantically correct);
- the ``num_traj_sets`` axis is preserved through ``BatchedModelOutput``
  so the alpagym selector can index ``pred_xyz[set_ix, sample_ix]``;
- replay metadata flows into ``extra`` with a real leading batch axis for unbind.

The model is a MagicMock substitute and ``tokenize_for_generation`` is
monkeypatched; no real ExpertModel is loaded.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from alpagym_alpamayo_r1.inference_model import AlpamayoR1InferenceModel
from alpagym_host.config import DiffusionSamplingConfig, SamplingParamsConfig
from alpagym_runtime.inference.types import BatchedModelInput, ModelInput
from alpagym_runtime.replay import ActionSelection

_B = 1
_NUM_TRAJ_SETS = 1
_NUM_TRAJ_SAMPLES = 1
_NUM_FUTURE = 8
_NUM_HISTORY = 4
_NUM_CONTEXT_FRAMES = 2
_NUM_CAMERAS = 2
_H = 4
_W = 4


def _make_batched_input(
    batch_size: int = _B,
    seed: torch.Tensor | None = None,
) -> BatchedModelInput:
    """Construct a batched input mirroring AlpamayoPolicy._preprocess."""
    camera_frames = torch.full(
        (batch_size, _NUM_CAMERAS * _NUM_CONTEXT_FRAMES, 3, _H, _W),
        127,
        dtype=torch.uint8,
    )
    camera_indices = torch.tensor([[0, 0, 1, 1]], dtype=torch.int64).repeat(batch_size, 1)
    relative_timestamps = torch.zeros(
        (batch_size, _NUM_CAMERAS * _NUM_CONTEXT_FRAMES), dtype=torch.int64
    )
    ego_history_xyz = torch.zeros((batch_size, _NUM_HISTORY, 3), dtype=torch.float32)
    ego_history_rot = (
        torch.eye(3, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(_NUM_HISTORY, 1, 1)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
    )
    route_xy = torch.zeros((batch_size, 4, 2), dtype=torch.float32)
    return BatchedModelInput(
        ego_history_xyz=ego_history_xyz,
        ego_history_rot=ego_history_rot,
        camera_frames=camera_frames,
        camera_indices=camera_indices,
        relative_timestamps=relative_timestamps,
        route_xy=route_xy,
        seed=seed,
    )


def _single_model_input(batched: BatchedModelInput, index: int = 0) -> ModelInput:
    """Slice a test batch into the per-output input consumed by replay packing."""
    return ModelInput(
        ego_history_xyz=batched.ego_history_xyz[index],
        ego_history_rot=batched.ego_history_rot[index],
        camera_frames=batched.camera_frames[index],
        camera_indices=batched.camera_indices[index],
        relative_timestamps=batched.relative_timestamps[index],
        route_xy=batched.route_xy[index],
        seed=batched.seed[index] if batched.seed is not None else None,
    )


def _make_sampling(
    num_traj_samples: int = _NUM_TRAJ_SAMPLES,
    num_traj_sets: int = _NUM_TRAJ_SETS,
    force_determinism: bool = False,
    retry_on_nonfinite: bool = True,
) -> SamplingParamsConfig:
    """Build a sampling config exercising the R1 diffusion knobs."""
    return SamplingParamsConfig(
        top_p=0.98,
        top_k=None,
        temperature=0.6,
        num_traj_samples=num_traj_samples,
        num_traj_sets=num_traj_sets,
        diffusion_kwargs=DiffusionSamplingConfig(
            int_method="sde",
            noise_level=0.4,
            inference_step=10,
        ),
        force_determinism=force_determinism,
        retry_on_nonfinite=retry_on_nonfinite,
    )


def _make_sde_tuple(
    batch_size: int = _B,
    num_traj_samples: int = _NUM_TRAJ_SAMPLES,
    num_traj_sets: int = _NUM_TRAJ_SETS,
    with_replay: bool = True,
    sample_offset: float = 0.0,
) -> tuple[Any, ...]:
    """Build a fake ExpertModel SDE 4-tuple matching the documented contract."""
    pred_xyz = torch.randn((batch_size, num_traj_sets, num_traj_samples, _NUM_FUTURE, 3))
    pred_rot = (
        torch.eye(3).expand(batch_size, num_traj_sets, num_traj_samples, _NUM_FUTURE, 3, 3).clone()
    )
    logprob = torch.randn((batch_size, num_traj_sets, num_traj_samples))
    sde_info: dict[str, Any] = {}
    if with_replay:
        num_candidates = num_traj_sets * num_traj_samples
        sde_info["noise_level"] = 0.4
        sde_info["samples_list"] = (
            torch.arange(
                batch_size * num_candidates * (_NUM_FUTURE + 1) * _NUM_FUTURE * 3,
                dtype=torch.float32,
            ).reshape(batch_size * num_candidates, _NUM_FUTURE + 1, _NUM_FUTURE, 3)
            + sample_offset
        )
        sde_info["timesteps"] = torch.linspace(0.0, 1.0, _NUM_FUTURE + 1)
        # One VLM generation per rollout row (per-row leaf), not per SDE candidate.
        sde_info["vlm_generated_ids_list"] = [
            torch.full((32,), i, dtype=torch.int64) for i in range(batch_size)
        ]
    return pred_xyz, pred_rot, logprob, sde_info


def _make_mock_model(sde_tuple: tuple[Any, ...]) -> MagicMock:
    """Stand-in for ExpertModelRL with the sampler method the adapter calls."""
    model = MagicMock()
    model.sample_trajectories_from_data = MagicMock(return_value=sde_tuple)
    return model


@pytest.fixture(autouse=True)
def tokenize_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the module-level tokenize_for_generation; return the mock for assertion."""
    mock_fn = MagicMock(return_value={"input_ids": torch.zeros((1, 64), dtype=torch.int64)})
    monkeypatch.setattr("alpagym_alpamayo_r1.inference_model.tokenize_for_generation", mock_fn)
    return mock_fn


def test_pack_data_replaces_tokenized_data_with_prompt_td(tokenize_mock: MagicMock) -> None:
    """The data dict passed to the sampler must carry ``prompt_td`` as tokenized_data."""
    model = _make_mock_model(_make_sde_tuple())
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)
    adapter.sample_trajectories_from_data(_make_batched_input(), _make_sampling())

    sample_call = model.sample_trajectories_from_data.call_args
    data_arg = sample_call.args[0]
    prompt_td = tokenize_mock.return_value
    assert "tokenized_data" in data_arg
    assert torch.equal(data_arg["tokenized_data"]["input_ids"], prompt_td["input_ids"])


def test_traj_future_skips_vlm_rollout() -> None:
    """``last_component='traj_future'`` is the prefill-only fast path."""
    model = _make_mock_model(_make_sde_tuple())
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)
    adapter.sample_trajectories_from_data(_make_batched_input(), _make_sampling())

    kwargs = model.sample_trajectories_from_data.call_args.kwargs
    assert kwargs["with_vlm_rollout"] is False
    assert kwargs["last_component"] == "traj_future"


def test_cot_enables_vlm_rollout() -> None:
    """``last_component='cot'`` runs the autoregressive VLM step."""
    model = _make_mock_model(_make_sde_tuple())
    adapter = AlpamayoR1InferenceModel(
        model=model, num_context_frames=_NUM_CONTEXT_FRAMES, last_component="cot"
    )
    adapter.sample_trajectories_from_data(_make_batched_input(), _make_sampling())

    kwargs = model.sample_trajectories_from_data.call_args.kwargs
    assert kwargs["with_vlm_rollout"] is True
    assert kwargs["last_component"] == "cot"


def test_diffusion_kwargs_filter_none_and_force_return_info() -> None:
    """None-valued diffusion knobs are dropped; ``return_info=True`` is set internally."""
    sampling = SamplingParamsConfig(
        top_p=0.98,
        top_k=None,
        temperature=0.6,
        num_traj_samples=_NUM_TRAJ_SAMPLES,
        num_traj_sets=_NUM_TRAJ_SETS,
        diffusion_kwargs=DiffusionSamplingConfig(
            int_method="sde",
            noise_level=0.4,
            inference_step=10,
            temperature=None,
        ),
    )
    model = _make_mock_model(_make_sde_tuple())
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)
    adapter.sample_trajectories_from_data(_make_batched_input(), sampling)

    diffusion_kwargs = model.sample_trajectories_from_data.call_args.kwargs["diffusion_kwargs"]
    assert diffusion_kwargs == {
        "int_method": "sde",
        "noise_level": 0.4,
        "inference_step": 10,
        "return_info": True,
    }


def test_deterministic_sampling_adds_seeded_generator() -> None:
    """Deterministic R1 sampling seeds the diffusion generator for the request."""
    seeded_inputs = _make_batched_input(seed=torch.tensor([31], dtype=torch.int64))
    model = _make_mock_model(_make_sde_tuple())
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)

    adapter.sample_trajectories_from_data(
        seeded_inputs,
        _make_sampling(force_determinism=True),
    )

    diffusion_kwargs = model.sample_trajectories_from_data.call_args.kwargs["diffusion_kwargs"]
    assert diffusion_kwargs["int_method"] == "sde"
    assert diffusion_kwargs["noise_level"] == 0.4
    assert diffusion_kwargs["return_info"] is True
    generator = diffusion_kwargs["generator"]
    assert isinstance(generator, torch.Generator)
    assert generator.initial_seed() == 31


def test_deterministic_sampling_splits_batched_inputs_by_seed() -> None:
    """A deterministic B>1 call is replayed as B single-row calls with per-session seeds."""
    model = MagicMock()

    def sample_side_effect(data: dict[str, Any], **kwargs: Any) -> tuple[Any, ...]:
        generator = kwargs["diffusion_kwargs"]["generator"]
        seed = generator.initial_seed()
        assert data["image_frames"].shape[0] == 1
        return _make_sde_tuple(batch_size=1, sample_offset=float(seed))

    model.sample_trajectories_from_data = MagicMock(side_effect=sample_side_effect)
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)
    seeded_inputs = _make_batched_input(
        batch_size=2,
        seed=torch.tensor([31, 37], dtype=torch.int64),
    )

    output = adapter.sample_trajectories_from_data(
        seeded_inputs,
        _make_sampling(force_determinism=True),
        return_trace_for_rl=True,
    )

    assert model.sample_trajectories_from_data.call_count == 2
    generators = [
        call.kwargs["diffusion_kwargs"]["generator"]
        for call in model.sample_trajectories_from_data.call_args_list
    ]
    assert [generator.initial_seed() for generator in generators] == [31, 37]
    assert output.pred_xyz.shape[0] == 2
    assert output.extra["samples_list"].shape[0] == 2
    assert output.extra["timesteps"].shape[0] == 2
    assert len(output.unbind()) == 2

    # Merge must keep input row order: row 0 ran with seed 31, row 1 with seed 37, and the
    # side effect offsets every ``samples_list`` value by the seed (arange starts at 0). A
    # reversed merge would swap these while still passing every assertion above.
    assert output.extra["samples_list"][0].flatten()[0] == 31.0
    assert output.extra["samples_list"][1].flatten()[0] == 37.0


def test_return_extra_follows_trace_flag() -> None:
    """``return_extra`` is forwarded as-is to the underlying model call."""
    for flag in (False, True):
        model = _make_mock_model(_make_sde_tuple())
        adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)
        adapter.sample_trajectories_from_data(
            _make_batched_input(), _make_sampling(), return_trace_for_rl=flag
        )

        assert model.sample_trajectories_from_data.call_args.kwargs["return_extra"] is flag


def test_outputs_preserve_5d_layout_for_alpagym_unbind_contract() -> None:
    """BatchedModelOutput shape stays ``(B, num_traj_sets, num_traj_samples, T, 3)``.

    The alpagym selector indexes ``pred_xyz[set_ix, sample_ix]`` after
    ``BatchedModelOutput.unbind`` strips the leading ``B`` axis, and
    ``AlpamayoPolicy._postprocess`` reads ``shape[2]`` as the horizon.
    Collapsing the ``num_traj_sets`` axis here would feed the policy a
    rank-3 tensor and corrupt both reads.
    """
    model = _make_mock_model(_make_sde_tuple())
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)
    output = adapter.sample_trajectories_from_data(
        _make_batched_input(), _make_sampling(), return_trace_for_rl=True
    )

    assert output.pred_xyz.shape == (_B, _NUM_TRAJ_SETS, _NUM_TRAJ_SAMPLES, _NUM_FUTURE, 3)
    assert output.pred_rot.shape == (_B, _NUM_TRAJ_SETS, _NUM_TRAJ_SAMPLES, _NUM_FUTURE, 3, 3)
    assert output.logprob is not None
    assert output.logprob.shape == (_B, _NUM_TRAJ_SETS, _NUM_TRAJ_SAMPLES)


def test_logprob_and_extra_omitted_when_trace_flag_off() -> None:
    """The RL trace payload (logprob + extra) is gated on ``return_trace_for_rl``.

    ``AlpamayoPolicy._postprocess`` derives ``chosen_logprob`` and
    ``replay_data`` from a non-None logprob; emitting one with the flag off
    would leak training artifacts into a trace-disabled run.
    """
    model = _make_mock_model(_make_sde_tuple())
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)
    output = adapter.sample_trajectories_from_data(
        _make_batched_input(), _make_sampling(), return_trace_for_rl=False
    )

    assert output.logprob is None
    assert output.extra == {}


def test_extra_replay_payload_has_leading_batch_axis_for_unbind() -> None:
    """Replay extra exposes a real leading B axis for every unbindable leaf.

    `samples_list` carries the full `[B, K, T+1, Tf, D]` candidate trace and
    `timesteps` is broadcast to `[B, T+1]`. The rollout `ModelInput` is
    persisted by `build_policy_replay_data` directly via `asdict`, so it does
    not appear in `extra`. Candidate selection happens in
    `build_policy_replay_data`.
    """
    batch_size = 2
    model = _make_mock_model(_make_sde_tuple(batch_size=batch_size))
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)
    output = adapter.sample_trajectories_from_data(
        _make_batched_input(batch_size=batch_size), _make_sampling(), return_trace_for_rl=True
    )

    assert "input_data" not in output.extra

    assert output.extra["samples_list"].shape == (
        batch_size,
        _NUM_TRAJ_SETS * _NUM_TRAJ_SAMPLES,
        _NUM_FUTURE + 1,
        _NUM_FUTURE,
        3,
    )
    assert output.extra["timesteps"].shape == (batch_size, _NUM_FUTURE + 1)
    assert output.extra["noise_level"] == [0.4, 0.4]
    assert len(output.extra["vlm_generated_ids"]) == batch_size
    assert output.extra["vlm_generated_ids"][0].shape == (32,)


def test_extra_replay_payload_survives_batched_unbind_roundtrip() -> None:
    """The SDE replay ``extra`` leaves pass ``BatchedModelOutput.unbind`` with row identity.

    Covers the ``traj_future`` path's leaves (``samples_list`` / ``timesteps`` /
    ``noise_level`` / ``vlm_generated_ids``); ``cot`` is not exercised here as it only
    appears under the single-candidate ``last_component='cot'`` path.
    """
    batch_size = 2
    model = _make_mock_model(_make_sde_tuple(batch_size=batch_size))
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)
    inputs = _make_batched_input(batch_size=batch_size)
    output = adapter.sample_trajectories_from_data(
        inputs, _make_sampling(), return_trace_for_rl=True
    )

    per_output = output.unbind()
    assert len(per_output) == batch_size
    for row, model_output in enumerate(per_output):
        for key in ("samples_list", "timesteps", "noise_level", "vlm_generated_ids"):
            assert key in model_output.extra, f"missing {key!r} in per-output extra"
        assert torch.equal(model_output.extra["samples_list"], output.extra["samples_list"][row])
        assert torch.equal(model_output.extra["timesteps"], output.extra["timesteps"][row])
        assert model_output.extra["noise_level"] == 0.4
        assert torch.equal(model_output.extra["vlm_generated_ids"], torch.full((32,), row))

    replay = adapter.build_policy_replay_data(
        _single_model_input(inputs, index=1),
        per_output[1],
        ActionSelection(set_ix=0, sample_ix=0),
    )
    assert torch.equal(replay.payload["samples_list"], output.extra["samples_list"][1, 0])
    assert torch.equal(replay.payload["timesteps"], output.extra["timesteps"][1])


def test_replay_selection_indexes_set_then_sample() -> None:
    """``build_policy_replay_data`` selects the (set_ix, sample_ix) candidate, not its transpose."""
    num_traj_sets, num_traj_samples = 2, 2
    model = _make_mock_model(
        _make_sde_tuple(
            batch_size=1,
            num_traj_sets=num_traj_sets,
            num_traj_samples=num_traj_samples,
        )
    )
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)
    inputs = _make_batched_input(batch_size=1)
    output = adapter.sample_trajectories_from_data(
        inputs,
        _make_sampling(num_traj_sets=num_traj_sets, num_traj_samples=num_traj_samples),
        return_trace_for_rl=True,
    )

    # Candidates are flattened in (set, sample) order, so set_ix=1/sample_ix=0 is flat index
    # 1*num_traj_samples+0 = 2 -- distinct from the transposed 0*num_traj_sets+1 = 1, which a
    # set/sample swap in the selector would pick. The arange fixture gives each candidate a
    # distinct value range, so the two assertions below fail under a transpose.
    replay = adapter.build_policy_replay_data(
        _single_model_input(inputs, index=0),
        output.unbind()[0],
        ActionSelection(set_ix=1, sample_ix=0),
    )
    assert torch.equal(replay.payload["samples_list"], output.extra["samples_list"][0, 2])
    assert not torch.equal(replay.payload["samples_list"], output.extra["samples_list"][0, 1])


def test_cot_with_multi_sample_rejected_early() -> None:
    """``last_component='cot'`` + ``num_traj_samples>1`` is rejected pre-call."""
    model = _make_mock_model(_make_sde_tuple(num_traj_samples=2))
    adapter = AlpamayoR1InferenceModel(
        model=model, num_context_frames=_NUM_CONTEXT_FRAMES, last_component="cot"
    )
    sampling = _make_sampling(num_traj_samples=2)

    with pytest.raises(ValueError, match="num_traj_samples=1"):
        adapter.sample_trajectories_from_data(_make_batched_input(), sampling)
    model.sample_trajectories_from_data.assert_not_called()


def test_cot_with_multi_set_rejected_early() -> None:
    """``last_component='cot'`` + ``num_traj_sets>1`` is rejected pre-call."""
    model = _make_mock_model(_make_sde_tuple())
    adapter = AlpamayoR1InferenceModel(
        model=model, num_context_frames=_NUM_CONTEXT_FRAMES, last_component="cot"
    )
    sampling = _make_sampling(num_traj_sets=2)

    with pytest.raises(ValueError, match="num_traj_sets=1"):
        adapter.sample_trajectories_from_data(_make_batched_input(), sampling)
    model.sample_trajectories_from_data.assert_not_called()


def test_non_4_tuple_return_rejected() -> None:
    """A 3-tuple (non-SDE path) is a contract violation; raise loudly."""
    model = MagicMock()
    model.sample_trajectories_from_data = MagicMock(
        return_value=(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    )
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)

    with pytest.raises(ValueError, match="expected 4"):
        adapter.sample_trajectories_from_data(_make_batched_input(), _make_sampling())


def test_retry_on_nonfinite_reruns_prefill_until_finite(tokenize_mock: MagicMock) -> None:
    """A non-finite trajectory triggers a re-tokenize + re-sample that lands finite."""
    nan_xyz, pred_rot, logprob, sde_info = _make_sde_tuple()
    nan_tuple = (torch.full_like(nan_xyz, float("nan")), pred_rot, logprob, sde_info)
    model = _make_mock_model(_make_sde_tuple())
    model.sample_trajectories_from_data = MagicMock(side_effect=[nan_tuple, _make_sde_tuple()])
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)

    output = adapter.sample_trajectories_from_data(_make_batched_input(), _make_sampling())

    assert torch.isfinite(output.pred_xyz).all()
    assert model.sample_trajectories_from_data.call_count == 2
    # Each re-sample re-runs the VLM prefill: initial tokenize + one retry tokenize.
    assert tokenize_mock.call_count == 2


def test_retry_on_nonfinite_disabled_propagates_nonfinite() -> None:
    """With the retry off, a non-finite trajectory is returned as-is (no re-sample)."""
    nan_xyz, pred_rot, logprob, sde_info = _make_sde_tuple()
    nan_tuple = (torch.full_like(nan_xyz, float("nan")), pred_rot, logprob, sde_info)
    model = _make_mock_model(nan_tuple)
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)

    output = adapter.sample_trajectories_from_data(
        _make_batched_input(), _make_sampling(retry_on_nonfinite=False)
    )

    assert not torch.isfinite(output.pred_xyz).all()
    assert model.sample_trajectories_from_data.call_count == 1


def test_retry_on_nonfinite_raises_when_retries_exhausted() -> None:
    """A trajectory that stays non-finite through every retry fails loudly."""
    nan_xyz, pred_rot, logprob, sde_info = _make_sde_tuple()
    nan_tuple = (torch.full_like(nan_xyz, float("nan")), pred_rot, logprob, sde_info)
    model = _make_mock_model(nan_tuple)
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)

    with pytest.raises(ValueError, match="non-finite trajectory after 3 retries"):
        adapter.sample_trajectories_from_data(_make_batched_input(), _make_sampling())

    # Initial sample plus three retries.
    assert model.sample_trajectories_from_data.call_count == 4


def test_pack_data_includes_route_xy_with_n_traj_group_axis(tokenize_mock: MagicMock) -> None:
    """``route_xy`` gets the n_traj_group=1 axis the model expects."""
    model = _make_mock_model(_make_sde_tuple())
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)
    adapter.sample_trajectories_from_data(_make_batched_input(), _make_sampling())

    data_arg = tokenize_mock.call_args.args[1]
    assert "route_xy" in data_arg
    assert data_arg["route_xy"].shape == (_B, 1, 4, 2)


def test_transform_inputs_normalizes_uint8_to_float32_neg1_to_1(tokenize_mock: MagicMock) -> None:
    """Adapter feeds float32 [-1, 1] frames to the sampler even if the helper rescales in place."""

    def helper_side_effect(model, data, last_component=None):
        data["image_frames"].copy_((data["image_frames"] + 1.0) / 2.0)
        return {"input_ids": torch.zeros((1, 64), dtype=torch.int64)}

    tokenize_mock.side_effect = helper_side_effect
    model = _make_mock_model(_make_sde_tuple())
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)
    adapter.sample_trajectories_from_data(_make_batched_input(), _make_sampling())

    sampler_data = model.sample_trajectories_from_data.call_args.args[0]
    image_frames = sampler_data["image_frames"]
    assert image_frames.dtype == torch.float32
    assert image_frames.shape == (_B, _NUM_CAMERAS * _NUM_CONTEXT_FRAMES, 1, 3, _H, _W)
    assert torch.allclose(image_frames, torch.full_like(image_frames, 127 / 127.5 - 1.0), atol=1e-6)


def test_pack_data_rejects_non_uint8_camera_frames() -> None:
    """``BatchedModelInput`` contract is raw uint8; adapter refuses anything else."""
    batched = _make_batched_input()
    batched = type(batched)(
        ego_history_xyz=batched.ego_history_xyz,
        ego_history_rot=batched.ego_history_rot,
        camera_frames=torch.zeros_like(batched.camera_frames, dtype=torch.float32),
        camera_indices=batched.camera_indices,
        relative_timestamps=batched.relative_timestamps,
        route_xy=batched.route_xy,
    )
    adapter = AlpamayoR1InferenceModel(
        model=_make_mock_model(_make_sde_tuple()), num_context_frames=_NUM_CONTEXT_FRAMES
    )
    with pytest.raises(ValueError, match="camera_frames must be uint8"):
        adapter.sample_trajectories_from_data(batched, _make_sampling())


def test_sample_trajectories_accepts_batch_size_greater_than_one() -> None:
    """The adapter dispatches a real batched model call when determinism is off."""
    batch_size = 2
    model = _make_mock_model(_make_sde_tuple(batch_size=batch_size))
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)

    output = adapter.sample_trajectories_from_data(
        _make_batched_input(batch_size=batch_size),
        _make_sampling(),
        return_trace_for_rl=True,
    )

    assert model.sample_trajectories_from_data.call_count == 1
    data_arg = model.sample_trajectories_from_data.call_args.args[0]
    assert data_arg["image_frames"].shape[0] == batch_size
    assert output.pred_xyz.shape[0] == batch_size
    assert len(output.unbind()) == batch_size


def test_pack_data_adds_n_traj_group_axis_to_history(tokenize_mock: MagicMock) -> None:
    """``ego_history_xyz`` / ``ego_history_rot`` get a dim=1 axis for n_traj_group."""
    model = _make_mock_model(_make_sde_tuple())
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)
    adapter.sample_trajectories_from_data(_make_batched_input(), _make_sampling())

    data_arg = tokenize_mock.call_args.args[1]
    assert data_arg["ego_history_xyz"].shape == (_B, 1, _NUM_HISTORY, 3)
    assert data_arg["ego_history_rot"].shape == (_B, 1, _NUM_HISTORY, 3, 3)


def test_pack_data_passes_through_policy_4d_history_layout(tokenize_mock: MagicMock) -> None:
    """When the policy feeds the n_traj_group=1 axis already, don't double-unsqueeze.

    ``AlpamayoPolicy._extract_historical_motion`` inserts the n_traj_group axis
    before stacking, so the batched input arrives with ego_history_xyz as 4-D
    and ego_history_rot as 5-D. The adapter must not add another axis on top.
    """
    model = _make_mock_model(_make_sde_tuple())
    adapter = AlpamayoR1InferenceModel(model=model, num_context_frames=_NUM_CONTEXT_FRAMES)
    inputs = _make_batched_input()
    # Reshape per the AlpamayoPolicy layout: per-sample (1, T, 3) and (1, T, 3, 3)
    # so the stacked batch is (B, 1, T, 3) / (B, 1, T, 3, 3).
    policy_xyz = inputs.ego_history_xyz.unsqueeze(1)
    policy_rot = inputs.ego_history_rot
    if policy_rot.ndim == 4:
        policy_rot = policy_rot.unsqueeze(1)
    policy_inputs = BatchedModelInput(
        ego_history_xyz=policy_xyz,
        ego_history_rot=policy_rot,
        camera_frames=inputs.camera_frames,
        camera_indices=inputs.camera_indices,
        relative_timestamps=inputs.relative_timestamps,
        route_xy=inputs.route_xy,
    )

    adapter.sample_trajectories_from_data(policy_inputs, _make_sampling())

    data_arg = tokenize_mock.call_args.args[1]
    assert data_arg["ego_history_xyz"].shape == (_B, 1, _NUM_HISTORY, 3)
    assert data_arg["ego_history_rot"].shape == (_B, 1, _NUM_HISTORY, 3, 3)
