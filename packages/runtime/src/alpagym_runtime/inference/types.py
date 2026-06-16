# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed I/O between `AlpamayoPolicy`, inference engine, and inference models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, cast, runtime_checkable

import numpy as np
import torch
from alpagym_host.config import SamplingParamsConfig

from alpagym_runtime.replay import ActionSelection, PolicyReplayData


@dataclass(frozen=True)
class _ModelInputFields:
    """Shared field set for `ModelInput` and `BatchedModelInput`.

    Private base used purely to dedupe the field list. Not part of the public
    API; never instantiate directly.
    """

    ego_history_xyz: torch.Tensor
    ego_history_rot: torch.Tensor
    # Raw uint8 camera frames as written by ``AlpamayoPolicy``. Per-model
    # normalization (e.g. float32 [-1, 1] for Alpamayo R1) is the inference
    # adapter's responsibility so this dataclass stays model-agnostic.
    camera_frames: torch.Tensor
    camera_indices: torch.Tensor
    relative_timestamps: torch.Tensor
    route_xy: torch.Tensor
    seed: torch.Tensor | None = None


@dataclass(frozen=True)
class ModelInput(_ModelInputFields):
    """A single model input (no batch axis)."""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ModelInput:
        """Rehydrate a `ModelInput` from a persisted replay payload mapping.

        Reads the six tensor fields the replay payload carries; ``seed`` is
        rollout-only and stays ``None`` on the trainer-side rehydration.
        """
        return cls(
            ego_history_xyz=torch.as_tensor(payload["ego_history_xyz"], dtype=torch.float32),
            ego_history_rot=torch.as_tensor(payload["ego_history_rot"], dtype=torch.float32),
            camera_frames=torch.as_tensor(payload["camera_frames"], dtype=torch.uint8),
            camera_indices=torch.as_tensor(payload["camera_indices"], dtype=torch.int64),
            relative_timestamps=torch.as_tensor(payload["relative_timestamps"], dtype=torch.int64),
            route_xy=torch.as_tensor(payload["route_xy"], dtype=torch.float32),
        )


@dataclass(frozen=True)
class BatchedModelInput(_ModelInputFields):
    """A batch of model inputs (leading batch axis on every field)."""

    @classmethod
    def stack(cls, model_inputs: list[ModelInput]) -> BatchedModelInput:
        """Collate `ModelInput` objects into a batched input."""
        if not model_inputs:
            raise ValueError("BatchedModelInput.stack requires at least one ModelInput")
        return cls(
            ego_history_xyz=torch.stack(
                [model_input.ego_history_xyz for model_input in model_inputs], dim=0
            ),
            ego_history_rot=torch.stack(
                [model_input.ego_history_rot for model_input in model_inputs], dim=0
            ),
            camera_frames=torch.stack(
                [model_input.camera_frames for model_input in model_inputs], dim=0
            ),
            camera_indices=torch.stack(
                [model_input.camera_indices for model_input in model_inputs], dim=0
            ),
            relative_timestamps=torch.stack(
                [model_input.relative_timestamps for model_input in model_inputs], dim=0
            ),
            route_xy=torch.stack([model_input.route_xy for model_input in model_inputs], dim=0),
            seed=_stack_optional_tensors([model_input.seed for model_input in model_inputs]),
        )


@dataclass(frozen=True)
class _ModelOutputFields:
    """Shared field set for `ModelOutput` and `BatchedModelOutput`.

    Private base used purely to dedupe the field list. Not part of the public
    API; never instantiate directly.
    """

    pred_xyz: torch.Tensor
    pred_rot: torch.Tensor
    logprob: torch.Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelOutput(_ModelOutputFields):
    """Model output for one inference call (no batch axis)."""


@dataclass(frozen=True)
class BatchedModelOutput(_ModelOutputFields):
    """A batch of model outputs (leading batch axis on every tensor field)."""

    def unbind(self) -> list[ModelOutput]:
        """Slice this batched output along axis 0 into per-call `ModelOutput` objects.

        The dataclass-level analog of `torch.unbind` on dim 0: removes the
        leading batch axis and returns one `ModelOutput` per index.
        """
        batch_size = self.pred_xyz.shape[0]
        if self.pred_rot.shape[0] != batch_size:
            raise ValueError(
                "BatchedModelOutput.unbind: pred_rot leading dim "
                f"{self.pred_rot.shape[0]} != pred_xyz leading dim {batch_size}"
            )
        if self.logprob is not None and self.logprob.shape[0] != batch_size:
            raise ValueError(
                "BatchedModelOutput.unbind: logprob leading dim "
                f"{self.logprob.shape[0]} != pred_xyz leading dim {batch_size}"
            )
        model_outputs: list[ModelOutput] = []
        for i in range(batch_size):
            model_outputs.append(
                ModelOutput(
                    pred_xyz=self.pred_xyz[i],
                    pred_rot=self.pred_rot[i],
                    logprob=self.logprob[i] if self.logprob is not None else None,
                    extra=_split_extra_per_model_output(self.extra, i, batch_size),
                )
            )
        return model_outputs


@runtime_checkable
class InferenceModel(Protocol):
    """Per-bundle adapter that owns the released-model dialect."""

    def sample_trajectories_from_data(
        self,
        model_input: BatchedModelInput,
        sampling: SamplingParamsConfig,
        return_trace_for_rl: bool = False,
    ) -> BatchedModelOutput:
        """Run one batched forward and return a normalized `BatchedModelOutput`."""

    def build_policy_replay_data(
        self,
        model_input: ModelInput,
        model_output: ModelOutput,
        action_selection: ActionSelection,
    ) -> PolicyReplayData:
        """Build model-family-specific replay data for the selected action."""

    def get_model(self) -> torch.nn.Module:
        """Return the model object that serves rollout inference."""

    def set_model(self, model: torch.nn.Module) -> None:
        """Replace the model object used by rollout inference."""


NUM_ROUTE_WAYPOINTS = 20
"""Trainer-input contract: every `ModelInput.route_xy` is a `[NUM_ROUTE_WAYPOINTS, 2]`
float32 tensor. Trailing rows are NaN when the live route is shorter."""


def _stack_optional_tensors(values: list[torch.Tensor | None]) -> torch.Tensor | None:
    """Stack optional tensors along a new leading axis."""
    if all(value is None for value in values):
        return None
    return torch.stack(cast(list[torch.Tensor], values), dim=0)


def drop_single_batch_axis(value: Any) -> Any:
    """Strip the synthetic leading B=1 axis added to reuse a batched packer.

    Trainer-side packers call a family forward-input builder on
    ``BatchedModelInput.stack([single])`` to share the rollout seam, then
    undo the size-1 batch axis on every (possibly nested) tensor leaf.

    Raises:
        ValueError: a tensor leaf does not have leading dim 1.
    """
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or value.shape[0] != 1:
            raise ValueError(
                f"Expected forward input with leading batch 1, got {tuple(value.shape)}"
            )
        return value[0]
    if isinstance(value, dict):
        return {key: drop_single_batch_axis(item) for key, item in value.items()}
    if value is None:
        return None
    return value


def _split_extra_per_model_output(
    extra: dict[str, Any], index: int, batch_size: int
) -> dict[str, Any]:
    """Slice each `extra` leaf along its leading batch axis; every leaf must be per-output."""
    out: dict[str, Any] = {}
    for key, value in extra.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(
                    f"extra[{key!r}]: leading dim must equal batch_size={batch_size}; "
                    f"got shape {tuple(value.shape)}"
                )
            out[key] = value[index]
        elif isinstance(value, (list, tuple)):
            if len(value) != batch_size:
                raise ValueError(
                    f"extra[{key!r}]: len must equal batch_size={batch_size}; got {len(value)}"
                )
            out[key] = value[index]
        else:
            raise TypeError(
                f"extra[{key!r}]: unsupported leaf type {type(value).__name__}; "
                "use torch.Tensor, np.ndarray, list, or tuple."
            )
    return out
