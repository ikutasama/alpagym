# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the rollout's inference engine and per-session policy factory."""

from typing import Callable

import torch
from alpagym_host.config import RunConfig

from alpagym_runtime.inference.inference_engine import InferenceEngine
from alpagym_runtime.policies.alpamayo.policy import AlpamayoPolicy
from alpagym_runtime.policies.determinism import set_deterministic
from alpagym_runtime.policies.registry import get_policy_bundle
from alpagym_runtime.types import Policy, RolloutCalibration

_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def build_inference_engine(run_config: RunConfig) -> InferenceEngine:
    """Build the rollout's inference engine for the configured policy family.

    Reads `run_config.policy.model` and `.inference`, then dispatches loading to
    the installed `PolicyBundle` for `policy.model.kind`.
    """
    policy_cfg = run_config.policy
    model_cfg = policy_cfg.model
    inference_cfg = policy_cfg.inference
    device = torch.device(model_cfg.device)
    dtype = _DTYPE_BY_NAME[model_cfg.dtype]
    if inference_cfg.sampling.force_determinism:
        set_deterministic()

    inference_model = get_policy_bundle(model_cfg.kind).load_inference_model(
        run_config, device, dtype
    )

    return InferenceEngine(
        inference_model=inference_model,
        sampling=inference_cfg.sampling,
        return_trace_for_rl=inference_cfg.return_trace_for_rl,
        max_batch_size=inference_cfg.max_batch_size,
    )


def build_policy_factory(
    run_config: RunConfig,
    inference_engine: InferenceEngine,
) -> Callable[[str, RolloutCalibration, int], Policy]:
    """Build the per-session policy factory bound to `inference_engine`.

    The returned closure is invoked by the egodriver servicer at
    `start_session` with `(session_uuid, calibration, random_seed)` to
    construct one `AlpamayoPolicy` per AlpaSim session.
    """
    policy_cfg = run_config.policy
    model_cfg = policy_cfg.model
    device = torch.device(model_cfg.device)
    dtype = _DTYPE_BY_NAME[model_cfg.dtype]

    def policy_factory(
        session_uuid: str,
        calibration: RolloutCalibration,
        random_seed: int,
    ) -> Policy:
        """Build one `AlpamayoPolicy` bound to the shared inference engine."""
        del calibration
        return AlpamayoPolicy(
            inference_engine=inference_engine,
            session_uuid=session_uuid,
            config=policy_cfg,
            device=device,
            dtype=dtype,
            seed=random_seed,
        )

    return policy_factory
