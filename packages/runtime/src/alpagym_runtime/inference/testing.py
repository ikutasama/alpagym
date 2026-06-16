# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test helpers that drive `InferenceEngine` synchronously for one model input."""

import threading
from contextlib import contextmanager
from typing import Iterator

from alpagym_host.config import SamplingParamsConfig

from alpagym_runtime.inference.inference_engine import InferenceEngine
from alpagym_runtime.inference.types import InferenceModel


@contextmanager
def driven_inference_engine(
    inference_model: InferenceModel,
    sampling: SamplingParamsConfig,
    return_trace_for_rl: bool,
) -> Iterator[InferenceEngine]:
    """Yield an `InferenceEngine` running in a background thread."""
    inference_engine = InferenceEngine(
        inference_model=inference_model,
        sampling=sampling,
        return_trace_for_rl=return_trace_for_rl,
        max_batch_size=1,
    )
    thread = threading.Thread(target=inference_engine.run_loop, daemon=True)
    thread.start()
    try:
        yield inference_engine
    finally:
        inference_engine.shutdown()
        thread.join(timeout=5.0)
