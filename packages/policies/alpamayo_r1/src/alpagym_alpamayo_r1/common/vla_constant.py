# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Constants for the VLA models

from alpamayo_r1.models.base_model import SPECIAL_TOKENS_KEYS as _PUBLIC_SPECIAL_TOKENS_KEYS

IGNORE_LABEL_ID = -100

# Δt marker format for the streaming-aware text-token signal. Plain text
# ``<dt NNN ms>`` with 3-digit zero-padded ms value — clamped to [0, 999] so
# Qwen3-VL's tokenizer always produces exactly 8 tokens (``<``, ``dt``, `` ``,
# d1, d2, d3, `` ms``, ``>``). No special tokens / no embedding resize: we
# reuse Qwen3-VL's digit+"ms" priors directly. Every tick uses the same
# ``<dt NNN ms>`` form, including the first tick — so training and steady-state
# streaming inference share a single 8-token-per-tick alphabet.
# Max dt we represent exactly; beyond this we clamp (catastrophic gaps are
# coarser-grained anyway).
DT_CLAMP_MS_MAX = 999


def dt_marker(dt_microseconds: int) -> str:
    """Format a Δt (int µs) as a fixed-width plain-text marker.

    Input is int64 µs (matches the upstream ``absolute_timestamps`` schema —
    subtraction stays exact). Clamps to ``[0, DT_CLAMP_MS_MAX]`` ms so
    Qwen3-VL's tokenizer output stays at a constant 8 tokens (streaming
    eviction friendly).
    """
    # µs → ms with integer rounding (half-up via +500 bias, safe for non-neg).
    dt_ms = max(0, int(dt_microseconds) + 500) // 1000
    dt_ms = min(dt_ms, DT_CLAMP_MS_MAX)
    return f"<dt {dt_ms:03d} ms>"


# Upstream Alpamayo-R1 special tokens plus the two bounding-box keys AlpaGym
# appends. ``box_start`` / ``box_end`` already exist in the Qwen3VL tokenizer, so
# appending them does not change the tokenizer size. The box keys go last so the
# special-token order (and thus token IDs) stays aligned with upstream + the
# checkpoint.
SPECIAL_TOKENS_KEYS = [*_PUBLIC_SPECIAL_TOKENS_KEYS, "box_start", "box_end"]
SPECIAL_TOKENS = {k: "<|" + k + "|>" for k in SPECIAL_TOKENS_KEYS}
