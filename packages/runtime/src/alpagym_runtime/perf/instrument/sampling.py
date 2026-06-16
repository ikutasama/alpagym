# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reservoir sampling and percentile helpers shared by every store."""

from __future__ import annotations

import random


class ReservoirSampler:
    """Vitter Algorithm R reservoir sampler over a single numeric series.

    Gives an unbiased uniform sample of size at most `capacity` from the
    sequence of values added via `add`. Used by timing and resource stores
    to bound memory while still producing meaningful percentiles over the
    whole run.
    """

    def __init__(self, capacity: int, rng: random.Random | None = None) -> None:
        """Build an empty reservoir with the given bound.

        Args:
            capacity: Maximum samples retained in the reservoir. Must be > 0.
            rng: Optional RNG override (tests pass a seeded `random.Random`).

        Raises:
            ValueError: If `capacity` is not positive.
        """
        if capacity <= 0:
            raise ValueError(f"ReservoirSampler capacity must be > 0, got {capacity}")
        self._capacity = capacity
        self._n = 0
        self._sum = 0.0
        self._buf: list[float] = []
        self._rng = rng if rng is not None else random.Random()

    def add(self, value: float) -> None:
        """Offer one sample to the reservoir using Algorithm R."""
        self._n += 1
        self._sum += value
        if len(self._buf) < self._capacity:
            self._buf.append(value)
            return
        j = self._rng.randrange(self._n)
        if j < self._capacity:
            self._buf[j] = value

    def samples(self) -> list[float]:
        """Return a snapshot copy of the current reservoir contents."""
        return list(self._buf)

    @property
    def total_seen(self) -> int:
        """Return the total number of values offered to this reservoir."""
        return self._n

    @property
    def mean(self) -> float:
        """Return the exact mean over every value offered, not the retained sample.

        Tracked from a running sum so the mean stays correct once the reservoir
        caps; the bounded `samples()` buffer is only for percentiles.
        """
        return self._sum / self._n if self._n else 0.0


def percentile(samples: list[float], q: float) -> float:
    """Return the `q`-th percentile of `samples` (0–100) using linear interpolation.

    Empty input returns 0.0. The implementation uses NumPy-style linear
    interpolation between order statistics so small reservoirs still produce a
    meaningful number.
    """
    if not samples:
        return 0.0
    if q < 0.0 or q > 100.0:
        raise ValueError(f"percentile q must be in [0, 100], got {q}")
    sorted_samples = sorted(samples)
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    rank = (len(sorted_samples) - 1) * (q / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_samples) - 1)
    frac = rank - lower
    return float(sorted_samples[lower] * (1.0 - frac) + sorted_samples[upper] * frac)
