# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AlpaGym runtime performance instrumentation.

Two subpackages separate runtime hooks from offline tooling:

- `instrument/` — the runtime hot-path API: `initialize_perf`/`shutdown_perf`
  wrap each worker, `@measure_perf` / `timed_scope` time a scope, and
  `record_perf_marker` records a CPU/GPU snapshot at a named instant.
- `analysis/` — the offline `cli` that summarizes the JSON artifacts.

Import from the submodules directly. See `README.md` in this directory for the
design and JSON contract.
"""
