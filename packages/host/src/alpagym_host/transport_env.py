# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Env-var writer the host uses to hand NCCL settings to cosmos workers.

Lives in its own module so tests can import the helper without pulling in
``alpagym_host.cli`` (which transitively requires ``tomli_w`` via
``run_artifacts``). ``run_lifecycle.execute_run`` calls it just before launching
the cosmos subprocess, so the NCCL env propagates to the cosmos workers via
process inheritance (locally through ``subprocess.run``; on Slurm through the
cosmos ``srun --export=ALL``).

``apply_transport_env_vars`` mutates the host process's ``os.environ`` on
purpose: that inheritance is how the NCCL env reaches the cosmos workers. It
runs after the AlpaSim wizard processes have already started, so the wizard
keeps the env it launched with -- AlpaSim has no NCCL relationship and needs
none of these vars.

The resolved config path travels separately, in the cosmos launcher TOML's
``[custom].resolved_config_path`` (passed as ``--config`` and read by the worker
entrypoint), so it is not duplicated here.
"""

import os

from alpagym_host.config import TransportConfig, TransportKind


def apply_transport_env_vars(transport_config: TransportConfig) -> None:
    """Write NCCL env vars to ``os.environ`` for the cosmos subprocess to inherit.

    Disk runs leave ``nccl_env`` empty, so this is a no-op for them. NCCL runs
    launch with the exact NCCL env declared in ``transport.nccl_env``; that dict
    is the single source of truth for the NCCL keys this run sets.
    """
    if transport_config.kind != TransportKind.nccl:
        return
    # Drop any inherited NCCL_*/TORCH_NCCL_* first so ``transport.nccl_env`` is
    # the authoritative NCCL surface for the workers this host launches.
    for name in list(os.environ):
        if name.startswith("NCCL_") or name.startswith("TORCH_NCCL_"):
            os.environ.pop(name)
    for name, value in transport_config.nccl_env.items():
        os.environ[name] = str(value)
