# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.alpasim_e2e
def test_host_runs_real_alpasim_rollout(tmp_path: Path) -> None:
    """Starts real AlpaSim Wizard and runs one Cosmos rollout through RuntimeService."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "alpagym_host.cli",
            f"run_root={tmp_path}",
            "cosmos.train.max_num_steps=1",
            "cosmos.rollout.n_generation=1",
            "cosmos.rollout.batch_size=1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    (run_dir,) = sorted(tmp_path.iterdir())

    assert (run_dir / "alpasim" / "generated-runtime-server.yaml").is_file()
    artifacts = sorted((run_dir / "artifacts").glob("*.json"))
    assert len(artifacts) == 1
