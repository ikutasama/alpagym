# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from alpagym_runtime.transport.disk import read_episode_json

_SMOKE_SCENE_ID = "clipgt-01d503d4-449b-46fc-8d78-9085e70d3554"


def _cosmos_smoke_available() -> bool:
    """Return whether the local environment can launch Cosmos-RL workers."""
    try:
        cosmos_spec = importlib.util.find_spec("cosmos_rl")
    except ValueError:
        return False
    if cosmos_spec is None:
        return False
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


def _free_tcp_port() -> int:
    """Return an available local TCP port for one smoke launch."""
    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


@pytest.mark.alpasim_e2e
@pytest.mark.skipif(
    not _cosmos_smoke_available(),
    reason="cosmos_rl with CUDA is not available",
)
def test_host_launches_cosmos_smoke(tmp_path: Path) -> None:
    """Launches Cosmos-RL against the real AlpaSim-backed host entrypoint."""
    controller_port = _free_tcp_port()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alpagym_host.cli",
            f"run_root={tmp_path}",
            f"dataset.scene_ids=[{_SMOKE_SCENE_ID}]",
            "policy.model.num_future_waypoints=2",
            f"cosmos.launch.controller_port={controller_port}",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    run_dirs = sorted(tmp_path.iterdir())

    assert result.returncode == 0
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "cosmos_config.toml").is_file()
    artifact_paths = sorted((run_dirs[0] / "artifacts").glob(f"{_SMOKE_SCENE_ID}_*.json"))
    assert len(artifact_paths) == 1
    artifact = read_episode_json(artifact_paths[0])
    assert artifact.scene_id == _SMOKE_SCENE_ID
    assert artifact.num_steps > 0
