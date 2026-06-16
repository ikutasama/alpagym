# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the one-shot NCCL communicator bootstrap."""

import logging
import socket

import pytest
from alpagym_runtime.transport.nccl.comm_init import (
    CommInitConfig,
    assign_rank_once,
    init_communicator,
)
from torch.distributed import TCPStore


def _free_port() -> int:
    """Pick an OS-assigned ephemeral port for the test's TCPStore."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def store() -> TCPStore:
    """Create a fresh master TCPStore on an ephemeral port for this test."""
    return TCPStore(
        host_name="127.0.0.1",
        port=_free_port(),
        world_size=1,
        is_master=True,
        wait_for_workers=False,
    )


def _logger() -> logging.Logger:
    return logging.getLogger("test_comm_init")


def test_assign_rank_once_for_policy_role(store: TCPStore) -> None:
    """The first Policy rank assignment yields rank 0."""
    rank = assign_rank_once(
        role="Policy",
        prefix="alpamayo:test:job",
        world_size=8,
        num_policy_processes=4,
        store=store,
        logger=_logger(),
    )
    assert rank == 0


def test_assign_rank_once_for_rollout_role_uses_policy_offset(store: TCPStore) -> None:
    """Rollout ranks start right after the last policy rank."""
    rank = assign_rank_once(
        role="Rollout",
        prefix="alpamayo:test:job",
        world_size=8,
        num_policy_processes=4,
        store=store,
        logger=_logger(),
    )
    assert rank == 4


def test_assign_rank_once_raises_for_unknown_role(store: TCPStore) -> None:
    """An unsupported role string fails fast."""
    with pytest.raises(RuntimeError, match="Unknown role"):
        assign_rank_once(
            role="Trainer",
            prefix="alpamayo:test:job",
            world_size=8,
            num_policy_processes=4,
            store=store,
            logger=_logger(),
        )


def test_assign_rank_once_raises_when_policy_rank_exceeds_policy_processes(
    store: TCPStore,
) -> None:
    """Rank-counter overflow is reported immediately."""
    prefix = "alpamayo:test:job"
    assign_rank_once(
        role="Policy",
        prefix=prefix,
        world_size=1,
        num_policy_processes=1,
        store=store,
        logger=_logger(),
    )
    with pytest.raises(RuntimeError, match="exceeds policy process count"):
        assign_rank_once(
            role="Policy",
            prefix=prefix,
            world_size=1,
            num_policy_processes=1,
            store=store,
            logger=_logger(),
        )
    assert int(store.add(f"{prefix}:policy_rank_counter", 0)) == 1


def test_assign_rank_once_raises_when_rollout_rank_exceeds_world_size(
    store: TCPStore,
) -> None:
    """Rollout rank overflow is checked against the combined world size."""
    prefix = "alpamayo:test:job"
    with pytest.raises(RuntimeError, match="exceeds world_size"):
        assign_rank_once(
            role="Rollout",
            prefix=prefix,
            world_size=1,
            num_policy_processes=1,
            store=store,
            logger=_logger(),
        )
    assert int(store.add(f"{prefix}:rollout_rank_counter", 0)) == 0


def test_init_communicator_raises_when_rank_assignment_fails(store: TCPStore) -> None:
    """init_communicator surfaces rank-assignment failures without retry."""
    # Pre-bump the counter so the only allowed slot is already burned.
    prefix = "alpamayo:test:job"
    store.add(f"{prefix}:policy_rank_counter", 1)
    with pytest.raises(RuntimeError, match="exceeds policy process count"):
        init_communicator(
            role="Policy",
            prefix=prefix,
            world_size=1,
            num_policy_processes=1,
            store=store,
            create_nccl_uid=lambda: [1, 2, 3],
            create_nccl_comm=lambda *args, **kwargs: 17,
            logger=_logger(),
        )


def test_init_communicator_times_out_when_uid_never_appears(store: TCPStore) -> None:
    """A non-zero rank that can't read the UID raises a clear timeout."""
    prefix = "alpamayo:test:job"
    # Pre-bump the counter so the next Policy assignment lands at rank 1,
    # forcing the function into UID-wait mode.
    store.add(f"{prefix}:policy_rank_counter", 1)
    config = CommInitConfig(barrier_wait_timeout_seconds=0.2)
    with pytest.raises(RuntimeError, match="Timed out waiting for NCCL UID"):
        init_communicator(
            role="Policy",
            prefix=prefix,
            world_size=2,
            num_policy_processes=2,
            store=store,
            create_nccl_uid=lambda: [1, 2, 3],
            create_nccl_comm=lambda *args, **kwargs: 17,
            logger=_logger(),
            config=config,
        )


def test_init_communicator_completes_when_world_ready(store: TCPStore) -> None:
    """With the UID and the peer's ready marker present, init returns cleanly."""
    prefix = "alpamayo:test:job"
    # Pre-seed the UID and the peer's ready marker.
    store.set(f"{prefix}:nccl_uid", "1,2,3")
    store.set(f"{prefix}:nccl_ready:1", "2")
    config = CommInitConfig(barrier_wait_timeout_seconds=1.0)
    rank, comm_idx = init_communicator(
        role="Policy",
        prefix=prefix,
        world_size=2,
        num_policy_processes=2,
        store=store,
        create_nccl_uid=lambda: [1, 2, 3],
        create_nccl_comm=lambda *args, **kwargs: 42,
        logger=_logger(),
        config=config,
    )
    assert rank == 0
    assert comm_idx == 42


def test_init_communicator_rejects_duplicate_ready_writer(store: TCPStore) -> None:
    """A second writer for the same ready rank fails before the barrier."""
    prefix = "alpamayo:test:job"
    store.set(f"{prefix}:nccl_uid", "1,2,3")
    store.set(f"{prefix}:nccl_ready:1", "2")
    store.add(f"{prefix}:nccl_ready:0:writers", 1)
    config = CommInitConfig(barrier_wait_timeout_seconds=1.0)
    with pytest.raises(RuntimeError, match="Ready marker for rank 0 was already written"):
        init_communicator(
            role="Policy",
            prefix=prefix,
            world_size=2,
            num_policy_processes=2,
            store=store,
            create_nccl_uid=lambda: [1, 2, 3],
            create_nccl_comm=lambda *args, **kwargs: 42,
            logger=_logger(),
            config=config,
        )
    assert int(store.add(f"{prefix}:nccl_ready:0:writers", 0)) == 1
