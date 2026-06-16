# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the NCCL consumer-side receiver (TCPStore-backed)."""

import socket
import time
from typing import Any

import pytest
import torch
from alpagym_runtime.transport.nccl.protocol import (
    NCCL_RENDEZVOUS_ACCEPTED,
    NcclTransferAck,
    build_nccl_prefix,
    build_rollout_prefix,
)
from alpagym_runtime.transport.nccl.receiver import NcclReceiver, NcclReceiverConfig
from torch.distributed import TCPStore

_PREFIX = build_nccl_prefix(experiment_name="exp", job_id="job")


def _free_port() -> int:
    """Pick an OS-assigned ephemeral port for the test's TCPStore."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def store() -> TCPStore:
    """Create a fresh master TCPStore on an ephemeral port."""
    return TCPStore(
        host_name="127.0.0.1",
        port=_free_port(),
        world_size=1,
        is_master=True,
        wait_for_workers=False,
    )


class _AlwaysAcceptingRendezvous:
    """Rendezvous double whose request_ack returns ``accepted`` instantly."""

    def request_ack(self, **kwargs: Any) -> NcclTransferAck:
        return NcclTransferAck(
            request_id="r1",
            transfer_id=kwargs["transfer_id"],
            status=NCCL_RENDEZVOUS_ACCEPTED,
            message="ok",
            pending_count=1,
            accepted_at_ms=1,
        )

    def process_request(self, **kwargs: Any) -> Any:
        raise NotImplementedError


def _build_receiver(
    store: TCPStore,
    nccl_recv: Any = None,
    nccl_abort: Any = None,
    rendezvous: Any = None,
    num_rollout_replicas: int = 1,
    num_policy_replicas: int = 1,
    config: NcclReceiverConfig | None = None,
) -> NcclReceiver:
    """Construct a receiver with no-op cosmos-rl callables and a tiny world."""
    return NcclReceiver(
        experiment_name="exp",
        job_id="job",
        num_policy_replicas=num_policy_replicas,
        dp_shard_size=1,
        num_rollout_replicas=num_rollout_replicas,
        store=store,
        rendezvous=rendezvous or _AlwaysAcceptingRendezvous(),
        create_nccl_uid=lambda: [1, 2, 3],
        create_nccl_comm=lambda *args, **kwargs: 17,
        nccl_recv=nccl_recv or (lambda *args, **kwargs: None),
        nccl_abort=nccl_abort or (lambda comm_idx: None),
        config=config,
    )


def test_constructor_rejects_nonpositive_dp_shard_size(store: TCPStore) -> None:
    """dp_shard_size <= 0 fails fast at construction time."""
    with pytest.raises(ValueError, match="dp_shard_size must be positive"):
        NcclReceiver(
            experiment_name="exp",
            job_id="job",
            num_policy_replicas=1,
            dp_shard_size=0,
            num_rollout_replicas=1,
            store=store,
            rendezvous=_AlwaysAcceptingRendezvous(),
            create_nccl_uid=lambda: [1, 2, 3],
            create_nccl_comm=lambda *args, **kwargs: 17,
            nccl_recv=lambda *args, **kwargs: None,
            nccl_abort=lambda comm_idx: None,
        )


def test_constructor_rejects_nonpositive_num_policy_replicas(store: TCPStore) -> None:
    """num_policy_replicas <= 0 fails fast at construction time."""
    with pytest.raises(ValueError, match="num_policy_replicas must be positive"):
        _build_receiver(store=store, num_policy_replicas=0)


def test_recv_raises_before_setup(store: TCPStore) -> None:
    """recv() before setup() fails with a clear NCCL-not-initialized error."""
    receiver = _build_receiver(store=store)
    with pytest.raises(RuntimeError, match="NCCL not initialized"):
        receiver.recv(transfer_id="0:abc", tensor_specs={})


def test_recv_raises_without_resolved_rollout_idx(store: TCPStore) -> None:
    """A transfer_id without a parseable rollout-idx prefix fails fast."""
    receiver = _build_receiver(store=store)
    receiver._rollout_comms = {0: 17}
    receiver._policy_ranks = {0: 0}
    receiver._rollout_prefixes = {0: build_rollout_prefix(_PREFIX, 0)}
    with pytest.raises(ValueError, match="invalid literal for int"):
        receiver.recv(transfer_id="bad-format", tensor_specs={})


def test_recv_raises_when_rollout_comm_missing(store: TCPStore) -> None:
    """Recv on a rollout-idx without a matching comm raises immediately."""
    receiver = _build_receiver(store=store, num_rollout_replicas=2)
    receiver._rollout_comms = {0: 17}
    receiver._policy_ranks = {0: 0}
    receiver._rollout_prefixes = {0: build_rollout_prefix(_PREFIX, 0)}
    with pytest.raises(RuntimeError, match="not initialized for rollout_idx=1"):
        receiver.recv(transfer_id="1:abc", tensor_specs={})


def test_recv_routes_to_right_comm_and_calls_nccl_recv(store: TCPStore) -> None:
    """Recv resolves rollout_idx, picks the matching comm, and runs nccl_recv."""
    nccl_calls: list[tuple[int, int]] = []

    def fake_recv(tensor: torch.Tensor, sender_rank: int, comm_idx: int, **kwargs: Any) -> None:
        nccl_calls.append((sender_rank, comm_idx))
        tensor.fill_(1.0)

    receiver = _build_receiver(store=store, nccl_recv=fake_recv, num_rollout_replicas=2)
    receiver._rollout_comms = {0: 17, 1: 29}
    receiver._policy_ranks = {0: 0, 1: 0}
    receiver._rollout_prefixes = {
        0: build_rollout_prefix(_PREFIX, 0),
        1: build_rollout_prefix(_PREFIX, 1),
    }
    result = receiver.recv(
        transfer_id="1:abc",
        tensor_specs={"k": ((2,), "torch.float32")},
    )
    assert "k" in result
    assert result["k"].shape == (2,)
    assert nccl_calls == [(1, 29)]


def test_close_aborts_all_communicators(store: TCPStore) -> None:
    """close() walks the rollout-comm set and aborts each entry."""
    aborted: list[int] = []
    receiver = _build_receiver(store=store)
    receiver._nccl_abort = aborted.append
    receiver._rollout_comms = {0: 17, 1: 29, 2: 31}
    receiver.close()
    assert sorted(aborted) == [17, 29, 31]


def test_recv_raises_timeout_when_nccl_recv_hangs(store: TCPStore) -> None:
    """A hung nccl_recv must surface as TimeoutError within recv_timeout_seconds.

    This guards the watchdog Event path: without it, a stuck sender would
    park the receiver inside nccl_recv until ``NCCL_TIMEOUT`` kills the
    process — exactly the asymmetric-commitment failure the protocol exists
    to avoid.
    """

    def hang_recv(tensor: torch.Tensor, sender_rank: int, comm_idx: int, **kwargs: Any) -> None:
        time.sleep(60)  # blocks well past the test's recv timeout

    aborted: list[int] = []
    receiver = _build_receiver(
        store=store,
        nccl_recv=hang_recv,
        nccl_abort=aborted.append,
        config=NcclReceiverConfig(recv_timeout_seconds=0.2),
    )
    receiver._rollout_comms = {0: 17}
    receiver._policy_ranks = {0: 0}
    receiver._rollout_prefixes = {0: build_rollout_prefix(_PREFIX, 0)}
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="NCCL recv TIMEOUT"):
        receiver.recv(
            transfer_id="0:abc",
            tensor_specs={"k": ((2,), "torch.float32")},
        )
    # Watchdog must fire close to the configured deadline, not a full minute later.
    assert time.monotonic() - start < 5.0
    assert aborted == [17]
    assert 0 not in receiver._rollout_comms
    assert 0 not in receiver._policy_ranks
    assert 0 not in receiver._rollout_prefixes


def test_recv_error_drops_comm_after_accept(store: TCPStore) -> None:
    """Any post-accept nccl_recv error drops the poisoned rollout comm."""
    aborted: list[int] = []

    def fail_recv(tensor: torch.Tensor, sender_rank: int, comm_idx: int, **kwargs: Any) -> None:
        raise RuntimeError("simulated recv failure")

    receiver = _build_receiver(store=store, nccl_recv=fail_recv, nccl_abort=aborted.append)
    receiver._rollout_comms = {0: 17}
    receiver._policy_ranks = {0: 0}
    receiver._rollout_prefixes = {0: build_rollout_prefix(_PREFIX, 0)}
    with pytest.raises(RuntimeError, match="simulated recv failure"):
        receiver.recv(
            transfer_id="0:abc",
            tensor_specs={"k": ((2,), "torch.float32")},
        )
    assert aborted == [17]
    assert 0 not in receiver._rollout_comms
