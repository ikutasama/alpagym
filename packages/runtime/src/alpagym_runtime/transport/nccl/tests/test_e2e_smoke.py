# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke tests for the NCCL transport with a fake data plane.

A real in-process ``torch.distributed.TCPStore`` backs the rendezvous, and
the NCCL data plane is faked with an in-process queue. The sender thread, the
receiver watchdog, the rendezvous state machine, and the cross-process
TCPStore coordination are all exercised against real components.
"""

import queue
import socket
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from alpagym_runtime.transport.nccl.comm_init import CommInitConfig
from alpagym_runtime.transport.nccl.endpoints import NcclDataPackerMixin, NcclEpisodeWriter
from alpagym_runtime.transport.nccl.protocol import normalize_nccl_handle
from alpagym_runtime.transport.nccl.receiver import NcclReceiver, NcclReceiverConfig
from alpagym_runtime.transport.nccl.rendezvous import AckRendezvous, NcclRendezvousError
from alpagym_runtime.transport.nccl.sender import NcclSender, NcclSenderConfig
from alpagym_runtime.types import EpisodeOutput, PolicyOutput
from torch.distributed import TCPStore


class _Resolver(NcclDataPackerMixin):
    """Trainer-side NCCL read path isolated from replay collation.

    ``NcclDataPackerMixin._resolve_nccl_handle`` uses only ``_store``,
    ``_receiver``, and ``_target_device``; the e2e smoke exercises the full
    resolve / rendezvous / recv / unpack path through it.
    """

    def __init__(self, store: TCPStore, receiver: NcclReceiver) -> None:
        """Wire the resolver to a store and receiver, reading onto CPU."""
        self._store = store
        self._receiver = receiver
        self._target_device = torch.device("cpu")


def _free_port() -> int:
    """Pick an OS-assigned ephemeral port for the test's TCPStore."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def store_endpoint() -> Iterator[tuple[str, int]]:
    """Start a master TCPStore on an ephemeral port and yield its address.

    Each test actor (sender, receiver, transport) must create its own client
    TCPStore connected to this master — TCPStore's master-side reads serialize
    across threads in the same process, so sharing one Python TCPStore object
    across the main thread + the sender's background thread deadlocks.
    """
    host = "127.0.0.1"
    port = _free_port()
    master = TCPStore(
        host_name=host,
        port=port,
        world_size=1,
        is_master=True,
        wait_for_workers=False,
    )
    yield host, port
    del master  # keep the master alive for the duration of the test


class _MockNcclChannel:
    """In-process fake for ``nccl_send`` / ``nccl_recv``."""

    def __init__(self) -> None:
        self._queues: dict[tuple[int, int], queue.Queue[torch.Tensor]] = defaultdict(queue.Queue)

    def send(
        self,
        tensor: torch.Tensor,
        dst_rank: int,
        comm_idx: int,
        stream: Any = None,
    ) -> None:
        """Sender side: enqueue a clone of ``tensor`` for ``(dst_rank, comm_idx)``."""
        del stream
        self._queues[(dst_rank, comm_idx)].put(tensor.detach().cpu().clone())

    def make_recv_for_rank(self, my_rank: int) -> Callable[..., None]:
        """Build a ``nccl_recv`` callable for a receiver running at ``my_rank``."""

        def nccl_recv(
            tensor: torch.Tensor,
            src_rank: int,
            comm_idx: int,
            stream: Any = None,
        ) -> None:
            del src_rank, stream
            received = self._queues[(my_rank, comm_idx)].get(timeout=10.0)
            tensor.copy_(received)

        return nccl_recv


class _MockCommFactory:
    """Builds pynccl ``create_nccl_uid`` / ``create_nccl_comm`` shims."""

    def __init__(self) -> None:
        self._uid_counter = 0
        self._next_comm_idx = 1
        self._comm_idx_by_uid: dict[tuple[int, ...], int] = {}

    def create_nccl_uid(self) -> list[int]:
        self._uid_counter += 3
        return [self._uid_counter, self._uid_counter + 1, self._uid_counter + 2]

    def create_nccl_comm(
        self,
        uid: list[int],
        rank: int,
        world_size: int,
        **kwargs: Any,
    ) -> int:
        del rank, world_size, kwargs
        uid_key = tuple(uid)
        if uid_key not in self._comm_idx_by_uid:
            self._comm_idx_by_uid[uid_key] = self._next_comm_idx
            self._next_comm_idx += 1
        return self._comm_idx_by_uid[uid_key]


_EXPERIMENT_NAME = "e2e_smoke"
_JOB_ID = "test_job"
_FAST_COMM_INIT = CommInitConfig(
    barrier_wait_timeout_seconds=10.0,
    communicator_timeout_ms=10_000,
)
_FAST_ACK_TIMEOUT_S = 5.0
_FAST_SENDER = NcclSenderConfig(
    comm_init=_FAST_COMM_INIT,
    sender_poll_interval_seconds=0.02,
)
_FAST_RECEIVER = NcclReceiverConfig(
    recv_timeout_seconds=10.0,
    comm_init=_FAST_COMM_INIT,
)


def _setup_concurrently(*setups: Callable[[], None], timeout: float = 30.0) -> None:
    """Run multiple setup callables in parallel; raise the first error seen."""
    errors: list[BaseException | None] = [None] * len(setups)

    def _runner(idx: int, fn: Callable[[], None]) -> Callable[[], None]:
        def _do() -> None:
            try:
                fn()
            except BaseException as error:
                errors[idx] = error

        return _do

    threads = [threading.Thread(target=_runner(i, fn), daemon=True) for i, fn in enumerate(setups)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
    for err in errors:
        if err is not None:
            raise err


def _make_client(store_endpoint: tuple[str, int]) -> TCPStore:
    """Open one fresh client TCPStore connected to the test's master."""
    host, port = store_endpoint
    return TCPStore(host_name=host, port=port, world_size=1, is_master=False)


def _build_sender(
    rollout_idx: int,
    store: TCPStore,
    channel: _MockNcclChannel,
    comm_factory: _MockCommFactory,
) -> NcclSender:
    """Construct a sender wired against the shared mock channel + comm factory."""
    return NcclSender(
        experiment_name=_EXPERIMENT_NAME,
        job_id=_JOB_ID,
        rollout_idx=rollout_idx,
        num_policy_replicas=1,
        dp_shard_size=1,
        store=store,
        rendezvous=AckRendezvous(_FAST_ACK_TIMEOUT_S),
        create_nccl_uid=comm_factory.create_nccl_uid,
        create_nccl_comm=comm_factory.create_nccl_comm,
        nccl_send=channel.send,
        nccl_abort=lambda comm_idx: None,
        config=_FAST_SENDER,
    )


def _build_receiver(
    num_rollout_replicas: int,
    store: TCPStore,
    channel: _MockNcclChannel,
    comm_factory: _MockCommFactory,
) -> NcclReceiver:
    """Construct a receiver wired against the shared mock channel + comm factory."""
    return NcclReceiver(
        experiment_name=_EXPERIMENT_NAME,
        job_id=_JOB_ID,
        num_policy_replicas=1,
        dp_shard_size=1,
        num_rollout_replicas=num_rollout_replicas,
        store=store,
        rendezvous=AckRendezvous(_FAST_ACK_TIMEOUT_S),
        create_nccl_uid=comm_factory.create_nccl_uid,
        create_nccl_comm=comm_factory.create_nccl_comm,
        nccl_recv=channel.make_recv_for_rank(0),
        nccl_abort=lambda comm_idx: None,
        config=_FAST_RECEIVER,
    )


def _minimal_episode(
    scene_id: str = "scene_alpha",
    chosen_dt_us: int = 0,
) -> EpisodeOutput:
    """Build a minimal EpisodeOutput with one PolicyOutput; one tensor per field.

    ``chosen_dt_us`` lets callers stamp distinct tensor values into otherwise
    identical episodes so cross-routing tests can detect wrong-comm regressions
    instead of just verifying the manifest round-trip.
    """
    return EpisodeOutput(
        scene_id=scene_id,
        session_uuid="session_zero",
        num_steps=1,
        policy_outputs=(
            PolicyOutput(
                chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
                chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                chosen_dt_us=torch.tensor([chosen_dt_us], dtype=torch.int64),
            ),
        ),
    )


def test_single_transfer_roundtrip(store_endpoint: tuple[str, int]) -> None:
    """A complete EpisodeOutput round-trips through sender + receiver + TCPStore."""
    channel = _MockNcclChannel()
    comm_factory = _MockCommFactory()
    sender_store = _make_client(store_endpoint)
    receiver_store = _make_client(store_endpoint)
    sender = _build_sender(
        rollout_idx=0,
        store=sender_store,
        channel=channel,
        comm_factory=comm_factory,
    )
    receiver = _build_receiver(
        num_rollout_replicas=1,
        store=receiver_store,
        channel=channel,
        comm_factory=comm_factory,
    )
    _setup_concurrently(sender.setup, receiver.setup)
    write_transport = NcclEpisodeWriter(
        store=sender_store, sender=sender, experiment_name=_EXPERIMENT_NAME, job_id=_JOB_ID
    )
    read_transport = _Resolver(receiver_store, receiver)
    try:
        episode = _minimal_episode()
        handle = write_transport.write(episode)
        loaded = read_transport._resolve_nccl_handle(handle)
        assert loaded.scene_id == episode.scene_id
        assert loaded.session_uuid == episode.session_uuid
        original = episode.policy_outputs[0]
        new = loaded.policy_outputs[0]
        assert torch.equal(new.chosen_xyz, original.chosen_xyz)
        assert torch.equal(new.chosen_quat, original.chosen_quat)
        assert torch.equal(new.chosen_dt_us, original.chosen_dt_us)
    finally:
        sender.close()
        receiver.close()


def test_concurrent_transfers_route_correctly_through_multiple_rollouts(
    store_endpoint: tuple[str, int],
) -> None:
    """Two senders + one receiver: each transfer reaches the right per-rollout comm."""
    channel = _MockNcclChannel()
    comm_factory = _MockCommFactory()
    sender_zero_store = _make_client(store_endpoint)
    sender_one_store = _make_client(store_endpoint)
    receiver_store = _make_client(store_endpoint)
    sender_zero = _build_sender(
        rollout_idx=0,
        store=sender_zero_store,
        channel=channel,
        comm_factory=comm_factory,
    )
    sender_one = _build_sender(
        rollout_idx=1,
        store=sender_one_store,
        channel=channel,
        comm_factory=comm_factory,
    )
    receiver = _build_receiver(
        num_rollout_replicas=2,
        store=receiver_store,
        channel=channel,
        comm_factory=comm_factory,
    )
    _setup_concurrently(sender_zero.setup, sender_one.setup, receiver.setup)
    transport_zero = NcclEpisodeWriter(
        store=sender_zero_store,
        sender=sender_zero,
        experiment_name=_EXPERIMENT_NAME,
        job_id=_JOB_ID,
    )
    transport_one = NcclEpisodeWriter(
        store=sender_one_store,
        sender=sender_one,
        experiment_name=_EXPERIMENT_NAME,
        job_id=_JOB_ID,
    )
    read_transport = _Resolver(receiver_store, receiver)
    try:
        # Distinct tensor payloads so cross-routing detects wrong-comm
        # delivery: scene_id rides on the manifest but chosen_dt_us rides on
        # the NCCL data plane.
        episode_zero = _minimal_episode(scene_id="scene_zero", chosen_dt_us=1000)
        episode_one = _minimal_episode(scene_id="scene_one", chosen_dt_us=2000)
        handle_zero = transport_zero.write(episode_zero)
        handle_one = transport_one.write(episode_one)
        assert handle_zero.startswith("nccl:0:")
        assert handle_one.startswith("nccl:1:")
        loaded_one = read_transport._resolve_nccl_handle(handle_one)
        loaded_zero = read_transport._resolve_nccl_handle(handle_zero)
        assert loaded_zero.scene_id == "scene_zero"
        assert loaded_one.scene_id == "scene_one"
        assert int(loaded_zero.policy_outputs[0].chosen_dt_us.item()) == 1000
        assert int(loaded_one.policy_outputs[0].chosen_dt_us.item()) == 2000
    finally:
        sender_zero.close()
        sender_one.close()
        receiver.close()


def test_rendezvous_returns_missing_when_payload_was_discarded_after_manifest_read(
    store_endpoint: tuple[str, int],
) -> None:
    """A late request for a discard-released transfer gets an event-driven ``missing``."""
    channel = _MockNcclChannel()
    comm_factory = _MockCommFactory()
    sender_store = _make_client(store_endpoint)
    receiver_store = _make_client(store_endpoint)
    sender = _build_sender(
        rollout_idx=0,
        store=sender_store,
        channel=channel,
        comm_factory=comm_factory,
    )
    receiver = _build_receiver(
        num_rollout_replicas=1,
        store=receiver_store,
        channel=channel,
        comm_factory=comm_factory,
    )
    _setup_concurrently(sender.setup, receiver.setup)
    write_transport = NcclEpisodeWriter(
        store=sender_store, sender=sender, experiment_name=_EXPERIMENT_NAME, job_id=_JOB_ID
    )
    read_transport = _Resolver(receiver_store, receiver)
    try:
        handle = write_transport.write(_minimal_episode(scene_id="discarded"))
        transfer_id = normalize_nccl_handle(handle)
        # Simulate a receiver that already observed metadata before the rollout
        # cleanup thread discarded the sender-side tensor payload.
        sender.release(transfer_id, reason="cosmos_cleanup")
        with pytest.raises(NcclRendezvousError, match="status=missing"):
            read_transport._resolve_nccl_handle(handle)
        deadline = time.time() + 2.0
        while transfer_id in sender._released_transfers and time.time() < deadline:
            time.sleep(0.01)
        assert transfer_id not in sender._released_transfers
    finally:
        sender.close()
        receiver.close()


def test_clean_shutdown_drops_pending_payloads(store_endpoint: tuple[str, int]) -> None:
    """Sender close() with pending payloads returns cleanly and aborts the comm."""
    channel = _MockNcclChannel()
    comm_factory = _MockCommFactory()
    abort_calls: list[int] = []
    sender_store = _make_client(store_endpoint)
    receiver_store = _make_client(store_endpoint)
    sender = NcclSender(
        experiment_name=_EXPERIMENT_NAME,
        job_id=_JOB_ID,
        rollout_idx=0,
        num_policy_replicas=1,
        dp_shard_size=1,
        store=sender_store,
        rendezvous=AckRendezvous(_FAST_ACK_TIMEOUT_S),
        create_nccl_uid=comm_factory.create_nccl_uid,
        create_nccl_comm=comm_factory.create_nccl_comm,
        nccl_send=channel.send,
        nccl_abort=abort_calls.append,
        config=_FAST_SENDER,
    )
    receiver = _build_receiver(
        num_rollout_replicas=1,
        store=receiver_store,
        channel=channel,
        comm_factory=comm_factory,
    )
    _setup_concurrently(sender.setup, receiver.setup)
    try:
        write_transport = NcclEpisodeWriter(
            store=sender_store, sender=sender, experiment_name=_EXPERIMENT_NAME, job_id=_JOB_ID
        )
        transfer_id = normalize_nccl_handle(write_transport.write(_minimal_episode()))
        assert transfer_id in sender._pending_sends
        time.sleep(0.05)
    finally:
        sender.close()
        receiver.close()
    assert sender.comm_idx in abort_calls
    assert sender._pending_sends == {}
