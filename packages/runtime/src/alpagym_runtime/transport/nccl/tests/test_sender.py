# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the NCCL producer-side sender (TCPStore-backed)."""

import socket
import threading
import time
from typing import Any

import pytest
import torch
from alpagym_runtime.transport.nccl.protocol import (
    NCCL_RENDEZVOUS_ACCEPTED,
    NCCL_RENDEZVOUS_CANCELLED,
    build_nccl_prefix,
    build_req_key,
)
from alpagym_runtime.transport.nccl.rendezvous import SendDecision
from alpagym_runtime.transport.nccl.sender import NcclSender, NcclSenderConfig, assign_rollout_idx
from torch.distributed import TCPStore


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


class _StubRendezvous:
    """No-op Rendezvous used when the test doesn't exercise process_request."""

    def request_ack(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def process_request(self, **kwargs: Any) -> Any:
        raise NotImplementedError


def _build_sender(
    store: TCPStore,
    rendezvous: Any = None,
    nccl_send: Any = None,
    config: NcclSenderConfig | None = None,
    num_policy_replicas: int = 1,
) -> NcclSender:
    """Construct a sender wired with no-op cosmos-rl callables for unit tests."""
    return NcclSender(
        experiment_name="exp",
        job_id="job",
        rollout_idx=0,
        num_policy_replicas=num_policy_replicas,
        dp_shard_size=1,
        store=store,
        rendezvous=rendezvous or _StubRendezvous(),
        create_nccl_uid=lambda: [1, 2, 3],
        create_nccl_comm=lambda *args, **kwargs: 17,
        nccl_send=nccl_send or (lambda *args, **kwargs: None),
        nccl_abort=lambda comm_idx: None,
        config=config,
    )


def test_constructor_rejects_nonpositive_dp_shard_size(store: TCPStore) -> None:
    """dp_shard_size <= 0 fails fast at construction time."""
    with pytest.raises(ValueError, match="dp_shard_size must be positive"):
        NcclSender(
            experiment_name="exp",
            job_id="job",
            rollout_idx=0,
            num_policy_replicas=1,
            dp_shard_size=0,
            store=store,
            rendezvous=_StubRendezvous(),
            create_nccl_uid=lambda: [1, 2, 3],
            create_nccl_comm=lambda *args, **kwargs: 17,
            nccl_send=lambda *args, **kwargs: None,
            nccl_abort=lambda comm_idx: None,
        )


def test_constructor_rejects_nonpositive_num_policy_replicas(store: TCPStore) -> None:
    """num_policy_replicas <= 0 fails fast at construction time."""
    with pytest.raises(ValueError, match="num_policy_replicas must be positive"):
        _build_sender(store=store, num_policy_replicas=0)


def test_send_raises_when_nccl_not_initialized(store: TCPStore) -> None:
    """send() before setup() raises rather than silently dropping the payload."""
    sender = _build_sender(store=store)
    with pytest.raises(RuntimeError, match="NCCL not initialized"):
        sender.send({"key": torch.zeros(1)}, "0:abc")


def test_send_registers_tensors_under_transfer_id(store: TCPStore) -> None:
    """send() places the tensors in the pending registry."""
    sender = _build_sender(store=store)
    sender._comm_idx = 17  # bypass setup for this unit test
    tensors = {"k": torch.tensor([1.0])}
    sender.send(tensors, "0:abc")
    assert "0:abc" in sender._pending_sends
    assert sender._pending_sends["0:abc"] is tensors


def test_send_raises_when_registry_is_full(store: TCPStore) -> None:
    """A full registry refuses new payloads instead of evicting a live one."""
    sender = _build_sender(store=store, config=NcclSenderConfig(max_pending_transfers=2))
    sender._comm_idx = 17
    sender.send({"k": torch.tensor([1.0])}, "0:a")
    sender.send({"k": torch.tensor([2.0])}, "0:b")
    with pytest.raises(RuntimeError, match="Pending NCCL send registry is full"):
        sender.send({"k": torch.tensor([3.0])}, "0:c")


def test_release_returns_true_for_known_transfer_and_false_for_unknown(store: TCPStore) -> None:
    """release() returns True for a registered transfer, False otherwise; idempotent.

    The public release() is the single drop-from-registry path for both internal
    post-send cleanup and external rollback callers, so the return-value contract
    matters: callers can tell whether they observed the transfer.
    """
    sender = _build_sender(store=store)
    sender._comm_idx = 17
    sender._pending_sends["0:abc"] = {"k": torch.tensor([1.0])}
    assert sender.release("0:abc", reason="test") is True
    assert "0:abc" not in sender._pending_sends
    # Second call is a no-op that returns False.
    assert sender.release("0:abc", reason="test") is False
    # Unknown transfer ids are also a no-op.
    assert sender.release("0:never-registered", reason="test") is False


def test_claimed_transfer_survives_concurrent_release(store: TCPStore) -> None:
    """A release that loses the held-to-claimed race does not revoke tensors."""
    sender = _build_sender(store=store)
    sender._comm_idx = 17
    sender._pending_sends["0:abc"] = {"k": torch.tensor([1.0])}
    snapshot, _, count = sender._claim_transfer_snapshot("0:abc")
    assert sender.release("0:abc", reason="cosmos_cleanup") is True
    assert snapshot is not None
    assert torch.equal(snapshot["k"], torch.tensor([1.0]))
    assert count == 1
    assert "0:abc" not in sender._pending_sends
    assert "0:abc" in sender._claimed_sends
    assert "0:abc" not in sender._released_transfers
    sender._complete_claimed_transfer("0:abc")
    assert "0:abc" not in sender._claimed_sends


def test_release_marks_discarded_transfer_for_late_request(store: TCPStore) -> None:
    """Discard-like releases leave a short marker; terminal send releases do not."""
    sender = _build_sender(store=store)
    sender._comm_idx = 17
    sender._pending_sends["0:discard"] = {"k": torch.tensor([1.0])}
    sender._pending_sends["0:published-error"] = {"k": torch.tensor([2.0])}
    sender._pending_sends["0:sent"] = {"k": torch.tensor([3.0])}

    assert sender.release("0:discard", reason="cosmos_cleanup") is True
    assert sender.release("0:published-error", reason="metadata_publish_error") is True
    assert sender.release("0:sent", reason="read_complete") is True

    assert "0:discard" in sender._released_transfers
    assert "0:published-error" not in sender._released_transfers
    assert "0:sent" not in sender._released_transfers


def test_sender_loop_processes_transfer_id_request_and_removes_after_success(
    store: TCPStore,
) -> None:
    """Sender loop polls transfer-id request keys and evicts each shipped transfer."""

    class _AcceptingRendezvous:
        """Rendezvous double that always accepts the registered payload."""

        def request_ack(self, **kwargs: Any) -> Any:
            """Unused by the sender loop; raise to make a misuse obvious."""
            raise NotImplementedError

        def process_request(self, **kwargs: Any) -> SendDecision:
            """Return an ACCEPTED decision pointing at the pre-seeded payload."""
            snapshot, _, count = kwargs["claim_transfer_snapshot"](kwargs["transfer_id"])
            return SendDecision(
                status=NCCL_RENDEZVOUS_ACCEPTED,
                request_id="r1",
                transfer_id=kwargs["transfer_id"],
                requester_rank=1,
                tensors_map=snapshot,
                pending_count=count,
            )

    sent_signal = threading.Event()

    def _nccl_send_signal(tensor: Any, dst_rank: int, comm_idx: int, **kwargs: Any) -> None:
        sent_signal.set()

    sender = _build_sender(
        store=store,
        rendezvous=_AcceptingRendezvous(),
        nccl_send=_nccl_send_signal,
        config=NcclSenderConfig(sender_poll_interval_seconds=0.01),
    )
    sender._comm_idx = 17
    transfer_id = "0:xfr"
    sender._pending_sends[transfer_id] = {"k": torch.tensor([1.0])}
    sender._device = torch.device("cpu")
    store.set(build_req_key(sender._rollout_prefix, transfer_id), "request-present")
    loop_thread = threading.Thread(target=sender._run_sender_loop, daemon=True)
    loop_thread.start()
    try:
        assert sent_signal.wait(timeout=2.0)
        # Brief settle for the post-send remove_transfer.
        deadline = time.time() + 1.0
        while (
            transfer_id in sender._pending_sends or transfer_id in sender._claimed_sends
        ) and time.time() < deadline:
            time.sleep(0.01)
        assert transfer_id not in sender._pending_sends
        assert transfer_id not in sender._claimed_sends
    finally:
        sender._stop_event.set()
        loop_thread.join(timeout=2.0)


def test_wait_until_drained_returns_true_when_registry_empty(store: TCPStore) -> None:
    """An empty pending registry drains immediately (flush before weight sync)."""
    sender = _build_sender(store=store)
    sender._comm_idx = 17
    start = time.monotonic()
    assert sender.wait_until_drained(timeout_seconds=1.0) is True
    # Returns on the first check, well within the timeout.
    assert time.monotonic() - start < 0.5


def test_wait_until_drained_ignores_held_pending_sends(store: TCPStore) -> None:
    """A held payload nobody has requested is inert and must not block the flush.

    Held sends run no NCCL op, so they cannot collide with the R2R broadcast.
    Waiting on them is what deadlocked the flush against the off-policy buffer.
    """
    sender = _build_sender(store=store, config=NcclSenderConfig(sender_poll_interval_seconds=0.01))
    sender._comm_idx = 17
    sender._pending_sends["0:held"] = {"k": torch.tensor([1.0])}
    assert sender.wait_until_drained(timeout_seconds=0.1) is True


def test_wait_until_drained_times_out_with_an_in_flight_send(store: TCPStore) -> None:
    """A claimed transfer is mid-flight on the GPU, so the bounded wait returns False."""
    sender = _build_sender(store=store, config=NcclSenderConfig(sender_poll_interval_seconds=0.01))
    sender._comm_idx = 17
    sender._claimed_sends["0:inflight"] = {"k": torch.tensor([1.0])}
    assert sender.wait_until_drained(timeout_seconds=0.1) is False


def test_wait_until_drained_returns_true_once_the_send_completes(store: TCPStore) -> None:
    """The bounded wait observes the in-flight send finishing while it polls."""
    sender = _build_sender(store=store, config=NcclSenderConfig(sender_poll_interval_seconds=0.01))
    sender._comm_idx = 17
    sender._claimed_sends["0:inflight"] = {"k": torch.tensor([1.0])}

    def _drain_after_delay() -> None:
        time.sleep(0.05)
        with sender._pending_sends_lock:
            sender._claimed_sends.clear()

    drainer = threading.Thread(target=_drain_after_delay, daemon=True)
    drainer.start()
    try:
        assert sender.wait_until_drained(timeout_seconds=2.0) is True
    finally:
        drainer.join(timeout=2.0)


def test_close_stops_worker_thread_and_aborts_comm(store: TCPStore) -> None:
    """close() sets stop_event, joins the worker thread, and aborts NCCL."""
    aborted: list[int] = []
    sender = _build_sender(store=store)
    sender._comm_idx = 31
    sender._nccl_abort = aborted.append
    fake_worker = _FakeJoinable()
    sender._worker_thread = fake_worker  # type: ignore[assignment]
    sender.close()
    assert sender._stop_event.is_set()
    assert fake_worker.joined
    assert aborted == [31]


class _FakeJoinable:
    """Stand-in for a Thread whose ``join`` we want to observe."""

    def __init__(self) -> None:
        self.joined = False

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.joined = True

    def is_alive(self) -> bool:
        return False


def test_assign_rollout_idx_raises_when_counter_exceeds_replicas(
    store: TCPStore,
) -> None:
    """A worker that lands on an out-of-range rollout_idx fails fast."""
    counter_key = (
        f"{build_nccl_prefix(experiment_name='exp', job_id='job')}:rollout_replica_counter"
    )
    # Pre-bump the counter so the next caller's assigned idx == num_rollout_replicas.
    store.add(counter_key, 2)
    with pytest.raises(RuntimeError, match="exceeds rollout replica count"):
        assign_rollout_idx(
            experiment_name="exp",
            job_id="job",
            num_rollout_replicas=2,
            store=store,
        )
    assert int(store.add(counter_key, 0)) == 2


def test_assign_rollout_idx_rolls_back_after_counter_overflow(
    store: TCPStore,
) -> None:
    """A rejected rollout worker must not burn the replica slot permanently."""
    counter_key = (
        f"{build_nccl_prefix(experiment_name='exp', job_id='job')}:rollout_replica_counter"
    )
    store.add(counter_key, 1)
    with pytest.raises(RuntimeError, match="exceeds rollout replica count"):
        assign_rollout_idx(
            experiment_name="exp",
            job_id="job",
            num_rollout_replicas=1,
            store=store,
        )
    assert int(store.add(counter_key, 0)) == 1


def test_sender_loop_continues_after_process_request_raises(store: TCPStore) -> None:
    """A transient process_request failure must not stop the poll loop.

    One bad transfer-id request raises while another valid held transfer is
    still processed in the same local scan.
    """
    sent_signals: list[str] = []

    class _FailingRendezvous:
        """Reject one transfer id and accept the other."""

        def request_ack(self, **kwargs: Any) -> Any:
            """Unused by the sender loop; raise to make a misuse obvious."""
            raise NotImplementedError

        def process_request(self, **kwargs: Any) -> SendDecision:
            """Raise for the first transfer and accept the second."""
            if kwargs["transfer_id"] == "0:first":
                raise RuntimeError("simulated transient failure")
            snapshot, _, count = kwargs["claim_transfer_snapshot"]("0:second")
            return SendDecision(
                status=NCCL_RENDEZVOUS_ACCEPTED,
                request_id="r2",
                transfer_id="0:second",
                requester_rank=1,
                tensors_map=snapshot,
                pending_count=count,
            )

    second_sent = threading.Event()

    def _nccl_send(tensor: Any, dst_rank: int, comm_idx: int, **kwargs: Any) -> None:
        sent_signals.append("sent")
        second_sent.set()

    sender = _build_sender(
        store=store,
        rendezvous=_FailingRendezvous(),
        nccl_send=_nccl_send,
        config=NcclSenderConfig(sender_poll_interval_seconds=0.01),
    )
    sender._comm_idx = 17
    sender._pending_sends["0:first"] = {"k": torch.tensor([0.0])}
    sender._pending_sends["0:second"] = {"k": torch.tensor([1.0])}
    sender._device = torch.device("cpu")
    store.set(build_req_key(sender._rollout_prefix, "0:first"), "request-present")
    store.set(build_req_key(sender._rollout_prefix, "0:second"), "request-present")
    loop_thread = threading.Thread(target=sender._run_sender_loop, daemon=True)
    loop_thread.start()
    try:
        assert second_sent.wait(timeout=2.0)
        assert sent_signals == ["sent"]
    finally:
        sender._stop_event.set()
        loop_thread.join(timeout=2.0)


def test_sender_loop_drops_cancelled_claimed_transfer(store: TCPStore) -> None:
    """A receiver-side rendezvous cancel drops the corresponding claimed payload."""

    class _CancellingRendezvous:
        """Rendezvous double that returns a cancelled decision for the registered payload."""

        def request_ack(self, **kwargs: Any) -> Any:
            """Unused by the sender loop; raise to make a misuse obvious."""
            raise NotImplementedError

        def process_request(self, **kwargs: Any) -> SendDecision:
            """Return a CANCELLED decision that names the pending transfer."""
            snapshot, _, count = kwargs["claim_transfer_snapshot"]("0:xfr")
            assert snapshot is not None
            return SendDecision(
                status=NCCL_RENDEZVOUS_CANCELLED,
                request_id="r-cancel",
                transfer_id="0:xfr",
                requester_rank=1,
                tensors_map=None,
                pending_count=count,
            )

    sender = _build_sender(
        store=store,
        rendezvous=_CancellingRendezvous(),
        config=NcclSenderConfig(sender_poll_interval_seconds=0.01),
    )
    sender._comm_idx = 17
    sender._pending_sends["0:xfr"] = {"k": torch.tensor([1.0])}
    sender._device = torch.device("cpu")
    store.set(build_req_key(sender._rollout_prefix, "0:xfr"), "request-present")
    loop_thread = threading.Thread(target=sender._run_sender_loop, daemon=True)
    loop_thread.start()
    try:
        deadline = time.time() + 2.0
        while "0:xfr" in sender._pending_sends and time.time() < deadline:
            time.sleep(0.01)
        assert "0:xfr" not in sender._pending_sends
        assert "0:xfr" not in sender._claimed_sends
    finally:
        sender._stop_event.set()
        loop_thread.join(timeout=2.0)


def test_post_accept_send_failure_aborts_comm_and_drops_it(store: TCPStore) -> None:
    """A post-accept nccl_send failure aborts the comm so the receiver's
    watchdog fires immediately, drops ``_comm_idx`` so subsequent sends
    fail fast, and removes the transfer from the pending registry.
    """

    class _AcceptingRendezvous:
        """Rendezvous double that always accepts the registered payload."""

        def request_ack(self, **kwargs: Any) -> Any:
            """Unused by the sender loop; raise to make a misuse obvious."""
            raise NotImplementedError

        def process_request(self, **kwargs: Any) -> SendDecision:
            """Return an ACCEPTED decision pointing at the pre-seeded payload."""
            snapshot, _, count = kwargs["claim_transfer_snapshot"]("0:xfr")
            return SendDecision(
                status=NCCL_RENDEZVOUS_ACCEPTED,
                request_id="r1",
                transfer_id="0:xfr",
                requester_rank=1,
                tensors_map=snapshot,
                pending_count=count,
            )

    aborted_comms: list[int] = []

    def _failing_send(tensor: Any, dst_rank: int, comm_idx: int, **kwargs: Any) -> None:
        raise RuntimeError("simulated nccl_send failure")

    sender = _build_sender(
        store=store,
        rendezvous=_AcceptingRendezvous(),
        nccl_send=_failing_send,
        config=NcclSenderConfig(sender_poll_interval_seconds=0.01),
    )
    sender._comm_idx = 17
    sender._nccl_abort = aborted_comms.append
    sender._pending_sends["0:xfr"] = {"k": torch.tensor([1.0])}
    sender._device = torch.device("cpu")
    store.set(build_req_key(sender._rollout_prefix, "0:xfr"), "request-present")
    loop_thread = threading.Thread(target=sender._run_sender_loop, daemon=True)
    loop_thread.start()
    try:
        deadline = time.time() + 2.0
        while not aborted_comms and time.time() < deadline:
            time.sleep(0.01)
        assert aborted_comms == [17]
        # comm dropped so subsequent sends fail fast at the not-initialized check.
        assert sender._comm_idx is None
        # transfer evicted from the pending registry.
        assert "0:xfr" not in sender._pending_sends
        assert "0:xfr" not in sender._claimed_sends
    finally:
        sender._stop_event.set()
        loop_thread.join(timeout=2.0)


def test_dead_comm_snapshot_returns_missing_payload(store: TCPStore) -> None:
    """A dead communicator makes later rendezvous requests non-accepted."""
    sender = _build_sender(store=store)
    sender._comm_idx = 17
    sender._pending_sends["0:next"] = {"k": torch.tensor([1.0])}
    sender._handle_post_accept_send_failure(
        transfer_id="0:failed",
        error=RuntimeError("simulated send failure"),
    )
    snapshot, _, count = sender._claim_transfer_snapshot("0:next")
    assert snapshot is None
    assert count == 0
