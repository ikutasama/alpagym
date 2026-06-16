# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the NCCL rendezvous against a real TCPStore."""

import logging
import socket
import threading
import time

import pytest
from alpagym_runtime.transport.nccl.protocol import (
    NCCL_RENDEZVOUS_ACCEPTED,
    NCCL_RENDEZVOUS_CANCELLED,
    NCCL_RENDEZVOUS_MISSING,
    NCCL_RENDEZVOUS_REQUESTED,
    NcclTransferAck,
    NcclTransferRequest,
    build_ack_key,
    build_nccl_prefix,
    build_req_key,
    build_state_key,
)
from alpagym_runtime.transport.nccl.rendezvous import AckRendezvous, NcclRendezvousError
from torch.distributed import TCPStore

_PREFIX = build_nccl_prefix(experiment_name="test", job_id="job")
_TRANSFER_ID = "0:payload"
_LOGGER = logging.getLogger("test_rendezvous")
_FAST_ACK_TIMEOUT_S = 0.2


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


def test_request_ack_raises_when_no_sender_appears(store: TCPStore) -> None:
    """request_ack with no live sender hits the deadline and cancels its request."""
    rendezvous = AckRendezvous(_FAST_ACK_TIMEOUT_S)
    with pytest.raises(NcclRendezvousError, match="status=no_ack"):
        rendezvous.request_ack(
            prefix=_PREFIX,
            transfer_id=_TRANSFER_ID,
            requester_rank=1,
            rollout_idx=0,
            store=store,
            role="Policy",
            logger=_LOGGER,
        )
    state_value = store.get(build_state_key(_PREFIX, _TRANSFER_ID))
    state_str = state_value.decode("utf-8") if isinstance(state_value, bytes) else state_value
    assert state_str == NCCL_RENDEZVOUS_CANCELLED
    assert store.check([build_req_key(_PREFIX, _TRANSFER_ID)])

    decision = rendezvous.process_request(
        prefix=_PREFIX,
        rollout_idx=0,
        transfer_id=_TRANSFER_ID,
        store=store,
        claim_transfer_snapshot=lambda _: (None, [], 0),
        logger=_LOGGER,
    )
    assert decision.status == NCCL_RENDEZVOUS_CANCELLED
    assert not _rendezvous_keys_exist(store, _TRANSFER_ID)


def test_process_request_accepts_when_snapshot_present(store: TCPStore) -> None:
    """A request whose snapshot is non-None transitions to accepted."""
    rendezvous = AckRendezvous(_FAST_ACK_TIMEOUT_S)
    _seed_request(store, _TRANSFER_ID)

    decision = rendezvous.process_request(
        prefix=_PREFIX,
        rollout_idx=0,
        transfer_id=_TRANSFER_ID,
        store=store,
        claim_transfer_snapshot=_snapshot_with_payload,
        logger=_LOGGER,
    )

    assert decision.status == NCCL_RENDEZVOUS_ACCEPTED
    assert decision.tensors_map is not None


def test_process_request_writes_missing_ack_when_snapshot_none(store: TCPStore) -> None:
    """When the sender has no payload registered, ACK reads ``missing``."""
    rendezvous = AckRendezvous(_FAST_ACK_TIMEOUT_S)
    _seed_request(store, _TRANSFER_ID)

    decision = rendezvous.process_request(
        prefix=_PREFIX,
        rollout_idx=0,
        transfer_id=_TRANSFER_ID,
        store=store,
        claim_transfer_snapshot=lambda _: (None, [], 0),
        logger=_LOGGER,
    )
    ack = NcclTransferAck.from_json(store.get(build_ack_key(_PREFIX, _TRANSFER_ID)))

    assert decision.status == NCCL_RENDEZVOUS_MISSING
    assert decision.tensors_map is None
    assert ack.status == NCCL_RENDEZVOUS_MISSING


def test_process_request_raises_on_rollout_idx_mismatch(store: TCPStore) -> None:
    """A request routed to the wrong rollout communicator is rejected."""
    rendezvous = AckRendezvous(_FAST_ACK_TIMEOUT_S)
    _seed_request(store, _TRANSFER_ID)

    with pytest.raises(NcclRendezvousError, match="routed to rollout_idx=2"):
        rendezvous.process_request(
            prefix=_PREFIX,
            rollout_idx=2,
            transfer_id=_TRANSFER_ID,
            store=store,
            claim_transfer_snapshot=lambda _: (None, [], 0),
            logger=_LOGGER,
        )

    state_value = store.get(build_state_key(_PREFIX, _TRANSFER_ID))
    state_str = state_value.decode("utf-8") if isinstance(state_value, bytes) else state_value
    assert state_str == NCCL_RENDEZVOUS_REQUESTED


def test_process_request_skips_send_when_receiver_already_cancelled(
    store: TCPStore,
) -> None:
    """Sender losing the CAS to a prior cancel returns a no-send decision."""
    rendezvous = AckRendezvous(_FAST_ACK_TIMEOUT_S)
    _seed_request(store, _TRANSFER_ID, state=NCCL_RENDEZVOUS_CANCELLED)

    decision = rendezvous.process_request(
        prefix=_PREFIX,
        rollout_idx=0,
        transfer_id=_TRANSFER_ID,
        store=store,
        claim_transfer_snapshot=_snapshot_with_payload,
        logger=_LOGGER,
    )

    assert decision.status == NCCL_RENDEZVOUS_CANCELLED
    assert decision.tensors_map is None
    assert not _rendezvous_keys_exist(store, _TRANSFER_ID)


def test_request_ack_returns_ack_when_sender_wins_during_cancel_window() -> None:
    """If the sender accepts during timeout cancellation, request_ack returns the ACK."""
    host = "127.0.0.1"
    port = _free_port()
    master = TCPStore(
        host_name=host, port=port, world_size=1, is_master=True, wait_for_workers=False
    )
    receiver_store = TCPStore(host_name=host, port=port, world_size=1, is_master=False)

    class _RacingRendezvous(AckRendezvous):
        """Inject the sender-winning race between receiver wait timeout and cancel CAS."""

        def _cancel_unaccepted_request(
            self,
            *,
            prefix: str,
            transfer_id: str,
            store: TCPStore,
        ) -> str:
            store.compare_set(
                build_state_key(prefix, transfer_id),
                NCCL_RENDEZVOUS_REQUESTED,
                NCCL_RENDEZVOUS_ACCEPTED,
            )
            ack = NcclTransferAck(
                request_id="late-claim",
                transfer_id=transfer_id,
                status=NCCL_RENDEZVOUS_ACCEPTED,
                message="late accept",
                pending_count=0,
                accepted_at_ms=int(time.time() * 1000),
            )
            store.set(build_ack_key(prefix, transfer_id), ack.to_json())
            return super()._cancel_unaccepted_request(
                prefix=prefix,
                transfer_id=transfer_id,
                store=store,
            )

    rendezvous = _RacingRendezvous(_FAST_ACK_TIMEOUT_S)
    try:
        ack = rendezvous.request_ack(
            prefix=_PREFIX,
            transfer_id=_TRANSFER_ID,
            requester_rank=1,
            rollout_idx=0,
            store=receiver_store,
            role="Policy",
            logger=_LOGGER,
        )
        assert ack.status == NCCL_RENDEZVOUS_ACCEPTED
    finally:
        del master


def test_request_ack_succeeds_when_sender_processes_in_parallel() -> None:
    """End-to-end: receiver's request_ack succeeds once a sender thread claims it."""
    host = "127.0.0.1"
    port = _free_port()
    master = TCPStore(
        host_name=host,
        port=port,
        world_size=1,
        is_master=True,
        wait_for_workers=False,
    )
    receiver_store = TCPStore(host_name=host, port=port, world_size=1, is_master=False)
    sender_store = TCPStore(host_name=host, port=port, world_size=1, is_master=False)
    rendezvous = AckRendezvous(ack_timeout_seconds=5.0)

    def _sender_loop() -> None:
        """Poll the transfer-id request key and claim the first request that appears."""
        deadline = time.time() + 3.0
        req_key = build_req_key(_PREFIX, _TRANSFER_ID)
        while time.time() < deadline:
            if sender_store.check([req_key]):
                rendezvous.process_request(
                    prefix=_PREFIX,
                    rollout_idx=0,
                    transfer_id=_TRANSFER_ID,
                    store=sender_store,
                    claim_transfer_snapshot=_snapshot_with_payload,
                    logger=_LOGGER,
                )
                return
            time.sleep(0.02)

    sender_thread = threading.Thread(target=_sender_loop, daemon=True)
    sender_thread.start()
    try:
        ack = rendezvous.request_ack(
            prefix=_PREFIX,
            transfer_id=_TRANSFER_ID,
            requester_rank=1,
            rollout_idx=0,
            store=receiver_store,
            role="Policy",
            logger=_LOGGER,
        )
        assert ack.status == NCCL_RENDEZVOUS_ACCEPTED
    finally:
        sender_thread.join(timeout=3.0)
        del master


def test_request_ack_deletes_rendezvous_keys_when_ack_payload_is_malformed(
    store: TCPStore,
) -> None:
    """Malformed ACK payload still triggers per-transfer key cleanup."""
    rendezvous = AckRendezvous(_FAST_ACK_TIMEOUT_S)
    store.set(build_ack_key(_PREFIX, _TRANSFER_ID), "{not-json")

    with pytest.raises(Exception):
        rendezvous.request_ack(
            prefix=_PREFIX,
            transfer_id=_TRANSFER_ID,
            requester_rank=1,
            rollout_idx=0,
            store=store,
            role="Policy",
            logger=_LOGGER,
        )

    assert not _rendezvous_keys_exist(store, _TRANSFER_ID)


def _seed_request(
    store: TCPStore,
    transfer_id: str,
    state: str = NCCL_RENDEZVOUS_REQUESTED,
) -> None:
    """Seed one transfer-id keyed request as the receiver would."""
    request = NcclTransferRequest(
        request_id="req-1",
        transfer_id=transfer_id,
        requester_rank=1,
        rollout_idx=0,
        deadline_ms=9_999_999_999_999,
        state=NCCL_RENDEZVOUS_REQUESTED,
    )
    store.set(build_req_key(_PREFIX, transfer_id), request.to_json())
    store.set(build_state_key(_PREFIX, transfer_id), state)


def _snapshot_with_payload(_transfer_id: str):
    """Return one fake tensor snapshot for rendezvous tests."""
    return {"k": object()}, ["0:payload"], 1  # type: ignore[dict-item]


def _rendezvous_keys_exist(store: TCPStore, transfer_id: str) -> bool:
    """Return whether any control-plane key remains for ``transfer_id``."""
    return store.check(
        [
            build_state_key(_PREFIX, transfer_id),
            build_req_key(_PREFIX, transfer_id),
            build_ack_key(_PREFIX, transfer_id),
        ]
    )
