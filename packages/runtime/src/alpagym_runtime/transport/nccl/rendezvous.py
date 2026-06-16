# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-transfer rendezvous: a control-plane handshake that pairs one NCCL
sender with one receiver before any bulk tensors move.

See ``README.md`` in this directory for the protocol design — why a
handshake is needed, why TCPStore is the control plane, the no-deadlock
state machine, and the step-by-step flow.
"""

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

import torch
from torch.distributed import TCPStore

from alpagym_runtime.transport.nccl.protocol import (
    NCCL_RENDEZVOUS_ACCEPTED,
    NCCL_RENDEZVOUS_CANCELLED,
    NCCL_RENDEZVOUS_MISSING,
    NCCL_RENDEZVOUS_REQUESTED,
    NcclTransferAck,
    NcclTransferRequest,
    build_ack_key,
    build_req_key,
    build_state_key,
)


class NcclRendezvousError(RuntimeError):
    """Raised when the rendezvous fails before NCCL should start."""


@dataclass(frozen=True)
class SendDecision:
    """One sender-side decision derived from a rendezvous wakeup.

    ``status`` is one of ``NCCL_RENDEZVOUS_*`` and indicates whether the
    sender should proceed with NCCL transmission. ``tensors_map`` is the
    snapshot of the registered payload tensors and is non-None only when
    ``status == NCCL_RENDEZVOUS_ACCEPTED``.
    """

    status: str
    request_id: str
    transfer_id: str
    requester_rank: int
    tensors_map: dict[str, torch.Tensor] | None
    pending_count: int


class AckRendezvous:
    """Per-transfer handshake backed by explicit ACK writes and ``compare_set``.

    Two entry points, one for each side:

    - :meth:`request_ack` — receiver-side. Publishes the request and blocks
      until the sender ACKs or the deadline fires.
    - :meth:`process_request` — sender-side. Called once per transfer id by
      the sender's poll loop; decides whether to ship or bail.

    See ``README.md`` in this directory for the full protocol design.
    """

    def __init__(self, ack_timeout_seconds: float = 120.0):
        """Configure the per-transfer ACK timeout in seconds."""
        self._ack_timeout_seconds = ack_timeout_seconds

    def request_ack(
        self,
        prefix: str,
        transfer_id: str,
        requester_rank: int,
        rollout_idx: int,
        store: TCPStore,
        role: str,
        logger: logging.Logger,
    ) -> NcclTransferAck:
        """Receiver-side rendezvous: write request, wait for ACK."""
        start_ms = _now_ms()
        request = self._publish_request(
            prefix=prefix,
            transfer_id=transfer_id,
            requester_rank=requester_rank,
            rollout_idx=rollout_idx,
            start_ms=start_ms,
            store=store,
        )
        ack_key = build_ack_key(prefix, transfer_id)
        try:
            store.wait([ack_key], timedelta(seconds=self._ack_timeout_seconds))
        except RuntimeError:
            # Wait timed out. Try to cancel; if the sender won the CAS race
            # in the meantime, the ACK is (or will be) in the store and we
            # fall through to read it instead of falsely raising no_ack.
            final_state = self._cancel_unaccepted_request(
                prefix=prefix,
                transfer_id=transfer_id,
                store=store,
            )
            if final_state == NCCL_RENDEZVOUS_CANCELLED:
                # The sender never accepted, so it deletes the per-transfer keys
                # when it observes this CANCELLED state. Deleting them here could
                # race a sender still reading req/state for this transfer.
                raise _build_rendezvous_error(
                    role=role,
                    status="no_ack",
                    message="no sender ACK before rendezvous deadline",
                    request=request,
                    start_ms=start_ms,
                )
            # Sender committed before our cancel; the ACK write is a single
            # store.set following the CAS. Wait the full configured budget for it
            # to land -- a hardcoded short grace can expire under TCPStore
            # contention, leaving the receiver not posting nccl_recv while the
            # committed sender posts nccl_send (a poisoned comm).
            try:
                store.wait([ack_key], timedelta(seconds=self._ack_timeout_seconds))
            except RuntimeError:
                self._delete_rendezvous_keys(
                    prefix=prefix,
                    transfer_id=transfer_id,
                    store=store,
                )
                raise _build_rendezvous_error(
                    role=role,
                    status="no_ack",
                    message=(
                        f"sender transitioned state to {final_state} but ACK "
                        "did not land within the grace window"
                    ),
                    request=request,
                    start_ms=start_ms,
                )
        try:
            ack = NcclTransferAck.from_json(_decode(store.get(ack_key)))
        finally:
            self._delete_rendezvous_keys(prefix=prefix, transfer_id=transfer_id, store=store)
        if ack.status != NCCL_RENDEZVOUS_ACCEPTED:
            logger.info(
                "[NCCL:Rendezvous] request_id=%s transfer_id=%s status=%s pending=%s",
                request.request_id,
                transfer_id,
                ack.status,
                ack.pending_count,
            )
            raise _build_rendezvous_error(
                role=role,
                status=ack.status,
                message=ack.message,
                request=request,
                start_ms=start_ms,
            )
        logger.info(
            "[NCCL:Rendezvous] request_id=%s transfer_id=%s status=accepted ack_ms=%s",
            request.request_id,
            transfer_id,
            max(0, _now_ms() - start_ms),
        )
        return ack

    def _publish_request(
        self,
        prefix: str,
        transfer_id: str,
        requester_rank: int,
        rollout_idx: int,
        start_ms: int,
        store: TCPStore,
    ) -> NcclTransferRequest:
        """Write the request and initialize state to ``"requested"``."""
        request = NcclTransferRequest(
            request_id=uuid.uuid4().hex,
            transfer_id=transfer_id,
            requester_rank=requester_rank,
            rollout_idx=rollout_idx,
            deadline_ms=start_ms + int(self._ack_timeout_seconds * 1000),
            state=NCCL_RENDEZVOUS_REQUESTED,
        )
        store.set(build_state_key(prefix, transfer_id), NCCL_RENDEZVOUS_REQUESTED)
        store.set(build_req_key(prefix, transfer_id), request.to_json())
        return request

    def process_request(
        self,
        prefix: str,
        rollout_idx: int,
        transfer_id: str,
        store: TCPStore,
        claim_transfer_snapshot: Callable[
            [str], tuple[dict[str, torch.Tensor] | None, list[str], int]
        ],
        logger: logging.Logger,
    ) -> SendDecision:
        """Sender-side: decide what to do for the request at ``transfer_id``."""
        request = self._read_request(prefix=prefix, transfer_id=transfer_id, store=store)
        if request.rollout_idx != rollout_idx:
            raise NcclRendezvousError(
                f"NCCL rendezvous routed to rollout_idx={rollout_idx}, "
                f"but request targets rollout_idx={request.rollout_idx}"
            )
        tensors_map, pending_preview, pending_count = claim_transfer_snapshot(request.transfer_id)
        logger.debug(
            "[NCCL:Rendezvous] request_id=%s transfer_id=%s found=%s pending=%s keys=%s",
            request.request_id,
            request.transfer_id[:16],
            tensors_map is not None,
            pending_count,
            [key[:16] for key in pending_preview],
        )
        if tensors_map is None:
            return self._handle_missing_payload(
                prefix=prefix,
                request=request,
                pending_count=pending_count,
                store=store,
                logger=logger,
            )
        return self._handle_present_payload(
            prefix=prefix,
            request=request,
            tensors_map=tensors_map,
            pending_count=pending_count,
            store=store,
            logger=logger,
        )

    def _read_request(
        self,
        prefix: str,
        transfer_id: str,
        store: TCPStore,
    ) -> NcclTransferRequest:
        """Read the receiver's request payload for ``transfer_id``."""
        req_key = build_req_key(prefix, transfer_id)
        if not store.check([req_key]):
            raise NcclRendezvousError(f"NCCL request key {req_key} is not present")
        return NcclTransferRequest.from_json(_decode(store.get(req_key)))

    def _handle_missing_payload(
        self,
        prefix: str,
        request: NcclTransferRequest,
        pending_count: int,
        store: TCPStore,
        logger: logging.Logger,
    ) -> SendDecision:
        """Sender lacks the payload: try to mark the request ``"missing"``."""
        status = self._try_transition_state(
            prefix=prefix,
            transfer_id=request.transfer_id,
            expected=NCCL_RENDEZVOUS_REQUESTED,
            desired=NCCL_RENDEZVOUS_MISSING,
            store=store,
        )
        if status == NCCL_RENDEZVOUS_MISSING:
            self._write_ack(
                prefix=prefix,
                request=request,
                status=NCCL_RENDEZVOUS_MISSING,
                message="sender no longer owns transfer payload",
                pending_count=pending_count,
                accepted_at_ms=None,
                store=store,
            )
        elif status == NCCL_RENDEZVOUS_CANCELLED:
            self._delete_rendezvous_keys(
                prefix=prefix,
                transfer_id=request.transfer_id,
                store=store,
            )
        else:
            raise NcclRendezvousError(
                f"NCCL rendezvous transfer_id={request.transfer_id} reached "
                "missing-payload handling with "
                f"unexpected state={status!r} (expected missing or cancelled)"
            )
        logger.info(
            "[NCCL:Rendezvous] request_id=%s transfer_id=%s status=%s pending=%s",
            request.request_id,
            request.transfer_id,
            status,
            pending_count,
        )
        return _build_no_send_decision(request, status=status, pending_count=pending_count)

    def _handle_present_payload(
        self,
        prefix: str,
        request: NcclTransferRequest,
        tensors_map: dict[str, torch.Tensor],
        pending_count: int,
        store: TCPStore,
        logger: logging.Logger,
    ) -> SendDecision:
        """Sender has the payload: try to claim the rendezvous as ``"accepted"``."""
        status = self._try_transition_state(
            prefix=prefix,
            transfer_id=request.transfer_id,
            expected=NCCL_RENDEZVOUS_REQUESTED,
            desired=NCCL_RENDEZVOUS_ACCEPTED,
            store=store,
        )
        if status == NCCL_RENDEZVOUS_CANCELLED:
            logger.info(
                "[NCCL:Rendezvous] request_id=%s transfer_id=%s status=%s",
                request.request_id,
                request.transfer_id,
                status,
            )
            self._delete_rendezvous_keys(
                prefix=prefix,
                transfer_id=request.transfer_id,
                store=store,
            )
            return _build_no_send_decision(request, status=status, pending_count=pending_count)
        if status != NCCL_RENDEZVOUS_ACCEPTED:
            raise NcclRendezvousError(
                f"NCCL rendezvous transfer_id={request.transfer_id} reached "
                "present-payload accept with "
                f"unexpected state={status!r} (expected accepted or cancelled)"
            )
        self._write_ack(
            prefix=prefix,
            request=request,
            status=NCCL_RENDEZVOUS_ACCEPTED,
            message="sender accepted rendezvous",
            pending_count=pending_count,
            accepted_at_ms=_now_ms(),
            store=store,
        )
        logger.info(
            "[NCCL:Rendezvous] request_id=%s transfer_id=%s status=accepted pending=%s",
            request.request_id,
            request.transfer_id,
            pending_count,
        )
        return SendDecision(
            status=status,
            request_id=request.request_id,
            transfer_id=request.transfer_id,
            requester_rank=request.requester_rank,
            tensors_map=tensors_map,
            pending_count=pending_count,
        )

    def _try_transition_state(
        self,
        prefix: str,
        transfer_id: str,
        expected: str,
        desired: str,
        store: TCPStore,
    ) -> str:
        """Atomic state transition via compare_set; return the resulting state.

        torch's ``TCPStore.compare_set`` returns the value the key holds after
        the call: ``desired`` when the current value equaled ``expected`` (the
        swap happened), otherwise the unchanged current value. That post-call
        value is exactly the resulting state the callers need.
        """
        state_key = build_state_key(prefix, transfer_id)
        return _decode(store.compare_set(state_key, expected, desired))

    def _cancel_unaccepted_request(
        self,
        prefix: str,
        transfer_id: str,
        store: TCPStore,
    ) -> str:
        """Cancel a request that has not yet been accepted by a sender."""
        return self._try_transition_state(
            prefix=prefix,
            transfer_id=transfer_id,
            expected=NCCL_RENDEZVOUS_REQUESTED,
            desired=NCCL_RENDEZVOUS_CANCELLED,
            store=store,
        )

    def _write_ack(
        self,
        prefix: str,
        request: NcclTransferRequest,
        status: str,
        message: str,
        pending_count: int,
        accepted_at_ms: int | None,
        store: TCPStore,
    ) -> None:
        """Write the per-transfer ACK payload so the receiver's wait unblocks.

        The single ``store.set`` is load-bearing: ``request_ack`` reads the ACK
        with one ``store.get`` once its ``store.wait`` returns, so the payload
        must land atomically. Splitting it into multiple sets breaks that read.
        """
        ack = NcclTransferAck(
            request_id=request.request_id,
            transfer_id=request.transfer_id,
            status=status,
            message=message,
            pending_count=pending_count,
            accepted_at_ms=accepted_at_ms,
        )
        store.set(build_ack_key(prefix, request.transfer_id), ack.to_json())

    def _delete_rendezvous_keys(self, prefix: str, transfer_id: str, store: TCPStore) -> None:
        """Best-effort cleanup for per-transfer rendezvous control-plane keys."""
        for key in (
            build_req_key(prefix, transfer_id),
            build_state_key(prefix, transfer_id),
            build_ack_key(prefix, transfer_id),
        ):
            try:
                store.delete_key(key)
            except Exception:
                pass


def _build_no_send_decision(
    request: NcclTransferRequest,
    status: str,
    pending_count: int,
) -> SendDecision:
    """Build a SendDecision that signals no NCCL send should occur."""
    return SendDecision(
        status=status,
        request_id=request.request_id,
        transfer_id=request.transfer_id,
        requester_rank=request.requester_rank,
        tensors_map=None,
        pending_count=pending_count,
    )


def _build_rendezvous_error(
    role: str,
    status: str,
    message: str,
    request: NcclTransferRequest,
    start_ms: int,
) -> NcclRendezvousError:
    """Compose the canonical rendezvous-failure error."""
    elapsed_ms = max(0, _now_ms() - start_ms)
    return NcclRendezvousError(
        f"[{role}] NCCL rendezvous failed: status={status}, "
        f"transfer_id={request.transfer_id}, request_id={request.request_id}, "
        f"rollout_idx={request.rollout_idx}, message={message}, "
        f"ack_wait_ms={elapsed_ms}."
    )


def _now_ms() -> int:
    """Current time in milliseconds since the epoch."""
    return int(time.time() * 1000)


def _decode(value: Any) -> str:
    """Decode a TCPStore value into a Python str (TCPStore.get/compare_set return bytes)."""
    return value.decode("utf-8")
