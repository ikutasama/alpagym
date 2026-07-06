# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NCCL transport producer side, backed by ``torch.distributed.TCPStore``.

:class:`NcclSender` owns three things:

- One NCCL communicator per rollout-worker process.
- The in-process pending-payload registry.
- A background thread that polls content-addressed request keys for payloads
  it owns or recently discarded.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.distributed import TCPStore

from alpagym_runtime.transport.nccl.comm_init import (
    CommInitConfig,
    compute_nccl_topology,
    init_communicator,
    safe_abort,
)
from alpagym_runtime.transport.nccl.protocol import (
    NCCL_RENDEZVOUS_ACCEPTED,
    NCCL_RENDEZVOUS_CANCELLED,
    NCCL_RENDEZVOUS_MISSING,
    build_nccl_prefix,
    build_req_key,
    build_rollout_prefix,
    normalize_nccl_handle,
)
from alpagym_runtime.transport.nccl.rendezvous import AckRendezvous, SendDecision


@dataclass(frozen=True)
class NcclSenderConfig:
    """Tunables for :class:`NcclSender`."""

    max_pending_transfers: int = 128
    sender_poll_interval_seconds: float = 0.05
    released_transfer_retention_seconds: float = 30.0
    comm_init: CommInitConfig = field(default_factory=CommInitConfig)


def assign_rollout_idx(
    experiment_name: str,
    job_id: str,
    num_rollout_replicas: int,
    store: TCPStore,
) -> int:
    """Acquire the next rollout-replica id from the shared TCPStore counter."""
    prefix = build_nccl_prefix(experiment_name=experiment_name, job_id=job_id)
    counter_key = f"{prefix}:rollout_replica_counter"
    # Each rollout worker calls this once at startup, so a correct launch has
    # exactly num_rollout_replicas callers, each claiming one id. The
    # decrement-and-raise below guards against an over-launch (more workers than
    # replicas); it is not a race-free slot allocator.
    rollout_idx = int(store.add(counter_key, 1)) - 1
    if rollout_idx >= num_rollout_replicas:
        store.add(counter_key, -1)
        raise RuntimeError(
            f"Assigned rollout_idx {rollout_idx} exceeds rollout replica count "
            f"{num_rollout_replicas}."
        )
    return rollout_idx


class NcclSender:
    """Producer-side of the NCCL transport (one per rollout worker)."""

    def __init__(
        self,
        experiment_name: str,
        job_id: str,
        rollout_idx: int,
        num_policy_replicas: int,
        dp_shard_size: int,
        store: TCPStore,
        rendezvous: AckRendezvous,
        create_nccl_uid: Callable[[], list[int]],
        create_nccl_comm: Callable[..., int],
        nccl_send: Callable[..., None],
        nccl_abort: Callable[[int], Any],
        config: NcclSenderConfig | None = None,
    ):
        """Wire the sender; ``rollout_idx`` is this worker's stable replica id."""
        self._num_policy_processes, self._comm_world_size = compute_nccl_topology(
            num_policy_replicas=num_policy_replicas, dp_shard_size=dp_shard_size
        )
        self._rollout_idx = rollout_idx
        self._store = store
        self._rendezvous = rendezvous
        self._create_nccl_uid = create_nccl_uid
        self._create_nccl_comm = create_nccl_comm
        self._nccl_send = nccl_send
        self._nccl_abort = nccl_abort
        self._config = config or NcclSenderConfig()
        self._prefix = build_nccl_prefix(experiment_name=experiment_name, job_id=job_id)
        self._rollout_prefix = build_rollout_prefix(self._prefix, self._rollout_idx)
        self._logger = logging.getLogger(__name__).getChild(f"sender.{self._rollout_idx}")
        self._comm_idx: int | None = None
        self._pending_sends: dict[str, dict[str, torch.Tensor]] = {}
        self._claimed_sends: dict[str, dict[str, torch.Tensor]] = {}
        self._released_transfers: dict[str, float] = {}
        self._pending_sends_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._device_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        device_name = f"cuda:{self._device_idx}" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_name)

    @property
    def rollout_idx(self) -> int:
        """Rollout-replica id this sender services."""
        return self._rollout_idx

    @property
    def comm_idx(self) -> int | None:
        """NCCL communicator index (None until :meth:`setup` returns)."""
        return self._comm_idx

    def setup(self) -> None:
        """Bootstrap the communicator and start the sender thread."""
        _, comm_idx = init_communicator(
            role="Rollout",
            prefix=self._rollout_prefix,
            world_size=self._comm_world_size,
            num_policy_processes=self._num_policy_processes,
            store=self._store,
            create_nccl_uid=self._create_nccl_uid,
            create_nccl_comm=self._create_nccl_comm,
            logger=self._logger,
            config=self._config.comm_init,
        )
        self._comm_idx = comm_idx
        self._logger.info(
            "Bootstrapped communicator (rollout_idx=%s, comm_idx=%s)",
            self._rollout_idx,
            comm_idx,
        )
        self._worker_thread = threading.Thread(target=self._run_sender_loop, daemon=True)
        self._worker_thread.start()

    def send(self, tensors: dict[str, torch.Tensor], transfer_id: str) -> None:
        """Register ``tensors`` under ``transfer_id`` so the sender loop can ship them.

        Raises if NCCL isn't initialized or if the pending registry is full.
        """
        max_pending = self._config.max_pending_transfers
        with self._pending_sends_lock:
            if self._comm_idx is None:
                raise RuntimeError("NCCL not initialized; call setup() first")
            if (
                max_pending > 0
                and transfer_id not in self._pending_sends
                and len(self._pending_sends) >= max_pending
            ):
                raise RuntimeError(
                    "Pending NCCL send registry is full; refusing to evict a live payload."
                )
            self._pending_sends[transfer_id] = tensors
            self._released_transfers.pop(transfer_id, None)
            pending_count = len(self._pending_sends)
        self._logger.debug(
            "state=registered transfer_id=%s... tensors=%s pending=%s",
            transfer_id[:16],
            len(tensors),
            pending_count,
        )

    def wait_until_drained(self, timeout_seconds: float) -> bool:
        """Block until no send is in flight, or ``timeout_seconds`` elapses.

        Only ``_claimed_sends`` is running an NCCL op: a claimed transfer is
        being shipped by the background loop right now. Held ``_pending_sends``
        are inert tensors awaiting a receiver request, so they are left in place
        and ship after the R2R weight broadcast. A held send cannot be claimed
        mid-broadcast: the policy publishes receive requests only synchronously
        inside ``get_policy_input``, never during the weight-sync phase that R2R
        belongs to, so the poll loop has no request to claim while R2R runs (see
        the transport README's "Coexistence with weight sync"). Returns whether
        nothing was in flight when it returned; the NCCL writer's
        ``flush_pending_sends`` treats a ``False`` return as fail-fast and raises
        before weight sync.
        """
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            with self._pending_sends_lock:
                if not self._claimed_sends:
                    return True
            time.sleep(self._config.sender_poll_interval_seconds)
        with self._pending_sends_lock:
            return not self._claimed_sends

    def close(self) -> None:
        """Stop the sender thread and abort the NCCL communicator."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            if self._worker_thread.is_alive():
                self._logger.warning("Sender thread did not stop within close() timeout")
        with self._pending_sends_lock:
            self._pending_sends.clear()
            self._claimed_sends.clear()
            self._released_transfers.clear()
        if self._comm_idx is not None:
            safe_abort(
                nccl_abort=self._nccl_abort,
                comm_idx=self._comm_idx,
                logger=self._logger,
                context="close",
            )

    def _run_sender_loop(self) -> None:
        """Background thread: poll content-addressed request keys.

        The discovery set is local and bounded: transfers whose tensors are
        currently held, plus recently discarded transfers that may have a
        receiver between manifest-read and request-publish. Rendezvous state in
        TCPStore remains keyed by transfer id.
        """
        if self._comm_idx is None:
            return
        if torch.cuda.is_available():
            torch.cuda.set_device(self._device_idx)
        # R2P sends run on a dedicated stream to keep them off the rollout worker's
        # main inference stream. The stream alone does not make concurrent NCCL comms
        # safe against R2R broadcasts: cosmos drains this sender via flush_pending_sends()
        # (wait_until_drained) before each R2R, and that drain -- not the stream -- is
        # what serializes the two communicators and avoids a deadlock.
        stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self._logger.info(
            "Sender loop polling transfer-id request keys under %s",
            self._rollout_prefix,
        )
        while not self._stop_event.is_set():
            time.sleep(self._config.sender_poll_interval_seconds)
            self._gc_released_transfers()
            for transfer_id in self._request_scan_ids():
                if not self._store.check([build_req_key(self._rollout_prefix, transfer_id)]):
                    continue
                decision: SendDecision | None = None
                try:
                    decision = self._rendezvous.process_request(
                        prefix=self._rollout_prefix,
                        rollout_idx=self._rollout_idx,
                        transfer_id=transfer_id,
                        store=self._store,
                        claim_transfer_snapshot=self._claim_transfer_snapshot,
                        logger=self._logger,
                    )
                except Exception as error:
                    self._drop_claimed_transfer(transfer_id)
                    self._logger.error(
                        "Error in rendezvous for transfer_id=%s: %s", transfer_id, error
                    )
                if (
                    decision is not None
                    and decision.status == NCCL_RENDEZVOUS_ACCEPTED
                    and decision.tensors_map is not None
                ):
                    try:
                        self._send_registered_transfer(decision, stream=stream)
                        self._complete_claimed_transfer(decision.transfer_id)
                    except Exception as send_error:
                        self._handle_post_accept_send_failure(
                            transfer_id=decision.transfer_id, error=send_error
                        )
                elif decision is not None and decision.status == NCCL_RENDEZVOUS_CANCELLED:
                    self._drop_claimed_transfer(decision.transfer_id)
                    self._drop_released_marker(decision.transfer_id)
                elif decision is not None and decision.status == NCCL_RENDEZVOUS_MISSING:
                    self._drop_released_marker(decision.transfer_id)

    def _request_scan_ids(self) -> list[str]:
        """Return transfer ids whose request keys this sender can answer."""
        with self._pending_sends_lock:
            transfer_ids = set(self._pending_sends)
            transfer_ids.update(self._released_transfers)
        return sorted(transfer_ids)

    def _gc_released_transfers(self) -> None:
        """Drop old discard markers after their manifest-to-request window has elapsed."""
        retention = self._config.released_transfer_retention_seconds
        if retention < 0:
            return
        cutoff = time.monotonic() - retention
        with self._pending_sends_lock:
            expired = [
                transfer_id
                for transfer_id, released_at in self._released_transfers.items()
                if released_at < cutoff
            ]
            for transfer_id in expired:
                self._released_transfers.pop(transfer_id, None)

    def _claim_transfer_snapshot(
        self,
        transfer_id: str,
    ) -> tuple[dict[str, torch.Tensor] | None, list[str], int]:
        """Move a held transfer into claimed state and return its tensor snapshot.

        A discard release can only win while the transfer is still held. Once this
        method claims the tensor map, external release calls leave sender tensor
        ownership unchanged until success or post-accept failure drops it.
        """
        with self._pending_sends_lock:
            pending_preview = list(self._pending_sends.keys())[:3]
            pending_count = len(self._pending_sends)
            if self._comm_idx is None:
                return None, pending_preview, pending_count
            tensors_map = self._pending_sends.pop(transfer_id, None)
            if tensors_map is None:
                return None, pending_preview, pending_count
            self._claimed_sends[transfer_id] = tensors_map
            return dict(tensors_map), pending_preview, pending_count

    def _complete_claimed_transfer(self, transfer_id: str) -> None:
        """Drop one claimed transfer after its NCCL send has completed."""
        with self._pending_sends_lock:
            self._claimed_sends.pop(transfer_id, None)
            pending_count = len(self._pending_sends)
        self._logger.info(
            "state=sent transfer_id=%s... reason=read_complete pending=%s",
            transfer_id[:16],
            pending_count,
        )

    def _drop_claimed_transfer(self, transfer_id: str) -> None:
        """Drop a claimed transfer when the sender will not complete a send."""
        with self._pending_sends_lock:
            self._claimed_sends.pop(transfer_id, None)

    def _drop_released_marker(self, transfer_id: str) -> None:
        """Drop a discard marker after it has produced a terminal negative ACK."""
        with self._pending_sends_lock:
            self._released_transfers.pop(transfer_id, None)

    def _handle_post_accept_send_failure(
        self,
        transfer_id: str,
        error: BaseException,
    ) -> None:
        """Abort the comm + drop the transfer after a failed post-accept send.

        The rendezvous already accepted the transfer, so the receiver is
        parked inside ``nccl_recv`` expecting the full tensor batch. A
        partial ``nccl_send`` leaves NCCL in an undefined state on this
        communicator. Abort it so the receiver's ``nccl_recv`` raises
        immediately rather than waiting out its watchdog timeout, and
        clear ``self._comm_idx`` so subsequent ``send()`` calls fail
        fast at the registry-not-initialized check rather than
        producing more partial batches on a dead comm.
        """
        self._logger.error(
            "state=poisoned: post-accept NCCL send failure for transfer_id=%s; "
            "aborting comm_idx=%s: %s",
            transfer_id,
            self._comm_idx,
            error,
        )
        with self._pending_sends_lock:
            self._pending_sends.clear()
            self._claimed_sends.clear()
            self._released_transfers.clear()
            comm_idx = self._comm_idx
            self._comm_idx = None
        if comm_idx is None:
            return
        safe_abort(
            nccl_abort=self._nccl_abort,
            comm_idx=comm_idx,
            logger=self._logger,
            context="after send failure",
        )

    def _send_registered_transfer(
        self,
        decision: SendDecision,
        stream: Any,
    ) -> None:
        """Issue one ``nccl_send`` per tensor in the accepted snapshot."""
        assert self._comm_idx is not None
        assert decision.tensors_map is not None
        self._logger.debug(
            "Sending %s... to rank %s", decision.transfer_id[:16], decision.requester_rank
        )
        if stream is not None:
            with torch.cuda.stream(stream):
                for key in sorted(decision.tensors_map.keys()):
                    tensor = decision.tensors_map[key]
                    if not tensor.is_cuda:
                        tensor = tensor.to(self._device)
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                    self._nccl_send(tensor, decision.requester_rank, self._comm_idx, stream=stream)
            stream.synchronize()
        else:
            for key in sorted(decision.tensors_map.keys()):
                tensor = decision.tensors_map[key]
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                self._nccl_send(tensor, decision.requester_rank, self._comm_idx)

    def release(self, transfer_id: str, reason: str) -> bool:
        """Drop ``transfer_id`` from the pending registry; return whether it was present.

        Single public release path for both the sender's own post-send cleanup
        and external callers — write-side rollback when metadata publish fails,
        and controller-driven stale-discard release. Idempotent: releasing an
        unknown transfer is a no-op that returns ``False``.
        """
        transfer_id = normalize_nccl_handle(transfer_id)
        with self._pending_sends_lock:
            was_held = transfer_id in self._pending_sends
            was_claimed = transfer_id in self._claimed_sends
            if was_held:
                self._pending_sends.pop(transfer_id, None)
                if _release_needs_discard_marker(reason):
                    self._released_transfers[transfer_id] = time.monotonic()
            pending_count = len(self._pending_sends)
        self._logger.info(
            "state=released transfer_id=%s... reason=%s was_present=%s pending=%s",
            transfer_id[:16],
            reason,
            was_held or was_claimed,
            pending_count,
        )
        return was_held or was_claimed


def _release_needs_discard_marker(reason: str) -> bool:
    """Return whether a release reason can race a receiver that already read metadata."""
    return reason not in {"read_complete", "metadata_publish_error", "rendezvous_cancelled"}
