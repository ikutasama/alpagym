# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NCCL transport consumer side, backed by ``torch.distributed.TCPStore``.

``NcclReceiver`` owns one NCCL communicator per rollout-replica it
expects to receive from. ``recv`` blocks on the injected
``AckRendezvous`` until the sender claims the transfer, then issues
``nccl_recv`` per tensor in the caller-provided ``tensor_specs``. A
``threading.Event`` watchdog turns a hung NCCL call into an observable
``TimeoutError`` after the configured deadline.
"""

import logging
import threading
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
    build_nccl_prefix,
    build_rollout_prefix,
    parse_transfer_rollout_idx,
)
from alpagym_runtime.transport.nccl.rendezvous import AckRendezvous


@dataclass(frozen=True)
class NcclReceiverConfig:
    """Tunables for ``NcclReceiver``."""

    recv_timeout_seconds: float = 120.0
    comm_init: CommInitConfig = field(default_factory=CommInitConfig)


class NcclReceiver:
    """Consumer-side of the NCCL transport (one per trainer worker)."""

    def __init__(
        self,
        experiment_name: str,
        job_id: str,
        num_policy_replicas: int,
        dp_shard_size: int,
        num_rollout_replicas: int,
        store: TCPStore,
        rendezvous: AckRendezvous,
        create_nccl_uid: Callable[[], list[int]],
        create_nccl_comm: Callable[..., int],
        nccl_recv: Callable[..., None],
        nccl_abort: Callable[[int], Any],
        config: NcclReceiverConfig | None = None,
    ):
        """Wire the receiver; call ``setup`` before ``recv``."""
        self._num_policy_processes, self._comm_world_size = compute_nccl_topology(
            num_policy_replicas=num_policy_replicas, dp_shard_size=dp_shard_size
        )
        self._num_rollout_replicas = num_rollout_replicas
        self._sender_rank = self._num_policy_processes
        self._store = store
        self._rendezvous = rendezvous
        self._create_nccl_uid = create_nccl_uid
        self._create_nccl_comm = create_nccl_comm
        self._nccl_recv = nccl_recv
        self._nccl_abort = nccl_abort
        self._config = config or NcclReceiverConfig()
        self._prefix = build_nccl_prefix(experiment_name=experiment_name, job_id=job_id)
        self._logger = logging.getLogger(__name__).getChild("receiver")
        self._rollout_comms: dict[int, int] = {}
        self._policy_ranks: dict[int, int] = {}
        self._rollout_prefixes: dict[int, str] = {}

    @property
    def rollout_comms(self) -> dict[int, int]:
        """Map rollout-replica id -> NCCL communicator index."""
        return dict(self._rollout_comms)

    def setup(self) -> None:
        """Bootstrap one communicator per rollout replica we expect to receive from."""
        try:
            for rollout_idx in range(self._num_rollout_replicas):
                rollout_prefix = build_rollout_prefix(self._prefix, rollout_idx)
                rank, comm_idx = init_communicator(
                    role="Policy",
                    prefix=rollout_prefix,
                    world_size=self._comm_world_size,
                    num_policy_processes=self._num_policy_processes,
                    store=self._store,
                    create_nccl_uid=self._create_nccl_uid,
                    create_nccl_comm=self._create_nccl_comm,
                    logger=self._logger,
                    config=self._config.comm_init,
                )
                self._rollout_comms[rollout_idx] = comm_idx
                self._policy_ranks[rollout_idx] = rank
                self._rollout_prefixes[rollout_idx] = rollout_prefix
        except Exception:
            self._abort_all_communicators()
            self._rollout_comms.clear()
            self._policy_ranks.clear()
            self._rollout_prefixes.clear()
            raise
        self._logger.info("Receiver ready with rollout communicators: %s", self._rollout_comms)

    def recv(
        self,
        transfer_id: str,
        tensor_specs: dict[str, tuple[tuple[int, ...], str]],
    ) -> dict[str, torch.Tensor]:
        """Block on rendezvous, then receive every tensor in ``tensor_specs``."""
        resolved_rollout_idx = parse_transfer_rollout_idx(transfer_id)
        if not self._rollout_comms:
            raise RuntimeError("NCCL not initialized; call setup() first")
        comm_idx = self._rollout_comms.get(resolved_rollout_idx)
        policy_rank = self._policy_ranks.get(resolved_rollout_idx)
        rollout_prefix = self._rollout_prefixes.get(resolved_rollout_idx)
        if comm_idx is None or policy_rank is None or rollout_prefix is None:
            raise RuntimeError(
                f"NCCL communicator not initialized for rollout_idx={resolved_rollout_idx}"
            )
        self._rendezvous.request_ack(
            prefix=rollout_prefix,
            transfer_id=transfer_id,
            requester_rank=policy_rank,
            rollout_idx=resolved_rollout_idx,
            store=self._store,
            role="Policy",
            logger=self._logger,
        )
        try:
            return self._receive_with_watchdog(
                transfer_id=transfer_id,
                comm_idx=comm_idx,
                tensor_specs=tensor_specs,
            )
        except Exception:
            # Any post-accept recv failure poisons this communicator. Abort it
            # and drop routing state so the next recv on this rollout_idx fails
            # fast instead of reusing a dead NCCL comm.
            self._abort_and_drop_comm(resolved_rollout_idx)
            raise

    def close(self) -> None:
        """Abort every per-rollout NCCL communicator."""
        self._abort_all_communicators()

    def _receive_with_watchdog(
        self,
        transfer_id: str,
        comm_idx: int,
        tensor_specs: dict[str, tuple[tuple[int, ...], str]],
    ) -> dict[str, torch.Tensor]:
        """Run ``nccl_recv`` in a daemon thread guarded by a watchdog Event."""
        received_tensors: dict[str, torch.Tensor] = {}
        recv_error: list[Exception | None] = [None]
        recv_complete = threading.Event()
        caller_device_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        caller_device = (
            torch.device(f"cuda:{caller_device_idx}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        def do_nccl_recv() -> None:
            """Receive each expected tensor over NCCL on an isolated CUDA stream."""
            try:
                if torch.cuda.is_available():
                    torch.cuda.set_device(caller_device_idx)
                    stream = torch.cuda.Stream(device=caller_device)
                    with torch.cuda.stream(stream):
                        for key in sorted(tensor_specs.keys()):
                            shape, dtype_str = tensor_specs[key]
                            dtype = getattr(torch, dtype_str.split(".")[-1])
                            tensor = torch.empty(shape, dtype=dtype, device=caller_device)
                            self._nccl_recv(tensor, self._sender_rank, comm_idx, stream=stream)
                            received_tensors[key] = tensor
                    stream.synchronize()
                else:
                    for key in sorted(tensor_specs.keys()):
                        shape, dtype_str = tensor_specs[key]
                        dtype = getattr(torch, dtype_str.split(".")[-1])
                        tensor = torch.empty(shape, dtype=dtype, device=caller_device)
                        self._nccl_recv(tensor, self._sender_rank, comm_idx)
                        received_tensors[key] = tensor
                self._logger.info(
                    "NCCL recv completed for %s, received %s tensors",
                    transfer_id[:16],
                    len(received_tensors),
                )
            except Exception as error:
                recv_error[0] = error
                self._logger.error("NCCL recv error: %s", error)
            finally:
                recv_complete.set()

        recv_thread = threading.Thread(target=do_nccl_recv, daemon=True)
        recv_thread.start()
        if not recv_complete.wait(timeout=self._config.recv_timeout_seconds):
            raise TimeoutError(
                f"NCCL recv TIMEOUT after {self._config.recv_timeout_seconds}s! "
                f"transfer_id={transfer_id}, num_tensors={len(tensor_specs)}. "
                "Sender accepted rendezvous but NCCL did not complete."
            )
        if recv_error[0] is not None:
            raise recv_error[0]
        return received_tensors

    def _abort_all_communicators(self) -> None:
        """Abort every per-rollout NCCL communicator (idempotent)."""
        for comm_idx in sorted(set(self._rollout_comms.values())):
            safe_abort(
                nccl_abort=self._nccl_abort,
                comm_idx=comm_idx,
                logger=self._logger,
            )

    def _abort_and_drop_comm(self, rollout_idx: int) -> None:
        """Abort the per-rollout comm and drop it from the receiver's routing tables."""
        comm_idx = self._rollout_comms.pop(rollout_idx, None)
        self._policy_ranks.pop(rollout_idx, None)
        self._rollout_prefixes.pop(rollout_idx, None)
        if comm_idx is None:
            return
        safe_abort(
            nccl_abort=self._nccl_abort,
            comm_idx=comm_idx,
            logger=self._logger,
            context=f"after recv failure, rollout_idx={rollout_idx}",
        )
