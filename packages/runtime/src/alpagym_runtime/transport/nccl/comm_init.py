# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot NCCL communicator bootstrap across peers.

``init_communicator`` runs four steps on each process:

1. Assign a unique NCCL rank from a ``TCPStore`` counter. Policy and rollout
   roles use separate counters, so the combined rank ordering is stable.
2. Rank 0 generates the NCCL UID and writes it to the store. Other ranks
   call ``store.wait`` until the UID appears, then read it.
3. Every rank writes a ready marker. All ranks wait until the full ready-key
   set is published.
4. Call ``create_nccl_comm`` to obtain the communicator index.

All coordination state lives under a caller-provided ``prefix`` so multiple
experiments can share one ``TCPStore`` without colliding.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from torch.distributed import TCPStore


@dataclass(frozen=True)
class CommInitConfig:
    """Coordination-policy values consumed by ``init_communicator``."""

    barrier_wait_timeout_seconds: float = 600.0
    communicator_timeout_ms: int = 600000


def compute_nccl_topology(num_policy_replicas: int, dp_shard_size: int) -> tuple[int, int]:
    """Return ``(num_policy_processes, comm_world_size)`` for one sender + N policy peers.

    Each NCCL communicator joins one rollout-worker process with
    ``num_policy_replicas * dp_shard_size`` policy processes, hence the
    ``+ 1`` for the sender rank in the world size.

    Args:
        num_policy_replicas: number of policy replicas in the job.
        dp_shard_size: DP shard size per replica.

    Raises:
        ValueError: if ``num_policy_replicas`` or ``dp_shard_size`` is not positive.
    """
    if num_policy_replicas <= 0:
        raise ValueError(f"num_policy_replicas must be positive, got {num_policy_replicas}")
    if dp_shard_size <= 0:
        raise ValueError(f"dp_shard_size must be positive, got {dp_shard_size}")
    num_policy_processes = num_policy_replicas * dp_shard_size
    return num_policy_processes, 1 + num_policy_processes


def safe_abort(
    nccl_abort: Callable[[int], Any],
    comm_idx: int,
    logger: logging.Logger,
    context: str = "",
) -> None:
    """Abort an NCCL communicator; log a warning on failure and swallow the exception.

    Used on every teardown and post-accept failure path so a dead communicator
    never blocks shutdown.

    Args:
        nccl_abort: cosmos-rl ``pynccl`` abort entry point.
        comm_idx: communicator index to abort.
        logger: caller's logger.
        context: optional descriptor added to the warning (e.g. ``"after send failure"``).
    """
    try:
        nccl_abort(comm_idx)
    except Exception as error:
        if context:
            logger.warning("Failed to abort NCCL comm %s (%s): %s", comm_idx, context, error)
        else:
            logger.warning("Failed to abort NCCL comm %s: %s", comm_idx, error)


def init_communicator(
    role: str,
    prefix: str,
    world_size: int,
    num_policy_processes: int,
    store: TCPStore,
    create_nccl_uid: Callable[[], list[int]],
    create_nccl_comm: Callable[..., int],
    logger: logging.Logger,
    config: CommInitConfig = CommInitConfig(),
) -> tuple[int, int]:
    """Bootstrap a NCCL communicator. Returns ``(rank, comm_idx)``.

    Args:
        role: ``"Rollout"`` or ``"Policy"``; selects which counter assigns
            this process's rank.
        prefix: store key prefix scoping this experiment + job + rollout.
        world_size: total number of processes joining this communicator.
        num_policy_processes: offsets rollout-side ranks so the combined
            namespace is contiguous (policy ranks 0..N-1; rollouts N..N+R-1).
        store: a connected ``torch.distributed.TCPStore`` shared by all peers.
        create_nccl_uid: cosmos-rl ``pynccl`` entry point.
        create_nccl_comm: cosmos-rl ``pynccl`` entry point.
        logger: caller's logger.
        config: timeouts.

    Raises:
        RuntimeError: on rank-assignment failure, UID timeout, barrier
            timeout, or communicator creation failure.
    """
    if world_size <= 0:
        raise RuntimeError(f"[{role}] NCCL communicator world_size must be positive")
    rank = assign_rank_once(
        role=role,
        prefix=prefix,
        world_size=world_size,
        num_policy_processes=num_policy_processes,
        store=store,
        logger=logger,
    )
    uid = _get_or_create_uid(
        role=role,
        rank=rank,
        prefix=prefix,
        store=store,
        create_nccl_uid=create_nccl_uid,
        logger=logger,
        config=config,
    )
    _wait_for_all_ranks_ready(
        role=role,
        prefix=prefix,
        rank=rank,
        world_size=world_size,
        store=store,
        logger=logger,
        config=config,
    )
    logger.info(
        "[%s] Creating NCCL communicator (rank=%s, size=%s)...",
        role,
        rank,
        world_size,
    )
    comm_idx = create_nccl_comm(
        uid,
        rank,
        world_size,
        timeout_ms=config.communicator_timeout_ms,
    )
    logger.info("[%s] NCCL communicator created: idx=%s", role, comm_idx)
    return rank, comm_idx


def assign_rank_once(
    role: str,
    prefix: str,
    world_size: int,
    num_policy_processes: int,
    store: TCPStore,
    logger: logging.Logger,
) -> int:
    """Assign one NCCL rank from the role-specific store counter.

    Policy ranks come from ``policy_rank_counter`` (0-indexed); rollout ranks
    come from ``rollout_rank_counter`` with ``num_policy_processes`` added so
    the combined namespace is dense.

    Args:
        role: ``"Policy"`` or ``"Rollout"``; selects the rank counter.
        prefix: store key prefix scoping this experiment + job + rollout.
        world_size: total processes joining this communicator.
        num_policy_processes: offset added to rollout ranks for a dense namespace.
        store: connected ``torch.distributed.TCPStore`` shared by all peers.
        logger: caller's logger.
    """
    if role == "Policy":
        counter_key = _policy_rank_counter_key(prefix)
        rank_idx = store.add(counter_key, 1)
        rank = int(rank_idx) - 1
        if rank >= num_policy_processes:
            store.add(counter_key, -1)
            message = (
                f"Assigned Policy rank {rank} exceeds policy process count {num_policy_processes}."
            )
            logger.error("[%s] %s", role, message)
            raise RuntimeError(message)
    elif role == "Rollout":
        counter_key = _rollout_rank_counter_key(prefix)
        rank_idx = store.add(counter_key, 1)
        rank = num_policy_processes + int(rank_idx) - 1
    else:
        raise RuntimeError(f"Unknown role {role}, NCCL disabled.")
    if rank >= world_size:
        store.add(counter_key, -1)
        message = f"Assigned rank {rank} exceeds world_size {world_size}."
        logger.error("[%s] %s", role, message)
        raise RuntimeError(message)
    return rank


def _get_or_create_uid(
    role: str,
    rank: int,
    prefix: str,
    store: TCPStore,
    create_nccl_uid: Callable[[], list[int]],
    logger: logging.Logger,
    config: CommInitConfig,
) -> list[int]:
    """Rank-0 creates the NCCL UID and stores it; everyone else waits + reads."""
    uid_key = _uid_key(prefix)
    if rank == 0:
        logger.info("[%s] Creating NCCL UID...", role)
        uid = create_nccl_uid()
        store.set(uid_key, ",".join(str(value) for value in uid))
        logger.info("[%s] Wrote NCCL UID to store key %s", role, uid_key)
        return uid
    logger.info("[%s] Waiting for NCCL UID at store key %s...", role, uid_key)
    try:
        store.wait([uid_key], timedelta(seconds=config.barrier_wait_timeout_seconds))
    except RuntimeError as error:
        raise RuntimeError(
            f"[{role}] Timed out waiting for NCCL UID at store key {uid_key} "
            f"after {config.barrier_wait_timeout_seconds}s."
        ) from error
    uid_value = store.get(uid_key)
    uid_str = uid_value.decode("utf-8")
    logger.info("[%s] Read NCCL UID", role)
    return [int(value) for value in uid_str.split(",")]


def _wait_for_all_ranks_ready(
    role: str,
    prefix: str,
    rank: int,
    world_size: int,
    store: TCPStore,
    logger: logging.Logger,
    config: CommInitConfig,
) -> None:
    """Block until every rank has written its ready marker, or raise on timeout.

    Each rank writes ``ready:{rank}`` as a presence sentinel; everyone calls
    ``store.wait`` on the full ready-key set.
    """
    own_ready_key = _ready_key(prefix, rank)
    writer_count_key = f"{own_ready_key}:writers"
    writer_count = int(store.add(writer_count_key, 1))
    if writer_count != 1:
        store.add(writer_count_key, -1)
        raise RuntimeError(f"[{role}] Ready marker for rank {rank} was already written")
    store.set(own_ready_key, "1")
    logger.info("[%s] Rank %s ready, waiting for all %s ranks...", role, rank, world_size)
    all_ready_keys = [_ready_key(prefix, current_rank) for current_rank in range(world_size)]
    try:
        store.wait(all_ready_keys, timedelta(seconds=config.barrier_wait_timeout_seconds))
    except RuntimeError as error:
        missing_ranks = [
            current_rank
            for current_rank in range(world_size)
            if not store.check([_ready_key(prefix, current_rank)])
        ]
        logger.error(
            "[%s] Timeout waiting for all ranks to be ready. Missing ranks: %s",
            role,
            missing_ranks,
        )
        raise RuntimeError(
            f"[{role}] Timeout waiting for all ranks to be ready. Missing ranks: {missing_ranks}"
        ) from error
    logger.info("[%s] All %s ranks ready. Proceeding to create NCCL comm...", role, world_size)


def _policy_rank_counter_key(prefix: str) -> str:
    """Store key for the policy-side rank counter under ``prefix``."""
    return f"{prefix}:policy_rank_counter"


def _rollout_rank_counter_key(prefix: str) -> str:
    """Store key for the rollout-side rank counter under ``prefix``."""
    return f"{prefix}:rollout_rank_counter"


def _uid_key(prefix: str) -> str:
    """Store key where rank-0 publishes the NCCL UID."""
    return f"{prefix}:nccl_uid"


def _ready_key(prefix: str, rank: int) -> str:
    """Store key where ``rank`` writes its ready marker."""
    return f"{prefix}:nccl_ready:{rank}"
