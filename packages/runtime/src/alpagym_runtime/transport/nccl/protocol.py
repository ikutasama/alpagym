# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TCPStore-side wire format for the NCCL transport.

This module owns three things:

- The rendezvous state constants (``NCCL_RENDEZVOUS_REQUESTED`` etc.).
- The per-transfer protocol payloads
  (:class:`NcclTransferRequest`, :class:`NcclTransferAck`).
- The TCPStore key builders for the rendezvous control plane and the transfer
  manifest.

``NCCL_NAMESPACE`` lives here as a module-level constant because its value
must agree across producer and consumer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from cosmos_rl.utils.payload_transport.nccl import NCCL_COMPLETION_PREFIX

NCCL_NAMESPACE = "alpagym"
NCCL_RENDEZVOUS_REQUESTED = "requested"
NCCL_RENDEZVOUS_ACCEPTED = "accepted"
NCCL_RENDEZVOUS_MISSING = "missing"
NCCL_RENDEZVOUS_CANCELLED = "cancelled"


def build_nccl_prefix(experiment_name: str, job_id: str) -> str:
    """Return the store key prefix that scopes this experiment + job."""
    return f"{NCCL_NAMESPACE}:{experiment_name}:{job_id}"


def build_rollout_prefix(prefix: str, rollout_idx: int) -> str:
    """Return the per-rollout-replica subprefix under ``prefix``."""
    return f"{prefix}:rollout_comm:{rollout_idx}"


def build_req_key(prefix: str, transfer_id: str) -> str:
    """TCPStore key holding the request payload for ``transfer_id``."""
    return f"{prefix}:nccl_req:{transfer_id}"


def build_state_key(prefix: str, transfer_id: str) -> str:
    """TCPStore key holding the current rendezvous state for ``transfer_id``."""
    return f"{prefix}:nccl_state:{transfer_id}"


def build_ack_key(prefix: str, transfer_id: str) -> str:
    """TCPStore key holding the sender's ack payload for ``transfer_id``."""
    return f"{prefix}:nccl_ack:{transfer_id}"


def build_metadata_key(transfer_id: str) -> str:
    """TCPStore key holding the reconstruction manifest for ``transfer_id``."""
    return f"nccl_meta:{transfer_id}"


def to_external_nccl_handle(transfer_id: str) -> str:
    """Return the Cosmos-visible handle for a raw AlpaGym transfer id."""
    if transfer_id.startswith(NCCL_COMPLETION_PREFIX):
        return transfer_id
    return f"{NCCL_COMPLETION_PREFIX}{transfer_id}"


def normalize_nccl_handle(handle: str) -> str:
    """Strip the Cosmos NCCL completion prefix from ``handle`` if present."""
    if handle.startswith(NCCL_COMPLETION_PREFIX):
        return handle[len(NCCL_COMPLETION_PREFIX) :]
    return handle


def parse_transfer_rollout_idx(transfer_id: str) -> int:
    """Parse the rollout-idx prefix encoded in ``transfer_id``.

    Raises ``ValueError`` when ``transfer_id`` carries no integer
    ``<rollout_idx>:`` prefix; callers treat a malformed id as fatal.
    """
    prefix = normalize_nccl_handle(transfer_id).split(":", maxsplit=1)[0]
    return int(prefix)


@dataclass(frozen=True)
class NcclTransferRequest:
    """Receiver-published request payload stored at ``build_req_key(prefix, transfer_id)``."""

    request_id: str
    transfer_id: str
    requester_rank: int
    rollout_idx: int
    deadline_ms: int
    state: str

    def to_json(self) -> str:
        """Serialize this request payload as JSON for store storage."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray | dict[str, Any]) -> NcclTransferRequest:
        """Parse one stored request payload back into a typed dataclass."""
        data = _load_json_payload(payload)
        return cls(
            request_id=str(data["request_id"]),
            transfer_id=str(data["transfer_id"]),
            requester_rank=int(data["requester_rank"]),
            rollout_idx=int(data["rollout_idx"]),
            deadline_ms=int(data["deadline_ms"]),
            state=str(data["state"]),
        )


@dataclass(frozen=True)
class NcclTransferAck:
    """Sender-published ack payload stored at ``build_ack_key(prefix, transfer_id)``."""

    request_id: str
    transfer_id: str
    status: str
    message: str
    pending_count: int
    accepted_at_ms: int | None = None

    def to_json(self) -> str:
        """Serialize this ack payload as JSON for store storage."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray | dict[str, Any]) -> NcclTransferAck:
        """Parse one stored ack payload back into a typed dataclass."""
        data = _load_json_payload(payload)
        accepted_at_ms = None
        if data.get("accepted_at_ms") is not None:
            accepted_at_ms = int(data["accepted_at_ms"])
            if accepted_at_ms < 0:
                raise ValueError("accepted_at_ms must be non-negative")
        return cls(
            request_id=str(data["request_id"]),
            transfer_id=str(data["transfer_id"]),
            status=str(data["status"]),
            message=str(data["message"]),
            pending_count=int(data["pending_count"]),
            accepted_at_ms=accepted_at_ms,
        )


def _load_json_payload(
    payload: str | bytes | bytearray | dict[str, Any],
) -> dict[str, Any]:
    """Coerce a payload into a dict; accept pre-parsed dicts or JSON bytes/str."""
    if isinstance(payload, dict):
        return payload
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("NCCL protocol payload must be a JSON object")
    return data
