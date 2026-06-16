# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the NCCL protocol wire format and TCPStore key builders."""

import pytest
from alpagym_runtime.transport.nccl.protocol import (
    NCCL_RENDEZVOUS_REQUESTED,
    NcclTransferAck,
    NcclTransferRequest,
    build_ack_key,
    build_metadata_key,
    build_nccl_prefix,
    build_req_key,
    build_rollout_prefix,
    build_state_key,
    normalize_nccl_handle,
    parse_transfer_rollout_idx,
    to_external_nccl_handle,
)


def test_protocol_key_builders_match_current_contract() -> None:
    """Every TCPStore key builder emits the agreed-upon string format.

    Derive the prefix from ``build_nccl_prefix`` (the builder that stamps the
    cross-process ``NCCL_NAMESPACE``) so a namespace change surfaces here instead of
    silently making sender and receiver keys disagree.
    """
    prefix = build_nccl_prefix(experiment_name="exp", job_id="job")
    assert prefix == "alpagym:exp:job"
    rollout_prefix = build_rollout_prefix(prefix, rollout_idx=3)
    assert rollout_prefix == "alpagym:exp:job:rollout_comm:3"
    assert build_req_key(prefix, "7:abc") == "alpagym:exp:job:nccl_req:7:abc"
    assert build_state_key(prefix, "7:abc") == "alpagym:exp:job:nccl_state:7:abc"
    assert build_ack_key(prefix, "7:abc") == "alpagym:exp:job:nccl_ack:7:abc"
    assert build_metadata_key("7:abc") == "nccl_meta:7:abc"
    assert build_req_key(rollout_prefix, "7:abc") == (
        "alpagym:exp:job:rollout_comm:3:nccl_req:7:abc"
    )


def test_transfer_request_roundtrip() -> None:
    """NcclTransferRequest.to_json / from_json round-trip preserves every field."""
    request = NcclTransferRequest(
        request_id="req-1",
        transfer_id="3:payload",
        requester_rank=1,
        rollout_idx=3,
        deadline_ms=99_999,
        state=NCCL_RENDEZVOUS_REQUESTED,
    )
    parsed = NcclTransferRequest.from_json(request.to_json())
    assert parsed == request


def test_transfer_ack_rejects_negative_accepted_at_ms() -> None:
    """ACK parsing rejects invalid accepted timestamps instead of coercing to None."""
    ack = NcclTransferAck(
        request_id="req-1",
        transfer_id="3:payload",
        status="accepted",
        message="ok",
        pending_count=1,
        accepted_at_ms=-1,
    )
    with pytest.raises(ValueError, match="accepted_at_ms must be non-negative"):
        NcclTransferAck.from_json(ack.to_json())


def test_parse_transfer_rollout_idx_raises_on_bad_payloads() -> None:
    """parse_transfer_rollout_idx parses the prefix and raises on malformed ids."""
    assert parse_transfer_rollout_idx("9:payload") == 9
    assert parse_transfer_rollout_idx("nccl:9:payload") == 9
    with pytest.raises(ValueError):
        parse_transfer_rollout_idx("bad-format")
    with pytest.raises(ValueError):
        parse_transfer_rollout_idx("x:payload")


def test_nccl_external_handle_helpers_are_idempotent() -> None:
    """External handles carry Cosmos's nccl: prefix while internals use raw ids."""
    assert to_external_nccl_handle("1:payload") == "nccl:1:payload"
    assert to_external_nccl_handle("nccl:1:payload") == "nccl:1:payload"
    assert normalize_nccl_handle("nccl:1:payload") == "1:payload"
    assert normalize_nccl_handle("1:payload") == "1:payload"
