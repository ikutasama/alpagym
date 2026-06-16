# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the NCCL rollout-transport endpoints (TCPStore-backed)."""

import json
import socket
import threading
import time
from typing import Any

import pytest
import torch
from alpagym_runtime.transport.base import EpisodeWriter
from alpagym_runtime.transport.nccl.endpoints import (
    NcclAlpagymDataPacker,
    NcclDataPackerMixin,
    NcclEpisodeWriter,
)
from alpagym_runtime.transport.nccl.protocol import build_metadata_key, normalize_nccl_handle
from alpagym_runtime.transport.nccl.receiver import NcclReceiver
from alpagym_runtime.types import EpisodeOutput, PolicyOutput
from cosmos_rl.utils.payload_transport.nccl import (
    build_cleanup_channel,
    build_nccl_prefix,
    build_rollout_prefix,
)
from torch.distributed import TCPStore


class _Resolver(NcclDataPackerMixin):
    """Trainer-side read path under test, isolated from replay collation.

    ``NcclDataPackerMixin._resolve_nccl_handle`` uses only ``_store``,
    ``_receiver``, and ``_target_device``; setting them directly exercises the
    manifest-resolve / recv / unpack path without the full packer.
    """

    def __init__(self, store: TCPStore, receiver: NcclReceiver) -> None:
        """Wire the resolver to a store and (fake) receiver on CPU."""
        self._store = store
        self._receiver = receiver
        self._target_device = torch.device("cpu")


class _RecordingPolicyBase:
    """Base packer double that records the episode handed to replay collation."""

    def __init__(self) -> None:
        """Initialize recorded call state."""
        self.seen_rollout_output: Any | None = None
        self.seen_n_ignore_prefix_tokens: int | None = None
        self.seen_kwargs: dict[str, Any] | None = None

    def get_policy_input(
        self,
        sample: Any,
        rollout_output: Any,
        n_ignore_prefix_tokens: int = 0,
        **kwargs: Any,
    ) -> list[str]:
        """Record the resolved rollout output and return a sentinel batch."""
        del sample
        self.seen_rollout_output = rollout_output
        self.seen_n_ignore_prefix_tokens = n_ignore_prefix_tokens
        self.seen_kwargs = kwargs
        return ["collated"]


class _SyncPolicyPacker(NcclDataPackerMixin, _RecordingPolicyBase):
    """NCCL policy-input path with a fake synchronous resolver."""

    def __init__(self, episode: EpisodeOutput) -> None:
        """Wire the fake resolver to a fixed episode."""
        super().__init__()
        self._episode = episode
        self.resolved_handles: list[str] = []

    def _resolve_nccl_handle(self, handle: str) -> EpisodeOutput:
        """Record the handle and return the fixed episode."""
        self.resolved_handles.append(handle)
        return self._episode


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


class _FakeSender:
    """Records send() and release() calls for the writer."""

    def __init__(self, rollout_idx: int = 0) -> None:
        self.rollout_idx = rollout_idx
        self.sent: list[tuple[str, dict[str, torch.Tensor]]] = []
        self.released: list[tuple[str, str]] = []
        self.closed = False
        self.close_count = 0
        self.drain_result = True
        self.flush_timeout: float | None = None

    def send(self, tensors: dict[str, torch.Tensor], transfer_id: str) -> None:
        self.sent.append((transfer_id, dict(tensors)))

    def release(self, transfer_id: str, reason: str) -> bool:
        self.released.append((transfer_id, reason))
        return True

    def wait_until_drained(self, timeout_seconds: float) -> bool:
        self.flush_timeout = timeout_seconds
        return self.drain_result

    def close(self) -> None:
        self.closed = True
        self.close_count += 1


class _FakePubSub:
    """Minimal redis pub/sub double: delivers queued messages, then ``None``."""

    def __init__(self, messages: list[object]) -> None:
        self._messages = list(messages)
        self.subscribed: list[str] = []
        self.closed = False
        self.close_thread_names: list[str] = []

    def subscribe(self, channel: str) -> None:
        self.subscribed.append(channel)

    def get_message(self, ignore_subscribe_messages: bool = False, timeout: float = 0.0) -> object:
        del ignore_subscribe_messages, timeout
        if self._messages:
            return self._messages.pop(0)
        # Mimic the real client's blocking poll so the listener loop doesn't busy-spin.
        time.sleep(0.01)
        return None

    def close(self) -> None:
        self.closed = True
        self.close_thread_names.append(threading.current_thread().name)


class _FakeRedis:
    """Redis client double whose ``pubsub()`` returns a pre-seeded message queue."""

    def __init__(self, messages: list[object]) -> None:
        self._pubsub = _FakePubSub(messages)

    def pubsub(self) -> _FakePubSub:
        return self._pubsub


class _FailingSender(_FakeSender):
    """Sender double whose send() fails before metadata should publish."""

    def __init__(self) -> None:
        super().__init__()
        self.failed_transfer_id: str | None = None

    def send(self, tensors: dict[str, torch.Tensor], transfer_id: str) -> None:
        del tensors
        self.failed_transfer_id = transfer_id
        raise RuntimeError("registry full")


class _StoreFailingOnSet:
    """Minimal store stand-in whose set() raises (metadata-publish rollback test)."""

    def set(self, key: str, value: str) -> None:
        del key, value
        raise RuntimeError("simulated metadata publish failure")


class _FakeReceiver:
    """Returns whatever tensors were last passed to the matched sender."""

    def __init__(self, sender: _FakeSender) -> None:
        self._sender = sender
        self.closed = False

    def recv(
        self,
        transfer_id: str,
        tensor_specs: dict[str, tuple[tuple[int, ...], str]],
        rollout_idx: int | None = None,
    ) -> dict[str, torch.Tensor]:
        del rollout_idx
        for stored_id, tensors in self._sender.sent:
            if stored_id == transfer_id:
                assert set(tensors.keys()) == set(tensor_specs.keys())
                return tensors
        raise RuntimeError(f"FakeReceiver: no sender record for {transfer_id}")

    def close(self) -> None:
        self.closed = True


class _FailingReceiver(_FakeReceiver):
    """Receiver double whose recv() fails after metadata validation."""

    def recv(
        self,
        transfer_id: str,
        tensor_specs: dict[str, tuple[tuple[int, ...], str]],
        rollout_idx: int | None = None,
    ) -> dict[str, torch.Tensor]:
        del transfer_id, tensor_specs, rollout_idx
        raise RuntimeError("simulated receiver failure")


def _minimal_episode() -> EpisodeOutput:
    """Build a minimal EpisodeOutput suitable for round-trip testing."""
    return EpisodeOutput(
        scene_id="scene_alpha",
        session_uuid="session_zero",
        num_steps=1,
        policy_outputs=(
            PolicyOutput(
                chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
                chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                chosen_dt_us=torch.tensor([0], dtype=torch.int64),
            ),
        ),
    )


def _writer(store: TCPStore, sender: _FakeSender) -> NcclEpisodeWriter:
    """Build an NcclEpisodeWriter over a fake sender with a fixed run identity."""
    return NcclEpisodeWriter(store=store, sender=sender, experiment_name="exp", job_id="job")  # type: ignore[arg-type]


def test_writer_satisfies_role_protocol(store: TCPStore) -> None:
    """The NCCL writer structurally matches the rollout-writer role protocol."""
    assert isinstance(_writer(store, _FakeSender()), EpisodeWriter)


def test_round_trip_preserves_episode_through_writer_resolve_pair(store: TCPStore) -> None:
    """Writer.write produces a handle; the mixin resolve reconstructs the EpisodeOutput."""
    sender = _FakeSender(rollout_idx=0)
    writer = _writer(store, sender)
    resolver = _Resolver(store, _FakeReceiver(sender))  # type: ignore[arg-type]
    episode = _minimal_episode()
    handle = writer.write(episode)
    transfer_id = normalize_nccl_handle(handle)
    assert handle.startswith("nccl:0:")
    assert sender.sent[0][0] == transfer_id
    assert store.check([build_metadata_key(transfer_id)])
    loaded = resolver._resolve_nccl_handle(handle)
    assert loaded.scene_id == episode.scene_id
    assert loaded.session_uuid == episode.session_uuid
    original = episode.policy_outputs[0]
    new = loaded.policy_outputs[0]
    assert torch.equal(new.chosen_xyz, original.chosen_xyz)
    assert torch.equal(new.chosen_quat, original.chosen_quat)
    assert torch.equal(new.chosen_dt_us, original.chosen_dt_us)
    assert not store.check([build_metadata_key(transfer_id)])


def test_resolve_accepts_raw_transfer_id(store: TCPStore) -> None:
    """The resolver normalizes both raw transfer ids and external nccl: handles."""
    sender = _FakeSender(rollout_idx=0)
    writer = _writer(store, sender)
    resolver = _Resolver(store, _FakeReceiver(sender))  # type: ignore[arg-type]
    handle = writer.write(_minimal_episode())

    loaded = resolver._resolve_nccl_handle(normalize_nccl_handle(handle))

    assert loaded.scene_id == "scene_alpha"


def test_policy_input_resolves_nccl_handle_synchronously_without_prefetch_path() -> None:
    """The policy loop resolves NCCL handles inline, with no background prefetch entry point."""
    episode = _minimal_episode()
    packer = _SyncPolicyPacker(episode)

    result = packer.get_policy_input(
        sample="sample",
        rollout_output="nccl:0:abc",
        n_ignore_prefix_tokens=3,
        extra="value",
    )

    assert result == ["collated"]
    assert packer.resolved_handles == ["nccl:0:abc"]
    assert packer.seen_rollout_output is episode
    assert packer.seen_n_ignore_prefix_tokens == 3
    assert packer.seen_kwargs == {"extra": "value"}
    assert not hasattr(packer, "_fetch_batch")
    assert not hasattr(packer, "_sync_fetch")


def test_production_nccl_packer_class_has_no_background_prefetch_hooks() -> None:
    """The production NCCL policy packer must not expose the prefetch path."""
    assert not hasattr(NcclAlpagymDataPacker, "_fetch_batch")
    assert not hasattr(NcclAlpagymDataPacker, "_sync_fetch")
    assert all(cls.__name__ != "PrefetchDataPackerMixin" for cls in NcclAlpagymDataPacker.__mro__)


def test_resolve_raises_when_metadata_missing(store: TCPStore) -> None:
    """An unknown handle surfaces a clear error rather than silently failing."""
    resolver = _Resolver(store, _FakeReceiver(_FakeSender()))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="No metadata"):
        resolver._resolve_nccl_handle("0:unknown")


def test_write_does_not_publish_metadata_when_sender_registration_fails(store: TCPStore) -> None:
    """A send() failure cannot leave an unreadable metadata key behind."""
    sender = _FailingSender()
    writer = _writer(store, sender)
    with pytest.raises(RuntimeError, match="registry full"):
        writer.write(_minimal_episode())
    assert sender.failed_transfer_id is not None
    assert not store.check([build_metadata_key(sender.failed_transfer_id)])


def test_write_releases_registration_when_metadata_publish_fails() -> None:
    """A store.set() failure rolls back the sender registration.

    Without the rollback, the payload would leak in the sender's pending
    registry until shutdown, with no handle ever returned to drive a normal
    read-side cleanup.
    """
    sender = _FakeSender()
    writer = NcclEpisodeWriter(
        store=_StoreFailingOnSet(),  # type: ignore[arg-type]
        sender=sender,  # type: ignore[arg-type]
        experiment_name="exp",
        job_id="job",
    )
    with pytest.raises(RuntimeError, match="simulated metadata publish failure"):
        writer.write(_minimal_episode())
    assert len(sender.sent) == 1
    transfer_id = sender.sent[0][0]
    assert sender.released == [(transfer_id, "metadata_publish_error")]


def test_resolve_deletes_malformed_metadata(store: TCPStore) -> None:
    """Malformed manifests are deleted before the parse error is re-raised."""
    handle = "0:bad"
    metadata_key = build_metadata_key(handle)
    store.set(metadata_key, "{not-json")
    resolver = _Resolver(store, _FakeReceiver(_FakeSender()))  # type: ignore[arg-type]
    with pytest.raises(Exception):
        resolver._resolve_nccl_handle(handle)
    assert not store.check([metadata_key])


def test_resolve_rejects_invalid_tensor_dtype_before_receiver(store: TCPStore) -> None:
    """Malformed tensor specs fail before dtype parsing reaches the recv thread."""
    handle = "0:bad-dtype"
    metadata_key = build_metadata_key(handle)
    store.set(
        metadata_key,
        json.dumps(
            {"manifest": {"__tensor_key__": "x", "shape": [1], "dtype": "torch.not_a_dtype"}}
        ),
    )
    resolver = _Resolver(store, _FakeReceiver(_FakeSender()))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dtype is unsupported"):
        resolver._resolve_nccl_handle(handle)
    assert not store.check([metadata_key])


def test_resolve_deletes_metadata_when_receiver_fails(store: TCPStore) -> None:
    """Terminal recv/rendezvous failures delete the manifest before re-raising."""
    sender = _FakeSender(rollout_idx=0)
    handle = _writer(store, sender).write(_minimal_episode())
    transfer_id = normalize_nccl_handle(handle)
    metadata_key = build_metadata_key(transfer_id)
    assert store.check([metadata_key])

    resolver = _Resolver(store, _FailingReceiver(sender))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="simulated receiver failure"):
        resolver._resolve_nccl_handle(handle)
    assert not store.check([metadata_key])


def test_writer_release_deletes_metadata_and_sender_registration(store: TCPStore) -> None:
    """release() accepts external handles and frees raw-id transport state."""
    sender = _FakeSender()
    writer = _writer(store, sender)
    handle = writer.write(_minimal_episode())
    transfer_id = normalize_nccl_handle(handle)
    assert store.check([build_metadata_key(transfer_id)])

    writer.release(handle, "cosmos_cleanup")

    assert not store.check([build_metadata_key(transfer_id)])
    assert sender.released == [(transfer_id, "cosmos_cleanup")]


def test_writer_exposes_sender_rollout_idx(store: TCPStore) -> None:
    """The writer exposes the sender's rollout index (used to build the cleanup channel)."""
    assert _writer(store, _FakeSender(rollout_idx=7)).rollout_idx == 7


def test_cleanup_subscriber_frees_sender_buffer_on_discard_message(store: TCPStore) -> None:
    """F9a end-to-end: a controller discard on the cleanup channel frees the sender buffer.

    The outdated-rollout filter path is now cosmos's built-in
    ``NcclPayloadTransport`` publishing ``{"transfer_id": ...}`` on this rollout's
    cleanup channel; this asserts the writer's subscriber receives that message and
    releases the matching pending transfer (deleting its manifest).
    """
    sender = _FakeSender(rollout_idx=3)
    writer = _writer(store, sender)
    handle = writer.write(_minimal_episode())
    transfer_id = normalize_nccl_handle(handle)
    assert store.check([build_metadata_key(transfer_id)])

    redis = _FakeRedis([{"data": json.dumps({"transfer_id": transfer_id})}])
    writer.start_cleanup(redis)
    try:
        deadline = time.time() + 3.0
        while not sender.released and time.time() < deadline:
            time.sleep(0.01)
    finally:
        writer.close()

    assert sender.released == [(transfer_id, "cosmos_cleanup")]
    assert not store.check([build_metadata_key(transfer_id)])
    # The subscriber must listen on the exact channel the cosmos controller publishes
    # on -- built from (experiment_name, job_id, rollout_idx) via cosmos's own helpers.
    # Asserting the string (not just the count) catches a builder-format or input-order
    # drift that would otherwise silently route cleanup to a channel nobody hears.
    expected_channel = build_cleanup_channel(
        build_rollout_prefix(build_nccl_prefix(experiment_name="exp", job_id="job"), 3)
    )
    assert redis._pubsub.subscribed == [expected_channel]


def test_writer_close_lets_cleanup_thread_close_pubsub_once(store: TCPStore) -> None:
    """close() signals the cleanup listener; the listener owns pubsub.close()."""
    sender = _FakeSender(rollout_idx=2)
    writer = _writer(store, sender)
    redis = _FakeRedis([])

    writer.start_cleanup(redis)
    deadline = time.time() + 3.0
    while not redis._pubsub.subscribed and time.time() < deadline:
        time.sleep(0.01)

    writer.close()
    writer.close()

    assert sender.close_count == 1
    assert redis._pubsub.closed
    assert redis._pubsub.close_thread_names == ["alpagym-nccl-cleanup-2"]


def test_writer_rejects_cleanup_start_after_close(store: TCPStore) -> None:
    """A closed writer cannot restart the discard-cleanup subscriber."""
    writer = _writer(store, _FakeSender())

    writer.close()

    with pytest.raises(RuntimeError, match="after close"):
        writer.start_cleanup(_FakeRedis([]))


def test_cleanup_subscriber_ignores_malformed_messages(store: TCPStore) -> None:
    """A cleanup payload that breaks the {"transfer_id": ...} contract is ignored, not fatal."""
    sender = _FakeSender()
    writer = _writer(store, sender)

    writer._process_cleanup_message({"data": "{not-json"})
    writer._process_cleanup_message({"data": json.dumps({"wrong_key": "x"})})
    writer._process_cleanup_message({"data": json.dumps({"transfer_id": 123})})

    assert sender.released == []


def test_start_cleanup_requires_redis_client(store: TCPStore) -> None:
    """A rollout writer refuses to start its subscriber without an injected Redis client."""
    writer = _writer(store, _FakeSender())
    with pytest.raises(RuntimeError, match="requires an injected Redis client"):
        writer.start_cleanup(None)


def test_flush_pending_sends_forwards_to_sender_drain(store: TCPStore) -> None:
    """flush_pending_sends waits on the sender's bounded drain before weight sync."""
    sender = _FakeSender()
    writer = _writer(store, sender)
    writer.flush_pending_sends()
    assert sender.flush_timeout is not None


def test_flush_pending_sends_raises_on_drain_timeout(store: TCPStore) -> None:
    """A drain timeout fails fast so weight sync never overlaps in-flight sends."""
    sender = _FakeSender()
    sender.drain_result = False
    writer = _writer(store, sender)
    with pytest.raises(TimeoutError, match="in-flight transfer"):
        writer.flush_pending_sends()
