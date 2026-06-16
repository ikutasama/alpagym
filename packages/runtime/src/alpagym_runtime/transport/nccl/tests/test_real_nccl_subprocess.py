# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-NCCL multi-process round-trip for the alpagym NCCL transport.

Two subprocesses spawn on this node's GPUs — one as ``Rollout`` and one as
``Policy``. Both connect to a real ``TCPStore`` master in the parent, build
the production :class:`NcclSender` / :class:`NcclReceiver` pair against real
``cosmos_rl.utils.pynccl``, and pass one ``EpisodeOutput`` through the
:class:`NcclEpisodeWriter` egress and the :class:`NcclDataPackerMixin` resolve
path. The parent reads a success manifest back from the store to verify the
policy decoded the payload correctly.

Gated by ``pytest.mark.nccl_e2e`` and skipped when fewer than 2 CUDA devices
are visible. Complements :mod:`test_e2e_smoke` (which fakes ``nccl_send`` /
``nccl_recv``) by running real NCCL between separate processes on separate
GPUs, validating the full data plane end-to-end without Docker, cosmos-rl's
launcher, or AlpaSim.
"""

import json
import logging
import socket
import time
from datetime import timedelta

import pytest
import torch
import torch.multiprocessing as mp
from alpagym_runtime.transport.nccl.endpoints import NcclDataPackerMixin, NcclEpisodeWriter
from alpagym_runtime.transport.nccl.protocol import normalize_nccl_handle
from alpagym_runtime.transport.nccl.receiver import NcclReceiver
from alpagym_runtime.transport.nccl.rendezvous import AckRendezvous, NcclRendezvousError
from alpagym_runtime.transport.nccl.sender import NcclSender, assign_rollout_idx
from alpagym_runtime.types import EpisodeOutput, PolicyOutput
from torch.distributed import TCPStore

pytestmark = pytest.mark.nccl_e2e


class _Resolver(NcclDataPackerMixin):
    """Trainer-side NCCL read path isolated from replay collation.

    ``NcclDataPackerMixin._resolve_nccl_handle`` uses only ``_store``,
    ``_receiver``, and ``_target_device``; this real-NCCL test drives the full
    resolve / rendezvous / recv / unpack path through it.
    """

    def __init__(self, store: TCPStore, receiver: NcclReceiver) -> None:
        """Wire the resolver to a store and receiver, reading onto CPU."""
        self._store = store
        self._receiver = receiver
        self._target_device = torch.device("cpu")


def _free_port() -> int:
    """Pick an ephemeral local TCP port for the test's TCPStore master."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _minimal_episode() -> EpisodeOutput:
    """Build a small EpisodeOutput with one PolicyOutput; CPU tensors only."""
    return EpisodeOutput(
        scene_id="real_nccl_scene",
        session_uuid="session_real_nccl",
        num_steps=1,
        policy_outputs=(
            PolicyOutput(
                chosen_xyz=torch.zeros((1, 3), dtype=torch.float32),
                chosen_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
                chosen_dt_us=torch.tensor([100], dtype=torch.int64),
            ),
        ),
    )


def _build_sender(*, store: TCPStore, experiment_name: str) -> NcclSender:
    """Construct the production NcclSender wired against real ``pynccl``."""
    from cosmos_rl.utils.pynccl import create_nccl_comm, create_nccl_uid, nccl_abort, nccl_send

    rollout_idx = assign_rollout_idx(
        experiment_name=experiment_name,
        job_id="test",
        num_rollout_replicas=1,
        store=store,
    )
    return NcclSender(
        experiment_name=experiment_name,
        job_id="test",
        rollout_idx=rollout_idx,
        num_policy_replicas=1,
        dp_shard_size=1,
        store=store,
        rendezvous=AckRendezvous(),
        create_nccl_uid=create_nccl_uid,
        create_nccl_comm=create_nccl_comm,
        nccl_send=nccl_send,
        nccl_abort=nccl_abort,
    )


def _build_receiver(*, store: TCPStore, experiment_name: str) -> NcclReceiver:
    """Construct the production NcclReceiver wired against real ``pynccl``."""
    from cosmos_rl.utils.pynccl import create_nccl_comm, create_nccl_uid, nccl_abort, nccl_recv

    return NcclReceiver(
        experiment_name=experiment_name,
        job_id="test",
        num_policy_replicas=1,
        dp_shard_size=1,
        num_rollout_replicas=1,
        store=store,
        rendezvous=AckRendezvous(),
        create_nccl_uid=create_nccl_uid,
        create_nccl_comm=create_nccl_comm,
        nccl_recv=nccl_recv,
        nccl_abort=nccl_abort,
    )


def _wait_for_key(store: TCPStore, key: str, timeout_seconds: float) -> bool:
    """Poll ``store.check`` so the sender's background thread isn't serialized.

    The rollout's main thread shares its TCPStore client with the sender's
    polling thread; a blocking ``store.wait`` here would block both.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if store.check([key]):
            return True
        time.sleep(0.25)
    return False


def _run_rollout(*, store: TCPStore) -> None:
    """Rollout subprocess body: write one episode through the NCCL transport."""
    sender = _build_sender(store=store, experiment_name="real_nccl")
    sender.setup()
    try:
        transport = NcclEpisodeWriter(
            store=store, sender=sender, experiment_name="real_nccl", job_id="test"
        )
        handle = transport.write(_minimal_episode())
        store.set("real_nccl:handle", handle)
        _wait_for_key(store, "real_nccl:policy_done", timeout_seconds=60.0)
    finally:
        sender.close()


def _run_rollout_discarded_payload(*, store: TCPStore) -> None:
    """Rollout subprocess body: publish a handle, discard payload, then idle."""
    sender = _build_sender(store=store, experiment_name="real_nccl_missing")
    sender.setup()
    try:
        transport = NcclEpisodeWriter(
            store=store, sender=sender, experiment_name="real_nccl_missing", job_id="test"
        )
        handle = transport.write(_minimal_episode())
        sender.release(normalize_nccl_handle(handle), reason="cosmos_cleanup")
        store.set("real_nccl_missing:handle", handle)
        _wait_for_key(store, "real_nccl_missing:policy_done", timeout_seconds=60.0)
    finally:
        sender.close()


def _run_policy(*, store: TCPStore) -> None:
    """Policy subprocess body: read one episode and publish a success manifest."""
    receiver = _build_receiver(store=store, experiment_name="real_nccl")
    receiver.setup()
    try:
        transport = _Resolver(store, receiver)
        store.wait(["real_nccl:handle"], timedelta(seconds=60.0))
        handle_value = store.get("real_nccl:handle")
        handle = handle_value.decode("utf-8") if isinstance(handle_value, bytes) else handle_value
        episode = transport._resolve_nccl_handle(handle)
        store.set(
            "real_nccl:policy_done",
            json.dumps(
                {
                    "scene_id": episode.scene_id,
                    "session_uuid": episode.session_uuid,
                    "num_steps": episode.num_steps,
                    "chosen_dt_us": int(episode.policy_outputs[0].chosen_dt_us.item()),
                }
            ),
        )
    finally:
        receiver.close()


def _run_policy_expect_missing(*, store: TCPStore) -> None:
    """Policy subprocess body: request a handle whose payload was discarded."""
    receiver = _build_receiver(store=store, experiment_name="real_nccl_missing")
    receiver.setup()
    try:
        transport = _Resolver(store, receiver)
        store.wait(["real_nccl_missing:handle"], timedelta(seconds=60.0))
        handle_value = store.get("real_nccl_missing:handle")
        handle = handle_value.decode("utf-8") if isinstance(handle_value, bytes) else handle_value
        try:
            transport._resolve_nccl_handle(handle)
            store.set("real_nccl_missing:policy_done", "no_error_raised")
        except NcclRendezvousError as error:
            store.set("real_nccl_missing:policy_done", f"missing:{error}")
    finally:
        receiver.close()


_SCENARIO_HAPPY_PATH = "happy"
_SCENARIO_MISSING_PAYLOAD = "missing"


def _worker_entrypoint(
    rank: int, master_host: str, master_port: int, log_dir: str, scenario: str
) -> None:
    """``torch.multiprocessing.spawn`` entrypoint; rank 0 = Rollout, rank 1 = Policy."""
    handler = logging.FileHandler(f"{log_dir}/worker_{rank}.log", mode="w")
    handler.setFormatter(
        logging.Formatter(f"[w{rank}] %(asctime)s %(name)s %(levelname)s %(message)s")
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    torch.cuda.set_device(rank)
    store = TCPStore(
        host_name=master_host,
        port=master_port,
        world_size=1,
        is_master=False,
    )
    try:
        if scenario == _SCENARIO_HAPPY_PATH:
            (_run_rollout if rank == 0 else _run_policy)(store=store)
        elif scenario == _SCENARIO_MISSING_PAYLOAD:
            (_run_rollout_discarded_payload if rank == 0 else _run_policy_expect_missing)(
                store=store
            )
        else:
            raise ValueError(f"Unknown scenario: {scenario!r}")
    finally:
        handler.flush()
        handler.close()
        root_logger.removeHandler(handler)


def _run_two_workers(*, log_dir: str, scenario: str) -> TCPStore:
    """Spawn two NCCL workers and return the parent's TCPStore master.

    Caller asserts on per-scenario sentinel keys in the returned store, then
    must ``del`` the master to release the port. Each worker's logs land at
    ``<log_dir>/worker_<rank>.log`` so callers can dump them on failure.
    """
    master_host = "127.0.0.1"
    master_port = _free_port()
    master = TCPStore(
        host_name=master_host,
        port=master_port,
        world_size=1,
        is_master=True,
        wait_for_workers=False,
    )

    def _dump_worker_logs() -> str:
        """Concatenate per-worker log files into one diagnostic blob."""
        parts: list[str] = []
        for rank in (0, 1):
            log_path = f"{log_dir}/worker_{rank}.log"
            try:
                with open(log_path) as fh:
                    parts.append(f"--- worker_{rank}.log ---\n{fh.read()}")
            except FileNotFoundError:
                parts.append(f"--- worker_{rank}.log (missing) ---")
        return "\n".join(parts)

    context = mp.spawn(
        _worker_entrypoint,
        args=(master_host, master_port, log_dir, scenario),
        nprocs=2,
        join=False,
    )
    try:
        deadline = time.time() + 180.0
        # ProcessContext.join returns False as soon as one worker is reaped
        # while the other is still running, so loop until all are reaped.
        try:
            while not context.join(timeout=max(0.0, deadline - time.time())):
                if time.time() >= deadline:
                    raise AssertionError(f"worker subprocesses timed out\n\n{_dump_worker_logs()}")
        except AssertionError:
            raise
        except Exception as error:
            raise AssertionError(
                f"worker subprocess raised: {error}\n\n{_dump_worker_logs()}"
            ) from error
    except BaseException:
        for proc in context.processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5.0)
        raise
    return master


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least 2 visible CUDA devices",
)
def test_real_nccl_subprocess_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Two subprocesses exchange one EpisodeOutput through real pynccl."""
    master = _run_two_workers(log_dir=str(tmp_path), scenario=_SCENARIO_HAPPY_PATH)
    try:
        manifest_raw = master.get("real_nccl:policy_done")
        manifest_str = (
            manifest_raw.decode("utf-8") if isinstance(manifest_raw, bytes) else manifest_raw
        )
        assert json.loads(manifest_str) == {
            "scene_id": "real_nccl_scene",
            "session_uuid": "session_real_nccl",
            "num_steps": 1,
            "chosen_dt_us": 100,
        }
    finally:
        del master


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least 2 visible CUDA devices",
)
def test_real_nccl_subprocess_missing_payload(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A receiver requesting a discarded transfer gets a clean rendezvous error.

    Exercises the cross-process ``missing`` path: the policy has a manifest for
    a transfer whose sender-side payload was discard-released. The rendezvous
    state machine must transition ``requested -> missing`` and the receiver
    must raise :class:`NcclRendezvousError` *before* invoking ``nccl_recv``.
    """
    master = _run_two_workers(log_dir=str(tmp_path), scenario=_SCENARIO_MISSING_PAYLOAD)
    try:
        result_raw = master.get("real_nccl_missing:policy_done")
        result = result_raw.decode("utf-8") if isinstance(result_raw, bytes) else result_raw
        assert result.startswith("missing:"), result
        assert "status=missing" in result
    finally:
        del master
