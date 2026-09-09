# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import threading
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any, Callable

import grpc
from alpagym_host.endpoint_registry import TopologyEndpoint
from alpasim_grpc.v0.common_pb2 import Empty, SessionRequestStatus, VersionId
from alpasim_grpc.v0.egodriver_pb2 import (
    DriveRequest,
    DriveResponse,
    DriveSessionCloseRequest,
    DriveSessionRequest,
    GroundTruthRequest,
    RolloutCameraImage,
    RolloutEgoTrajectory,
    RouteRequest,
)
from alpasim_grpc.v0.egodriver_pb2_grpc import add_EgodriverServiceServicer_to_server

from alpagym_runtime.alpasim.proto_conversion import (
    calibration_from_proto,
    drive_response_from_policy_output,
    ground_truth_from_proto,
    policy_input_from_tick_buffer,
)
from alpagym_runtime.alpasim.tick_buffer import TickBuffer
from alpagym_runtime.perf.instrument.scope import measure_perf, timed_scope
from alpagym_runtime.types import (
    EgoPose,
    GroundTruth,
    Policy,
    PolicyInput,
    PolicyOutput,
    RolloutCalibration,
    Trajectory,
)

logger = logging.getLogger(__name__)

_EXPERT_ENABLED = os.environ.get("ALPAGYM_EXPERT_HOST", "") != ""
_EXPERT_HOST = os.environ.get("ALPAGYM_EXPERT_HOST", "localhost")
_EXPERT_PORT = int(os.environ.get("ALPAGYM_EXPERT_PORT", "5557"))
_DAGGER_MIX_BETA = float(os.environ.get("ALPAGYM_DAGGER_MIX_BETA", "0.0"))

_expert_client = None
if _EXPERT_ENABLED:
    try:
        import sys as _sys
        _dagger_scripts = str(
            __import__("pathlib").Path(__file__).resolve().parents[5]
            / "scripts" / "dagger"
        )
        if _dagger_scripts not in _sys.path:
            _sys.path.insert(0, _dagger_scripts)
        from expert_client import ExpertClient  # type: ignore[import-not-found]
        _expert_client = ExpertClient(host=_EXPERT_HOST, port=_EXPERT_PORT)
        logger.info("Expert client enabled: %s:%d", _EXPERT_HOST, _EXPERT_PORT)
    except Exception as e:
        logger.warning("Failed to init expert client: %s — expert labeling disabled", e)
        _EXPERT_ENABLED = False


@dataclass(frozen=True)
class SessionRecord:
    """Frozen per-session payload (outputs + executed trajectory + GT) drained by the runner."""

    outputs: tuple[PolicyOutput, ...]
    executed_ego_trajectory: Trajectory
    ground_truth: GroundTruth | None


@dataclass
class _Session:
    """Per-session record held by `EgodriverGrpcServicer`."""

    calibration: RolloutCalibration
    policy: Policy
    tick_buffer: TickBuffer = field(default_factory=TickBuffer)
    step_index: int = 0
    tick_lock: threading.Lock = field(default_factory=threading.Lock)
    ground_truth: GroundTruth | None = None
    outputs: list[PolicyOutput] = field(default_factory=list)
    executed_poses: list[EgoPose] = field(default_factory=list)
    expert_futures: list[Any] = field(default_factory=list)
    expert_action_tokens: list[Any] = field(default_factory=list)
    expert_collector_done: threading.Event = field(default_factory=threading.Event)
    ade_list: list[float] = field(default_factory=list)
    fde_list: list[float] = field(default_factory=list)

    def consume_tick(self) -> tuple[int, TickBuffer]:
        """Return the current tick and prepare the next tick buffer."""
        with self.tick_lock:
            snapshot = self.tick_buffer
            step_index = self.step_index
            self.step_index += 1
            self.tick_buffer = TickBuffer(route=snapshot.route)
        return step_index, snapshot

    def record_step(self, policy_input: PolicyInput, policy_output: PolicyOutput) -> None:
        """Append the output and any newly observed executed poses from this tick.

        Only poses strictly newer than the last recorded one are kept, sorted
        by timestamp. AlpaSim's drive only delivers one session at a time, so
        this runs single-threaded per session and needs no extra lock.
        """
        self.outputs.append(policy_output)
        last_us = self.executed_poses[-1].timestamp_us if self.executed_poses else None
        fresh = [
            pose
            for pose in policy_input.ego_trajectory.poses
            if last_us is None or pose.timestamp_us > last_us
        ]
        self.executed_poses.extend(sorted(fresh, key=lambda pose: pose.timestamp_us))

    def get_record(self) -> SessionRecord:
        """Return a frozen `SessionRecord` for this session's recorded data."""
        return SessionRecord(
            outputs=tuple(self.outputs),
            executed_ego_trajectory=Trajectory(poses=tuple(self.executed_poses)),
            ground_truth=self.ground_truth,
        )


class EgodriverGrpcServicer:
    """AlpaSim EgodriverService implementation backed by per-session policies."""

    def __init__(
        self,
        policy_factory: Callable[[str, RolloutCalibration, int], Policy],
    ) -> None:
        """Store the policy factory and initialize per-session bookkeeping."""
        self._policy_factory = policy_factory
        self._sessions: dict[str, _Session] = {}
        self._sessions_lock = threading.Lock()
        self._session_records: dict[str, SessionRecord] = {}
        self._expert_collector_events: dict[str, threading.Event] = {}

    @measure_perf("driver/session_start", category="orchestration", cpu_snapshot=True)
    def start_session(
        self,
        request: DriveSessionRequest,
        context: grpc.ServicerContext,
    ) -> SessionRequestStatus:
        """Open one AlpaSim driver session and instantiate its policy.

        Converts the rollout spec to calibration and stores the new session.
        """
        session_uuid = str(request.session_uuid)
        calibration = calibration_from_proto(request.rollout_spec)
        random_seed = int(request.random_seed)
        policy = self._policy_factory(session_uuid, calibration, random_seed)
        session = _Session(calibration=calibration, policy=policy)
        with self._sessions_lock:
            self._sessions[session_uuid] = session
        logger.info("Started AlpaGym driver session=%s seed=%d", session_uuid, random_seed)
        return SessionRequestStatus()

    def submit_image_observation(
        self,
        request: RolloutCameraImage,
        context: grpc.ServicerContext,
    ) -> Empty:
        """Append `request.camera_image` to the session's tick-local image list."""
        with self._sessions_lock:
            session = self._sessions[request.session_uuid]
        with session.tick_lock:
            session.tick_buffer.camera_images.append(request.camera_image)
        return Empty()

    def submit_egomotion_observation(
        self,
        request: RolloutEgoTrajectory,
        context: grpc.ServicerContext,
    ) -> Empty:
        """Append egomotion observations to the session's tick-local trajectory."""
        with self._sessions_lock:
            session = self._sessions[request.session_uuid]
        with session.tick_lock:
            session.tick_buffer.add_ego_trajectory(request.trajectory)
        return Empty()

    def submit_route(
        self,
        request: RouteRequest,
        context: grpc.ServicerContext,
    ) -> Empty:
        """Store `request.route` as the session's latest route."""
        with self._sessions_lock:
            session = self._sessions[request.session_uuid]
        with session.tick_lock:
            session.tick_buffer.route = request.route
        return Empty()

    def submit_recording_ground_truth(
        self,
        request: GroundTruthRequest,
        context: grpc.ServicerContext,
    ) -> Empty:
        """Store the per-session recording GT on the session record.

        Raises `KeyError` for an unknown session uuid.
        """
        ground_truth = ground_truth_from_proto(request.ground_truth)
        with self._sessions_lock:
            session = self._sessions[request.session_uuid]
        session.ground_truth = ground_truth
        return Empty()

    @measure_perf("driver/drive", category="external_rpc")
    def drive(
        self,
        request: DriveRequest,
        context: grpc.ServicerContext,
    ) -> DriveResponse:
        """Run one policy tick for one AlpaSim session and record the I/O pair."""
        with self._sessions_lock:
            session = self._sessions[request.session_uuid]
        step_index, snapshot = session.consume_tick()

        # `policy_input_from_tick_buffer` and `drive_response_from_policy_output` carry
        # their own `@measure_perf("driver/input_build" | "driver/output_build")`
        # decorators, so they record under the active `driver/drive` scope here. These
        # per-tick scopes stay timing-only: per-tick psutil snapshots (which read /proc
        # smaps for USS) would dominate their own small durations. CPU resource trends
        # come from the periodic monitor and the once-per-session driver/session_start
        # scope instead. The GPU snapshot stays on policy_step, where it is the valuable
        # attribution and cheap next to the model forward.
        policy_input = policy_input_from_tick_buffer(
            step_index=step_index,
            time_now_us=int(request.time_now_us),
            time_query_us=int(request.time_query_us),
            calibration=session.calibration,
            tick_buffer=snapshot,
        )
        try:
            with timed_scope("driver/policy_step", category="compute_gpu_wall", gpu_snapshot=True):
                policy_output = session.policy.step(policy_input)
        except Exception as oom_exc:
            import torch
            logger.warning("CUDA error in drive() step %d: %s; clearing cache", step_index, oom_exc)
            torch.cuda.empty_cache()
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, f"CUDA error in policy step: {oom_exc}")
        session.record_step(policy_input, policy_output)

        if _EXPERT_ENABLED and _expert_client is not None:
            try:
                rd = policy_output.replay_data
                if rd is not None and "model_input" in rd.payload:
                    mi = rd.payload["model_input"]
                    import torch as _torch
                    import numpy as _np
                    import random as _random
                    def _to_tensor(v):
                        return v if isinstance(v, _torch.Tensor) else _torch.as_tensor(v)
                    cf = _to_tensor(mi["camera_frames"])
                    ci = _to_tensor(mi["camera_indices"])
                    hist_xyz = _to_tensor(mi["ego_history_xyz"])
                    hist_rot = _to_tensor(mi["ego_history_rot"])
                    future = _expert_client.async_query(cf, ci, hist_xyz, hist_rot)

                    if _DAGGER_MIX_BETA > 0.0:
                        result = future.result(timeout=120)
                        if result is not None and "action_indices" in result:
                            action_tokens = _torch.tensor(
                                result["action_indices"], dtype=_torch.int64,
                            )
                            rd.payload["expert_action_tokens"] = action_tokens

                            use_expert = _random.random() < _DAGGER_MIX_BETA
                            if "expert_xyz" in result and "expert_rot" in result:
                                from alpagym_runtime.policies.alpamayo.output_trajectory import (
                                    build_policy_output_trajectory,
                                )
                                expert_xyz = _np.asarray(result["expert_xyz"])
                                expert_rot = _np.asarray(result["expert_rot"])
                                step = max(1, int(round(0.5 / 0.1)))
                                indices = [min(i * step, expert_xyz.shape[0] - 1) for i in range(10)]
                                future_xyz_ego = _torch.tensor(
                                    expert_xyz[indices], dtype=_torch.float32,
                                )
                                future_rot_ego = _torch.tensor(
                                    expert_rot[indices], dtype=_torch.float32,
                                )
                                future_dt_us = _torch.arange(1, 11, dtype=_torch.int64) * 500000
                                ego_pose_now = policy_input.ego_trajectory.poses[-1].pose
                                exp_xyz, exp_quat, exp_dt = build_policy_output_trajectory(
                                    ego_pose_now=ego_pose_now,
                                    future_xyz_ego=future_xyz_ego,
                                    future_rot_ego=future_rot_ego,
                                    future_dt_us=future_dt_us,
                                )

                                student_xyz = policy_output.chosen_xyz
                                ade = _torch.norm(
                                    student_xyz[1:] - exp_xyz[1:], dim=-1,
                                ).mean().item()
                                fde = _torch.norm(
                                    student_xyz[-1] - exp_xyz[-1],
                                ).item()
                                rd.payload["ade"] = ade
                                rd.payload["fde"] = fde
                                session.ade_list.append(ade)
                                session.fde_list.append(fde)

                                if use_expert:
                                    from dataclasses import replace as _replace
                                    policy_output = _replace(
                                        policy_output,
                                        chosen_xyz=exp_xyz,
                                        chosen_quat=exp_quat,
                                        chosen_dt_us=exp_dt,
                                    )
                                    logger.info(
                                        "  step %d EXPERT drives (beta=%.2f) tokens=%s "
                                        "ade=%.3f fde=%.3f (%.1fs)",
                                        step_index, _DAGGER_MIX_BETA, action_tokens.tolist(),
                                        ade, fde, result.get("elapsed", 0),
                                    )
                                else:
                                    logger.info(
                                        "  step %d AutoVLA drives (beta=%.2f) expert_tokens=%s "
                                        "ade=%.3f fde=%.3f",
                                        step_index, _DAGGER_MIX_BETA, action_tokens.tolist(),
                                        ade, fde,
                                    )
                            elif use_expert:
                                logger.warning(
                                    "  step %d expert drives but no expert_xyz, AutoVLA fallback",
                                    step_index,
                                )
                            else:
                                logger.info(
                                    "  step %d AutoVLA drives (beta=%.2f) expert_tokens=%s",
                                    step_index, _DAGGER_MIX_BETA, action_tokens.tolist(),
                                )
                            session.expert_futures.append(None)
                        else:
                            logger.warning("  step %d expert query failed, AutoVLA fallback", step_index)
                            session.expert_futures.append(None)
                    else:
                        session.expert_futures.append(future)
                else:
                    session.expert_futures.append(None)
            except Exception as e:
                logger.warning("Expert async_query failed at step %d: %s", step_index, e)
                session.expert_futures.append(None)

        return drive_response_from_policy_output(policy_input, policy_output)

    @measure_perf("driver/session_close", category="synchronization_wait")
    def close_session(
        self,
        request: DriveSessionCloseRequest,
        context: grpc.ServicerContext,
    ) -> Empty:
        """Close one AlpaSim driver session and freeze its record.

        If expert labeling is enabled, a background thread collects the
        pending expert futures so this RPC returns immediately without
        blocking the AlPaSim ``simulate()`` deadline.  ``pop_session_record``
        waits for the collector to finish before handing the record to the
        rollout worker.
        """
        session_uuid = str(request.session_uuid)
        with self._sessions_lock:
            session = self._sessions.pop(session_uuid)

        self._session_records[session_uuid] = session.get_record()

        if _EXPERT_ENABLED and session.expert_futures:
            self._expert_collector_events[session_uuid] = session.expert_collector_done
            logger.info(
                "Collecting %d expert futures for session=%s",
                len(session.expert_futures), session_uuid,
            )
            collector = threading.Thread(
                target=self._collect_expert_futures,
                args=(session_uuid, session),
                daemon=True,
            )
            collector.start()
        else:
            session.expert_collector_done.set()

        session.policy.close()
        import torch
        torch.cuda.empty_cache()
        logger.info(
            "Closed AlpaGym driver session=%s recorded_steps=%d",
            session_uuid,
            len(self._session_records[session_uuid].outputs),
        )
        return Empty()

    def _collect_expert_futures(self, session_uuid: str, session: _Session) -> None:
        """Collect expert futures in the background and inject tokens into the frozen record."""
        import torch
        expert_results: list[dict | None] = []
        for i, future in enumerate(session.expert_futures):
            if future is None:
                session.expert_action_tokens.append(None)
                expert_results.append(None)
                continue
            try:
                result = future.result(timeout=600)
                if result is not None and "action_indices" in result:
                    tokens = torch.tensor(result["action_indices"], dtype=torch.int64)
                    session.expert_action_tokens.append(tokens)
                    expert_results.append(result)
                    logger.info(
                        "  step %d expert tokens=%s (%.1fs)",
                        i, tokens.tolist(), result.get("elapsed", 0),
                    )
                else:
                    session.expert_action_tokens.append(None)
                    expert_results.append(None)
            except Exception as e:
                logger.warning("  step %d expert future failed: %s", i, e)
                session.expert_action_tokens.append(None)
                expert_results.append(None)

        record = self._session_records.get(session_uuid)
        if record is not None:
            for i, output in enumerate(record.outputs):
                if i >= len(session.expert_action_tokens):
                    break
                et = session.expert_action_tokens[i]
                if et is not None and output.replay_data is not None:
                    output.replay_data.payload["expert_action_tokens"] = et

                    result = expert_results[i]
                    if (
                        result is not None
                        and "expert_xyz" in result
                        and "expert_rot" in result
                    ):
                        import numpy as _np
                        from alpagym_runtime.policies.alpamayo.output_trajectory import (
                            build_policy_output_trajectory,
                        )
                        from alpagym_runtime.types import Pose, Vec3, Quaternion

                        expert_xyz = _np.asarray(result["expert_xyz"])
                        expert_rot = _np.asarray(result["expert_rot"])
                        step = max(1, int(round(0.5 / 0.1)))
                        indices = [min(i * step, expert_xyz.shape[0] - 1) for i in range(10)]
                        future_xyz_ego = torch.tensor(
                            expert_xyz[indices], dtype=torch.float32,
                        )
                        future_rot_ego = torch.tensor(
                            expert_rot[indices], dtype=torch.float32,
                        )
                        future_dt_us = torch.arange(1, 11, dtype=torch.int64) * 500000

                        student_xyz = output.chosen_xyz
                        student_quat = output.chosen_quat
                        ego_pose_now = Pose(
                            vec=Vec3(
                                x=float(student_xyz[0, 0]),
                                y=float(student_xyz[0, 1]),
                                z=float(student_xyz[0, 2]),
                            ),
                            quat=Quaternion(
                                w=float(student_quat[0, 0]),
                                x=float(student_quat[0, 1]),
                                y=float(student_quat[0, 2]),
                                z=float(student_quat[0, 3]),
                            ),
                        )
                        exp_xyz, _, _ = build_policy_output_trajectory(
                            ego_pose_now=ego_pose_now,
                            future_xyz_ego=future_xyz_ego,
                            future_rot_ego=future_rot_ego,
                            future_dt_us=future_dt_us,
                        )
                        ade = torch.norm(
                            student_xyz[1:] - exp_xyz[1:], dim=-1,
                        ).mean().item()
                        fde = torch.norm(
                            student_xyz[-1] - exp_xyz[-1],
                        ).item()
                        output.replay_data.payload["ade"] = ade
                        output.replay_data.payload["fde"] = fde
                        session.ade_list.append(ade)
                        session.fde_list.append(fde)

        expert_labels = sum(
            1 for x in self._session_records[session_uuid].outputs
            if x.replay_data is not None
            and x.replay_data.payload.get("expert_action_tokens") is not None
        ) if session_uuid in self._session_records else 0
        n_ade = len(session.ade_list)
        if n_ade > 0:
            import statistics as _stats
            logger.info(
                "Expert collection done session=%s expert_labels=%d "
                "ade_mean=%.3f fde_mean=%.3f (n=%d)",
                session_uuid, expert_labels,
                _stats.mean(session.ade_list),
                _stats.mean(session.fde_list),
                n_ade,
            )
        else:
            logger.info(
                "Expert collection done session=%s expert_labels=%d",
                session_uuid, expert_labels,
            )
        session.expert_collector_done.set()

    def pop_session_record(self, session_uuid: str) -> SessionRecord:
        """Remove and return the ``SessionRecord`` for ``session_uuid``.

        Blocks until the background expert-future collector (if any) has
        finished, so the rollout worker receives the fully-labeled record.
        """
        event = self._expert_collector_events.pop(session_uuid, None)
        if event is not None:
            event.wait(timeout=700)
        return self._session_records.pop(session_uuid)

    def get_version(self, request: Empty, context: grpc.ServicerContext) -> VersionId:
        """Return a minimal driver version."""
        return VersionId(version_id="alpagym-egodriver", git_hash="unknown")

    def shut_down(self, request: Empty, context: grpc.ServicerContext) -> Empty:
        """Reject AlpaSim shutdown requests; the rollout owns its lifecycle."""
        raise NotImplementedError("EgodriverService does not support remote shutdown.")


class EgodriverServer:
    """Lifecycle wrapper for the rollout worker's Egodriver gRPC server."""

    def __init__(
        self,
        name: str,
        max_concurrent_rollouts: int,
        policy_factory: Callable[[str, RolloutCalibration, int], Policy],
        publish_host: str = "localhost",
    ) -> None:
        """Build the gRPC server and bind a port without starting it."""
        self.name = name
        self.max_concurrent_rollouts = max_concurrent_rollouts
        self.host = publish_host
        # Publish the discoverable host, but bind all interfaces for remote workers.
        bind_host = "localhost" if publish_host == "localhost" else "[::]"
        self._servicer = EgodriverGrpcServicer(policy_factory=policy_factory)
        # Under streaming dispatch, up to `max_concurrent_rollouts` drive()
        # handlers can be in flight alongside up to `max_concurrent_rollouts`
        # close_session callbacks, plus headroom for non-session RPCs
        # (start_session, submit_*, get_version).
        self._grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=2 * max_concurrent_rollouts + 2),
            options=[
                ("grpc.max_send_message_length", 256 * 1024 * 1024),
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ],
        )
        add_EgodriverServiceServicer_to_server(self._servicer, self._grpc_server)
        bind_port = int(os.environ.get("ALPAGYM_DRIVER_PORT", "0"))
        self.port = self._grpc_server.add_insecure_port(f"{bind_host}:{bind_port}")
        if self.port == 0:
            raise RuntimeError(f"Failed to bind EgodriverService on {bind_host}:{bind_port}")

    @property
    def topology_endpoint(self) -> TopologyEndpoint:
        """Return this server's publishable topology endpoint."""
        return TopologyEndpoint(
            id=self.name,
            host=self.host,
            port=int(self.port),
            capacity=self.max_concurrent_rollouts,
        )

    def start(self) -> None:
        """Start serving EgodriverService requests."""
        self._grpc_server.start()

    def stop(self, grace_seconds: float = 30.0) -> None:
        """Stop serving EgodriverService requests, draining in-flight calls.

        Graceful shutdown is required by the rollout backend's tear-down
        ordering: a `drive()` handler that has reached `policy.step()` is
        waiting on the inference engine; if the engine is told to shut down
        before that handler returns, the handler blocks forever on
        `future.result()`. The grace window lets in-flight drive calls
        finish before the engine sentinel posts; `.wait()` blocks the
        caller until shutdown actually completes (grpc's `stop()` returns
        immediately and runs the drain on a background thread).
        """
        shutdown_event = self._grpc_server.stop(grace_seconds)
        shutdown_event.wait()

    @property
    def servicer(self) -> EgodriverGrpcServicer:
        """Return the gRPC servicer that owns the session-lifecycle state."""
        return self._servicer
