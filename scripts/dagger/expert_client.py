"""Async expert client for online DAgger during RL rollout.

Sends observations to the Alpamayo-1.5 expert service on GPU 2
asynchronously.  Each call returns a future; results are collected
when the rollout session closes, so the ~20 s expert inference
overlaps with the next simulation steps.

Usage::

    client = ExpertClient(host="localhost", port=5557)
    future = client.async_query(camera_frames, ego_history_xyz, ...)
    # ... continue rollout ...
    result = future.result(timeout=60)  # {"action_indices": [...], ...}
    client.close()
"""

from __future__ import annotations

import concurrent.futures
import io
import logging
import pickle
import threading
from typing import Any

import torch
import zmq

logger = logging.getLogger(__name__)


class ExpertFuture:
    """A handle for an in-flight expert query."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: dict[str, Any] | None = None
        self._error: Exception | None = None

    def set_result(self, result: dict[str, Any]) -> None:
        self._result = result
        self._event.set()

    def set_error(self, error: Exception) -> None:
        self._error = error
        self._event.set()

    def result(self, timeout: float = 120.0) -> dict[str, Any]:
        if not self._event.wait(timeout=timeout):
            raise TimeoutError("Expert query timed out")
        if self._error is not None:
            raise self._error
        return self._result  # type: ignore[return-value]

    @property
    def done(self) -> bool:
        return self._event.is_set()


class ExpertClient:
    """Async client that sends observations to the expert service.

    A background thread owns the ZMQ DEALER socket, sends requests, and
    resolves futures.  The main rollout thread calls ``async_query`` and
    later ``future.result()``.

    Args:
        host: Expert service host.
        port: Expert service port.
        enabled: When ``False``, ``async_query`` returns a future that
            immediately resolves to ``None``.  This lets the rollout run
            with or without expert labeling via a single code path.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5557,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._futures: dict[bytes, ExpertFuture] = {}
        self._futures_lock = threading.Lock()
        self._stop = threading.Event()

        if not enabled:
            self._ctx = None
            self._sock = None
            self._thread = None
            return

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.DEALER)
        self._sock.connect(f"tcp://{host}:{port}")
        self._sock.setsockopt(zmq.LINGER, 0)
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        logger.info("ExpertClient connected to %s:%d", host, port)

    def async_query(
        self,
        camera_frames: torch.Tensor,
        camera_indices: torch.Tensor,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
    ) -> ExpertFuture:
        """Send observation to the expert service and return a future."""
        future = ExpertFuture()
        if not self._enabled:
            future.set_result(None)  # type: ignore[arg-type]
            return future

        import os
        req_id = os.urandom(16)
        with self._futures_lock:
            self._futures[req_id] = future

        payload = {
            "camera_frames": camera_frames.cpu(),
            "camera_indices": camera_indices.cpu(),
            "ego_history_xyz": ego_history_xyz.cpu(),
            "ego_history_rot": ego_history_rot.cpu(),
        }
        buf = io.BytesIO()
        pickle.dump(payload, buf)
        msg = buf.getvalue()

        self._sock.send_multipart([req_id, msg])
        return future

    def _recv_loop(self) -> None:
        """Background thread: receive responses and resolve futures."""
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            events = dict(poller.poll(timeout=1000))
            if self._sock not in events:
                continue
            try:
                parts = self._sock.recv_multipart()
                if len(parts) < 2:
                    continue
                req_id = parts[0]
                msg = parts[1]
                result = pickle.loads(msg)
                with self._futures_lock:
                    future = self._futures.pop(req_id, None)
                if future is not None:
                    if "error" in result:
                        future.set_error(RuntimeError(result["error"]))
                    else:
                        future.set_result(result)
            except Exception as e:
                logger.warning("ExpertClient recv error: %s", e)

    def close(self) -> None:
        """Shut down the client and background thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._sock is not None:
            self._sock.close()
        if self._ctx is not None:
            self._ctx.term()
        logger.info("ExpertClient closed.")

    @property
    def enabled(self) -> bool:
        return self._enabled
