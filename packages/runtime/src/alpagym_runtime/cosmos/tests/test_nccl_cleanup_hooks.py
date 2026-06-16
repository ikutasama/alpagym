# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AlpaGym's Cosmos-RL NCCL cleanup hooks."""

import queue
from types import SimpleNamespace
from typing import Any

import pytest
from alpagym_runtime.cosmos import nccl_cleanup_hooks as hooks


def test_install_publisher_opt_in_publishes_cleanup_on_buffer_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After install + setup, a direct rollout-buffer clear publishes NCCL cleanup."""

    class PolicyStatusManager:
        """Minimal Cosmos status-manager stand-in."""

        def __init__(self) -> None:
            self.rollout_buffer: queue.Queue[Any] = queue.Queue()
            self.published: list[tuple[list[Any], list[Any]]] = []
            self.config = SimpleNamespace(
                train=SimpleNamespace(
                    train_policy=SimpleNamespace(data_dispatch_as_rank_in_mesh=False)
                )
            )

        def setup(self, *args: Any, **kwargs: Any) -> str:
            """Return a sentinel result after the wrapped setup runs."""
            del args, kwargs
            return "setup-result"

        def _publish_payload_transport_cleanup(
            self,
            rollouts: list[Any],
            filtered: list[Any],
        ) -> None:
            """Record a discard-cleanup publish for assertions."""
            self.published.append((rollouts, filtered))

    monkeypatch.setattr(hooks, "PolicyStatusManager", PolicyStatusManager)

    hooks.install_cosmos_nccl_cleanup_publisher_opt_in()
    manager = PolicyStatusManager()
    assert manager.setup() == "setup-result"

    manager.rollout_buffer.put(SimpleNamespace(completion="nccl:0:discarded"))
    manager.rollout_buffer.queue.clear()
    assert manager.published == [
        ([SimpleNamespace(completion="nccl:0:discarded")], []),
    ]
    assert manager.rollout_buffer.empty()


def test_rollout_buffer_clear_propagates_cleanup_publish_error() -> None:
    """The clear wrapper re-raises whatever its injected publisher raises.

    This pins the wrapper's local behavior only. In production cosmos's real
    publisher (``PolicyStatusManager._publish_payload_transport_cleanup`` ->
    ``PayloadTransportRegistry.handle_discarded``) swallows per-transport publish
    failures and lets the clear proceed, so a real NCCL cleanup-publish failure does
    not abort the buffer clear.
    """

    class PolicyStatusManager:
        """Minimal status manager with a failing cleanup publisher."""

        def __init__(self) -> None:
            self.rollout_buffer: queue.Queue[Any] = queue.Queue()
            self.config = SimpleNamespace(
                train=SimpleNamespace(
                    train_policy=SimpleNamespace(data_dispatch_as_rank_in_mesh=False)
                )
            )

        def _publish_payload_transport_cleanup(
            self,
            rollouts: list[Any],
            filtered: list[Any],
        ) -> None:
            """Fail the cleanup publish to exercise error propagation."""
            del rollouts, filtered
            raise RuntimeError("publish failed")

    manager = PolicyStatusManager()
    hooks.wrap_rollout_buffer_clear(manager)
    manager.rollout_buffer.put(SimpleNamespace(completion="nccl:0:discarded"))

    with pytest.raises(RuntimeError, match="publish failed"):
        manager.rollout_buffer.queue.clear()

    assert not manager.rollout_buffer.empty()
