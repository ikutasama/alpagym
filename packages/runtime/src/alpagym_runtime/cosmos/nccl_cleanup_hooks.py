# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Buffer-clear cleanup hook for the Cosmos-RL NCCL discard protocol.

Cosmos-RL owns the controller-to-rollout Redis cleanup for ``nccl:`` completions
via ``PayloadTransportRegistry``, which AlpaGym reuses with no code. The one path
cosmos does not route through the registry is the direct
``rollout_buffer.queue.clear()`` (end-of-run / web-panel buffer clear), so this
module installs a single Controller-only shim that publishes cleanup for the
queued rollouts before that clear, preventing leaked sender buffers.
"""

from collections import deque
from typing import Any, Callable

from cosmos_rl.dispatcher.status import PolicyStatusManager


def install_cosmos_nccl_cleanup_publisher_opt_in() -> None:
    """Wrap Cosmos's direct rollout-buffer clear so it publishes NCCL discard cleanup.

    cosmos's ``filter_outdated_rollouts`` already publishes cleanup on its own
    for the outdated-weight discard path, so only the direct
    ``rollout_buffer.queue.clear()`` path needs a wrapper. Idempotent: the guard
    reads a marker off the patched ``setup`` itself, so a repeat install is a
    no-op without tracking separate process-global state.
    """
    original_setup = PolicyStatusManager.setup
    if getattr(original_setup, "_alpagym_cleanup_wrap", False):
        return

    def setup_with_alpagym_nccl_cleanup(self: object, *args: Any, **kwargs: Any) -> Any:
        """Run Cosmos setup, then wrap the rollout buffer's direct clear path."""
        result = original_setup(self, *args, **kwargs)
        wrap_rollout_buffer_clear(self)
        return result

    setup_with_alpagym_nccl_cleanup._alpagym_cleanup_wrap = True  # type: ignore[attr-defined]
    PolicyStatusManager.setup = setup_with_alpagym_nccl_cleanup


def wrap_rollout_buffer_clear(policy_status_manager: PolicyStatusManager) -> None:
    """Publish NCCL cleanup before Cosmos directly clears the rollout buffer."""
    # This shim wraps only the aggregate ``rollout_buffer.queue``. cosmos routes
    # discards through per-rank buffers (``rollout_buffer_per_rank``) when
    # ``data_dispatch_as_rank_in_mesh`` is set, and those have no cleanup path, so a
    # clear there would silently leak sender buffers. AlpaGym never sets the flag;
    # fail fast if that ever changes rather than leaking invisibly.
    if policy_status_manager.config.train.train_policy.data_dispatch_as_rank_in_mesh:
        raise RuntimeError(
            "NCCL cleanup wraps only the aggregate rollout buffer, but "
            "train.train_policy.data_dispatch_as_rank_in_mesh routes discards through "
            "per-rank buffers with no cleanup path. Disable it for NCCL transport."
        )
    rollout_buffer = policy_status_manager.rollout_buffer
    with rollout_buffer.mutex:
        if isinstance(rollout_buffer.queue, _CleanupPublishingDeque):
            return
        rollout_buffer.queue = _CleanupPublishingDeque(
            rollout_buffer.queue,
            policy_status_manager._publish_payload_transport_cleanup,
        )


class _CleanupPublishingDeque(deque):
    """A deque that publishes cleanup for queued rollouts before clear()."""

    def __init__(
        self,
        values: deque[Any],
        cleanup_publisher: Callable[[list[Any], list[Any]], None],
    ) -> None:
        """Copy ``values`` and store cosmos's payload-transport cleanup publisher."""
        super().__init__(values, maxlen=values.maxlen)
        self._cleanup_publisher = cleanup_publisher

    def clear(self) -> None:
        """Publish cleanup for all queued items before clearing them."""
        discarded = list(self)
        if discarded:
            # cosmos's publisher takes (rollouts, filtered_rollouts); the
            # direct-clear path discards everything, so nothing is filtered out.
            self._cleanup_publisher(discarded, [])
        super().clear()
