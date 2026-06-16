# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-tick observation buffers owned by `AlpamayoPolicy`."""

import collections
import logging
from dataclasses import dataclass

import torch
import torchvision.io
import torchvision.transforms.functional as tvf

from alpagym_runtime.types import ChosenTrajectory, EgoPose, PolicyInput, RouteWaypoint

logger = logging.getLogger(__name__)


class CameraFrameBuffer:
    """Fixed-capacity ring buffer of decoded camera frames for one camera."""

    def __init__(
        self,
        capacity: int,
        frame_shape: tuple[int, int, int],
        camera_name: str,
        frame_device: torch.device,
        frame_dtype: torch.dtype = torch.uint8,
        tstamps_device: torch.device = torch.device("cpu"),
    ) -> None:
        """Allocate fixed-capacity storage for one camera's decoded frames."""
        if capacity <= 0:
            raise ValueError(f"Camera frame buffer capacity must be positive, got {capacity}.")
        self.capacity = capacity
        self.camera_name = camera_name
        self.frames_buffer = torch.empty(
            (capacity, *frame_shape),
            dtype=frame_dtype,
            device=frame_device,
        )
        self.tstamps_buffer = torch.empty((capacity,), dtype=torch.int64, device=tstamps_device)
        self.write_idx = 0
        self._ordered_cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def add(self, frame: torch.Tensor, timestamp_us: int | torch.Tensor) -> None:
        """Add one CHW frame and invalidate any cached ordered view."""
        if frame.dtype != self.frames_buffer.dtype or frame.ndim != 3:
            raise ValueError(
                f"Expected {self.frames_buffer.dtype} CHW camera frame for "
                f"{self.camera_name!r}, got dtype={frame.dtype}, "
                f"shape={tuple(frame.shape)}."
            )
        if tuple(self.frames_buffer.shape[1:]) != tuple(frame.shape):
            raise ValueError(
                f"Camera frame shape changed for {self.camera_name!r}: "
                f"{tuple(self.frames_buffer.shape[1:])} -> {tuple(frame.shape)}."
            )
        if frame.device != self.frames_buffer.device:
            raise ValueError(
                f"Camera frame device changed for {self.camera_name!r}: "
                f"{self.frames_buffer.device} -> {frame.device}."
            )
        slot = self.write_idx % self.capacity
        self.frames_buffer[slot] = frame
        self.tstamps_buffer[slot] = timestamp_us
        self.write_idx += 1
        self._ordered_cache = None

    def __len__(self) -> int:
        """Number of valid frames in the ring buffer."""
        return min(self.write_idx, self.capacity)

    def clear(self) -> None:
        """Drop valid frames while keeping the allocated ring storage."""
        self.write_idx = 0
        self._ordered_cache = None

    @property
    def frames_ordered(self) -> torch.Tensor:
        """Frames in chronological order as `[T, C, H, W]`."""
        return self._ordered_view()[0]

    @property
    def tstamps_ordered(self) -> torch.Tensor:
        """Frame timestamps in chronological order as `[T]` int64."""
        return self._ordered_view()[1]

    def _ordered_view(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return frames and timestamps oldest first."""
        if len(self) < self.capacity:
            return self.frames_buffer[: len(self)], self.tstamps_buffer[: len(self)]

        start = self.write_idx % self.capacity
        if start == 0:
            return self.frames_buffer, self.tstamps_buffer

        if self._ordered_cache is None:
            order = list(range(start, self.capacity)) + list(range(start))
            self._ordered_cache = (
                self.frames_buffer[order],
                self.tstamps_buffer[order],
            )
        return self._ordered_cache


@dataclass
class RouteState:
    """Latest route waypoints plus the simulator timestamp they were emitted at."""

    waypoints: tuple[RouteWaypoint, ...]
    timestamp_us: int


class EgomotionBuffer:
    """Fixed-capacity ego-pose buffer with strictly increasing timestamps."""

    def __init__(self, capacity: int) -> None:
        """Allocate an empty ego-pose ring."""
        self._poses: collections.deque[EgoPose] = collections.deque(maxlen=capacity)

    def add(self, poses: tuple[EgoPose, ...]) -> None:
        """Append ego poses after validating strict timestamp order."""
        previous_timestamp_us = self._poses[-1].timestamp_us if self._poses else None
        for ego_pose in poses:
            if previous_timestamp_us is not None and ego_pose.timestamp_us <= previous_timestamp_us:
                raise ValueError(
                    "Ego pose timestamps must increase strictly; "
                    f"got {ego_pose.timestamp_us} after {previous_timestamp_us}."
                )
            self._poses.append(ego_pose)
            previous_timestamp_us = ego_pose.timestamp_us

    def clear(self) -> None:
        """Drop buffered ego poses."""
        self._poses.clear()

    @property
    def poses(self) -> tuple[EgoPose, ...]:
        """Snapshot of buffered ego poses, oldest first."""
        return tuple(self._poses)


class ObservationBuffers:
    """Per-session ring buffers feeding `AlpamayoPolicy._preprocess`."""

    def __init__(
        self,
        frame_buffer_capacity: int,
        ego_history_capacity: int,
        use_cameras: tuple[str, ...],
        frame_size: tuple[int, int],
        frame_device: torch.device = torch.device("cpu"),
    ) -> None:
        """Allocate empty buffers sized for the per-session configuration."""
        if frame_device.type == "cpu":
            logger.warning(
                "Alpamayo observation buffers are on CPU; camera JPEG decode and frame "
                "buffering will run on CPU (libjpeg) and be slow. Production rollout "
                "expects a CUDA frame device for GPU (nvJPEG) decode."
            )
        self._frame_buffer_capacity = frame_buffer_capacity
        self._frame_device = frame_device
        self._use_cameras = use_cameras
        self._allowed_cameras: frozenset[str] = frozenset(use_cameras)
        self._frame_size = frame_size
        self._cameras = {
            camera_name: CameraFrameBuffer(
                capacity=self._frame_buffer_capacity,
                frame_shape=(3, frame_size[0], frame_size[1]),
                camera_name=camera_name,
                frame_device=self._frame_device,
            )
            for camera_name in use_cameras
        }
        self._ego_history = EgomotionBuffer(capacity=ego_history_capacity)
        self._route: RouteState | None = None
        self._last_chosen_traj: ChosenTrajectory | None = None

    def ingest(self, policy_input: PolicyInput) -> None:
        """Fold one drive-tick `PolicyInput` into the buffers."""
        for camera_image in policy_input.camera_images:
            if camera_image.logical_id not in self._allowed_cameras:
                raise ValueError(
                    f"Unexpected camera {camera_image.logical_id!r}; "
                    f"expected one of {sorted(self._allowed_cameras)!r}"
                )
            frame = self._decode_to_chw_uint8(
                camera_image.image_bytes, self._frame_size, self._frame_device
            )
            ring = self._cameras[camera_image.logical_id]
            ring.add(frame, camera_image.frame_end_us)

        self._ego_history.add(policy_input.ego_trajectory.poses)

        self._route = RouteState(
            waypoints=policy_input.route_waypoints,
            timestamp_us=policy_input.route_timestamp_us,
        )

    def update_last_chosen_traj(self, chosen_traj: ChosenTrajectory) -> None:
        """Record this tick's pick; next tick reads it as the previous trajectory."""
        self._last_chosen_traj = chosen_traj

    def clear(self) -> None:
        """Reset every buffer (frames, ego, route, last chosen trajectory)."""
        for ring in self._cameras.values():
            ring.clear()
        self._ego_history.clear()
        self._route = None
        self._last_chosen_traj = None

    @property
    def ego_history(self) -> tuple[EgoPose, ...]:
        """Snapshot of the ego-history deque, oldest first."""
        return self._ego_history.poses

    @property
    def cameras(self) -> dict[str, CameraFrameBuffer]:
        """Per-camera ring buffers keyed by logical camera id."""
        return dict(self._cameras)

    @property
    def route(self) -> RouteState:
        """The most recent ingested route state."""
        if self._route is None:
            raise ValueError(
                "ObservationBuffers.route accessed before any PolicyInput was ingested."
            )
        return self._route

    @property
    def last_chosen_traj(self) -> ChosenTrajectory | None:
        """The last chosen trajectory, if any has been recorded."""
        return self._last_chosen_traj

    @staticmethod
    def _decode_to_chw_uint8(
        image_bytes: bytes,
        frame_size: tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Decode JPEG bytes into a resized `[3, H, W]` CHW `uint8` tensor on `device`.

        Decoding and resizing both run on `device`: nvJPEG on CUDA, libjpeg
        on CPU. Keeping both on `device` avoids any host<->device copy of
        pixel data on the rollout hot path.
        """
        jpeg = torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8)
        frame = torchvision.io.decode_jpeg(
            jpeg, mode=torchvision.io.ImageReadMode.RGB, device=device
        )
        target_h, target_w = frame_size
        return tvf.resize(frame, [target_h, target_w], antialias=True)
