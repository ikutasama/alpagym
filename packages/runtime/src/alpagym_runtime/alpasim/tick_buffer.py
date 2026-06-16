# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from alpasim_grpc.v0.common_pb2 import Trajectory as ProtoTrajectory
from alpasim_grpc.v0.egodriver_pb2 import RolloutCameraImage, Route


@dataclass
class TickBuffer:
    """Per-tick proto holder."""

    camera_images: list[RolloutCameraImage.CameraImage] = field(default_factory=list)
    ego_trajectory: ProtoTrajectory | None = None
    route: Route | None = None

    def add_ego_trajectory(self, trajectory: ProtoTrajectory) -> None:
        """Append egomotion observations to this drive-window buffer."""
        if self.ego_trajectory is None:
            self.ego_trajectory = ProtoTrajectory()
        self.ego_trajectory.poses.extend(trajectory.poses)
