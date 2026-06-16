# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Final

CROSS_LEFT_CAMERA_NAME: Final = "camera_cross_left_120fov"
CROSS_RIGHT_CAMERA_NAME: Final = "camera_cross_right_120fov"
FRONT_TELE_CAMERA_NAME: Final = "camera_front_tele_30fov"
FRONT_WIDE_CAMERA_NAME: Final = "camera_front_wide_120fov"
REAR_LEFT_CAMERA_NAME: Final = "camera_rear_left_70fov"
REAR_RIGHT_CAMERA_NAME: Final = "camera_rear_right_70fov"
REAR_TELE_CAMERA_NAME: Final = "camera_rear_tele_30fov"

# Camera indices will be used for computing camera embeddings.
CAMERA_NAMES_TO_INDICES = {
    CROSS_LEFT_CAMERA_NAME: 0,
    FRONT_WIDE_CAMERA_NAME: 1,
    CROSS_RIGHT_CAMERA_NAME: 2,
    REAR_LEFT_CAMERA_NAME: 3,
    REAR_TELE_CAMERA_NAME: 4,
    REAR_RIGHT_CAMERA_NAME: 5,
    FRONT_TELE_CAMERA_NAME: 6,
}
CAMERA_NAMES_TO_DISPLAY_NAMES = {
    CROSS_LEFT_CAMERA_NAME: "Front left camera",
    CROSS_RIGHT_CAMERA_NAME: "Front right camera",
    FRONT_WIDE_CAMERA_NAME: "Front camera",
    FRONT_TELE_CAMERA_NAME: "Front telephoto camera",
    REAR_LEFT_CAMERA_NAME: "Rear left camera",
    REAR_TELE_CAMERA_NAME: "Rear camera",
    REAR_RIGHT_CAMERA_NAME: "Rear right camera",
}
CAMERA_INDICES_TO_DISPLAY_NAMES = {
    idx: CAMERA_NAMES_TO_DISPLAY_NAMES[name] for name, idx in CAMERA_NAMES_TO_INDICES.items()
}

S3_REGION: Final = "us-east-1"
"""The region of the S3/SwiftStack bucket/container."""
