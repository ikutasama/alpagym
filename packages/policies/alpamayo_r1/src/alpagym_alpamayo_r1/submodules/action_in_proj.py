# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from alpamayo_r1.models.action_in_proj import (
    PerWaypointActionInProjV2 as _PublicPerWaypointActionInProjV2,
)


class PerWaypointActionInProjV2(_PublicPerWaypointActionInProjV2):
    """Improved per-waypoint action input projection module.

    It uses FourierEncoderV2 with logarithmically-spaced frequencies and includes layer
    normalization. Projects action sequences with timestep information into a
    higher-dimensional representation.
    """

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass of the per-waypoint action projection V2.

        Args:
            x: Action tensor of shape (batch_size, num_waypoints, action_dim).
            timesteps: Timestep tensor of shape (batch_size, ...). The last dimension
                is used for encoding.

        Returns:
            Normalized projected action features of shape
            (batch_size, num_waypoints, out_dim).
        """
        B, T, _ = x.shape

        x = x.float()
        timesteps = timesteps.float()

        action_feats = torch.cat([s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1)
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
        timestep_feats = timestep_feats.repeat(1, T, 1)
        x = torch.cat((action_feats, timestep_feats), dim=-1)
        # ``x = x.float()`` above keeps the Fourier features in fp32 for stable
        # sin/cos accumulation; cast back to the encoder weight dtype here so
        # mixed-precision (bf16 encoder + fp32 features) matmul does not raise
        # "mat1 and mat2 must have the same dtype".
        x = x.to(self.encoder.trunk[0].weight.dtype)
        return self.norm(self.encoder(x.flatten(0, 1)).reshape(B, T, -1))
