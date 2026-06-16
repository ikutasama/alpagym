# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NCCL transport for completed rollout artifacts: one transfer moves one episode over NCCL.

Wire-format dataclasses + ``TCPStore`` key builders live in :mod:`protocol`;
the ``EpisodeOutput`` pack/unpack helpers live in :mod:`payload`.
"""
