#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

CI_CUDA_VERSION="${CI_CUDA_VERSION:-12.8.1}"
CI_UBUNTU_VERSION="${CI_UBUNTU_VERSION:-22.04}"

image="${ALPAGYM_IMAGE:-alpagym:local}"

build_args=(
  --platform linux/amd64 \
  --tag "$image" \
  --build-arg CI_CUDA_VERSION="${CI_CUDA_VERSION}" \
  --build-arg CI_UBUNTU_VERSION="${CI_UBUNTU_VERSION}" \
  --file install/Dockerfile
)

if [ -f "$HOME/.netrc" ]; then
  build_args+=("--secret" "id=netrc,src=$HOME/.netrc")
fi

docker build "${build_args[@]}" .
