#!/usr/bin/env bash
set -euo pipefail

cd "${ALPAGYM_ROOT:-$HOME/alpagym}"

# Clean up previous smoke run artifacts (each smoke is self-contained; failed
# runs leave ~3-6GB of JSON per rollout that is never read back).
find "${ALPAGYM_ROOT:-$HOME/alpagym}/tmp/alpagym-runs" -mindepth 1 -delete 2>/dev/null || true

docker compose down 2>/dev/null || true

export GRPC_ARG_ENABLE_HTTP_PROXY="${GRPC_ARG_ENABLE_HTTP_PROXY:-0}"
export grpc_proxy=""
export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"

export UV_NO_MANAGED_PYTHON="${UV_NO_MANAGED_PYTHON:-1}"
export UV_PYTHON="${UV_PYTHON:-$(command -v python)}"
export UV_SYSTEM_CERTS="${UV_SYSTEM_CERTS:-true}"
export UV_INSECURE_HOST="${UV_INSECURE_HOST:-github.com codeload.github.com objects.githubusercontent.com files.pythonhosted.org pypi.org download.pytorch.org download-r2.pytorch.org static.crates.io index.crates.io crates.io}"
export GIT_SSL_NO_VERIFY="${GIT_SSL_NO_VERIFY:-1}"
export CARGO_HTTP_CAINFO="${CARGO_HTTP_CAINFO:-$HOME/.config/cargo/cargo-ca.pem}"
export CARGO_HTTP_PROXY_CAINFO="${CARGO_HTTP_PROXY_CAINFO:-$HOME/.config/cargo/cargo-ca.pem}"
export CARGO_HTTP_CHECK_REVOKE="${CARGO_HTTP_CHECK_REVOKE:-false}"
export CARGO_NET_GIT_FETCH_WITH_CLI="${CARGO_NET_GIT_FETCH_WITH_CLI:-true}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Install camera config into the AlPaSim wizard configs directory so Hydra can
# resolve +cameras=${CAMERAS_PRESET}.
CAMERAS_PRESET="${CAMERAS_PRESET:-4cam_1080}"
CAM_SRC="${ALPAGYM_ROOT:-$HOME/alpagym}/scripts/cameras/${CAMERAS_PRESET}.yaml"
CAMERA_DIR="src/wizard/configs/cameras"

COPY_TARGETS=()
if [ -n "${ALPASIM_ROOT:-}" ] && [ -d "${ALPASIM_ROOT}/${CAMERA_DIR}" ]; then
  COPY_TARGETS+=("${ALPASIM_ROOT}/${CAMERA_DIR}/${CAMERAS_PRESET}.yaml")
fi

CACHE_BASE="${XDG_CACHE_HOME:-$HOME/.cache}/alpagym/alpasim"
for d in "${CACHE_BASE}"/*/; do
  if [ -d "${d}${CAMERA_DIR}" ]; then
    COPY_TARGETS+=("${d}${CAMERA_DIR}/${CAMERAS_PRESET}.yaml")
  fi
done

for dst in "${COPY_TARGETS[@]}"; do
  if [ -f "${CAM_SRC}" ]; then
    cp -f "${CAM_SRC}" "${dst}"
    echo "Installed ${CAMERAS_PRESET}.yaml -> ${dst}"
  fi
done

# HF token
if [ -z "${HF_TOKEN:-}" ]; then
  if [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
  elif [ -f "$HOME/.alpagym_env" ]; then
    source "$HOME/.alpagym_env"
  fi
fi

EXPERIMENT="${EXPERIMENT:-alpamayo_1_5_local_2gpu_smoke}"
REWARD="${REWARD:-progress_safety}"
MODEL_PATH="${MODEL_PATH:-/mnt/mnt_m181/z59900495/workspace/model/Alpamayo-1.5-10B}"
ALPASIM_EXTRA_OVERRIDES="${ALPASIM_EXTRA_OVERRIDES:-+cameras=${CAMERAS_PRESET} runtime.simulation_config.pose_reporting_interval_us=100000 scenes.local_usdz_dir=/mnt/mnt_m181/z59900495/workspace/DownloadTool-master/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec wizard.runtime_server_port=5011 topology=1gpu}"

exec uv run --no-sync --all-packages python -m alpagym_host.cli \
  "experiment=${EXPERIMENT}" \
  "policy.model.path=${MODEL_PATH}" \
  "reward=${REWARD}" \
  "alpasim.wizard_args.extra_overrides=\"${ALPASIM_EXTRA_OVERRIDES}\""
