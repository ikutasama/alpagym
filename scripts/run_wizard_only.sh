#!/usr/bin/env bash
# Generic wizard-only script (run on the 5090 / rendering machine).
# Starts AlPaSim Wizard for rendering + simulation, then blocks until Ctrl+C.
# Does NOT specify model path — that is chosen on the A100 side.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./scripts/run_wizard_only.sh [experiment_name]
#
# Default experiment: alpamayo_1_5_local_2gpu_smoke
# The experiment only determines AlPaSim settings (cameras, control_timestep,
# n_sim_steps, scenes). Model and training params are overridden on the A100.
set -euo pipefail
export ALPAGYM_WIZARD_ONLY=1

EXPERIMENT="${1:-alpamayo_1_5_local_2gpu_smoke}"

cd "${ALPAGYM_ROOT:-$HOME/alpagym}"

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Install camera config
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

ALPASIM_EXTRA_OVERRIDES="${ALPASIM_EXTRA_OVERRIDES:-+cameras=${CAMERAS_PRESET} runtime.simulation_config.pose_reporting_interval_us=100000 scenes.local_usdz_dir=/mnt/mnt_m181/z59900495/workspace/DownloadTool-master/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec wizard.runtime_server_port=5011}"

# Use a dummy model path to satisfy Hydra config resolution.
# Wizard-only mode skips model path validation (see config_validation.py).
DUMMY_MODEL_PATH="/tmp/alpagym_dummy_model"
mkdir -p "${DUMMY_MODEL_PATH}"

exec uv run --no-sync --all-packages python -m alpagym_host.cli \
  "experiment=${EXPERIMENT}" \
  "policy.model.path=${DUMMY_MODEL_PATH}" \
  "reward=progress_safety" \
  "alpasim.wizard_args.extra_overrides=\"${ALPASIM_EXTRA_OVERRIDES}\""
