#!/usr/bin/env bash
set -euo pipefail

# Qwen-Drive-1.0 closed-loop inference smoke test (Phase 3)
# Uses the conservative 10K SFT checkpoint on GPU 1.
# AlpaSim tunnel (port 5011) should already be established.

cd /data/mnt_m62/10_personal/z59900495/workspace/alpagym

# Clean up previous run artifacts
find tmp/alpagym-runs -mindepth 1 -delete 2>/dev/null || true

# Environment
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export GRPC_ARG_ENABLE_HTTP_PROXY=0
export UV_PROJECT_ENVIRONMENT=/tmp/alpagym_venv
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES=1

# Model paths
MODEL_PATH="/tmp/qd_model"
PLANNER_PATH="/tmp/qd_runs/sft_continued_conservative_b4/checkpoints/final"
UPSTREAM_PATH="/data/mnt_m62/10_personal/z59900495/workspace/a_0914_qwendrive/Qwen-Drive-1.0"

# Experiment config
EXPERIMENT="qwen_drive_a100_1gpu_inference"

# Camera preset (matching Qwen-Drive's 3-camera setup)
CAMERAS_PRESET="front_wide_120fov,cross_left_120fov,cross_right_120fov"

ALPASIM_EXTRA_OVERRIDES="+cameras=${CAMERAS_PRESET} runtime.simulation_config.pose_reporting_interval_us=500000 scenes.local_usdz_dir=/mnt/mnt_m181/z59900495/workspace/DownloadTool-master/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec wizard.runtime_server_port=5011"

echo "=== Starting Qwen-Drive closed-loop inference ==="
echo "Model: ${MODEL_PATH}"
echo "Planner: ${PLANNER_PATH}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Experiment: ${EXPERIMENT}"

/tmp/alpagym_venv/bin/python -m alpagym_host.cli \
  "experiment=${EXPERIMENT}" \
  "policy.model.path=${MODEL_PATH}" \
  "+policy.model.bundle_config.planner_path=${PLANNER_PATH}" \
  "+policy.model.bundle_config.upstream_path=${UPSTREAM_PATH}" \
  "alpasim.wizard_args.extra_overrides=\"${ALPASIM_EXTRA_OVERRIDES}\""
