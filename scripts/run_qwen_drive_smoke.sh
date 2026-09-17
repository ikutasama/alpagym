#!/usr/bin/env bash
set -euo pipefail

# Qwen-Drive closed-loop smoke test using a pre-existing remote AlPaSim runtime.
# The AlPaSim runtime is already running on a remote machine, tunneled to
# localhost:5011. We bypass the wizard entirely and connect the policy
# directly to the existing runtime.
#
# Strategy:
#   1. Run CLI briefly to create the run directory + config files, kill before wizard starts docker
#   2. Manually populate the topology registry with the existing runtime endpoint
#   3. Re-run CLI with ALPAGYM_COSMOS_ONLY=1 + execution.resolved_config_path to launch cosmos

ALPAGYM_ROOT=/data/mnt_m62/10_personal/z59900495/workspace/alpagym
ALPASIM_PATH=/data/mnt_m62/10_personal/z59900495/workspace/alpasim
UPSTREAM_PATH=/data/mnt_m62/10_personal/z59900495/workspace/a_0914_qwendrive/Qwen-Drive-1.0
MODEL_PATH="/tmp/qd_model"
PLANNER_PATH="/tmp/qd_runs/sft_continued_conservative_b4/checkpoints/final"
EXPERIMENT="qwen_drive_a100_1gpu_inference"
CAMERAS_PRESET="front_wide_120fov,cross_left_120fov,cross_right_120fov"
RUNTIME_PORT=5011

# A scene that exists on the remote AlPaSim runtime
SCENE_ID="clipgt-01d503d4-449b-46fc-8d78-9085e70d3554"

cd "${ALPAGYM_ROOT}"

# Clean up previous run artifacts
find tmp/alpagym-runs -mindepth 1 -delete 2>/dev/null || true

# Environment
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export GRPC_ARG_ENABLE_HTTP_PROXY=0
export UV_PROJECT_ENVIRONMENT=/tmp/alpagym_venv
export UV_LINK_MODE=copy
export UV_NO_MANAGED_PYTHON=1
export ALPAGYM_SKIP_ALPASIM_SYNC=1
export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV=/tmp/alpagym_venv
ALPASIM_EXTRA_OVERRIDES="+cameras=${CAMERAS_PRESET} runtime.simulation_config.pose_reporting_interval_us=500000 scenes.local_usdz_dir=/mnt/mnt_m181/z59900495/workspace/DownloadTool-master/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec wizard.runtime_server_port=${RUNTIME_PORT}"

CLI_ARGS=(
  "experiment=${EXPERIMENT}"
  "policy.model.path=${MODEL_PATH}"
  "+policy.model.bundle_config.planner_path=${PLANNER_PATH}"
  "+policy.model.bundle_config.upstream_path=${UPSTREAM_PATH}"
  "alpasim.repo_path=${ALPASIM_PATH}"
  "alpasim.repo_url=null"
  "alpasim.repo_ref=null"
  "dataset.scene_ids=[${SCENE_ID}]"
  "alpasim.wizard_args.extra_overrides=\"${ALPASIM_EXTRA_OVERRIDES}\""
)

LOG_FILE=/tmp/qd_smoke_phase1.log

echo "=== Phase 1: Create run directory (will kill before wizard starts docker) ==="

# Run CLI in background, capture output to log file
"${VENV}/bin/python" -m alpagym_host.cli "${CLI_ARGS[@]}" > "${LOG_FILE}" 2>&1 &
CLI_PID=$!

# Wait for the "Starting AlPaSim Wizard" log line 鈥?this means the run dir is created
# and the wizard is about to start (but docker hasn't started yet)
RUN_DIR=""
for i in $(seq 1 30); do
    sleep 1
    if grep -q "Wizard process" "${LOG_FILE}" 2>/dev/null; then
        # Extract run_dir from the first log line
        RUN_DIR=$(grep "Starting AlpaGym run" "${LOG_FILE}" | head -1 | sed 's/.*run_dir=\([^ ]*\).*/\1/')
        echo "Run directory: ${RUN_DIR}"
        # Wait 1 more second for the wizard to fully initialize (config files are written)
        sleep 2
        break
    fi
    # Check if process already exited (error)
    if ! kill -0 ${CLI_PID} 2>/dev/null; then
        echo "CLI exited early!"
        cat "${LOG_FILE}"
        exit 1
    fi
done

if [ -z "${RUN_DIR}" ]; then
    echo "ERROR: Run directory not detected within 30s"
    cat "${LOG_FILE}"
    kill ${CLI_PID} 2>/dev/null || true
    exit 1
fi

# Kill the CLI process and any wizard subprocess
kill ${CLI_PID} 2>/dev/null || true
sleep 1
kill -9 ${CLI_PID} 2>/dev/null || true
pkill -9 -f "alpasim_wizard" 2>/dev/null || true
pkill -9 -f "uv sync" 2>/dev/null || true
echo "CLI killed (run directory preserved)"

# Verify run directory has config files
ABS_RUN_DIR="${ALPAGYM_ROOT}/${RUN_DIR}"
if [ ! -f "${ABS_RUN_DIR}/resolved_config.yaml" ]; then
    echo "ERROR: resolved_config.yaml not found in ${ABS_RUN_DIR}"
    ls -la "${ABS_RUN_DIR}/"
    exit 1
fi
echo "Config files verified: ${ABS_RUN_DIR}"

echo ""
echo "=== Phase 2: Populate topology registry with existing runtime ==="

# Use Python to fetch runtime info and populate the registry
"${VENV}/bin/python" -W ignore << PYEOF
import grpc
import yaml
from pathlib import Path

run_dir = Path("${ABS_RUN_DIR}")

# Connect to existing AlPaSim runtime
channel = grpc.insecure_channel("localhost:${RUNTIME_PORT}")
from alpasim_grpc.v0 import runtime_pb2, runtime_pb2_grpc
from google.protobuf import empty_pb2
stub = runtime_pb2_grpc.RuntimeServiceStub(channel)
info = stub.get_runtime_info(empty_pb2.Empty(), timeout=10)
scene_ids = [s.scene_id for s in info.scenes]
capacity = info.max_supported_concurrent_rollouts
channel.close()

print(f"Runtime: localhost:${RUNTIME_PORT}, capacity={capacity}, scenes={len(scene_ids)}")

# Publish the runtime endpoint to the topology registry
registry_dir = run_dir / "topology_registry"
registry_dir.mkdir(parents=True, exist_ok=True)
endpoint_path = registry_dir / "alpasim_runtimes" / "alpasim-runtime-0.yaml"
endpoint_path.parent.mkdir(parents=True, exist_ok=True)
endpoint_data = {
    "id": "alpasim-runtime-0",
    "host": "localhost",
    "port": ${RUNTIME_PORT},
    "capacity": capacity,
}
endpoint_path.write_text(yaml.safe_dump(endpoint_data, sort_keys=False))
print(f"Published endpoint: {endpoint_path}")

# Write scene IDs
scene_ids_path = run_dir / "alpasim_scene_ids.yaml"
scene_ids_path.write_text(yaml.safe_dump({"scene_ids": scene_ids}, sort_keys=False))
print(f"Wrote {len(scene_ids)} scene IDs to {scene_ids_path}")
PYEOF

echo ""
echo "=== Phase 3: Launch Cosmos (policy + rollout) with ALPAGYM_COSMOS_ONLY=1 ==="
export ALPAGYM_COSMOS_ONLY=1

# Re-run the CLI with resolved_config_path pointing to the existing run dir
# This loads the existing config and skips wizard (cosmos_only=1)
"${VENV}/bin/python" -m alpagym_host.cli \
  "${CLI_ARGS[@]}" \
  "execution.resolved_config_path=${ABS_RUN_DIR}/resolved_config.yaml" \
  2>&1
COSMOS_EXIT=$?

echo ""
echo "=== Cosmos exit code: ${COSMOS_EXIT} ==="
exit ${COSMOS_EXIT}
