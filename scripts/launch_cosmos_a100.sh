#!/usr/bin/env bash
set -euo pipefail

LATEST_DIR=/data/mnt_m62/10_personal/z59900495/workspace/latest
ALPAGYM_DIR=/data/mnt_m62/10_personal/z59900495/workspace/alpagym
TMPDIR=/data/mnt_m62/10_personal/z59900495/workspace/tmp

# Four-A100 layout with the existing single 5012 reverse tunnel:
#   GPUs 0-2: three synchronized policy replicas
#   GPU 3:    one rollout replica / one Egodriver server on port 5012
# The rollout worker dispatches 3 prompt groups x 8 generations = 24 episodes,
# matching 3 policy replicas x 8 episodes per replica.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
POLICY_REPLICAS="${POLICY_REPLICAS:-3}"
ROLLOUT_REPLICAS="${ROLLOUT_REPLICAS:-1}"
N_GENERATION="${N_GENERATION:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-3}"
TRAIN_BATCH_PER_REPLICA="${TRAIN_BATCH_PER_REPLICA:-8}"
MINI_BATCH="${MINI_BATCH:-2}"
MAX_INFERENCE_BATCH_SIZE="${MAX_INFERENCE_BATCH_SIZE:-3}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-916}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
RATIO_CLIP="${RATIO_CLIP:-0.1}"
ALLOWED_OUTDATED_STEPS="${ALLOWED_OUTDATED_STEPS:-1}"
SAVE_FREQ="${SAVE_FREQ:-50}"
GRAD_NORM_CLIP="${GRAD_NORM_CLIP:-1.0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-autovla_a100_4gpu_grpo}"
RUNTIME_PORT="${RUNTIME_PORT:-5011}"
DRIVER_PORT="${DRIVER_PORT:-5012}"

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
EXPECTED_GPU_COUNT=$((POLICY_REPLICAS + ROLLOUT_REPLICAS))
if [ "${#GPU_IDS[@]}" -ne "$EXPECTED_GPU_COUNT" ]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES exposes ${#GPU_IDS[@]} GPUs, but policy + rollout" \
    "requires ${EXPECTED_GPU_COUNT}." >&2
  exit 2
fi
if [ "$ROLLOUT_REPLICAS" -ne 1 ]; then
  echo "ERROR: the current 5012 tunnel supports one rollout/Egodriver process." \
    "Add driver ports and tunnels before increasing ROLLOUT_REPLICAS." >&2
  exit 2
fi

echo "[1/4] Fixing paths in resolved_config.yaml and cosmos_config.toml ..."
sed -i \
  -e 's|tmp/alpagym-runs/[^/]*/|'"$LATEST_DIR"'/|g' \
  -e 's|/mnt/mnt_m62|/data/mnt_m62|g' \
  -e 's|/mnt/mnt_m181/z59900495/workspace/model|/tmp/model|g' \
  -e 's|/mnt/mnt_m62/10_personal/z59900495/workspace/DownloadTool-master/Qwen/Qwen2.5-VL-3B-Instruct|/tmp/model/Qwen/Qwen2.5-VL-3B-Instruct|g' \
  -e 's|/mnt/mnt_m62/10_personal/z59900495/workspace/DownloadTool-master/Zewei-Zhou/AutoVLA/AutoVLA_PDMS_89.ckpt|/tmp/model/AutoVLA/AutoVLA_PDMS_89.ckpt|g' \
  "$LATEST_DIR/resolved_config.yaml"

sed -i \
  -e 's|^resolved_config_path.*|resolved_config_path = "'"$LATEST_DIR"'/resolved_config.yaml"|' \
  -e 's|/mnt/mnt_m181/z59900495/workspace/model|/tmp/model|g' \
  -e 's|/mnt/mnt_m62|/data/mnt_m62|g' \
  -e 's|/mnt/mnt_m62/10_personal/z59900495/workspace/DownloadTool-master/Qwen/Qwen2.5-VL-3B-Instruct|/tmp/model/Qwen/Qwen2.5-VL-3B-Instruct|g' \
  -e 's|/mnt/mnt_m62/10_personal/z59900495/workspace/DownloadTool-master/Zewei-Zhou/AutoVLA/AutoVLA_PDMS_89.ckpt|/tmp/model/AutoVLA/AutoVLA_PDMS_89.ckpt|g' \
  "$LATEST_DIR/cosmos_config.toml"

echo "[2/4] Configuring matched four-GPU rollout and policy geometry ..."
cd "$ALPAGYM_DIR"
UV_NO_MANAGED_PYTHON=1 UV_PYTHON="$(command -v python)" \
uv run --no-sync --all-packages python -m alpagym_host.autovla_a100_config \
  --run-dir "$LATEST_DIR" \
  --policy-replicas "$POLICY_REPLICAS" \
  --rollout-replicas "$ROLLOUT_REPLICAS" \
  --n-generation "$N_GENERATION" \
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
  --train-batch-per-replica "$TRAIN_BATCH_PER_REPLICA" \
  --mini-batch "$MINI_BATCH" \
  --max-inference-batch-size "$MAX_INFERENCE_BATCH_SIZE" \
  --max-num-steps "$MAX_NUM_STEPS" \
  --num-epochs "$NUM_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --warmup-steps "$WARMUP_STEPS" \
  --ratio-clip "$RATIO_CLIP" \
  --allowed-outdated-steps "$ALLOWED_OUTDATED_STEPS" \
  --save-freq "$SAVE_FREQ" \
  --grad-norm-clip "$GRAD_NORM_CLIP" \
  --experiment-name "$EXPERIMENT_NAME"

echo "[3/4] Creating run directory symlinks on A100 ..."
RUN_ID=$(grep -oP 'alpagym-runs/\K[^/]+' "$LATEST_DIR/resolved_config.yaml" 2>/dev/null | head -1 || true)
if [ -n "$RUN_ID" ]; then
  RUN_DIR="$ALPAGYM_DIR/tmp/alpagym-runs/$RUN_ID"
  mkdir -p "$RUN_DIR"
  for f in alpasim_scene_ids.yaml resolved_config.yaml cosmos_config.toml; do
    [ -f "$LATEST_DIR/$f" ] && ln -sfn "$LATEST_DIR/$f" "$RUN_DIR/$f"
  done
  for d in topology alpasim artifacts logs perf; do
    [ -e "$LATEST_DIR/$d" ] && ln -sfn "$LATEST_DIR/$d" "$RUN_DIR/$d"
  done
  echo "  Linked run dir: $RUN_DIR"
else
  echo "  WARNING: Could not extract run ID from resolved_config.yaml"
fi

echo "[4/4] Launching cosmos ..."
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY grpc_proxy GRPC_PROXY all_proxy ALL_PROXY
export GRPC_ARG_ENABLE_HTTP_PROXY=0 no_proxy='localhost,127.0.0.1,0.0.0.0' NO_PROXY='localhost,127.0.0.1,0.0.0.0'
export TMPDIR="$TMPDIR" GLOO_TIMEOUT_SECONDS="${GLOO_TIMEOUT_SECONDS:-3600}"
export ALPAGYM_DRIVER_HOST=localhost ALPAGYM_DRIVER_PORT="$DRIVER_PORT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,^docker}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_VISIBLE_DEVICES

python - "$RUNTIME_PORT" "$DRIVER_PORT" <<'PY'
import socket
import sys

runtime_port = int(sys.argv[1])
driver_port = int(sys.argv[2])
with socket.create_connection(("127.0.0.1", runtime_port), timeout=3.0):
    pass
probe = socket.socket()
try:
    probe.bind(("127.0.0.1", driver_port))
finally:
    probe.close()
print(f"  Tunnel/runtime ready on {runtime_port}; driver port {driver_port} is free")
PY

cd "$ALPAGYM_DIR"
AUTOVLA_REPO_PATH=/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
UV_NO_MANAGED_PYTHON=1 UV_PYTHON="$(command -v python)" \
uv run --no-sync --all-packages python -m cosmos_rl.launcher.launch_all \
  --config "$LATEST_DIR/cosmos_config.toml" \
  --policy "$POLICY_REPLICAS" --rollout "$ROLLOUT_REPLICAS" \
  --num-workers 1 --worker-idx 0 --port 29500 \
  --log-dir "$LATEST_DIR/logs" \
  alpagym_runtime.cosmos.entrypoint
