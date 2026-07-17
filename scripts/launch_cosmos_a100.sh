#!/usr/bin/env bash
set -euo pipefail

LATEST_DIR=/data/mnt_m62/10_personal/z59900495/workspace/latest
ALPAGYM_DIR=/data/mnt_m62/10_personal/z59900495/workspace/alpagym
TMPDIR=/data/mnt_m62/10_personal/z59900495/workspace/tmp
RUNTIME_PORT="${RUNTIME_PORT:-5011}"
DRIVER_PORT="${DRIVER_PORT:-5012}"

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [1gpu|4gpu|/path/to/profile.yaml]" >&2
  exit 2
fi

PROFILE_SELECTOR="${1:-${AUTOVLA_A100_PROFILE:-4gpu}}"
case "$PROFILE_SELECTOR" in
  1gpu)
    PROFILE_PATH="$ALPAGYM_DIR/packages/policies/autovla/src/alpagym_autovla/configs/a100/autovla_a100_1gpu.yaml"
    ;;
  4gpu)
    PROFILE_PATH="$ALPAGYM_DIR/packages/policies/autovla/src/alpagym_autovla/configs/a100/autovla_a100_4gpu.yaml"
    ;;
  *)
    PROFILE_PATH="$PROFILE_SELECTOR"
    ;;
esac
if [ ! -f "$PROFILE_PATH" ]; then
  echo "ERROR: AutoVLA A100 profile not found: $PROFILE_PATH" >&2
  exit 2
fi

cd "$ALPAGYM_DIR"
PROFILE_FIELDS=$(UV_NO_MANAGED_PYTHON=1 UV_PYTHON="$(command -v python)" \
  uv run --no-sync --all-packages python -m alpagym_host.autovla_a100_config \
  --profile "$PROFILE_PATH" --print-launch-fields)
IFS=$'\t' read -r PROFILE_CUDA POLICY_REPLICAS ROLLOUT_REPLICAS COSMOS_MODE TRANSPORT_KIND PROFILE_NAME <<< "$PROFILE_FIELDS"
CUDA_VISIBLE_DEVICES="$PROFILE_CUDA"

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if [ "$COSMOS_MODE" = "colocated" ]; then
  EXPECTED_GPU_COUNT=1
else
  EXPECTED_GPU_COUNT=$((POLICY_REPLICAS + ROLLOUT_REPLICAS))
fi
if [ "${#GPU_IDS[@]}" -ne "$EXPECTED_GPU_COUNT" ]; then
  echo "ERROR: profile $PROFILE_NAME ($COSMOS_MODE) requires $EXPECTED_GPU_COUNT visible" \
    "GPU(s), but CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES exposes ${#GPU_IDS[@]}." >&2
  exit 2
fi

echo "[1/4] Loading $PROFILE_NAME and fixing copied-run paths ..."
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

echo "[2/4] Configuring matched rollout and policy geometry ..."
cd "$ALPAGYM_DIR"
CONFIG_ARGS=(--profile "$PROFILE_PATH" --run-dir "$LATEST_DIR")
if [ -n "${MAX_NUM_STEPS:-}" ]; then
  CONFIG_ARGS+=(--max-num-steps "$MAX_NUM_STEPS")
fi
if [ -n "${SAVE_FREQ:-}" ]; then
  CONFIG_ARGS+=(--save-freq "$SAVE_FREQ")
fi
UV_NO_MANAGED_PYTHON=1 UV_PYTHON="$(command -v python)" \
uv run --no-sync --all-packages python -m alpagym_host.autovla_a100_config \
  "${CONFIG_ARGS[@]}"

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
if [ "$TRANSPORT_KIND" = "nccl" ]; then
  export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,^docker}"
  export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
else
  unset NCCL_SHM_DISABLE NCCL_DEBUG NCCL_IB_DISABLE NCCL_SOCKET_IFNAME
  unset NCCL_TIMEOUT NCCL_P2P_DISABLE TORCH_NCCL_ASYNC_ERROR_HANDLING
fi
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
