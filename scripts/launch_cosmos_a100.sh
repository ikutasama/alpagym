#!/usr/bin/env bash
set -euo pipefail

LATEST_DIR=/data/mnt_m62/10_personal/z59900495/workspace/latest
ALPAGYM_DIR=/data/mnt_m62/10_personal/z59900495/workspace/alpagym
TMPDIR=/data/mnt_m62/10_personal/z59900495/workspace/tmp

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

echo "[2/4] Setting full training parameters in cosmos_config.toml ..."
sed -i \
  -e 's|^max_num_steps = .*|max_num_steps = 916|' \
  -e 's|^epoch = .*|epoch = 3|' \
  -e 's|^save_freq = .*|save_freq = 50|' \
  -e 's|^experiment_name = .*|experiment_name = "autovla_full_train"|' \
  -e 's|^n_generation = .*|n_generation = 4|' \
  -e 's|^train_batch_per_replica = .*|train_batch_per_replica = 4|' \
  -e 's|^max_response_length = .*|max_response_length = 500|' \
  "$LATEST_DIR/cosmos_config.toml"

# Set 4-GPU FSDP: both policy and rollout must match for colocated mode
sed -i 's/dp_shard_size.*=.*/dp_shard_size = 4/' "$LATEST_DIR/cosmos_config.toml"
grep -A1 '\[rollout.parallelism\]' "$LATEST_DIR/cosmos_config.toml" | grep -q 'dp_shard_size' || \
  sed -i '/\[rollout.parallelism\]/a dp_shard_size = 4' "$LATEST_DIR/cosmos_config.toml"

sed -i \
  -e 's|max_num_steps: .*|max_num_steps: 916|' \
  -e 's|num_epochs: .*|num_epochs: 3|' \
  -e 's|save_freq: .*|save_freq: 50|' \
  -e 's|experiment_name: autovla_local_smoke|experiment_name: autovla_full_train|' \
  "$LATEST_DIR/resolved_config.yaml"

# Set 4-GPU FSDP sharding
sed -i '/\[policy\.parallelism\]/,/^$/{s/^dp_shard_size = .*/dp_shard_size = 4/}' "$LATEST_DIR/cosmos_config.toml"
sed -i '/\[rollout\.parallelism\]/,/^$/{s/^dp_shard_size = .*/dp_shard_size = 4/}' "$LATEST_DIR/cosmos_config.toml"

echo "  max_num_steps=916, epoch=3, save_freq=50, n_generation=4, batch=2, max_response=200, dp_shard=4"

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
export no_proxy='' NO_PROXY='' TMPDIR="$TMPDIR" ALPAGYM_DRIVER_HOST=localhost ALPAGYM_DRIVER_PORT=5012
cd "$ALPAGYM_DIR"
AUTOVLA_REPO_PATH=/data/mnt_m62/10_personal/z59900495/workspace/AutoVLA \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES=2,3,4,5 UV_NO_MANAGED_PYTHON=1 UV_PYTHON=$(which python) \
uv run --no-sync --all-packages python -m cosmos_rl.launcher.launch_all \
  --config "$LATEST_DIR/cosmos_config.toml" \
  --policy 1 --rollout 1 --num-workers 1 --worker-idx 0 --port 29500 \
  --log-dir "$LATEST_DIR/logs" \
  alpagym_runtime.cosmos.entrypoint