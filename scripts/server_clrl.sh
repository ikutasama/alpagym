#!/usr/bin/env bash
# Prepare and run Alpamayo 1.5 closed-loop RL on a two-GPU Linux server.

set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${1:-preflight}"
model_path="${MODEL_PATH:-$repo_root/tmp/checkpoints/alpamayo-1.5-10B_alpagym_ckpt}"
release_path="${RELEASE_PATH:-$repo_root/tmp/checkpoints/Alpamayo-1.5-10B}"
min_vram_mib="${MIN_VRAM_MIB:-40000}"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
note() { printf '\n==> %s\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

usage() {
  cat <<'EOF'
Usage: scripts/server_clrl.sh prepare|preflight|p2p|smoke|train

Environment overrides:
  MODEL_PATH       Converted AlpaGym checkpoint directory
  RELEASE_PATH     Download directory for the released HF checkpoint
  MAX_NUM_STEPS    Override training steps (train mode only)
  SCENE_ID         Override the default NuRec scene
  ENABLE_WANDB=1   Enable console + W&B logging
  SKIP_P2P_CHECK=1 Skip the two-GPU NCCL probe during preflight
  NCCL_P2P_DISABLE=1  Use shared-memory NCCL if direct P2P is broken
  ALLOW_LOW_VRAM=1 Continue when either GPU has less than MIN_VRAM_MIB
EOF
}

check_model() {
  [[ -d "$model_path" ]] || die "converted checkpoint not found: $model_path (run prepare)"
  [[ -f "$model_path/config.json" ]] || die "checkpoint has no config.json: $model_path"
  compgen -G "$model_path/*.safetensors" >/dev/null \
    || compgen -G "$model_path/*.bin" >/dev/null \
    || [[ -f "$model_path/model.safetensors.index.json" ]] \
    || die "checkpoint has no supported weight files: $model_path"
  [[ -f "$model_path/tokenizer_config.json" ]] \
    || die "checkpoint has no tokenizer_config.json; reconvert it with prepare"
}

check_gpus() {
  need nvidia-smi
  mapfile -t gpu_memory < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  (( ${#gpu_memory[@]} >= 2 )) || die "Alpamayo 1.5 local training requires at least 2 visible GPUs"

  local i
  for i in 0 1; do
    if (( gpu_memory[i] < min_vram_mib )); then
      [[ "${ALLOW_LOW_VRAM:-0}" == "1" ]] \
        || die "GPU $i has ${gpu_memory[i]} MiB; recommended minimum is ${min_vram_mib} MiB (set ALLOW_LOW_VRAM=1 to try anyway)"
    fi
  done
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
}

run_p2p() {
  note "Checking NCCL point-to-point transport"
  timeout 45s uv run --no-sync torchrun --nproc-per-node=2 install/check_nccl_p2p.py \
    || die "NCCL P2P failed or hung. Retry with: NCCL_P2P_DISABLE=1 scripts/server_clrl.sh p2p"
}

preflight() {
  [[ "$(uname -s)" == "Linux" ]] || die "the CUDA workspace is Linux x86_64 only"
  need uv
  need docker
  need redis-server
  need git-lfs
  check_gpus
  install/check_env.sh
  check_model
  if [[ "${SKIP_P2P_CHECK:-0}" != "1" ]]; then
    run_p2p
  fi
  note "Preflight passed"
}

prepare() {
  [[ "$(uname -s)" == "Linux" ]] || die "the CUDA workspace is Linux x86_64 only"
  need uv
  install/check_env.sh
  note "Syncing the pinned workspace"
  uv sync --all-packages

  mkdir -p "$(dirname "$release_path")" "$(dirname "$model_path")"
  if [[ ! -f "$release_path/config.json" ]]; then
    note "Downloading nvidia/Alpamayo-1.5-10B"
    uv run --no-sync python -c \
      "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Alpamayo-1.5-10B', local_dir=r'$release_path')"
  else
    note "Release checkpoint already exists; skipping download"
  fi

  if [[ ! -f "$model_path/tokenizer_config.json" ]]; then
    note "Converting release checkpoint for AlpaGym/Cosmos-RL"
    uv run --no-sync --package alpagym-runtime python \
      packages/policies/alpamayo_r1/scripts/convert_release_to_alpagym_checkpoint.py \
      --input "$release_path" --output "$model_path" --overwrite
  else
    note "Converted checkpoint already exists; skipping conversion"
  fi
  check_model
  note "Preparation complete; run: scripts/server_clrl.sh smoke"
}

launch() {
  local experiment="$1"
  preflight

  local -a overrides=(
    "experiment=$experiment"
    "policy.model.path=$model_path"
    "reward=progress_safety"
  )
  [[ -z "${SCENE_ID:-}" ]] || overrides+=("dataset.scene_ids=[$SCENE_ID]")
  if [[ "$experiment" == "alpamayo_1_5_local_2gpu_train" && -n "${MAX_NUM_STEPS:-}" ]]; then
    [[ "$MAX_NUM_STEPS" =~ ^[1-9][0-9]*$ ]] || die "MAX_NUM_STEPS must be a positive integer"
    overrides+=("cosmos.train.max_num_steps=$MAX_NUM_STEPS")
  fi
  if [[ "${ENABLE_WANDB:-0}" == "1" ]]; then
    [[ -n "${WANDB_API_KEY:-}" ]] || die "ENABLE_WANDB=1 requires WANDB_API_KEY"
    overrides+=("cosmos.logging.logger=[console,wandb]")
  fi

  note "Launching $experiment"
  exec uv run --no-sync --all-packages python -m alpagym_host.cli "${overrides[@]}"
}

case "$mode" in
  prepare) prepare ;;
  preflight) preflight ;;
  p2p) need uv; check_gpus; run_p2p ;;
  smoke) launch alpamayo_1_5_local_2gpu_smoke ;;
  train) launch alpamayo_1_5_local_2gpu_train ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
