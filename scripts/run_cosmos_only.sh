#!/usr/bin/env bash
# Disaggregated mode: Cosmos-only (run on the A100 / training machine).
# Chooses the model and training config — the 5090 wizard is model-agnostic.
#
# Usage:
#   ./scripts/run_cosmos_only.sh <copied_run_dir> \
#     --model-path <path> \
#     [--experiment <name>] \
#     [--checkpoint-path <path>] \
#     [--path-remap <old>:<new>] \
#     [--topology <name>]
#
# Prerequisites:
#   1. SSH tunnel to the Wizard machine (start_ssh_tunnel.sh on 5090)
#   2. Run directory copied from Wizard machine (scp)
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <copied_run_dir> --model-path <path> [--experiment <name>] [--checkpoint-path <path>] [--path-remap <old>:<new>] [--topology <name>]" >&2
  exit 1
fi

RUN_DIR="$(cd "$1" && pwd)"
shift

MODEL_PATH=""
CHECKPOINT_PATH=""
EXPERIMENT=""
PATH_REMAP=""
TOPOLOGY="${TOPOLOGY:-local_disaggregated_8gpu}"
while [ $# -gt 0 ]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --checkpoint-path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --experiment) EXPERIMENT="$2"; shift 2 ;;
    --path-remap) PATH_REMAP="$2"; shift 2 ;;
    --topology) TOPOLOGY="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$MODEL_PATH" ]; then
  echo "Error: --model-path is required" >&2
  exit 1
fi

# Fix absolute paths in resolved_config.yaml so they point to local filesystem.
# Always start from the original backup to avoid double-remap corruption.
# Also override model path and checkpoint path to local values.
uv run --no-sync python -c "
import yaml, sys, os, shutil
run_dir = sys.argv[1]
path_remap = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
model_path = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
checkpoint_path = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
cfg_path = os.path.join(run_dir, 'resolved_config.yaml')
backup_path = os.path.join(run_dir, 'resolved_config.yaml.orig')
if os.path.isfile(backup_path):
    shutil.copy2(backup_path, cfg_path)
else:
    shutil.copy2(cfg_path, backup_path)
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

old_run_dir = cfg['artifact_paths']['run_dir']

def remap(v):
    if not isinstance(v, str):
        return v
    if v.startswith(old_run_dir):
        v = v.replace(old_run_dir, run_dir, 1)
    if path_remap:
        old, new = path_remap.split(':', 1)
        v = v.replace(old, new)
    return v

def walk(obj):
    if isinstance(obj, dict):
        return {k: walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [walk(v) for v in obj]
    return remap(obj)

cfg = walk(cfg)

# Override model path with local value
if model_path:
    cfg.setdefault('policy', {}).setdefault('model', {})['path'] = model_path
    print(f'Override model path: {model_path}', flush=True)

# Override checkpoint path if provided
if checkpoint_path:
    cfg.setdefault('policy', {}).setdefault('model', {}).setdefault('bundle_config', {})['checkpoint_path'] = checkpoint_path
    print(f'Override checkpoint: {checkpoint_path}', flush=True)

with open(cfg_path, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
print(f'Fixed paths: run_dir {old_run_dir} -> {run_dir}', flush=True)
if path_remap:
    print(f'Path remap: {path_remap}', flush=True)
" "$RUN_DIR" "$PATH_REMAP" "$MODEL_PATH" "$CHECKPOINT_PATH"

cd "${ALPAGYM_ROOT:-/data/mnt_m62/10_personal/z59900495/workspace/alpagym}"

export GRPC_ARG_ENABLE_HTTP_PROXY="${GRPC_ARG_ENABLE_HTTP_PROXY:-0}"
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"

export UV_NO_MANAGED_PYTHON="${UV_NO_MANAGED_PYTHON:-1}"
export UV_PYTHON="${UV_PYTHON:-$(command -v python)}"
export UV_SYSTEM_CERTS="${UV_SYSTEM_CERTS:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6}"

export ALPAGYM_COSMOS_ONLY=1

# Build Hydra overrides
OVERRIDES=(
  "execution.resolved_config_path=${RUN_DIR}/resolved_config.yaml"
  "topology=${TOPOLOGY}"
  "cosmos.mode=disaggregated"
)
if [ -n "$EXPERIMENT" ]; then
  OVERRIDES+=("experiment=${EXPERIMENT}")
fi
OVERRIDES+=("policy.model.path=${MODEL_PATH}")
if [ -n "$CHECKPOINT_PATH" ]; then
  OVERRIDES+=("+policy.model.bundle_config.checkpoint_path=${CHECKPOINT_PATH}")
fi

exec uv run --no-sync --all-packages python -m alpagym_host.cli "${OVERRIDES[@]}"
