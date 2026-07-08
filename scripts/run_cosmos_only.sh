#!/usr/bin/env bash
# Disaggregated mode: Cosmos-only (run on the A100 / training machine).
#
# Usage:
#   ./scripts/run_cosmos_only.sh <copied_run_dir> [--model-path <path>]
#
# Prerequisites:
#   1. SSH tunnel to the Wizard machine must already be established:
#      ssh -N -L <port>:localhost:<port> <user>@<wizard_host>
#      (port is printed by run_wizard_only.sh)
#   2. The run directory from the Wizard machine must be copied here:
#      scp -r <user>@<wizard_host>:<run_dir> <local_path>
#   3. The model checkpoint must be accessible locally (override with --model-path
#      or MODEL_PATH env var if the path differs from the Wizard machine).
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <copied_run_dir> [--model-path <path>]" >&2
  exit 1
fi

RUN_DIR="$(cd "$1" && pwd)"
shift

MODEL_PATH_OVERRIDE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model-path) MODEL_PATH_OVERRIDE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -n "${MODEL_PATH:-}" ] && [ -z "$MODEL_PATH_OVERRIDE" ]; then
  MODEL_PATH_OVERRIDE="$MODEL_PATH"
fi

# Fix absolute paths in resolved_config.yaml so they point to the local copy.
python3 -c "
import yaml, sys, os
run_dir = sys.argv[1]
model_override = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
cfg_path = os.path.join(run_dir, 'resolved_config.yaml')
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
old_run_dir = cfg['artifact_paths']['run_dir']
# Remap every artifact_paths field from old run_dir prefix to new.
for k, v in cfg['artifact_paths'].items():
    if isinstance(v, str) and v.startswith(old_run_dir):
        cfg['artifact_paths'][k] = v.replace(old_run_dir, run_dir, 1)
if model_override:
    cfg['policy']['model']['path'] = model_override
with open(cfg_path, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
print(f'Fixed paths in {cfg_path}: run_dir {old_run_dir} -> {run_dir}')
" "$RUN_DIR" "$MODEL_PATH_OVERRIDE"

cd "${ALPAGYM_ROOT:-$HOME/alpagym}"

export GRPC_ARG_ENABLE_HTTP_PROXY="${GRPC_ARG_ENABLE_HTTP_PROXY:-0}"
export grpc_proxy="" http_proxy="" https_proxy=""
export HTTP_PROXY="" HTTPS_PROXY=""
export no_proxy="localhost,127.0.0.1,0.0.0.0"
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"

export UV_NO_MANAGED_PYTHON="${UV_NO_MANAGED_PYTHON:-1}"
export UV_PYTHON="${UV_PYTHON:-$(command -v python)}"
export UV_SYSTEM_CERTS="${UV_SYSTEM_CERTS:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export ALPAGYM_COSMOS_ONLY=1

exec uv run --no-sync --all-packages python -m alpagym_host.cli \
  "execution.resolved_config_path=${RUN_DIR}/resolved_config.yaml"
