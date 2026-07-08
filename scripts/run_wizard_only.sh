#!/usr/bin/env bash
# Disaggregated mode: Wizard-only (run on the 5090 / rendering machine).
# Starts AlPaSim Wizard, publishes runtime endpoints, then blocks until Ctrl+C.
# The printed run directory must be copied to the Cosmos machine.
set -euo pipefail
export ALPAGYM_WIZARD_ONLY=1
exec "$(dirname "$0")/run_autovla_smoke.sh"
