#!/usr/bin/env bash
# Disaggregated mode: Wizard-only for Alpamayo 1.5 (run on the 5090 / rendering machine).
# Starts AlPaSim Wizard, publishes runtime endpoints, then blocks until Ctrl+C.
set -euo pipefail
export ALPAGYM_WIZARD_ONLY=1
exec "$(dirname "$0")/run_alpamayo_smoke.sh"
