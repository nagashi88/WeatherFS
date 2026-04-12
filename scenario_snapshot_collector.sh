#!/bin/bash
# Periodic snapshot collector for multi-scenario WeatherEdge runs.

set -euo pipefail

cd "$(dirname "$0")"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

python -m weatheredge.dashboard.scenario_snapshot_collector "$@"

