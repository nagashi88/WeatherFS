#!/bin/bash
# Multi-scenario dashboard launcher (rischioso/bilanciato/safer)

set -euo pipefail

cd "$(dirname "$0")"

if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

python -m weatheredge.dashboard.scenario_dashboard "$@"

