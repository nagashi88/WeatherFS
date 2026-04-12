#!/bin/bash
# WeatherEdge — Double-click to launch control panel
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p logs

echo ""
echo "  🌤  WeatherEdge Control Panel"
echo "  ─────────────────────────────"
echo "  Opening in browser..."
echo "  Close this window to stop."
echo ""

streamlit run weatheredge/dashboard/control_panel.py \
    --server.headless true \
    --server.port 8501 \
    --browser.gatherUsageStats false \
    2>&1
