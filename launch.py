#!/usr/bin/env python3
"""WeatherEdge Launcher — PRD v3.1.

Reads the active phase from current_phase.json, applies module configuration,
and starts the trading bot. Optionally launches the dashboard.

Usage:
    python launch.py                      # Start with current phase (default: phase_0)
    python launch.py --phase phase_1      # Switch to phase_1 and start
    python launch.py --dashboard          # Also launch Streamlit dashboard
    python launch.py --gate               # Check if current phase gate is passed
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("launch")

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="WeatherEdge Launcher")
    parser.add_argument("--phase", type=str, default=None,
                        help="Phase to activate (phase_0..phase_4)")
    parser.add_argument("--dashboard", action="store_true",
                        help="Also launch the Streamlit control panel")
    parser.add_argument("--gate", action="store_true",
                        help="Check gate status for current phase and exit")
    parser.add_argument("--status", action="store_true",
                        help="Show current phase and module status")
    args = parser.parse_args()

    from weatheredge.phase_config import (
        activate_phase, get_current_phase, get_gate_status,
        PHASE_CONFIGS, get_phase_modules,
    )

    # Determine phase
    if args.phase:
        phase = args.phase
        Path(PROJECT_ROOT / "current_phase.json").write_text(
            json.dumps({"phase": phase})
        )
    else:
        phase = get_current_phase()

    # Status mode
    if args.status:
        modules = get_phase_modules(phase)
        config = PHASE_CONFIGS[phase]
        print(f"\n  WeatherEdge — {config['name']}")
        print(f"  {config['description']}\n")
        on = [k for k, v in modules.items() if v]
        off = [k for k, v in modules.items() if not v]
        print(f"  ON  ({len(on)}): {', '.join(on)}")
        print(f"  OFF ({len(off)}): {', '.join(off)}")
        print(f"\n  Gate: {config.get('gate', 'N/A')}\n")
        return

    # Gate check mode
    if args.gate:
        result = get_gate_status(phase)
        print(f"\n  Phase: {phase}")
        print(f"  Gate: {result.get('gate', 'N/A')}")
        print(f"  Passed: {result.get('passed', 'unknown')}")
        if result.get("reason"):
            print(f"  Reason: {result['reason']}")
        print()
        sys.exit(0 if result.get("passed") else 1)

    # Activate phase
    log.info("WeatherEdge starting in: %s", phase)
    modules = activate_phase(phase)

    on_count = sum(1 for v in modules.values() if v)
    log.info("Modules: %d ON, %d OFF", on_count, len(modules) - on_count)

    # Launch dashboard in background if requested
    dashboard_proc = None
    if args.dashboard:
        log.info("Launching dashboard on http://localhost:8502")
        dashboard_proc = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run",
                str(PROJECT_ROOT / "weatheredge" / "dashboard" / "control_panel.py"),
                "--server.headless", "true",
                "--server.port", "8502",
                "--browser.gatherUsageStats", "false",
            ],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Start the trading bot
    log.info("Dashboard: http://localhost:8502")
    try:
        from weatheredge.main import main as bot_main
        bot_main()
    except KeyboardInterrupt:
        log.info("Shutdown requested")
    finally:
        if dashboard_proc:
            dashboard_proc.terminate()
            log.info("Dashboard stopped")


if __name__ == "__main__":
    main()
