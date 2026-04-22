"""WeatherEdge Phase Configuration — PRD v3.1.

Defines module configurations for each phase and provides activate_phase()
to switch between phases at runtime. Phase state persists in current_phase.json.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import weatheredge.config as cfg

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE_FILE = PROJECT_ROOT / "current_phase.json"

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

PHASE_CONFIGS: dict[str, dict] = {
    "phase_0": {
        "name": "Phase 0 — Baseline B",
        "description": (
            "Sistema semplificato. Solo blended forecast + fixed thresholds + risk caps. "
            "Nessun agent voting, nessun TimesFM, nessun learning dinamico."
        ),
        "modules": {
            "agent_mean_reversion": False,
            "agent_voting": False,
            "agent_qualitative": False,
            "timesfm": False,
            "student_t_fallback": True,
            "vpin": False,
            "exit_pnl_stoploss": False,
            "exit_edge_flip": True,
            "exit_time_decay": True,
            "exit_ev_based": False,
            "dynamic_model_weights": False,
            "dynamic_agent_weights": False,
            "weekly_error_analysis": True,
            "cross_platform_arb": False,
            "limit_orders": True,
            "burst_mode": False,
            "bias_correction": False,
            "rmse_floor_dynamic": False,
            "reliability_tracking": True,   # Enabled Phase 0: observational only, no decision impact
            "market_drift_tracking": False,
            "shrinkage_rmse": False,
            "seasonal_agent_weights": False,
            "meta_calibrator": False,
            "orderbook_logging": True,
            "temporal_guard": True,
            "volume_filter": True,
        },
        "details": [
            "HRRR-weighted blending attivo (pesi fissi per lead time)",
            "Student-T fallback per code grasse",
            "Kelly fraction: 5% warm-up / 8% normale",
            "Filtro volume: skip mercati < $2K o spread > 8%",
            "Temporal guard anti-leakage attivo",
            "Orderbook depth logging attivo (per Phase 4)",
            "Exit: solo edge-flip + time-decay",
            "STRUCTURAL KILL: MeanReversion, VPIN, P&L stop-loss",
        ],
        "gate": "20+ trade paper, Brier < 0.25, zero leakage violations",
        "gate_function": "check_phase0_gate",
    },
    "phase_1": {
        "name": "Phase 1 — Calibrazione modello",
        "description": (
            "Centra il bersaglio con bias correction e RMSE dinamico. "
            "Exit basate su Expected Value."
        ),
        "modules": {
            "bias_correction": True,
            "rmse_floor_dynamic": True,
            "exit_ev_based": True,
            "exit_edge_flip": True,         # KEEP as-is in Phase 1 (EV-based replaces only P&L stop-loss)
            "reliability_tracking": True,
        },
        "details": [
            "Tutto Phase 0 +",
            "Bias correction per stazione/modello/mese",
            "RMSE floor HRRR -> 1.5F sotto 12h",
            "Exit EV-based (sostituisce P&L stop-loss)",
            "Reliability diagram per decile di probabilita",
            "PnL separato per lead time (<12h / 12-24h / >24h)",
        ],
        "gate": "40+ trade, calibration gap < 15%, Baseline B profittevole after costs",
        "gate_function": "check_phase1_gate",
    },
    "phase_2": {
        "name": "Phase 2 — Burst mode & timing",
        "description": (
            "Sfrutta la finestra informativa post-model-run (00Z/06Z/12Z/18Z)."
        ),
        "modules": {
            "burst_mode": True,
            "market_drift_tracking": True,
        },
        "details": [
            "Tutto Phase 1 +",
            "Trigger burst su nuovo model run GFS/HRRR/ECMWF",
            "Scan aggressivo nei 30 min post-run",
            "Market drift tracking (il mercato converge verso di te?)",
            "Kelly 8% (non piu warm-up se gate Phase 1 superato)",
        ],
        "gate": "50+ trade, Brier < market-implied, drift rate > 50%",
        "gate_function": "check_phase2_gate",
    },
    "phase_3": {
        "name": "Phase 3 — Complessita giustificata",
        "description": (
            "Riattiva moduli SOLO se l'ablation test li promuove. "
            "Ogni pezzo deve dimostrare miglioramento."
        ),
        "modules": {
            "agent_voting": True,
            "timesfm": True,
            "shrinkage_rmse": True,
            "seasonal_agent_weights": True,
            "dynamic_model_weights": True,
            "meta_calibrator": True,
        },
        "details": [
            "Tutto Phase 2 +",
            "Agent voting CON convergence gate (ratio 3:1)",
            "TimesFM a weight 0.0 -- il learning loop decide",
            "Shrinkage gerarchico su RMSE (station -> region -> global)",
            "Pesi agenti stagionali (estate/inverno/spalla)",
            "Dynamic model weights (ogni 50 trade)",
            "Meta-calibrator logistic (solo se 300+ trade)",
            "REGOLA: ogni modulo deve passare ablation test matched",
        ],
        "gate": "Ogni modulo migliora Brier out-of-sample? Se no -> resta OFF",
        "gate_function": None,
    },
    "phase_4": {
        "name": "Phase 4 — Scale",
        "description": (
            "Solo dopo 200+ trade live profittevoli e Sharpe > 1.0."
        ),
        "modules": {
            "cross_platform_arb": True,
        },
        "details": [
            "Tutto Phase 3 +",
            "Cross-platform arbitrage automatico",
            "Execution simulator con dati orderbook storici",
            "Correlation-aware Kelly (richiede 500+ trade)",
            "Scala capitale",
        ],
        "gate": "200+ trade live, PnL positivo, Sharpe > 1.0 su 3+ mesi",
        "gate_function": None,
    },
}

PHASE_ORDER = ["phase_0", "phase_1", "phase_2", "phase_3", "phase_4"]


# ---------------------------------------------------------------------------
# Phase activation
# ---------------------------------------------------------------------------

def get_current_phase() -> str:
    """Read current phase from persistent file. Defaults to phase_0."""
    if PHASE_FILE.exists():
        try:
            data = json.loads(PHASE_FILE.read_text())
            return data.get("phase", "phase_0")
        except (json.JSONDecodeError, KeyError):
            pass
    return "phase_0"


def get_phase_activated_at() -> str | None:
    """Read ISO 8601 timestamp of the most recent phase activation, or None."""
    if PHASE_FILE.exists():
        try:
            data = json.loads(PHASE_FILE.read_text())
            ts = data.get("activated_at")
            return str(ts) if ts else None
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _merged_modules_for_phase(phase_name: str) -> dict[str, bool]:
    """Return cumulative module config for a phase without mutating globals."""
    if phase_name not in PHASE_CONFIGS:
        raise ValueError(f"Unknown phase: {phase_name}. Choose from: {list(PHASE_CONFIGS.keys())}")

    target_idx = PHASE_ORDER.index(phase_name)
    merged: dict[str, bool] = {}

    for i in range(target_idx + 1):
        p = PHASE_ORDER[i]
        phase_modules = PHASE_CONFIGS[p]["modules"]
        if i == 0:
            # Phase 0 is the full base — set ALL keys
            merged = dict(phase_modules)
        else:
            # Subsequent phases only override specific keys
            merged.update(phase_modules)
    return merged


def activate_phase(phase_name: str, persist: bool = True) -> dict[str, bool]:
    """Activate a phase by applying cumulative modules to cfg.MODULES.

    Phases are cumulative: phase_2 includes all of phase_0 + phase_1 + phase_2.
    Returns the final MODULES dict.
    """
    merged = _merged_modules_for_phase(phase_name)

    # Apply to global MODULES
    changes: list[str] = []
    for key, value in merged.items():
        old = cfg.MODULES.get(key)
        if old != value:
            changes.append(f"  {key}: {old} -> {value}")
        cfg.MODULES[key] = value

    if persist:
        # Persist
        PHASE_FILE.write_text(json.dumps({
            "phase": phase_name,
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "modules": dict(cfg.MODULES),
        }, indent=2))

    log.info(
        "Phase activated: %s (%s) [persist=%s]",
        phase_name,
        PHASE_CONFIGS[phase_name]["name"],
        persist,
    )
    if changes:
        log.info("Module changes:\n%s", "\n".join(changes))
    else:
        log.info("No module changes (already at this phase)")

    return dict(cfg.MODULES)


def get_phase_modules(phase_name: str) -> dict[str, bool]:
    """Get the merged module config for a phase without activating it."""
    try:
        return _merged_modules_for_phase(phase_name)
    except ValueError:
        return {}


def get_gate_status(phase_name: str) -> dict:
    """Check if a phase's gate conditions are met."""
    config = PHASE_CONFIGS.get(phase_name, {})
    gate_fn_name = config.get("gate_function")
    if not gate_fn_name:
        return {"phase": phase_name, "gate": config.get("gate", ""), "passed": None, "reason": "No gate function"}

    try:
        from weatheredge import main as _main
        gate_fn = getattr(_main, gate_fn_name, None)
        if gate_fn is None:
            return {"phase": phase_name, "gate": config.get("gate", ""), "passed": None, "reason": f"Function {gate_fn_name} not found"}
        passed = gate_fn()
        return {"phase": phase_name, "gate": config.get("gate", ""), "passed": passed}
    except Exception as exc:
        return {"phase": phase_name, "gate": config.get("gate", ""), "passed": False, "reason": str(exc)}
