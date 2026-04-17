from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from unittest.mock import AsyncMock, patch

import weatheredge.config as cfg


def test_activate_phase_no_persist_does_not_write(tmp_path, monkeypatch):
    from weatheredge import phase_config

    phase_file = tmp_path / "current_phase.json"
    monkeypatch.setattr(phase_config, "PHASE_FILE", phase_file)

    original_modules = dict(cfg.MODULES)
    try:
        modules = phase_config.activate_phase("phase_2", persist=False)
        assert modules["burst_mode"] is True
        assert modules["market_drift_tracking"] is True
        assert not phase_file.exists()
    finally:
        cfg.MODULES.clear()
        cfg.MODULES.update(original_modules)


def test_runtime_phase_override_uses_env_without_persist(monkeypatch):
    from weatheredge.main import _apply_runtime_phase_override

    mock_activate = MagicMock()
    monkeypatch.setenv("WEATHEREDGE_PHASE", "phase_1")
    monkeypatch.setattr("weatheredge.phase_config.activate_phase", mock_activate)

    result = _apply_runtime_phase_override()

    assert result == "phase_1"
    mock_activate.assert_called_once_with("phase_1", persist=False)


def test_runtime_order_mode_override_market(monkeypatch):
    from weatheredge.main import _apply_runtime_order_mode_override

    original_modules = dict(cfg.MODULES)
    try:
        cfg.MODULES["limit_orders"] = True
        monkeypatch.setenv("WEATHEREDGE_ORDER_MODE", "market")
        result = _apply_runtime_order_mode_override()
        assert result == "market"
        assert cfg.MODULES["limit_orders"] is False
    finally:
        cfg.MODULES.clear()
        cfg.MODULES.update(original_modules)


def test_phase2_effective_thresholds_burst_on():
    from weatheredge.main import AppState, _phase2_effective_thresholds
    from weatheredge.config import BURST_EDGE_DISCOUNT

    original_modules = dict(cfg.MODULES)
    try:
        cfg.MODULES["burst_mode"] = True
        state = AppState()
        state._burst_mode_until = datetime.now(timezone.utc) + timedelta(minutes=10)

        conservative, aggressive, qualitative = _phase2_effective_thresholds(state)

        assert abs(conservative - max(0.0, state._base_conservative_min_edge - BURST_EDGE_DISCOUNT)) < 1e-9
        assert abs(aggressive - max(0.0, state._base_aggressive_min_edge - BURST_EDGE_DISCOUNT)) < 1e-9
        assert abs(qualitative - max(0.0, 0.05 - (BURST_EDGE_DISCOUNT * 0.5))) < 1e-9
    finally:
        cfg.MODULES.clear()
        cfg.MODULES.update(original_modules)


def test_phase2_effective_thresholds_burst_off():
    from weatheredge.main import AppState, _phase2_effective_thresholds

    original_modules = dict(cfg.MODULES)
    try:
        cfg.MODULES["burst_mode"] = True
        state = AppState()
        state._burst_mode_until = datetime.now(timezone.utc) - timedelta(minutes=1)

        conservative, aggressive, qualitative = _phase2_effective_thresholds(state)

        assert abs(conservative - state._base_conservative_min_edge) < 1e-9
        assert abs(aggressive - state._base_aggressive_min_edge) < 1e-9
        assert abs(qualitative - 0.05) < 1e-9
    finally:
        cfg.MODULES.clear()
        cfg.MODULES.update(original_modules)


def test_check_model_runs_arms_burst_window_when_enabled():
    from weatheredge import main as main_module

    original_modules = dict(cfg.MODULES)
    try:
        cfg.MODULES["burst_mode"] = True
        state = main_module.init_app()
        state.model_run_detector = MagicMock()

        with patch.object(
            main_module,
            "_check_model_runs_async",
            new=AsyncMock(return_value=["event"]),
        ), patch.object(main_module, "run_decision_loop") as mock_loop:
            main_module.check_model_runs()

        assert state._burst_mode_until is not None
        mock_loop.assert_called_once()
    finally:
        cfg.MODULES.clear()
        cfg.MODULES.update(original_modules)


def test_startup_bootstrap_runs_immediate_steps_in_order():
    from weatheredge import main as main_module

    calls: list[str] = []

    with (
        patch.object(main_module, "fetch_market_data", side_effect=lambda: calls.append("market")),
        patch.object(main_module, "fetch_forecast_data", side_effect=lambda: calls.append("forecast")),
        patch.object(main_module, "run_decision_loop", side_effect=lambda: calls.append("decision")),
    ):
        main_module._startup_bootstrap_cycle(run_decision_loop_once=True)

    assert calls == ["market", "forecast", "decision"]


def test_startup_bootstrap_continues_when_one_step_fails():
    from weatheredge import main as main_module

    calls: list[str] = []

    def _fail_market() -> None:
        calls.append("market")
        raise RuntimeError("boom")

    with (
        patch.object(main_module, "fetch_market_data", side_effect=_fail_market),
        patch.object(main_module, "fetch_forecast_data", side_effect=lambda: calls.append("forecast")),
        patch.object(main_module, "run_decision_loop", side_effect=lambda: calls.append("decision")),
    ):
        main_module._startup_bootstrap_cycle(run_decision_loop_once=True)

    assert calls == ["market", "forecast", "decision"]
