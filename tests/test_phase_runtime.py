from __future__ import annotations

from unittest.mock import MagicMock

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


def test_phase_matrix_alias_resolution():
    from weatheredge.dashboard.phase_matrix import resolve_profile_name

    assert resolve_profile_name("phase1") == "phase_1"
    assert resolve_profile_name("phase2") == "phase_2"
    assert resolve_profile_name("phase_2") == "phase_2"
