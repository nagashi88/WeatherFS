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


def test_runtime_phase_falls_back_to_current_phase_json_when_env_unset(monkeypatch):
    """STEP 4: without WEATHEREDGE_PHASE, the bot must load the phase
    persisted in current_phase.json instead of silently running Phase 0."""
    from weatheredge.main import _apply_runtime_phase_override

    mock_activate = MagicMock()
    monkeypatch.delenv("WEATHEREDGE_PHASE", raising=False)
    monkeypatch.setattr("weatheredge.phase_config.activate_phase", mock_activate)
    monkeypatch.setattr("weatheredge.phase_config.get_current_phase", lambda: "phase_1")

    result = _apply_runtime_phase_override()

    assert result == "phase_1"
    mock_activate.assert_called_once_with("phase_1", persist=False)


def test_runtime_phase_phase_0_no_env_does_not_call_activate(monkeypatch):
    """STEP 4: when neither env var nor json declare a post-Phase-0 state,
    the function should be a no-op (no activate_phase call, no noisy logs)."""
    from weatheredge.main import _apply_runtime_phase_override

    mock_activate = MagicMock()
    monkeypatch.delenv("WEATHEREDGE_PHASE", raising=False)
    monkeypatch.setattr("weatheredge.phase_config.activate_phase", mock_activate)
    monkeypatch.setattr("weatheredge.phase_config.get_current_phase", lambda: "phase_0")

    result = _apply_runtime_phase_override()

    assert result == "phase_0"
    mock_activate.assert_not_called()


def test_runtime_phase_boot_assertion_logs_on_mismatch(monkeypatch, caplog):
    """STEP 4: boot_phase_check must log ERROR when MODULES diverge from
    the declared phase's expected modules. This is the safety net that
    would have surfaced the April-2026 bug where current_phase.json said
    phase_1 but MODULES silently stayed at Phase-0 defaults."""
    import logging
    from weatheredge.main import _apply_runtime_phase_override

    monkeypatch.delenv("WEATHEREDGE_PHASE", raising=False)
    monkeypatch.setattr("weatheredge.phase_config.get_current_phase", lambda: "phase_1")
    # activate_phase is mocked out → MODULES never actually get updated, so
    # the boot check must detect the divergence (exit_ev_based expected True
    # in phase_1 but config.py default is False).
    monkeypatch.setattr("weatheredge.phase_config.activate_phase", MagicMock())

    original_modules = dict(cfg.MODULES)
    cfg.MODULES["exit_ev_based"] = False
    cfg.MODULES["bias_correction"] = False
    try:
        with caplog.at_level(logging.ERROR, logger="weatheredge.main"):
            _apply_runtime_phase_override()
    finally:
        cfg.MODULES.clear()
        cfg.MODULES.update(original_modules)

    assert any(
        "boot_phase_check" in rec.message and "diverge" in rec.message
        for rec in caplog.records
    ), f"Expected boot_phase_check divergence ERROR, got: {[r.message for r in caplog.records]}"


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


# ---------------------------------------------------------------------------
# Finding 5: atomic phase-file write + malformed-file warning
# ---------------------------------------------------------------------------

def test_activate_phase_persists_full_payload_and_no_tmp_leak(tmp_path, monkeypatch):
    """activate_phase must persist phase + activated_at + modules atomically.

    A direct `write_text` is not atomic: a crash mid-write would leave the
    file empty and silently downgrade the bot to phase_0 on next start.
    The tmp-file + rename path must write the full payload and leave no
    `.tmp` sibling behind on success.
    """
    import json as _json
    from weatheredge import phase_config

    phase_file = tmp_path / "current_phase.json"
    monkeypatch.setattr(phase_config, "PHASE_FILE", phase_file)

    original_modules = dict(cfg.MODULES)
    try:
        phase_config.activate_phase("phase_1", persist=True)
        assert phase_file.exists()
        data = _json.loads(phase_file.read_text())
        assert data["phase"] == "phase_1"
        assert "activated_at" in data and data["activated_at"]
        assert isinstance(data.get("modules"), dict) and data["modules"]
        # No tmp leak after successful rename.
        assert not (tmp_path / "current_phase.json.tmp").exists()
    finally:
        cfg.MODULES.clear()
        cfg.MODULES.update(original_modules)


def test_malformed_phase_file_logs_warning_and_falls_back_to_phase_0(
    tmp_path, monkeypatch, caplog
):
    """A corrupted current_phase.json must log a WARNING, not silently
    downgrade to phase_0. Silent downgrade was the Phase-1-never-activated
    failure mode — future parse failures must be loud."""
    import logging
    from weatheredge import phase_config

    phase_file = tmp_path / "current_phase.json"
    phase_file.write_text("{not valid json")
    monkeypatch.setattr(phase_config, "PHASE_FILE", phase_file)

    caplog.set_level(logging.WARNING, logger="weatheredge.phase_config")
    current = phase_config.get_current_phase()
    assert current == "phase_0"
    assert any(
        "current_phase.json is malformed" in rec.message for rec in caplog.records
    ), f"Expected malformed-file warning, got: {[r.message for r in caplog.records]}"


def test_get_phase_activated_at_returns_none_on_malformed_file(
    tmp_path, monkeypatch
):
    """Finding-3's phase-scoped dashboard filter depends on activated_at;
    a corrupted file must return None (skip filter) rather than raise."""
    from weatheredge import phase_config

    phase_file = tmp_path / "current_phase.json"
    phase_file.write_text("{broken")
    monkeypatch.setattr(phase_config, "PHASE_FILE", phase_file)

    assert phase_config.get_phase_activated_at() is None


def test_atomic_write_uses_rename_not_truncate(tmp_path, monkeypatch):
    """Directly verify the tmp + rename path: once _atomic_write_json
    returns, the tmp sibling is gone and the target has the new payload.

    Regression guard: if someone reintroduces `write_text(...)` directly,
    this test passes, but the test_activate_phase_persists_full_payload
    one still exercises the public API. This one documents the mechanism.
    """
    import json as _json
    from weatheredge import phase_config

    target = tmp_path / "nested" / "phase.json"
    target.parent.mkdir(parents=True)
    payload = {"phase": "phase_2", "activated_at": "2026-04-22T10:00:00+00:00"}
    phase_config._atomic_write_json(target, payload)

    assert target.exists()
    # No stray `.tmp` sibling under the target directory.
    leftovers = [p.name for p in target.parent.iterdir() if p.name.startswith(target.name)
                 and p.suffix == ".tmp"]
    assert leftovers == [], f"Unexpected tmp leak: {leftovers}"
    assert _json.loads(target.read_text()) == payload


def test_atomic_write_is_safe_under_concurrent_writers(tmp_path):
    """Two threads calling `_atomic_write_json` on the same target must
    both succeed. The previous implementation shared a single
    `target.with_suffix('.tmp')` path, so the second writer raced into
    the same filename and one `os.replace` would race with the other's
    `open-truncate`, yielding FileNotFoundError. Using a unique tmp per
    writer (tempfile.mkstemp) makes the write clean last-writer-wins.
    """
    import json as _json
    import threading
    from weatheredge import phase_config

    target = tmp_path / "phase.json"
    errors: list[Exception] = []
    barrier = threading.Barrier(8)

    def _worker(i: int) -> None:
        try:
            barrier.wait(timeout=5)
            for k in range(10):
                phase_config._atomic_write_json(
                    target, {"phase": f"phase_{i}", "n": k}
                )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert errors == [], f"Concurrent writers raised: {errors}"
    # Final file must be valid JSON with the expected shape.
    data = _json.loads(target.read_text())
    assert "phase" in data and data["phase"].startswith("phase_")
    # No leftover .tmp siblings.
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == [], f"Unexpected tmp leak: {leftovers}"


def test_atomic_write_cleans_up_tmp_on_rename_failure(tmp_path, monkeypatch):
    """If os.replace fails (permissions, EXDEV), we must not leak a tmp
    file into the target directory, and the exception propagates."""
    from weatheredge import phase_config

    target = tmp_path / "phase.json"

    def _boom(*_args, **_kwargs):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(phase_config.os, "replace", _boom)

    import pytest as _pytest
    with _pytest.raises(OSError, match="simulated rename failure"):
        phase_config._atomic_write_json(target, {"phase": "phase_1"})

    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == [], f"Tmp leaked after rename failure: {leftovers}"


def test_launch_status_mode_does_not_touch_phase_file(tmp_path, monkeypatch, capsys):
    """Regression for Finding 5 Bug B: `launch.py --phase X --status` must
    NOT persist anything to current_phase.json — it is an inspection mode.
    Previously the truncated write at the top of main() clobbered the
    file before the status check returned."""
    import runpy
    import sys
    from weatheredge import phase_config

    phase_file = tmp_path / "current_phase.json"
    # Seed with a known-good payload; status mode must leave it untouched.
    import json as _json
    seed = {
        "phase": "phase_1",
        "activated_at": "2026-04-21T23:13:56+00:00",
        "modules": {"bias_correction": True},
    }
    phase_file.write_text(_json.dumps(seed, indent=2))
    monkeypatch.setattr(phase_config, "PHASE_FILE", phase_file)

    argv_backup = list(sys.argv)
    sys.argv = ["launch.py", "--phase", "phase_2", "--status"]
    try:
        runpy.run_path(
            str(phase_config.PROJECT_ROOT / "launch.py"),
            run_name="__main__",
        )
    except SystemExit:
        pass
    finally:
        sys.argv = argv_backup

    # File must still contain the seed payload unchanged.
    after = _json.loads(phase_file.read_text())
    assert after == seed, (
        f"--status clobbered current_phase.json. Expected {seed!r}, got {after!r}"
    )
