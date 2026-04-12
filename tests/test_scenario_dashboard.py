"""Tests for multi-scenario dashboard orchestration."""

from __future__ import annotations

import errno
import sqlite3

import pytest

from weatheredge.dashboard import scenario_dashboard as dash


def test_resolve_scenario_name_aliases():
    assert dash.resolve_scenario_name("risky") == "rischioso"
    assert dash.resolve_scenario_name("balanced") == "bilanciato"
    assert dash.resolve_scenario_name("safe") == "safer"
    assert dash.resolve_scenario_name("safer") == "safer"

    with pytest.raises(ValueError):
        dash.resolve_scenario_name("invalid")


def test_collect_status_reads_db_metrics_and_last_log(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")

    paths = dash.scenario_paths("bilanciato")
    paths.root.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(paths.db_file)) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE trades (status TEXT, pnl REAL)")
        cursor.execute("CREATE TABLE forecasts (id INTEGER PRIMARY KEY, forecast_timestamp TEXT)")
        cursor.execute("CREATE TABLE market_prices (id INTEGER PRIMARY KEY, timestamp TEXT)")
        cursor.execute("CREATE TABLE markets (id TEXT PRIMARY KEY)")
        cursor.executemany(
            "INSERT INTO trades(status, pnl) VALUES (?, ?)",
            [
                ("won", 5.0),
                ("won", 2.5),
                ("lost", -1.0),
                ("open", 0.0),
            ],
        )
        cursor.executemany(
            "INSERT INTO forecasts(id, forecast_timestamp) VALUES (?, ?)",
            [
                (1, "2026-04-01 11:58:00"),
                (2, "2026-04-01 12:00:00"),
            ],
        )
        cursor.executemany(
            "INSERT INTO market_prices(id, timestamp) VALUES (?, ?)",
            [
                (1, "2026-04-01 12:01:00"),
                (2, "2026-04-01 12:02:00"),
                (3, "2026-04-01 12:03:00"),
            ],
        )
        cursor.executemany("INSERT INTO markets(id) VALUES (?)", [("m1",), ("m2",)])
        conn.commit()

    paths.log_file.write_text("linea uno\nlinea due\n", encoding="utf-8")
    status = dash.collect_status("bilanciato")

    assert status.running is False
    assert status.stats.trades == 4
    assert status.stats.open_positions == 1
    assert status.stats.wins == 2
    assert status.stats.losses == 1
    assert status.stats.pnl == pytest.approx(6.5)
    assert status.stats.forecasts == 2
    assert status.stats.market_prices == 3
    assert status.stats.markets == 2
    assert status.stats.last_forecast_ts == "2026-04-01 12:00:00"
    assert status.stats.last_market_price_ts == "2026-04-01 12:03:00"
    assert status.stats.forecast_age_minutes is not None
    assert status.stats.price_age_minutes is not None
    assert status.stats.win_rate == pytest.approx(66.6666, rel=1e-3)
    assert status.last_log_line == "linea due"


def test_start_scenario_sets_env_and_writes_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)

    captured: dict[str, object] = {}

    class DummyProcess:
        pid = 43210

        @staticmethod
        def poll():
            return None

    def fake_popen(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return DummyProcess()

    monkeypatch.setattr(dash.subprocess, "Popen", fake_popen)

    ok, message = dash.start_scenario("rischioso")
    paths = dash.scenario_paths("rischioso")
    env = captured["kwargs"]["env"]  # type: ignore[index]

    assert ok is True
    assert "avviato" in message
    assert paths.pid_file.read_text(encoding="utf-8").strip() == "43210"
    assert env["WEATHEREDGE_DB_PATH"] == str(paths.db_file)
    assert env["WEATHEREDGE_LOG_FILE"] == str(paths.log_file)
    assert env["KELLY_FRACTION"] == dash.SCENARIO_ENV_OVERRIDES["rischioso"]["KELLY_FRACTION"]


def test_stop_scenario_removes_stale_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")

    paths = dash.scenario_paths("safer")
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.pid_file.write_text("99999", encoding="utf-8")

    monkeypatch.setattr(dash, "_is_pid_running", lambda _pid: False)

    ok, message = dash.stop_scenario("safer")
    assert ok is False
    assert "stale" in message
    assert not paths.pid_file.exists()


def test_is_pid_running_treats_eperm_as_alive(monkeypatch):
    def fake_kill(_pid, _sig):
        raise OSError(errno.EPERM, "operation not permitted")

    monkeypatch.setattr(dash.os, "kill", fake_kill)
    assert dash._is_pid_running(12345) is True


def test_run_start_or_stop_starts_collector_after_successful_start(monkeypatch):
    monkeypatch.setattr(dash, "_targets", lambda _target: ["rischioso"])
    monkeypatch.setattr(dash, "start_scenario", lambda _name: (True, "started"))
    called = {"collector": False}

    def fake_ensure_collector():
        called["collector"] = True
        return True, "collector started"

    monkeypatch.setattr(dash, "_ensure_snapshot_collector_running", fake_ensure_collector)

    rc = dash._run_start_or_stop("start", "all")
    assert rc == 0
    assert called["collector"] is True
