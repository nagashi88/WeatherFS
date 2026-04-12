"""Additional coverage tests for scenario_dashboard and scenario_snapshot_collector.

Targets uncovered paths identified in coverage report:
- scenario_dashboard.py: lines 172-175, 181, 192, 195-196, 202-203, 225, 233, 241,
  250-251, 256, 269-305, 340-341, 366-376, 395, 399-419, 423, 427-433, 437-440,
  444-477, 481-515, 525-526, 552, 571-576, 584, 590-609, 613, 617-631, 635-835, 839
- scenario_snapshot_collector.py: lines 55, 61, 119-121, 144-230, 245-282, 286-348, 352
"""

from __future__ import annotations

import csv
import errno
import io
import signal
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from weatheredge.dashboard import scenario_dashboard as dash
from weatheredge.dashboard import scenario_snapshot_collector as collector


# ---------------------------------------------------------------------------
# scenario_dashboard – helper functions
# ---------------------------------------------------------------------------


class TestTargets:
    def test_all_returns_all_scenarios(self):
        assert dash._targets("all") == list(dash.SCENARIO_ORDER)

    def test_single_alias_resolves(self):
        assert dash._targets("risky") == ["rischioso"]

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            dash._targets("nonexistent")


class TestReadPid:
    def test_returns_none_when_file_missing(self, tmp_path):
        assert dash._read_pid(tmp_path / "nope.pid") is None

    def test_returns_none_when_file_empty(self, tmp_path):
        p = tmp_path / "empty.pid"
        p.write_text("", encoding="utf-8")
        assert dash._read_pid(p) is None

    def test_returns_none_when_file_not_integer(self, tmp_path):
        p = tmp_path / "bad.pid"
        p.write_text("abc", encoding="utf-8")
        assert dash._read_pid(p) is None

    def test_returns_pid_integer(self, tmp_path):
        p = tmp_path / "ok.pid"
        p.write_text("12345\n", encoding="utf-8")
        assert dash._read_pid(p) == 12345


class TestSafeUnlink:
    def test_removes_existing_file(self, tmp_path):
        p = tmp_path / "file.txt"
        p.touch()
        dash._safe_unlink(p)
        assert not p.exists()

    def test_does_not_raise_on_missing_file(self, tmp_path):
        dash._safe_unlink(tmp_path / "nonexistent.txt")  # should not raise


class TestTailLines:
    def test_returns_empty_list_when_file_missing(self, tmp_path):
        assert dash._tail_lines(tmp_path / "nope.log") == []

    def test_returns_last_n_lines(self, tmp_path):
        p = tmp_path / "file.log"
        p.write_text("\n".join(str(i) for i in range(100)), encoding="utf-8")
        lines = dash._tail_lines(p, lines=5)
        assert len(lines) == 5
        assert lines[-1] == "99"

    def test_clamps_to_at_least_one_line(self, tmp_path):
        p = tmp_path / "file.log"
        p.write_text("only\n", encoding="utf-8")
        lines = dash._tail_lines(p, lines=0)
        assert lines == ["only"]


class TestLastLogLine:
    def test_returns_dash_when_file_missing(self, tmp_path):
        assert dash._last_log_line(tmp_path / "nope.log") == "-"

    def test_returns_dash_when_file_all_blank(self, tmp_path):
        p = tmp_path / "blank.log"
        p.write_text("   \n\n  \n", encoding="utf-8")
        assert dash._last_log_line(p) == "-"

    def test_returns_last_non_blank_line(self, tmp_path):
        p = tmp_path / "log.log"
        p.write_text("line one\nline two\n\n", encoding="utf-8")
        assert dash._last_log_line(p) == "line two"


class TestSafeScalar:
    def test_returns_zero_when_no_row(self):
        with sqlite3.connect(":memory:") as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE t (v REAL)")
            assert dash._safe_scalar(cursor, "SELECT SUM(v) FROM t") == 0.0

    def test_returns_zero_when_null(self):
        with sqlite3.connect(":memory:") as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE t (v REAL)")
            cursor.execute("INSERT INTO t VALUES (NULL)")
            assert dash._safe_scalar(cursor, "SELECT v FROM t") == 0.0

    def test_returns_float_value(self):
        with sqlite3.connect(":memory:") as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE t (v REAL)")
            cursor.execute("INSERT INTO t VALUES (3.5)")
            assert dash._safe_scalar(cursor, "SELECT v FROM t") == pytest.approx(3.5)


class TestSafeOptionalScalar:
    def test_returns_none_when_no_row(self):
        with sqlite3.connect(":memory:") as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE t (v REAL)")
            assert dash._safe_optional_scalar(cursor, "SELECT MAX(v) FROM t") is None

    def test_returns_float_when_present(self):
        with sqlite3.connect(":memory:") as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE t (v REAL)")
            cursor.execute("INSERT INTO t VALUES (7.5)")
            assert dash._safe_optional_scalar(cursor, "SELECT v FROM t") == pytest.approx(7.5)


class TestSafeText:
    def test_returns_none_when_no_row(self):
        with sqlite3.connect(":memory:") as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE t (v TEXT)")
            assert dash._safe_text(cursor, "SELECT MAX(v) FROM t") is None

    def test_returns_string_when_present(self):
        with sqlite3.connect(":memory:") as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE t (v TEXT)")
            cursor.execute("INSERT INTO t VALUES ('hello')")
            assert dash._safe_text(cursor, "SELECT v FROM t") == "hello"


class TestTableColumns:
    def test_returns_columns(self):
        with sqlite3.connect(":memory:") as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE trades (status TEXT, pnl REAL)")
            cols = dash._table_columns(cursor, "trades")
            assert "status" in cols
            assert "pnl" in cols

    def test_returns_empty_set_on_error(self):
        with sqlite3.connect(":memory:") as conn:
            cursor = conn.cursor()
            # Non-existent table returns empty PRAGMA result, not an error
            cols = dash._table_columns(cursor, "nonexistent_table")
            assert cols == set()


# ---------------------------------------------------------------------------
# scenario_dashboard – unrealized PnL query
# ---------------------------------------------------------------------------


class TestQueryUnrealizedPnl:
    def test_returns_none_when_db_missing(self, tmp_path):
        assert dash._query_unrealized_pnl(tmp_path / "nope.db") is None

    def test_returns_none_when_required_columns_missing(self, tmp_path):
        db = tmp_path / "partial.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("CREATE TABLE trades (status TEXT)")
            conn.execute("CREATE TABLE market_prices (id INTEGER)")
        assert dash._query_unrealized_pnl(db) is None

    def test_computes_buy_yes_unrealized(self, tmp_path):
        db = tmp_path / "test.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                "CREATE TABLE trades "
                "(id INTEGER, market_id TEXT, status TEXT, side TEXT, amount REAL, cost REAL)"
            )
            conn.execute(
                "CREATE TABLE market_prices "
                "(id INTEGER PRIMARY KEY, market_id TEXT, yes_price REAL, timestamp TEXT)"
            )
            conn.execute(
                "INSERT INTO trades VALUES (1, 'm1', 'open', 'buy_yes', 10.0, 4.0)"
            )
            conn.execute(
                "INSERT INTO market_prices VALUES (1, 'm1', 0.6, '2026-01-01T10:00:00')"
            )
        result = dash._query_unrealized_pnl(db)
        # value = 0.6 * 10 = 6, cost = 4, unrealized = 2
        assert result == pytest.approx(2.0)

    def test_computes_buy_no_unrealized(self, tmp_path):
        db = tmp_path / "test.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                "CREATE TABLE trades "
                "(id INTEGER, market_id TEXT, status TEXT, side TEXT, amount REAL, cost REAL)"
            )
            conn.execute(
                "CREATE TABLE market_prices "
                "(id INTEGER PRIMARY KEY, market_id TEXT, yes_price REAL, timestamp TEXT)"
            )
            conn.execute(
                "INSERT INTO trades VALUES (1, 'm1', 'open', 'buy_no', 10.0, 3.0)"
            )
            conn.execute(
                "INSERT INTO market_prices VALUES (1, 'm1', 0.7, '2026-01-01T10:00:00')"
            )
        result = dash._query_unrealized_pnl(db)
        # value = (1 - 0.7) * 10 = 3, cost = 3, unrealized = 0
        assert result == pytest.approx(0.0)

    def test_skips_rows_with_null_yes_price(self, tmp_path):
        db = tmp_path / "test.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                "CREATE TABLE trades "
                "(id INTEGER, market_id TEXT, status TEXT, side TEXT, amount REAL, cost REAL)"
            )
            conn.execute(
                "CREATE TABLE market_prices "
                "(id INTEGER PRIMARY KEY, market_id TEXT, yes_price REAL, timestamp TEXT)"
            )
            conn.execute(
                "INSERT INTO trades VALUES (1, 'm_no_price', 'open', 'buy_yes', 10.0, 4.0)"
            )
            # No market_prices row → yes_price will be NULL in LEFT JOIN
        result = dash._query_unrealized_pnl(db)
        assert result == pytest.approx(0.0)

    def test_returns_none_on_db_error(self, tmp_path):
        db = tmp_path / "bad.db"
        db.write_text("not a sqlite database", encoding="utf-8")
        assert dash._query_unrealized_pnl(db) is None


# ---------------------------------------------------------------------------
# scenario_dashboard – query stats
# ---------------------------------------------------------------------------


class TestQueryStats:
    def test_returns_empty_stats_when_db_missing(self, tmp_path):
        result = dash._query_stats(tmp_path / "nope.db")
        assert result == dash.EMPTY_STATS

    def test_returns_empty_stats_on_db_error(self, tmp_path):
        db = tmp_path / "bad.db"
        db.write_text("garbage", encoding="utf-8")
        result = dash._query_stats(db)
        assert result == dash.EMPTY_STATS

    def test_win_rate_is_none_when_no_resolved_trades(self, tmp_path):
        db = tmp_path / "empty.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("CREATE TABLE trades (status TEXT, pnl REAL)")
            conn.execute("CREATE TABLE forecasts (forecast_timestamp TEXT)")
            conn.execute("CREATE TABLE market_prices (id INTEGER, timestamp TEXT)")
            conn.execute("CREATE TABLE markets (id TEXT)")
        stats = dash._query_stats(db)
        assert stats.win_rate is None

    def test_win_rate_calculated_when_resolved_trades_exist(self, tmp_path):
        db = tmp_path / "trades.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute("CREATE TABLE trades (status TEXT, pnl REAL)")
            conn.execute("CREATE TABLE forecasts (forecast_timestamp TEXT)")
            conn.execute("CREATE TABLE market_prices (id INTEGER, timestamp TEXT)")
            conn.execute("CREATE TABLE markets (id TEXT)")
            conn.executemany(
                "INSERT INTO trades VALUES (?, ?)",
                [("won", 1.0), ("won", 1.0), ("lost", -1.0)],
            )
        stats = dash._query_stats(db)
        assert stats.win_rate == pytest.approx(200.0 / 3.0)


# ---------------------------------------------------------------------------
# scenario_dashboard – open trade ETA
# ---------------------------------------------------------------------------


class TestQueryOpenTradeEta:
    def test_returns_empty_when_db_missing(self, tmp_path):
        assert dash._query_open_trade_eta(tmp_path / "nope.db") == []

    def test_returns_empty_on_db_error(self, tmp_path):
        db = tmp_path / "bad.db"
        db.write_text("garbage", encoding="utf-8")
        assert dash._query_open_trade_eta(db) == []

    def test_returns_eta_entries(self, tmp_path):
        db = tmp_path / "trades.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                "CREATE TABLE trades "
                "(id INTEGER PRIMARY KEY, market_id TEXT, status TEXT, side TEXT, created_at TEXT)"
            )
            conn.execute(
                "CREATE TABLE markets (id TEXT PRIMARY KEY, date TEXT, station TEXT)"
            )
            conn.execute(
                "INSERT INTO trades VALUES (1, 'm1', 'open', 'buy_yes', '2026-01-01T10:00:00')"
            )
            conn.execute("INSERT INTO markets VALUES ('m1', NULL, NULL)")
        results = dash._query_open_trade_eta(db)
        assert len(results) == 1
        assert results[0].trade_id == 1
        assert results[0].hours_left is None  # date=NULL → no ETA computed

    def test_hours_left_is_clamped_to_zero(self, tmp_path):
        db = tmp_path / "trades.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                "CREATE TABLE trades "
                "(id INTEGER PRIMARY KEY, market_id TEXT, status TEXT, side TEXT, created_at TEXT)"
            )
            conn.execute(
                "CREATE TABLE markets (id TEXT PRIMARY KEY, date TEXT, station TEXT)"
            )
            conn.execute(
                "INSERT INTO trades VALUES (1, 'm1', 'open', 'buy_yes', '2026-01-01')"
            )
            # Use a past date to get negative hours → should be clamped to 0
            conn.execute("INSERT INTO markets VALUES ('m1', '2020-01-01', 'KNYC')")

        with patch("weatheredge.dashboard.scenario_dashboard.hours_until_market_close") as mock_h:
            mock_h.return_value = -5.0  # simulate past event
            results = dash._query_open_trade_eta(db)

        assert len(results) == 1
        assert results[0].hours_left == pytest.approx(0.0)

    def test_hours_left_exception_yields_none(self, tmp_path):
        db = tmp_path / "trades.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                "CREATE TABLE trades "
                "(id INTEGER PRIMARY KEY, market_id TEXT, status TEXT, side TEXT, created_at TEXT)"
            )
            conn.execute(
                "CREATE TABLE markets (id TEXT PRIMARY KEY, date TEXT, station TEXT)"
            )
            conn.execute("INSERT INTO trades VALUES (1, 'm1', 'open', 'buy_yes', '2026-01-01')")
            conn.execute("INSERT INTO markets VALUES ('m1', '2026-06-01', 'KNYC')")

        with patch("weatheredge.dashboard.scenario_dashboard.hours_until_market_close") as mock_h:
            mock_h.side_effect = RuntimeError("boom")
            results = dash._query_open_trade_eta(db)

        assert results[0].hours_left is None


# ---------------------------------------------------------------------------
# scenario_dashboard – collector helpers
# ---------------------------------------------------------------------------


class TestCollectorHelpers:
    def test_read_collector_pid_delegates_to_read_pid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", tmp_path / "collector.pid")
        assert dash._read_collector_pid() is None

    def test_collector_running_false_when_no_pid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", tmp_path / "collector.pid")
        assert dash._collector_running() is False

    def test_collector_running_true_when_pid_alive(self, tmp_path, monkeypatch):
        pid_file = tmp_path / "collector.pid"
        pid_file.write_text("12345", encoding="utf-8")
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", pid_file)
        monkeypatch.setattr(dash, "_is_pid_running", lambda _pid: True)
        assert dash._collector_running() is True

    def test_collector_running_false_and_cleans_stale_pid(self, tmp_path, monkeypatch):
        pid_file = tmp_path / "collector.pid"
        pid_file.write_text("12345", encoding="utf-8")
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", pid_file)
        monkeypatch.setattr(dash, "_is_pid_running", lambda _pid: False)
        assert dash._collector_running() is False
        assert not pid_file.exists()

    def test_collector_status_text_stop_when_no_pid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", tmp_path / "c.pid")
        assert dash._collector_status_text() == "STOP"

    def test_collector_status_text_run_when_running(self, tmp_path, monkeypatch):
        pid_file = tmp_path / "c.pid"
        pid_file.write_text("99", encoding="utf-8")
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", pid_file)
        monkeypatch.setattr(dash, "_is_pid_running", lambda _pid: True)
        text = dash._collector_status_text()
        assert "RUN" in text
        assert "99" in text


# ---------------------------------------------------------------------------
# scenario_dashboard – _ensure_snapshot_collector_running
# ---------------------------------------------------------------------------


class TestEnsureSnapshotCollectorRunning:
    def test_returns_false_if_already_running(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "_collector_running", lambda: True)
        pid_file = tmp_path / "c.pid"
        pid_file.write_text("55", encoding="utf-8")
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", pid_file)
        ok, msg = dash._ensure_snapshot_collector_running()
        assert ok is False
        assert "55" in msg

    def test_returns_true_on_successful_launch(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "_collector_running", lambda: False)
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", tmp_path / "c.pid")
        monkeypatch.setattr(dash, "COLLECTOR_LOG_FILE", tmp_path / "c.log")
        monkeypatch.setattr(dash, "COLLECTOR_OUTPUT_FILE", tmp_path / "c.csv")
        monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)

        class FakeProc:
            pid = 777
            def poll(self): return None

        monkeypatch.setattr(dash.subprocess, "Popen", lambda *a, **kw: FakeProc())
        monkeypatch.setattr(dash.time, "sleep", lambda _: None)

        ok, msg = dash._ensure_snapshot_collector_running()
        assert ok is True
        assert "777" in msg

    def test_returns_false_when_process_exits_immediately(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "_collector_running", lambda: False)
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", tmp_path / "c.pid")
        monkeypatch.setattr(dash, "COLLECTOR_LOG_FILE", tmp_path / "c.log")
        monkeypatch.setattr(dash, "COLLECTOR_OUTPUT_FILE", tmp_path / "c.csv")
        monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)

        class FakeProc:
            pid = 888
            def poll(self): return 1  # process already dead

        monkeypatch.setattr(dash.subprocess, "Popen", lambda *a, **kw: FakeProc())
        monkeypatch.setattr(dash.time, "sleep", lambda _: None)

        ok, msg = dash._ensure_snapshot_collector_running()
        assert ok is False
        assert "fallito" in msg

    def test_returns_false_on_exception(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "_collector_running", lambda: False)
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", tmp_path / "c.pid")
        monkeypatch.setattr(dash, "COLLECTOR_LOG_FILE", tmp_path / "c.log")
        monkeypatch.setattr(dash, "COLLECTOR_OUTPUT_FILE", tmp_path / "c.csv")
        monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)

        def boom(*a, **kw):
            raise OSError("no such file")

        monkeypatch.setattr(dash.subprocess, "Popen", boom)

        ok, msg = dash._ensure_snapshot_collector_running()
        assert ok is False
        assert "errore" in msg


# ---------------------------------------------------------------------------
# scenario_dashboard – any scenario running / stop collector if idle
# ---------------------------------------------------------------------------


class TestAnyScenarioRunning:
    def test_returns_false_when_no_pid_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        assert dash._any_scenario_running() is False

    def test_returns_true_when_one_scenario_has_running_pid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        paths = dash.scenario_paths("bilanciato")
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.pid_file.write_text("1234", encoding="utf-8")
        monkeypatch.setattr(dash, "_is_pid_running", lambda _pid: True)
        assert dash._any_scenario_running() is True


class TestStopSnapshotCollectorIfIdle:
    def test_returns_none_when_scenario_still_running(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "_any_scenario_running", lambda: True)
        assert dash._stop_snapshot_collector_if_idle() is None

    def test_returns_none_when_no_collector_pid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "_any_scenario_running", lambda: False)
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", tmp_path / "c.pid")
        assert dash._stop_snapshot_collector_if_idle() is None

    def test_cleans_stale_pid(self, tmp_path, monkeypatch):
        pid_file = tmp_path / "c.pid"
        pid_file.write_text("999", encoding="utf-8")
        monkeypatch.setattr(dash, "_any_scenario_running", lambda: False)
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", pid_file)
        monkeypatch.setattr(dash, "_is_pid_running", lambda _pid: False)
        msg = dash._stop_snapshot_collector_if_idle()
        assert msg is not None
        assert "stale" in msg.lower() or "pulito" in msg.lower()
        assert not pid_file.exists()

    def test_sends_sigterm_and_reports_stopped(self, tmp_path, monkeypatch):
        pid_file = tmp_path / "c.pid"
        pid_file.write_text("9001", encoding="utf-8")
        monkeypatch.setattr(dash, "_any_scenario_running", lambda: False)
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", pid_file)

        call_count = {"n": 0}

        def fake_is_running(pid):
            call_count["n"] += 1
            return call_count["n"] <= 1  # running once, then stops

        monkeypatch.setattr(dash, "_is_pid_running", fake_is_running)
        monkeypatch.setattr(dash.os, "kill", lambda pid, sig: None)
        monkeypatch.setattr(dash.time, "sleep", lambda _: None)

        msg = dash._stop_snapshot_collector_if_idle()
        assert msg is not None
        assert "fermato" in msg

    def test_force_kills_on_timeout(self, tmp_path, monkeypatch):
        pid_file = tmp_path / "c.pid"
        pid_file.write_text("9002", encoding="utf-8")
        monkeypatch.setattr(dash, "_any_scenario_running", lambda: False)
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", pid_file)
        monkeypatch.setattr(dash, "_is_pid_running", lambda _pid: True)  # never stops
        monkeypatch.setattr(dash.os, "kill", lambda pid, sig: None)

        import time as _time
        # Force deadline to be immediately passed
        monkeypatch.setattr(dash.time, "sleep", lambda _: None)
        original_time = _time.time
        call_count = {"n": 0}

        def fake_time():
            call_count["n"] += 1
            # First call sets deadline, subsequent calls exceed it
            return original_time() + (100 if call_count["n"] > 1 else 0)

        monkeypatch.setattr(dash.time, "time", fake_time)

        msg = dash._stop_snapshot_collector_if_idle(timeout_seconds=0.001)
        assert msg is not None
        assert "forzatamente" in msg or "non riuscito" in msg


# ---------------------------------------------------------------------------
# scenario_dashboard – stop_scenario (SIGTERM → SIGKILL path)
# ---------------------------------------------------------------------------


class TestStopScenarioSignalPaths:
    def test_sigterm_oserror_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        paths = dash.scenario_paths("safer")
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.pid_file.write_text("8888", encoding="utf-8")
        monkeypatch.setattr(dash, "_is_pid_running", lambda _pid: True)

        def fake_kill(pid, sig):
            raise OSError("no process")

        monkeypatch.setattr(dash.os, "kill", fake_kill)

        ok, msg = dash.stop_scenario("safer")
        assert ok is False
        assert "non trovato" in msg

    def test_sigkill_sent_after_timeout(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        paths = dash.scenario_paths("safer")
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.pid_file.write_text("7777", encoding="utf-8")

        killed = {"sigkill": False}

        def fake_is_running(pid):
            return not killed["sigkill"]

        def fake_kill(pid, sig):
            if sig == signal.SIGKILL:
                killed["sigkill"] = True

        monkeypatch.setattr(dash, "_is_pid_running", fake_is_running)
        monkeypatch.setattr(dash.os, "kill", fake_kill)
        monkeypatch.setattr(dash.time, "sleep", lambda _: None)

        import time as _time
        original_time = _time.time
        n = {"i": 0}

        def fake_time():
            n["i"] += 1
            return original_time() + (100 if n["i"] > 1 else 0)

        monkeypatch.setattr(dash.time, "time", fake_time)

        ok, msg = dash.stop_scenario("safer", timeout_seconds=0.001)
        assert ok is True
        assert "SIGKILL" in msg

    def test_start_scenario_already_running(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)

        paths = dash.scenario_paths("rischioso")
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.pid_file.write_text("5555", encoding="utf-8")
        monkeypatch.setattr(dash, "_is_pid_running", lambda _pid: True)

        ok, msg = dash.start_scenario("rischioso")
        assert ok is False
        assert "gia avviato" in msg

    def test_start_scenario_popen_exception(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)

        def boom(*a, **kw):
            raise OSError("popen failed")

        monkeypatch.setattr(dash.subprocess, "Popen", boom)

        ok, msg = dash.start_scenario("bilanciato")
        assert ok is False
        assert "errore" in msg


# ---------------------------------------------------------------------------
# scenario_dashboard – formatting helpers
# ---------------------------------------------------------------------------


class TestFmtHelpers:
    def test_fmt_int_none(self):
        assert dash._fmt_int(None) == "-"

    def test_fmt_int_value(self):
        assert dash._fmt_int(42) == "42"

    def test_fmt_float_none(self):
        assert dash._fmt_float(None) == "-"

    def test_fmt_float_value(self):
        assert dash._fmt_float(3.14159, 2) == "3.14"

    def test_fmt_age_minutes_none(self):
        assert dash._fmt_age_minutes(None) == "-"

    def test_fmt_age_minutes_negative_clamped(self):
        assert dash._fmt_age_minutes(-10.0) == "0m"

    def test_fmt_age_minutes_positive(self):
        assert dash._fmt_age_minutes(90.0) == "90m"

    def test_fmt_ts_none(self):
        assert dash._fmt_ts(None) == "-"

    def test_fmt_ts_empty(self):
        assert dash._fmt_ts("") == "-"

    def test_fmt_ts_replaces_T_and_truncates(self):
        assert dash._fmt_ts("2026-04-01T12:00:00") == "2026-04-01 12:00:00"

    def test_fmt_ts_truncates_long_string(self):
        assert dash._fmt_ts("2026-04-01T12:00:00.123456") == "2026-04-01 12:00:00"


# ---------------------------------------------------------------------------
# scenario_dashboard – print_status_table, print_logs, build_parser, main
# ---------------------------------------------------------------------------


class TestPrintStatusTable:
    def test_print_status_table_runs_without_error(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", tmp_path / "c.pid")
        output = capsys.readouterr()
        # Just verify it runs and produces output
        dash.print_status_table()
        out = capsys.readouterr().out
        assert "WeatherEdge" in out
        assert "rischioso" in out
        assert "bilanciato" in out
        assert "safer" in out


class TestPrintLogs:
    def test_print_logs_invalid_scenario(self, capsys):
        rc = dash.print_logs("nonexistent")
        assert rc == 2
        assert "non valido" in capsys.readouterr().out

    def test_print_logs_no_log_file(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        rc = dash.print_logs("bilanciato")
        assert rc == 0
        assert "nessun log" in capsys.readouterr().out

    def test_print_logs_shows_lines(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        paths = dash.scenario_paths("bilanciato")
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.log_file.write_text("line1\nline2\nline3\n", encoding="utf-8")
        rc = dash.print_logs("bilanciato", lines=2)
        assert rc == 0
        out = capsys.readouterr().out
        assert "line2" in out
        assert "line3" in out


class TestBuildParser:
    def test_build_parser_returns_parser(self):
        p = dash.build_parser()
        args = p.parse_args(["status"])
        assert args.command == "status"

    def test_build_parser_start_default_target(self):
        p = dash.build_parser()
        args = p.parse_args(["start"])
        assert args.target == "all"

    def test_build_parser_logs_command(self):
        p = dash.build_parser()
        args = p.parse_args(["logs", "bilanciato", "--lines", "10"])
        assert args.scenario == "bilanciato"
        assert args.lines == 10


class TestMain:
    def test_main_status_command_returns_0(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        monkeypatch.setattr(dash, "COLLECTOR_PID_FILE", tmp_path / "c.pid")
        rc = dash.main(["status"])
        assert rc == 0

    def test_main_logs_command(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        rc = dash.main(["logs", "bilanciato"])
        assert rc == 0

    def test_main_start_command_calls_run_start_or_stop(self, monkeypatch):
        called = {}

        def fake_run(action, target):
            called["action"] = action
            called["target"] = target
            return 0

        monkeypatch.setattr(dash, "_run_start_or_stop", fake_run)
        rc = dash.main(["start", "all"])
        assert rc == 0
        assert called["action"] == "start"

    def test_main_stop_command(self, monkeypatch):
        called = {}

        def fake_run(action, target):
            called["action"] = action
            return 0

        monkeypatch.setattr(dash, "_run_start_or_stop", fake_run)
        rc = dash.main(["stop", "bilanciato"])
        assert rc == 0
        assert called["action"] == "stop"

    def test_main_no_command_calls_run_dashboard(self, monkeypatch):
        monkeypatch.setattr(dash, "run_dashboard", lambda: 0)
        rc = dash.main([])
        assert rc == 0

    def test_main_dashboard_command_calls_run_dashboard(self, monkeypatch):
        monkeypatch.setattr(dash, "run_dashboard", lambda: 0)
        rc = dash.main(["dashboard"])
        assert rc == 0


class TestRunStartOrStop:
    def test_invalid_target_returns_2(self, capsys):
        rc = dash._run_start_or_stop("start", "garbage")
        assert rc == 2

    def test_stop_action_invokes_collector_stop(self, monkeypatch):
        monkeypatch.setattr(dash, "_targets", lambda _: ["safer"])
        monkeypatch.setattr(dash, "stop_scenario", lambda _name: (True, "stopped"))
        stop_called = {"called": False}

        def fake_stop_if_idle():
            stop_called["called"] = True
            return "collector stopped"

        monkeypatch.setattr(dash, "_stop_snapshot_collector_if_idle", fake_stop_if_idle)

        rc = dash._run_start_or_stop("stop", "safer")
        assert rc == 0
        assert stop_called["called"] is True

    def test_partial_success_returns_1(self, monkeypatch):
        scenarios = ["rischioso", "bilanciato"]
        monkeypatch.setattr(dash, "_targets", lambda _: scenarios)
        results = {"rischioso": (True, "ok"), "bilanciato": (False, "fail")}
        monkeypatch.setattr(dash, "start_scenario", lambda name: results[name])
        monkeypatch.setattr(dash, "_ensure_snapshot_collector_running", lambda: (True, "collector ok"))

        rc = dash._run_start_or_stop("start", "all")
        assert rc == 1


# ---------------------------------------------------------------------------
# scenario_snapshot_collector – _to_float
# ---------------------------------------------------------------------------


class TestToFloat:
    def test_none_returns_zero(self):
        assert collector._to_float(None) == 0.0

    def test_string_number_converts(self):
        assert collector._to_float("3.14") == pytest.approx(3.14)

    def test_int_converts(self):
        assert collector._to_float(5) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# scenario_snapshot_collector – _compute_mark_to_market
# ---------------------------------------------------------------------------


class TestComputeMarkToMarket:
    def test_returns_defaults_when_db_missing(self, tmp_path):
        result = collector._compute_mark_to_market(tmp_path / "nope.db", 500.0)
        assert result["open_cost"] == 0.0
        assert result["equity_now"] == 500.0
        assert result["missing_prices"] == 0

    def test_missing_prices_counted(self, tmp_path):
        db = tmp_path / "test.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                "CREATE TABLE trades (id INTEGER, market_id TEXT, status TEXT, side TEXT, amount REAL, cost REAL, pnl REAL)"
            )
            conn.execute(
                "CREATE TABLE market_prices (id INTEGER PRIMARY KEY, market_id TEXT, yes_price REAL, timestamp TEXT)"
            )
            # open trade with no matching price row
            conn.execute("INSERT INTO trades VALUES (1, 'no_price_market', 'open', 'buy_yes', 10.0, 5.0, NULL)")

        result = collector._compute_mark_to_market(db, 1000.0)
        assert result["missing_prices"] == 1
        assert result["open_cost"] == pytest.approx(5.0)
        # mark_value uses cost as fallback when price is missing
        assert result["mark_value"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# scenario_snapshot_collector – collect_snapshot_rows
# ---------------------------------------------------------------------------


class TestCollectSnapshotRows:
    def test_returns_rows_for_all_scenarios_plus_total(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")

        rows = collector.collect_snapshot_rows(initial_bankroll=500.0, include_total=True)
        # 3 scenarios + 1 TOTAL
        assert len(rows) == 4
        scenarios_in = {r["scenario"] for r in rows}
        assert "TOTAL" in scenarios_in
        assert "rischioso" in scenarios_in

    def test_returns_rows_without_total(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        rows = collector.collect_snapshot_rows(include_total=False)
        assert len(rows) == 3
        assert all(r["scenario"] != "TOTAL" for r in rows)

    def test_total_row_aggregates_fields(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        rows = collector.collect_snapshot_rows(initial_bankroll=1000.0, include_total=True)
        total = next(r for r in rows if r["scenario"] == "TOTAL")
        assert total["running"] == ""
        assert total["pid"] == ""
        assert isinstance(total["trades"], int)


# ---------------------------------------------------------------------------
# scenario_snapshot_collector – _print_brief
# ---------------------------------------------------------------------------


class TestPrintBrief:
    def test_prints_scenario_and_total(self, capsys):
        rows = [
            {
                "scenario": "rischioso",
                "open_positions": 2,
                "unrealized_pnl": 3.5,
                "equity_now": 1003.5,
            },
            {
                "scenario": "TOTAL",
                "unrealized_pnl": 3.5,
                "realized_pnl": 0.0,
                "equity_now": 3003.5,
            },
        ]
        collector._print_brief(rows)
        out = capsys.readouterr().out
        assert "rischioso" in out
        assert "TOTAL" in out
        assert "3003.5" in out


# ---------------------------------------------------------------------------
# scenario_snapshot_collector – run_collector
# ---------------------------------------------------------------------------


class TestRunCollector:
    def test_run_collector_one_iteration(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        out = tmp_path / "snapshots.csv"
        rc = collector.run_collector(
            output_path=out,
            interval_seconds=1,
            iterations=1,
            initial_bankroll=500.0,
            print_stdout=False,
        )
        assert rc == 0
        assert out.exists()

    def test_run_collector_prints_when_stdout_enabled(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        out = tmp_path / "snapshots.csv"
        collector.run_collector(
            output_path=out,
            interval_seconds=1,
            iterations=1,
            print_stdout=True,
        )
        captured = capsys.readouterr().out
        assert "snapshot saved" in captured


# ---------------------------------------------------------------------------
# scenario_snapshot_collector – build_parser and main
# ---------------------------------------------------------------------------


class TestSnapshotCollectorBuildParser:
    def test_default_interval(self):
        p = collector.build_parser()
        args = p.parse_args(["--output", "/tmp/x.csv"])
        assert args.interval_seconds == 1800

    def test_once_flag(self):
        p = collector.build_parser()
        args = p.parse_args(["--output", "/tmp/x.csv", "--once"])
        assert args.once is True

    def test_no_total_flag(self):
        p = collector.build_parser()
        args = p.parse_args(["--output", "/tmp/x.csv", "--no-total"])
        assert args.no_total is True

    def test_quiet_flag(self):
        p = collector.build_parser()
        args = p.parse_args(["--output", "/tmp/x.csv", "--quiet"])
        assert args.quiet is True

    def test_initial_bankroll(self):
        p = collector.build_parser()
        args = p.parse_args(["--output", "/tmp/x.csv", "--initial-bankroll", "2000.0"])
        assert args.initial_bankroll == pytest.approx(2000.0)


class TestSnapshotCollectorMain:
    def test_main_once_flag(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        out = tmp_path / "snapshots.csv"
        rc = collector.main(["--output", str(out), "--once", "--quiet"])
        assert rc == 0
        assert out.exists()

    def test_main_keyboard_interrupt_returns_130(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, "RUN_ROOT", tmp_path / "runs")
        out = tmp_path / "s.csv"

        def fake_run_collector(**kwargs):
            raise KeyboardInterrupt()

        monkeypatch.setattr(collector, "run_collector", fake_run_collector)
        rc = collector.main(["--output", str(out), "--iterations", "1"])
        assert rc == 130
