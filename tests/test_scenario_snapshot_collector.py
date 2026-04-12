"""Tests for scenario snapshot collector utilities."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from weatheredge.dashboard import scenario_snapshot_collector as collector


def _create_test_db(path: Path) -> None:
    with sqlite3.connect(str(path)) as conn:
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE trades ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "market_id TEXT, status TEXT, side TEXT, amount REAL, cost REAL, pnl REAL)"
        )
        cur.execute(
            "CREATE TABLE market_prices ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "market_id TEXT, yes_price REAL, timestamp TEXT)"
        )

        # Open positions
        cur.execute(
            "INSERT INTO trades (market_id, status, side, amount, cost, pnl) "
            "VALUES ('m_yes', 'open', 'buy_yes', 10, 5, NULL)"
        )
        cur.execute(
            "INSERT INTO trades (market_id, status, side, amount, cost, pnl) "
            "VALUES ('m_no', 'open', 'buy_no', 10, 4, NULL)"
        )
        # Closed trade (realized pnl)
        cur.execute(
            "INSERT INTO trades (market_id, status, side, amount, cost, pnl) "
            "VALUES ('m_done', 'won', 'buy_yes', 10, 3, 2)"
        )

        cur.execute(
            "INSERT INTO market_prices (market_id, yes_price, timestamp) "
            "VALUES ('m_yes', 0.7, '2026-04-01 12:00:00')"
        )
        cur.execute(
            "INSERT INTO market_prices (market_id, yes_price, timestamp) "
            "VALUES ('m_no', 0.8, '2026-04-01 12:00:00')"
        )
        conn.commit()


def test_compute_mark_to_market_buy_yes_and_buy_no(tmp_path):
    db = tmp_path / "scenario.db"
    _create_test_db(db)

    mtm = collector._compute_mark_to_market(db_path=db, initial_bankroll=1000.0)

    # buy_yes value = 0.7*10 = 7, cost=5 => +2
    # buy_no value = (1-0.8)*10 = 2, cost=4 => -2
    # unrealized = 0, realized = +2
    assert mtm["open_cost"] == pytest.approx(9.0)
    assert mtm["mark_value"] == pytest.approx(9.0)
    assert mtm["unrealized_pnl"] == pytest.approx(0.0)
    assert mtm["realized_pnl"] == pytest.approx(2.0)
    assert mtm["equity_now"] == pytest.approx(1002.0)
    assert mtm["missing_prices"] == 0


def test_append_snapshot_rows_writes_header_once(tmp_path):
    out = tmp_path / "snapshots.csv"
    row = {
        "collected_at_utc": "2026-04-01T12:00:00+00:00",
        "scenario": "rischioso",
        "running": 1,
        "pid": 123,
        "uptime": "00:10",
        "trades": 1,
        "open_positions": 1,
        "wins": 0,
        "losses": 0,
        "win_rate_pct": "",
        "pnl_reported": 0.0,
        "forecasts": 10,
        "market_prices": 20,
        "markets": 3,
        "forecast_age_min": 5.0,
        "price_age_min": 1.0,
        "forecast_ts": "2026-04-01 12:00:00",
        "price_ts": "2026-04-01 12:01:00",
        "open_cost": 10.0,
        "mark_value": 10.5,
        "unrealized_pnl": 0.5,
        "realized_pnl": 0.0,
        "equity_now": 1000.5,
        "missing_prices": 0,
        "last_log_line": "ok",
    }

    collector.append_snapshot_rows(out, [row])
    collector.append_snapshot_rows(out, [row])

    with out.open("r", encoding="utf-8", newline="") as handle:
        lines = handle.read().strip().splitlines()
    assert len(lines) == 3  # header + 2 rows

    with out.open("r", encoding="utf-8", newline="") as handle:
        parsed = list(csv.DictReader(handle))
    assert len(parsed) == 2
    assert parsed[0]["scenario"] == "rischioso"

