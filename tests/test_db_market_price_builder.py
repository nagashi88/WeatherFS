"""Tests for DB -> backtest market price CSV builder."""

from __future__ import annotations

import csv
import sqlite3
from datetime import date

from weatheredge.backtesting.db_market_price_builder import build_market_prices_csv_from_db


def _init_db(path):
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE market_prices (
                id INTEGER PRIMARY KEY,
                market_id TEXT,
                timestamp TEXT,
                yes_price FLOAT,
                no_price FLOAT,
                volume FLOAT,
                spread FLOAT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE markets (
                id TEXT PRIMARY KEY,
                platform TEXT NOT NULL,
                city TEXT NOT NULL,
                station TEXT NOT NULL,
                date TEXT NOT NULL,
                market_type TEXT NOT NULL,
                threshold_low FLOAT,
                threshold_high FLOAT,
                resolution_source TEXT,
                created_at TEXT
            )
            """
        )
        conn.commit()


def test_build_market_prices_csv_from_db_uses_latest_snapshot(tmp_path):
    db_path = tmp_path / "weatheredge.db"
    out_path = tmp_path / "prices.csv"
    _init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO market_prices (market_id, timestamp, yes_price) VALUES (?, ?, ?)",
            [
                ("KXHIGHDEN-26MAR30-T76", "2026-03-30 10:00:00", 0.55),
                ("KXHIGHDEN-26MAR30-T76", "2026-03-30 11:00:00", 0.61),
                ("KXHIGHDEN-26MAR30-B76", "2026-03-30 11:00:00", 0.42),
                ("NOT-A-WEATHER-TICKER", "2026-03-30 11:00:00", 0.90),  # ignored
            ],
        )
        conn.commit()

    report = build_market_prices_csv_from_db(db_path=db_path, output_csv=out_path)
    assert report.output_rows == 2

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    by_key = {(r["station"], r["date"], r["threshold"], r["direction"]): r for r in rows}
    assert float(by_key[("KDEN", "2026-03-30", "76.00", "above")]["yes_price"]) == 0.61
    assert float(by_key[("KDEN", "2026-03-30", "76.00", "below")]["yes_price"]) == 0.42


def test_build_market_prices_csv_from_db_applies_contract_date_filters(tmp_path):
    db_path = tmp_path / "weatheredge.db"
    out_path = tmp_path / "prices.csv"
    _init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO market_prices (market_id, timestamp, yes_price) VALUES (?, ?, ?)",
            [
                ("KXHIGHDEN-26MAR29-T76", "2026-03-29 11:00:00", 0.51),
                ("KXHIGHDEN-26MAR30-T76", "2026-03-30 11:00:00", 0.61),
            ],
        )
        conn.commit()

    report = build_market_prices_csv_from_db(
        db_path=db_path,
        output_csv=out_path,
        start_date=date.fromisoformat("2026-03-30"),
        end_date=date.fromisoformat("2026-03-30"),
    )
    assert report.output_rows == 1

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["date"] == "2026-03-30"


def test_build_market_prices_csv_from_db_uses_markets_metadata_for_numeric_ids(tmp_path):
    db_path = tmp_path / "weatheredge.db"
    out_path = tmp_path / "prices.csv"
    _init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO markets (
                id, platform, city, station, date, market_type, threshold_high, resolution_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "1781474",
                "polymarket",
                "denver",
                "KDEN",
                "2026-04-01",
                "daily_high",
                62.0,
                '{"direction":"above","threshold":62.0}',
            ),
        )
        conn.execute(
            "INSERT INTO market_prices (market_id, timestamp, yes_price) VALUES (?, ?, ?)",
            ("1781474", "2026-04-01 07:00:00", 0.33),
        )
        conn.commit()

    report = build_market_prices_csv_from_db(db_path=db_path, output_csv=out_path)
    assert report.output_rows == 1

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["station"] == "KDEN"
    assert row["date"] == "2026-04-01"
    assert row["threshold"] == "62.00"
    assert row["direction"] == "above"
