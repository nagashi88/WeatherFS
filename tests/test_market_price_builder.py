"""Tests for Kalshi parquet -> backtest CSV builder."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd

from weatheredge.backtesting.market_price_builder import (
    build_market_prices_csv_from_kalshi_parquet,
    kalshi_dataframe_to_backtest_rows,
)


def test_kalshi_dataframe_to_backtest_rows_filters_and_uses_midpoint():
    df = pd.DataFrame(
        [
            {
                "ticker": "KXHIGHNY-26MAR30-T76",
                "yes_bid": 60,
                "yes_ask": 64,
                "_fetched_at": "2026-03-30T10:00:00Z",
            },
            {
                "ticker": "PRES-2028-DJT",  # non-weather
                "yes_bid": 20,
                "yes_ask": 30,
                "_fetched_at": "2026-03-30T10:00:00Z",
            },
        ]
    )

    rows = kalshi_dataframe_to_backtest_rows(df)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "KXHIGHNY-26MAR30-T76"
    assert rows[0]["yes_price"] == 0.62


def test_build_market_prices_csv_uses_latest_snapshot(tmp_path, monkeypatch):
    older = pd.DataFrame(
        [
            {
                "ticker": "KXHIGHNY-26MAR30-T76",
                "yes_ask": 55,
                "_fetched_at": "2026-03-30T10:00:00Z",
            }
        ]
    )
    newer = pd.DataFrame(
        [
            {
                "ticker": "KXHIGHNY-26MAR30-T76",
                "yes_ask": 65,
                "_fetched_at": "2026-03-30T12:00:00Z",
            }
        ]
    )
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    p1.touch()
    p2.touch()

    def fake_read_parquet(path: Path):
        return older if Path(path).name == "a.parquet" else newer

    monkeypatch.setattr("weatheredge.backtesting.market_price_builder.pd.read_parquet", fake_read_parquet)

    out = tmp_path / "market_prices.csv"
    report = build_market_prices_csv_from_kalshi_parquet([tmp_path], out)

    assert report.output_rows == 2  # above + complement below
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    by_dir = {r["direction"]: r for r in rows}
    assert by_dir["above"]["station"] == "KNYC"
    assert by_dir["above"]["date"] == "2026-03-30"
    assert float(by_dir["above"]["yes_price"]) == 0.65
    assert float(by_dir["below"]["yes_price"]) == 0.35


def test_build_market_prices_csv_date_filters(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        [
            {"ticker": "KXHIGHNY-26MAR30-T76", "yes_ask": 60, "_fetched_at": "2026-03-30T10:00:00Z"},
            {"ticker": "KXHIGHNY-26MAR31-T76", "yes_ask": 62, "_fetched_at": "2026-03-31T10:00:00Z"},
        ]
    )
    p = tmp_path / "snap.parquet"
    p.touch()

    monkeypatch.setattr("weatheredge.backtesting.market_price_builder.pd.read_parquet", lambda _path: frame)

    out = tmp_path / "market_prices_filtered.csv"
    report = build_market_prices_csv_from_kalshi_parquet(
        [tmp_path],
        out,
        start_date=date(2026, 3, 31),
    )

    assert report.output_rows == 2  # one date retained, two directions
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert {r["date"] for r in rows} == {"2026-03-31"}
