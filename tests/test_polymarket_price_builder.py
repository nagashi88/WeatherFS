"""Tests for poly_data -> backtest price CSV builder."""

from __future__ import annotations

import csv
from datetime import date, datetime, timezone

import pandas as pd

from weatheredge.backtesting.polymarket_price_builder import (
    _parse_ts_to_date,
    _side_to_yes_price,
    build_market_prices_csv_from_poly_data,
)


def test_parse_ts_seconds_and_ms():
    d = date(2026, 3, 30)
    ts = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc).timestamp()
    assert _parse_ts_to_date(ts) == d
    assert _parse_ts_to_date(ts * 1000.0) == d


def test_side_to_yes_price_with_explicit_yes_no_labels():
    assert _side_to_yes_price(0.62, "token1", "Yes", "No") == 0.62
    assert _side_to_yes_price(0.62, "token2", "Yes", "No") == 0.38
    assert _side_to_yes_price(0.62, "token1", "No", "Yes") == 0.38
    assert _side_to_yes_price(0.62, "token2", "No", "Yes") == 0.62


def test_build_market_prices_csv_from_poly_data(tmp_path):
    markets = pd.DataFrame(
        [
            {
                "id": "m1",
                "question": "Will the temperature in NYC on March 30 be 76°F or above?",
                "end_date_iso": "2024-03-30T23:00:00Z",
                "answer1": "Yes",
                "answer2": "No",
            },
            {
                "id": "m2",
                "question": "Non weather market",
                "answer1": "Yes",
                "answer2": "No",
            },
        ]
    )
    # Trade timestamps can be before contract resolution day; key date should stay contract date.
    ts1 = int(datetime(2024, 3, 29, 23, tzinfo=timezone.utc).timestamp())
    ts2 = int(datetime(2024, 3, 30, 1, tzinfo=timezone.utc).timestamp())
    trades = pd.DataFrame(
        [
            {"timestamp": ts1, "market_id": "m1", "nonusdc_side": "token1", "price": 0.55},
            {"timestamp": ts2, "market_id": "m1", "nonusdc_side": "token1", "price": 0.60},  # latest same day
            {"timestamp": ts1, "market_id": "m2", "nonusdc_side": "token1", "price": 0.70},  # ignored
        ]
    )

    markets_path = tmp_path / "markets.csv"
    trades_path = tmp_path / "trades.csv"
    out_path = tmp_path / "prices.csv"
    markets.to_csv(markets_path, index=False)
    trades.to_csv(trades_path, index=False)

    report = build_market_prices_csv_from_poly_data(markets_path, trades_path, out_path)
    assert report.output_rows == 1

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["station"] == "KNYC"
    assert row["date"] == "2024-03-30"
    assert row["threshold"] == "76.00"
    assert row["direction"] == "above"
    assert float(row["yes_price"]) == 0.60


def test_build_market_prices_tolerates_non_numeric_timestamp(tmp_path):
    markets = pd.DataFrame(
        [
            {
                "id": "m1",
                "question": "Will the temperature in NYC on March 30 be 76°F or above?",
                "end_date_iso": "2024-03-30T23:00:00Z",
                "answer1": "Yes",
                "answer2": "No",
            },
        ]
    )
    trades = pd.DataFrame(
        [
            {"timestamp": "not-a-timestamp", "market_id": "m1", "nonusdc_side": "token1", "price": 0.40},
            {"timestamp": "", "market_id": "m1", "nonusdc_side": "token1", "price": 0.45},
        ]
    )

    markets_path = tmp_path / "markets.csv"
    trades_path = tmp_path / "trades.csv"
    out_path = tmp_path / "prices.csv"
    markets.to_csv(markets_path, index=False)
    trades.to_csv(trades_path, index=False)

    report = build_market_prices_csv_from_poly_data(markets_path, trades_path, out_path)
    assert report.output_rows == 1

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["date"] == "2024-03-30"
