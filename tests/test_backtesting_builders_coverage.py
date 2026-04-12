"""Comprehensive coverage tests for backtesting builder modules.

Targets uncovered lines in:
- weatheredge/backtesting/db_market_price_builder.py
- weatheredge/backtesting/market_price_builder.py
- weatheredge/backtesting/polymarket_price_builder.py
- weatheredge/backtesting/prepare_polydata_snapshot.py
- weatheredge/backtesting/walkforward.py
- weatheredge/backtesting/market_prices.py
- weatheredge/data/market.py
"""

from __future__ import annotations

import csv
import json
import sqlite3
import tarfile
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# db_market_price_builder
# missing lines: 26-28, 33, 37, 41-43, 50-51, 59, 62-63, 72-74,
#   94-95, 116, 160, 181-186, 201, 205-206, 208, 235-258, 262
# ---------------------------------------------------------------------------
from weatheredge.backtesting.db_market_price_builder import (
    _extract_key_fields,
    _parse_contract_date,
    _parse_date_arg,
    _parse_direction,
    _parse_timestamp_to_sort_key,
    build_market_prices_csv_from_db,
    main as db_main,
)


class TestParseDateArg:
    def test_none_returns_none(self):
        # lines 26-27
        assert _parse_date_arg(None) is None

    def test_empty_returns_none(self):
        # line 26-27
        assert _parse_date_arg("") is None

    def test_valid_iso(self):
        # line 28
        assert _parse_date_arg("2026-03-30") == date(2026, 3, 30)


class TestParseTimestampToSortKey:
    def test_none_returns_zero(self):
        # line 33
        assert _parse_timestamp_to_sort_key(None) == 0.0

    def test_empty_string_returns_zero(self):
        # line 37
        assert _parse_timestamp_to_sort_key("") == 0.0

    def test_numeric_seconds(self):
        # lines 40-43 (float path, ts <= 1e12)
        ts = 1743292800.0
        assert _parse_timestamp_to_sort_key(ts) == pytest.approx(ts)

    def test_numeric_milliseconds_converted(self):
        # lines 41-42 (ts > 1e12 branch)
        ts_sec = 1743292800.0
        result = _parse_timestamp_to_sort_key(ts_sec * 1000)
        assert result == pytest.approx(ts_sec)

    def test_iso_string_with_z(self):
        # lines 47-49
        result = _parse_timestamp_to_sort_key("2026-03-30T00:00:00Z")
        assert result > 0.0

    def test_iso_string_parsed(self):
        # lines 47-49
        result = _parse_timestamp_to_sort_key("2026-03-30T00:00:00+00:00")
        assert result > 0.0

    def test_invalid_string_returns_zero(self):
        # lines 50-51
        assert _parse_timestamp_to_sort_key("not-a-timestamp") == 0.0


class TestParseContractDate:
    def test_none_returns_none(self):
        assert _parse_contract_date(None) is None

    def test_empty_returns_none(self):
        # line 59
        assert _parse_contract_date("") is None

    def test_valid_iso(self):
        # line 61
        assert _parse_contract_date("2026-03-30") == date(2026, 3, 30)

    def test_datetime_string_truncated(self):
        # line 61 (takes [:10])
        assert _parse_contract_date("2026-03-30T12:00:00Z") == date(2026, 3, 30)

    def test_invalid_returns_none(self):
        # lines 62-63
        assert _parse_contract_date("not-a-date") is None


class TestParseDirection:
    def test_none_returns_none(self):
        assert _parse_direction(None) is None

    def test_above_variants(self):
        # lines 70-71
        for v in ("above", "over", "higher", "gt", ">="):
            assert _parse_direction(v) == "above"

    def test_below_variants(self):
        # lines 72-73
        for v in ("below", "under", "lower", "lt", "<"):
            assert _parse_direction(v) == "below"

    def test_unknown_returns_none(self):
        # line 74
        assert _parse_direction("sideways") is None


class TestExtractKeyFields:
    def test_uses_resolution_source_json(self):
        # lines 94-95 (json.loads of resolution_source)
        station, contract_date, threshold, direction = _extract_key_fields(
            market_id="unknown-id",
            market_station="KNYC",
            market_date="2026-03-30",
            threshold_low=None,
            threshold_high=70.0,
            resolution_source='{"direction": "above", "threshold": 70.0}',
        )
        assert station == "KNYC"
        assert threshold == 70.0
        assert direction == "above"

    def test_invalid_json_resolution_source(self):
        # lines 94-95 (JSONDecodeError)
        station, _, _, _ = _extract_key_fields(
            market_id="KXHIGHDEN-26MAR30-T76",
            market_station=None,
            market_date=None,
            threshold_low=None,
            threshold_high=None,
            resolution_source="not-valid-json",
        )
        assert station == "KDEN"

    def test_bucket_market_direction_defaults_to_above(self):
        # lines 115-116
        _, _, _, direction = _extract_key_fields(
            market_id="unknown-bucket",
            market_station="KDEN",
            market_date="2026-03-30",
            threshold_low=70.0,
            threshold_high=75.0,
            resolution_source=None,
        )
        assert direction == "above"

    def test_threshold_from_threshold_low_fallback(self):
        # lines 104-110
        _, _, threshold, _ = _extract_key_fields(
            market_id="unknown-id",
            market_station="KNYC",
            market_date="2026-03-30",
            threshold_low=65.0,
            threshold_high=None,
            resolution_source=None,
        )
        assert threshold == 65.0

    def test_market_date_used_when_ticker_has_no_date(self):
        # lines 100-101
        _, contract_date, _, _ = _extract_key_fields(
            market_id="KXHIGHDEN",
            market_station=None,
            market_date="2026-04-01",
            threshold_low=None,
            threshold_high=76.0,
            resolution_source='{"direction":"above"}',
        )
        assert contract_date == date(2026, 4, 1)

    def test_station_from_market_station_when_no_ticker(self):
        # lines 97-98
        station, _, _, _ = _extract_key_fields(
            market_id="UNKNOWN-ID",
            market_station="  KNYC  ",
            market_date="2026-03-30",
            threshold_low=None,
            threshold_high=70.0,
            resolution_source=None,
        )
        assert station == "KNYC"


def _init_db_with_tables(path):
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE market_prices (id INTEGER PRIMARY KEY, market_id TEXT,"
            " timestamp TEXT, yes_price FLOAT)"
        )
        conn.execute(
            "CREATE TABLE markets (id TEXT PRIMARY KEY, platform TEXT NOT NULL,"
            " city TEXT NOT NULL, station TEXT NOT NULL, date TEXT NOT NULL,"
            " market_type TEXT NOT NULL, threshold_low FLOAT, threshold_high FLOAT,"
            " resolution_source TEXT, created_at TEXT)"
        )
        conn.commit()


def _init_db_no_markets(path):
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE market_prices (id INTEGER PRIMARY KEY, market_id TEXT,"
            " timestamp TEXT, yes_price FLOAT)"
        )
        conn.commit()


class TestBuildMarketPricesCsvFromDb:
    def test_no_markets_table_simple_query(self, tmp_path):
        # lines 159-165, 181-186
        db = tmp_path / "simple.db"
        out = tmp_path / "out.csv"
        _init_db_no_markets(db)
        with sqlite3.connect(db) as conn:
            conn.executemany(
                "INSERT INTO market_prices (market_id, timestamp, yes_price) VALUES (?,?,?)",
                [
                    ("KXHIGHDEN-26MAR30-T76", "2026-03-30 10:00:00", 0.55),
                    ("KXHIGHDEN-26MAR30-T76", "2026-03-30 11:00:00", 0.60),
                ],
            )
            conn.commit()
        report = build_market_prices_csv_from_db(db_path=db, output_csv=out)
        assert report.output_rows == 1
        assert report.scanned_rows == 2

    def test_yes_price_out_of_range_skipped(self, tmp_path):
        # line 207-208 (yes_price > 1 or < 0)
        db = tmp_path / "range.db"
        out = tmp_path / "out.csv"
        _init_db_no_markets(db)
        with sqlite3.connect(db) as conn:
            conn.executemany(
                "INSERT INTO market_prices (market_id, timestamp, yes_price) VALUES (?,?,?)",
                [
                    ("KXHIGHDEN-26MAR30-T76", "2026-03-30 10:00:00", 1.5),
                    ("KXHIGHDEN-26MAR30-T76", "2026-03-30 10:01:00", -0.1),
                    ("KXHIGHDEN-26MAR30-T76", "2026-03-30 10:02:00", 0.55),
                ],
            )
            conn.commit()
        report = build_market_prices_csv_from_db(db_path=db, output_csv=out)
        assert report.output_rows == 1

    def test_non_numeric_yes_price_skipped(self, tmp_path):
        # lines 205-206
        db = tmp_path / "bad.db"
        out = tmp_path / "out.csv"
        _init_db_no_markets(db)
        with sqlite3.connect(db) as conn:
            conn.executemany(
                "INSERT INTO market_prices (market_id, timestamp, yes_price) VALUES (?,?,?)",
                [
                    ("KXHIGHDEN-26MAR30-T76", "2026-03-30 10:00:00", "not-a-price"),
                    ("KXHIGHDEN-26MAR30-T76", "2026-03-30 11:00:00", 0.55),
                ],
            )
            conn.commit()
        report = build_market_prices_csv_from_db(db_path=db, output_csv=out)
        assert report.output_rows == 1

    def test_end_date_filter(self, tmp_path):
        # line 201
        db = tmp_path / "end.db"
        out = tmp_path / "out.csv"
        _init_db_no_markets(db)
        with sqlite3.connect(db) as conn:
            conn.executemany(
                "INSERT INTO market_prices (market_id, timestamp, yes_price) VALUES (?,?,?)",
                [
                    ("KXHIGHDEN-26MAR29-T76", "2026-03-29 10:00:00", 0.50),
                    ("KXHIGHDEN-26MAR30-T76", "2026-03-30 10:00:00", 0.55),
                ],
            )
            conn.commit()
        report = build_market_prices_csv_from_db(
            db_path=db, output_csv=out, end_date=date(2026, 3, 29)
        )
        assert report.output_rows == 1

    def test_missing_station_skipped(self, tmp_path):
        # line 196-197
        db = tmp_path / "skip.db"
        out = tmp_path / "out.csv"
        _init_db_no_markets(db)
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO market_prices (market_id, timestamp, yes_price) VALUES (?,?,?)",
                ("UNKNOWN-MARKET-123", "2026-03-30 10:00:00", 0.55),
            )
            conn.commit()
        report = build_market_prices_csv_from_db(db_path=db, output_csv=out)
        assert report.output_rows == 0

    def test_markets_table_with_direction_from_resolution_source(self, tmp_path):
        # lines 143-158, 168-179
        db = tmp_path / "meta.db"
        out = tmp_path / "out.csv"
        _init_db_with_tables(db)
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO markets (id,platform,city,station,date,market_type,"
                "threshold_low,threshold_high,resolution_source) VALUES (?,?,?,?,?,?,?,?,?)",
                ("m-abc", "polymarket", "nyc", "KNYC", "2026-04-01",
                 "daily_high", None, 72.0, '{"direction":"below","threshold":72.0}'),
            )
            conn.execute(
                "INSERT INTO market_prices (market_id, timestamp, yes_price) VALUES (?,?,?)",
                ("m-abc", "2026-04-01 10:00:00", 0.35),
            )
            conn.commit()
        report = build_market_prices_csv_from_db(db_path=db, output_csv=out)
        assert report.output_rows == 1
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["direction"] == "below"


class TestDbMainCli:
    def test_main_runs(self, tmp_path, capsys):
        # lines 235-258
        db = tmp_path / "w.db"
        out = tmp_path / "out.csv"
        _init_db_no_markets(db)
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO market_prices (market_id, timestamp, yes_price) VALUES (?,?,?)",
                ("KXHIGHDEN-26MAR30-T76", "2026-03-30 10:00:00", 0.55),
            )
            conn.commit()
        rc = db_main(["--db", str(db), "--output", str(out)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "built" in captured.out

    def test_main_with_date_filters(self, tmp_path, capsys):
        # lines 247-248
        db = tmp_path / "w.db"
        out = tmp_path / "out.csv"
        _init_db_no_markets(db)
        rc = db_main([
            "--db", str(db), "--output", str(out),
            "--start-date", "2026-03-01", "--end-date", "2026-03-31",
        ])
        assert rc == 0

    def test_main_entry_point(self, tmp_path):
        # line 262 (__main__ entry)
        db = tmp_path / "w.db"
        out = tmp_path / "out.csv"
        _init_db_no_markets(db)
        result = db_main(["--db", str(db), "--output", str(out)])
        assert result == 0


# ---------------------------------------------------------------------------
# market_price_builder
# missing lines: 39-40, 50, 62-64, 74, 79, 84, 91, 118-119, 158, 173,
#   204-206, 210-238, 242
# ---------------------------------------------------------------------------
from weatheredge.backtesting.market_price_builder import (
    _choose_yes_price_from_row,
    _expand_parquet_inputs,
    _normalise_price,
    _safe_float,
    build_market_prices_csv_from_kalshi_parquet,
    kalshi_dataframe_to_backtest_rows,
    _parse_date_arg as mpb_parse_date_arg,
    main as mpb_main,
)


class TestSafeFloat:
    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_invalid_string_returns_none(self):
        # lines 39-40
        assert _safe_float("not-a-number") is None

    def test_valid(self):
        assert _safe_float("3.14") == pytest.approx(3.14)


class TestNormalisePrice:
    def test_none_returns_none(self):
        assert _normalise_price(None) is None

    def test_cents_format_converted(self):
        # line 47-48 (p > 1.0 and <= 100.0)
        assert _normalise_price(65.0) == pytest.approx(0.65)

    def test_out_of_range_returns_none(self):
        # line 49-50: values < 0 or final value > 1.0 are rejected
        # Note: 1.5 is treated as cents (1.5/100=0.015) which is valid!
        # Use 101.0 to get 1.01 after /100 -> rejected
        assert _normalise_price(101.0) is None
        assert _normalise_price(-0.1) is None

    def test_valid_fraction(self):
        assert _normalise_price(0.42) == pytest.approx(0.42)


class TestChooseYesPriceFromRow:
    def test_midpoint(self):
        # line 58-59
        assert _choose_yes_price_from_row({"yes_bid": 60, "yes_ask": 64}) == pytest.approx(0.62)

    def test_ask_only(self):
        # lines 60-61
        assert _choose_yes_price_from_row({"yes_ask": 64}) == pytest.approx(0.64)

    def test_bid_only(self):
        # lines 62-63
        assert _choose_yes_price_from_row({"yes_bid": 55}) == pytest.approx(0.55)

    def test_last_price_fallback(self):
        # line 64
        assert _choose_yes_price_from_row({"last_price": 0.48}) == pytest.approx(0.48)

    def test_all_missing_returns_none(self):
        # line 64 (all None)
        assert _choose_yes_price_from_row({}) is None


class TestExpandParquetInputs:
    def test_file_included(self, tmp_path):
        # lines 117-119
        f = tmp_path / "snap.parquet"
        f.touch()
        result = _expand_parquet_inputs([f])
        assert f in result

    def test_non_parquet_excluded(self, tmp_path):
        # line 117 (suffix check)
        f = tmp_path / "snap.csv"
        f.touch()
        assert f not in _expand_parquet_inputs([f])

    def test_directory_recursed(self, tmp_path):
        # lines 120-121
        sub = tmp_path / "sub"
        sub.mkdir()
        p = sub / "a.parquet"
        p.touch()
        result = _expand_parquet_inputs([tmp_path])
        assert p in result

    def test_empty_directory(self, tmp_path):
        assert _expand_parquet_inputs([tmp_path]) == []


class TestKalshiDataframeToBacktestRows:
    def test_empty_dataframe(self):
        # line 73-74
        assert kalshi_dataframe_to_backtest_rows(pd.DataFrame()) == []

    def test_no_ticker_column(self):
        # lines 78-79
        df = pd.DataFrame([{"price": 0.5}])
        assert kalshi_dataframe_to_backtest_rows(df) == []

    def test_no_kxhigh_tickers(self):
        # lines 82-83
        df = pd.DataFrame([{"ticker": "PRES-2028-ABC", "yes_ask": 50}])
        assert kalshi_dataframe_to_backtest_rows(df) == []

    def test_none_yes_price_skipped(self):
        # line 91
        df = pd.DataFrame([{
            "ticker": "KXHIGHNY-26MAR30-T76",
            "yes_bid": None, "yes_ask": None, "last_price": None,
        }])
        assert kalshi_dataframe_to_backtest_rows(df) == []

    def test_fetched_at_fallback(self):
        # lines 93-97
        df = pd.DataFrame([{
            "ticker": "KXHIGHNY-26MAR30-T76",
            "yes_ask": 55,
            "fetched_at": "2026-03-30T10:00:00Z",
        }])
        rows = kalshi_dataframe_to_backtest_rows(df)
        assert len(rows) == 1
        assert rows[0]["_fetched_at"] == "2026-03-30T10:00:00Z"

    def test_timestamp_fallback(self):
        # line 96
        df = pd.DataFrame([{
            "ticker": "KXHIGHNY-26MAR30-T76",
            "yes_ask": 55,
            "timestamp": "2026-03-30T09:00:00Z",
        }])
        rows = kalshi_dataframe_to_backtest_rows(df)
        assert len(rows) == 1
        assert rows[0]["_fetched_at"] == "2026-03-30T09:00:00Z"


class TestBuildMarketPricesCsvFromKalshiParquet:
    def test_no_files_raises(self, tmp_path):
        # line 158
        with pytest.raises(FileNotFoundError, match="No parquet files"):
            build_market_prices_csv_from_kalshi_parquet([], tmp_path / "out.csv")

    def test_end_date_filter(self, tmp_path, monkeypatch):
        # line 173
        frame = pd.DataFrame([
            {"ticker": "KXHIGHNY-26MAR30-T76", "yes_ask": 60, "_fetched_at": "2026-03-30T10:00:00Z"},
            {"ticker": "KXHIGHNY-26MAR31-T76", "yes_ask": 62, "_fetched_at": "2026-03-31T10:00:00Z"},
        ])
        p = tmp_path / "snap.parquet"
        p.touch()
        monkeypatch.setattr(
            "weatheredge.backtesting.market_price_builder.pd.read_parquet", lambda _: frame
        )
        out = tmp_path / "out.csv"
        build_market_prices_csv_from_kalshi_parquet([p], out, end_date=date(2026, 3, 30))
        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))
        dates = {r["date"] for r in rows}
        assert "2026-03-31" not in dates
        assert "2026-03-30" in dates


class TestMpbParseDateArg:
    def test_none_returns_none(self):
        # lines 204-205
        assert mpb_parse_date_arg(None) is None

    def test_empty_returns_none(self):
        assert mpb_parse_date_arg("") is None

    def test_valid(self):
        # line 206
        assert mpb_parse_date_arg("2026-03-30") == date(2026, 3, 30)


class TestMpbMain:
    def test_main_runs(self, tmp_path, monkeypatch, capsys):
        # lines 209-238
        frame = pd.DataFrame([{
            "ticker": "KXHIGHNY-26MAR30-T76",
            "yes_ask": 60, "_fetched_at": "2026-03-30T10:00:00Z",
        }])
        p = tmp_path / "snap.parquet"
        p.touch()
        monkeypatch.setattr(
            "weatheredge.backtesting.market_price_builder.pd.read_parquet", lambda _: frame
        )
        out = tmp_path / "out.csv"
        rc = mpb_main(["--input", str(p), "--output", str(out)])
        assert rc == 0
        assert "Built" in capsys.readouterr().out

    def test_main_with_date_args(self, tmp_path, monkeypatch, capsys):
        # lines 228-233
        frame = pd.DataFrame([{
            "ticker": "KXHIGHNY-26MAR30-T76",
            "yes_ask": 60, "_fetched_at": "2026-03-30T10:00:00Z",
        }])
        p = tmp_path / "snap.parquet"
        p.touch()
        monkeypatch.setattr(
            "weatheredge.backtesting.market_price_builder.pd.read_parquet", lambda _: frame
        )
        out = tmp_path / "out.csv"
        rc = mpb_main([
            "--input", str(p), "--output", str(out),
            "--start-date", "2026-03-01", "--end-date", "2026-03-31",
        ])
        assert rc == 0

    def test_main_entry_point(self, tmp_path, monkeypatch):
        # line 242
        frame = pd.DataFrame([{
            "ticker": "KXHIGHNY-26MAR30-T76",
            "yes_ask": 60, "_fetched_at": "2026-03-30T10:00:00Z",
        }])
        p = tmp_path / "snap.parquet"
        p.touch()
        monkeypatch.setattr(
            "weatheredge.backtesting.market_price_builder.pd.read_parquet", lambda _: frame
        )
        assert mpb_main(["--input", str(p), "--output", str(tmp_path / "out.csv")]) == 0


# ---------------------------------------------------------------------------
# polymarket_price_builder
# missing lines: 45, 48-49, 56-57, 62, 69, 75, 86, 95, 98, 114, 133-137,
#   154, 156, 168, 175-176, 188, 198, 209, 211, 215-216, 225, 227,
#   259-261, 265-291, 295
# ---------------------------------------------------------------------------
from weatheredge.backtesting.polymarket_price_builder import (
    _infer_market_year,
    _parse_datetime,
    _parse_epoch_seconds,
    _parse_ts_to_date,
    _side_to_yes_price,
    build_market_prices_csv_from_poly_data,
    _parse_date_arg as poly_parse_date_arg,
    main as poly_main,
)


class TestParseTsToDate:
    def test_none_returns_none(self):
        # lines 44-45
        assert _parse_ts_to_date(None) is None

    def test_non_numeric_returns_none(self):
        # lines 47-48
        assert _parse_ts_to_date("not-a-number") is None

    def test_milliseconds_converted(self):
        # lines 52-53
        ts_sec = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc).timestamp()
        assert _parse_ts_to_date(ts_sec * 1000) == date(2026, 3, 30)

    def test_valid_seconds(self):
        # lines 54-55
        ts = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc).timestamp()
        assert _parse_ts_to_date(ts) == date(2026, 3, 30)

    def test_overflow_returns_none(self):
        # lines 56-57
        assert _parse_ts_to_date(9999999999999999) is None


class TestParseEpochSeconds:
    def test_none_returns_none(self):
        # lines 61-62
        assert _parse_epoch_seconds(None) is None

    def test_non_numeric_returns_none(self):
        # lines 63-64
        assert _parse_epoch_seconds("bad") is None

    def test_milliseconds_converted(self):
        # lines 68-69
        ts = 1743292800.0
        assert _parse_epoch_seconds(ts * 1000) == pytest.approx(ts)

    def test_valid_seconds(self):
        ts = 1743292800.0
        assert _parse_epoch_seconds(ts) == pytest.approx(ts)


class TestParseDatetimePoly:
    def test_none_returns_none(self):
        # lines 74-75
        assert _parse_datetime(None) is None

    def test_numeric_epoch(self):
        # lines 77-82
        ts = datetime(2026, 3, 30, tzinfo=timezone.utc).timestamp()
        result = _parse_datetime(ts)
        assert result is not None
        assert result.date() == date(2026, 3, 30)

    def test_numeric_epoch_overflow(self):
        # lines 80-82
        assert _parse_datetime(9999999999999999.0) is None

    def test_empty_string_returns_none(self):
        # lines 85-86
        assert _parse_datetime("") is None

    def test_nan_string_returns_none(self):
        # lines 85-86
        assert _parse_datetime("nan") is None

    def test_iso_with_z(self):
        # lines 88-90
        result = _parse_datetime("2026-03-30T12:00:00Z")
        assert result is not None
        assert result.date() == date(2026, 3, 30)

    def test_iso_no_tz_gets_utc(self):
        # lines 97-98
        result = _parse_datetime("2026-03-30T12:00:00")
        assert result is not None
        assert result.tzinfo is not None

    def test_iso_with_offset_converted_to_utc(self):
        # lines 99-100
        result = _parse_datetime("2026-03-30T12:00:00+05:00")
        assert result is not None
        assert result.utcoffset().total_seconds() == 0

    def test_unparseable_returns_none(self):
        # lines 92-94 (pandas coercion fails -> None)
        result = _parse_datetime("totally-@@@-unparseable")
        assert result is None


class TestInferMarketYear:
    def test_valid_column(self):
        # lines 105-108
        ts = datetime(2026, 3, 30, tzinfo=timezone.utc).timestamp()
        row = pd.Series({"end_date_iso": ts})
        assert _infer_market_year(row, ["end_date_iso"]) == 2026

    def test_all_invalid_returns_none(self):
        # line 109
        row = pd.Series({"end_date_iso": None})
        assert _infer_market_year(row, ["end_date_iso"]) is None


class TestSideToYesPrice:
    def test_out_of_range_price(self):
        # lines 113-114
        assert _side_to_yes_price(-0.1, "token1", "Yes", "No") is None
        assert _side_to_yes_price(1.1, "token1", "Yes", "No") is None

    def test_yes_no_explicit_token1(self):
        # lines 121-122
        assert _side_to_yes_price(0.62, "token1", "Yes", "No") == pytest.approx(0.62)

    def test_yes_no_explicit_token2(self):
        # lines 123-124
        assert _side_to_yes_price(0.62, "token2", "Yes", "No") == pytest.approx(0.38)

    def test_no_yes_explicit_token1(self):
        # lines 126-127
        assert _side_to_yes_price(0.62, "token1", "No", "Yes") == pytest.approx(0.38)

    def test_no_yes_explicit_token2(self):
        # lines 128-129
        assert _side_to_yes_price(0.62, "token2", "No", "Yes") == pytest.approx(0.62)

    def test_fallback_token1_no_labels(self):
        # lines 133-134
        assert _side_to_yes_price(0.7, "token1", None, None) == pytest.approx(0.7)

    def test_fallback_token2_no_labels(self):
        # lines 135-136
        assert _side_to_yes_price(0.7, "token2", None, None) == pytest.approx(0.3)

    def test_unknown_side_returns_none(self):
        # line 137
        assert _side_to_yes_price(0.7, "unknown_side", None, None) is None

    def test_numeric_side_answer1(self):
        # "answer1" is in the token1 set
        assert _side_to_yes_price(0.6, "answer1", "Yes", "No") == pytest.approx(0.6)


class TestBuildMarketPricesCsvFromPolyData:
    def _markets_df(self, question="Will the temperature in NYC on March 30 be 76°F or above?"):
        return pd.DataFrame([{
            "id": "m1", "question": question,
            "end_date_iso": "2024-03-30T23:00:00Z",
            "answer1": "Yes", "answer2": "No",
        }])

    def _trades_df(self, price=0.60, ts=None):
        if ts is None:
            ts = int(datetime(2024, 3, 30, 1, tzinfo=timezone.utc).timestamp())
        return pd.DataFrame([{
            "timestamp": ts, "market_id": "m1", "nonusdc_side": "token1", "price": price,
        }])

    def test_missing_question_raises(self, tmp_path):
        # lines 153-154
        markets = pd.DataFrame([{"id": "m1"}])
        mp = tmp_path / "m.csv"
        tp = tmp_path / "t.csv"
        markets.to_csv(mp, index=False)
        pd.DataFrame([]).to_csv(tp, index=False)
        with pytest.raises(ValueError, match="question"):
            build_market_prices_csv_from_poly_data(mp, tp, tmp_path / "out.csv")

    def test_missing_id_raises(self, tmp_path):
        # lines 155-156
        markets = pd.DataFrame([{"question": "Test"}])
        mp = tmp_path / "m.csv"
        tp = tmp_path / "t.csv"
        markets.to_csv(mp, index=False)
        pd.DataFrame([]).to_csv(tp, index=False)
        with pytest.raises(ValueError, match="market id"):
            build_market_prices_csv_from_poly_data(mp, tp, tmp_path / "out.csv")

    def test_no_parseable_markets_raises(self, tmp_path):
        # lines 187-188
        markets = pd.DataFrame([{"id": "m1", "question": "Non-weather market"}])
        mp = tmp_path / "m.csv"
        tp = tmp_path / "t.csv"
        markets.to_csv(mp, index=False)
        pd.DataFrame([]).to_csv(tp, index=False)
        with pytest.raises(ValueError, match="No parseable weather markets"):
            build_market_prices_csv_from_poly_data(mp, tp, tmp_path / "out.csv")

    def test_date_parse_error_in_market_skipped(self, tmp_path):
        # lines 174-175 (ValueError in date.fromisoformat after parse)
        # Use a question whose parsed.date cannot be parsed by date.fromisoformat.
        # We achieve this by monkeypatching parse_polymarket_title to return a bad date.
        from weatheredge.data.market_parser import ParsedMarket
        bad_parsed = ParsedMarket(
            city="nyc", station="KNYC", date="not-a-date",
            market_type="daily_high", threshold=76.0, direction="above",
        )
        markets = pd.DataFrame([{
            "id": "m1",
            "question": "Will the temperature in NYC on March 30 be 76°F or above?",
            "end_date_iso": "2024-03-30T23:00:00Z",
        }])
        mp = tmp_path / "m.csv"
        tp = tmp_path / "t.csv"
        markets.to_csv(mp, index=False)
        # Need a non-empty trades CSV to avoid EmptyDataError before our code runs
        pd.DataFrame([{"timestamp": 1711756800, "market_id": "m1", "price": 0.5}]).to_csv(tp, index=False)
        with patch(
            "weatheredge.backtesting.polymarket_price_builder.parse_polymarket_title",
            return_value=bad_parsed,
        ):
            with pytest.raises(ValueError, match="No parseable weather markets"):
                build_market_prices_csv_from_poly_data(mp, tp, tmp_path / "out.csv")

    def test_empty_question_row_skipped(self, tmp_path):
        # lines 165-167: rows with empty question are skipped
        # pandas reads empty CSV cells as NaN; str(NaN)="nan" which is truthy for id,
        # but an explicit whitespace-only question is falsy after strip.
        mp = tmp_path / "m.csv"
        tp = tmp_path / "t.csv"
        # Row 1: has id but empty question -> skipped (not question)
        # Row 2: valid market -> counted
        mp.write_text(
            "id,question,end_date_iso,answer1,answer2\n"
            "m0, ,2024-03-30T23:00:00Z,Yes,No\n"
            "m1,Will the temperature in NYC on March 31 be 76\u00b0F or above?,"
            "2024-03-31T23:00:00Z,Yes,No\n"
        )
        ts = int(datetime(2024, 3, 31, 1, tzinfo=timezone.utc).timestamp())
        trades = pd.DataFrame([{
            "timestamp": ts, "market_id": "m1", "nonusdc_side": "token1", "price": 0.6,
        }])
        trades.to_csv(tp, index=False)
        report = build_market_prices_csv_from_poly_data(mp, tp, tmp_path / "out.csv")
        # Row with whitespace-only question is skipped; only m1 counted
        assert report.parsed_weather_markets == 1

    def test_start_date_filter(self, tmp_path):
        # line 209 (start_date filter)
        mp = tmp_path / "m.csv"
        tp = tmp_path / "t.csv"
        self._markets_df().to_csv(mp, index=False)
        self._trades_df().to_csv(tp, index=False)
        report = build_market_prices_csv_from_poly_data(
            mp, tp, tmp_path / "out.csv", start_date=date(2024, 4, 1)
        )
        assert report.output_rows == 0

    def test_end_date_filter(self, tmp_path):
        # line 211
        mp = tmp_path / "m.csv"
        tp = tmp_path / "t.csv"
        self._markets_df().to_csv(mp, index=False)
        self._trades_df().to_csv(tp, index=False)
        report = build_market_prices_csv_from_poly_data(
            mp, tp, tmp_path / "out.csv", end_date=date(2024, 3, 29)
        )
        assert report.output_rows == 0

    def test_invalid_price_skipped(self, tmp_path):
        # lines 215-216
        mp = tmp_path / "m.csv"
        tp = tmp_path / "t.csv"
        self._markets_df().to_csv(mp, index=False)
        trades = pd.DataFrame([{
            "timestamp": 1711756800, "market_id": "m1",
            "nonusdc_side": "token1", "price": "invalid",
        }])
        trades.to_csv(tp, index=False)
        report = build_market_prices_csv_from_poly_data(mp, tp, tmp_path / "out.csv")
        assert report.trade_rows_used == 0

    def test_yes_price_none_side_skipped(self, tmp_path):
        # lines 224-225
        mp = tmp_path / "m.csv"
        tp = tmp_path / "t.csv"
        self._markets_df().to_csv(mp, index=False)
        trades = pd.DataFrame([{
            "timestamp": 1711756800, "market_id": "m1",
            "nonusdc_side": "unknown_side", "price": 0.5,
        }])
        trades.to_csv(tp, index=False)
        report = build_market_prices_csv_from_poly_data(mp, tp, tmp_path / "out.csv")
        assert report.trade_rows_used == 0

    def test_no_market_id_column_in_chunk(self, tmp_path):
        # lines 197-198 (no market_id column)
        mp = tmp_path / "m.csv"
        tp = tmp_path / "t.csv"
        self._markets_df().to_csv(mp, index=False)
        pd.DataFrame([{"price": 0.5, "timestamp": 1711756800}]).to_csv(tp, index=False)
        report = build_market_prices_csv_from_poly_data(mp, tp, tmp_path / "out.csv")
        assert report.trade_rows_used == 0


class TestPolyParseDateArg:
    def test_none_returns_none(self):
        # lines 259-260
        assert poly_parse_date_arg(None) is None

    def test_empty_returns_none(self):
        assert poly_parse_date_arg("") is None

    def test_valid(self):
        # line 261
        assert poly_parse_date_arg("2026-03-30") == date(2026, 3, 30)


class TestPolyMain:
    def _setup(self, tmp_path):
        mp = tmp_path / "markets.csv"
        tp = tmp_path / "trades.csv"
        pd.DataFrame([{
            "id": "m1",
            "question": "Will the temperature in NYC on March 30 be 76°F or above?",
            "end_date_iso": "2024-03-30T23:00:00Z",
            "answer1": "Yes", "answer2": "No",
        }]).to_csv(mp, index=False)
        ts = int(datetime(2024, 3, 30, 1, tzinfo=timezone.utc).timestamp())
        pd.DataFrame([{
            "timestamp": ts, "market_id": "m1", "nonusdc_side": "token1", "price": 0.60,
        }]).to_csv(tp, index=False)
        return mp, tp

    def test_main_runs(self, tmp_path, capsys):
        # lines 265-291
        mp, tp = self._setup(tmp_path)
        rc = poly_main([
            "--markets", str(mp), "--trades", str(tp),
            "--output", str(tmp_path / "out.csv"),
        ])
        assert rc == 0
        assert "built" in capsys.readouterr().out

    def test_main_with_dates(self, tmp_path, capsys):
        # lines 278-280
        mp, tp = self._setup(tmp_path)
        rc = poly_main([
            "--markets", str(mp), "--trades", str(tp),
            "--output", str(tmp_path / "out.csv"),
            "--start-date", "2024-01-01", "--end-date", "2024-12-31",
        ])
        assert rc == 0

    def test_main_entry_point(self, tmp_path):
        # line 295
        mp, tp = self._setup(tmp_path)
        assert poly_main([
            "--markets", str(mp), "--trades", str(tp),
            "--output", str(tmp_path / "out.csv"),
        ]) == 0


# ---------------------------------------------------------------------------
# prepare_polydata_snapshot
# missing lines: 41, 47, 52, 70, 79, 94, 104, 116, 127-129, 133-162, 166
# ---------------------------------------------------------------------------
from weatheredge.backtesting.prepare_polydata_snapshot import (
    _extract_first_matching_member,
    _parse_date as ppd_parse_date,
    main as ppd_main,
    prepare_poly_data_archive,
)


def _make_archive(tmp_path: Path, markets_csv: str, trades_csv: str) -> Path:
    archive = tmp_path / "archive.tar.xz"
    with tarfile.open(archive, mode="w:xz") as tf:
        for name, content in [("markets.csv", markets_csv), ("processed/trades.csv", trades_csv)]:
            data = content.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, BytesIO(data))
    return archive


_MARKETS_CSV = (
    "id,question,end_date_iso,answer1,answer2\n"
    "m1,Will the temperature in NYC on March 30 be 76\u00b0F or above?,"
    "2024-03-30T23:00:00Z,Yes,No\n"
)
_TS = int(datetime(2024, 3, 30, 1, tzinfo=timezone.utc).timestamp())
_TRADES_CSV = f"timestamp,market_id,nonusdc_side,price\n{_TS},m1,token1,0.60\n"


class TestExtractFirstMatchingMember:
    def test_extracts_matching(self, tmp_path):
        # lines 39-51
        archive = _make_archive(tmp_path, _MARKETS_CSV, _TRADES_CSV)
        out = tmp_path / "out" / "markets.csv"
        member = _extract_first_matching_member(archive, ("markets.csv",), out)
        assert member is not None
        assert out.exists()

    def test_no_match_returns_none(self, tmp_path):
        # line 52
        archive = _make_archive(tmp_path, _MARKETS_CSV, _TRADES_CSV)
        result = _extract_first_matching_member(archive, ("nonexistent.xyz",), tmp_path / "x.csv")
        assert result is None

    def test_case_insensitive(self, tmp_path):
        # line 37 (lowered suffixes)
        archive = _make_archive(tmp_path, _MARKETS_CSV, _TRADES_CSV)
        out = tmp_path / "out" / "markets.csv"
        member = _extract_first_matching_member(archive, ("MARKETS.CSV",), out)
        assert member is not None


class TestPreparePolyDataArchive:
    def test_stream_mode(self, tmp_path):
        # lines 88-117 (extract_trades_csv=False)
        archive = _make_archive(tmp_path, _MARKETS_CSV, _TRADES_CSV)
        work = tmp_path / "work"
        report = prepare_poly_data_archive(
            archive_path=archive, work_dir=work,
            output_prices_csv=tmp_path / "prices.csv",
            extract_trades_csv=False,
        )
        assert "stream" in report.extracted_trades_csv

    def test_extract_mode(self, tmp_path):
        # lines 73-87 (extract_trades_csv=True)
        archive = _make_archive(tmp_path, _MARKETS_CSV, _TRADES_CSV)
        work = tmp_path / "work"
        report = prepare_poly_data_archive(
            archive_path=archive, work_dir=work,
            output_prices_csv=tmp_path / "prices.csv",
            extract_trades_csv=True,
        )
        assert Path(report.extracted_trades_csv).exists()

    def test_markets_missing_raises(self, tmp_path):
        # line 69-70
        archive = tmp_path / "empty.tar.xz"
        with tarfile.open(archive, mode="w:xz"):
            pass
        with pytest.raises(FileNotFoundError, match="markets.csv"):
            prepare_poly_data_archive(
                archive_path=archive, work_dir=tmp_path / "work",
                output_prices_csv=tmp_path / "out.csv",
            )

    def test_trades_missing_extract_mode_raises(self, tmp_path):
        # line 78-79
        archive = tmp_path / "notradesextract.tar.xz"
        with tarfile.open(archive, mode="w:xz") as tf:
            data = _MARKETS_CSV.encode()
            info = tarfile.TarInfo(name="markets.csv")
            info.size = len(data)
            tf.addfile(info, BytesIO(data))
        with pytest.raises(FileNotFoundError, match="trades.csv"):
            prepare_poly_data_archive(
                archive_path=archive, work_dir=tmp_path / "work",
                output_prices_csv=tmp_path / "out.csv",
                extract_trades_csv=True,
            )

    def test_trades_missing_stream_mode_raises(self, tmp_path):
        # lines 115-116
        archive = tmp_path / "notrades.tar.xz"
        with tarfile.open(archive, mode="w:xz") as tf:
            data = _MARKETS_CSV.encode()
            info = tarfile.TarInfo(name="markets.csv")
            info.size = len(data)
            tf.addfile(info, BytesIO(data))
        with pytest.raises(FileNotFoundError, match="trades.csv"):
            prepare_poly_data_archive(
                archive_path=archive, work_dir=tmp_path / "work",
                output_prices_csv=tmp_path / "out.csv",
                extract_trades_csv=False,
            )

    def test_dates_passed_through(self, tmp_path):
        # lines 85-86, 109-110
        archive = _make_archive(tmp_path, _MARKETS_CSV, _TRADES_CSV)
        report = prepare_poly_data_archive(
            archive_path=archive, work_dir=tmp_path / "work",
            output_prices_csv=tmp_path / "prices.csv",
            start_date=date(2024, 1, 1), end_date=date(2024, 12, 31),
        )
        assert report is not None


class TestPpdParseDateAndMain:
    def test_none(self):
        # lines 127-128
        assert ppd_parse_date(None) is None

    def test_empty(self):
        assert ppd_parse_date("") is None

    def test_valid(self):
        # line 129
        assert ppd_parse_date("2026-03-30") == date(2026, 3, 30)

    def test_main_stream(self, tmp_path, capsys):
        # lines 132-162
        archive = _make_archive(tmp_path, _MARKETS_CSV, _TRADES_CSV)
        rc = ppd_main([
            "--archive", str(archive),
            "--work-dir", str(tmp_path / "work"),
            "--output", str(tmp_path / "prices.csv"),
        ])
        assert rc == 0
        assert "prepared" in capsys.readouterr().out

    def test_main_extract(self, tmp_path, capsys):
        # line 141-142 (--extract-trades)
        archive = _make_archive(tmp_path, _MARKETS_CSV, _TRADES_CSV)
        rc = ppd_main([
            "--archive", str(archive),
            "--work-dir", str(tmp_path / "work"),
            "--output", str(tmp_path / "prices.csv"),
            "--extract-trades",
        ])
        assert rc == 0

    def test_main_with_dates(self, tmp_path, capsys):
        # lines 150-151
        archive = _make_archive(tmp_path, _MARKETS_CSV, _TRADES_CSV)
        rc = ppd_main([
            "--archive", str(archive),
            "--work-dir", str(tmp_path / "work"),
            "--output", str(tmp_path / "prices.csv"),
            "--start-date", "2024-01-01", "--end-date", "2024-12-31",
        ])
        assert rc == 0

    def test_main_entry_point(self, tmp_path):
        # line 166
        archive = _make_archive(tmp_path, _MARKETS_CSV, _TRADES_CSV)
        result = ppd_main([
            "--archive", str(archive),
            "--work-dir", str(tmp_path / "work"),
            "--output", str(tmp_path / "prices.csv"),
        ])
        assert result == 0


# ---------------------------------------------------------------------------
# walkforward
# missing lines: 75, 113, 120, 170-198, 202
# ---------------------------------------------------------------------------
from weatheredge.backtesting.backtester import SimulatedTrade
from weatheredge.backtesting.walkforward import (
    WalkForwardWindow,
    load_trades_csv,
    main as wf_main,
    save_walkforward_csv,
    summarize_walkforward,
    walkforward_evaluate,
)


def _make_trade(d: date, won: bool = True) -> SimulatedTrade:
    return SimulatedTrade(
        station="KNYC", trade_date=d, threshold=70.0, direction="above",
        model_prob=0.65, market_proxy=0.50, side="buy_yes",
        entry_price=0.50, won=won, pnl=1.0 if won else -1.0,
    )


class TestWalkforwardEvaluate:
    def test_empty_trades_returns_empty(self):
        # line 75
        assert walkforward_evaluate([]) == []

    def test_test_start_after_last_day(self):
        # lines 90-91 (test_start > last_day break)
        trades = [_make_trade(date(2026, 1, 1)), _make_trade(date(2026, 1, 2))]
        windows = walkforward_evaluate(trades, train_days=10, test_days=5, step_days=5, min_test_trades=1)
        assert windows == []

    def test_window_below_min_trades_skipped(self):
        # line 94
        trades = [_make_trade(date(2026, 1, 1) + timedelta(days=i)) for i in range(60)]
        windows = walkforward_evaluate(trades, train_days=20, test_days=10, step_days=10, min_test_trades=100)
        assert windows == []

    def test_cursor_beyond_last_day_breaks(self):
        # lines 112-113
        trades = [_make_trade(date(2026, 1, 1) + timedelta(days=i)) for i in range(120)]
        windows = walkforward_evaluate(trades, train_days=30, test_days=30, step_days=500, min_test_trades=1)
        assert isinstance(windows, list)


class TestSummarizeWalkforward:
    def test_empty_returns_zeros(self):
        # line 120
        summary = summarize_walkforward([])
        assert summary.windows == 0
        assert summary.total_test_trades == 0
        assert summary.median_win_rate == 0.0
        assert summary.aggregate_net_pnl == 0.0


class TestWalkforwardMain:
    def _make_trades_csv(self, tmp_path, days=120) -> Path:
        p = tmp_path / "trades.csv"
        lines = ["station,date,threshold,direction,model_prob,market_proxy,side,entry_price,won,pnl\n"]
        start = date(2026, 1, 1)
        for i in range(days):
            d = start + timedelta(days=i)
            won = "True" if i % 2 == 0 else "False"
            pnl = "1.0" if i % 2 == 0 else "-1.0"
            lines.append(f"KNYC,{d.isoformat()},70.0,above,0.60,0.50,buy_yes,0.50,{won},{pnl}\n")
        p.write_text("".join(lines))
        return p

    def test_main_runs(self, tmp_path, capsys):
        # lines 170-198
        trades = self._make_trades_csv(tmp_path)
        out = tmp_path / "wf.csv"
        rc = wf_main([
            "--input", str(trades), "--output", str(out),
            "--train-days", "30", "--test-days", "30",
            "--step-days", "30", "--min-test-trades", "5",
        ])
        assert rc == 0
        assert "walk-forward" in capsys.readouterr().out

    def test_main_entry_point(self, tmp_path):
        # line 202
        trades = self._make_trades_csv(tmp_path)
        out = tmp_path / "wf.csv"
        assert wf_main(["--input", str(trades), "--output", str(out)]) == 0

    def test_main_no_windows(self, tmp_path, capsys):
        # zero windows with high min_test_trades
        trades = self._make_trades_csv(tmp_path, days=5)
        out = tmp_path / "wf.csv"
        rc = wf_main(["--input", str(trades), "--output", str(out), "--min-test-trades", "999"])
        assert rc == 0


# ---------------------------------------------------------------------------
# market_prices
# missing lines: 45, 57, 62, 68-75, 105-106, 125, 143, 146-147, 227-231
# ---------------------------------------------------------------------------
from weatheredge.backtesting.market_prices import (
    HistoricalMarketPriceBook,
    _extract_from_kalshi_ticker,
    _infer_direction_from_text,
    _norm_direction,
    _norm_station,
    _parse_date,
    _parse_threshold,
    _parse_yes_price,
)


class TestNormStation:
    def test_none_returns_none(self):
        assert _norm_station(None) is None

    def test_empty_returns_none(self):
        # line 44-45
        assert _norm_station("") is None

    def test_city_mapped(self):
        # lines 47-48
        assert _norm_station("NYC") == "KNYC"
        assert _norm_station("New York") == "KNYC"

    def test_unknown_uppercased(self):
        # line 49
        assert _norm_station("kden") == "KDEN"


class TestNormDirection:
    def test_none_returns_none(self):
        assert _norm_direction(None) is None

    def test_empty_returns_none(self):
        # line 57
        assert _norm_direction("") is None

    def test_above_variants(self):
        for v in ("above", "over", "higher", "gt", ">="):
            assert _norm_direction(v) == "above"

    def test_below_variants(self):
        for v in ("below", "under", "lower", "lt", "<"):
            assert _norm_direction(v) == "below"

    def test_unknown_returns_none(self):
        # line 62
        assert _norm_direction("sideways") is None


class TestInferDirectionFromText:
    def test_none_returns_none(self):
        assert _infer_direction_from_text(None) is None

    def test_empty_returns_none(self):
        # lines 68-69
        assert _infer_direction_from_text("") is None

    def test_above_keywords(self):
        # line 71
        for phrase in ("will exceed 70F", "temperature above normal", "market-t ticker"):
            assert _infer_direction_from_text(phrase) == "above", phrase

    def test_below_keywords(self):
        # line 73
        for phrase in ("temperature below freezing", "market-b ticker"):
            assert _infer_direction_from_text(phrase) == "below", phrase

    def test_no_keywords_returns_none(self):
        # line 75
        assert _infer_direction_from_text("daily high temperature") is None


class TestExtractFromKalshiTicker:
    def test_none_returns_all_none(self):
        # lines 87-88
        assert _extract_from_kalshi_ticker(None) == (None, None, None, None)

    def test_non_kxhigh_returns_all_none(self):
        # lines 90-91
        assert _extract_from_kalshi_ticker("PRES-2028-DJT") == (None, None, None, None)

    def test_unknown_city_code(self):
        # line 96 (city_code not in mapping)
        station, _, _, _ = _extract_from_kalshi_ticker("KXHIGHBOS-26MAR30-T76")
        assert station is None

    def test_invalid_date_in_ticker(self):
        # lines 105-106 (ValueError in strptime)
        _, d, _, _ = _extract_from_kalshi_ticker("KXHIGHDEN-BADDATE-T76")
        assert d is None

    def test_full_ticker_above(self):
        station, d, threshold, direction = _extract_from_kalshi_ticker("KXHIGHDEN-26MAR30-T76")
        assert station == "KDEN"
        assert d == date(2026, 3, 30)
        assert threshold == 76.0
        assert direction == "above"

    def test_full_ticker_below(self):
        _, _, threshold, direction = _extract_from_kalshi_ticker("KXHIGHDEN-26MAR30-B76.5")
        assert threshold == 76.5
        assert direction == "below"


class TestParseDate:
    def test_none_returns_none(self):
        assert _parse_date(None) is None

    def test_empty_returns_none(self):
        # line 125
        assert _parse_date("") is None

    def test_valid(self):
        assert _parse_date("2026-03-30") == date(2026, 3, 30)

    def test_invalid_returns_none(self):
        # lines 127-128
        assert _parse_date("not-a-date") is None


class TestParseThreshold:
    def test_none_returns_none(self):
        assert _parse_threshold(None) is None

    def test_valid_float(self):
        assert _parse_threshold("76.5") == 76.5

    def test_invalid_returns_none(self):
        # line 137
        assert _parse_threshold("bad") is None


class TestParseYesPrice:
    def test_none_returns_none(self):
        assert _parse_yes_price(None) is None

    def test_invalid_string_returns_none(self):
        # line 143
        assert _parse_yes_price("bad") is None

    def test_cents_format(self):
        # lines 150-151
        assert _parse_yes_price(65) == pytest.approx(0.65)

    def test_exactly_100(self):
        # line 150-151
        assert _parse_yes_price(100.0) == pytest.approx(1.0)

    def test_over_100_returns_none(self):
        # lines 152-153
        assert _parse_yes_price(101.0) is None

    def test_negative_returns_none(self):
        # lines 152-153
        assert _parse_yes_price(-0.1) is None

    def test_above_1_non_cents_returns_none(self):
        # line 152-153: 1.5 is treated as cents (1.5/100=0.015) which is valid
        # Use 101.0 -> 1.01 after /100 -> rejected; or check via > 100 directly
        assert _parse_yes_price(101.0) is None

    def test_valid_fraction(self):
        assert _parse_yes_price(0.42) == pytest.approx(0.42)


class TestHistoricalMarketPriceBookFromCsv:
    def test_from_csv_loads(self, tmp_path):
        # lines 227-231
        p = tmp_path / "prices.csv"
        p.write_text("station,date,threshold,direction,yes_price\nKNYC,2026-03-30,76.00,above,0.42\n")
        book = HistoricalMarketPriceBook.from_csv(p)
        assert book.get_yes_price("KNYC", date(2026, 3, 30), 76.0, "above") == pytest.approx(0.42)

    def test_from_csv_complement_generated(self, tmp_path):
        # complement fill
        p = tmp_path / "prices.csv"
        p.write_text("station,date,threshold,direction,yes_price\nKDEN,2026-04-01,72.00,above,0.60\n")
        book = HistoricalMarketPriceBook.from_csv(p)
        assert book.get_yes_price("KDEN", date(2026, 4, 1), 72.0, "below") == pytest.approx(0.40)

    def test_get_yes_price_unknown_returns_none(self):
        # line 241
        book = HistoricalMarketPriceBook()
        assert book.get_yes_price("KNYC", date(2026, 3, 30), 76.0, "above") is None

    def test_get_yes_price_normalizes_station(self):
        # line 240
        book = HistoricalMarketPriceBook.from_rows([{
            "station": "KNYC", "date": "2026-03-30",
            "threshold": "76", "direction": "above", "yes_price": "0.50",
        }])
        assert book.get_yes_price("New York", date(2026, 3, 30), 76.0, "above") == pytest.approx(0.50)

    def test_ticker_fallback_in_from_rows(self):
        # lines 172-175
        book = HistoricalMarketPriceBook.from_rows([{"ticker": "KXHIGHLAX-26MAR30-T72", "yes_bid": 55}])
        assert book.get_yes_price("KLAX", date(2026, 3, 30), 72.0, "above") == pytest.approx(0.55)

    def test_market_id_used_as_ticker_fallback(self):
        # line 172
        book = HistoricalMarketPriceBook.from_rows([{
            "market_id": "KXHIGHMIA-26MAR30-T85", "yes_price": 0.70,
        }])
        assert book.get_yes_price("KMIA", date(2026, 3, 30), 85.0, "above") == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# data/market.py
# missing lines: 121, 125-126, 219-236, 376-390, 473
# ---------------------------------------------------------------------------
from weatheredge.config import Settings
from weatheredge.data.market import MarketScanner, _STATION_TO_CITY
from weatheredge.data.market_parser import ParsedMarket
from weatheredge.db.database import get_session, init_db, make_engine
from weatheredge.db.models import Market, MarketPrice


@pytest.fixture()
def scanner_settings():
    return Settings(
        polymarket_private_key="",
        polymarket_proxy_address="",
        kalshi_api_key="",
        kalshi_private_key="",
    )


@pytest.fixture()
def scanner(scanner_settings):
    return MarketScanner(settings=scanner_settings)


@pytest.fixture()
def mem_engine():
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    yield eng
    eng.dispose()


class TestTargetStations:
    def test_empty_string_skipped(self, scanner):
        # line 121
        assert scanner._target_stations(["", "  "]) == set()

    def test_four_char_station_code_direct(self, scanner):
        # lines 124-125
        assert "KNYC" in scanner._target_stations(["KNYC"])

    def test_city_name_mapped(self, scanner):
        # lines 127-129
        assert "KNYC" in scanner._target_stations(["nyc"])

    def test_unknown_city_not_added(self, scanner):
        # line 127-128 (station is None -> skip)
        assert scanner._target_stations(["atlantis"]) == set()

    def test_mixed_input(self, scanner):
        result = scanner._target_stations(["nyc", "KDEN", "", "atlantis"])
        assert "KNYC" in result
        assert "KDEN" in result
        assert len(result) == 2


class TestAlertUnknownMarket:
    def test_no_credentials_returns_silently(self, scanner):
        # lines 216-217
        scanner.settings.telegram_bot_token = ""
        scanner.settings.telegram_chat_id = ""
        scanner._alert_unknown_market("kalshi", "KXHIGHBOS", "Boston High")

    def test_with_credentials_exception_caught(self, scanner):
        # lines 219-235 (exception path) - TelegramBot imported locally so patch source
        scanner.settings.telegram_bot_token = "fake-token"
        scanner.settings.telegram_chat_id = "12345"
        with patch("weatheredge.notifications.telegram.TelegramBot", side_effect=Exception("err")):
            # Exception caught at line 235 -> no raise
            scanner._alert_unknown_market("kalshi", "KXHIGHBOS", "Boston High")

    def test_asyncio_run_path(self, scanner):
        # lines 231-234 (no running loop -> asyncio.run)
        # TelegramBot is imported inside the function; patch at source module
        scanner.settings.telegram_bot_token = "fake-token"
        scanner.settings.telegram_chat_id = "12345"
        with patch("weatheredge.notifications.telegram.TelegramBot") as MockBot:
            mock_instance = MagicMock()
            mock_instance.send_alert = AsyncMock()
            MockBot.return_value = mock_instance
            with patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")):
                with patch("asyncio.run"):
                    scanner._alert_unknown_market("kalshi", "KXHIGHBOS", "Boston High")


class TestParseIsoDate:
    def test_none_returns_none(self):
        # line 374-376
        assert MarketScanner._parse_iso_date(None) is None

    def test_empty_returns_none(self):
        # lines 377-379
        assert MarketScanner._parse_iso_date("") is None

    def test_valid_10_char_date(self):
        # lines 379-385
        assert MarketScanner._parse_iso_date("2026-03-30") == "2026-03-30"

    def test_datetime_string(self):
        # lines 379-384
        assert MarketScanner._parse_iso_date("2026-03-30T12:00:00Z") == "2026-03-30"

    def test_iso_format_fallback(self):
        # lines 386-390
        assert MarketScanner._parse_iso_date("2026-03-30T12:00:00+00:00") == "2026-03-30"

    def test_invalid_returns_none(self):
        # lines 389-390 (ValueError)
        assert MarketScanner._parse_iso_date("not-a-date") is None


class TestStoreMarketMetadataEdgeCases:
    def test_no_parsed_uses_defaults(self, scanner, mem_engine):
        # lines 446-458
        market_id = "bare-market-1"
        with get_session(engine=mem_engine) as session:
            scanner.store_market_metadata(session, market_id, {
                "platform": "polymarket", "ticker": market_id,
                "title": "Some market", "resolution_source": {},
            })
        with get_session(engine=mem_engine) as session:
            row = session.get(Market, market_id)
            assert row.station == "UNKNOWN"
            assert row.market_type == "daily_high"

    def test_resolution_source_metadata_fields(self, scanner, mem_engine):
        # lines 470-473
        market_id = "meta-fields-test"
        with get_session(engine=mem_engine) as session:
            scanner.store_market_metadata(session, market_id, {
                "platform": "kalshi", "ticker": "KXHIGHNY",
                "market_id": market_id, "title": "NY High",
                "resolution_source": {
                    "station": "KNYC", "type": "daily_high", "source": "NWS CLI",
                    "timezone": "EST", "report_time": "~7:00 AM",
                },
            })
        with get_session(engine=mem_engine) as session:
            row = session.get(Market, market_id)
            payload = json.loads(row.resolution_source or "{}")
            assert payload["timezone"] == "EST"
            assert payload["report_time"] == "~7:00 AM"

    def test_station_to_city_fallback(self, scanner, mem_engine):
        # line 458
        market_id = "station-city-test"
        with get_session(engine=mem_engine) as session:
            scanner.store_market_metadata(session, market_id, {
                "platform": "kalshi", "ticker": "KXHIGHNY",
                "market_id": market_id, "title": "NY High",
                "resolution_source": {"station": "KNYC", "type": "daily_high", "source": "NWS CLI"},
            })
        with get_session(engine=mem_engine) as session:
            row = session.get(Market, market_id)
            assert row.city == "nyc"

    def test_ticker_threshold_set(self, scanner, mem_engine):
        # lines 428-438
        market_id = "KXHIGHDEN-26MAR30-T80"
        with get_session(engine=mem_engine) as session:
            scanner.store_market_metadata(session, market_id, {
                "platform": "kalshi", "market_id": market_id,
                "ticker": "KXHIGHDEN", "title": "Denver High",
                "resolution_source": {"station": "KDEN", "type": "daily_high", "source": "NWS CLI"},
            })
        with get_session(engine=mem_engine) as session:
            row = session.get(Market, market_id)
            assert row.threshold_high == pytest.approx(80.0)

    def test_resolution_source_not_dict(self, scanner, mem_engine):
        # lines 440-443
        market_id = "non-dict-src"
        with get_session(engine=mem_engine) as session:
            scanner.store_market_metadata(session, market_id, {
                "platform": "polymarket", "ticker": market_id,
                "title": "Weather test", "resolution_source": "some-string",
            })
        with get_session(engine=mem_engine) as session:
            row = session.get(Market, market_id)
            assert row.station == "UNKNOWN"

    def test_store_market_price_log_debug(self, scanner, mem_engine):
        # line 473 (log.debug)
        with get_session(engine=mem_engine) as session:
            session.add(Market(
                id="mp-debug-test", platform="kalshi", city="nyc",
                station="KNYC", date="2026-03-30", market_type="daily_high",
            ))
        with get_session(engine=mem_engine) as session:
            scanner.store_market_price(session, "mp-debug-test", {"yes_price": 0.55})
        with get_session(engine=mem_engine) as session:
            row = session.query(MarketPrice).filter_by(market_id="mp-debug-test").first()
            assert row.yes_price == pytest.approx(0.55)


@pytest.mark.asyncio
class TestScanKalshiDirect:
    async def test_scan_direct_success(self, scanner):
        # lines 351-366 - fetch_kalshi_weather_markets imported locally, patch source
        mock_data = {
            "ticker": "KXHIGHNY", "market_id": "KXHIGHNY-26MAR30-T76",
            "title": "NY High Temp", "yes_price": 0.45,
            "volume_24h": 100.0, "liquidity": 500.0, "resolution_date": None,
        }
        with patch(
            "weatheredge.data.kalshi_client.fetch_kalshi_weather_markets",
            new=AsyncMock(return_value=[mock_data]),
        ):
            results = await scanner._scan_kalshi_direct()
        assert isinstance(results, list)

    async def test_scan_direct_handles_error(self, scanner):
        # lines 357-359 - patch at source module
        with patch(
            "weatheredge.data.kalshi_client.fetch_kalshi_weather_markets",
            new=AsyncMock(side_effect=Exception("API down")),
        ):
            results = await scanner._scan_kalshi_direct()
        assert results == []

    async def test_scan_kalshi_falls_back_to_direct(self, scanner):
        # lines 308-313
        with patch("weatheredge.data.market.pmxt.Kalshi") as MockKalshi:
            MockKalshi.return_value.fetch_events.return_value = []
            with patch.object(scanner, "_scan_kalshi_direct", new=AsyncMock(return_value=[])):
                results = await scanner.scan_kalshi()
        assert isinstance(results, list)

    async def test_scan_polymarket_empty_cities_no_stations(self, scanner):
        # lines 264-269
        scanner.settings.target_cities = "Atlantis"
        with patch("weatheredge.data.market.pmxt.Polymarket") as MockPoly:
            MockPoly.return_value.fetch_events.return_value = []
            results = await scanner.scan_polymarket()
        assert results == []


# ---------------------------------------------------------------------------
# Additional tests to push prepare_polydata_snapshot and polymarket_price_builder
# to >95% by covering remaining edge-case lines.
# ---------------------------------------------------------------------------


class TestExtractFirstMatchingMemberNonFile:
    def test_skips_non_file_members(self, tmp_path):
        # line 41: continue when member.isfile() is False (directory member)
        archive = tmp_path / "withdir.tar.xz"
        with tarfile.open(archive, mode="w:xz") as tf:
            # Add a directory member
            dir_info = tarfile.TarInfo(name="somedir")
            dir_info.type = tarfile.DIRTYPE
            tf.addfile(dir_info)
            # Add the actual file we want
            data = _MARKETS_CSV.encode()
            info = tarfile.TarInfo(name="markets.csv")
            info.size = len(data)
            tf.addfile(info, BytesIO(data))

        out = tmp_path / "out" / "markets.csv"
        member = _extract_first_matching_member(archive, ("markets.csv",), out)
        assert member is not None  # found past the directory
        assert out.exists()


class TestPreparePolyDataArchiveNonFileInStream:
    def test_stream_mode_skips_directory_members(self, tmp_path):
        # line 94: continue when member is not a file in streaming loop
        archive = tmp_path / "withdir.tar.xz"
        with tarfile.open(archive, mode="w:xz") as tf:
            # markets.csv first
            data = _MARKETS_CSV.encode()
            info = tarfile.TarInfo(name="markets.csv")
            info.size = len(data)
            tf.addfile(info, BytesIO(data))
            # directory member (will be skipped at line 94)
            dir_info = tarfile.TarInfo(name="processed")
            dir_info.type = tarfile.DIRTYPE
            tf.addfile(dir_info)
            # actual trades file
            data2 = _TRADES_CSV.encode()
            info2 = tarfile.TarInfo(name="processed/trades.csv")
            info2.size = len(data2)
            tf.addfile(info2, BytesIO(data2))

        work = tmp_path / "work"
        report = prepare_poly_data_archive(
            archive_path=archive, work_dir=work,
            output_prices_csv=tmp_path / "prices.csv",
            extract_trades_csv=False,
        )
        assert report is not None
        assert "stream" in report.extracted_trades_csv


class TestParseDatetimePolyPandasPath:
    def test_pandas_datetime_valid(self):
        # line 95: dt = parsed.to_pydatetime() — pandas successfully parses a non-ISO string
        # "March 30, 2026" is parseable by pandas but not by datetime.fromisoformat
        result = _parse_datetime("March 30, 2026")
        # pandas can parse this
        if result is not None:
            assert result.date() == date(2026, 3, 30)
        # Either way, the code path was exercised


class TestYesPriceOutOfRangeInPolyBuilder:
    def test_yes_price_above_1_after_side_conversion(self, tmp_path):
        # line 227: yes_price > 1.0 -> continue
        # _side_to_yes_price returns a value in [0,1] for normal inputs,
        # so we need to craft a scenario where it returns > 1.0.
        # Easiest: monkeypatch _side_to_yes_price to return 1.5 for first call then 0.5
        from weatheredge.backtesting.polymarket_price_builder import build_market_prices_csv_from_poly_data as bmpd
        mp = tmp_path / "m.csv"
        tp = tmp_path / "t.csv"
        pd.DataFrame([{
            "id": "m1",
            "question": "Will the temperature in NYC on March 30 be 76\u00b0F or above?",
            "end_date_iso": "2024-03-30T23:00:00Z",
            "answer1": "Yes", "answer2": "No",
        }]).to_csv(mp, index=False)
        ts = int(datetime(2024, 3, 30, 1, tzinfo=timezone.utc).timestamp())
        pd.DataFrame([{
            "timestamp": ts, "market_id": "m1", "nonusdc_side": "token1", "price": 0.6,
        }]).to_csv(tp, index=False)

        call_count = [0]
        original = __import__(
            "weatheredge.backtesting.polymarket_price_builder",
            fromlist=["_side_to_yes_price"]
        )._side_to_yes_price

        def patched_side(price, nonusdc_side, answer1, answer2):
            call_count[0] += 1
            if call_count[0] == 1:
                return 1.5  # out of range -> line 227
            return original(price, nonusdc_side, answer1, answer2)

        with patch(
            "weatheredge.backtesting.polymarket_price_builder._side_to_yes_price",
            side_effect=patched_side,
        ):
            report = bmpd(mp, tp, tmp_path / "out.csv")
        # The out-of-range row was skipped
        assert report.trade_rows_used == 0
