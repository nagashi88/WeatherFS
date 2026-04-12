"""Tests for historical market price CSV loader."""

from __future__ import annotations

from datetime import date

from weatheredge.backtesting.market_prices import HistoricalMarketPriceBook


def test_from_rows_normalizes_city_direction_and_threshold():
    rows = [
        {
            "city": "New York",
            "date": "2026-03-30",
            "threshold": "76",
            "direction": "OVER",
            "yes_price": "0.42",
        }
    ]
    book = HistoricalMarketPriceBook.from_rows(rows)
    price = book.get_yes_price("KNYC", date(2026, 3, 30), 76.0, "above")
    assert price == 0.42


def test_from_rows_accepts_cents_format_and_builds_complement():
    rows = [
        {
            "station": "KMDW",
            "date": "2026-03-30",
            "threshold": 70,
            "direction": "above",
            "yes_price": 65,  # cents format
        }
    ]
    book = HistoricalMarketPriceBook.from_rows(rows)
    above_price = book.get_yes_price("KMDW", date(2026, 3, 30), 70.0, "above")
    below_price = book.get_yes_price("KMDW", date(2026, 3, 30), 70.0, "below")
    assert above_price == 0.65
    assert below_price == 0.35


def test_from_rows_skips_invalid_rows():
    rows = [
        {"station": "KNYC", "date": "bad-date", "threshold": 70, "direction": "above", "yes_price": 0.5},
        {"station": "KNYC", "date": "2026-03-30", "threshold": "bad", "direction": "above", "yes_price": 0.5},
        {"station": "KNYC", "date": "2026-03-30", "threshold": 70, "direction": "above", "yes_price": 999},
    ]
    book = HistoricalMarketPriceBook.from_rows(rows)
    assert len(book) == 0


def test_missing_direction_defaults_to_above_and_complement_is_available():
    rows = [
        {"station": "KMIA", "date": "2026-03-30", "threshold": 80, "yes_price": 0.25}
    ]
    book = HistoricalMarketPriceBook.from_rows(rows)
    assert book.get_yes_price("KMIA", date(2026, 3, 30), 80.0, "above") == 0.25
    assert book.get_yes_price("KMIA", date(2026, 3, 30), 80.0, "below") == 0.75


def test_case_insensitive_column_names_supported():
    rows = [
        {
            "STATION": "KDEN",
            "DATE": "2026-03-30",
            "THRESHOLD": "75",
            "DIRECTION": "below",
            "YES_PRICE": "0.61",
        }
    ]
    book = HistoricalMarketPriceBook.from_rows(rows)
    assert book.get_yes_price("KDEN", date(2026, 3, 30), 75.0, "below") == 0.61


def test_kalshi_ticker_parsing_for_station_threshold_direction_and_date():
    rows = [
        {
            "ticker": "KXHIGHDEN-26MAR30-T76",
            "yes_ask": 64,
        }
    ]
    book = HistoricalMarketPriceBook.from_rows(rows)
    assert book.get_yes_price("KDEN", date(2026, 3, 30), 76.0, "above") == 0.64
    assert book.get_yes_price("KDEN", date(2026, 3, 30), 76.0, "below") == 0.36
