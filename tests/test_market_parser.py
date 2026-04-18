"""Tests for weatheredge.data.market_parser — Polymarket title parsing.

Covers:
- Bucket parsing ("between 50-51°F")
- Above/higher threshold parsing ("55°F or above")
- Below/lower threshold parsing ("43°F or below")
- Exceed threshold parsing ("exceed 60°F")
- City name variants (NYC, New York City, New York)
- Date extraction ("on April 4")
- Station mapping for multiple cities
- Unknown city returns None
- Missing date returns None
- Real Polymarket titles from production logs
- Integration with MarketScanner.process_market()
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from weatheredge.config import Settings
from weatheredge.data.market_parser import ParsedMarket, parse_polymarket_title
from weatheredge.data.market import MarketScanner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def settings():
    """Minimal Settings instance with no real credentials."""
    return Settings(
        polymarket_private_key="",
        polymarket_proxy_address="",
        kalshi_api_key="",
        kalshi_private_key="",
    )


@pytest.fixture()
def scanner(settings):
    """MarketScanner with minimal settings."""
    return MarketScanner(settings=settings)


# ---------------------------------------------------------------------------
# Bucket parsing
# ---------------------------------------------------------------------------

def test_parse_bucket():
    """'between 50-51°F' parses to threshold_low=50, threshold_high=51."""
    result = parse_polymarket_title(
        "Will the highest temperature in New York City be between 50-51°F on April 4?"
    )
    assert result is not None
    assert result.threshold_low == 50.0
    assert result.threshold_high == 51.0
    assert result.threshold is None


# ---------------------------------------------------------------------------
# Above / higher
# ---------------------------------------------------------------------------

def test_parse_above():
    """'55°F or above' parses to threshold=55, direction='above'."""
    result = parse_polymarket_title(
        "Will the highest temperature in Denver be 55°F or above on April 4?"
    )
    assert result is not None
    assert result.threshold == 55.0
    assert result.direction == "above"


# ---------------------------------------------------------------------------
# Below / lower
# ---------------------------------------------------------------------------

def test_parse_below():
    """'43°F or below' parses to threshold=43, direction='below'."""
    result = parse_polymarket_title(
        "Will the highest temperature in Denver be 43°F or below on April 4?"
    )
    assert result is not None
    assert result.threshold == 43.0
    assert result.direction == "below"
    # Regression: substring "low" in "below" must NOT flip a daily-high
    # contract to daily_low. METAR override picks the wrong sensor reading
    # (daily MIN vs MAX) when this misclassifies, so re-evaluation goes off.
    assert result.market_type == "daily_high"


def test_parse_below_does_not_misclassify_as_daily_low():
    """A 'highest temperature ... or below' contract is direction-down on the
    daily HIGH, not a daily_low contract. The naive `'low' in title` check
    used to flip these because 'below' contains 'low'."""
    result = parse_polymarket_title(
        "Will the highest temperature in Chicago be 55°F or below on April 20?"
    )
    assert result is not None
    assert result.market_type == "daily_high"
    assert result.direction == "below"


def test_parse_lowest_temperature_still_classified_as_daily_low():
    """Counter-test: a real 'Lowest temperature' contract must still parse as
    daily_low, ensuring the regex tightening didn't over-correct."""
    result = parse_polymarket_title(
        "Will the lowest temperature in Denver be 30°F or below on April 4?"
    )
    assert result is not None
    assert result.market_type == "daily_low"


# ---------------------------------------------------------------------------
# Exceed
# ---------------------------------------------------------------------------

def test_parse_exceed():
    """'exceed 60°F' parses to threshold=60, direction='above'."""
    result = parse_polymarket_title(
        "Will the highest temperature in Chicago exceed 60°F on April 4?"
    )
    assert result is not None
    assert result.threshold == 60.0
    assert result.direction == "above"


# ---------------------------------------------------------------------------
# City variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("city_text,expected_station", [
    ("New York City", "KNYC"),
    ("NYC", "KNYC"),
    ("New York", "KNYC"),
])
def test_parse_nyc_variants(city_text, expected_station):
    """NYC, New York City, and New York all map to KNYC."""
    result = parse_polymarket_title(
        f"Will the highest temperature in {city_text} be between 50-51°F on April 4?"
    )
    assert result is not None
    assert result.station == expected_station


def test_parse_chicago():
    """Chicago maps to KMDW."""
    result = parse_polymarket_title(
        "Will the highest temperature in Chicago be between 56-57°F on April 3?"
    )
    assert result is not None
    assert result.station == "KMDW"


def test_parse_denver():
    """Denver maps to KDEN."""
    result = parse_polymarket_title(
        "Will the highest temperature in Denver be between 50-51°F on April 3?"
    )
    assert result is not None
    assert result.station == "KDEN"


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

def test_parse_date():
    """'on April 4' extracts to '2026-04-04'."""
    result = parse_polymarket_title(
        "Will the highest temperature in NYC be between 50-51°F on April 4?"
    )
    assert result is not None
    assert result.date == "2026-04-04"


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------

def test_parse_unknown_city():
    """Unknown city 'Timbuktu' returns None."""
    result = parse_polymarket_title(
        "Will the highest temperature in Timbuktu be between 50-51°F on April 4?"
    )
    assert result is None


def test_parse_no_date():
    """Missing date returns None."""
    result = parse_polymarket_title(
        "Will the highest temperature in NYC be between 50-51°F?"
    )
    assert result is None


def test_unknown_city_logs_warning(caplog):
    """A genuinely unmapped city must still emit WARNING for alerting."""
    with caplog.at_level(logging.WARNING, logger="weatheredge.data.market_parser"):
        parse_polymarket_title(
            "Will the highest temperature in Timbuktu be 50°F or above on April 4?"
        )
    assert any("UNKNOWN_CITY_HIGH_PRIORITY" in r.message for r in caplog.records)


@pytest.mark.parametrize("city", [
    "Cape Town", "Lagos", "Helsinki", "Amsterdam", "Jeddah",
    "Karachi", "Jakarta", "Manila", "Kuala Lumpur", "Guangzhou",
    "Busan", "Panama City",
])
def test_known_untradable_cities_log_debug_not_warning(caplog, city):
    """Cities we know about but cannot trade (no RMSE/timezone/HRRR) should
    NOT spam the WARNING channel — they're a known-skip, not a real alert."""
    with caplog.at_level(logging.DEBUG, logger="weatheredge.data.market_parser"):
        result = parse_polymarket_title(
            f"Will the highest temperature in {city} be 50°F or above on April 4?"
        )
    assert result is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not warnings, f"Expected no WARNING for {city}, got: {[r.message for r in warnings]}"


def test_month_false_positive_does_not_warn(caplog):
    """Global-temperature-anomaly markets (no city, contain 'in April 2026')
    falsely match the 'in {Capitalized}' regex and extract the month name as
    a city. Observed ~252/h in production. Must not pollute WARNING channel."""
    with caplog.at_level(logging.DEBUG, logger="weatheredge.data.market_parser"):
        parse_polymarket_title(
            "April 2026 Temperature Increase (ºC) - Will global temperature "
            "increase by less than 0.6°C in April 2026?"
        )
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not warnings, f"unexpected: {[r.message for r in warnings]}"


# ---------------------------------------------------------------------------
# Real Polymarket titles from production logs
# ---------------------------------------------------------------------------

REAL_TITLES = [
    (
        "Highest temperature in NYC on April 4? - Will the highest temperature in New York City be between 50-51°F on April 4?",
        "KNYC", "2026-04-04", 50.0, 51.0, None, "above",
    ),
    (
        "Highest temperature in Denver on April 3? - Will the highest temperature in Denver be between 50-51°F on April 3?",
        "KDEN", "2026-04-03", 50.0, 51.0, None, "above",
    ),
    (
        "Highest temperature in Denver on April 4? - Will the highest temperature in Denver be 43°F or below on April 4?",
        "KDEN", "2026-04-04", None, None, 43.0, "below",
    ),
    (
        "Highest temperature in Denver on April 4? - Will the highest temperature in Denver be 62°F or higher on April 4?",
        "KDEN", "2026-04-04", None, None, 62.0, "above",
    ),
    (
        "Highest temperature in Chicago on April 3? - Will the highest temperature in Chicago be between 56-57°F on April 3?",
        "KMDW", "2026-04-03", 56.0, 57.0, None, "above",
    ),
]


@pytest.mark.parametrize(
    "title,station,date,t_low,t_high,threshold,direction",
    REAL_TITLES,
    ids=[
        "nyc_bucket_apr4",
        "denver_bucket_apr3",
        "denver_below_apr4",
        "denver_above_apr4",
        "chicago_bucket_apr3",
    ],
)
def test_parse_real_titles(title, station, date, t_low, t_high, threshold, direction):
    """Parse real Polymarket titles exactly as seen in production logs."""
    result = parse_polymarket_title(title)
    assert result is not None, f"Failed to parse: {title}"
    assert result.station == station
    assert result.date == date
    assert result.threshold_low == t_low
    assert result.threshold_high == t_high
    assert result.threshold == threshold
    assert result.direction == direction


# ---------------------------------------------------------------------------
# Integration: process_market uses parser for Polymarket
# ---------------------------------------------------------------------------

def test_process_market_uses_parser(scanner):
    """process_market falls back to title parsing for unknown Polymarket tickers."""
    market_data = {
        "ticker": "1781240",
        "title": "Highest temperature in NYC on April 4? - Will the highest temperature in New York City be between 50-51°F on April 4?",
    }
    result = scanner.process_market(market_data, "polymarket")

    assert result is not None
    assert result["platform"] == "polymarket"
    assert result["resolution_source"]["station"] == "KNYC"
    assert result["resolution_source"]["type"] == "daily_high"
    assert result["resolution_source"]["source"] == "parsed_from_title"
    assert "parsed" in result
    assert result["parsed"].threshold_low == 50.0
    assert result["parsed"].threshold_high == 51.0
