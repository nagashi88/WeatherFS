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
from weatheredge.data import market_parser as mp_module
from weatheredge.data.market_parser import (
    ParsedMarket,
    get_parse_failure_stats,
    parse_polymarket_title,
    reset_parse_history_for_tests,
)
from weatheredge.data.market import MarketScanner


@pytest.fixture(autouse=True)
def _clean_parse_history():
    """Each test starts with an empty rolling parse-failure window so the
    new failure-rate observability does not bleed between tests."""
    reset_parse_history_for_tests()
    yield
    reset_parse_history_for_tests()


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


# ---------------------------------------------------------------------------
# Negative thresholds (P1 fix) — winter US-mountain markets routinely list
# buckets like [-5, 0]. The pre-fix regex `(\d+)` silently dropped these.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title,expected_low,expected_high", [
    (
        "Will the highest temperature in Chicago be between -5-0°F on January 15?",
        -5.0, 0.0,
    ),
    (
        "Will the highest temperature in Denver be between -10--5°F on January 20?",
        -10.0, -5.0,
    ),
    (
        "Will the highest temperature in NYC be between -3-2°F on February 1?",
        -3.0, 2.0,
    ),
])
def test_parse_negative_bucket(title, expected_low, expected_high):
    result = parse_polymarket_title(title)
    assert result is not None, f"Failed to parse: {title}"
    assert result.threshold_low == expected_low
    assert result.threshold_high == expected_high
    assert result.threshold is None


@pytest.mark.parametrize("title,expected_threshold", [
    ("Will the highest temperature in Chicago be -5°F or above on January 15?", -5.0),
    ("Will the highest temperature in Denver be -10°F or higher on January 20?", -10.0),
])
def test_parse_negative_above(title, expected_threshold):
    result = parse_polymarket_title(title)
    assert result is not None, f"Failed to parse: {title}"
    assert result.threshold == expected_threshold
    assert result.direction == "above"


@pytest.mark.parametrize("title,expected_threshold", [
    ("Will the highest temperature in Chicago be -5°F or below on January 15?", -5.0),
    ("Will the highest temperature in Denver be -10°F or lower on January 20?", -10.0),
])
def test_parse_negative_below(title, expected_threshold):
    result = parse_polymarket_title(title)
    assert result is not None, f"Failed to parse: {title}"
    assert result.threshold == expected_threshold
    assert result.direction == "below"


def test_parse_negative_exceed():
    result = parse_polymarket_title(
        "Will the highest temperature in Denver exceed -5°F on January 20?"
    )
    assert result is not None
    assert result.threshold == -5.0
    assert result.direction == "above"


def test_parse_negative_celsius_bucket():
    """°C markets also need negative support (every continental winter)."""
    result = parse_polymarket_title(
        "Will the highest temperature in Helsinki be between -10--5°C on January 15?"
    )
    # Helsinki is in KNOWN_UNTRADABLE_CITIES so this won't parse to a city,
    # but Amsterdam is also untradable. Use a tradable city instead:
    result = parse_polymarket_title(
        "Will the highest temperature in NYC be between -10--5°C on January 15?"
    )
    assert result is not None
    assert result.threshold_low == -10.0
    assert result.threshold_high == -5.0
    assert result.unit == "C"


def test_parse_positive_bucket_not_misread_as_negative():
    """Regression: greedy `(-?\\d+)` must not steal a leading `-` that's
    actually the bucket separator. 'between 5-10°F' must remain (5, 10),
    NOT (5, -10) or (5-1, 0)."""
    result = parse_polymarket_title(
        "Will the highest temperature in NYC be between 5-10°F on January 15?"
    )
    assert result is not None
    assert result.threshold_low == 5.0
    assert result.threshold_high == 10.0


# ---------------------------------------------------------------------------
# P2 — parse-failure rate observability
# ---------------------------------------------------------------------------


def test_parse_failure_stats_zero_when_no_attempts():
    stats = get_parse_failure_stats()
    assert stats == {"attempts": 0, "failures": 0, "failure_rate": 0.0}


def test_parse_failure_stats_records_success_and_failure():
    parse_polymarket_title("Will the highest temperature in NYC be 50°F or above on April 4?")
    parse_polymarket_title("totally garbage title — no city, no date, no threshold")
    parse_polymarket_title("Will the highest temperature in Chicago be 50°F or above on April 4?")
    stats = get_parse_failure_stats()
    assert stats["attempts"] == 3
    assert stats["failures"] == 1
    assert stats["failure_rate"] == pytest.approx(1 / 3)


def test_parse_failure_alert_fires_above_threshold(caplog):
    """30% failure rate over the rolling window must emit one throttled WARNING."""
    # Need at least _PARSE_FAILURE_MIN_SAMPLES attempts.
    for _ in range(15):
        parse_polymarket_title("Will the highest temperature in NYC be 50°F or above on April 4?")
    # Add 10 unparseable titles → 10/25 = 40% failure rate, above 30%.
    with caplog.at_level(logging.WARNING, logger="weatheredge.data.market_parser"):
        for _ in range(10):
            parse_polymarket_title("malformed " + "x" * 5)

    warnings = [r for r in caplog.records if "PARSE_FAILURE_RATE_HIGH" in r.message]
    assert len(warnings) >= 1, "Expected at least one rate-high WARNING"
    # Throttle: cooldown means we must NOT see N warnings — at most a couple.
    assert len(warnings) <= 3


def test_parse_failure_alert_silent_below_threshold(caplog):
    """A small number of failures — under the 30% threshold — must stay silent."""
    # 25 attempts, 4 failures → 16% (below 30%).
    for _ in range(21):
        parse_polymarket_title("Will the highest temperature in NYC be 50°F or above on April 4?")
    with caplog.at_level(logging.WARNING, logger="weatheredge.data.market_parser"):
        for _ in range(4):
            parse_polymarket_title("malformed nope")

    warnings = [r for r in caplog.records if "PARSE_FAILURE_RATE_HIGH" in r.message]
    assert warnings == []


def test_parse_failure_alert_silent_below_min_samples(caplog):
    """Even at 100% failure rate, warning is suppressed until min_samples reached.

    Threshold is _PARSE_FAILURE_MIN_SAMPLES (currently 5); with strictly
    fewer attempts than that, no warning fires regardless of failure rate.
    """
    with caplog.at_level(logging.WARNING, logger="weatheredge.data.market_parser"):
        # 4 failures only — strictly under min_samples=5.
        for _ in range(mp_module._PARSE_FAILURE_MIN_SAMPLES - 1):
            parse_polymarket_title("totally bogus title")
    warnings = [r for r in caplog.records if "PARSE_FAILURE_RATE_HIGH" in r.message]
    assert warnings == []


def test_parse_failure_alert_throttled(caplog):
    """Once fired, the WARNING must be throttled by the cooldown — no flood."""
    for _ in range(15):
        parse_polymarket_title("Will the highest temperature in NYC be 50°F or above on April 4?")
    with caplog.at_level(logging.WARNING, logger="weatheredge.data.market_parser"):
        for _ in range(50):  # many more failures
            parse_polymarket_title("malformed nope " + str(_))
    warnings = [r for r in caplog.records if "PARSE_FAILURE_RATE_HIGH" in r.message]
    # Cooldown is 5 minutes — within a single test run we should see exactly 1.
    assert len(warnings) == 1


def test_parse_failure_window_pruning():
    """Records older than the window must not count toward the failure rate.

    History tuples now use `time.monotonic()` (float seconds since process
    start) instead of UTC datetimes — this makes pruning robust against
    backward NTP clock jumps. Backdate the entry to "2 hours ago" by
    subtracting from its monotonic timestamp.
    """
    parse_polymarket_title("malformed first")
    # Manually backdate the only entry beyond the window (1h).
    with mp_module._PARSE_LOCK:
        ts, ok = mp_module._PARSE_HISTORY[0]
        mp_module._PARSE_HISTORY[0] = (
            ts - 2 * 3600,  # 2 hours into the past, well outside the 1h window
            ok,
        )

    # A fresh attempt triggers the prune.
    parse_polymarket_title("Will the highest temperature in NYC be 50°F or above on April 4?")
    stats = get_parse_failure_stats()
    assert stats["attempts"] == 1  # old failure pruned, only the success counts
    assert stats["failures"] == 0
