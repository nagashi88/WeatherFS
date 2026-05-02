"""Tests for weatheredge.data.market — MarketScanner and RESOLUTION_SOURCES.

Covers:
- RESOLUTION_SOURCES contains all 5 Kalshi tickers with correct station codes
- get_resolution_source returns correct data for known ticker
- get_resolution_source returns None for unknown ticker
- process_market returns None + logs warning for unknown market (Addendum Gap 7b)
- process_market returns enriched dict for known market
- store_market_price writes to DB correctly
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from weatheredge.config import Settings
from weatheredge.data.market import RESOLUTION_SOURCES, MarketScanner
from weatheredge.data.market_parser import ParsedMarket
from weatheredge.db.database import init_db, get_session, make_engine
from weatheredge.db.models import Market, MarketPrice


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


@pytest.fixture()
def engine():
    """In-memory SQLite engine with all tables."""
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    yield eng
    eng.dispose()


# ---------------------------------------------------------------------------
# RESOLUTION_SOURCES structure
# ---------------------------------------------------------------------------

EXPECTED_KALSHI_TICKERS = {
    "KXHIGHNY": "KNYC",
    "KXHIGHCHI": "KMDW",
    "KXHIGHMIA": "KMIA",
    "KXHIGHLAX": "KLAX",
    "KXHIGHDEN": "KDEN",
}


def test_resolution_sources_has_all_5_kalshi_tickers():
    """RESOLUTION_SOURCES must contain all 5 Kalshi weather tickers."""
    kalshi = RESOLUTION_SOURCES["kalshi"]
    missing = set(EXPECTED_KALSHI_TICKERS) - set(kalshi)
    assert not missing, f"Missing Kalshi tickers: {missing}"


@pytest.mark.parametrize("ticker,station", EXPECTED_KALSHI_TICKERS.items())
def test_kalshi_ticker_has_correct_station(ticker, station):
    """Each Kalshi ticker must map to the correct NWS station code."""
    entry = RESOLUTION_SOURCES["kalshi"][ticker]
    assert entry["station"] == station, (
        f"{ticker}: expected station {station!r}, got {entry['station']!r}"
    )


def test_resolution_sources_has_polymarket_templates():
    """RESOLUTION_SOURCES must contain both Polymarket templates."""
    poly = RESOLUTION_SOURCES["polymarket"]
    assert "london_temp" in poly
    assert "nyc_precipitation" in poly


def test_kalshi_tickers_have_required_fields():
    """Every Kalshi entry must have station, type, source, timezone, report_time."""
    required = {"station", "type", "source", "timezone", "report_time"}
    for ticker, entry in RESOLUTION_SOURCES["kalshi"].items():
        missing = required - set(entry)
        assert not missing, f"{ticker} missing fields: {missing}"


# ---------------------------------------------------------------------------
# get_resolution_source
# ---------------------------------------------------------------------------

def test_get_resolution_source_returns_correct_data_for_known_ticker(scanner):
    """get_resolution_source returns the full mapping dict for a known ticker."""
    result = scanner.get_resolution_source("kalshi", "KXHIGHNY")
    assert result is not None
    assert result["station"] == "KNYC"
    assert result["type"] == "daily_high"
    assert result["source"] == "NWS CLI"
    assert result["timezone"] == "EST"


def test_get_resolution_source_returns_none_for_unknown_ticker(scanner):
    """get_resolution_source returns None for a ticker not in RESOLUTION_SOURCES."""
    result = scanner.get_resolution_source("kalshi", "KXHIGHBOS")
    assert result is None


def test_get_resolution_source_returns_none_for_unknown_platform(scanner):
    """get_resolution_source returns None for an entirely unknown platform."""
    result = scanner.get_resolution_source("unknown_platform", "KXHIGHNY")
    assert result is None


def test_get_resolution_source_polymarket_known(scanner):
    """get_resolution_source returns data for known Polymarket template."""
    result = scanner.get_resolution_source("polymarket", "london_temp")
    assert result is not None
    assert result["station"] == "EGLC"


# ---------------------------------------------------------------------------
# process_market — Addendum Gap 7b
# ---------------------------------------------------------------------------

def test_process_market_returns_none_for_unknown_market(scanner, caplog):
    """process_market returns None and logs a warning for unknown markets (Gap 7b)."""
    market_data = {
        "ticker": "KXHIGHBOS",
        "title": "Will Boston hit 90F?",
    }
    with caplog.at_level(logging.WARNING, logger="weatheredge.data.market"):
        result = scanner.process_market(market_data, "kalshi")

    assert result is None
    assert any("Unknown market" in r.message for r in caplog.records), (
        "Expected a WARNING log containing 'Unknown market'"
    )


def test_process_market_warning_includes_platform_and_ticker(scanner, caplog):
    """Warning log for unknown market must include platform and ticker."""
    market_data = {"ticker": "MYSTERY_TICKER", "title": "Mystery market"}
    with caplog.at_level(logging.WARNING, logger="weatheredge.data.market"):
        scanner.process_market(market_data, "polymarket")

    combined = " ".join(r.message for r in caplog.records)
    assert "polymarket" in combined
    assert "MYSTERY_TICKER" in combined


def test_process_market_returns_enriched_dict_for_known_market(scanner):
    """process_market returns an enriched dict for a market in RESOLUTION_SOURCES."""
    market_data = {
        "ticker": "KXHIGHNY",
        "title": "New York High Temp",
        "yes_price": 0.45,
    }
    result = scanner.process_market(market_data, "kalshi")

    assert result is not None
    # Original fields preserved
    assert result["ticker"] == "KXHIGHNY"
    assert result["title"] == "New York High Temp"
    assert result["yes_price"] == 0.45
    # Enriched fields added
    assert result["platform"] == "kalshi"
    assert result["resolution_source"]["station"] == "KNYC"
    assert result["resolution_source"]["type"] == "daily_high"


def test_process_market_no_warning_for_known_market(scanner, caplog):
    """process_market must NOT log a warning for a known market."""
    market_data = {"ticker": "KXHIGHCHI", "title": "Chicago High Temp"}
    with caplog.at_level(logging.WARNING, logger="weatheredge.data.market"):
        scanner.process_market(market_data, "kalshi")

    assert not caplog.records, "Expected no warnings for a known market"


# ---------------------------------------------------------------------------
# store_market_price
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_with_market(engine):
    """Engine with a market row pre-inserted for FK references."""
    with get_session(engine=engine) as session:
        session.add(Market(
            id="test-market-kalshi-1",
            platform="kalshi",
            city="NYC",
            station="KNYC",
            date="2026-03-29",
            market_type="daily_high",
        ))
    return engine


def test_store_market_price_writes_to_db(scanner, db_with_market):
    """store_market_price must insert a MarketPrice row with correct values."""
    price_data = {
        "yes_price": 0.42,
        "no_price": 0.58,
        "volume": 1234.56,
        "spread": 0.02,
    }

    with get_session(engine=db_with_market) as session:
        scanner.store_market_price(session, "test-market-kalshi-1", price_data)

    with get_session(engine=db_with_market) as session:
        row = session.query(MarketPrice).filter_by(market_id="test-market-kalshi-1").first()
        assert row is not None
        assert row.yes_price == pytest.approx(0.42)
        assert row.no_price == pytest.approx(0.58)
        assert row.volume == pytest.approx(1234.56)
        assert row.spread == pytest.approx(0.02)
        assert row.market_id == "test-market-kalshi-1"
        assert isinstance(row.timestamp, datetime)


def test_store_market_price_partial_data(scanner, db_with_market):
    """store_market_price handles partial price_data (missing keys become None)."""
    price_data = {"yes_price": 0.60}

    with get_session(engine=db_with_market) as session:
        scanner.store_market_price(session, "test-market-kalshi-1", price_data)

    with get_session(engine=db_with_market) as session:
        row = session.query(MarketPrice).filter_by(market_id="test-market-kalshi-1").first()
        assert row is not None
        assert row.yes_price == pytest.approx(0.60)
        assert row.no_price is None
        assert row.volume is None
        assert row.spread is None


def test_store_market_metadata_upserts_parsed_market(scanner, engine):
    market_id = "pm-weather-1"
    parsed = ParsedMarket(
        city="nyc",
        station="KNYC",
        date="2026-04-01",
        market_type="daily_high",
        threshold=76.0,
        direction="above",
    )

    market_data = {
        "platform": "polymarket",
        "ticker": market_id,
        "title": "Will the temperature in NYC on April 1 be 76°F or above?",
        "parsed": parsed,
        "resolution_source": {"station": "KNYC", "type": "daily_high", "source": "parsed_from_title"},
    }

    with get_session(engine=engine) as session:
        scanner.store_market_metadata(session, market_id, market_data)

    with get_session(engine=engine) as session:
        row = session.get(Market, market_id)
        assert row is not None
        assert row.platform == "polymarket"
        assert row.station == "KNYC"
        assert row.city == "nyc"
        assert row.date == "2026-04-01"
        assert row.threshold_high == pytest.approx(76.0)
        payload = json.loads(row.resolution_source or "{}")
        assert payload.get("direction") == "above"
        assert payload.get("threshold") == pytest.approx(76.0)

    # Update same market and ensure we upsert (no duplicate rows).
    parsed_v2 = ParsedMarket(
        city="nyc",
        station="KNYC",
        date="2026-04-01",
        market_type="daily_high",
        threshold=77.0,
        direction="above",
    )
    with get_session(engine=engine) as session:
        scanner.store_market_metadata(session, market_id, {**market_data, "parsed": parsed_v2})

    with get_session(engine=engine) as session:
        row = session.get(Market, market_id)
        assert row is not None
        assert row.threshold_high == pytest.approx(77.0)
        count = session.query(Market).filter_by(id=market_id).count()
        assert count == 1


def test_store_market_metadata_infers_fields_from_kalshi_ticker(scanner, engine):
    market_id = "KXHIGHDEN-26MAR30-B76.5"
    market_data = {
        "platform": "kalshi",
        "market_id": market_id,
        "ticker": "KXHIGHDEN",
        "title": "Will Denver's high temperature on March 30 be 76.5°F or below?",
        "resolution_source": {"station": "KDEN", "type": "daily_high", "source": "NWS CLI"},
    }

    with get_session(engine=engine) as session:
        scanner.store_market_metadata(session, market_id, market_data)

    with get_session(engine=engine) as session:
        row = session.get(Market, market_id)
        assert row is not None
        assert row.platform == "kalshi"
        assert row.station == "KDEN"
        assert row.city == "denver"
        assert row.date == "2026-03-30"
        assert row.threshold_high == pytest.approx(76.5)
        payload = json.loads(row.resolution_source or "{}")
        assert payload.get("direction") == "below"


def test_store_market_metadata_parses_kalshi_bucket_title(scanner, engine):
    market_id = "KXHIGHNY-26APR23-B72.5"
    market_data = {
        "platform": "kalshi",
        "market_id": market_id,
        "ticker": "KXHIGHNY",
        "title": "Will the **high temp in NYC** be 72-73° on Apr 23, 2026?",
        "resolution_source": {"station": "KNYC", "type": "daily_high", "source": "NWS CLI"},
    }
    processed = scanner.process_market(market_data, "kalshi")

    with get_session(engine=engine) as session:
        scanner.store_market_metadata(session, market_id, processed)

    with get_session(engine=engine) as session:
        row = session.get(Market, market_id)
        assert row is not None
        assert row.threshold_low == pytest.approx(71.5)
        assert row.threshold_high == pytest.approx(73.5)
        payload = json.loads(row.resolution_source or "{}")
        assert payload.get("direction") == "above"


# ---------------------------------------------------------------------------
# scan_polymarket — mock pmxt.Polymarket
# ---------------------------------------------------------------------------

def _make_mock_market(market_id, title, yes_price=0.5):
    """Build a mock UnifiedMarket."""
    m = MagicMock()
    m.market_id = market_id
    m.title = title
    m.yes = MagicMock()
    m.yes.price = yes_price
    m.yes.outcome_id = None
    m.no = MagicMock()
    m.no.price = 1.0 - yes_price
    m.no.outcome_id = None
    m.volume_24h = 100.0
    m.liquidity = 500.0
    m.resolution_date = None
    return m


def _make_mock_event(event_id, title, markets):
    """Build a mock UnifiedEvent."""
    e = MagicMock()
    e.id = event_id
    e.title = title
    e.markets = markets
    return e


@pytest.mark.asyncio
async def test_scan_polymarket_returns_known_markets(scanner):
    """scan_polymarket filters by city and returns markets with known resolution sources."""
    mock_market = _make_mock_market("london_temp", "London daily high temperature")
    mock_event = _make_mock_event("evt-1", "NYC weather", [mock_market])

    with patch("weatheredge.data.market.pmxt.Polymarket") as MockPoly:
        instance = MockPoly.return_value
        instance.fetch_events.return_value = [mock_event]
        results = await scanner.scan_polymarket()

    # london_temp is a known Polymarket ticker but the event title contains NYC
    # so it passes the city filter
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_scan_polymarket_matches_target_city_from_market_title_alias(scanner):
    """Target-city filter should use parsed station, not only event-title substrings."""
    scanner.settings.target_cities = "NYC"

    mock_market = _make_mock_market(
        "0xnyc_alias_market",
        "Will the temperature in New York on April 3 be 70F or above?",
    )
    mock_event = _make_mock_event("evt-nyc", "Daily weather contracts", [mock_market])

    with patch("weatheredge.data.market.pmxt.Polymarket") as MockPoly:
        instance = MockPoly.return_value
        instance.fetch_events.return_value = [mock_event]
        results = await scanner.scan_polymarket()

    assert len(results) == 1
    assert results[0]["resolution_source"]["station"] == "KNYC"
    assert results[0]["yes_price"] == pytest.approx(0.5)
    assert results[0]["no_price"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_scan_polymarket_attaches_real_orderbook_spread(scanner):
    """Polymarket spread should come from the CLOB order book, not YES+NO prices."""
    scanner.settings.target_cities = "NYC"
    scanner.settings.polymarket_orderbook_min_volume = 2000.0
    scanner.settings.polymarket_orderbook_max_markets_per_scan = 10

    mock_market = _make_mock_market(
        "0xnyc_orderbook_market",
        "Will the temperature in New York on April 3 be 70F or above?",
        yes_price=0.41,
    )
    mock_market.yes.outcome_id = "yes-token-1"
    mock_market.volume_24h = 5000.0
    mock_event = _make_mock_event("evt-nyc", "Daily weather contracts", [mock_market])
    order_book = SimpleNamespace(
        bids=[SimpleNamespace(price=0.38, size=200.0), SimpleNamespace(price=0.40, size=120.0)],
        asks=[SimpleNamespace(price=0.45, size=500.0), SimpleNamespace(price=0.42, size=80.0)],
    )

    with patch("weatheredge.data.market.pmxt.Polymarket") as MockPoly:
        instance = MockPoly.return_value
        instance.fetch_events.return_value = [mock_event]
        instance.fetch_order_book.return_value = order_book
        results = await scanner.scan_polymarket()

    assert len(results) == 1
    assert results[0]["best_bid"] == pytest.approx(0.40)
    assert results[0]["best_ask"] == pytest.approx(0.42)
    assert results[0]["bid_size"] == pytest.approx(120.0)
    assert results[0]["ask_size"] == pytest.approx(80.0)
    assert results[0]["spread"] == pytest.approx(0.02)
    instance.fetch_order_book.assert_called_once_with("yes-token-1")


@pytest.mark.asyncio
async def test_scan_polymarket_leaves_spread_missing_without_two_sided_book(scanner):
    """One-sided books can be logged, but they are not liquid entry candidates."""
    scanner.settings.target_cities = "NYC"

    mock_market = _make_mock_market(
        "0xnyc_one_sided_market",
        "Will the temperature in New York on April 3 be 70F or above?",
        yes_price=0.01,
    )
    mock_market.yes.outcome_id = "yes-token-2"
    mock_market.volume_24h = 5000.0
    mock_event = _make_mock_event("evt-nyc", "Daily weather contracts", [mock_market])
    order_book = SimpleNamespace(
        bids=[],
        asks=[SimpleNamespace(price=0.02, size=100.0)],
    )

    with patch("weatheredge.data.market.pmxt.Polymarket") as MockPoly:
        instance = MockPoly.return_value
        instance.fetch_events.return_value = [mock_event]
        instance.fetch_order_book.return_value = order_book
        results = await scanner.scan_polymarket()

    assert len(results) == 1
    assert results[0]["best_ask"] == pytest.approx(0.02)
    assert results[0].get("best_bid") is None
    assert results[0].get("spread") is None


@pytest.mark.asyncio
async def test_scan_polymarket_fetches_adaptive_volume_orderbook_for_shadow(scanner):
    """Shadow-only adaptive liquidity needs real spreads below the main volume floor."""
    scanner.settings.target_cities = "NYC"
    scanner.settings.polymarket_orderbook_min_volume = 2000.0
    scanner.settings.shadow_twin_enabled = True
    scanner.settings.shadow_twin_adaptive_liquidity_enabled = True
    scanner.settings.shadow_twin_adaptive_liquidity_min_volume = 250.0

    mock_market = _make_mock_market(
        "0xnyc_adaptive_market",
        "Will the temperature in New York on April 3 be 70F or above?",
        yes_price=0.41,
    )
    mock_market.yes.outcome_id = "adaptive-token"
    mock_market.volume_24h = 300.0
    mock_event = _make_mock_event("evt-nyc", "Daily weather contracts", [mock_market])
    order_book = SimpleNamespace(
        bids=[SimpleNamespace(price=0.40, size=120.0)],
        asks=[SimpleNamespace(price=0.42, size=80.0)],
    )

    with patch("weatheredge.data.market.pmxt.Polymarket") as MockPoly:
        instance = MockPoly.return_value
        instance.fetch_events.return_value = [mock_event]
        instance.fetch_order_book.return_value = order_book
        results = await scanner.scan_polymarket()

    assert len(results) == 1
    assert results[0]["spread"] == pytest.approx(0.02)
    instance.fetch_order_book.assert_called_once_with("adaptive-token")


@pytest.mark.asyncio
async def test_scan_polymarket_prioritizes_main_volume_orderbooks(scanner):
    """Limited orderbook budget should cover main-liquidity candidates first."""
    scanner.settings.target_cities = "NYC"
    scanner.settings.polymarket_orderbook_min_volume = 2000.0
    scanner.settings.polymarket_orderbook_max_markets_per_scan = 1
    scanner.settings.shadow_twin_enabled = True
    scanner.settings.shadow_twin_adaptive_liquidity_enabled = True
    scanner.settings.shadow_twin_adaptive_liquidity_min_volume = 250.0

    low_volume = _make_mock_market(
        "0xnyc_low_volume_market",
        "Will the temperature in New York on April 3 be 69F or above?",
        yes_price=0.41,
    )
    low_volume.yes.outcome_id = "low-token"
    low_volume.volume_24h = 300.0
    main_volume = _make_mock_market(
        "0xnyc_main_volume_market",
        "Will the temperature in New York on April 3 be 70F or above?",
        yes_price=0.41,
    )
    main_volume.yes.outcome_id = "main-token"
    main_volume.volume_24h = 5000.0
    mock_event = _make_mock_event("evt-nyc", "Daily weather contracts", [low_volume, main_volume])
    order_book = SimpleNamespace(
        bids=[SimpleNamespace(price=0.40, size=120.0)],
        asks=[SimpleNamespace(price=0.42, size=80.0)],
    )

    with patch("weatheredge.data.market.pmxt.Polymarket") as MockPoly:
        instance = MockPoly.return_value
        instance.fetch_events.return_value = [mock_event]
        instance.fetch_order_book.return_value = order_book
        results = await scanner.scan_polymarket()

    by_id = {row["ticker"]: row for row in results}
    assert by_id["0xnyc_main_volume_market"]["spread"] == pytest.approx(0.02)
    assert by_id["0xnyc_low_volume_market"].get("spread") is None
    instance.fetch_order_book.assert_called_once_with("main-token")


@pytest.mark.asyncio
async def test_scan_polymarket_skips_markets_outside_target_stations(scanner):
    """Known markets outside configured target stations should be skipped."""
    scanner.settings.target_cities = "NYC"
    scanner.settings.shadow_twin_enabled = False

    mock_market = _make_mock_market("london_temp", "London daily high temperature")
    mock_event = _make_mock_event("evt-london", "NYC weather", [mock_market])

    with patch("weatheredge.data.market.pmxt.Polymarket") as MockPoly:
        instance = MockPoly.return_value
        instance.fetch_events.return_value = [mock_event]
        results = await scanner.scan_polymarket()

    assert results == []


@pytest.mark.asyncio
async def test_scan_polymarket_returns_empty_if_target_cities_unmapped(scanner):
    """If configured target cities cannot map to stations, scanner should skip all."""
    scanner.settings.target_cities = "Atlantis"

    mock_market = _make_mock_market(
        "0xnyc_alias_market",
        "Will the temperature in New York on April 3 be 70F or above?",
    )
    mock_event = _make_mock_event("evt-nyc", "Daily weather contracts", [mock_market])

    with patch("weatheredge.data.market.pmxt.Polymarket") as MockPoly:
        instance = MockPoly.return_value
        instance.fetch_events.return_value = [mock_event]
        results = await scanner.scan_polymarket()

    assert results == []


@pytest.mark.asyncio
async def test_scan_polymarket_handles_connection_error(scanner):
    """scan_polymarket returns empty list on connection error (Addendum Gap 12)."""
    with patch("weatheredge.data.market.pmxt.Polymarket") as MockPoly:
        MockPoly.side_effect = Exception("Connection refused")
        results = await scanner.scan_polymarket()

    assert results == []


@pytest.mark.asyncio
async def test_scan_kalshi_handles_connection_error(scanner):
    """scan_kalshi returns empty list on connection error (Addendum Gap 12)."""
    with patch("weatheredge.data.market.pmxt.Kalshi") as MockKalshi:
        MockKalshi.side_effect = Exception("Connection refused")
        with patch.object(scanner, "_scan_kalshi_direct", new=AsyncMock(return_value=[])):
            results = await scanner.scan_kalshi()

    assert results == []


@pytest.mark.asyncio
async def test_scan_kalshi_returns_known_markets(scanner):
    """scan_kalshi returns processed markets for known KXHIGH tickers."""
    mock_market = _make_mock_market("KXHIGHNY-26MAR29", "NY High Temp Mar 29", 0.40)
    mock_event = _make_mock_event("kx-evt-1", "KXHIGH weather", [mock_market])

    with patch("weatheredge.data.market.pmxt.Kalshi") as MockKalshi:
        instance = MockKalshi.return_value
        instance.fetch_events.return_value = [mock_event]
        results = await scanner.scan_kalshi()

    assert len(results) == 1
    assert results[0]["ticker"] == "KXHIGHNY"
    assert results[0]["platform"] == "kalshi"
    assert results[0]["resolution_source"]["station"] == "KNYC"
    assert results[0]["yes_price"] == pytest.approx(0.40)
    assert results[0]["no_price"] == pytest.approx(0.60)


@pytest.mark.asyncio
async def test_scan_kalshi_skips_unknown_tickers(scanner, caplog):
    """scan_kalshi skips markets whose ticker prefix is not in RESOLUTION_SOURCES."""
    mock_market = _make_mock_market("KXHIGHBOS-26MAR29", "Boston High Temp", 0.35)
    mock_event = _make_mock_event("kx-evt-2", "KXHIGH weather", [mock_market])

    with patch("weatheredge.data.market.pmxt.Kalshi") as MockKalshi:
        instance = MockKalshi.return_value
        instance.fetch_events.return_value = [mock_event]
        with patch.object(scanner, "_scan_kalshi_direct", new=AsyncMock(return_value=[])):
            with caplog.at_level(logging.WARNING, logger="weatheredge.data.market"):
                results = await scanner.scan_kalshi()

    assert results == []
    assert any("Unknown market" in r.message for r in caplog.records)
