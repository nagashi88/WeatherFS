"""Tests for weatheredge.data.kalshi_client — direct Kalshi public API client.

Covers:
- test_fetch_returns_normalized_markets: mock 3 markets → 3 dicts with correct fields
- test_fetch_handles_api_error: mock 500 response → empty list (no crash)
- test_fetch_multiple_series: mock 2 series → combined list
- test_market_data_has_expected_fields: ticker, market_id, title, yes_price present
- test_prices_converted_from_cents: yes_ask=45 → yes_price=0.45
- test_scan_kalshi_falls_back_to_direct: pmxt empty → results from direct API
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from weatheredge.data.kalshi_client import fetch_kalshi_weather_markets
from weatheredge.config import Settings
from weatheredge.data.market import MarketScanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(status: int, json_data: dict) -> MagicMock:
    """Build a mock aiohttp response context manager."""
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_mock_session(responses: list) -> MagicMock:
    """Build a mock aiohttp.ClientSession whose .get() yields responses in order."""
    session = MagicMock()
    session.get = MagicMock(side_effect=responses)
    # close is a coroutine
    session.close = AsyncMock()
    return session


def _sample_market(
    ticker: str = "KXHIGHNY-26MAR29",
    yes_ask_dollars: str = "0.4500",
    no_ask_dollars: str = "0.5500",
    volume_24h_fp: str = "807.88",
    volume_fp: str = "988.94",
    open_interest_fp: str = "909.94",
) -> dict:
    return {
        "ticker": ticker,
        "title": f"Will NY high temp exceed 60F?",
        "subtitle": "Daily high temperature",
        "yes_ask_dollars": yes_ask_dollars,
        "no_ask_dollars": no_ask_dollars,
        "volume_24h_fp": volume_24h_fp,
        "volume_fp": volume_fp,
        "open_interest_fp": open_interest_fp,
        "close_time": "2026-03-29T23:59:00Z",
        "status": "open",
    }


# ---------------------------------------------------------------------------
# test_fetch_returns_normalized_markets
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_returns_normalized_markets():
    """Mock 3 markets in one series → returns 3 normalized dicts."""
    markets = [_sample_market(f"KXHIGHNY-26MAR{i:02d}") for i in range(1, 4)]
    resp_cm = _make_mock_response(200, {"markets": markets})
    session = _make_mock_session([resp_cm])

    results = await fetch_kalshi_weather_markets(
        session=session, series_tickers=["KXHIGHNY"]
    )

    assert len(results) == 3
    for r in results:
        assert r["ticker"] == "KXHIGHNY"
        assert "market_id" in r
        assert "title" in r
        assert "yes_price" in r


# ---------------------------------------------------------------------------
# test_fetch_handles_api_error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_handles_api_error():
    """Mock 500 response → returns empty list without raising."""
    resp_cm = _make_mock_response(500, {})
    session = _make_mock_session([resp_cm])

    results = await fetch_kalshi_weather_markets(
        session=session, series_tickers=["KXHIGHNY"]
    )

    assert results == []


# ---------------------------------------------------------------------------
# test_fetch_multiple_series
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_multiple_series():
    """Mock responses for 2 series → combined list of markets."""
    resp_ny = _make_mock_response(200, {"markets": [_sample_market("KXHIGHNY-26MAR29")]})
    resp_chi = _make_mock_response(200, {"markets": [_sample_market("KXHIGHCHI-26MAR29")]})
    session = _make_mock_session([resp_ny, resp_chi])

    results = await fetch_kalshi_weather_markets(
        session=session, series_tickers=["KXHIGHNY", "KXHIGHCHI"]
    )

    assert len(results) == 2
    tickers = {r["ticker"] for r in results}
    assert tickers == {"KXHIGHNY", "KXHIGHCHI"}


# ---------------------------------------------------------------------------
# test_market_data_has_expected_fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_market_data_has_expected_fields():
    """Every returned dict must contain ticker, market_id, title, yes_price, volumes."""
    resp_cm = _make_mock_response(200, {"markets": [_sample_market()]})
    session = _make_mock_session([resp_cm])

    results = await fetch_kalshi_weather_markets(
        session=session, series_tickers=["KXHIGHNY"]
    )

    assert len(results) == 1
    r = results[0]
    for field in (
        "ticker", "market_id", "title", "yes_price", "no_price",
        "volume", "volume_24h", "open_interest",
    ):
        assert field in r, f"Missing field: {field}"
    # Volumes must be parsed from *_fp string fields
    assert r["volume_24h"] == pytest.approx(807.88)
    assert r["volume"] == pytest.approx(988.94)
    assert r["open_interest"] == pytest.approx(909.94)


# ---------------------------------------------------------------------------
# test_prices_parsed_from_dollar_strings
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prices_parsed_from_dollar_strings():
    """yes_ask_dollars='0.4500' in API response → yes_price=0.45 in normalized dict."""
    market = _sample_market(yes_ask_dollars="0.4500", no_ask_dollars="0.5500")
    resp_cm = _make_mock_response(200, {"markets": [market]})
    session = _make_mock_session([resp_cm])

    results = await fetch_kalshi_weather_markets(
        session=session, series_tickers=["KXHIGHNY"]
    )

    assert len(results) == 1
    assert results[0]["yes_price"] == pytest.approx(0.45)
    assert results[0]["no_price"] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# test_prices_none_when_missing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prices_none_when_missing():
    """yes_ask_dollars absent, '0.0000', or '1.0000' → yes_price=None."""
    # Case 1: fields absent
    market = _sample_market()
    del market["yes_ask_dollars"]
    del market["no_ask_dollars"]
    resp_cm = _make_mock_response(200, {"markets": [market]})
    session = _make_mock_session([resp_cm])
    results = await fetch_kalshi_weather_markets(
        session=session, series_tickers=["KXHIGHNY"]
    )
    assert results[0]["yes_price"] is None
    assert results[0]["no_price"] is None

    # Case 2: fields set to '0.0000' (no ask/bid on book)
    market_zero = _sample_market(yes_ask_dollars="0.0000", no_ask_dollars="0.0000")
    resp_cm2 = _make_mock_response(200, {"markets": [market_zero]})
    session2 = _make_mock_session([resp_cm2])
    results2 = await fetch_kalshi_weather_markets(
        session=session2, series_tickers=["KXHIGHNY"]
    )
    assert results2[0]["yes_price"] is None
    assert results2[0]["no_price"] is None


# ---------------------------------------------------------------------------
# test_fetch_continues_after_one_series_exception
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_continues_after_one_series_exception(caplog):
    """If one series raises an exception, remaining series are still fetched."""
    # First series exhausts its retry budget, second series still returns a market.
    boom = MagicMock()
    boom.__aenter__ = AsyncMock(side_effect=Exception("network error"))
    boom.__aexit__ = AsyncMock(return_value=False)
    boom2 = MagicMock()
    boom2.__aenter__ = AsyncMock(side_effect=Exception("network error"))
    boom2.__aexit__ = AsyncMock(return_value=False)
    boom3 = MagicMock()
    boom3.__aenter__ = AsyncMock(side_effect=Exception("network error"))
    boom3.__aexit__ = AsyncMock(return_value=False)
    resp_chi = _make_mock_response(200, {"markets": [_sample_market("KXHIGHCHI-26MAR29")]})

    session = _make_mock_session([boom, boom2, boom3, resp_chi])

    with patch("weatheredge.data.kalshi_client.asyncio.sleep", new=AsyncMock()):
        with caplog.at_level(logging.WARNING, logger="weatheredge.data.kalshi_client"):
            results = await fetch_kalshi_weather_markets(
                session=session, series_tickers=["KXHIGHNY", "KXHIGHCHI"]
            )

    assert len(results) == 1
    assert results[0]["ticker"] == "KXHIGHCHI"
    assert any("Kalshi fetch failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# test_scan_kalshi_falls_back_to_direct
# ---------------------------------------------------------------------------

@pytest.fixture()
def scanner():
    settings = Settings(
        polymarket_private_key="",
        polymarket_proxy_address="",
        kalshi_api_key="",
        kalshi_private_key="",
    )
    return MarketScanner(settings=settings)


@pytest.mark.asyncio
async def test_scan_kalshi_falls_back_to_direct(scanner):
    """When pmxt returns 0 markets, scan_kalshi uses the direct API fallback."""
    # pmxt returns empty list
    with patch("weatheredge.data.market.pmxt.Kalshi") as MockKalshi:
        instance = MockKalshi.return_value
        instance.fetch_events.return_value = []

        # direct API returns one known market
        direct_market = {
            "ticker": "KXHIGHNY",
            "market_id": "KXHIGHNY-26MAR29",
            "title": "NY High Temp",
            "subtitle": "",
            "yes_price": 0.45,
            "no_price": 0.55,
            "volume": 1000,
            "open_interest": 500,
            "close_time": "2026-03-29T23:59:00Z",
            "status": "open",
        }
        with patch(
            "weatheredge.data.market.MarketScanner._scan_kalshi_direct",
            new_callable=AsyncMock,
            return_value=[
                {
                    **direct_market,
                    "platform": "kalshi",
                    "resolution_source": {"station": "KNYC", "type": "daily_high"},
                }
            ],
        ):
            results = await scanner.scan_kalshi()

    assert len(results) == 1
    assert results[0]["ticker"] == "KXHIGHNY"
    assert results[0]["platform"] == "kalshi"


@pytest.mark.asyncio
async def test_scan_kalshi_direct_processes_known_tickers(scanner):
    """_scan_kalshi_direct filters out unknown tickers via process_market."""
    raw_markets = [
        {
            "ticker": "KXHIGHNY",
            "market_id": "KXHIGHNY-26MAR29",
            "title": "NY High",
            "subtitle": "",
            "yes_price": 0.45,
            "no_price": 0.55,
            "volume": 100,
            "open_interest": 50,
            "close_time": "",
            "status": "open",
        },
        {
            "ticker": "KXHIGHBOS",  # Unknown ticker — should be filtered
            "market_id": "KXHIGHBOS-26MAR29",
            "title": "Boston High",
            "subtitle": "",
            "yes_price": 0.40,
            "no_price": 0.60,
            "volume": 100,
            "open_interest": 50,
            "close_time": "",
            "status": "open",
        },
    ]

    with patch(
        "weatheredge.data.kalshi_client.fetch_kalshi_weather_markets",
        new_callable=AsyncMock,
        return_value=raw_markets,
    ):
        results = await scanner._scan_kalshi_direct()

    # Only KXHIGHNY passes; KXHIGHBOS is unknown
    assert len(results) == 1
    assert results[0]["ticker"] == "KXHIGHNY"


@pytest.mark.asyncio
async def test_scan_kalshi_direct_handles_exception(scanner, caplog):
    """_scan_kalshi_direct returns [] and logs a warning when fetch raises."""
    with patch(
        "weatheredge.data.kalshi_client.fetch_kalshi_weather_markets",
        new_callable=AsyncMock,
        side_effect=Exception("API down"),
    ):
        with caplog.at_level(logging.WARNING, logger="weatheredge.data.market"):
            results = await scanner._scan_kalshi_direct()

    assert results == []
    assert any("Kalshi direct API failed" in r.message for r in caplog.records)
