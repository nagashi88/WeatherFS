"""Direct Kalshi public API client for weather market data.

No authentication required for read-only market data.
Fallback when pmxt returns 0 results for Kalshi.
"""

from __future__ import annotations

import logging
import aiohttp

log = logging.getLogger(__name__)

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
WEATHER_SERIES = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHLAX", "KXHIGHDEN"]


def _parse_dollar_price(value: object) -> float | None:
    """Parse a Kalshi dollar-string price field (e.g. '0.1600') into a float in (0,1)."""
    if value is None:
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if price <= 0.0 or price >= 1.0:
        return None
    return price


async def fetch_kalshi_weather_markets(
    session: aiohttp.ClientSession | None = None,
    series_tickers: list[str] | None = None,
) -> list[dict]:
    """Fetch active weather markets from Kalshi public API.

    No auth needed. Returns list of normalized market dicts.
    """
    tickers = series_tickers or WEATHER_SERIES
    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()

    all_markets = []
    try:
        for series in tickers:
            url = f"{KALSHI_API_BASE}/markets"
            params = {"series_ticker": series, "status": "open"}
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        log.warning("Kalshi API returned %d for %s", resp.status, series)
                        continue
                    data = await resp.json()
                    markets = data.get("markets", [])
                    for m in markets:
                        # Kalshi public API returns prices as dollar strings (e.g. '0.1600')
                        # in `yes_ask_dollars`/`no_ask_dollars`. None / 0 / out-of-range → None.
                        yes_price = _parse_dollar_price(m.get("yes_ask_dollars"))
                        no_price = _parse_dollar_price(m.get("no_ask_dollars"))
                        all_markets.append({
                            "ticker": series,
                            "market_id": m.get("ticker", ""),
                            "title": m.get("title", ""),
                            "subtitle": m.get("subtitle", ""),
                            "yes_price": yes_price,
                            "no_price": no_price,
                            "volume": m.get("volume", 0),
                            "open_interest": m.get("open_interest", 0),
                            "close_time": m.get("close_time", ""),
                            "status": m.get("status", ""),
                        })
            except Exception as e:
                log.warning("Kalshi fetch failed for %s: %s", series, e)
                continue
    finally:
        if own_session:
            await session.close()

    log.info(
        "Kalshi public API: fetched %d weather markets across %d series",
        len(all_markets),
        len(tickers),
    )
    return all_markets
