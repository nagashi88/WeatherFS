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
                        # BugFix M10: Sanity check Kalshi prices — API returns cents (0-100)
                        raw_yes = m.get("yes_ask", 0) or 0
                        raw_no = m.get("no_ask", 0) or 0
                        yes_price = raw_yes / 100.0 if raw_yes > 1.0 else raw_yes if raw_yes > 0 else None
                        no_price = raw_no / 100.0 if raw_no > 1.0 else raw_no if raw_no > 0 else None
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
