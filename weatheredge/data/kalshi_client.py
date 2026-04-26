"""Direct Kalshi public API client for weather market data.

No authentication required for read-only market data.
Fallback when pmxt returns 0 results for Kalshi.
"""

from __future__ import annotations

import asyncio
import logging
import aiohttp

log = logging.getLogger(__name__)

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
WEATHER_SERIES = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHLAX", "KXHIGHDEN"]

# Retry policy mirrors data/weather.py and data/noaa.py: 3 attempts with
# exponential backoff between attempts (1s / 4s). 5xx and network errors retry; 4xx
# is treated as a config error and returns None without retrying.
_MAX_ATTEMPTS = 3
_RETRY_DELAYS = [1, 4]


async def _get_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    params: dict | None = None,
) -> dict | None:
    """GET *url* with up to 3 attempts and exponential backoff.

    Returns the parsed JSON dict on success, or None if every attempt
    failed (caller skips the series rather than aborting the cycle).
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            async with session.get(url, params=params or {}) as resp:
                if resp.status == 200:
                    return await resp.json()
                if 400 <= resp.status < 500:
                    log.warning(
                        "Kalshi API returned %d for %s — not retrying (4xx)",
                        resp.status, url,
                    )
                    return None
                # 5xx → fall through to retry path
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history, status=resp.status,
                )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                next_delay = _RETRY_DELAYS[attempt]
                log.warning(
                    "Kalshi fetch attempt %d/%d failed (%s). Retrying in %ds.",
                    attempt + 1, _MAX_ATTEMPTS, exc, next_delay,
                )
                await asyncio.sleep(next_delay)
            else:
                log.error(
                    "Kalshi fetch failed after %d attempts for %s: %s",
                    _MAX_ATTEMPTS, url, exc,
                )
    return None


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


def _parse_float(value: object, default: float = 0.0) -> float:
    """Parse a Kalshi numeric-string field (e.g. volume_24h_fp='807.88') into a float."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
                data = await _get_with_retry(session, url, params)
                if data is None:
                    continue
                markets = data.get("markets", [])
                for m in markets:
                        # Kalshi public API returns prices as dollar strings (e.g. '0.1600')
                        # in `yes_ask_dollars`/`no_ask_dollars`. None / 0 / out-of-range → None.
                        yes_price = _parse_dollar_price(m.get("yes_ask_dollars"))
                        no_price = _parse_dollar_price(m.get("no_ask_dollars"))
                        # Kalshi returns volume/open_interest as stringified floats in
                        # `*_fp` fields (e.g. volume_24h_fp='807.88'). Expose both
                        # `volume` and `volume_24h` so downstream (main.fetch_market_data)
                        # matches the pmxt path which reads `volume_24h` first.
                        volume_24h = _parse_float(m.get("volume_24h_fp"))
                        volume_total = _parse_float(m.get("volume_fp"))
                        open_interest = _parse_float(m.get("open_interest_fp"))
                        all_markets.append({
                            "ticker": series,
                            "market_id": m.get("ticker", ""),
                            "title": m.get("title", ""),
                            "subtitle": m.get("subtitle", ""),
                            "yes_price": yes_price,
                            "no_price": no_price,
                            "volume_24h": volume_24h,
                            "volume": volume_total,
                            "open_interest": open_interest,
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
