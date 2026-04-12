"""NOAA NWS API data fetcher for WeatherEdge.

Two-step lookup per PRD Section 3.1.2:
  1. GET /points/{lat},{lon}  → returns forecastHourly URL
  2. GET {forecastHourly}     → returns hourly periods

Retry policy per Addendum Gap 12:
  3 attempts, exponential backoff: 1s, 4s, 16s (base ** attempt, base=1, exponents 0,1,2 → 1,4,16 … using 4**attempt).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TypedDict

import aiohttp
from sqlalchemy.orm import Session

from weatheredge.db.models import Forecast

log = logging.getLogger(__name__)

NOAA_BASE = "https://api.weather.gov"
USER_AGENT = "WeatherEdge/0.1 (weather trading bot)"
_RETRY_DELAYS = (1, 4, 16)  # seconds between attempts (Gap 12)


# ---------------------------------------------------------------------------
# Typed return structures
# ---------------------------------------------------------------------------

class HourlyPeriod(TypedDict):
    start_time: str          # ISO-8601 string from NOAA
    temperature_f: float
    wind_speed: str          # raw string, e.g. "10 mph"
    short_forecast: str


class NoaaForecast(TypedDict):
    lat: float
    lon: float
    forecast_hourly_url: str
    periods: list[HourlyPeriod]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _get_with_retry(
    session: aiohttp.ClientSession,
    url: str,
) -> dict:
    """GET *url* with up to 3 attempts and exponential backoff.

    Returns the parsed JSON dict on success.
    Raises the last aiohttp / ValueError on exhausted retries.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt, delay in enumerate(_RETRY_DELAYS):
        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
                body = await resp.text()
                raise ValueError(
                    f"NOAA API returned HTTP {resp.status} for {url}: {body[:200]}"
                )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < len(_RETRY_DELAYS) - 1:
                log.warning(
                    "NOAA fetch attempt %d/%d failed (%s). Retrying in %ds.",
                    attempt + 1,
                    len(_RETRY_DELAYS),
                    exc,
                    _RETRY_DELAYS[attempt + 1],
                )
                await asyncio.sleep(_RETRY_DELAYS[attempt + 1])
            else:
                log.error(
                    "NOAA fetch failed after %d attempts for %s: %s",
                    len(_RETRY_DELAYS),
                    url,
                    exc,
                )

    raise last_exc


def _parse_periods(raw_periods: list[dict]) -> list[HourlyPeriod]:
    """Extract fields of interest from NOAA hourly period objects."""
    result: list[HourlyPeriod] = []
    for p in raw_periods:
        result.append(
            HourlyPeriod(
                start_time=p["startTime"],
                temperature_f=float(p["temperature"]),
                wind_speed=p.get("windSpeed", ""),
                short_forecast=p.get("shortForecast", ""),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_noaa_forecast(
    lat: float,
    lon: float,
    session: aiohttp.ClientSession,
) -> NoaaForecast:
    """Fetch hourly NOAA NWS forecast for *lat*/*lon*.

    Step 1: GET /points/{lat},{lon}  → forecastHourly URL
    Step 2: GET {forecastHourly}     → hourly periods

    Returns a :class:`NoaaForecast` TypedDict with all hourly periods.

    Raises:
        ValueError: if the NOAA API returns unexpected JSON structure.
        aiohttp.ClientError: on network-level failure after all retries.
    """
    points_url = f"{NOAA_BASE}/points/{lat},{lon}"
    log.debug("NOAA points lookup: %s", points_url)
    points_data = await _get_with_retry(session, points_url)

    try:
        forecast_hourly_url: str = points_data["properties"]["forecastHourly"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unexpected /points response structure: {exc}. "
            f"Got keys: {list((points_data.get('properties') or {}).keys())}"
        ) from exc

    log.debug("NOAA hourly forecast lookup: %s", forecast_hourly_url)
    hourly_data = await _get_with_retry(session, forecast_hourly_url)

    try:
        raw_periods: list[dict] = hourly_data["properties"]["periods"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unexpected forecast/hourly response structure: {exc}"
        ) from exc

    periods = _parse_periods(raw_periods)
    log.info("Fetched %d NOAA hourly periods for (%.4f, %.4f)", len(periods), lat, lon)

    return NoaaForecast(
        lat=lat,
        lon=lon,
        forecast_hourly_url=forecast_hourly_url,
        periods=periods,
    )


def store_noaa_forecast(
    db_session: Session,
    station: str,
    target_date: str,
    data: NoaaForecast,
) -> list[Forecast]:
    """Persist NOAA hourly forecast periods to the forecasts table.

    One :class:`Forecast` row is written per hourly period.  The
    ``model`` column is set to ``'noaa_nws'`` and ``lead_time_hours``
    is derived from the period's ``startTime`` relative to *now*.

    Args:
        db_session: Active SQLAlchemy session (caller manages commit).
        station:    NWS station identifier, e.g. ``'KNYC'``.
        target_date: The market resolution date as ``'YYYY-MM-DD'``.
        data:        Parsed :class:`NoaaForecast` returned by
                     :func:`fetch_noaa_forecast`.

    Returns:
        List of :class:`Forecast` objects added to the session.
    """
    now = datetime.now(timezone.utc)
    rows: list[Forecast] = []

    for period in data["periods"]:
        # Parse ISO-8601 startTime; keep timezone info if present.
        start_dt = datetime.fromisoformat(period["start_time"])
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)

        lead_hours = max(0, int((start_dt - now).total_seconds() / 3600))

        row = Forecast(
            station=station,
            target_date=target_date,
            model="noaa_nws",
            forecast_timestamp=now,
            lead_time_hours=lead_hours,
            temperature_forecast=period["temperature_f"],
            precipitation_forecast=None,  # not in hourly endpoint
            ensemble_members=None,
        )
        db_session.add(row)
        rows.append(row)

    log.info(
        "Stored %d NOAA forecast rows for station=%s date=%s",
        len(rows),
        station,
        target_date,
    )
    return rows
