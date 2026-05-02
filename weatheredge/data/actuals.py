"""Observed temperature ingestion for live calibration.

Fetches real daily max temperatures from Open-Meteo Archive API and writes
Forecast.actual_temp for matching (station, target_date) rows.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
from sqlalchemy import select
from sqlalchemy.orm import Session

from weatheredge.db.models import Forecast

log = logging.getLogger(__name__)

_ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
_RETRY_DELAYS = (1, 4, 16)
_REQUEST_TIMEOUT_SECONDS = 12


def _is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status", None)
    if status == 429:
        return True
    message = str(exc).lower()
    return "429" in message or "too many requests" in message


async def _get_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """GET with exponential-backoff retries."""
    last_exc: Exception | None = None

    for attempt in range(len(_RETRY_DELAYS)):
        try:
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
            async with session.get(url, params=params, timeout=timeout) as resp:
                resp.raise_for_status()
                payload: dict[str, Any] = await resp.json()
                return payload
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_rate_limit_error(exc):
                log.warning("Actuals request to %s hit rate limit: %s — not retrying", url, exc)
                raise
            if attempt < len(_RETRY_DELAYS) - 1:
                await asyncio.sleep(_RETRY_DELAYS[attempt + 1])

    raise last_exc if last_exc is not None else RuntimeError("actuals request failed")


async def fetch_daily_max_actual(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    session: aiohttp.ClientSession,
) -> dict[str, float]:
    """Fetch observed daily max temperatures (°F) keyed by YYYY-MM-DD."""
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max",
        "timezone": "UTC",
        "temperature_unit": "fahrenheit",
    }
    data = await _get_with_retry(session, _ARCHIVE_BASE_URL, params)

    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    max_temps = daily.get("temperature_2m_max") or []

    result: dict[str, float] = {}
    for raw_date, raw_temp in zip(dates, max_temps):
        if raw_temp is None:
            continue
        try:
            result[str(raw_date)] = float(raw_temp)
        except (TypeError, ValueError):
            continue

    return result


def store_actual_temperatures(
    db_session: Session,
    station: str,
    daily_actuals: dict[str, float],
    station_aliases: tuple[str, ...] = (),
) -> int:
    """Write observed temperatures into Forecast.actual_temp.

    Updates all forecast-model rows for matching dates so bias/RMSE refresh jobs
    can aggregate across existing forecast entries.
    """
    if not daily_actuals:
        return 0

    stations = (station,) + tuple(s for s in station_aliases if s and s != station)
    target_dates = tuple(sorted(daily_actuals.keys()))

    rows = db_session.execute(
        select(Forecast).where(
            Forecast.station.in_(stations),
            Forecast.target_date.in_(target_dates),
            Forecast.temperature_forecast.is_not(None),
        )
    ).scalars().all()

    updated = 0
    for row in rows:
        actual = daily_actuals.get(str(row.target_date))
        if actual is None:
            continue

        if row.actual_temp is None or abs(float(row.actual_temp) - actual) > 1e-6:
            row.actual_temp = actual
            updated += 1

    return updated
