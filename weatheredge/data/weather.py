"""Open-Meteo weather data fetcher for WeatherEdge.

Implements fetch_forecast, fetch_ensemble, fetch_ecmwf, and fetch_nbm with retry logic
per Addendum Gap 12: 3 retries with exponential backoff (1s, 4s, 16s).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any

import aiohttp

from weatheredge.db.models import Forecast

log = logging.getLogger(__name__)

BASE_URL = "https://api.open-meteo.com"
ENSEMBLE_BASE_URL = "https://ensemble-api.open-meteo.com"

HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "precipitation_probability",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "relative_humidity_2m",
]

ENSEMBLE_VARS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
]

# Retry policy per Addendum Gap 12: 3 attempts, backoff 1s / 4s / 16s
_RETRY_COUNT = 3
_RETRY_DELAYS = [1, 4, 16]
_REQUEST_TIMEOUT_SECONDS = 15


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
    """Perform a GET request with exponential backoff retries.

    Args:
        session: An active aiohttp ClientSession.
        url: Full URL to request.
        params: Query parameters.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        aiohttp.ClientError: After all retries are exhausted.
        Exception: Any non-HTTP exception after all retries exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(_RETRY_COUNT):
        try:
            timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
            async with session.get(url, params=params, timeout=timeout) as resp:
                resp.raise_for_status()
                data: dict[str, Any] = await resp.json()
                return data
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit_error(exc):
                log.warning("Request to %s hit rate limit: %s — not retrying", url, exc)
                raise
            if attempt < _RETRY_COUNT - 1:
                delay = _RETRY_DELAYS[attempt]
                log.warning(
                    "Request to %s failed (attempt %d/%d): %s — retrying in %ds",
                    url,
                    attempt + 1,
                    _RETRY_COUNT,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                log.error(
                    "Request to %s failed after %d attempts: %s",
                    url,
                    _RETRY_COUNT,
                    exc,
                )
    raise last_exc  # type: ignore[misc]


async def fetch_forecast(
    lat: float,
    lon: float,
    session: aiohttp.ClientSession,
) -> dict[str, Any]:
    """Fetch hourly forecast data from Open-Meteo /v1/forecast.

    Uses HOURLY_VARS (temperature_2m, apparent_temperature, precipitation,
    precipitation_probability, rain, snowfall, cloud_cover, wind_speed_10m,
    wind_gusts_10m, relative_humidity_2m).

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        session: Active aiohttp ClientSession.

    Returns:
        Raw API response dict containing 'hourly' key with time-indexed arrays.
    """
    url = f"{BASE_URL}/v1/forecast"
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "temperature_unit": "fahrenheit",  # Match NOAA/ensemble units — thresholds are in °F
    }
    log.info("Fetching forecast for lat=%s lon=%s", lat, lon)
    return await _get_with_retry(session, url, params)


async def fetch_ensemble(
    lat: float,
    lon: float,
    session: aiohttp.ClientSession,
) -> dict[str, Any]:
    """Fetch GFS ensemble forecast data from Open-Meteo /v1/ensemble.

    Requests 31 GFS ensemble members with ENSEMBLE_VARS (temperature_2m,
    precipitation, wind_speed_10m).

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        session: Active aiohttp ClientSession.

    Returns:
        Raw API response dict containing 'hourly' key with ensemble member arrays.
    """
    url = f"{ENSEMBLE_BASE_URL}/v1/ensemble"
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(ENSEMBLE_VARS),
        "models": "gfs_seamless",
        "timezone": "UTC",
        "temperature_unit": "fahrenheit",  # BugFix E2: match NOAA NWS unit so ensemble_probability() compares correctly
    }
    log.info("Fetching ensemble for lat=%s lon=%s", lat, lon)
    return await _get_with_retry(session, url, params)


async def fetch_ecmwf(
    lat: float,
    lon: float,
    session: aiohttp.ClientSession,
) -> dict[str, Any]:
    """Fetch ECMWF forecast data from Open-Meteo /v1/ecmwf.

    ECMWF provides highest-accuracy global model at 9km resolution, up to 15 days.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        session: Active aiohttp ClientSession.

    Returns:
        Raw API response dict containing 'hourly' key with ECMWF forecast arrays.
    """
    url = f"{BASE_URL}/v1/ecmwf"
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
        "temperature_unit": "fahrenheit",  # BugFix E3: match forecast/ensemble units — thresholds are in °F
    }
    log.info("Fetching ECMWF for lat=%s lon=%s", lat, lon)
    return await _get_with_retry(session, url, params)


async def fetch_nbm(
    lat: float,
    lon: float,
    session: aiohttp.ClientSession,
) -> dict[str, Any]:
    """Fetch NBM Conus forecast data from Open-Meteo /v1/gfs.

    NBM is available for U.S. Conus locations as ``ncep_nbm_conus`` and updates
    hourly. It is fetched as a separate model so the blend can compare it with
    GFS/HRRR/ECMWF rather than silently replacing them.
    """
    url = f"{BASE_URL}/v1/gfs"
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "models": "ncep_nbm_conus",
        "timezone": "UTC",
        "temperature_unit": "fahrenheit",
    }
    log.info("Fetching NBM for lat=%s lon=%s", lat, lon)
    return await _get_with_retry(session, url, params)


def store_forecast(
    db_session: Any,
    station: str,
    target_date: str,
    model: str,
    forecast_data: dict[str, Any],
) -> Forecast:
    """Write a parsed forecast response to the forecasts table.

    Extracts daily max/min temperatures for target_date from hourly series. The
    legacy ``temperature_forecast`` column remains the daily max for backward
    compatibility; daily-low decisions use ``temperature_min_forecast``.
    precipitation_forecast still uses the first available hourly point.
    For ensemble data, serialises all temperature_2m values as a JSON array
    into ensemble_members.

    Args:
        db_session: A SQLAlchemy Session (from get_session context manager).
        station: Station identifier (e.g. 'KNYC').
        target_date: ISO date string 'YYYY-MM-DD' the forecast is valid for.
        model: Model name: 'forecast', 'ensemble', or 'ecmwf'.
        forecast_data: Raw API response dict from one of the fetch_* functions.

    Returns:
        The committed Forecast ORM instance.
    """
    hourly = forecast_data.get("hourly", {})

    time_values: list[str] | None = hourly.get("time")

    # Parse target date for day-aligned extraction where possible.
    try:
        target_day = datetime.strptime(target_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        target_day = None

    parsed_times: list[datetime] = []
    if time_values:
        for raw_time in time_values:
            try:
                parsed_time = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                if parsed_time.tzinfo is None:
                    parsed_time = parsed_time.replace(tzinfo=timezone.utc)
                parsed_times.append(parsed_time)
            except (AttributeError, ValueError):
                continue

    # Extract target-date max/min temperature, with first-value fallback.
    temp_values: list[float] | None = hourly.get("temperature_2m")
    target_day_temps: list[float] = []
    if target_day and temp_values and time_values:
        for raw_time, temp_val in zip(time_values, temp_values):
            if temp_val is None:
                continue
            try:
                ts = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except (AttributeError, ValueError):
                continue
            if ts.date() == target_day:
                target_day_temps.append(float(temp_val))

    if target_day_temps:
        temperature_max_forecast: float | None = max(target_day_temps)
        temperature_min_forecast: float | None = min(target_day_temps)
    else:
        fallback_temp = temp_values[0] if temp_values else None
        temperature_max_forecast = fallback_temp
        temperature_min_forecast = fallback_temp

    # Extract first available precipitation value
    precip_values: list[float] | None = hourly.get("precipitation")
    precipitation_forecast: float | None = precip_values[0] if precip_values else None

    # For ensemble model: extract per-member daily max/min for the target date.
    # BugFix E1: The old code stored the control member's 168-value hourly time series.
    # ensemble_probability() expects one value per member (e.g. daily-max), not hourly points
    # of a single member. The GFS ensemble API returns temperature_2m_member01..member30
    # as separate hourly series alongside temperature_2m (deterministic control member).
    ensemble_members_json: str | None = None
    ensemble_min_members_json: str | None = None
    if model == "ensemble":
        member_max_temps: list[float] = []
        member_min_temps: list[float] = []
        member_keys = sorted(k for k in hourly if k.startswith("temperature_2m_member"))
        for member_key in member_keys:
            member_series: list[float] | None = hourly.get(member_key)
            if not member_series:
                continue
            if target_day and time_values:
                day_temps: list[float] = []
                for raw_time, val in zip(time_values, member_series):
                    if val is None:
                        continue
                    try:
                        ts = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                    except (AttributeError, ValueError):
                        continue
                    if ts.date() == target_day:
                        day_temps.append(float(val))
                if day_temps:
                    member_max_temps.append(max(day_temps))
                    member_min_temps.append(min(day_temps))
            else:
                first = next((float(v) for v in member_series if v is not None), None)
                if first is not None:
                    member_max_temps.append(first)
                    member_min_temps.append(first)

        if member_max_temps:
            ensemble_members_json = json.dumps(member_max_temps)
            ensemble_min_members_json = json.dumps(member_min_temps) if member_min_temps else None
            # Use ensemble mean max/min for this row
            temperature_max_forecast = sum(member_max_temps) / len(member_max_temps)
            if member_min_temps:
                temperature_min_forecast = sum(member_min_temps) / len(member_min_temps)
            log.debug(
                "store_forecast: ensemble %s %s — %d members, mean_max=%.1f°F mean_min=%.1f°F",
                station,
                target_date,
                len(member_max_temps),
                temperature_max_forecast,
                temperature_min_forecast,
            )
        elif temp_values:
            # Fallback: API did not return member keys (unexpected) — store control member daily-max only
            log.warning(
                "store_forecast: ensemble %s %s — no member keys found; storing control member only",
                station, target_date,
            )
            if target_day_temps:
                ensemble_members_json = json.dumps([max(target_day_temps)])
                ensemble_min_members_json = json.dumps([min(target_day_temps)])
            else:
                ensemble_members_json = json.dumps([temp_values[0]])
                ensemble_min_members_json = json.dumps([temp_values[0]])

    # Calculate lead time from the first timestamp in the response
    lead_time_hours = 0
    if parsed_times:
        try:
            # Use earliest timestamp for target_date when available.
            first_time = min(parsed_times)
            if target_day:
                target_times = [ts for ts in parsed_times if ts.date() == target_day]
                if target_times:
                    first_time = min(target_times)
            now = datetime.now(timezone.utc)
            lead_time_hours = max(0, int((first_time - now).total_seconds() / 3600))
        except (ValueError, IndexError):
            lead_time_hours = 0

    forecast = Forecast(
        station=station,
        target_date=target_date,
        model=model,
        forecast_timestamp=datetime.now(timezone.utc),
        lead_time_hours=lead_time_hours,
        temperature_max_forecast=temperature_max_forecast,
        temperature_min_forecast=temperature_min_forecast,
        temperature_forecast=temperature_max_forecast,
        ensemble_members=ensemble_members_json,
        ensemble_min_members=ensemble_min_members_json,
        precipitation_forecast=precipitation_forecast,
    )
    db_session.add(forecast)
    return forecast
