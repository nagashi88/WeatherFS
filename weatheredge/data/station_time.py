"""Station timezone helpers for market close calculations."""

from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

# Canonical IANA timezones per resolution station.
_STATION_TIMEZONES: dict[str, str] = {
    "KNYC": "America/New_York",
    "KMDW": "America/Chicago",
    "KMIA": "America/New_York",
    "KDEN": "America/Denver",
    "KLAX": "America/Los_Angeles",
    "KAUS": "America/Chicago",
    "KSEA": "America/Los_Angeles",
    "KATL": "America/New_York",
    "KDFW": "America/Chicago",
    "KPHX": "America/Phoenix",
    "KIAH": "America/Chicago",
    "KSFO": "America/Los_Angeles",
    "KBOS": "America/New_York",
    "EGLC": "Europe/London",
    "SAEZ": "America/Argentina/Buenos_Aires",
    "LTAC": "Europe/Istanbul",
    "RKSS": "Asia/Seoul",
}


def station_timezone_name(station: str) -> str | None:
    """Return IANA timezone name for a station code."""
    return _STATION_TIMEZONES.get((station or "").upper())


def station_local_today_str(
    station: str,
    now_utc: datetime | None = None,
) -> str:
    """Return the current date in the station's local timezone as YYYY-MM-DD.

    Falls back to the UTC date when the station's timezone is unknown.
    Used to align market_date (station-local) with live observations whose
    wall-clock labeling differs from UTC during the overnight window.
    """
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    tz_name = station_timezone_name(station)
    if tz_name is None:
        return now.astimezone(timezone.utc).strftime("%Y-%m-%d")
    return now.astimezone(ZoneInfo(tz_name)).strftime("%Y-%m-%d")


def market_close_time_utc(market_date: str, station: str) -> datetime | None:
    """Return UTC timestamp for local end-of-day (23:59) market close."""
    tz_name = station_timezone_name(station)
    if tz_name is None:
        return None

    try:
        market_day = datetime.strptime(market_date, "%Y-%m-%d").date()
        local_tz = ZoneInfo(tz_name)
        local_close = datetime.combine(market_day, time(23, 59), tzinfo=local_tz)
        return local_close.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def hours_until_market_close(
    market_date: str,
    station: str,
    now_utc: datetime | None = None,
) -> float | None:
    """Return hours until local-day close for a station/date."""
    close_utc = market_close_time_utc(market_date, station)
    if close_utc is None:
        return None

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    return (close_utc - now).total_seconds() / 3600.0
