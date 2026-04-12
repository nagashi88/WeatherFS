"""Real-time METAR current conditions tracker.

Enhancement 3: Polls aviationweather.gov for live METAR observations at
resolution stations. Tracks running daily high/low and provides probability
overrides when observed temperature already exceeds/precludes thresholds.

No API key required. Updated every hour (or on significant change).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

log = logging.getLogger(__name__)

METAR_API_URL = "https://aviationweather.gov/api/data/metar"

# IEM archive for historical ASOS (10-minute ingest)
IEM_BASE_URL = "https://mesonet.agron.iastate.edu/request/download.phtml"


@dataclass
class StationObservation:
    """Latest observation for a single station."""
    station: str
    temperature_c: float | None = None
    temperature_f: float | None = None
    dewpoint_c: float | None = None
    wind_speed_kt: float | None = None
    wind_dir: int | None = None
    altimeter_hpa: float | None = None
    observation_time: datetime | None = None
    raw_metar: str = ""


# Station timezone names for DST-aware local hour calculations
_STATION_TZ: dict[str, str] = {
    "KNYC": "America/New_York", "KMDW": "America/Chicago",
    "KMIA": "America/New_York", "KDEN": "America/Denver",
    "KLAX": "America/Los_Angeles", "KAUS": "America/Chicago",
    "KSEA": "America/Los_Angeles", "KATL": "America/New_York",
    "KDFW": "America/Chicago", "KPHX": "America/Phoenix",
    "KIAH": "America/Chicago", "KSFO": "America/Los_Angeles",
    "KBOS": "America/New_York",
    "EGLC": "Europe/London", "SAEZ": "America/Argentina/Buenos_Aires",
    "LTAC": "Europe/Istanbul", "RKSS": "Asia/Seoul",
}

# Fallback static offsets (standard time) if zoneinfo is unavailable
_STATION_UTC_OFFSET_HOURS_FALLBACK: dict[str, int] = {
    "KNYC": -5, "KMDW": -6, "KMIA": -5, "KDEN": -7,
    "KLAX": -8, "KAUS": -6, "KSEA": -8, "KATL": -5,
    "KDFW": -6, "KPHX": -7, "KIAH": -6, "KSFO": -8, "KBOS": -5,
    "EGLC": 0, "SAEZ": -3, "LTAC": 3, "RKSS": 9,
}


def _station_local_hour(station: str) -> int:
    """Get current local hour for a station, respecting DST."""
    tz_name = _STATION_TZ.get(station)
    if tz_name:
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo(tz_name))
            return now.hour
        except Exception:
            pass
    # Fallback to static offset
    offset = _STATION_UTC_OFFSET_HOURS_FALLBACK.get(station, 0)
    return (datetime.now(timezone.utc).hour + offset) % 24


@dataclass
class DailyTracker:
    """Tracks running daily high/low from actual observations for one station."""
    station: str
    date: str  # YYYY-MM-DD
    daily_high_f: float = -999.0
    daily_low_f: float = 999.0
    observation_count: int = 0
    last_observation: StationObservation | None = None
    prev_observation: StationObservation | None = None  # second-to-last obs for trend detection

    def update(self, obs: StationObservation) -> None:
        """Update running daily high/low with new observation."""
        if obs.temperature_f is None:
            return
        if obs.temperature_f > self.daily_high_f:
            self.daily_high_f = obs.temperature_f
        if obs.temperature_f < self.daily_low_f:
            self.daily_low_f = obs.temperature_f
        self.observation_count += 1
        self.prev_observation = self.last_observation
        self.last_observation = obs


class CurrentConditionsTracker:
    """Polls METAR for target stations and tracks running daily high/low.

    The key edge: if observed temp already exceeds a market threshold,
    the probability is near 1.0 regardless of any forecast model.
    """

    def __init__(self) -> None:
        # station -> DailyTracker (reset daily)
        self._trackers: dict[str, DailyTracker] = {}

    def get_tracker(self, station: str) -> DailyTracker | None:
        """Return the daily tracker for a station, or None if no data yet."""
        return self._trackers.get(station)

    async def poll_station(
        self,
        station: str,
        session: aiohttp.ClientSession | None = None,
    ) -> StationObservation | None:
        """Fetch latest METAR for a station from aviationweather.gov.

        Args:
            station: ICAO station code (e.g., 'KNYC').
            session: Optional aiohttp session (creates one if None).

        Returns:
            StationObservation or None on failure.
        """
        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession()

        try:
            params = {"ids": station, "format": "json"}
            async with session.get(METAR_API_URL, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    log.warning("METAR API returned %d for %s", resp.status, station)
                    return None
                data = await resp.json()

            if not data:
                log.debug("METAR: no data returned for %s", station)
                return None

            # API returns a list; take the first (most recent) observation
            obs_data = data[0] if isinstance(data, list) else data

            temp_c = obs_data.get("temp")
            temp_f = (temp_c * 9 / 5 + 32) if temp_c is not None else None
            dewpoint_c = obs_data.get("dewp")

            obs_time_str = obs_data.get("reportTime") or obs_data.get("obsTime")
            obs_time = None
            if obs_time_str:
                try:
                    obs_time = datetime.fromisoformat(obs_time_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            obs = StationObservation(
                station=station,
                temperature_c=temp_c,
                temperature_f=round(temp_f, 1) if temp_f is not None else None,
                dewpoint_c=dewpoint_c,
                wind_speed_kt=obs_data.get("wspd"),
                wind_dir=obs_data.get("wdir"),
                altimeter_hpa=obs_data.get("altim"),
                observation_time=obs_time,
                raw_metar=obs_data.get("rawOb", ""),
            )

            # Update daily tracker
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            tracker = self._trackers.get(station)
            if tracker is None or tracker.date != today:
                tracker = DailyTracker(station=station, date=today)
                self._trackers[station] = tracker
            tracker.update(obs)

            log.info(
                "METAR %s: %.1f°F (high=%.1f low=%.1f, obs#%d)",
                station,
                obs.temperature_f or 0,
                tracker.daily_high_f,
                tracker.daily_low_f,
                tracker.observation_count,
            )
            return obs

        except Exception as exc:
            log.warning("METAR fetch failed for %s: %s", station, exc)
            return None
        finally:
            if own_session:
                await session.close()

    async def poll_all_stations(
        self,
        stations: list[str],
        session: aiohttp.ClientSession | None = None,
    ) -> dict[str, StationObservation | None]:
        """Poll METAR for multiple stations.

        Returns:
            Dict of station -> StationObservation (or None on failure).
        """
        results: dict[str, StationObservation | None] = {}
        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession()
        try:
            for station in stations:
                results[station] = await self.poll_station(station, session)
        finally:
            if own_session:
                await session.close()
        return results

    def metar_probability_override(
        self,
        station: str,
        threshold: float,
        direction: str = "above",
        market_type: str = "daily_high",
        market_date: str | None = None,
    ) -> float | None:
        """Check if METAR observation already resolves the market.

        Rules (per strategic review):
        - Future dates (market_date != today): NEVER override — use forecast models only.
        - "above" market: override to 0.99 ONLY if observed daily high already >= threshold.
        - "below" market: override to 0.01 if observed daily high already >= threshold (proven NO).
          Override to 0.99 only if ALL three conditions met:
            a) Local time is after 17:00 at the station
            b) Current temp is falling (last obs cooler than previous obs)
            c) Current temp is at least 5°F below threshold
          Otherwise return None and let forecast models decide.

        Returns:
            Override probability (0.01 or 0.99) if METAR is decisive, else None.
        """
        # Never override for future-date markets
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if market_date is not None and market_date != today:
            return None

        tracker = self._trackers.get(station)
        if tracker is None or tracker.last_observation is None:
            return None

        obs = tracker.last_observation
        if obs.temperature_f is None:
            return None

        if market_type == "daily_high" and direction == "above":
            # Correct override: observed daily HIGH already exceeds threshold → YES is certain
            if tracker.daily_high_f >= threshold:
                log.info(
                    "METAR OVERRIDE %s: daily high %.1f°F >= threshold %.1f°F → 0.99",
                    station, tracker.daily_high_f, threshold,
                )
                return 0.99
            # Otherwise use forecast models — current temp says nothing about afternoon high
            return None

        elif market_type == "daily_high" and direction == "below":
            # Unambiguous NO: daily high already exceeded threshold
            if tracker.daily_high_f >= threshold:
                log.info(
                    "METAR OVERRIDE %s: high %.1f°F >= threshold %.1f°F → below is 0.01",
                    station, tracker.daily_high_f, threshold,
                )
                return 0.01

            # Override to 0.99 (YES for "below") only if all three conditions are satisfied:
            # a) Local time past 17:00 at the station (DST-aware)
            local_hour = _station_local_hour(station)
            if local_hour < 17:
                return None

            # b) Temperature is falling (current obs cooler than previous)
            if tracker.prev_observation is None or tracker.prev_observation.temperature_f is None:
                return None
            if obs.temperature_f >= tracker.prev_observation.temperature_f:
                return None  # Not falling

            # c) Current temp at least 5°F below threshold
            if obs.temperature_f > threshold - 5.0:
                return None

            log.info(
                "METAR OVERRIDE %s: after 17:00 local, temp falling %.1f→%.1f°F, "
                "%.1f°F below threshold %.1f°F → below is 0.99",
                station,
                tracker.prev_observation.temperature_f,
                obs.temperature_f,
                threshold - obs.temperature_f,
                threshold,
            )
            return 0.99

        # No decisive override for other market types
        return None
