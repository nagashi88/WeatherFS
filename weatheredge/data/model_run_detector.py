"""GFS Model Run Detector — Enhancement 4.

Detects when new GFS model runs become available by polling Open-Meteo
and comparing forecast values against cached previous run. Triggers
immediate decision cycles on significant delta.

GFS runs: 00Z, 06Z, 12Z, 18Z UTC — available ~3.5h after initialization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

import aiohttp

log = logging.getLogger(__name__)

# Expected availability windows for GFS runs (UTC hours)
GFS_RUN_WINDOWS: list[tuple[int, int, str]] = [
    # (start_hour, end_hour, run_label)
    (3, 5, "00Z"),    # 00Z run available 03:00-05:00 UTC
    (9, 11, "06Z"),   # 06Z run available 09:00-10:30 UTC
    (15, 17, "12Z"),  # 12Z run available 15:00-16:30 UTC
    (21, 23, "18Z"),  # 18Z run available 21:00-22:30 UTC
]

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Temperature delta threshold (°F) to trigger an immediate decision cycle
DELTA_THRESHOLD_F = 0.5


@dataclass
class ModelRunEvent:
    """Describes a detected new model run."""
    run_label: str           # e.g., "12Z"
    detected_at: datetime
    city: str
    delta_f: float           # Temperature shift in °F
    old_temp_f: float
    new_temp_f: float


@dataclass
class CityForecastCache:
    """Cached forecast for a single city from the previous model run."""
    city: str
    lat: float
    lon: float
    temperature_f: float | None = None
    last_check: datetime | None = None
    last_model_run_detected: str = ""


class ModelRunDetector:
    """Detects new GFS model runs by monitoring forecast shifts.

    Polls Open-Meteo during expected availability windows. When a temperature
    delta > DELTA_THRESHOLD_F is detected for any target city, fires a callback
    to trigger an immediate decision cycle.
    """

    def __init__(
        self,
        city_coords: dict[str, tuple[float, float]],
        on_new_run: Callable[[ModelRunEvent], Awaitable[None]] | None = None,
    ) -> None:
        """
        Args:
            city_coords: Mapping of city name -> (lat, lon).
            on_new_run: Async callback fired when a new model run is detected.
        """
        self._city_coords = city_coords
        self._on_new_run = on_new_run
        self._cache: dict[str, CityForecastCache] = {}
        self._detection_log: list[ModelRunEvent] = []

        # Initialize cache entries
        for city, (lat, lon) in city_coords.items():
            self._cache[city] = CityForecastCache(city=city, lat=lat, lon=lon)

    @property
    def detection_log(self) -> list[ModelRunEvent]:
        return list(self._detection_log)

    def is_in_check_window(self, now: datetime | None = None) -> tuple[bool, str]:
        """Check if current time falls within a GFS run availability window.

        Returns:
            (is_in_window, run_label) — e.g. (True, "12Z")
        """
        if now is None:
            now = datetime.now(timezone.utc)
        hour = now.hour
        for start_h, end_h, label in GFS_RUN_WINDOWS:
            if start_h <= hour < end_h:
                return True, label
        return False, ""

    async def check_for_new_run(
        self,
        session: aiohttp.ClientSession | None = None,
    ) -> list[ModelRunEvent]:
        """Poll Open-Meteo for all cities and detect forecast shifts.

        Should be called every 90 seconds during expected availability windows.

        Returns:
            List of ModelRunEvent for cities where delta > threshold.
        """
        now = datetime.now(timezone.utc)
        in_window, run_label = self.is_in_check_window(now)
        if not in_window:
            return []

        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession()

        events: list[ModelRunEvent] = []
        try:
            for city, cache in self._cache.items():
                # Skip if we already detected this run for this city
                if cache.last_model_run_detected == run_label:
                    continue

                try:
                    new_temp_f = await self._fetch_forecast_temp(
                        cache.lat, cache.lon, session
                    )
                except Exception as exc:
                    log.debug("ModelRunDetector: fetch failed for %s: %s", city, exc)
                    continue

                if new_temp_f is None:
                    continue

                old_temp_f = cache.temperature_f

                if old_temp_f is not None:
                    delta = abs(new_temp_f - old_temp_f)
                    if delta >= DELTA_THRESHOLD_F:
                        event = ModelRunEvent(
                            run_label=run_label,
                            detected_at=now,
                            city=city,
                            delta_f=round(new_temp_f - old_temp_f, 1),
                            old_temp_f=old_temp_f,
                            new_temp_f=new_temp_f,
                        )
                        events.append(event)
                        self._detection_log.append(event)
                        cache.last_model_run_detected = run_label
                        log.info(
                            "NEW GFS RUN DETECTED [%s]: %s shifted %.1f°F (%.1f→%.1f)",
                            run_label, city, event.delta_f,
                            old_temp_f, new_temp_f,
                        )

                # Update cache regardless
                cache.temperature_f = new_temp_f
                cache.last_check = now

        finally:
            if own_session:
                await session.close()

        # Fire callback for each event
        if events and self._on_new_run:
            for event in events:
                try:
                    await self._on_new_run(event)
                except Exception as exc:
                    log.warning("ModelRunDetector: callback failed for %s: %s", event.city, exc)

        return events

    async def _fetch_forecast_temp(
        self,
        lat: float,
        lon: float,
        session: aiohttp.ClientSession,
    ) -> float | None:
        """Fetch current temperature_2m forecast from Open-Meteo.

        Returns temperature in °F, or None on failure.
        """
        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m",
            "timezone": "UTC",
            "forecast_days": 1,
            "temperature_unit": "fahrenheit",
        }
        async with session.get(
            OPEN_METEO_FORECAST_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()

        temps = data.get("hourly", {}).get("temperature_2m", [])
        if not temps:
            return None

        # Return the max temperature in the first 24h as proxy for daily high
        valid_temps = [t for t in temps if t is not None]
        return max(valid_temps) if valid_temps else None

    def arm_status(self) -> dict[str, Any]:
        """Return current detector status for reporting."""
        now = datetime.now(timezone.utc)
        in_window, run_label = self.is_in_check_window(now)

        # Find next window
        hour = now.hour
        next_window = ""
        for start_h, _, label in GFS_RUN_WINDOWS:
            if start_h > hour:
                next_window = f"{label} (starts {start_h:02d}:00 UTC)"
                break
        if not next_window:
            next_window = f"00Z (starts 03:00 UTC tomorrow)"

        return {
            "armed": True,
            "in_check_window": in_window,
            "current_window": run_label if in_window else None,
            "next_window": next_window,
            "total_detections": len(self._detection_log),
            "cached_cities": {
                city: {
                    "temp_f": c.temperature_f,
                    "last_check": c.last_check.isoformat() if c.last_check else None,
                    "last_run": c.last_model_run_detected or "none",
                }
                for city, c in self._cache.items()
            },
        }
