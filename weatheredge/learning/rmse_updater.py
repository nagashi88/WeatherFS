from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from weatheredge.db.database import get_session
from weatheredge.db.models import Calibration, Forecast
from weatheredge.probability import bias_correction

log = logging.getLogger(__name__)

# Forecast.station historically used city names; keep aliases for back-compat.
_STATION_TO_CITY: dict[str, str] = {
    "KNYC": "NYC",
    "KMDW": "Chicago",
    "KMIA": "Miami",
    "KDEN": "Denver",
    "KLAX": "LAX",
    "KAUS": "Austin",
    "KSEA": "Seattle",
    "KATL": "Atlanta",
    "KDFW": "Dallas",
    "KPHX": "Phoenix",
    "KIAH": "Houston",
    "KSFO": "San Francisco",
    "KBOS": "Boston",
    "EGLC": "London",
    "SAEZ": "Buenos Aires",
    "LTAC": "Ankara",
    "RKSS": "Seoul",
}

# Open-Meteo deterministic feed is stored as model="forecast" and is used as GFS proxy.
_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "gfs": ("gfs", "forecast"),
    "hrrr": ("hrrr", "forecast"),
    "ecmwf": ("ecmwf",),
}


def _station_aliases(station: str) -> tuple[str, ...]:
    city = _STATION_TO_CITY.get(station)
    if city:
        return station, city
    return (station,)


def _model_aliases(model: str) -> tuple[str, ...]:
    return _MODEL_ALIASES.get(model, (model,))


class RMSEUpdater:
    """Recalculates station RMSE from forecast-vs-actual data. Runs monthly."""

    def __init__(self, engine):
        self.engine = engine

    def _load_pairs(self, station: str, model: str, days: int) -> list[tuple[float, float]]:
        """Load (forecast, actual) pairs using model/station aliases."""
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        station_values = _station_aliases(station)
        model_values = _model_aliases(model)

        with get_session(self.engine) as session:
            rows = session.execute(
                select(Forecast.temperature_forecast, Forecast.actual_temp).where(
                    Forecast.station.in_(station_values),
                    Forecast.model.in_(model_values),
                    Forecast.actual_temp.is_not(None),
                    Forecast.temperature_forecast.is_not(None),
                    Forecast.target_date >= cutoff_date,
                )
            ).all()

        return [(float(forecast), float(actual)) for forecast, actual in rows]

    def _upsert_calibration_row(
        self,
        station: str,
        model: str,
        rmse: float,
        sample_size: int,
        days: int,
    ) -> None:
        """Persist the latest RMSE snapshot for dashboard/status visibility."""
        now_utc = datetime.now(timezone.utc)
        period_end = now_utc.date().isoformat()
        period_start = (now_utc - timedelta(days=days)).date().isoformat()

        with get_session(self.engine) as session:
            row = session.execute(
                select(Calibration).where(
                    Calibration.station == station,
                    Calibration.model == model,
                    Calibration.lead_time_hours == 24,
                )
            ).scalar_one_or_none()

            if row is None:
                row = Calibration(
                    station=station,
                    model=model,
                    lead_time_hours=24,
                )
                session.add(row)

            row.rmse = rmse
            row.sample_size = sample_size
            row.period_start = period_start
            row.period_end = period_end
            row.updated_at = now_utc

    def recalculate_rmse(self, station: str, model: str, days: int = 30) -> float | None:
        """
        Compute RMSE from last N days of forecasts that have actual outcomes.
        RMSE = sqrt(mean((forecast - actual)^2))
        """
        rows = self._load_pairs(station, model, days)

        if not rows:
            log.warning(
                "No forecast-vs-actual data for %s/%s in last %d days",
                station,
                model,
                days,
            )
            return None

        squared_errors = [(forecast_temp - actual_temp) ** 2 for forecast_temp, actual_temp in rows]
        rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

        log.info("RMSE %s/%s: %.3f (from %d samples)", station, model, rmse, len(rows))
        return rmse

    def update_all_rmse(self, days: int = 30):
        """Recalculate RMSE for all station/model combos and update bias_correction.STATION_RMSE."""
        for station, models in bias_correction.STATION_RMSE.items():
            for model_name in models:
                rows = self._load_pairs(station, model_name, days)
                if not rows:
                    log.warning(
                        "No forecast-vs-actual data for %s/%s in last %d days",
                        station,
                        model_name,
                        days,
                    )
                    continue

                squared_errors = [
                    (forecast_temp - actual_temp) ** 2
                    for forecast_temp, actual_temp in rows
                ]
                rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
                rmse = max(rmse, 3.0)  # avoid overconfident probabilities

                # Update in-memory runtime lookup used by get_rmse().
                bias_correction.STATION_RMSE[station][model_name][1] = rmse
                log.info("Updated STATION_RMSE[%s][%s][1] = %.3f", station, model_name, rmse)

                # Persist for visibility in DB/dashboard.
                self._upsert_calibration_row(
                    station=station,
                    model=model_name,
                    rmse=rmse,
                    sample_size=len(rows),
                    days=days,
                )
