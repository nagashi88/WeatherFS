"""Bias correction and hierarchical RMSE shrinkage — Phase 1.

Provides:
- STATION_BIAS dict: persistent in-memory store of per-station/model/month bias
- get_bias_correction(): systematic bias to subtract from forecasts
- update_station_bias(): rolling 30-day window update, min 10 samples
- get_rmse_floor(): dynamic RMSE floor by model and lead time
- get_shrunk_rmse(): hierarchical shrinkage (30% global + 70% station)

Reference: WeatherEdge PRD v3.1, Phase 1 (sections 1.1, 1.2, and 3.3).
Builds on top of STATION_RMSE in weatheredge.probability.bias_correction.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select

from weatheredge.db.database import get_session
from weatheredge.db.models import Forecast
from weatheredge.probability.bias_correction import STATION_RMSE

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# STATION_BIAS: persistent in-memory bias store
# Structure: STATION_BIAS[station][model][month] = mean(forecast - actual)
# Positive value = model runs hot = subtract from forecast.
# Negative value = model runs cold = add to forecast (subtracting a negative).
# ---------------------------------------------------------------------------
STATION_BIAS: dict[str, dict[str, dict[int, float]]] = {}

# Shrinkage: 30% weight toward global mean, 70% toward station-specific.
_SHRINKAGE_FACTOR = 0.3

# Minimum sample count before writing a bias or shrunk-RMSE estimate.
_MIN_BIAS_SAMPLES = 10

# Observation threshold below which we lean heavier on the global RMSE.
_LOW_OBS_THRESHOLD = 20


# ---------------------------------------------------------------------------
# Public API — bias correction
# ---------------------------------------------------------------------------

def get_bias_correction(station: str, model: str, month: int) -> float:
    """Return the systematic bias to subtract from the forecast.

    Positive bias means the model runs hot; subtracting it cools the forecast.
    Negative bias means the model runs cold; subtracting it warms the forecast.

    Fallback chain:
    1. Exact (station, model, month) entry.
    2. Station-level mean across all models for that month.
    3. 0.0 — no correction.
    """
    try:
        return STATION_BIAS[station][model][month]
    except KeyError:
        pass

    # Fall back to station-level mean across models for this month
    try:
        all_model_biases = [
            STATION_BIAS[station][m][month]
            for m in STATION_BIAS.get(station, {})
            if month in STATION_BIAS[station].get(m, {})
        ]
        if all_model_biases:
            return sum(all_model_biases) / len(all_model_biases)
    except Exception:
        pass

    return 0.0


def update_station_bias(
    station: str,
    model: str,
    month: int,
    forecasts: list[float],
    actuals: list[float],
) -> None:
    """Update the in-memory bias for a (station, model, month) tuple.

    Uses a rolling 30-day window of residuals (forecast - actual).
    Writes only when at least 10 samples are available.

    Args:
        station: Station identifier, e.g. "KNYC".
        model:   Model name, e.g. "gfs", "ecmwf", "hrrr".
        month:   Calendar month as integer 1–12.
        forecasts: Sequence of forecast temperatures (°F).
        actuals:   Sequence of observed temperatures (°F), same length.
    """
    residuals = [f - a for f, a in zip(forecasts, actuals)]
    if len(residuals) < _MIN_BIAS_SAMPLES:
        log.debug(
            "Bias update skipped for %s/%s/month%d: only %d samples (need %d)",
            station, model, month, len(residuals), _MIN_BIAS_SAMPLES,
        )
        return

    bias = sum(residuals) / len(residuals)
    STATION_BIAS.setdefault(station, {}).setdefault(model, {})[month] = bias
    log.info(
        "Bias update: %s/%s/month%d = %+.2f°F (n=%d)",
        station, model, month, bias, len(residuals),
    )


# ---------------------------------------------------------------------------
# Public API — RMSE floor
# ---------------------------------------------------------------------------

def get_rmse_floor(model: str, lead_hours: float) -> float:
    """Return the dynamic RMSE floor for a given model and lead time.

    HRRR at short ranges is much tighter than coarser global models:
    - HRRR < 12 h  → 1.5°F  (3 km resolution, near-term skill)
    - HRRR < 24 h  → 2.0°F  (degrading but still better than global)
    - All other    → 3.0°F  (GFS, ECMWF, long lead times)
    """
    if model == "hrrr":
        if lead_hours < 12:
            return 1.5
        if lead_hours < 24:
            return 2.0
    return 3.0


# ---------------------------------------------------------------------------
# Public API — hierarchical shrinkage
# ---------------------------------------------------------------------------

def get_shrunk_rmse(
    station: str,
    model: str,
    lead_days: int,
    station_rmse_dict: Optional[dict] = None,
) -> float:
    """Return a shrinkage-adjusted RMSE for a station/model/lead-days combo.

    Hierarchical shrinkage pulls the station-specific RMSE toward the global
    mean to prevent overfitting when few observations exist.

    With >= 20 observations:  shrunk = 0.30 * global + 0.70 * station
    With < 20 observations:   shrinkage weight on global increases linearly
                               toward 1.0 as n → 0.

    The result is floored by get_rmse_floor() to prevent implausibly tight
    uncertainty estimates.

    Args:
        station:           Station identifier, e.g. "KNYC".
        model:             Model name, e.g. "gfs", "ecmwf", "hrrr".
        lead_days:         Forecast lead time in days (integer).
        station_rmse_dict: Optional override for the station RMSE lookup.
                           Defaults to STATION_RMSE from
                           weatheredge.probability.bias_correction.
    """
    rmse_source = station_rmse_dict if station_rmse_dict is not None else STATION_RMSE
    station_rmse: Optional[float] = (
        rmse_source.get(station, {}).get(model, {}).get(lead_days)
    )
    global_rmse = _get_global_mean_rmse(model, lead_days)

    if station_rmse is None:
        # No station data — return global, respecting the floor
        floor = get_rmse_floor(model, lead_days * 24)
        return max(global_rmse, floor)

    n_obs = _get_observation_count(station, model, lead_days)

    if n_obs < _LOW_OBS_THRESHOLD:
        # Few observations: lean heavier on the global mean
        # shrinkage rises linearly from _SHRINKAGE_FACTOR (at n=20) toward 1.0 (at n=0)
        shrinkage = _SHRINKAGE_FACTOR + (1 - _SHRINKAGE_FACTOR) * (1 - n_obs / _LOW_OBS_THRESHOLD)
    else:
        shrinkage = _SHRINKAGE_FACTOR

    shrunk = shrinkage * global_rmse + (1 - shrinkage) * station_rmse
    floor = get_rmse_floor(model, lead_days * 24)
    return max(shrunk, floor)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_global_mean_rmse(model: str, lead_days: int) -> float:
    """Compute the mean RMSE across all stations for a given model/lead_days.

    Falls back to a conservative default if no data is available.
    """
    values: list[float] = []
    for station_data in STATION_RMSE.values():
        model_data = station_data.get(model, {})
        rmse = model_data.get(lead_days)
        if rmse is not None:
            values.append(rmse)

    if not values:
        # Conservative fallback: use the floor for this model at this lead
        return get_rmse_floor(model, lead_days * 24)

    return sum(values) / len(values)


def _get_observation_count(station: str, model: str, lead_days: int) -> int:
    """Return the number of verified forecasts (with actuals) for a given key.

    Queries the Forecast table for rows where:
    - station and model match
    - actual_temp is not null (verified)
    - lead_time_hours falls within the lead_days bucket

    Returns 0 if the database is unreachable or the table is empty.
    """
    lead_hours_min = (lead_days - 1) * 24
    lead_hours_max = lead_days * 24

    try:
        with get_session() as session:
            count = session.execute(
                select(Forecast.id).where(
                    Forecast.station == station,
                    Forecast.model == model,
                    Forecast.actual_temp.is_not(None),
                    Forecast.lead_time_hours > lead_hours_min,
                    Forecast.lead_time_hours <= lead_hours_max,
                )
            ).fetchall()
        return len(count)
    except Exception as exc:
        log.warning(
            "Could not query observation count for %s/%s/lead_days=%d: %s",
            station, model, lead_days, exc,
        )
        return 0
