from __future__ import annotations

from typing import Any

from weatheredge.probability.gaussian import gaussian_probability
from weatheredge.probability.ensemble import ensemble_probability


def calculate_probability(
    forecast_temp: float,
    threshold: float,
    rmse: float,
    ensemble_temps: list[float] | None = None,
    direction: str = "above",
) -> tuple[float, str]:
    """
    Returns (probability, model_used).

    Priority (per Addendum Gap 3):
    1. Ensemble counting if >=20 members available
    2. Gaussian CDF as fallback

    For HRRR-weighted blended probability, use calculate_blended_probability().
    """
    if ensemble_temps and len(ensemble_temps) >= 20:
        prob = ensemble_probability(ensemble_temps, threshold, direction)
        return prob, "ensemble"
    else:
        prob = gaussian_probability(forecast_temp, threshold, rmse, direction)
        return prob, "gaussian_fallback"


def calculate_blended_probability(
    threshold: float,
    direction: str = "above",
    rmse: float = 5.0,
    is_us_station: bool = True,
    lead_time_hours: float = 24.0,
    ensemble_temps: list[float] | None = None,
    gfs_forecast_temp: float | None = None,
    ecmwf_forecast_temp: float | None = None,
    hrrr_forecast_temp: float | None = None,
    metar_override: float | None = None,
    learned_weights: dict[str, float] | None = None,
    timesfm_prob: float | None = None,
    timesfm_error_adj: float = 0.0,
) -> tuple[float, str, dict[str, Any]]:
    """HRRR-weighted blended probability (Enhancement 5) + TimesFM integration.

    Delegates to weatheredge.probability.blended.blended_probability.
    Falls back to simple calculate_probability if blended module unavailable.
    """
    from weatheredge.probability.blended import blended_probability
    return blended_probability(
        threshold=threshold,
        direction=direction,
        rmse=rmse,
        is_us_station=is_us_station,
        lead_time_hours=lead_time_hours,
        ensemble_temps=ensemble_temps,
        gfs_forecast_temp=gfs_forecast_temp,
        ecmwf_forecast_temp=ecmwf_forecast_temp,
        hrrr_forecast_temp=hrrr_forecast_temp,
        metar_override=metar_override,
        learned_weights=learned_weights,
        timesfm_prob=timesfm_prob,
        timesfm_error_adj=timesfm_error_adj,
    )
