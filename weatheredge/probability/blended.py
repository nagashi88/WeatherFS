"""Blended probability engine with HRRR priority weighting — Enhancement 5.

Replaces the simple ensemble-or-gaussian fallback with a proper weighted blend
that accounts for HRRR (High-Resolution Rapid Refresh) dominance at short lead
times for US cities.

Weight scheme (with TimesFM):
  Lead <12h (US):  HRRR 40%, NBM 15%, GFS ensemble 15%, ECMWF 10%, TimesFM 20%
  Lead 12-24h (US): NBM 25%, HRRR 25%, GFS ensemble 15%, ECMWF 15%, TimesFM 20%
  Lead >24h (US):  NBM 30%, GFS ensemble 25%, ECMWF 25%, HRRR 5%, TimesFM 15%
  Non-US:          GFS ensemble 40%, ECMWF 40%, TimesFM 20%

Without TimesFM (fallback — original weights):
  Lead <12h (US):  HRRR 45%, NBM 25%, GFS ensemble 20%, ECMWF 10%
  Lead 12-24h (US): NBM 35%, HRRR 25%, GFS ensemble 25%, ECMWF 15%
  Lead >24h (US):  NBM 35%, GFS ensemble 30%, ECMWF 30%, HRRR 5%
  Non-US:          GFS ensemble 50%, ECMWF 50%

Also integrates METAR override from Enhancement 3 and TimesFM NWP error
correction (the highest-alpha integration).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from weatheredge.probability.gaussian import gaussian_probability
from weatheredge.probability.ensemble import ensemble_probability

log = logging.getLogger(__name__)

# HRRR weight tables by lead time bucket (with TimesFM)
WEIGHTS_US: dict[str, dict[str, float]] = {
    "short":  {"hrrr": 0.40, "nbm": 0.15, "gfs_ensemble": 0.15, "ecmwf": 0.10, "timesfm": 0.20},  # <12h
    "medium": {"hrrr": 0.25, "nbm": 0.25, "gfs_ensemble": 0.15, "ecmwf": 0.15, "timesfm": 0.20},  # 12-24h
    "long":   {"hrrr": 0.05, "nbm": 0.30, "gfs_ensemble": 0.25, "ecmwf": 0.25, "timesfm": 0.15},  # >24h
}

WEIGHTS_NON_US: dict[str, float] = {
    "gfs_ensemble": 0.40,
    "ecmwf": 0.40,
    "timesfm": 0.20,
}

# Fallback weights when TimesFM is unavailable (original Enhancement 5 values)
WEIGHTS_US_NO_TIMESFM: dict[str, dict[str, float]] = {
    "short":  {"hrrr": 0.45, "nbm": 0.25, "gfs_ensemble": 0.20, "ecmwf": 0.10},
    "medium": {"hrrr": 0.25, "nbm": 0.35, "gfs_ensemble": 0.25, "ecmwf": 0.15},
    "long":   {"hrrr": 0.05, "nbm": 0.35, "gfs_ensemble": 0.30, "ecmwf": 0.30},
}

WEIGHTS_NON_US_NO_TIMESFM: dict[str, float] = {
    "gfs_ensemble": 0.50,
    "ecmwf": 0.50,
}


def _get_lead_bucket(lead_time_hours: float) -> str:
    if lead_time_hours < 12:
        return "short"
    elif lead_time_hours <= 24:
        return "medium"
    else:
        return "long"


def blended_probability(
    threshold: float,
    direction: str = "above",
    rmse: float = 5.0,
    is_us_station: bool = True,
    lead_time_hours: float = 24.0,
    # GFS ensemble data
    ensemble_temps: list[float] | None = None,
    # GFS/HRRR point forecast (Open-Meteo /v1/gfs includes HRRR for US)
    gfs_forecast_temp: float | None = None,
    # ECMWF point forecast
    ecmwf_forecast_temp: float | None = None,
    # HRRR-specific forecast temp (from /v1/gfs for US locations)
    hrrr_forecast_temp: float | None = None,
    # NBM-specific forecast temp (from /v1/gfs model ncep_nbm_conus)
    nbm_forecast_temp: float | None = None,
    # METAR override probability (from CurrentConditionsTracker)
    metar_override: float | None = None,
    # Learned model weights from Brier score self-learning (C4 fix)
    learned_weights: dict[str, float] | None = None,
    # TimesFM integration
    timesfm_prob: float | None = None,
    timesfm_error_adj: float = 0.0,
) -> tuple[float, str, dict[str, Any]]:
    """Calculate blended probability with HRRR priority weighting.

    Args:
        timesfm_prob: Direct probability from TimesFM quantile outputs.
        timesfm_error_adj: NWP error correction (degrees) from TimesFM.
            Applied to all Gaussian CDF calculations before blending.

    Returns:
        (probability, model_used, details_dict)
    """
    # METAR override takes absolute precedence (Enhancement 3)
    if metar_override is not None:
        return metar_override, "metar_override", {
            "metar_probability": metar_override,
            "reason": "observed temperature already decisive",
        }

    details: dict[str, Any] = {}

    # TimesFM NWP error correction: shift forecast temperatures before probability calc.
    # This is the highest-alpha integration — adjusts for systematic NWP biases
    # that TimesFM detects from historical error patterns.
    if timesfm_error_adj != 0.0:
        details["timesfm_error_adj"] = round(timesfm_error_adj, 2)
        if gfs_forecast_temp is not None:
            gfs_forecast_temp += timesfm_error_adj
            details["gfs_temp_corrected"] = round(gfs_forecast_temp, 2)
        if ecmwf_forecast_temp is not None:
            ecmwf_forecast_temp += timesfm_error_adj
            details["ecmwf_temp_corrected"] = round(ecmwf_forecast_temp, 2)
        if hrrr_forecast_temp is not None:
            hrrr_forecast_temp += timesfm_error_adj
            details["hrrr_temp_corrected"] = round(hrrr_forecast_temp, 2)
        if nbm_forecast_temp is not None:
            nbm_forecast_temp += timesfm_error_adj
            details["nbm_temp_corrected"] = round(nbm_forecast_temp, 2)

    # Calculate individual model probabilities
    # GFS ensemble
    gfs_prob: float | None = None
    gfs_prob_raw: float | None = None
    if ensemble_temps and len(ensemble_temps) >= 20:
        # Apply error correction to ensemble members too
        if timesfm_error_adj != 0.0:
            ensemble_temps = [t + timesfm_error_adj for t in ensemble_temps]
        gfs_prob = ensemble_probability(ensemble_temps, threshold, direction)
        gfs_prob_raw = ensemble_probability(ensemble_temps, threshold, direction, clamp=False)
        details["gfs_ensemble_prob"] = round(gfs_prob, 4)
        details["gfs_ensemble_members"] = len(ensemble_temps)

    # ECMWF (Gaussian with ECMWF forecast temp)
    ecmwf_prob: float | None = None
    ecmwf_prob_raw: float | None = None
    if ecmwf_forecast_temp is not None:
        # L-IMP-3: ECMWF has ~10% lower error than GFS for daily-max; conservative tightening.
        ecmwf_rmse = rmse * 0.90
        ecmwf_prob = gaussian_probability(ecmwf_forecast_temp, threshold, ecmwf_rmse, direction)
        ecmwf_prob_raw = gaussian_probability(ecmwf_forecast_temp, threshold, ecmwf_rmse, direction, clamp=False)
        details["ecmwf_prob"] = round(ecmwf_prob, 4)
        details["ecmwf_temp"] = ecmwf_forecast_temp

    # HRRR (Gaussian with HRRR forecast temp, tighter RMSE since higher resolution)
    hrrr_prob: float | None = None
    hrrr_prob_raw: float | None = None
    if hrrr_forecast_temp is not None and is_us_station:
        # HRRR at 3km resolution has lower RMSE than GFS at 25km, but for daily-max
        # temperature the advantage shrinks — use conservative 15% tightening.
        # Phase 0 evidence: 0.70 multiplier produced overconfident P→0 outputs (Brier 0.50).
        hrrr_rmse = rmse * 0.85
        # G4: upstream (main._derive_probability) floors rmse at the PRIMARY model's
        # floor — when primary is HRRR<12h the floor is 1.5°F, so 1.5 * 0.85 = 1.275°F
        # would re-tighten σ below the floor the floor was meant to enforce. When
        # primary is GFS (long lead), the floor is 3.0°F and 3.0 * 0.85 = 2.55°F
        # sits below HRRR's own 3.0°F floor at that lead. Re-apply the HRRR floor
        # only when rmse_floor_dynamic is active (matches upstream gate, so tests
        # that don't enable it keep original behavior).
        from weatheredge.config import MODULES
        if MODULES.get("rmse_floor_dynamic", False):
            from weatheredge.bias_correction import get_rmse_floor
            hrrr_rmse = max(hrrr_rmse, get_rmse_floor("hrrr", lead_time_hours))
        hrrr_prob = gaussian_probability(hrrr_forecast_temp, threshold, hrrr_rmse, direction)
        hrrr_prob_raw = gaussian_probability(hrrr_forecast_temp, threshold, hrrr_rmse, direction, clamp=False)
        details["hrrr_prob"] = round(hrrr_prob, 4)
        details["hrrr_temp"] = hrrr_forecast_temp
        details["hrrr_rmse"] = round(hrrr_rmse, 2)

    # NBM (Gaussian with NBM forecast temp). Treat as high-resolution U.S. model
    # but do not over-tighten: until we have WeatherEdge-specific calibration,
    # use a conservative 10% RMSE tightening versus the primary model.
    nbm_prob: float | None = None
    nbm_prob_raw: float | None = None
    if nbm_forecast_temp is not None and is_us_station:
        nbm_rmse = rmse * 0.90
        nbm_prob = gaussian_probability(nbm_forecast_temp, threshold, nbm_rmse, direction)
        nbm_prob_raw = gaussian_probability(nbm_forecast_temp, threshold, nbm_rmse, direction, clamp=False)
        details["nbm_prob"] = round(nbm_prob, 4)
        details["nbm_temp"] = nbm_forecast_temp
        details["nbm_rmse"] = round(nbm_rmse, 2)

    # TimesFM probability (from quantile outputs — captures non-Gaussian tails)
    timesfm_prob_val: float | None = timesfm_prob
    if timesfm_prob_val is not None:
        details["timesfm_prob"] = round(timesfm_prob_val, 4)

    # Select weights — use TimesFM-aware tables if TimesFM data is available,
    # otherwise fall back to original weights to avoid diluting the blend.
    has_timesfm = timesfm_prob_val is not None
    if is_us_station:
        bucket = _get_lead_bucket(lead_time_hours)
        if has_timesfm:
            weights = dict(WEIGHTS_US[bucket])
        else:
            weights = dict(WEIGHTS_US_NO_TIMESFM[bucket])
        details["weight_bucket"] = bucket
    else:
        if has_timesfm:
            weights = dict(WEIGHTS_NON_US)
        else:
            weights = dict(WEIGHTS_NON_US_NO_TIMESFM)
        weights.setdefault("hrrr", 0.0)
        weights.setdefault("nbm", 0.0)
        details["weight_bucket"] = "non_us"

    details["timesfm_active"] = has_timesfm

    # Blend with learned weights if available (C4 fix: use Brier-based model weights)
    if learned_weights:
        for model_key in weights:
            if model_key in learned_weights:
                # 50/50 blend of static and learned weights for stability
                weights[model_key] = 0.5 * weights[model_key] + 0.5 * learned_weights[model_key]
        # Renormalize
        w_total = sum(weights.values())
        if w_total > 0:
            weights = {k: v / w_total for k, v in weights.items()}
        details["learned_weights_applied"] = True

    details["weights"] = weights

    # Build weighted blend from available models
    # CCG-bug-6: A1 made raw_weighted_sum identical to weighted_sum (both blend on
    # unclamped sub-probabilities), so the dual accumulator was dead code. The
    # raw-vs-clamped distinction is preserved by the single _clamp_prob call below.
    weighted_sum = 0.0
    total_weight = 0.0

    # A1: Blend on RAW (unclamped) sub-probabilities — clamping each component
    # before blending collapsed distinct tails to identical PROB_FLOOR values
    # and destroyed ranking. Single final clamp on the blended result below.
    # CCG-bug-2 (audit-2): drop non-finite sub-probabilities before adding —
    # otherwise NaN propagates into weighted_sum and _clamp_prob silently turns
    # it into PROB_CEIL (0.95), promoting a single bad model input into
    # near-certainty for the whole blend.
    if (
        gfs_prob is not None
        and weights.get("gfs_ensemble", 0) > 0
    ):
        gfs_for_blend = gfs_prob_raw if gfs_prob_raw is not None else gfs_prob
        if math.isfinite(gfs_for_blend):
            w = weights["gfs_ensemble"]
            weighted_sum += gfs_for_blend * w
            total_weight += w

    if ecmwf_prob is not None and weights.get("ecmwf", 0) > 0:
        ecmwf_for_blend = ecmwf_prob_raw if ecmwf_prob_raw is not None else ecmwf_prob
        if math.isfinite(ecmwf_for_blend):
            w = weights["ecmwf"]
            weighted_sum += ecmwf_for_blend * w
            total_weight += w

    if hrrr_prob is not None and weights.get("hrrr", 0) > 0:
        hrrr_for_blend = hrrr_prob_raw if hrrr_prob_raw is not None else hrrr_prob
        if math.isfinite(hrrr_for_blend):
            w = weights["hrrr"]
            weighted_sum += hrrr_for_blend * w
            total_weight += w

    if nbm_prob is not None and weights.get("nbm", 0) > 0:
        nbm_for_blend = nbm_prob_raw if nbm_prob_raw is not None else nbm_prob
        if math.isfinite(nbm_for_blend):
            w = weights["nbm"]
            weighted_sum += nbm_for_blend * w
            total_weight += w

    if (
        timesfm_prob_val is not None
        and weights.get("timesfm", 0) > 0
        and math.isfinite(timesfm_prob_val)
    ):
        w = weights["timesfm"]
        weighted_sum += timesfm_prob_val * w
        # TimesFM probability comes pre-shaped from upstream, no separate raw available.
        total_weight += w

    # Fallback: if no model data at all, use GFS point forecast with Gaussian
    if total_weight == 0:
        if nbm_forecast_temp is not None:
            fallback = gaussian_probability(nbm_forecast_temp, threshold, rmse, direction)
            details["fallback"] = "gaussian_nbm_point"
            return fallback, "gaussian_nbm_fallback", details
        if gfs_forecast_temp is not None:
            fallback = gaussian_probability(gfs_forecast_temp, threshold, rmse, direction)
            details["fallback"] = "gaussian_gfs_point"
            return fallback, "gaussian_fallback", details
        # Ultimate fallback: return 0.5 (no information)
        details["fallback"] = "no_data"
        return 0.50, "no_data", details

    # Normalize and clamp to [PROB_FLOOR, PROB_CEIL] safety floor/ceiling
    from weatheredge.probability.gaussian import _clamp_prob
    raw_blended = weighted_sum / total_weight  # un-clamped, used by C4 phantom-NO guard
    blended = _clamp_prob(raw_blended)
    details["weighted_sum"] = round(weighted_sum, 4)
    details["total_weight"] = round(total_weight, 4)
    details["raw_blended"] = round(raw_blended, 6)
    details["blended_probability"] = round(blended, 4)

    # Determine model label
    models_used = []
    if gfs_prob is not None:
        models_used.append("gfs_ensemble")
    if ecmwf_prob is not None:
        models_used.append("ecmwf")
    if hrrr_prob is not None:
        models_used.append("hrrr")
    if nbm_prob is not None:
        models_used.append("nbm")
    if timesfm_prob_val is not None:
        models_used.append("timesfm")
    model_label = "blended_" + "+".join(models_used)

    log.debug(
        "blended_probability: %.3f (%s) — weights=%s lead=%.0fh us=%s",
        blended, model_label, details.get("weight_bucket"), lead_time_hours, is_us_station,
    )

    return blended, model_label, details
