"""TimesFM integration for WeatherEdge.

Three alpha-generating capabilities:

1. **Quantile Probability** — P(T > threshold) directly from TimesFM's quantile
   outputs, capturing non-Gaussian tails (fat tails, skewness) that the current
   Gaussian CDF misses.

2. **NWP Error Correction** — Predicts systematic NWP forecast errors by
   forecasting the (actual - NWP_forecast) residual time series, then shifts the
   Gaussian CDF mean.  This is the highest-alpha integration: it turns a generic
   NWP forecast into a locally-tuned one.

3. **Raw Temperature Forecast** — Zero-shot temperature forecast from historical
   station observations, used as an independent model in the blended probability
   engine.

The model is loaded lazily on first use and cached for the process lifetime.
Inference for a single station takes ~100-300ms on CPU (200M params).

API: timesfm 1.3.0 uses TimesFm + TimesFmHparams + TimesFmCheckpoint.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Singleton model instance, protected by lock for thread safety
_model = None
_model_lock = threading.Lock()
_load_failed = False

# TimesFM quantile output layout (10 columns):
#   0=mean, 1=p10, 2=p20, 3=p30, 4=p40, 5=p50(median), 6=p60, 7=p70, 8=p80, 9=p90
_Q_MEAN = 0
_Q_P10 = 1
_Q_P20 = 2
_Q_P30 = 3
_Q_P40 = 4
_Q_P50 = 5  # median = point forecast
_Q_P60 = 6
_Q_P70 = 7
_Q_P80 = 8
_Q_P90 = 9


def _load_model(model_id: str = "google/timesfm-1.0-200m-pytorch", context_length: int = 512):
    """Load TimesFM model (lazy, thread-safe singleton).

    Returns the loaded model or None if loading fails.
    """
    global _model, _load_failed

    if _load_failed:
        return None

    with _model_lock:
        if _model is not None:
            return _model
        if _load_failed:
            return None

        try:
            import timesfm

            tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    context_len=context_length,
                    horizon_len=128,
                    backend="cpu",
                    per_core_batch_size=32,
                    quantiles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=model_id,
                ),
            )
            _model = tfm
            log.info("TimesFM model loaded: %s (context=%d)", model_id, context_length)
            return _model
        except Exception as exc:
            _load_failed = True
            log.warning("TimesFM model failed to load: %s — TimesFM features disabled", exc)
            return None


def _ensure_model(model_id: str, context_length: int):
    """Get model or raise RuntimeError."""
    model = _load_model(model_id, context_length)
    if model is None:
        raise RuntimeError("TimesFM model unavailable")
    return model


def forecast_temperature(
    historical_temps: list[float],
    horizon: int = 24,
    model_id: str = "google/timesfm-1.0-200m-pytorch",
    context_length: int = 512,
) -> dict[str, Any]:
    """Zero-shot temperature forecast from historical hourly observations.

    Args:
        historical_temps: Hourly temperature observations (most recent last).
        horizon: Number of hours to forecast ahead.
        model_id: HuggingFace model identifier.
        context_length: Max context window.

    Returns:
        dict with keys:
            point_forecast: np.ndarray of shape (horizon,)
            quantiles: np.ndarray of shape (horizon, 10) — quantile forecasts
            forecast_max: float — predicted daily high (max of point forecast)
            quantile_p90_max: float — 90th percentile daily high
            quantile_p10_max: float — 10th percentile daily high
    """
    model = _ensure_model(model_id, context_length)

    context = np.array(historical_temps[-context_length:], dtype=np.float64)
    # freq=[0] = high frequency (hourly/sub-daily)
    point, quantiles = model.forecast([context], freq=[0])

    point_forecast = point[0][:horizon]  # shape (horizon,) — trim to requested horizon
    quantile_forecast = quantiles[0][:horizon]  # shape (horizon, 10)

    return {
        "point_forecast": point_forecast,
        "quantiles": quantile_forecast,
        "forecast_max": float(np.max(point_forecast)),
        "quantile_p90_max": float(np.max(quantile_forecast[:, _Q_P90])),
        "quantile_p10_max": float(np.max(quantile_forecast[:, _Q_P10])),
        "forecast_median": float(np.max(quantile_forecast[:, _Q_P50])),
    }


def quantile_probability(
    historical_temps: list[float],
    threshold: float,
    direction: str = "above",
    horizon: int = 24,
    model_id: str = "google/timesfm-1.0-200m-pytorch",
    context_length: int = 512,
) -> tuple[float, dict[str, Any]]:
    """Estimate P(T > threshold) or P(T < threshold) from TimesFM quantile outputs.

    Uses linear interpolation between quantile levels to estimate the probability
    that the daily max temperature exceeds or falls below a threshold.
    This captures non-Gaussian tails that the current Gaussian CDF approach misses.

    Args:
        historical_temps: Hourly temperature observations.
        threshold: Temperature threshold (F or C).
        direction: "above" or "below".
        horizon: Hours to forecast.
        model_id: Model identifier.
        context_length: Context window.

    Returns:
        (probability, details_dict)
    """
    result = forecast_temperature(historical_temps, horizon, model_id, context_length)
    quantiles = result["quantiles"]  # (horizon, 10)

    # For daily-high markets, we care about the max temperature across the forecast window.
    # Extract the max across the horizon for each quantile level.
    max_per_quantile = np.max(quantiles, axis=0)  # shape (10,)

    # Quantile levels: [mean, p10, p20, p30, p40, p50, p60, p70, p80, p90]
    # For probability estimation, use the ordered percentiles (skip mean at index 0)
    percentile_levels = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    percentile_values = max_per_quantile[1:]  # skip mean, get p10-p90

    # Interpolate: P(T <= threshold)
    # If threshold is below all quantiles, P(T <= threshold) ~ 0
    # If threshold is above all quantiles, P(T <= threshold) ~ 1
    if threshold <= percentile_values[0]:
        # Below p10 — extrapolate conservatively
        prob_below = 0.05
    elif threshold >= percentile_values[-1]:
        # Above p90 — extrapolate conservatively
        prob_below = 0.95
    else:
        # Linear interpolation between quantile values
        prob_below = float(np.interp(threshold, percentile_values, percentile_levels))

    if direction == "above":
        probability = 1.0 - prob_below
    else:
        probability = prob_below

    # Clamp to [0.01, 0.99] to avoid certainty
    probability = max(0.01, min(0.99, probability))

    details = {
        "timesfm_point_max": result["forecast_max"],
        "timesfm_p90_max": result["quantile_p90_max"],
        "timesfm_p10_max": result["quantile_p10_max"],
        "timesfm_median_max": result["forecast_median"],
        "timesfm_threshold": threshold,
        "timesfm_direction": direction,
        "timesfm_prob_raw": round(probability, 4),
        "timesfm_quantile_values": [round(float(v), 2) for v in percentile_values],
    }

    return probability, details


def predict_nwp_error(
    historical_errors: list[float],
    horizon: int = 1,
    model_id: str = "google/timesfm-1.0-200m-pytorch",
    context_length: int = 512,
) -> tuple[float, dict[str, Any]]:
    """Predict NWP forecast error using TimesFM on the residual time series.

    This is the HIGHEST-ALPHA integration. Instead of predicting weather directly,
    we predict the systematic error of GFS/HRRR/ECMWF by forecasting the
    (actual_temp - nwp_forecast_temp) residual series.

    If TimesFM predicts a +1.5F bias, we shift the Gaussian CDF mean by +1.5F,
    turning a generic NWP forecast into a locally-tuned one.

    Args:
        historical_errors: Time series of (actual - NWP_forecast) residuals.
            Positive = NWP underestimated, Negative = NWP overestimated.
        horizon: Number of steps ahead to predict (typically 1 for next forecast).
        model_id: Model identifier.
        context_length: Context window.

    Returns:
        (predicted_error_adjustment, details_dict)
        predicted_error_adjustment: float — shift to apply to NWP forecast mean.
            Positive means NWP is likely underestimating, so add this to forecast.
    """
    if len(historical_errors) < 10:
        return 0.0, {"reason": "insufficient_error_history", "n": len(historical_errors)}

    model = _ensure_model(model_id, context_length)

    context = np.array(historical_errors[-context_length:], dtype=np.float64)
    # freq=[0] = high frequency (hourly/sub-daily)
    point, quantiles = model.forecast([context], freq=[0])

    predicted_error = float(point[0][0])  # next-step error prediction
    quantile_data = quantiles[0][0]  # shape (10,) for the single horizon step

    # Safety clamp: don't adjust more than 5F — larger corrections are likely noise
    predicted_error = max(-5.0, min(5.0, predicted_error))

    # Confidence: if the p10-p90 spread is tight, we're more confident
    error_spread = float(quantile_data[_Q_P90] - quantile_data[_Q_P10])
    confidence = max(0.0, min(1.0, 1.0 - error_spread / 10.0))

    # Scale adjustment by confidence — uncertain predictions get smaller shifts
    adjusted_error = predicted_error * confidence

    details = {
        "raw_predicted_error": round(predicted_error, 2),
        "adjusted_error": round(adjusted_error, 2),
        "confidence": round(confidence, 3),
        "error_spread_p10_p90": round(error_spread, 2),
        "context_points": len(historical_errors),
        "recent_mean_error": round(float(np.mean(historical_errors[-10:])), 2),
    }

    return adjusted_error, details


class TimesFMForecaster:
    """Stateful wrapper for TimesFM integration with WeatherEdge.

    Manages per-station historical data and provides cached forecasts
    within a decision cycle to avoid redundant inference.
    """

    def __init__(self, settings):
        self.enabled = settings.enable_timesfm
        self.model_id = settings.timesfm_model_id
        self.context_length = settings.timesfm_context_length
        self.max_horizon = settings.timesfm_max_horizon
        self.error_correction_enabled = settings.timesfm_error_correction
        self.quantile_enabled = settings.timesfm_quantile_probability

        # Per-station historical data (populated from DB)
        self._station_temps: dict[str, list[float]] = {}
        # Per-station NWP error history: (actual - forecast) residuals
        self._station_errors: dict[str, list[float]] = {}
        # Cycle cache to avoid re-running inference
        self._cycle_cache: dict[str, dict] = {}

    def update_station_history(self, station: str, temps: list[float]):
        """Update historical temperature observations for a station."""
        self._station_temps[station] = temps

    def update_error_history(self, station: str, errors: list[float]):
        """Update NWP error history for a station."""
        self._station_errors[station] = errors

    def clear_cycle_cache(self):
        """Clear per-cycle cache. Call at the start of each decision cycle."""
        self._cycle_cache.clear()

    def get_probability(
        self,
        station: str,
        threshold: float,
        direction: str = "above",
        lead_time_hours: float = 24.0,
    ) -> tuple[float | None, dict[str, Any]]:
        """Get TimesFM probability for a market, using quantile outputs.

        Returns (probability, details) or (None, details) if unavailable.
        """
        if not self.enabled or not self.quantile_enabled:
            return None, {"reason": "timesfm_disabled"}

        cache_key = f"prob:{station}:{threshold}:{direction}"
        if cache_key in self._cycle_cache:
            cached = self._cycle_cache[cache_key]
            return cached["probability"], cached["details"]

        temps = self._station_temps.get(station)
        if not temps or len(temps) < 24:
            return None, {"reason": "insufficient_station_data", "n": len(temps) if temps else 0}

        try:
            horizon = min(int(lead_time_hours) + 6, self.max_horizon)
            prob, details = quantile_probability(
                historical_temps=temps,
                threshold=threshold,
                direction=direction,
                horizon=horizon,
                model_id=self.model_id,
                context_length=self.context_length,
            )
            self._cycle_cache[cache_key] = {"probability": prob, "details": details}
            return prob, details
        except Exception as exc:
            log.warning("TimesFM quantile_probability failed for %s: %s", station, exc)
            return None, {"reason": "inference_failed", "error": str(exc)}

    def get_error_correction(
        self,
        station: str,
    ) -> tuple[float, dict[str, Any]]:
        """Get NWP error correction for a station.

        Returns (error_adjustment, details). error_adjustment is 0.0 if unavailable.
        """
        if not self.enabled or not self.error_correction_enabled:
            return 0.0, {"reason": "error_correction_disabled"}

        cache_key = f"error:{station}"
        if cache_key in self._cycle_cache:
            cached = self._cycle_cache[cache_key]
            return cached["adjustment"], cached["details"]

        errors = self._station_errors.get(station)
        if not errors or len(errors) < 10:
            return 0.0, {"reason": "insufficient_error_history", "n": len(errors) if errors else 0}

        try:
            adj, details = predict_nwp_error(
                historical_errors=errors,
                horizon=1,
                model_id=self.model_id,
                context_length=self.context_length,
            )
            self._cycle_cache[cache_key] = {"adjustment": adj, "details": details}
            return adj, details
        except Exception as exc:
            log.warning("TimesFM error prediction failed for %s: %s", station, exc)
            return 0.0, {"reason": "inference_failed", "error": str(exc)}

    def is_available(self) -> bool:
        """Check if TimesFM model is loaded and available."""
        if not self.enabled:
            return False
        model = _load_model(self.model_id, self.context_length)
        return model is not None
