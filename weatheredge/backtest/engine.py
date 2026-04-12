from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from weatheredge.probability.engine import calculate_probability
from weatheredge.learning.calibration import CalibrationTracker

log = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    total_forecasts: int = 0
    brier_score: float = 0.0  # Average Brier over all forecasts
    rmse: float = 0.0  # RMSE of probability estimates
    per_model_brier: dict[str, float] = field(default_factory=dict)
    reliability_bins: list[dict] = field(default_factory=list)  # 10-bin reliability

@dataclass
class ForecastRecord:
    station: str
    model: str
    forecast_temp: float
    actual_temp: float
    threshold: float
    direction: str  # 'above' or 'below'
    rmse: float
    ensemble_temps: list[float] | None = None
    lead_time_days: int = 1

class BacktestEngine:
    """Forecast-accuracy-only backtest. No trading P&L simulation (addendum Gap 18)."""

    def run_backtest(self, records: list[ForecastRecord]) -> BacktestResult:
        """
        Compare historical forecasts against actual outcomes.
        Returns calibration metrics (Brier, RMSE, reliability).
        """
        if not records:
            return BacktestResult()

        brier_scores = []
        squared_errors = []
        model_briers: dict[str, list[float]] = {}
        predictions_and_actuals = []

        for rec in records:
            # Calculate what our model would have predicted
            prob, model_used = calculate_probability(
                forecast_temp=rec.forecast_temp,
                threshold=rec.threshold,
                rmse=rec.rmse,
                ensemble_temps=rec.ensemble_temps,
                direction=rec.direction,
            )

            # Determine actual outcome
            if rec.direction == "above":
                # Match live market resolution convention.
                actual = 1 if rec.actual_temp >= rec.threshold else 0
            else:
                actual = 1 if rec.actual_temp < rec.threshold else 0

            # Brier score
            brier = CalibrationTracker.calculate_brier_score(prob, actual)
            brier_scores.append(brier)
            squared_errors.append((prob - actual) ** 2)
            predictions_and_actuals.append((prob, actual))

            # Per-model tracking
            model_key = rec.model
            if model_key not in model_briers:
                model_briers[model_key] = []
            model_briers[model_key].append(brier)

        # Aggregate
        avg_brier = sum(brier_scores) / len(brier_scores)
        rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
        per_model = {m: sum(bs) / len(bs) for m, bs in model_briers.items()}

        # Reliability bins
        reliability = self._compute_reliability(predictions_and_actuals)

        return BacktestResult(
            total_forecasts=len(records),
            brier_score=avg_brier,
            rmse=rmse,
            per_model_brier=per_model,
            reliability_bins=reliability,
        )

    def _compute_reliability(self, predictions_and_actuals: list[tuple[float, int]]) -> list[dict]:
        bins = []
        for i in range(10):
            low, high = i / 10.0, (i + 1) / 10.0
            center = (low + high) / 2
            in_bin = [(p, a) for p, a in predictions_and_actuals if low <= p < high]
            if not in_bin:
                bins.append({"bin_center": center, "predicted_avg": None, "actual_freq": None, "count": 0})
            else:
                bins.append({
                    "bin_center": center,
                    "predicted_avg": sum(p for p, _ in in_bin) / len(in_bin),
                    "actual_freq": sum(a for _, a in in_bin) / len(in_bin),
                    "count": len(in_bin),
                })
        return bins
