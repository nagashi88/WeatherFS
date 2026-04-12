"""XGBoost forecast error prediction model — STEP 6.

Predicts HOW MUCH the GFS forecast will err for a specific station/day,
enabling a corrected forecast that's better than the raw model output.

Target: forecast_error = GFS_predicted_daily_high - actual_observed_daily_high
  (positive = model too warm, negative = model too cold)

Features:
- GFS forecast high (°F)
- ECMWF forecast high (°F), if available
- HRRR forecast high (°F), if available
- Inter-model spread: max - min of available model temps
- METAR current temperature (°F)
- METAR wind direction (categorical → encoded)
- METAR wind speed (kts)
- METAR cloud cover (categorical → encoded)
- Month (1-12)
- 7-day rolling mean forecast error for this station
- Real-time error: GFS_current_hour_temp - METAR_current_temp

Validation: walk-forward (train on months 1..N, test on N+1).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import numpy as np

log = logging.getLogger(__name__)

# Wind regime encoding (ordinal for tree-based models)
WIND_ENCODE: dict[str, int] = {
    "N": 0, "NE": 1, "E": 2, "SE": 3,
    "S": 4, "SW": 5, "W": 6, "NW": 7, "calm": 8,
}

# Cloud cover encoding
CLOUD_ENCODE: dict[str, int] = {
    "clear": 0, "scattered": 1, "broken": 2, "overcast": 3, "unknown": 4,
}


@dataclass
class ErrorSample:
    """One training/test sample for the error prediction model."""
    station: str
    sample_date: date
    gfs_temp: float
    actual_temp: float
    ecmwf_temp: float | None = None
    hrrr_temp: float | None = None
    metar_temp: float | None = None
    wind_dir: str = "calm"       # regime string
    wind_speed: float = 0.0
    cloud_cover: str = "unknown"
    rolling_7d_error: float = 0.0
    realtime_error: float = 0.0  # GFS current hour - METAR current

    @property
    def target(self) -> float:
        """Forecast error (positive = model too warm)."""
        return self.gfs_temp - self.actual_temp

    def to_feature_vector(self) -> list[float]:
        """Convert to numeric feature vector for XGBoost."""
        # Inter-model spread
        temps = [self.gfs_temp]
        if self.ecmwf_temp is not None:
            temps.append(self.ecmwf_temp)
        if self.hrrr_temp is not None:
            temps.append(self.hrrr_temp)
        spread = max(temps) - min(temps) if len(temps) > 1 else 0.0

        return [
            self.gfs_temp,
            self.ecmwf_temp if self.ecmwf_temp is not None else self.gfs_temp,
            self.hrrr_temp if self.hrrr_temp is not None else self.gfs_temp,
            spread,
            self.metar_temp if self.metar_temp is not None else self.gfs_temp,
            WIND_ENCODE.get(self.wind_dir, 8),
            self.wind_speed,
            CLOUD_ENCODE.get(self.cloud_cover, 4),
            self.sample_date.month,
            self.rolling_7d_error,
            self.realtime_error,
        ]


FEATURE_NAMES = [
    "gfs_temp", "ecmwf_temp", "hrrr_temp", "inter_model_spread",
    "metar_temp", "wind_dir", "wind_speed", "cloud_cover",
    "month", "rolling_7d_error", "realtime_error",
]


@dataclass
class WalkForwardResult:
    """Result from one fold of walk-forward validation."""
    train_months: int
    test_month: int
    test_year: int
    n_train: int
    n_test: int
    mae: float
    rmse: float
    r_squared: float


@dataclass
class ErrorPredictorReport:
    """Aggregate report from walk-forward validation."""
    folds: list[WalkForwardResult] = field(default_factory=list)
    overall_mae: float = 0.0
    overall_rmse: float = 0.0
    overall_r_squared: float = 0.0
    feature_importance: dict[str, float] = field(default_factory=dict)


class ErrorPredictor:
    """XGBoost model that predicts forecast error for bias correction.

    Usage:
        predictor = ErrorPredictor()
        report = predictor.walk_forward_validate(samples)
        # If R² > 0.15, the model adds value
        predictor.train(samples)
        predicted_error = predictor.predict(new_sample)
        corrected_forecast = gfs_raw - predicted_error
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        min_child_weight: int = 5,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        self._model = None

    def _build_model(self):
        """Create a fresh XGBoost regressor."""
        from xgboost import XGBRegressor
        return XGBRegressor(**self.params)

    def _prepare_data(self, samples: list[ErrorSample]) -> tuple[np.ndarray, np.ndarray]:
        """Convert samples to X, y numpy arrays."""
        X = np.array([s.to_feature_vector() for s in samples], dtype=np.float32)
        y = np.array([s.target for s in samples], dtype=np.float32)
        return X, y

    def train(self, samples: list[ErrorSample]) -> None:
        """Train the model on all provided samples."""
        if len(samples) < 20:
            log.warning("Too few samples (%d) to train error predictor", len(samples))
            return

        X, y = self._prepare_data(samples)
        self._model = self._build_model()
        self._model.fit(X, y)
        log.info("ErrorPredictor trained on %d samples", len(samples))

    def predict(self, sample: ErrorSample) -> float:
        """Predict the forecast error for a single sample.

        Returns predicted error (positive = model too warm).
        Use: corrected_forecast = gfs_raw - predicted_error
        """
        if self._model is None:
            return 0.0  # No correction if model not trained

        X = np.array([sample.to_feature_vector()], dtype=np.float32)
        return float(self._model.predict(X)[0])

    def predict_batch(self, samples: list[ErrorSample]) -> list[float]:
        """Predict errors for multiple samples."""
        if self._model is None:
            return [0.0] * len(samples)

        X = np.array([s.to_feature_vector() for s in samples], dtype=np.float32)
        return [float(p) for p in self._model.predict(X)]

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores."""
        if self._model is None:
            return {}
        importances = self._model.feature_importances_
        return {name: float(imp) for name, imp in zip(FEATURE_NAMES, importances)}

    def walk_forward_validate(
        self,
        samples: list[ErrorSample],
        min_train_months: int = 3,
    ) -> ErrorPredictorReport:
        """Walk-forward validation: train on months 1..N, test on N+1.

        This ensures we never test on data used for training.
        """
        # Group samples by (year, month)
        by_month: dict[tuple[int, int], list[ErrorSample]] = defaultdict(list)
        for s in samples:
            by_month[(s.sample_date.year, s.sample_date.month)].append(s)

        sorted_months = sorted(by_month.keys())
        if len(sorted_months) < min_train_months + 1:
            log.warning(
                "Not enough months for walk-forward (%d, need %d+1)",
                len(sorted_months), min_train_months,
            )
            return ErrorPredictorReport()

        folds: list[WalkForwardResult] = []
        all_test_predictions: list[float] = []
        all_test_actuals: list[float] = []

        for i in range(min_train_months, len(sorted_months)):
            # Train on months 0..i-1, test on month i
            train_samples = []
            for j in range(i):
                train_samples.extend(by_month[sorted_months[j]])

            test_ym = sorted_months[i]
            test_samples = by_month[test_ym]

            if len(train_samples) < 20 or len(test_samples) < 5:
                continue

            X_train, y_train = self._prepare_data(train_samples)
            X_test, y_test = self._prepare_data(test_samples)

            model = self._build_model()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Compute metrics for this fold
            errors = predictions - y_test
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors ** 2)))

            # R² = 1 - SS_res / SS_tot
            ss_res = float(np.sum(errors ** 2))
            ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            folds.append(WalkForwardResult(
                train_months=i,
                test_month=test_ym[1],
                test_year=test_ym[0],
                n_train=len(train_samples),
                n_test=len(test_samples),
                mae=mae,
                rmse=rmse,
                r_squared=r_squared,
            ))

            all_test_predictions.extend(predictions.tolist())
            all_test_actuals.extend(y_test.tolist())

        # Overall metrics across all folds
        if all_test_predictions:
            preds = np.array(all_test_predictions)
            actuals = np.array(all_test_actuals)
            overall_errors = preds - actuals
            overall_mae = float(np.mean(np.abs(overall_errors)))
            overall_rmse = float(np.sqrt(np.mean(overall_errors ** 2)))
            ss_res = float(np.sum(overall_errors ** 2))
            ss_tot = float(np.sum((actuals - np.mean(actuals)) ** 2))
            overall_r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            overall_mae = overall_rmse = overall_r2 = 0.0

        # Get feature importance from the last model trained
        fi = {}
        if folds:
            # Retrain on all data for feature importance
            self.train(samples)
            fi = self.get_feature_importance()

        report = ErrorPredictorReport(
            folds=folds,
            overall_mae=overall_mae,
            overall_rmse=overall_rmse,
            overall_r_squared=overall_r2,
            feature_importance=fi,
        )

        log.info(
            "Walk-forward: %d folds, overall MAE=%.2f RMSE=%.2f R²=%.3f",
            len(folds), overall_mae, overall_rmse, overall_r2,
        )
        return report

    def print_report(self, report: ErrorPredictorReport) -> str:
        """Print a formatted report. Returns the report string."""
        lines = [
            "=" * 60,
            "FORECAST ERROR PREDICTION MODEL REPORT",
            "=" * 60,
            f"Walk-forward folds: {len(report.folds)}",
            f"Overall MAE:        {report.overall_mae:.2f}°F",
            f"Overall RMSE:       {report.overall_rmse:.2f}°F",
            f"Overall R²:         {report.overall_r_squared:.3f}",
            "",
        ]

        if report.overall_r_squared < 0.15:
            lines.append("WARNING: R² < 0.15 — forecast error is too random for ML correction.")
            lines.append("Recommendation: Use simple bias correction (STEP 5) instead.")
        else:
            lines.append("Model shows predictive power. ML correction may improve forecasts.")

        if report.feature_importance:
            lines.extend(["", "FEATURE IMPORTANCE:", "-" * 40])
            sorted_fi = sorted(report.feature_importance.items(), key=lambda x: -x[1])
            for name, imp in sorted_fi:
                lines.append(f"  {name:<25} {imp:.3f}")

        if report.folds:
            lines.extend(["", "PER-FOLD RESULTS:", "-" * 60])
            for f in report.folds:
                lines.append(
                    f"  {f.test_year}-{f.test_month:02d}: "
                    f"train={f.n_train}, test={f.n_test}, "
                    f"MAE={f.mae:.2f}, RMSE={f.rmse:.2f}, R²={f.r_squared:.3f}"
                )

        lines.append("=" * 60)
        report_str = "\n".join(lines)
        print(report_str)
        return report_str
