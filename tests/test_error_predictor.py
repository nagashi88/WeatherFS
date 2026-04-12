"""Tests for XGBoost forecast error prediction model (STEP 6).

Verifies:
- Feature vector generation from ErrorSample
- Training on synthetic data
- Walk-forward validation pipeline
- Prediction returns float
- Report generation
"""

from __future__ import annotations

import math
from datetime import date

import pytest

try:
    import xgboost  # noqa: F401
    HAS_XGBOOST = True
except (ImportError, OSError):
    HAS_XGBOOST = False

from weatheredge.learning.error_predictor import (
    FEATURE_NAMES,
    ErrorPredictor,
    ErrorPredictorReport,
    ErrorSample,
    WalkForwardResult,
)

xgboost_required = pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not available")


# ---------------------------------------------------------------------------
# Fixtures: synthetic samples
# ---------------------------------------------------------------------------

def _make_samples(n_months: int = 6, days_per_month: int = 28, station: str = "KNYC") -> list[ErrorSample]:
    """Create synthetic samples with a known bias pattern.

    GFS has a +2°F warm bias that's correlated with wind direction.
    SW wind → +3°F bias, NW wind → +1°F bias, others → +2°F.
    """
    samples = []
    for m in range(n_months):
        month = 4 + m  # Apr through Sep
        for d in range(1, min(days_per_month + 1, 29)):
            # Rotate wind directions
            winds = ["SW", "NW", "N", "SE"]
            wind = winds[d % len(winds)]
            bias = {"SW": 3.0, "NW": 1.0, "N": 2.0, "SE": 2.0}[wind]

            actual = 70.0 + 5 * math.sin(2 * math.pi * d / 28) + m * 2
            gfs = actual + bias + 0.5 * math.sin(d)  # systematic + slight noise

            samples.append(ErrorSample(
                station=station,
                sample_date=date(2025, month, d),
                gfs_temp=round(gfs, 1),
                actual_temp=round(actual, 1),
                ecmwf_temp=round(actual + bias * 0.8, 1),
                hrrr_temp=round(actual + bias * 0.5, 1) if station.startswith("K") else None,
                metar_temp=round(actual - 5, 1),
                wind_dir=wind,
                wind_speed=10.0,
                cloud_cover="clear" if d % 3 == 0 else "scattered",
                rolling_7d_error=bias + 0.1 * d,
                realtime_error=bias * 0.5,
            ))
    return samples


# ---------------------------------------------------------------------------
# ErrorSample
# ---------------------------------------------------------------------------

class TestErrorSample:
    def test_target_is_forecast_minus_actual(self):
        s = ErrorSample(
            station="KNYC", sample_date=date(2025, 7, 1),
            gfs_temp=82.0, actual_temp=80.0,
        )
        assert s.target == pytest.approx(2.0)

    def test_feature_vector_length(self):
        s = ErrorSample(
            station="KNYC", sample_date=date(2025, 7, 1),
            gfs_temp=82.0, actual_temp=80.0,
            ecmwf_temp=81.0, hrrr_temp=81.5,
            metar_temp=78.0, wind_dir="SW", wind_speed=10.0,
            cloud_cover="clear", rolling_7d_error=1.5, realtime_error=0.8,
        )
        vec = s.to_feature_vector()
        assert len(vec) == len(FEATURE_NAMES)

    def test_feature_vector_none_handling(self):
        """Missing model temps should fallback to gfs_temp."""
        s = ErrorSample(
            station="KNYC", sample_date=date(2025, 7, 1),
            gfs_temp=82.0, actual_temp=80.0,
        )
        vec = s.to_feature_vector()
        # ecmwf and hrrr should be filled with gfs_temp
        assert vec[1] == 82.0  # ecmwf fallback
        assert vec[2] == 82.0  # hrrr fallback
        assert vec[3] == 0.0   # spread = 0 with one model

    def test_inter_model_spread(self):
        s = ErrorSample(
            station="KNYC", sample_date=date(2025, 7, 1),
            gfs_temp=82.0, actual_temp=80.0,
            ecmwf_temp=79.0, hrrr_temp=84.0,
        )
        vec = s.to_feature_vector()
        assert vec[3] == pytest.approx(5.0)  # 84 - 79


# ---------------------------------------------------------------------------
# Training & Prediction
# ---------------------------------------------------------------------------

@xgboost_required
class TestTraining:
    def test_train_and_predict(self):
        """Model should train and return a prediction."""
        samples = _make_samples(n_months=4, days_per_month=20)
        predictor = ErrorPredictor(n_estimators=10, max_depth=3)
        predictor.train(samples)

        # Predict on a new sample
        new = ErrorSample(
            station="KNYC", sample_date=date(2025, 9, 15),
            gfs_temp=85.0, actual_temp=83.0,
            wind_dir="SW", wind_speed=12.0, cloud_cover="clear",
            rolling_7d_error=2.5, realtime_error=1.5,
        )
        prediction = predictor.predict(new)
        assert isinstance(prediction, float)
        # Should predict a positive error (GFS warm bias)
        assert prediction > 0

    def test_predict_without_training_returns_zero(self):
        predictor = ErrorPredictor()
        s = ErrorSample("KNYC", date(2025, 7, 1), 82.0, 80.0)
        assert predictor.predict(s) == 0.0

    def test_predict_batch(self):
        samples = _make_samples(n_months=3, days_per_month=20)
        predictor = ErrorPredictor(n_estimators=10, max_depth=3)
        predictor.train(samples)
        preds = predictor.predict_batch(samples[:5])
        assert len(preds) == 5
        assert all(isinstance(p, float) for p in preds)

    def test_feature_importance(self):
        samples = _make_samples(n_months=4, days_per_month=20)
        predictor = ErrorPredictor(n_estimators=10, max_depth=3)
        predictor.train(samples)
        fi = predictor.get_feature_importance()
        assert len(fi) == len(FEATURE_NAMES)
        assert all(v >= 0 for v in fi.values())

    def test_too_few_samples(self):
        """Training with < 20 samples should not crash."""
        predictor = ErrorPredictor()
        samples = _make_samples(n_months=1, days_per_month=5)
        predictor.train(samples)  # Should log warning and skip
        assert predictor._model is None


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

@xgboost_required
class TestWalkForward:
    def test_walk_forward_produces_folds(self):
        samples = _make_samples(n_months=6, days_per_month=25)
        predictor = ErrorPredictor(n_estimators=10, max_depth=3)
        report = predictor.walk_forward_validate(samples, min_train_months=3)
        assert len(report.folds) > 0
        assert report.overall_mae > 0
        assert report.overall_rmse > 0

    def test_walk_forward_r_squared(self):
        """With systematic bias in data, R² should be > 0."""
        samples = _make_samples(n_months=6, days_per_month=25)
        predictor = ErrorPredictor(n_estimators=50, max_depth=4)
        report = predictor.walk_forward_validate(samples, min_train_months=3)
        # Our synthetic data has clear patterns, so R² should be positive
        assert report.overall_r_squared > 0.0

    def test_walk_forward_not_enough_months(self):
        samples = _make_samples(n_months=2, days_per_month=20)
        predictor = ErrorPredictor()
        report = predictor.walk_forward_validate(samples, min_train_months=3)
        assert len(report.folds) == 0

    def test_fold_metrics_reasonable(self):
        samples = _make_samples(n_months=6, days_per_month=25)
        predictor = ErrorPredictor(n_estimators=10, max_depth=3)
        report = predictor.walk_forward_validate(samples, min_train_months=3)
        for fold in report.folds:
            assert fold.mae >= 0
            assert fold.rmse >= 0
            assert fold.n_train > 0
            assert fold.n_test > 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestReport:
    def test_print_report(self, capsys):
        predictor = ErrorPredictor()
        report = ErrorPredictorReport(
            folds=[
                WalkForwardResult(3, 7, 2025, 100, 25, 1.5, 2.0, 0.25),
            ],
            overall_mae=1.5,
            overall_rmse=2.0,
            overall_r_squared=0.25,
            feature_importance={"gfs_temp": 0.30, "wind_dir": 0.20},
        )
        text = predictor.print_report(report)
        assert "MAE" in text
        assert "R²" in text or "R\u00b2" in text
        assert "gfs_temp" in text

    def test_low_r2_warning(self, capsys):
        predictor = ErrorPredictor()
        report = ErrorPredictorReport(overall_r_squared=0.05)
        text = predictor.print_report(report)
        assert "WARNING" in text
