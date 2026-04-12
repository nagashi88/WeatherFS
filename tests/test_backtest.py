from __future__ import annotations
import math
import pytest
from weatheredge.backtest.engine import BacktestEngine, BacktestResult, ForecastRecord
from weatheredge.backtest.metrics import sharpe_ratio, sortino_ratio, max_drawdown


# ---------------------------------------------------------------------------
# BacktestEngine tests
# ---------------------------------------------------------------------------

def _make_record(
    forecast_temp: float,
    actual_temp: float,
    threshold: float = 70.0,
    direction: str = "above",
    rmse: float = 2.0,
    model: str = "gfs",
    ensemble_temps: list[float] | None = None,
) -> ForecastRecord:
    return ForecastRecord(
        station="KORD",
        model=model,
        forecast_temp=forecast_temp,
        actual_temp=actual_temp,
        threshold=threshold,
        direction=direction,
        rmse=rmse,
        ensemble_temps=ensemble_temps,
    )


def test_backtest_brier_perfect():
    """All forecasts strongly correct -> brier near 0."""
    engine = BacktestEngine()
    # forecast well above threshold, actual also above -> prob ~1, actual=1 -> brier~0
    records = [_make_record(forecast_temp=80.0, actual_temp=75.0, threshold=70.0) for _ in range(10)]
    result = engine.run_backtest(records)
    assert result.total_forecasts == 10
    assert result.brier_score < 0.05


def test_backtest_brier_wrong():
    """All forecasts strongly wrong -> brier near 1."""
    engine = BacktestEngine()
    # forecast well above threshold -> prob ~1, but actual is well below -> actual=0 -> brier~1
    records = [_make_record(forecast_temp=80.0, actual_temp=60.0, threshold=70.0) for _ in range(10)]
    result = engine.run_backtest(records)
    assert result.total_forecasts == 10
    assert result.brier_score > 0.8


def test_backtest_per_model():
    """Two models appear in per_model_brier."""
    engine = BacktestEngine()
    records = [
        _make_record(forecast_temp=80.0, actual_temp=75.0, model="gfs"),
        _make_record(forecast_temp=80.0, actual_temp=75.0, model="gfs"),
        _make_record(forecast_temp=80.0, actual_temp=75.0, model="ecmwf"),
        _make_record(forecast_temp=80.0, actual_temp=75.0, model="ecmwf"),
    ]
    result = engine.run_backtest(records)
    assert "gfs" in result.per_model_brier
    assert "ecmwf" in result.per_model_brier
    assert len(result.per_model_brier) == 2


def test_backtest_empty():
    """No records -> default empty BacktestResult."""
    engine = BacktestEngine()
    result = engine.run_backtest([])
    assert isinstance(result, BacktestResult)
    assert result.total_forecasts == 0
    assert result.brier_score == 0.0
    assert result.rmse == 0.0
    assert result.per_model_brier == {}
    assert result.reliability_bins == []


def test_backtest_reliability_10_bins():
    """Reliability diagram always returns exactly 10 bins."""
    engine = BacktestEngine()
    records = [_make_record(forecast_temp=float(t), actual_temp=75.0, threshold=70.0)
               for t in range(60, 90)]
    result = engine.run_backtest(records)
    assert len(result.reliability_bins) == 10
    for b in result.reliability_bins:
        assert "bin_center" in b
        assert "predicted_avg" in b
        assert "actual_freq" in b
        assert "count" in b


def test_backtest_above_boundary_includes_equal_threshold():
    """actual_temp == threshold counts as YES for above markets."""
    engine = BacktestEngine()
    records = [_make_record(forecast_temp=80.0, actual_temp=70.0, threshold=70.0, direction="above")]
    result = engine.run_backtest(records)
    assert result.total_forecasts == 1
    assert result.brier_score < 0.10


def test_backtest_below_boundary_excludes_equal_threshold():
    """actual_temp == threshold counts as NO for below markets."""
    engine = BacktestEngine()
    records = [_make_record(forecast_temp=60.0, actual_temp=70.0, threshold=70.0, direction="below")]
    result = engine.run_backtest(records)
    assert result.total_forecasts == 1
    assert result.brier_score > 0.80


# ---------------------------------------------------------------------------
# metrics tests
# ---------------------------------------------------------------------------

def test_sharpe_ratio_positive():
    """Positive mean daily PnL with some variance -> positive Sharpe."""
    import random
    random.seed(42)
    pnl = [1.0 + random.uniform(-0.1, 0.1) for _ in range(100)]
    sr = sharpe_ratio(pnl)
    assert sr > 0


def test_sharpe_ratio_zero_std():
    """Constant PnL with zero std -> returns 0.0 (guard against division by zero)."""
    pnl = [5.0] * 50
    sr = sharpe_ratio(pnl)
    assert sr == 0.0


def test_sortino_ratio():
    """Mixed PnL with positive mean -> positive Sortino."""
    pnl = [2.0, -0.5, 3.0, -0.5, 2.0, 1.0, -0.5, 4.0]
    sr = sortino_ratio(pnl)
    assert sr > 0


def test_max_drawdown():
    """Known PnL series produces expected drawdown."""
    # cumulative: 1, 2, 3, 2, 1, 2, 3
    # peak=3, max drawdown = (3-1)/3 = 2/3
    pnl = [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0]
    dd = max_drawdown(pnl)
    expected = 2.0 / 3.0
    assert abs(dd - expected) < 1e-9
