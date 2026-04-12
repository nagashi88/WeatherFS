"""Tests for RMSEUpdater."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from weatheredge.db.database import init_db, make_engine, get_session
from weatheredge.db.models import Forecast
from weatheredge.learning.rmse_updater import RMSEUpdater
from weatheredge.probability import bias_correction


@pytest.fixture()
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def updater(engine):
    return RMSEUpdater(engine)


def _make_forecast(station: str, model: str, forecast_temp: float, actual_temp: float) -> Forecast:
    """Helper to build a Forecast row with both forecast and actual temps."""
    return Forecast(
        station=station,
        target_date="2026-03-29",
        model=model,
        forecast_timestamp=datetime.now(timezone.utc),
        lead_time_hours=24,
        temperature_forecast=forecast_temp,
        actual_temp=actual_temp,
        created_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# test_rmse_calculation
# ---------------------------------------------------------------------------

def test_rmse_calculation(engine, updater):
    """3 forecasts with known errors → RMSE matches manual calculation."""
    # errors: 1, -2, 3  →  squared: 1, 4, 9  →  mean: 14/3  →  rmse: sqrt(14/3)
    rows = [
        _make_forecast("KNYC", "gfs", forecast_temp=70.0, actual_temp=69.0),  # error=1
        _make_forecast("KNYC", "gfs", forecast_temp=70.0, actual_temp=72.0),  # error=-2
        _make_forecast("KNYC", "gfs", forecast_temp=70.0, actual_temp=67.0),  # error=3
    ]
    with get_session(engine) as session:
        for row in rows:
            session.add(row)

    rmse = updater.recalculate_rmse("KNYC", "gfs", days=90)

    expected = math.sqrt((1 + 4 + 9) / 3)
    assert rmse is not None
    assert abs(rmse - expected) < 1e-9


# ---------------------------------------------------------------------------
# test_rmse_no_data
# ---------------------------------------------------------------------------

def test_rmse_no_data(updater):
    """No forecasts → returns None."""
    result = updater.recalculate_rmse("KNYC", "gfs", days=30)
    assert result is None


def test_rmse_no_actual_temp_excluded(engine, updater):
    """Forecasts without actual_temp are excluded; no actual data → returns None."""
    row = Forecast(
        station="KNYC",
        target_date="2026-03-29",
        model="gfs",
        forecast_timestamp=datetime.now(timezone.utc),
        lead_time_hours=24,
        temperature_forecast=70.0,
        actual_temp=None,  # no actual — should be excluded
        created_at=datetime.now(timezone.utc),
    )
    with get_session(engine) as session:
        session.add(row)

    result = updater.recalculate_rmse("KNYC", "gfs", days=90)
    assert result is None


# ---------------------------------------------------------------------------
# test_update_propagates_to_bias_correction
# ---------------------------------------------------------------------------

def test_update_propagates_to_bias_correction(engine, updater):
    """After update_all_rmse, STATION_RMSE dict reflects new RMSE value."""
    # Insert data for KNYC/gfs — error of exactly 4.0 each → RMSE = 4.0 (above 3.0 floor)
    rows = [
        _make_forecast("KNYC", "gfs", forecast_temp=70.0, actual_temp=66.0),  # error=4
        _make_forecast("KNYC", "gfs", forecast_temp=70.0, actual_temp=66.0),  # error=4
    ]
    with get_session(engine) as session:
        for row in rows:
            session.add(row)

    # Capture original value to restore after test
    original = bias_correction.STATION_RMSE["KNYC"]["gfs"][1]

    try:
        updater.update_all_rmse(days=90)
        assert bias_correction.STATION_RMSE["KNYC"]["gfs"][1] == pytest.approx(4.0, abs=1e-9)
    finally:
        # Restore to avoid polluting other tests
        bias_correction.STATION_RMSE["KNYC"]["gfs"][1] = original
