"""Tests for the Brier Score Calibration system."""

from __future__ import annotations

import pytest
from weatheredge.db.database import init_db, make_engine
from weatheredge.learning.calibration import CalibrationTracker


@pytest.fixture()
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def tracker(engine):
    return CalibrationTracker(engine)


# ---------------------------------------------------------------------------
# Brier score calculation
# ---------------------------------------------------------------------------

def test_brier_perfect_prediction():
    """predicted=1.0, actual=1 → brier=0.0"""
    assert CalibrationTracker.calculate_brier_score(1.0, 1) == 0.0


def test_brier_worst_prediction():
    """predicted=1.0, actual=0 → brier=1.0"""
    assert CalibrationTracker.calculate_brier_score(1.0, 0) == 1.0


def test_brier_moderate():
    """predicted=0.7, actual=1 → brier=0.09"""
    result = CalibrationTracker.calculate_brier_score(0.7, 1)
    assert abs(result - 0.09) < 1e-9


# ---------------------------------------------------------------------------
# update_calibration + get_calibration_stats
# ---------------------------------------------------------------------------

def test_update_and_get_stats(tracker):
    """Insert 3 calibration rows; get_calibration_stats returns correct avg."""
    tracker.update_calibration("gfs", "KNYC", 1, 0.8, 1)   # brier = 0.04
    tracker.update_calibration("gfs", "KNYC", 1, 0.6, 0)   # brier = 0.36
    tracker.update_calibration("gfs", "KNYC", 1, 0.5, 1)   # brier = 0.25

    stats = tracker.get_calibration_stats()
    assert stats["count"] == 3
    expected_avg = (0.04 + 0.36 + 0.25) / 3
    assert abs(stats["avg_brier"] - expected_avg) < 1e-9


def test_stats_with_filters(tracker):
    """Filter by model → only matching rows are included."""
    tracker.update_calibration("gfs", "KNYC", 1, 0.8, 1)     # brier=0.04
    tracker.update_calibration("ecmwf", "KNYC", 1, 0.5, 0)   # brier=0.25
    tracker.update_calibration("ecmwf", "KORD", 2, 0.3, 1)   # brier=0.49

    stats_gfs = tracker.get_calibration_stats(model="gfs")
    assert stats_gfs["count"] == 1
    assert abs(stats_gfs["avg_brier"] - 0.04) < 1e-9

    stats_ecmwf = tracker.get_calibration_stats(model="ecmwf")
    assert stats_ecmwf["count"] == 2
    expected = (0.25 + 0.49) / 2
    assert abs(stats_ecmwf["avg_brier"] - expected) < 1e-9

    stats_station = tracker.get_calibration_stats(station="KORD")
    assert stats_station["count"] == 1

    stats_lead = tracker.get_calibration_stats(lead_time_days=2)
    assert stats_lead["count"] == 1


# ---------------------------------------------------------------------------
# get_reliability_data
# ---------------------------------------------------------------------------

def test_reliability_data_10_bins(tracker):
    """Insert data across multiple bins; verify 10 bins returned with correct frequencies."""
    # Bin 0 (0.0-0.1): predicted=0.05, actual=0
    tracker.update_calibration("gfs", "KNYC", 1, 0.05, 0)
    # Bin 4 (0.4-0.5): predicted=0.45, actual=1
    tracker.update_calibration("gfs", "KNYC", 1, 0.45, 1)
    # Bin 4 (0.4-0.5): predicted=0.42, actual=0
    tracker.update_calibration("gfs", "KNYC", 1, 0.42, 0)
    # Bin 9 (0.9-1.0): predicted=0.95, actual=1
    tracker.update_calibration("gfs", "KNYC", 1, 0.95, 1)

    bins = tracker.get_reliability_data()
    assert len(bins) == 10

    # Check bin centers are correct
    for i, b in enumerate(bins):
        expected_center = (i * 0.1 + (i + 1) * 0.1) / 2
        assert abs(b["bin_center"] - expected_center) < 1e-9

    # Bin 0: 1 point, actual_freq=0.0
    b0 = bins[0]
    assert b0["count"] == 1
    assert b0["actual_freq"] == 0.0
    assert abs(b0["predicted_avg"] - 0.05) < 1e-9

    # Bin 4: 2 points, actual_freq=0.5
    b4 = bins[4]
    assert b4["count"] == 2
    assert abs(b4["actual_freq"] - 0.5) < 1e-9
    assert abs(b4["predicted_avg"] - (0.45 + 0.42) / 2) < 1e-9

    # Bin 9: 1 point, actual_freq=1.0
    b9 = bins[9]
    assert b9["count"] == 1
    assert b9["actual_freq"] == 1.0

    # Empty bins have count=0 and actual_freq=None
    empty_bins = [b for b in bins if b["count"] == 0]
    assert len(empty_bins) == 7
    for b in empty_bins:
        assert b["actual_freq"] is None


def test_reliability_empty(tracker):
    """No data → 10 bins all with count=0 and actual_freq=None."""
    bins = tracker.get_reliability_data()
    assert len(bins) == 10
    for b in bins:
        assert b["count"] == 0
        assert b["actual_freq"] is None
