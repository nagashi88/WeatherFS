"""Tests for station-specific bias correction (STEP 5).

Verifies:
- Wind direction to regime conversion
- BiasAnalyzer computes correct mean/std per bin
- apply_bias_correction uses populated table
- Fallback behavior when bin has too few samples
- Integration with DailyRecord from backtester
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from weatheredge.probability.bias_correction import (
    BIAS_COUNT_TABLE,
    BIAS_STD_TABLE,
    BIAS_TABLE,
    BiasAnalyzer,
    BiasRecord,
    apply_bias_correction,
    month_to_name,
    wind_direction_to_regime,
)


@pytest.fixture(autouse=True)
def clean_tables():
    """Clear global bias tables before each test."""
    BiasAnalyzer.clear_tables()
    yield
    BiasAnalyzer.clear_tables()


# ---------------------------------------------------------------------------
# Wind direction conversion
# ---------------------------------------------------------------------------

class TestWindDirectionToRegime:
    @pytest.mark.parametrize("degrees,expected", [
        (0, "N"), (10, "N"), (350, "N"), (359, "N"),
        (45, "NE"), (90, "E"), (135, "SE"),
        (180, "S"), (225, "SW"), (270, "W"), (315, "NW"),
        (None, "calm"), (-1, "calm"),
    ])
    def test_conversion(self, degrees, expected):
        assert wind_direction_to_regime(degrees) == expected


class TestMonthToName:
    def test_all_months(self):
        assert month_to_name(1) == "Jan"
        assert month_to_name(6) == "Jun"
        assert month_to_name(12) == "Dec"


# ---------------------------------------------------------------------------
# BiasAnalyzer
# ---------------------------------------------------------------------------

class TestBiasAnalyzer:
    def _make_records(self, station, n, bias, month=7, wind="SW", cloud="clear"):
        """Create n records with known bias (forecast = actual + bias)."""
        return [
            BiasRecord(
                station=station,
                record_date=date(2025, month, min(1 + i, 28)),
                forecast_high_f=80.0 + bias,
                actual_high_f=80.0,
                wind_regime=wind,
                cloud_cover=cloud,
            )
            for i in range(n)
        ]

    def test_mean_error_correct(self):
        """Mean error should match the injected bias."""
        analyzer = BiasAnalyzer(min_samples=5)
        records = self._make_records("KNYC", 20, bias=1.8)
        summary = analyzer.analyze(records)

        assert "KNYC" in summary
        entry = summary["KNYC"][0]
        assert entry["mean_error"] == pytest.approx(1.8, abs=0.01)
        assert entry["sample_size"] == 20

    def test_std_error_zero_for_constant_bias(self):
        """If all errors are identical, std should be 0."""
        analyzer = BiasAnalyzer(min_samples=5)
        records = self._make_records("KNYC", 20, bias=2.0)
        summary = analyzer.analyze(records)

        entry = summary["KNYC"][0]
        assert entry["std_error"] == pytest.approx(0.0, abs=0.01)

    def test_populates_global_tables(self):
        """analyze() should populate BIAS_TABLE, BIAS_STD_TABLE, BIAS_COUNT_TABLE."""
        analyzer = BiasAnalyzer(min_samples=5)
        records = self._make_records("KNYC", 15, bias=1.5, wind="SW", cloud="clear")
        analyzer.analyze(records)

        key = ("KNYC", "Jul", "SW", "clear")
        assert key in BIAS_TABLE
        assert BIAS_TABLE[key] == pytest.approx(1.5, abs=0.01)
        assert BIAS_COUNT_TABLE[key] == 15

    def test_multiple_stations(self):
        """Should handle records from multiple stations."""
        analyzer = BiasAnalyzer(min_samples=5)
        records = (
            self._make_records("KNYC", 15, bias=1.5) +
            self._make_records("KDEN", 15, bias=3.2)
        )
        summary = analyzer.analyze(records)

        assert "KNYC" in summary
        assert "KDEN" in summary

    def test_multiple_bins(self):
        """Different wind/cloud combos should create separate bins."""
        analyzer = BiasAnalyzer(min_samples=5)
        records = (
            self._make_records("KNYC", 15, bias=1.5, wind="SW", cloud="clear") +
            self._make_records("KNYC", 15, bias=-0.5, wind="NW", cloud="overcast")
        )
        summary = analyzer.analyze(records)

        entries = summary["KNYC"]
        assert len(entries) == 2


# ---------------------------------------------------------------------------
# apply_bias_correction
# ---------------------------------------------------------------------------

class TestApplyBiasCorrection:
    def test_correction_applied(self):
        """When bias table is populated, correction subtracts mean error."""
        key = ("KNYC", "Jul", "SW", "clear")
        BIAS_TABLE[key] = 1.8
        BIAS_COUNT_TABLE[key] = 47

        corrected = apply_bias_correction(82.0, "KNYC", "Jul", "SW", "clear")
        assert corrected == pytest.approx(80.2, abs=0.01)

    def test_no_correction_when_no_data(self):
        """With empty table, returns original forecast."""
        corrected = apply_bias_correction(82.0, "KNYC", "Jul", "SW", "clear")
        assert corrected == 82.0

    def test_no_correction_below_min_samples(self):
        """If bin has < min_samples, returns original forecast."""
        key = ("KNYC", "Jul", "SW", "clear")
        BIAS_TABLE[key] = 1.8
        BIAS_COUNT_TABLE[key] = 3  # Below default min of 10

        corrected = apply_bias_correction(82.0, "KNYC", "Jul", "SW", "clear")
        assert corrected == 82.0

    def test_month_as_int(self):
        """Month can be passed as integer."""
        key = ("KNYC", "Jul", "SW", "clear")
        BIAS_TABLE[key] = 1.5
        BIAS_COUNT_TABLE[key] = 20

        corrected = apply_bias_correction(82.0, "KNYC", 7, "SW", "clear")
        assert corrected == pytest.approx(80.5, abs=0.01)

    def test_negative_bias_adds_temperature(self):
        """Negative bias (model too cold) should increase corrected temp."""
        key = ("KNYC", "Mar", "NW", "overcast")
        BIAS_TABLE[key] = -2.0
        BIAS_COUNT_TABLE[key] = 30

        corrected = apply_bias_correction(65.0, "KNYC", "Mar", "NW", "overcast")
        assert corrected == pytest.approx(67.0, abs=0.01)


# ---------------------------------------------------------------------------
# Integration: from DailyRecords
# ---------------------------------------------------------------------------

class TestBuildFromDailyRecords:
    def test_from_daily_records(self):
        """build_from_daily_records should work with backtester DailyRecords."""
        from weatheredge.backtesting.backtester import DailyRecord

        daily_records = [
            DailyRecord("KNYC", date(2025, 7, 1 + i), 82.0, 80.0)
            for i in range(20)
        ]
        analyzer = BiasAnalyzer(min_samples=5)
        summary = analyzer.build_from_daily_records(daily_records)

        assert "KNYC" in summary
        # All records have same bias of 2.0°F, wind=calm, cloud=unknown
        entry = summary["KNYC"][0]
        assert entry["mean_error"] == pytest.approx(2.0, abs=0.01)
        assert entry["wind_regime"] == "calm"
        assert entry["cloud_cover"] == "unknown"


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

class TestPrintTable:
    def test_report_output(self, capsys):
        analyzer = BiasAnalyzer(min_samples=5)
        summary = {
            "KNYC": [
                {"month": "Jul", "wind_regime": "SW", "cloud_cover": "clear",
                 "mean_error": 1.8, "std_error": 2.1, "sample_size": 47},
            ],
        }
        report = analyzer.print_table(summary)
        assert "KNYC" in report
        assert "+1.80" in report
        assert "47" in report
