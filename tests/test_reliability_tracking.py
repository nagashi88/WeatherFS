"""Tests for US-RT — reliability tracking via compute_reliability_data."""

from __future__ import annotations

import pytest

from weatheredge.learning.calibration import compute_reliability_data


def _make_records(n: int = 100) -> list[dict]:
    """Create synthetic calibration records with known properties.

    For predicted_probability in [0, 1), actual_outcome = 1 if p > 0.5 else 0.
    This means bins 0.5-1.0 should have empirical_freq ~1.0 and bins 0.0-0.5 ~0.0.
    """
    records = []
    for i in range(n):
        p = i / n
        outcome = 1 if p >= 0.5 else 0
        records.append({"predicted_probability": p, "actual_outcome": outcome})
    return records


class TestComputeReliabilityData:
    def test_returns_10_bins(self):
        records = _make_records(100)
        bins = compute_reliability_data(records)
        assert len(bins) == 10

    def test_bin_boundaries(self):
        records = _make_records(100)
        bins = compute_reliability_data(records)
        for i, b in enumerate(bins):
            assert b["bin_low"] == pytest.approx(i / 10.0)
            assert b["bin_high"] == pytest.approx((i + 1) / 10.0)

    def test_empirical_frequencies_correct(self):
        records = _make_records(100)
        bins = compute_reliability_data(records)
        # Bins 0-4 (p < 0.5): all outcomes are 0 → empirical_freq = 0.0
        for b in bins[:5]:
            assert b["empirical_freq"] == pytest.approx(0.0, abs=0.01)
        # Bins 5-9 (p >= 0.5): all outcomes are 1 → empirical_freq = 1.0
        for b in bins[5:]:
            assert b["empirical_freq"] == pytest.approx(1.0, abs=0.01)

    def test_predicted_mean_in_bin(self):
        records = _make_records(100)
        bins = compute_reliability_data(records)
        for b in bins:
            if b["n"] > 0:
                assert b["bin_low"] <= b["predicted_mean"] < b["bin_high"]

    def test_empty_records_returns_empty_bins(self):
        bins = compute_reliability_data([])
        assert len(bins) == 10
        for b in bins:
            assert b["n"] == 0
            assert b["empirical_freq"] is None

    def test_custom_n_bins(self):
        records = _make_records(100)
        bins = compute_reliability_data(records, n_bins=5)
        assert len(bins) == 5
