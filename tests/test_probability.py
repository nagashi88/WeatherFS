"""Tests for the probability engine (PRD Sections 4.1-4.6, Addendum Gaps 2 and 3)."""

import pytest

from weatheredge.probability.gaussian import gaussian_probability
from weatheredge.probability.ensemble import ensemble_probability
from weatheredge.probability.bucket import bucket_probability
from weatheredge.probability.engine import calculate_probability
from weatheredge.probability.edge import calculate_net_edge, PLATFORM_FEES
from weatheredge.probability.bias_correction import (
    BIAS_TABLE,
    apply_bias_correction,
    get_rmse,
)


# ---------------------------------------------------------------------------
# gaussian_probability
# ---------------------------------------------------------------------------

class TestGaussianProbability:
    def test_high_forecast_above_threshold(self):
        """forecast 85, threshold 80, rmse 2.5 → P high (clamped to 0.95 max)"""
        p = gaussian_probability(85.0, 80.0, 2.5, direction="above")
        assert p >= 0.90  # Student-T + clamp at 0.95

    def test_forecast_equals_threshold(self):
        """forecast 80, threshold 80, rmse 2.5 → P ≈ 0.50"""
        p = gaussian_probability(80.0, 80.0, 2.5, direction="above")
        assert abs(p - 0.50) < 0.01

    def test_low_forecast_above_threshold(self):
        """forecast 75, threshold 80, rmse 2.5 → P low (clamped to 0.05 min)"""
        p = gaussian_probability(75.0, 80.0, 2.5, direction="above")
        assert p <= 0.10  # Student-T + clamp at 0.05


# ---------------------------------------------------------------------------
# ensemble_probability
# ---------------------------------------------------------------------------

class TestEnsembleProbability:
    def test_all_members_above(self):
        """31/31 above → P = 0.95 (clamped from 1.0)"""
        temps = [85.0] * 31
        p = ensemble_probability(temps, 80.0, direction="above")
        assert p == 0.95  # clamped to PROB_CEIL

    def test_sixteen_of_thirty_one_above(self):
        """16/31 above → P ≈ 0.516"""
        temps = [85.0] * 16 + [75.0] * 15
        p = ensemble_probability(temps, 80.0, direction="above")
        assert abs(p - 16 / 31) < 1e-9


# ---------------------------------------------------------------------------
# bucket_probability
# ---------------------------------------------------------------------------

class TestBucketProbability:
    def test_forecast_82_bucket_80_85(self):
        """forecast 82, bucket [80, 85], rmse 2.5 → P ≈ 0.65 (within 0.1)"""
        p = bucket_probability(82.0, 80.0, 85.0, 2.5)
        assert abs(p - 0.65) < 0.1


# ---------------------------------------------------------------------------
# calculate_probability (engine — Addendum Gap 3)
# ---------------------------------------------------------------------------

class TestCalculateProbability:
    def test_with_25_ensemble_members_returns_ensemble(self):
        """25 ensemble members → model_used == 'ensemble'"""
        temps = [85.0] * 25
        prob, model_used = calculate_probability(85.0, 80.0, 2.5, ensemble_temps=temps)
        assert model_used == "ensemble"
        assert 0.0 <= prob <= 1.0

    def test_with_15_ensemble_members_returns_gaussian_fallback(self):
        """15 ensemble members (< 20) → model_used == 'gaussian_fallback'"""
        temps = [85.0] * 15
        prob, model_used = calculate_probability(85.0, 80.0, 2.5, ensemble_temps=temps)
        assert model_used == "gaussian_fallback"
        assert 0.0 <= prob <= 1.0

    def test_with_none_ensemble_returns_gaussian_fallback(self):
        """No ensemble → model_used == 'gaussian_fallback'"""
        prob, model_used = calculate_probability(85.0, 80.0, 2.5, ensemble_temps=None)
        assert model_used == "gaussian_fallback"
        assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# calculate_net_edge (Addendum Gap 2)
# ---------------------------------------------------------------------------

class TestCalculateNetEdge:
    def test_positive_edge_polymarket(self):
        """model_prob 0.60, market_price 0.40, polymarket → net_edge ≈ 0.185"""
        net = calculate_net_edge(0.60, 0.40, "polymarket")
        assert abs(net - 0.185) < 1e-9

    def test_negative_edge_when_model_below_market_plus_fee(self):
        """model_prob < market_price + fee → negative net edge"""
        # gross edge = 0.01, fee = 0.015 → net = -0.005
        net = calculate_net_edge(0.41, 0.40, "polymarket")
        assert net < 0.0


# ---------------------------------------------------------------------------
# bias_correction
# ---------------------------------------------------------------------------

class TestBiasCorrection:
    def test_known_bias_applied(self):
        """Known bias entry → corrected value equals forecast - bias."""
        from weatheredge.probability.bias_correction import BIAS_COUNT_TABLE
        key = ("KMDW", "Jul", "SW", "clear")
        BIAS_TABLE[key] = 1.2
        BIAS_COUNT_TABLE[key] = 50
        result = apply_bias_correction(90.0, "KMDW", "Jul", "SW", "clear")
        assert abs(result - 88.8) < 1e-9
        # Clean up to avoid test pollution
        del BIAS_TABLE[key]
        del BIAS_COUNT_TABLE[key]

    def test_no_bias_entry_returns_unchanged(self):
        """No bias entry → forecast returned unchanged."""
        result = apply_bias_correction(85.0, "KNYC", "Jan", "NW", "overcast")
        assert result == 85.0


# ---------------------------------------------------------------------------
# get_rmse
# ---------------------------------------------------------------------------

class TestGetRmse:
    def test_knyc_gfs_day1(self):
        """KNYC / gfs / day 1 → 4.5 (raised from 3.0 per Phase 0 Brier evidence of overconfidence)"""
        assert get_rmse("KNYC", "gfs", 1) == 4.5

    def test_knyc_ecmwf_day2(self):
        assert get_rmse("KNYC", "ecmwf", 2) == 5.5

    def test_kmdw_hrrr_day1(self):
        assert get_rmse("KMDW", "hrrr", 1) == 4.0

    def test_missing_station_returns_default(self):
        # BugFix H3: Unknown stations return sensible defaults instead of KeyError
        result = get_rmse("KUNKNOWN", "gfs", 1)
        assert isinstance(result, float)
        assert result > 0
