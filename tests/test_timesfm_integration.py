"""Tests for TimesFM integration into WeatherEdge.

Tests cover:
- Blended probability with/without TimesFM (backward compat)
- TimesFM error correction shifts
- TimesFM quantile strategy logic
- TimesFMForecaster class behavior with mock model
- Graceful fallback when TimesFM unavailable
"""

import pytest
from unittest.mock import patch, MagicMock

from weatheredge.probability.blended import (
    blended_probability,
    WEIGHTS_US,
    WEIGHTS_US_NO_TIMESFM,
    WEIGHTS_NON_US,
    WEIGHTS_NON_US_NO_TIMESFM,
)
from weatheredge.probability.engine import calculate_blended_probability
from weatheredge.agents.base import MarketOpportunity, Signal
from weatheredge.agents.strategies.timesfm_quantile import timesfm_quantile_strategy
from weatheredge.config import Settings


# ---------------------------------------------------------------------------
# Weight table structure
# ---------------------------------------------------------------------------

class TestWeightTables:
    def test_timesfm_weights_sum_to_one(self):
        for bucket, weights in WEIGHTS_US.items():
            assert abs(sum(weights.values()) - 1.0) < 1e-6, f"US {bucket} weights don't sum to 1"
            assert "timesfm" in weights, f"US {bucket} missing timesfm key"

    def test_fallback_weights_sum_to_one(self):
        for bucket, weights in WEIGHTS_US_NO_TIMESFM.items():
            assert abs(sum(weights.values()) - 1.0) < 1e-6, f"Fallback US {bucket} don't sum to 1"
            assert "timesfm" not in weights

    def test_non_us_weights_sum_to_one(self):
        assert abs(sum(WEIGHTS_NON_US.values()) - 1.0) < 1e-6
        assert "timesfm" in WEIGHTS_NON_US

    def test_non_us_fallback_weights_sum_to_one(self):
        assert abs(sum(WEIGHTS_NON_US_NO_TIMESFM.values()) - 1.0) < 1e-6
        assert "timesfm" not in WEIGHTS_NON_US_NO_TIMESFM


# ---------------------------------------------------------------------------
# Blended probability backward compatibility
# ---------------------------------------------------------------------------

class TestBlendedBackwardCompat:
    """Ensure blended_probability still works without TimesFM params."""

    def test_without_timesfm_uses_fallback_weights(self):
        prob, model, details = blended_probability(
            threshold=70.0, direction="above", rmse=3.0,
            is_us_station=True, lead_time_hours=6.0,
            gfs_forecast_temp=72.0, hrrr_forecast_temp=73.0,
        )
        assert details["timesfm_active"] is False
        assert "timesfm" not in model

    def test_with_timesfm_prob_uses_timesfm_weights(self):
        prob, model, details = blended_probability(
            threshold=70.0, direction="above", rmse=3.0,
            is_us_station=True, lead_time_hours=6.0,
            gfs_forecast_temp=72.0, hrrr_forecast_temp=73.0,
            timesfm_prob=0.80,
        )
        assert details["timesfm_active"] is True
        assert "timesfm" in model
        assert details.get("timesfm_prob") == 0.80

    def test_no_data_still_returns_05(self):
        prob, model, details = blended_probability(threshold=70.0)
        assert prob == 0.50
        assert model == "no_data"


# ---------------------------------------------------------------------------
# NWP Error correction
# ---------------------------------------------------------------------------

class TestErrorCorrection:
    def test_positive_error_adj_shifts_temps_up(self):
        """If NWP underestimates by 2F, all temps should shift up."""
        prob_base, _, _ = blended_probability(
            threshold=70.0, direction="above", rmse=3.0,
            gfs_forecast_temp=68.0, hrrr_forecast_temp=68.0,
            is_us_station=True, lead_time_hours=12.0,
        )
        prob_corrected, _, details = blended_probability(
            threshold=70.0, direction="above", rmse=3.0,
            gfs_forecast_temp=68.0, hrrr_forecast_temp=68.0,
            is_us_station=True, lead_time_hours=12.0,
            timesfm_error_adj=2.0,
        )
        # Corrected probability should be higher (temps shifted toward threshold)
        assert prob_corrected > prob_base
        assert details.get("gfs_temp_corrected") == 70.0
        assert details.get("hrrr_temp_corrected") == 70.0

    def test_negative_error_adj_shifts_temps_down(self):
        prob_base, _, _ = blended_probability(
            threshold=70.0, direction="above", rmse=3.0,
            gfs_forecast_temp=72.0, is_us_station=True, lead_time_hours=12.0,
        )
        prob_corrected, _, _ = blended_probability(
            threshold=70.0, direction="above", rmse=3.0,
            gfs_forecast_temp=72.0, is_us_station=True, lead_time_hours=12.0,
            timesfm_error_adj=-2.0,
        )
        assert prob_corrected < prob_base

    def test_zero_error_adj_is_noop(self):
        prob1, _, d1 = blended_probability(
            threshold=70.0, direction="above", rmse=3.0,
            gfs_forecast_temp=72.0, is_us_station=True, lead_time_hours=12.0,
        )
        prob2, _, d2 = blended_probability(
            threshold=70.0, direction="above", rmse=3.0,
            gfs_forecast_temp=72.0, is_us_station=True, lead_time_hours=12.0,
            timesfm_error_adj=0.0,
        )
        assert abs(prob1 - prob2) < 1e-6


# ---------------------------------------------------------------------------
# Engine passthrough
# ---------------------------------------------------------------------------

class TestEnginePassthrough:
    def test_calculate_blended_probability_passes_timesfm(self):
        prob, model, details = calculate_blended_probability(
            threshold=70.0, direction="above", rmse=3.0,
            is_us_station=True, lead_time_hours=12.0,
            gfs_forecast_temp=72.0, hrrr_forecast_temp=73.0,
            timesfm_prob=0.90, timesfm_error_adj=1.0,
        )
        assert details["timesfm_active"] is True
        assert "timesfm" in model


# ---------------------------------------------------------------------------
# TimesFM Quantile Strategy
# ---------------------------------------------------------------------------

def _make_opp(**kwargs) -> MarketOpportunity:
    """Helper to build a MarketOpportunity with sensible defaults."""
    defaults = dict(
        market_id="TEST-001", platform="polymarket", city="NYC",
        station="KNYC", threshold=70.0, direction="above",
        market_price=0.50, model_probability=0.65,
        net_edge=0.13, net_edge_no=-0.05, market_date="2026-04-03",
        forecast_temp=72.0, rmse=3.0, lead_time_days=1,
        gaussian_prob=0.60, ensemble_prob=0.63, blended_prob=0.65,
        timesfm_prob=None, timesfm_error_adj=None,
    )
    defaults.update(kwargs)
    return MarketOpportunity(**defaults)


class TestTimesFMQuantileStrategy:
    def setup_method(self):
        self.settings = Settings()

    def test_no_timesfm_prob_returns_no_trade(self):
        opp = _make_opp(timesfm_prob=None)
        sig = timesfm_quantile_strategy(opp, self.settings)
        assert sig.side == "no_trade"

    def test_no_gaussian_prob_returns_no_trade(self):
        opp = _make_opp(timesfm_prob=0.80, gaussian_prob=None)
        sig = timesfm_quantile_strategy(opp, self.settings)
        assert sig.side == "no_trade"

    def test_small_kurtosis_gap_returns_no_trade(self):
        """Gaussian and TimesFM agree — no tail alpha."""
        opp = _make_opp(timesfm_prob=0.63, gaussian_prob=0.60)
        sig = timesfm_quantile_strategy(opp, self.settings)
        assert sig.side == "no_trade"
        assert "agree" in sig.reasoning.lower()

    def test_large_kurtosis_gap_buy_yes(self):
        """TimesFM sees fat right tail → buy YES with conviction."""
        opp = _make_opp(
            timesfm_prob=0.75, gaussian_prob=0.55,
            market_price=0.50, model_probability=0.65,
            net_edge=0.13,
        )
        sig = timesfm_quantile_strategy(opp, self.settings)
        assert sig.side == "buy_yes"
        assert sig.agent_name == "TimesFM_Quantile"
        assert "kurtosis_gap" in sig.reasoning

    def test_large_kurtosis_gap_buy_no(self):
        """TimesFM sees fat left tail (lower prob) → buy NO."""
        opp = _make_opp(
            timesfm_prob=0.30, gaussian_prob=0.50,
            market_price=0.55, model_probability=0.40,
            net_edge=-0.05, net_edge_no=0.13,
        )
        sig = timesfm_quantile_strategy(opp, self.settings)
        assert sig.side == "buy_no"

    def test_blended_disagrees_returns_no_trade(self):
        """TimesFM bullish but blended model bearish → no trade (safety)."""
        opp = _make_opp(
            timesfm_prob=0.75, gaussian_prob=0.55,
            market_price=0.70, model_probability=0.60,  # blended < market
            net_edge=-0.05,
        )
        sig = timesfm_quantile_strategy(opp, self.settings)
        assert sig.side == "no_trade"

    def test_error_correction_bonus(self):
        """NWP error correction supporting trade direction → higher confidence."""
        opp_no_bonus = _make_opp(
            timesfm_prob=0.75, gaussian_prob=0.55,
            market_price=0.50, model_probability=0.65,
            net_edge=0.13, timesfm_error_adj=0.0,
        )
        opp_bonus = _make_opp(
            timesfm_prob=0.75, gaussian_prob=0.55,
            market_price=0.50, model_probability=0.65,
            net_edge=0.13, timesfm_error_adj=1.5,  # NWP underestimates → supports YES
        )
        sig_no = timesfm_quantile_strategy(opp_no_bonus, self.settings)
        sig_yes = timesfm_quantile_strategy(opp_bonus, self.settings)
        assert sig_yes.confidence >= sig_no.confidence


# ---------------------------------------------------------------------------
# TimesFMForecaster class
# ---------------------------------------------------------------------------

class TestTimesFMForecaster:
    def test_disabled_returns_none(self):
        settings = Settings()
        settings.enable_timesfm = False
        from weatheredge.probability.timesfm_forecaster import TimesFMForecaster
        forecaster = TimesFMForecaster(settings)
        prob, details = forecaster.get_probability("KNYC", 70.0, "above")
        assert prob is None
        assert details["reason"] == "timesfm_disabled"

    def test_insufficient_data_returns_none(self):
        settings = Settings()
        from weatheredge.probability.timesfm_forecaster import TimesFMForecaster
        forecaster = TimesFMForecaster(settings)
        forecaster.update_station_history("KNYC", [70.0] * 5)  # only 5 points
        prob, details = forecaster.get_probability("KNYC", 70.0, "above")
        assert prob is None
        assert "insufficient" in details["reason"]

    def test_error_correction_disabled(self):
        settings = Settings()
        settings.timesfm_error_correction = False
        from weatheredge.probability.timesfm_forecaster import TimesFMForecaster
        forecaster = TimesFMForecaster(settings)
        adj, details = forecaster.get_error_correction("KNYC")
        assert adj == 0.0
        assert details["reason"] == "error_correction_disabled"

    def test_cycle_cache_cleared(self):
        settings = Settings()
        from weatheredge.probability.timesfm_forecaster import TimesFMForecaster
        forecaster = TimesFMForecaster(settings)
        forecaster._cycle_cache["test"] = {"probability": 0.5, "details": {}}
        forecaster.clear_cycle_cache()
        assert len(forecaster._cycle_cache) == 0


# ---------------------------------------------------------------------------
# MarketOpportunity dataclass
# ---------------------------------------------------------------------------

class TestMarketOpportunityTimesFM:
    def test_timesfm_fields_default_none(self):
        opp = MarketOpportunity(
            market_id="X", platform="p", city="c", station="s",
            threshold=70.0, direction="above", market_price=0.5,
            model_probability=0.6, net_edge=0.1,
        )
        assert opp.timesfm_prob is None
        assert opp.timesfm_error_adj is None

    def test_timesfm_fields_set(self):
        opp = _make_opp(timesfm_prob=0.85, timesfm_error_adj=1.5)
        assert opp.timesfm_prob == 0.85
        assert opp.timesfm_error_adj == 1.5
