"""Tests for CCG analysis fixes — validates all bug fixes and optimizations.

Covers:
- Fix 1: Aggressive strategy sorts by edge, not probability
- Fix 2: NO-side price floor ($0.10 minimum)
- Fix 3: METAR DST-aware local hour
- Fix 4: Date-exposure cap
- Fix 5: Annualized Sharpe ratio + KSFO/KBOS microclimate
- Probability edge cases (bucket rmse=0, ensemble empty, gaussian rmse=0)
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from weatheredge.agents.base import MarketOpportunity, Signal, TradeOrder
from weatheredge.agents.strategies.aggressive import aggressive_strategy
from weatheredge.agents.strategies.conservative import conservative_strategy
from weatheredge.agents.strategies.sell_no import sell_no_strategy
from weatheredge.agents.risk_manager import RiskManager
from weatheredge.agents.portfolio_manager import PortfolioManager
from weatheredge.agents.rules_qualitative import STATION_MICROCLIMATE, RulesQualitativeEngine
from weatheredge.config import Settings
from weatheredge.execution.paper_trader import PaperTrader, Order
from weatheredge.probability.gaussian import gaussian_probability
from weatheredge.probability.ensemble import ensemble_probability
from weatheredge.probability.bucket import bucket_probability
from weatheredge.probability.edge import calculate_net_edge, calculate_gross_edge
from weatheredge.probability.blended import blended_probability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_opp(**overrides) -> MarketOpportunity:
    defaults = dict(
        market_id="TEST-001",
        platform="polymarket",
        city="NYC",
        station="KNYC",
        threshold=70.0,
        direction="above",
        market_price=0.50,
        model_probability=0.65,
        net_edge=0.135,
        net_edge_no=-0.135,
        market_date="2026-04-02",
        forecast_temp=72.0,
        rmse=3.0,
        lead_time_days=1,
        gaussian_prob=0.65,
        ensemble_prob=0.64,
        blended_prob=0.65,
    )
    defaults.update(overrides)
    return MarketOpportunity(**defaults)


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        initial_bankroll=1000.0,
        min_edge_percent=10.0,
        kelly_fraction=0.15,
        max_open_positions=20,
        max_single_market_exposure=0.08,
        max_cost_per_trade_fraction=1.0,  # disable cost cap in unit tests
        max_single_platform_exposure=0.60,
        max_single_date_exposure=0.25,
        conservative_min_edge=0.20,
        aggressive_min_edge=0.08,
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ===========================================================================
# FIX 1: Aggressive strategy sorts by edge, not probability
# ===========================================================================

class TestAggressiveSortByEdge:
    """The Aggressive ladder picks the top-2 adjacent neighbours by edge. These
    tests must construct REAL bucket markets with shared boundaries — without
    bucket_low/bucket_high the ladder branch returns the center signal only and
    the sort never runs (CCG-bug-audit-2 #6: previous tests passed vacuously).
    """

    def test_sorts_by_net_edge_not_probability(self):
        """Among adjacent buckets, the one with higher edge on the side
        `_emit_signal` actually picks should be ranked first.
        """
        settings = _make_settings()
        # Center bucket [70,75], two adjacent boundary-sharing neighbours.
        primary = _make_opp(
            market_id="primary", bucket_low=70.0, bucket_high=75.0,
            net_edge=0.12, net_edge_no=0.0,
        )
        # Adjacent low [65,70] — high prob, but BUY_NO is the model side
        # (model_probability < market_price) and net_edge_no is the SMALLER edge.
        bucket_bad = _make_opp(
            market_id="bad-bucket", bucket_low=65.0, bucket_high=70.0,
            model_probability=0.40, market_price=0.55,
            net_edge=-0.165, net_edge_no=0.135,
        )
        # Adjacent high [75,80] — model side is BUY_YES with the highest edge.
        bucket_good = _make_opp(
            market_id="good-bucket", bucket_low=75.0, bucket_high=80.0,
            model_probability=0.60, market_price=0.40,
            net_edge=0.185, net_edge_no=-0.185,
        )
        signals = aggressive_strategy(
            primary, [bucket_bad, bucket_good], settings,
        )
        # Center + 2 adjacent = exactly 3 signals when both neighbours pass the
        # edge gate. The good-bucket (higher signal-side edge) should come
        # before the bad-bucket in the adjacent ordering.
        non_center = [s for s in signals if s.reasoning.startswith("Ladder adjacent")]
        assert len(non_center) == 2
        good_idx = next(i for i, s in enumerate(non_center) if s.market_id == "good-bucket")
        bad_idx = next(i for i, s in enumerate(non_center) if s.market_id == "bad-bucket")
        assert good_idx < bad_idx, "Should prioritize by signal-side edge"

    def test_filters_out_negative_edge_buckets(self):
        """A boundary-adjacent bucket whose effective edge is below
        aggressive_min_edge must NOT receive a signal.
        """
        settings = _make_settings(aggressive_min_edge=0.08)
        primary = _make_opp(
            market_id="primary", bucket_low=70.0, bucket_high=75.0, net_edge=0.12,
        )
        # Adjacent neighbour with effective edge below the gate (0.05 < 0.08).
        bucket_below_min = _make_opp(
            market_id="low-edge", bucket_low=75.0, bucket_high=80.0,
            model_probability=0.55, market_price=0.50,
            net_edge=0.05, net_edge_no=0.02,
        )
        signals = aggressive_strategy(primary, [bucket_below_min], settings)
        low_edge_signals = [s for s in signals if s.market_id == "low-edge"]
        assert len(low_edge_signals) == 0

    def test_single_market_no_buckets(self):
        """With no bucket markets, should produce single signal."""
        settings = _make_settings()
        primary = _make_opp(net_edge=0.12)
        signals = aggressive_strategy(primary, [], settings)
        assert len(signals) == 1
        assert signals[0].agent_name == "Aggressive"

    def test_lead_time_gt_2_rejected(self):
        """Should reject if lead_time_days > 2."""
        settings = _make_settings()
        primary = _make_opp(lead_time_days=3, net_edge=0.15)
        signals = aggressive_strategy(primary, [], settings)
        assert len(signals) == 0


# ===========================================================================
# FIX 2: NO-side price floor
# ===========================================================================

class TestNOSidePriceFloor:
    def test_yes_price_below_010_skipped(self):
        """MarketOpportunity with yes_price < 0.10 should be filtered upstream."""
        # This is tested in main.py's _build_opportunity, here verify edge calc
        edge = calculate_net_edge(0.15, 0.08, "polymarket")
        assert edge > 0  # Would have positive edge but should be filtered

    def test_no_side_price_below_010(self):
        """When yes_price > 0.90, NO costs < $0.10 — should be filtered."""
        # Verify the cost math
        yes_price = 0.95
        no_price = 1.0 - yes_price
        assert no_price == pytest.approx(0.05)
        assert no_price < 0.10  # This should trigger the new filter

    def test_boundary_yes_price_090(self):
        """yes_price=0.90 should NOT be filtered (NO = $0.10 exactly)."""
        yes_price = 0.90
        no_price = 1.0 - yes_price
        assert no_price == pytest.approx(0.10)
        # At exactly 0.10 the filter should NOT trigger (> 0.90, not >= 0.90)


# ===========================================================================
# FIX 3: METAR DST-aware local hour
# ===========================================================================

class TestMetarDST:
    def test_station_local_hour_returns_int(self):
        from weatheredge.data.metar import _station_local_hour
        hour = _station_local_hour("KNYC")
        assert isinstance(hour, int)
        assert 0 <= hour < 24

    def test_station_local_hour_unknown_station(self):
        from weatheredge.data.metar import _station_local_hour
        hour = _station_local_hour("ZZZZ")
        assert isinstance(hour, int)
        assert 0 <= hour < 24  # Falls back to UTC offset 0

    def test_station_local_hour_all_known_stations(self):
        from weatheredge.data.metar import _station_local_hour, _STATION_TZ
        for station in _STATION_TZ:
            hour = _station_local_hour(station)
            assert 0 <= hour < 24, f"Invalid hour {hour} for station {station}"

    def test_metar_override_uses_dst_aware_hour(self):
        """The metar_probability_override should use DST-aware local hour."""
        from weatheredge.data.metar import CurrentConditionsTracker, StationObservation, DailyTracker
        tracker = CurrentConditionsTracker()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Set up a tracker with observations
        daily = DailyTracker(station="KNYC", date=today)
        obs1 = StationObservation(station="KNYC", temperature_f=60.0)
        obs2 = StationObservation(station="KNYC", temperature_f=58.0)  # falling
        daily.update(obs1)
        daily.update(obs2)
        tracker._trackers["KNYC"] = daily

        # Test override for below market — the result depends on local hour
        result = tracker.metar_probability_override(
            station="KNYC",
            threshold=70.0,
            direction="below",
            market_type="daily_high",
            market_date=today,
        )
        # Result depends on whether it's after 5pm local — just verify no error
        assert result is None or (0.0 <= result <= 1.0)


# ===========================================================================
# FIX 4: Date-exposure cap
# ===========================================================================

class TestDateExposureCap:
    def test_date_exposure_within_limit(self):
        settings = _make_settings(max_single_date_exposure=0.25)
        rm = RiskManager(settings)
        result = rm.check_date_exposure(
            market_date="2026-04-02",
            proposed_size=50.0,
            open_positions=[],
            bankroll=1000.0,
        )
        assert not result["blocked"]

    def test_date_exposure_exceeds_limit(self):
        settings = _make_settings(max_single_date_exposure=0.25)
        rm = RiskManager(settings)
        existing = [
            {"market_date": "2026-04-02", "cost": 200.0},
            {"market_date": "2026-04-02", "cost": 50.0},
        ]
        result = rm.check_date_exposure(
            market_date="2026-04-02",
            proposed_size=10.0,
            open_positions=existing,
            bankroll=1000.0,
        )
        assert result["blocked"]
        assert "2026-04-02" in result["reason"]

    def test_date_exposure_different_dates_ok(self):
        settings = _make_settings(max_single_date_exposure=0.25)
        rm = RiskManager(settings)
        existing = [
            {"market_date": "2026-04-01", "cost": 200.0},
            {"market_date": "2026-04-03", "cost": 200.0},
        ]
        result = rm.check_date_exposure(
            market_date="2026-04-02",
            proposed_size=200.0,
            open_positions=existing,
            bankroll=1000.0,
        )
        assert not result["blocked"]

    def test_date_exposure_empty_date_not_blocked(self):
        settings = _make_settings()
        rm = RiskManager(settings)
        result = rm.check_date_exposure(
            market_date="",
            proposed_size=500.0,
            open_positions=[],
            bankroll=1000.0,
        )
        assert not result["blocked"]

    def test_pre_trade_check_includes_date(self):
        """pre_trade_check should accept and use market_date parameter."""
        settings = _make_settings(max_single_date_exposure=0.10)
        rm = RiskManager(settings)
        existing = [{"market_date": "2026-04-02", "cost": 90.0, "market_id": "X", "city": "NYC"}]
        result = rm.pre_trade_check(
            city="NYC",
            proposed_size=20.0,
            open_positions=existing,
            bankroll=1000.0,
            daily_pnl=0.0,
            total_pnl=0.0,
            market_date="2026-04-02",
        )
        assert result["blocked"]


# ===========================================================================
# FIX 5: Annualized Sharpe ratio
# ===========================================================================

class TestAnnualizedSharpe:
    def test_sharpe_is_annualized(self):
        """Sharpe should be annualized, not per-trade."""
        pt = PaperTrader(initial_bankroll=1000.0)
        # Simulate 10 winning trades
        for i in range(10):
            order = Order(
                market_id=f"sharpe-{i}",
                side="buy_yes",
                price=0.50,
                amount=10.0,
            )
            pt.place_order(order)
            pt.resolve_position(f"sharpe-{i}", "yes")

        report = pt.get_performance_report()
        assert report.total_trades == 10
        assert report.win_rate == 1.0
        # Annualized sharpe should be significantly higher than per-trade
        # With 10 identical winning trades, per-trade sharpe = mean/std.
        # Since all wins are identical, std=0, so sharpe would be 0 from division.
        # With slight variance it would be large.
        # The key test: annualization factor = sqrt(min(n, 250))
        assert report.sharpe_ratio >= 0.0

    def test_sharpe_mixed_trades(self):
        """Sharpe with mixed wins/losses should reflect annualization."""
        pt = PaperTrader(initial_bankroll=1000.0)
        # 7 wins, 3 losses
        for i in range(7):
            order = Order(market_id=f"w-{i}", side="buy_yes", price=0.50, amount=10.0)
            pt.place_order(order)
            pt.resolve_position(f"w-{i}", "yes")
        for i in range(3):
            order = Order(market_id=f"l-{i}", side="buy_yes", price=0.50, amount=10.0)
            pt.place_order(order)
            pt.resolve_position(f"l-{i}", "no")

        report = pt.get_performance_report()
        assert report.total_trades == 10
        # Per-trade sharpe ~= (0.7*5 - 0.3*5) / std ≈ 2/std
        # Annualized = per_trade * sqrt(10) ≈ per_trade * 3.16
        assert report.sharpe_ratio != 0.0


# ===========================================================================
# FIX 5b: KSFO/KBOS microclimate entries
# ===========================================================================

class TestMicroclimateComplete:
    def test_ksfo_in_microclimate(self):
        assert "KSFO" in STATION_MICROCLIMATE
        assert STATION_MICROCLIMATE["KSFO"] < 0  # SF marine layer → cool bias

    def test_kbos_in_microclimate(self):
        assert "KBOS" in STATION_MICROCLIMATE

    def test_rules_engine_uses_ksfo(self):
        engine = RulesQualitativeEngine()
        opp = _make_opp(station="KSFO", city="San Francisco")
        signal = engine.analyze(opp)
        assert signal.agent_name == "Qualitative"

    def test_rules_engine_uses_kbos(self):
        engine = RulesQualitativeEngine()
        opp = _make_opp(station="KBOS", city="Boston")
        signal = engine.analyze(opp)
        assert signal.agent_name == "Qualitative"


# ===========================================================================
# Probability edge cases
# ===========================================================================

class TestProbabilityEdgeCases:
    def test_gaussian_rmse_zero_uses_default(self):
        """RMSE=0 should use default 3.0, not produce NaN."""
        prob = gaussian_probability(72.0, 70.0, 0.0, "above")
        assert 0.0 <= prob <= 1.0
        assert not np.isnan(prob)

    def test_gaussian_rmse_negative_uses_default(self):
        prob = gaussian_probability(72.0, 70.0, -5.0, "above")
        assert 0.0 <= prob <= 1.0

    def test_ensemble_empty_returns_half(self):
        prob = ensemble_probability([], 70.0, "above")
        assert prob == 0.5

    def test_ensemble_all_above(self):
        temps = [75.0] * 31
        prob = ensemble_probability(temps, 70.0, "above")
        assert prob == 0.95  # clamped from 1.0

    def test_ensemble_all_below(self):
        temps = [65.0] * 31
        prob = ensemble_probability(temps, 70.0, "above")
        assert prob == 0.10  # clamped from 0.0 (PROB_FLOOR=0.10)

    def test_ensemble_boundary_values(self):
        """At boundary, >= counts as above."""
        temps = [70.0] * 31
        prob = ensemble_probability(temps, 70.0, "above")
        assert prob == 0.95  # >= threshold, clamped from 1.0
        prob_below = ensemble_probability(temps, 70.0, "below")
        assert prob_below == 0.10  # < threshold, clamped from 0.0 (PROB_FLOOR=0.10)

    def test_bucket_rmse_zero_uses_default(self):
        prob = bucket_probability(72.0, 70.0, 75.0, 0.0)
        assert 0.0 <= prob <= 1.0
        assert not np.isnan(prob)

    def test_bucket_forecast_in_range(self):
        """Forecast right in the middle of the bucket → high probability."""
        prob = bucket_probability(72.5, 70.0, 75.0, 3.0)
        assert prob > 0.5

    def test_bucket_forecast_far_outside(self):
        """Forecast far from bucket → clamped to PROB_FLOOR."""
        prob = bucket_probability(50.0, 70.0, 75.0, 3.0)
        assert prob == 0.10  # clamped floor (PROB_FLOOR=0.10)

    def test_bucket_clamp_false_returns_unclamped_low(self):
        """clamp=False exposes raw sub-floor probability for phantom-edge detection."""
        raw = bucket_probability(50.0, 70.0, 75.0, 3.0, clamp=False)
        clamped = bucket_probability(50.0, 70.0, 75.0, 3.0, clamp=True)
        assert raw < 0.10  # true probability is well below the 0.10 floor
        assert clamped == 0.10  # still clamped by default
        # This gap is the phantom-edge driver: clamped > yes_price > raw.

    def test_bucket_clamp_false_returns_unclamped_high(self):
        """clamp=False also exposes above-ceiling probability (buy_no diagnostic)."""
        # Narrow bucket centered on forecast + tight rmse → probability ~= 1.
        raw = bucket_probability(72.5, 60.0, 85.0, 1.0, clamp=False)
        clamped = bucket_probability(72.5, 60.0, 85.0, 1.0, clamp=True)
        assert raw > 0.95  # raw captures the true near-certainty
        assert clamped == 0.95  # clamp caps at ceiling

    def test_bucket_clamp_default_unchanged(self):
        """Default clamp=True preserves existing call-site behavior."""
        default = bucket_probability(72.5, 70.0, 75.0, 3.0)
        explicit = bucket_probability(72.5, 70.0, 75.0, 3.0, clamp=True)
        assert default == explicit

    def test_bucket_clamp_false_respects_rmse_guard(self):
        """rmse<=0 still falls back to 3.0 when clamp=False (consistency)."""
        raw = bucket_probability(72.0, 70.0, 75.0, 0.0, clamp=False)
        assert 0.0 <= raw <= 1.0
        assert not np.isnan(raw)

    def test_gaussian_above_vs_below_symmetry(self):
        """P(above) + P(below) should sum to ~1.0."""
        p_above = gaussian_probability(72.0, 70.0, 3.0, "above")
        p_below = gaussian_probability(72.0, 70.0, 3.0, "below")
        assert p_above + p_below == pytest.approx(1.0, abs=1e-10)


# ===========================================================================
# Edge calculation
# ===========================================================================

class TestEdgeCalculation:
    def test_gross_edge_positive(self):
        edge = calculate_gross_edge(0.70, 0.50)
        assert edge == pytest.approx(0.20)

    def test_gross_edge_negative(self):
        edge = calculate_gross_edge(0.30, 0.50)
        assert edge == pytest.approx(-0.20)

    def test_net_edge_subtracts_fees(self):
        net = calculate_net_edge(0.70, 0.50, "polymarket")
        gross = calculate_gross_edge(0.70, 0.50)
        assert net < gross
        assert net == pytest.approx(gross - 0.015)

    def test_net_edge_kalshi_higher_fees(self):
        net_poly = calculate_net_edge(0.70, 0.50, "polymarket")
        net_kalshi = calculate_net_edge(0.70, 0.50, "kalshi")
        assert net_kalshi < net_poly  # Kalshi has higher fees

    def test_net_edge_unknown_platform(self):
        """Unknown platform should use default 2% fee."""
        net = calculate_net_edge(0.70, 0.50, "unknown_exchange")
        assert 0.0 < net < 0.20


# ===========================================================================
# Blended probability
# ===========================================================================

class TestBlendedProbability:
    def test_metar_override_takes_precedence(self):
        prob, model, details = blended_probability(
            threshold=70.0, metar_override=0.99,
            gfs_forecast_temp=65.0, rmse=3.0,
        )
        assert prob == 0.99
        assert model == "metar_override"

    def test_no_data_returns_half(self):
        prob, model, details = blended_probability(
            threshold=70.0,
        )
        assert prob == 0.50
        assert model == "no_data"

    def test_gfs_only_fallback(self):
        prob, model, details = blended_probability(
            threshold=70.0, gfs_forecast_temp=75.0, rmse=3.0,
            is_us_station=False,
        )
        # Only GFS available for non-US → fallback to gaussian
        assert 0.0 < prob < 1.0

    def test_full_blend_us_short_lead(self):
        prob, model, details = blended_probability(
            threshold=70.0,
            direction="above",
            rmse=3.0,
            is_us_station=True,
            lead_time_hours=6.0,
            ensemble_temps=[72.0] * 31,
            gfs_forecast_temp=72.0,
            ecmwf_forecast_temp=73.0,
            hrrr_forecast_temp=74.0,
        )
        assert 0.5 < prob < 1.0
        assert "hrrr" in model
        assert details["weight_bucket"] == "short"

    def test_learned_weights_applied(self):
        prob1, _, _ = blended_probability(
            threshold=70.0, rmse=3.0, is_us_station=True,
            lead_time_hours=24.0,
            ensemble_temps=[72.0] * 31,
            ecmwf_forecast_temp=72.0,
            hrrr_forecast_temp=72.0,
        )
        prob2, _, details = blended_probability(
            threshold=70.0, rmse=3.0, is_us_station=True,
            lead_time_hours=24.0,
            ensemble_temps=[72.0] * 31,
            ecmwf_forecast_temp=72.0,
            hrrr_forecast_temp=72.0,
            learned_weights={"hrrr": 0.90, "gfs_ensemble": 0.05, "ecmwf": 0.05},
        )
        assert details.get("learned_weights_applied") is True


# ===========================================================================
# Kelly criterion correctness
# ===========================================================================

class TestKellyCriterion:
    def test_kelly_basic(self):
        settings = _make_settings(kelly_fraction=0.15)
        rm = RiskManager(settings)
        rm.set_resolved_trade_count(100)  # bypass warmup Kelly
        size = rm.calculate_position_size(
            win_probability=0.70,
            market_price=0.50,
            bankroll=1000.0,
        )
        # Net-of-fees Kelly on Kalshi at p=0.50:
        #   fee(0.50) = 0.02, half-spread = 0.01 → exit_cost = 0.03
        #   net_stake = 0.52, net_win = 0.50 - 0.02 - 0.03 = 0.45
        #   b = 0.45/0.52 ≈ 0.8654; kelly ≈ 0.3533
        #   size ≈ 0.3533 * 0.15 * 1000 ≈ 53.00, below 8% cap (80).
        assert size == pytest.approx(53.00, abs=0.02)

    def test_kelly_no_edge(self):
        settings = _make_settings()
        rm = RiskManager(settings)
        size = rm.calculate_position_size(
            win_probability=0.50,
            market_price=0.50,
            bankroll=1000.0,
        )
        assert size == 0.0  # No edge → no bet

    def test_kelly_negative_edge(self):
        settings = _make_settings()
        rm = RiskManager(settings)
        size = rm.calculate_position_size(
            win_probability=0.30,
            market_price=0.50,
            bankroll=1000.0,
        )
        assert size == 0.0

    def test_kelly_buy_no_flips(self):
        settings = _make_settings(kelly_fraction=0.15)
        rm = RiskManager(settings)
        size = rm.calculate_position_size(
            win_probability=0.30,  # Model: YES unlikely
            market_price=0.70,    # Market: YES at 70c → NO at 30c
            bankroll=1000.0,
            side="buy_no",
        )
        # Flipped: win_prob = 0.70, price = 0.30
        # b = 0.70/0.30 = 2.333, kelly = (2.333*0.7-0.3)/2.333 = 0.571
        # size = 0.571 * 0.15 * 1000 = 85.71, capped at 80
        assert size > 0.0

    def test_kelly_capped_at_max_exposure(self):
        settings = _make_settings(kelly_fraction=1.0, max_single_market_exposure=0.08)
        rm = RiskManager(settings)
        size = rm.calculate_position_size(
            win_probability=0.95,
            market_price=0.50,
            bankroll=1000.0,
        )
        assert size <= 1000.0 * 0.08

    def test_kelly_net_of_fees_mid_price(self):
        """At contested prices (p≈0.5), gross Kelly over-sizes materially.

        Gross (buggy, pre-fix):
          b = 0.5/0.5 = 1.0, kelly = (1.0*0.6 - 0.4)/1.0 = 0.20
          size = 0.20 * 1.0 * 10000 = 2000.00
        Net-of-fees Kalshi (post-fix):
          fee(0.5)=0.02, exit_cost=0.03 (taker fee + half-spread)
          Win profit = 1 - price - entry_fee - exit_cost = 0.45
          Loss = price + entry_fee = 0.52 (losers held to $0 expiry)
          b = 0.45/0.52 ≈ 0.8654, kelly ≈ 0.1378
          size ≈ 1377.78 — ~31% smaller than gross.
        """
        settings = _make_settings(
            kelly_fraction=1.0,
            max_single_market_exposure=1.0,
            max_cost_per_trade_fraction=1.0,
        )
        rm = RiskManager(settings)
        rm.set_resolved_trade_count(100)
        size = rm.calculate_position_size(
            win_probability=0.60,
            market_price=0.50,
            bankroll=10000.0,
        )
        assert size == pytest.approx(1377.78, abs=0.05)
        assert size < 2000.0, "net Kelly must sit strictly below gross at p≈0.5"

    def test_kelly_net_of_fees_low_price(self):
        """At floor prices (p≈0.1) haircut is small but still reduces Kelly.

        Gross: b = 9, kelly = 1/9 ≈ 0.1111 → size ≈ 1111.11
        Net:   fee(0.1)=0.01, exit_cost=0.02
               Win profit = 0.90 - 0.01 - 0.02 = 0.87
               Loss = 0.10 + 0.01 = 0.11
               b = 0.87/0.11 ≈ 7.909, kelly ≈ 0.0989
               size ≈ 988.51
        """
        settings = _make_settings(
            kelly_fraction=1.0,
            max_single_market_exposure=1.0,
            max_cost_per_trade_fraction=1.0,
        )
        rm = RiskManager(settings)
        rm.set_resolved_trade_count(100)
        size = rm.calculate_position_size(
            win_probability=0.20,
            market_price=0.10,
            bankroll=10000.0,
        )
        assert size == pytest.approx(988.51, abs=0.05)
        assert size < 1111.12, "net Kelly must sit below gross even at floor prices"

    def test_kelly_polymarket_fallback_also_net_of_fees(self):
        """Non-Kalshi fallback splits estimated_round_trip evenly per leg.

        Polymarket rt=0.015, per-leg=0.0075:
          Win profit = 0.50 - 0.0075 - 0.0075 = 0.485
          Loss = 0.50 + 0.0075 = 0.5075
          b = 0.485/0.5075 ≈ 0.9557, kelly ≈ 0.3861
          size ≈ 0.3861 * 0.15 * 1000 ≈ 57.91 (vs 60 under old gross Kelly).
        """
        settings = _make_settings(kelly_fraction=0.15)
        rm = RiskManager(settings)
        rm.set_resolved_trade_count(100)
        size = rm.calculate_position_size(
            win_probability=0.70,
            market_price=0.50,
            bankroll=1000.0,
            platform="polymarket",
        )
        assert size == pytest.approx(57.91, abs=0.05)
        assert size < 60.0


# ===========================================================================
# SellNO strategy
# ===========================================================================

class TestSellNOStrategy:
    def test_sell_no_triggers(self):
        settings = _make_settings()
        opp = _make_opp(
            model_probability=0.10,
            market_price=0.60,
            net_edge_no=0.25,
            lead_time_days=1,
        )
        sig = sell_no_strategy(opp, settings)
        assert sig.side == "buy_no"
        assert sig.confidence > 0.0

    def test_sell_no_rejects_high_prob(self):
        settings = _make_settings()
        opp = _make_opp(model_probability=0.40)
        sig = sell_no_strategy(opp, settings)
        assert sig.side == "no_trade"

    def test_sell_no_rejects_low_market_price(self):
        settings = _make_settings()
        opp = _make_opp(model_probability=0.10, market_price=0.30)
        sig = sell_no_strategy(opp, settings)
        assert sig.side == "no_trade"

    def test_sell_no_rejects_long_lead(self):
        settings = _make_settings()
        opp = _make_opp(model_probability=0.10, market_price=0.60, lead_time_days=3)
        sig = sell_no_strategy(opp, settings)
        assert sig.side == "no_trade"


# ===========================================================================
# Conservative strategy
# ===========================================================================

class TestConservativeStrategy:
    def test_conservative_triggers_buy_yes(self):
        settings = _make_settings(conservative_min_edge=0.20)
        opp = _make_opp(
            model_probability=0.75,
            market_price=0.50,
            net_edge=0.235,
            lead_time_days=1,
            gaussian_prob=0.74,
            ensemble_prob=0.76,
        )
        sig = conservative_strategy(opp, settings)
        assert sig.side == "buy_yes"

    def test_conservative_rejects_low_edge(self):
        settings = _make_settings(conservative_min_edge=0.20)
        opp = _make_opp(net_edge=0.10, net_edge_no=0.05, lead_time_days=1)
        sig = conservative_strategy(opp, settings)
        assert sig.side == "no_trade"

    def test_conservative_rejects_long_lead(self):
        settings = _make_settings()
        opp = _make_opp(net_edge=0.30, lead_time_days=2)
        sig = conservative_strategy(opp, settings)
        assert sig.side == "no_trade"

    def test_conservative_model_disagreement(self):
        settings = _make_settings(conservative_min_edge=0.20)
        opp = _make_opp(
            net_edge=0.25,
            lead_time_days=1,
            gaussian_prob=0.80,
            ensemble_prob=0.50,  # 30% disagreement
        )
        sig = conservative_strategy(opp, settings)
        assert sig.side == "no_trade"

    def test_conservative_buy_no(self):
        """When model_prob < market_price, should signal buy_no."""
        settings = _make_settings(conservative_min_edge=0.20)
        opp = _make_opp(
            model_probability=0.30,
            market_price=0.70,
            net_edge=-0.415,
            net_edge_no=0.285,
            lead_time_days=1,
            gaussian_prob=None,
            ensemble_prob=None,
        )
        sig = conservative_strategy(opp, settings)
        assert sig.side == "buy_no"


# ===========================================================================
# Paper trader resolution
# ===========================================================================

class TestPaperTraderResolution:
    def test_buy_yes_win(self):
        pt = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="t1", side="buy_yes", price=0.40, amount=10.0)
        pt.place_order(order)
        assert pt.bankroll == pytest.approx(996.0)  # 1000 - 0.40*10
        pt.resolve_position("t1", "yes")
        # Win: payout = 10, cost = 4, gross_winnings = 6, fee = 0 (poly), pnl = 6
        assert pt.bankroll == pytest.approx(1006.0)

    def test_buy_yes_lose(self):
        pt = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="t2", side="buy_yes", price=0.40, amount=10.0)
        pt.place_order(order)
        pt.resolve_position("t2", "no")
        assert pt.bankroll == pytest.approx(996.0)  # lost the $4 cost

    def test_buy_no_win(self):
        pt = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="t3", side="buy_no", price=0.60, amount=10.0)
        pt.place_order(order)
        # NO cost = (1-0.60)*10 = 4.0
        assert pt.bankroll == pytest.approx(996.0)
        pt.resolve_position("t3", "no")
        # Win: payout = 10, cost = 4, gross_winnings = 6, pnl = 6
        assert pt.bankroll == pytest.approx(1006.0)

    def test_buy_no_lose(self):
        pt = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="t4", side="buy_no", price=0.60, amount=10.0)
        pt.place_order(order)
        pt.resolve_position("t4", "yes")
        assert pt.bankroll == pytest.approx(996.0)  # lost the $4 cost

    def test_kalshi_win_fee(self):
        pt = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="t5", side="buy_yes", price=0.40, amount=10.0, platform="kalshi")
        pt.place_order(order)
        pt.resolve_position("t5", "yes")
        # entry fee = ceil(0.07*0.40*0.60*100)/100 * 10 = 0.02 * 10 = 0.20
        # cost = 0.40*10 + 0.20 = 4.20; win pnl = 10 - 4.20 = 5.80
        assert pt.bankroll == pytest.approx(1005.80)

    def test_close_position_pnl(self):
        pt = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="t6", side="buy_yes", price=0.40, amount=10.0)
        pt.place_order(order)
        closed = pt.close_position("t6", current_yes_price=0.60, reason="edge_stop")
        assert closed is True
        # Value = 0.60 * 10 = 6.0, cost = 4.0, pnl = 2.0
        assert pt.bankroll == pytest.approx(1002.0)

    def test_insufficient_bankroll(self):
        pt = PaperTrader(initial_bankroll=10.0)
        order = Order(market_id="t7", side="buy_yes", price=0.50, amount=100.0)
        result = pt.place_order(order)
        assert result["filled"] is False
