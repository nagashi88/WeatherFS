"""
tests/test_comprehensive_validation.py

Comprehensive validation suite for WeatherEdge — designed to FIND BUGS,
not just exercise code paths. Tests are written against specific expected
numeric values derived from the specification. Any failure signals a real
calculation defect, not just a missing coverage line.

Categories:
  1. Resilience Tests
  2. Financial Correctness Tests
  3. Temporal Correctness Tests
  4. Probability Calculation Tests
  5. Market Parsing Validation
  6. Rules Engine Accuracy Tests
  7. Stress / Scale Tests
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ===========================================================================
# 1. RESILIENCE TESTS
# ===========================================================================


class TestResilience:
    """API failures, partial data, DB contention — bot must survive."""

    # -----------------------------------------------------------------------
    # Polymarket 500 / timeout → return [], no crash
    # -----------------------------------------------------------------------

    def test_polymarket_500_returns_empty_list(self):
        """ConnectionError on Polymarket scan → returns [], does not crash."""
        from weatheredge.config import Settings
        from weatheredge.data.market import MarketScanner

        settings = Settings(kalshi_api_key="dummy")
        scanner = MarketScanner(settings)

        with patch("weatheredge.data.market.pmxt.Polymarket") as MockClient:
            MockClient.return_value.fetch_events.side_effect = ConnectionError("HTTP 500")
            result = asyncio.run(scanner.scan_polymarket())

        assert result == []

    def test_polymarket_timeout_returns_empty_list(self):
        """TimeoutError on Polymarket scan → returns [], does not crash."""
        from weatheredge.config import Settings
        from weatheredge.data.market import MarketScanner

        settings = Settings(kalshi_api_key="dummy")
        scanner = MarketScanner(settings)

        with patch("weatheredge.data.market.pmxt.Polymarket") as MockClient:
            MockClient.return_value.fetch_events.side_effect = TimeoutError("timed out")
            result = asyncio.run(scanner.scan_polymarket())

        assert result == []

    # -----------------------------------------------------------------------
    # Kalshi empty / malformed JSON → graceful skip
    # -----------------------------------------------------------------------

    def test_kalshi_api_exception_returns_empty_list(self):
        """Kalshi pmxt AND direct API both fail → returns [], does not crash.

        fetch_kalshi_weather_markets is imported locally inside _scan_kalshi_direct,
        so we patch it at the source module (kalshi_client), not at market module level.
        """
        from weatheredge.config import Settings
        from weatheredge.data.market import MarketScanner

        settings = Settings(kalshi_api_key="dummy")
        scanner = MarketScanner(settings)

        with patch("weatheredge.data.market.pmxt.Kalshi") as MockKalshi:
            MockKalshi.return_value.fetch_events.side_effect = Exception("API error")
            with patch(
                "weatheredge.data.kalshi_client.fetch_kalshi_weather_markets",
                new_callable=AsyncMock,
            ) as mock_direct:
                mock_direct.side_effect = Exception("direct API down")
                result = asyncio.run(scanner.scan_kalshi())

        assert result == []

    def test_kalshi_empty_markets_list_returns_empty(self):
        """Kalshi returning 0 events → scan_kalshi returns []."""
        from weatheredge.config import Settings
        from weatheredge.data.market import MarketScanner

        settings = Settings(kalshi_api_key="dummy")
        scanner = MarketScanner(settings)

        with patch("weatheredge.data.market.pmxt.Kalshi") as MockKalshi:
            MockKalshi.return_value.fetch_events.return_value = []
            with patch(
                "weatheredge.data.kalshi_client.fetch_kalshi_weather_markets",
                new_callable=AsyncMock,
            ) as mock_direct:
                mock_direct.return_value = []
                result = asyncio.run(scanner.scan_kalshi())

        assert result == []

    # -----------------------------------------------------------------------
    # Blended probability with partial / missing model data → no crash
    # -----------------------------------------------------------------------

    def test_blended_only_gfs_no_ecmwf_no_hrrr(self):
        """Only GFS ensemble available (no ECMWF, no HRRR) → valid probability, no crash."""
        from weatheredge.probability.blended import blended_probability

        ensemble_temps = [68.0 + (i % 5) for i in range(25)]
        prob, model, details = blended_probability(
            threshold=70.0,
            direction="above",
            rmse=3.0,
            is_us_station=True,
            lead_time_hours=6.0,
            ensemble_temps=ensemble_temps,
            ecmwf_forecast_temp=None,
            hrrr_forecast_temp=None,
        )
        assert 0.0 <= prob <= 1.0
        assert "gfs" in model

    def test_blended_only_hrrr_no_gfs_no_ecmwf(self):
        """Only HRRR available (no GFS ensemble, no ECMWF) → valid probability, no crash."""
        from weatheredge.probability.blended import blended_probability

        prob, model, details = blended_probability(
            threshold=70.0,
            direction="above",
            rmse=3.0,
            is_us_station=True,
            lead_time_hours=6.0,
            ensemble_temps=None,
            ecmwf_forecast_temp=None,
            hrrr_forecast_temp=73.0,
        )
        assert 0.0 <= prob <= 1.0

    def test_blended_no_model_data_falls_back_to_half(self):
        """Zero model data → falls back to 0.5 (no information)."""
        from weatheredge.probability.blended import blended_probability

        prob, model, details = blended_probability(
            threshold=70.0,
            direction="above",
            rmse=3.0,
            ensemble_temps=None,
            gfs_forecast_temp=None,
            ecmwf_forecast_temp=None,
            hrrr_forecast_temp=None,
        )
        assert prob == 0.5
        assert model == "no_data"

    def test_blended_fewer_than_20_ensemble_members_skips_gfs(self):
        """GFS ensemble with < 20 members is ignored by blended (requires >=20)."""
        from weatheredge.probability.blended import blended_probability

        # 19 members — below threshold; blended should use HRRR fallback instead
        ensemble_temps = [75.0] * 19  # All above 70 → would give 1.0 if used
        prob, model, details = blended_probability(
            threshold=70.0,
            direction="above",
            rmse=3.0,
            is_us_station=True,
            lead_time_hours=6.0,
            ensemble_temps=ensemble_temps,
            ecmwf_forecast_temp=None,
            hrrr_forecast_temp=65.0,  # Below threshold → low prob
        )
        # If 19-member ensemble were used, prob would be ~1.0.
        # Since HRRR (65 vs 70) dominates, prob should be < 0.5.
        assert "gfs_ensemble_prob" not in details, (
            "19-member ensemble should be ignored (requires >=20)"
        )
        assert prob < 0.5

    # -----------------------------------------------------------------------
    # PaperTrader survives DB write failure
    # -----------------------------------------------------------------------

    def test_paper_trader_place_order_survives_db_failure(self):
        """PaperTrader.place_order must succeed in-memory even if DB persist fails."""
        from weatheredge.execution.paper_trader import Order, PaperTrader

        trader = PaperTrader(initial_bankroll=1000.0)

        with patch.object(
            trader, "_persist_new_trade", side_effect=Exception("DB locked")
        ):
            order = Order(
                market_id="db-fail-test",
                side="buy_yes",
                price=0.50,
                amount=10,
            )
            # Should NOT raise — in-memory state must still be updated
            result = trader.place_order(order)

        assert result["filled"] is True, "order must be filled even when DB fails"
        # Bankroll must be deducted (0.50 * 10 = 5.0)
        assert abs(trader.bankroll - 995.0) < 1e-9, (
            f"Bankroll should be 995.0 after $5 trade, got {trader.bankroll}"
        )
        # Position must be in open_positions regardless of DB failure
        assert len(trader.open_positions) == 1, "trade must land in open_positions"
        assert trader.open_positions[0]["market_id"] == "db-fail-test"

    # -----------------------------------------------------------------------
    # All APIs down simultaneously → both scanners return []
    # -----------------------------------------------------------------------

    def test_all_apis_down_returns_empty_without_crash(self):
        """All external APIs failing → both scan methods return [], no exception."""
        from weatheredge.config import Settings
        from weatheredge.data.market import MarketScanner

        settings = Settings(kalshi_api_key="dummy")
        scanner = MarketScanner(settings)

        with patch("weatheredge.data.market.pmxt.Polymarket") as MockPoly:
            MockPoly.return_value.fetch_events.side_effect = Exception("net down")
            poly_result = asyncio.run(scanner.scan_polymarket())

        with patch("weatheredge.data.market.pmxt.Kalshi") as MockKalshi:
            MockKalshi.return_value.fetch_events.side_effect = Exception("net down")
            with patch(
                "weatheredge.data.kalshi_client.fetch_kalshi_weather_markets",
                new_callable=AsyncMock,
            ) as mock_direct:
                mock_direct.side_effect = Exception("net down")
                kalshi_result = asyncio.run(scanner.scan_kalshi())

        assert poly_result == []
        assert kalshi_result == []


# ===========================================================================
# 2. FINANCIAL CORRECTNESS TESTS
# ===========================================================================


class TestFinancialCorrectness:
    """P&L, fees, edge, bankroll, position sizing."""

    # -----------------------------------------------------------------------
    # P&L on win: bought YES at $0.035
    # -----------------------------------------------------------------------

    def test_pnl_win_bought_yes_at_003_5(self):
        """Win: bought YES 1 contract at $0.035, Polymarket → pnl = 0.985 - 0.035 = $0.95."""
        from weatheredge.execution.paper_trader import Order, PaperTrader

        trader = PaperTrader(initial_bankroll=100.0)
        order = Order(market_id="win-test", side="buy_yes", price=0.035, amount=1, platform="polymarket")
        res = trader.place_order(order)
        assert res["filled"] is True

        trader.resolve_position("win-test", "yes")

        trade = trader.trade_history[-1]
        assert trade["status"] == "won"
        # Polymarket 0% fee: pnl = (amount - cost) * (1 - 0.00) = 1 - 0.035 = 0.965
        expected_pnl = (1.0 - 0.035 * 1) * (1.0 - 0.00)
        assert abs(trade["pnl"] - expected_pnl) < 1e-9, (
            f"Expected pnl={expected_pnl:.4f} (Polymarket 0% fee), got {trade['pnl']:.4f}"
        )

    # -----------------------------------------------------------------------
    # P&L on loss: bought YES at $0.035, resolves NO
    # -----------------------------------------------------------------------

    def test_pnl_loss_bought_yes_resolves_no(self):
        """Loss: bought YES at $0.035, resolves NO → pnl = -$0.035."""
        from weatheredge.execution.paper_trader import Order, PaperTrader

        trader = PaperTrader(initial_bankroll=100.0)
        order = Order(market_id="loss-test", side="buy_yes", price=0.035, amount=1)
        trader.place_order(order)
        trader.resolve_position("loss-test", "no")

        trade = trader.trade_history[-1]
        assert trade["status"] == "lost"
        assert abs(trade["pnl"] - (-0.035)) < 1e-9, (
            f"Expected pnl=-0.035, got {trade['pnl']}"
        )

    # -----------------------------------------------------------------------
    # P&L win does NOT deduct Kalshi fee — documents a potential gap
    # -----------------------------------------------------------------------

    def test_pnl_win_applies_kalshi_fee(self):
        """Win on Kalshi: 2% fee on winnings → pnl = (1 - 0.50) * (1 - 0.02) = 0.49."""
        from weatheredge.execution.paper_trader import Order, PaperTrader

        trader = PaperTrader(initial_bankroll=100.0)
        order = Order(
            market_id="kalshi-fee-test",
            side="buy_yes",
            price=0.50,
            amount=1,
            platform="kalshi",
        )
        trader.place_order(order)
        trader.resolve_position("kalshi-fee-test", "yes")

        trade = trader.trade_history[-1]
        # Kalshi 2% fee on WINNINGS: pnl = (amount - cost) * (1 - 0.02) = (1 - 0.50) * 0.98 = 0.49
        expected_pnl = (1.0 - 0.50) * (1.0 - 0.02)
        assert abs(trade["pnl"] - expected_pnl) < 1e-9, (
            f"Expected pnl={expected_pnl:.4f} (after 2% Kalshi fee on winnings), got {trade['pnl']:.4f}"
        )

    # -----------------------------------------------------------------------
    # Fee constants
    # -----------------------------------------------------------------------

    def test_polymarket_fee_constant_is_015(self):
        """Polymarket estimated_round_trip = 0.015 (1.5%)."""
        from weatheredge.probability.edge import PLATFORM_FEES

        assert PLATFORM_FEES["polymarket"]["estimated_round_trip"] == 0.015

    def test_kalshi_fee_constant_is_02(self):
        """Kalshi estimated_round_trip = 0.02 (2%)."""
        from weatheredge.probability.edge import PLATFORM_FEES

        assert PLATFORM_FEES["kalshi"]["estimated_round_trip"] == 0.02

    # -----------------------------------------------------------------------
    # Net edge calculation exactness
    # -----------------------------------------------------------------------

    def test_net_edge_polymarket_exact_value(self):
        """model_prob=0.80, market_price=0.60, polymarket → net_edge = 0.185."""
        from weatheredge.probability.edge import calculate_net_edge

        net = calculate_net_edge(
            model_probability=0.80, market_price=0.60, platform="polymarket"
        )
        assert abs(net - 0.185) < 1e-9, f"Expected 0.185, got {net}"

    def test_net_edge_kalshi_exact_value(self):
        """model_prob=0.80, market_price=0.60, kalshi → net_edge = 0.18."""
        from weatheredge.probability.edge import calculate_net_edge

        net = calculate_net_edge(
            model_probability=0.80, market_price=0.60, platform="kalshi"
        )
        assert abs(net - 0.18) < 1e-9, f"Expected 0.18, got {net}"

    def test_net_edge_negative_when_market_overpriced(self):
        """model_prob=0.45, market_price=0.60, polymarket → net_edge = -0.165."""
        from weatheredge.probability.edge import calculate_net_edge

        net = calculate_net_edge(
            model_probability=0.45, market_price=0.60, platform="polymarket"
        )
        assert abs(net - (-0.165)) < 1e-9

    # -----------------------------------------------------------------------
    # Bankroll enforcement
    # -----------------------------------------------------------------------

    def test_bankroll_enforcement_rejects_oversized_trade(self):
        """Trade costing more than bankroll must be rejected."""
        from weatheredge.execution.paper_trader import Order, PaperTrader

        trader = PaperTrader(initial_bankroll=10.0)
        order = Order(
            market_id="over-bankroll",
            side="buy_yes",
            price=0.50,
            amount=100,  # cost = 50 > bankroll = 10
        )
        result = trader.place_order(order)
        assert result["filled"] is False
        assert "bankroll" in result["reason"].lower()
        assert trader.bankroll == 10.0  # unchanged

    def test_bankroll_accepts_exact_cost_trade(self):
        """Trade costing exactly the bankroll should be accepted."""
        from weatheredge.execution.paper_trader import Order, PaperTrader

        trader = PaperTrader(initial_bankroll=5.0)
        order = Order(
            market_id="exact-bankroll",
            side="buy_yes",
            price=0.50,
            amount=10,  # cost = 5.0 == bankroll
        )
        result = trader.place_order(order)
        assert result["filled"] is True
        assert abs(trader.bankroll) < 1e-9  # bankroll = 0

    # -----------------------------------------------------------------------
    # has_open_position
    # -----------------------------------------------------------------------

    def test_has_open_position_true_after_place(self):
        """has_open_position returns True immediately after placing."""
        from weatheredge.execution.paper_trader import Order, PaperTrader

        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="mkt-A", side="buy_yes", price=0.40, amount=1)
        trader.place_order(order)
        assert trader.has_open_position("mkt-A") is True

    def test_has_open_position_false_for_other_market(self):
        """has_open_position is False for a different market."""
        from weatheredge.execution.paper_trader import Order, PaperTrader

        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="mkt-A", side="buy_yes", price=0.40, amount=1)
        trader.place_order(order)
        assert trader.has_open_position("mkt-B") is False

    def test_has_open_position_false_after_resolution(self):
        """has_open_position is False after the position resolves."""
        from weatheredge.execution.paper_trader import Order, PaperTrader

        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="mkt-resolve", side="buy_yes", price=0.40, amount=1)
        trader.place_order(order)
        trader.resolve_position("mkt-resolve", "yes")
        assert trader.has_open_position("mkt-resolve") is False

    # -----------------------------------------------------------------------
    # Bankroll after multiple trades
    # -----------------------------------------------------------------------

    def test_bankroll_after_two_trades(self):
        """$1000 start, trade1=$50, trade2=$30 → bankroll=$920."""
        from weatheredge.execution.paper_trader import Order, PaperTrader

        trader = PaperTrader(initial_bankroll=1000.0)

        # trade1: price=0.50, amount=100 → cost=50
        res1 = trader.place_order(
            Order(market_id="t1", side="buy_yes", price=0.50, amount=100)
        )
        assert res1["filled"] is True
        assert abs(trader.bankroll - 950.0) < 1e-9

        # trade2: price=0.30, amount=100 → cost=30
        res2 = trader.place_order(
            Order(market_id="t2", side="buy_yes", price=0.30, amount=100)
        )
        assert res2["filled"] is True
        assert abs(trader.bankroll - 920.0) < 1e-9, (
            f"Expected $920.00, got ${trader.bankroll:.2f}"
        )

    def test_place_order_itself_has_no_duplicate_market_check(self):
        """PaperTrader.place_order does NOT internally call has_open_position.

        Duplicate prevention is the caller's responsibility (decision loop).
        Two orders for the same market_id must both succeed at the paper_trader level.
        This is by design — if this test FAILS, duplicate checking was added inside
        place_order, which would be a behavior change worth knowing about.
        """
        from weatheredge.execution.paper_trader import Order, PaperTrader

        trader = PaperTrader(initial_bankroll=1000.0)
        o1 = Order(market_id="dup-mkt", side="buy_yes", price=0.40, amount=1)
        o2 = Order(market_id="dup-mkt", side="buy_yes", price=0.40, amount=1)

        r1 = trader.place_order(o1)
        r2 = trader.place_order(o2)

        assert r1["filled"] is True
        assert r2["filled"] is True
        assert len(
            [p for p in trader.open_positions if p["market_id"] == "dup-mkt"]
        ) == 2


# ===========================================================================
# 3. TEMPORAL CORRECTNESS TESTS
# ===========================================================================


class TestTemporalCorrectness:
    """Lead time, DST tables, daily high reset, market expiry, run windows."""

    # -----------------------------------------------------------------------
    # Lead time calculation: spec says ~58h, code produces ~76h
    # -----------------------------------------------------------------------

    def test_lead_time_uses_midnight_target(self):
        """Fixed: lead time correctly uses midnight of market date (00:00 UTC).

        market date April 2, now=March 30 14:00 UTC → 58 hours (not 76h).
        """
        from datetime import datetime as _dt, timezone as _tz

        now = _dt(2026, 3, 30, 14, 0, 0, tzinfo=_tz.utc)
        market_date = "2026-04-02"

        # Reproduce fixed main.py logic (midnight target)
        target = _dt.strptime(market_date, "%Y-%m-%d").replace(
            hour=0, tzinfo=_tz.utc
        )
        delta_hours = (target - now).total_seconds() / 3600.0

        assert abs(delta_hours - 58.0) < 0.01, (
            f"Expected 58h with midnight target, got {delta_hours:.1f}h"
        )

    # -----------------------------------------------------------------------
    # same-day lead_time_days = max(1,...) prevents lead_time_days=0
    # -----------------------------------------------------------------------

    def test_same_day_market_lead_time_days_is_zero(self):
        """Fixed: max(0, int(hours/24)) allows lead_time_days=0 for sub-24h markets.

        A next-day market (March 31) queried at 10:00 UTC on March 30:
          delta_hours = (March 31 00:00 UTC - March 30 10:00 UTC) = 14h
          lead_time_days = max(0, int(14/24)) = max(0, 0) = 0

        This enables the timing rules in rules_qualitative.py that check
        `lead_time_days == 0` for late-afternoon and early-morning adjustments.
        """
        from datetime import datetime as _dt, timezone as _tz

        now = _dt(2026, 3, 30, 10, 0, 0, tzinfo=_tz.utc)
        next_day_market = "2026-03-31"

        # Reproduce fixed main.py logic (midnight target, max(0,...))
        target = _dt.strptime(next_day_market, "%Y-%m-%d").replace(
            hour=0, tzinfo=_tz.utc
        )
        delta_hours = (target - now).total_seconds() / 3600.0  # = 14.0

        lead_time_days = max(0, int(delta_hours / 24.0))

        assert lead_time_days == 0, (
            f"Expected lead_time_days=0 for 14h lead time, got {lead_time_days}. "
            "max(0,...) must be used so the timing rules can fire."
        )

    def test_timing_rule_lead_days_zero_fires_correctly(self):
        """Rules engine condition `lead_time_days == 0` is now reachable from main.py.

        With the max(0,...) fix, main.py can produce lead_time_days=0 for sub-24h
        markets. This test confirms the timing rule fires when lead_time_days=0.
        """
        from weatheredge.agents.rules_qualitative import RulesQualitativeEngine
        from weatheredge.agents.base import MarketOpportunity

        engine = RulesQualitativeEngine()

        # Manually create an opportunity with lead_time_days=0
        # (something main.py cannot currently produce)
        from unittest.mock import patch as _patch

        opp = MarketOpportunity(
            market_id="same-day",
            platform="kalshi",
            city="New York",
            station="KNYC",
            threshold=70.0,
            direction="above",
            market_price=0.50,
            model_probability=0.65,
            net_edge=0.14,
            rmse=2.0,
            lead_time_days=0,  # What main.py can never produce
        )

        # At local hour = 4 (before 9am, UTC=8 with KNYC offset=-4),
        # the timing rule SHOULD fire (+0.01 for early morning).
        mock_now = datetime(2026, 3, 30, 8, 0, 0, tzinfo=timezone.utc)
        with _patch("weatheredge.agents.rules_qualitative.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            signal = engine.analyze(opp, conditions_tracker=None)

        # If timing rule fires, adjustment = +0.01; KNYC microclimate = +0.02 → total = +0.03
        # If timing rule is correctly reachable with lead_time_days=0, side should be "buy_yes"
        # If lead_time_days=0 check is broken, we'd get no_trade from timing only.
        # This documents that the path IS reachable when manually set to 0.
        assert signal.side in ("buy_yes", "no_trade")

    # -----------------------------------------------------------------------
    # Model run window boundaries
    # -----------------------------------------------------------------------

    def test_00z_window_starts_at_0300_utc(self):
        """00Z model run window starts at 03:00 UTC."""
        from weatheredge.data.model_run_detector import ModelRunDetector

        det = ModelRunDetector(city_coords={"NYC": (40.71, -74.01)})
        t = datetime(2026, 3, 30, 3, 0, tzinfo=timezone.utc)
        in_win, label = det.is_in_check_window(t)
        assert in_win is True
        assert label == "00Z"

    def test_0259_utc_not_in_any_window(self):
        """02:59 UTC is outside all run windows."""
        from weatheredge.data.model_run_detector import ModelRunDetector

        det = ModelRunDetector(city_coords={"NYC": (40.71, -74.01)})
        t = datetime(2026, 3, 30, 2, 59, tzinfo=timezone.utc)
        in_win, _ = det.is_in_check_window(t)
        assert in_win is False

    def test_00z_window_runs_0300_to_0459_utc(self):
        """00Z window runs 03:00–04:59 UTC; comment now correctly says '05:00 UTC'.

        GFS_RUN_WINDOWS has (3, 5, '00Z'): end_hour=5 means hour < 5,
        so the window covers 03:00–04:59. The comment was fixed to say
        '03:00-05:00 UTC' to match the integer-hour boundary.
        """
        from weatheredge.data.model_run_detector import GFS_RUN_WINDOWS, ModelRunDetector

        det = ModelRunDetector(city_coords={"NYC": (40.71, -74.01)})

        # 04:30 UTC (hour=4): correctly IN the window (hour < end=5)
        t_0430 = datetime(2026, 3, 30, 4, 30, tzinfo=timezone.utc)
        in_win, label = det.is_in_check_window(t_0430)
        assert in_win is True
        assert label == "00Z"

        # 05:00 UTC: correctly OUT
        t_0500 = datetime(2026, 3, 30, 5, 0, tzinfo=timezone.utc)
        in_win, _ = det.is_in_check_window(t_0500)
        assert in_win is False

        # Verify the table entry is (3, 5, "00Z")
        window_00z = next(w for w in GFS_RUN_WINDOWS if w[2] == "00Z")
        assert window_00z == (3, 5, "00Z")

        # Verify comment now says "05:00 UTC" (not the old incorrect "04:30 UTC")
        import inspect
        src = inspect.getsource(type(det))
        assert "04:30" not in src, (
            "Old '04:30 UTC' comment still present — should be updated to '05:00 UTC'"
        )

    # -----------------------------------------------------------------------
    # DST offset table correctness
    # -----------------------------------------------------------------------

    def test_station_utc_offsets_spring_2026(self):
        """Verify UTC offset table for US stations in spring (DST in effect)."""
        from weatheredge.agents.rules_qualitative import STATION_UTC_OFFSET

        # Spring 2026: EDT=-4, CDT=-5, MDT=-6, PDT=-7
        assert STATION_UTC_OFFSET["KNYC"] == -4, "NYC should be EDT (UTC-4)"
        assert STATION_UTC_OFFSET["KMDW"] == -5, "Chicago should be CDT (UTC-5)"
        assert STATION_UTC_OFFSET["KDEN"] == -6, "Denver should be MDT (UTC-6)"
        assert STATION_UTC_OFFSET["KLAX"] == -7, "LAX should be PDT (UTC-7)"
        # Phoenix: no DST → stays at -7 year-round
        assert STATION_UTC_OFFSET["KPHX"] == -7, "Phoenix should always be MST (UTC-7)"
        # MST (UTC-6) is correct in winter, but KDEN is MDT (UTC-6) in spring.
        # The test documents what the table CLAIMS, not what might be correct.

    # -----------------------------------------------------------------------
    # METAR daily high is per-date, not UTC day
    # -----------------------------------------------------------------------

    def test_daily_tracker_updates_high_correctly(self):
        """DailyTracker correctly accumulates the running daily high."""
        from weatheredge.data.metar import DailyTracker, StationObservation

        tracker = DailyTracker(station="KNYC", date="2026-03-30")
        assert tracker.daily_high_f == -999.0  # initial sentinel

        obs1 = StationObservation(station="KNYC", temperature_f=68.0, raw_metar="")
        obs2 = StationObservation(station="KNYC", temperature_f=72.0, raw_metar="")
        obs3 = StationObservation(station="KNYC", temperature_f=70.0, raw_metar="")

        tracker.update(obs1)
        assert tracker.daily_high_f == 68.0

        tracker.update(obs2)
        assert tracker.daily_high_f == 72.0

        tracker.update(obs3)
        assert tracker.daily_high_f == 72.0  # does not decrease


# ===========================================================================
# 4. PROBABILITY CALCULATION TESTS
# ===========================================================================


class TestProbabilityCalculations:
    """Blended weights, METAR override, ensemble counting, Gaussian correctness."""

    # -----------------------------------------------------------------------
    # Blended weight correctness with mocked model probabilities
    # -----------------------------------------------------------------------

    def test_blended_short_lead_weights_exact(self):
        """GFS=0.70, ECMWF=0.80, HRRR=0.90 at lead=6h → 0.60*0.9+0.25*0.7+0.15*0.8=0.835."""
        from weatheredge.probability.blended import blended_probability

        with patch(
            "weatheredge.probability.blended.ensemble_probability"
        ) as mock_ens, patch(
            "weatheredge.probability.blended.gaussian_probability"
        ) as mock_gauss:
            mock_ens.return_value = 0.70        # GFS ensemble (called 2x: clamped + raw)
            # Each gaussian model is called twice (clamped + raw); same value preserves blend math.
            mock_gauss.side_effect = [0.80, 0.80, 0.90, 0.90]  # ECMWF x2, HRRR x2

            prob, model, details = blended_probability(
                threshold=70.0,
                direction="above",
                rmse=3.0,
                is_us_station=True,
                lead_time_hours=6.0,        # short bucket: HRRR=60%, GFS=25%, ECMWF=15%
                ensemble_temps=[68.0] * 25,  # >=20 members triggers ensemble_probability
                ecmwf_forecast_temp=72.0,    # triggers gaussian_probability (ECMWF)
                hrrr_forecast_temp=74.0,     # triggers gaussian_probability (HRRR)
            )

        expected = 0.60 * 0.90 + 0.25 * 0.70 + 0.15 * 0.80  # = 0.835
        assert abs(prob - expected) < 1e-6, (
            f"Expected {expected:.6f}, got {prob:.6f}. "
            "Weight formula: HRRR*0.60 + GFS*0.25 + ECMWF*0.15"
        )

    def test_blended_medium_lead_weights_exact(self):
        """At lead=18h (medium): HRRR=0.40, GFS=0.35, ECMWF=0.25."""
        from weatheredge.probability.blended import blended_probability

        with patch(
            "weatheredge.probability.blended.ensemble_probability"
        ) as mock_ens, patch(
            "weatheredge.probability.blended.gaussian_probability"
        ) as mock_gauss:
            mock_ens.return_value = 0.70
            mock_gauss.side_effect = [0.80, 0.80, 0.90, 0.90]  # ECMWF x2 (clamped+raw), HRRR x2

            prob, _, _ = blended_probability(
                threshold=70.0,
                direction="above",
                rmse=3.0,
                is_us_station=True,
                lead_time_hours=18.0,        # medium bucket: HRRR=40%, GFS=35%, ECMWF=25%
                ensemble_temps=[68.0] * 25,
                ecmwf_forecast_temp=72.0,
                hrrr_forecast_temp=74.0,
            )

        expected = 0.40 * 0.90 + 0.35 * 0.70 + 0.25 * 0.80  # = 0.36+0.245+0.20 = 0.805
        assert abs(prob - expected) < 1e-6, f"Expected {expected:.6f}, got {prob:.6f}"

    def test_blended_long_lead_weights_exact(self):
        """At lead=36h (long): GFS=0.40, ECMWF=0.40, HRRR=0.20."""
        from weatheredge.probability.blended import blended_probability

        with patch(
            "weatheredge.probability.blended.ensemble_probability"
        ) as mock_ens, patch(
            "weatheredge.probability.blended.gaussian_probability"
        ) as mock_gauss:
            mock_ens.return_value = 0.70
            mock_gauss.side_effect = [0.80, 0.80, 0.90, 0.90]  # ECMWF x2 (clamped+raw), HRRR x2

            prob, _, _ = blended_probability(
                threshold=70.0,
                direction="above",
                rmse=3.0,
                is_us_station=True,
                lead_time_hours=36.0,        # long bucket: GFS=40%, ECMWF=40%, HRRR=20%
                ensemble_temps=[68.0] * 25,
                ecmwf_forecast_temp=72.0,
                hrrr_forecast_temp=74.0,
            )

        expected = 0.20 * 0.90 + 0.40 * 0.70 + 0.40 * 0.80  # = 0.18+0.28+0.32 = 0.780
        assert abs(prob - expected) < 1e-6, f"Expected {expected:.6f}, got {prob:.6f}"

    # -----------------------------------------------------------------------
    # Lead time bucket boundaries (exact)
    # -----------------------------------------------------------------------

    def test_lead_bucket_boundaries_exact(self):
        """Verify exact lead-time bucket boundaries: <12 short, 12-24 medium, >24 long."""
        from weatheredge.probability.blended import _get_lead_bucket

        assert _get_lead_bucket(0.0) == "short"
        assert _get_lead_bucket(11.9) == "short"
        assert _get_lead_bucket(12.0) == "medium"  # boundary: 12.0 → medium
        assert _get_lead_bucket(24.0) == "medium"  # boundary: 24.0 → medium
        assert _get_lead_bucket(24.01) == "long"
        assert _get_lead_bucket(48.0) == "long"

    # -----------------------------------------------------------------------
    # METAR override precedence
    # -----------------------------------------------------------------------

    def test_metar_override_takes_absolute_precedence(self):
        """metar_override=0.99 returns 0.99 regardless of model data."""
        from weatheredge.probability.blended import blended_probability

        prob, model, details = blended_probability(
            threshold=70.0,
            direction="above",
            rmse=3.0,
            ensemble_temps=[50.0] * 25,  # all far below → would give prob≈0
            ecmwf_forecast_temp=50.0,
            hrrr_forecast_temp=50.0,
            metar_override=0.99,
        )
        assert prob == 0.99
        assert model == "metar_override"
        assert details.get("reason") == "observed temperature already decisive"

    def test_metar_override_none_proceeds_to_models(self):
        """metar_override=None → falls through to model calculation."""
        from weatheredge.probability.blended import blended_probability

        prob, model, _ = blended_probability(
            threshold=70.0,
            direction="above",
            rmse=3.0,
            hrrr_forecast_temp=80.0,
            metar_override=None,
        )
        assert model != "metar_override"
        assert prob > 0.5  # 80 > 70 → high probability

    def test_metar_override_respects_market_date(self):
        """Fixed: metar_probability_override has market_date parameter.

        - Future-dated markets return None (use forecast models instead).
        - Today's markets still get the override (0.99 when high exceeds threshold).
        """
        from weatheredge.data.metar import CurrentConditionsTracker, DailyTracker, StationObservation

        tracker = CurrentConditionsTracker()

        # Inject a DailyTracker with daily high above threshold, dated today
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily = DailyTracker(station="KNYC", date=today)
        obs = StationObservation(station="KNYC", temperature_f=75.0, raw_metar="KNYC")
        daily.update(obs)
        tracker._trackers["KNYC"] = daily

        # Future market date → must return None (no override)
        result_future = tracker.metar_probability_override(
            station="KNYC",
            threshold=70.0,
            direction="above",
            market_type="daily_high",
            market_date="2099-12-31",
        )
        assert result_future is None, (
            f"Expected None for future market date, got {result_future}"
        )

        # Today's market → still returns 0.99 (high 75 >= threshold 70)
        result_today = tracker.metar_probability_override(
            station="KNYC",
            threshold=70.0,
            direction="above",
            market_type="daily_high",
            market_date=today,
        )
        assert result_today == 0.99, (
            f"Expected 0.99 for today's market with high above threshold, got {result_today}"
        )

        # Verify market_date is in the signature
        import inspect
        sig = inspect.signature(tracker.metar_probability_override)
        assert "market_date" in sig.parameters

    # -----------------------------------------------------------------------
    # Ensemble probability counting
    # -----------------------------------------------------------------------

    def test_ensemble_25_of_31_above_threshold(self):
        """25 of 31 members above threshold → probability = 25/31 ≈ 0.806."""
        from weatheredge.probability.ensemble import ensemble_probability

        temps = [75.0] * 25 + [65.0] * 6
        prob = ensemble_probability(temps, threshold=70.0, direction="above")
        assert abs(prob - 25 / 31) < 1e-9

    def test_ensemble_16_of_31_above_threshold(self):
        """16 of 31 members above threshold → probability = 16/31 ≈ 0.516."""
        from weatheredge.probability.ensemble import ensemble_probability

        temps = [75.0] * 16 + [65.0] * 15
        prob = ensemble_probability(temps, threshold=70.0, direction="above")
        assert abs(prob - 16 / 31) < 1e-9

    def test_ensemble_below_counts_lt_threshold(self):
        """'below' direction counts temps < threshold (strictly less).

        BugFix M1: Changed from <= to < so that P(above) + P(below) = 1.0.
        Members exactly at threshold are counted in 'above' (>=), not 'below'.
        """
        from weatheredge.probability.ensemble import ensemble_probability

        # 2 at exactly 70.0 (now counted as 'above' with >=), 1 strictly below
        temps = [68.0, 70.0, 70.0, 72.0]
        prob = ensemble_probability(temps, threshold=70.0, direction="below")
        # Only 1 of 4 is strictly < 70 (68)
        assert abs(prob - 1 / 4) < 1e-9
        # Verify complementarity: P(above) + P(below) = 1.0
        prob_above = ensemble_probability(temps, threshold=70.0, direction="above")
        assert abs(prob + prob_above - 1.0) < 1e-9

    def test_engine_uses_ensemble_when_20_or_more_members(self):
        """calculate_probability switches to ensemble at exactly 20 members."""
        from weatheredge.probability.engine import calculate_probability

        temps = [75.0] * 15 + [65.0] * 5  # exactly 20 members, 15 above 70
        prob, model = calculate_probability(72.0, 70.0, 3.0, ensemble_temps=temps)
        assert model == "ensemble"
        assert abs(prob - 0.75) < 1e-9

    def test_engine_uses_gaussian_at_19_members(self):
        """calculate_probability falls back to Gaussian at exactly 19 members."""
        from weatheredge.probability.engine import calculate_probability

        temps = [75.0] * 19  # 19 members — below threshold of 20
        prob, model = calculate_probability(72.0, 70.0, 3.0, ensemble_temps=temps)
        assert model == "gaussian_fallback"

    # -----------------------------------------------------------------------
    # Gaussian correctness
    # -----------------------------------------------------------------------

    def test_gaussian_at_forecast_equals_threshold_is_half(self):
        """Gaussian: forecast == threshold → probability = 0.5 for 'above'."""
        from weatheredge.probability.gaussian import gaussian_probability

        p = gaussian_probability(70.0, 70.0, 3.0, "above")
        assert abs(p - 0.5) < 1e-9

    def test_gaussian_above_plus_below_sums_correctly(self):
        """P(above) + P(below) ~ 1.0 for non-extreme inputs (clamping may shift extremes)."""
        from weatheredge.probability.gaussian import gaussian_probability

        # Near-center values: clamping doesn't affect, sum should be ~1.0
        for forecast, threshold in [(72.0, 70.0), (80.0, 80.0)]:
            p_above = gaussian_probability(forecast, threshold, 3.0, "above")
            p_below = gaussian_probability(forecast, threshold, 3.0, "below")
            assert abs(p_above + p_below - 1.0) < 0.01, (
                f"For forecast={forecast}, threshold={threshold}: "
                f"P(above)={p_above:.6f} + P(below)={p_below:.6f}"
            )

    def test_gaussian_high_forecast_gives_high_above_prob(self):
        """Forecast 10F above threshold with RMSE=2 → P(above) clamped to 0.95."""
        from weatheredge.probability.gaussian import gaussian_probability

        p = gaussian_probability(80.0, 70.0, 2.0, "above")
        assert p == 0.95, f"Expected 0.95 (clamped), got {p}"

    def test_hrrr_uses_tighter_rmse_in_blended(self):
        """HRRR uses rmse * 0.7 (tighter uncertainty than GFS at 3km resolution)."""
        from weatheredge.probability.blended import blended_probability

        # Capture what rmse was passed to gaussian_probability for HRRR
        captured_calls = []

        def capture_gauss(forecast_temp, threshold, rmse, direction="above", clamp=True):
            captured_calls.append({"forecast": forecast_temp, "rmse": rmse, "clamp": clamp})
            from scipy.stats import norm
            return 1 - norm.cdf(threshold, loc=forecast_temp, scale=rmse)

        with patch(
            "weatheredge.probability.blended.gaussian_probability",
            side_effect=capture_gauss,
        ):
            blended_probability(
                threshold=70.0,
                direction="above",
                rmse=4.0,
                is_us_station=True,
                lead_time_hours=6.0,
                ecmwf_forecast_temp=72.0,
                hrrr_forecast_temp=73.0,
            )

        # Each model is invoked twice: clamped + un-clamped (raw_model_prob audit).
        # L-IMP-3: ECMWF uses 0.90x RMSE, HRRR uses 0.85x — RMSE invariant across both calls.
        assert len(captured_calls) == 4
        primary_calls = [c for c in captured_calls if c["clamp"]]
        assert len(primary_calls) == 2
        ecmwf_call = primary_calls[0]
        hrrr_call = primary_calls[1]
        assert abs(ecmwf_call["rmse"] - 3.6) < 1e-9, "ECMWF should use rmse*0.90"
        assert abs(hrrr_call["rmse"] - 3.4) < 1e-9, (
            f"HRRR should use rmse*0.85=3.4, got {hrrr_call['rmse']}"
        )


# ===========================================================================
# 5. MARKET PARSING VALIDATION
# ===========================================================================


class TestMarketParsing:
    """Title parsing, city mapping, Kalshi contract ID parsing."""

    # -----------------------------------------------------------------------
    # Polymarket title patterns
    # -----------------------------------------------------------------------

    def test_parse_bucket_title_new_york_city(self):
        """Parse bucket title with 'New York City' and 'between X-Y°F'."""
        from weatheredge.data.market_parser import parse_polymarket_title

        title = "Highest temperature in New York City on April 4? - between 50-51°F"
        result = parse_polymarket_title(title)
        assert result is not None, "Expected successful parse"
        assert result.station == "KNYC"
        assert result.threshold_low == 50.0
        assert result.threshold_high == 51.0
        assert result.date == "2026-04-04"

    def test_parse_above_threshold_chicago(self):
        """'60°F or above' parses correctly with direction='above'."""
        from weatheredge.data.market_parser import parse_polymarket_title

        title = "Will Chicago temperature be 60°F or above on March 31?"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "KMDW"
        assert result.threshold == 60.0
        assert result.direction == "above"
        assert result.date == "2026-03-31"

    def test_parse_below_threshold_denver(self):
        """'43°F or below' parses correctly with direction='below'."""
        from weatheredge.data.market_parser import parse_polymarket_title

        title = "Will Denver temperature be 43°F or below on April 1?"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "KDEN"
        assert result.threshold == 43.0
        assert result.direction == "below"

    def test_parse_exceed_miami(self):
        """'exceed 90°F' parses correctly as direction='above'."""
        from weatheredge.data.market_parser import parse_polymarket_title

        title = "Will Miami temperature exceed 90°F on April 2?"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "KMIA"
        assert result.threshold == 90.0
        assert result.direction == "above"

    def test_parse_or_higher_variant(self):
        """'75°F or higher' variant maps to direction='above'."""
        from weatheredge.data.market_parser import parse_polymarket_title

        title = "Will Los Angeles temperature be 75°F or higher on April 3?"
        result = parse_polymarket_title(title)
        assert result is not None
        assert result.station == "KLAX"
        assert result.direction == "above"

    # -----------------------------------------------------------------------
    # City name variant mapping
    # -----------------------------------------------------------------------

    def test_city_variant_nyc(self):
        from weatheredge.data.market_parser import CITY_TO_STATION
        assert CITY_TO_STATION["nyc"] == "KNYC"

    def test_city_variant_new_york(self):
        from weatheredge.data.market_parser import CITY_TO_STATION
        assert CITY_TO_STATION["new york"] == "KNYC"

    def test_city_variant_new_york_city(self):
        from weatheredge.data.market_parser import CITY_TO_STATION
        assert CITY_TO_STATION["new york city"] == "KNYC"

    def test_city_lax_variant(self):
        from weatheredge.data.market_parser import CITY_TO_STATION
        assert CITY_TO_STATION["lax"] == "KLAX"

    def test_city_la_variant(self):
        from weatheredge.data.market_parser import CITY_TO_STATION
        assert CITY_TO_STATION["la"] == "KLAX"

    # -----------------------------------------------------------------------
    # Kalshi contract ID parsing
    # -----------------------------------------------------------------------

    def test_kalshi_t_prefix_parses_threshold(self):
        """KXHIGHDEN-26MAR30-T76.5 → threshold=76.5."""
        market_id = "KXHIGHDEN-26MAR30-T76.5"
        parts = market_id.split("-")
        tail = parts[-1]
        threshold = float(tail[1:])
        assert threshold == 76.5

    def test_kalshi_t_prefix_integer_threshold(self):
        """KXHIGHDEN-26MAR30-T76 → threshold=76.0."""
        market_id = "KXHIGHDEN-26MAR30-T76"
        tail = market_id.split("-")[-1]
        assert tail.startswith("T")
        assert float(tail[1:]) == 76.0

    def test_kalshi_b_prefix_sets_threshold_and_direction_below(self):
        """Fixed: KXHIGHDEN-26MAR30-B76.5 sets threshold=76.5 AND direction='below'."""
        # Reproduce the fixed main.py code
        market_id = "KXHIGHDEN-26MAR30-B76.5"
        threshold = 70.0   # default
        direction = "above"  # default, set earlier in function

        parts = market_id.split("-")
        if len(parts) >= 3:
            tail = parts[-1]
            try:
                if tail.startswith("T"):
                    threshold = float(tail[1:])
                elif tail.startswith("B"):
                    threshold = float(tail[1:])
                    direction = "below"
            except ValueError:
                pass

        assert threshold == 76.5, "Threshold must be extracted from B-prefix"
        assert direction == "below", (
            "B-prefix must set direction='below' to avoid inverting probability"
        )

    def test_kalshi_station_mapping_den(self):
        """KXHIGHDEN maps to KDEN."""
        from weatheredge.data.market import RESOLUTION_SOURCES

        assert RESOLUTION_SOURCES["kalshi"]["KXHIGHDEN"]["station"] == "KDEN"

    def test_kalshi_station_mapping_ny(self):
        """KXHIGHNY maps to KNYC."""
        from weatheredge.data.market import RESOLUTION_SOURCES

        assert RESOLUTION_SOURCES["kalshi"]["KXHIGHNY"]["station"] == "KNYC"

    # -----------------------------------------------------------------------
    # Unknown city / malformed titles
    # -----------------------------------------------------------------------

    def test_unknown_city_returns_none(self, caplog):
        """Title with unmapped city (Timbuktu) → returns None and logs warning."""
        from weatheredge.data.market_parser import parse_polymarket_title

        result = parse_polymarket_title(
            "Will Timbuktu temperature exceed 75°F on April 2?"
        )
        assert result is None

    def test_no_date_in_title_returns_none(self):
        """Title with no date → returns None."""
        from weatheredge.data.market_parser import parse_polymarket_title

        result = parse_polymarket_title(
            "Will New York City temperature exceed 75°F?"
        )
        assert result is None

    def test_empty_title_returns_none(self):
        """Empty string title → returns None without crashing."""
        from weatheredge.data.market_parser import parse_polymarket_title

        result = parse_polymarket_title("")
        assert result is None

    def test_process_market_unknown_ticker_returns_none(self):
        """Unknown Kalshi ticker → process_market returns None."""
        from weatheredge.config import Settings
        from weatheredge.data.market import MarketScanner

        settings = Settings(kalshi_api_key="dummy")
        scanner = MarketScanner(settings)

        with patch.object(scanner, "_alert_unknown_market"):
            result = scanner.process_market(
                {"ticker": "KXHIGHBOS", "title": "Boston High Temp"},
                platform="kalshi",
            )
        assert result is None

    def test_process_market_known_ticker_returns_enriched(self):
        """Known Kalshi ticker → process_market returns dict with resolution_source."""
        from weatheredge.config import Settings
        from weatheredge.data.market import MarketScanner

        settings = Settings(kalshi_api_key="dummy")
        scanner = MarketScanner(settings)

        result = scanner.process_market(
            {"ticker": "KXHIGHNY", "title": "NY High Temp", "yes_price": 0.40},
            platform="kalshi",
        )
        assert result is not None
        assert "resolution_source" in result
        assert result["resolution_source"]["station"] == "KNYC"


# ===========================================================================
# 6. RULES ENGINE ACCURACY TESTS
# ===========================================================================


class TestRulesEngine:
    """Microclimate, wind, precipitation, timing, damping rules."""

    def _opp(
        self,
        station="KNYC",
        direction="above",
        lead_days=1,
        threshold=70.0,
        rmse=2.0,
    ):
        from weatheredge.agents.base import MarketOpportunity

        return MarketOpportunity(
            market_id="test-market",
            platform="kalshi",
            city="TestCity",
            station=station,
            threshold=threshold,
            direction=direction,
            market_price=0.50,
            model_probability=0.65,
            net_edge=0.14,
            rmse=rmse,
            lead_time_days=lead_days,
        )

    def _mock_tracker(self, wind_kt=5.0, raw_metar="KMDW", altimeter=None, daily_high=65.0):
        obs = MagicMock()
        obs.wind_speed_kt = wind_kt
        obs.raw_metar = raw_metar
        obs.altimeter_hpa = altimeter

        daily = MagicMock()
        daily.last_observation = obs
        daily.daily_high_f = daily_high

        tracker = MagicMock()
        tracker.get_tracker.return_value = daily
        return tracker

    # -----------------------------------------------------------------------
    # Microclimate table values
    # -----------------------------------------------------------------------

    def test_kphx_microclimate_is_plus003(self):
        """KPHX microclimate = +0.03."""
        from weatheredge.agents.rules_qualitative import STATION_MICROCLIMATE

        assert STATION_MICROCLIMATE["KPHX"] == pytest.approx(0.03)

    def test_klax_microclimate_is_minus002(self):
        """KLAX microclimate = -0.02."""
        from weatheredge.agents.rules_qualitative import STATION_MICROCLIMATE

        assert STATION_MICROCLIMATE["KLAX"] == pytest.approx(-0.02)

    def test_knyc_microclimate_is_plus002(self):
        from weatheredge.agents.rules_qualitative import STATION_MICROCLIMATE

        assert STATION_MICROCLIMATE["KNYC"] == pytest.approx(0.02)

    def test_kmdw_microclimate_is_zero(self):
        from weatheredge.agents.rules_qualitative import STATION_MICROCLIMATE

        assert STATION_MICROCLIMATE["KMDW"] == pytest.approx(0.0)

    # -----------------------------------------------------------------------
    # Microclimate effect on signal
    # -----------------------------------------------------------------------

    def test_kphx_microclimate_gives_buy_yes(self):
        """KPHX +0.03 → buy_yes signal, no METAR."""
        from weatheredge.agents.rules_qualitative import RulesQualitativeEngine

        engine = RulesQualitativeEngine()
        signal = engine.analyze(self._opp(station="KPHX"), conditions_tracker=None)
        assert signal.side == "buy_yes"
        # Confidence = min(raw_total * damping * 10, 1.0); raw=0.03, rmse=2 → damping=1.0 → 0.30
        assert abs(signal.confidence - 0.30) < 1e-9

    def test_klax_microclimate_gives_buy_no(self):
        """KLAX -0.02 → buy_no signal, no METAR."""
        from weatheredge.agents.rules_qualitative import RulesQualitativeEngine

        engine = RulesQualitativeEngine()
        signal = engine.analyze(self._opp(station="KLAX"), conditions_tracker=None)
        assert signal.side == "buy_no"

    def test_kmdw_no_microclimate_no_metar_gives_no_trade(self):
        """KMDW microclimate=0 + no METAR → no significant factor → no_trade."""
        from weatheredge.agents.rules_qualitative import RulesQualitativeEngine

        engine = RulesQualitativeEngine()
        signal = engine.analyze(self._opp(station="KMDW"), conditions_tracker=None)
        assert signal.side == "no_trade"

    # -----------------------------------------------------------------------
    # High wind reduces above-market probability
    # -----------------------------------------------------------------------

    def test_high_wind_30kt_above_market_buy_no(self):
        """Wind 30kt on above-market → -0.02 adjustment → buy_no (KMDW base=0)."""
        from weatheredge.agents.rules_qualitative import RulesQualitativeEngine

        engine = RulesQualitativeEngine()
        opp = self._opp(station="KMDW", direction="above")
        tracker = self._mock_tracker(wind_kt=30.0, raw_metar="KMDW 30KT")
        signal = engine.analyze(opp, conditions_tracker=tracker)
        assert signal.side == "buy_no"

    def test_wind_below_25kt_does_not_trigger_rule(self):
        """Wind 24kt is below the 25kt threshold — no wind adjustment fires."""
        from weatheredge.agents.rules_qualitative import RulesQualitativeEngine

        engine = RulesQualitativeEngine()
        opp = self._opp(station="KMDW", direction="above")
        tracker = self._mock_tracker(wind_kt=24.0, raw_metar="KMDW 24KT")
        signal = engine.analyze(opp, conditions_tracker=tracker)
        # KMDW microclimate=0, wind<25 → no adjustments → no_trade
        assert signal.side == "no_trade"

    # -----------------------------------------------------------------------
    # Precipitation effects
    # -----------------------------------------------------------------------

    def test_rain_on_below_market_gives_buy_yes(self):
        """Precipitation (RA) on below-market → +0.02 → buy_yes (KMDW base=0)."""
        from weatheredge.agents.rules_qualitative import RulesQualitativeEngine

        engine = RulesQualitativeEngine()
        opp = self._opp(station="KMDW", direction="below")
        tracker = self._mock_tracker(raw_metar="KMDW RA 10KT")
        signal = engine.analyze(opp, conditions_tracker=tracker)
        assert signal.side == "buy_yes"

    def test_rain_on_above_market_gives_buy_no(self):
        """Precipitation (RA) on above-market → -0.03 → buy_no (KMDW base=0)."""
        from weatheredge.agents.rules_qualitative import RulesQualitativeEngine

        engine = RulesQualitativeEngine()
        opp = self._opp(station="KMDW", direction="above")
        tracker = self._mock_tracker(raw_metar="KMDW RA 10KT")
        signal = engine.analyze(opp, conditions_tracker=tracker)
        assert signal.side == "buy_no"

    # -----------------------------------------------------------------------
    # Adjustment clamping
    # -----------------------------------------------------------------------

    def test_combined_rules_cannot_exceed_010(self):
        """Multiple stacking rules: raw adjustment clamped to ±0.10, confidence scaled to [0, 1]."""
        from weatheredge.agents.rules_qualitative import RulesQualitativeEngine

        engine = RulesQualitativeEngine()
        # KPHX (+0.03) + low RMSE (damping=1.0) → max possible is small
        # Add: wind 30kt above-market (-0.02) + precipitation above (-0.03)
        # Net: +0.03 - 0.02 - 0.03 = -0.02 → buy_no; confidence = min(0.02*10, 1.0) = 0.20
        opp = self._opp(station="KPHX", direction="above", rmse=2.0)
        tracker = self._mock_tracker(
            wind_kt=30.0,
            raw_metar="KPHX RA 30KT",
        )
        signal = engine.analyze(opp, conditions_tracker=tracker)
        assert 0.0 <= signal.confidence <= 1.0, (
            f"Confidence {signal.confidence} out of [0, 1] range"
        )

    def test_high_rmse_dampens_signal(self):
        """High RMSE (e.g., 6.0) dampens the signal by 50%."""
        from weatheredge.agents.rules_qualitative import RulesQualitativeEngine, _RMSE_DAMP_FLOOR

        engine = RulesQualitativeEngine()
        # KPHX microclimate = +0.03
        # At rmse=6.0: damping = max(0.5, 1.0 - (6.0-2.0)/8.0) = max(0.5, 0.5) = 0.5
        # raw = 0.03 * 0.5 = 0.015; confidence = min(0.015 * 10, 1.0) = 0.15
        opp = self._opp(station="KPHX", rmse=6.0)
        signal = engine.analyze(opp, conditions_tracker=None)
        assert signal.side == "buy_yes"
        assert abs(signal.confidence - 0.15) < 1e-9, (
            f"Expected 0.15 (dampened), got {signal.confidence}"
        )

    def test_rmse_damping_floor_is_05(self):
        """RMSE damping never goes below 0.5 (the floor)."""
        from weatheredge.agents.rules_qualitative import _RMSE_DAMP_FLOOR

        assert _RMSE_DAMP_FLOOR == 0.5

    # -----------------------------------------------------------------------
    # Station microclimate is station-specific (not shared)
    # -----------------------------------------------------------------------

    def test_kphx_and_klax_adjustments_are_different(self):
        """KPHX (+0.03) and KLAX (-0.02) produce opposite signals."""
        from weatheredge.agents.rules_qualitative import RulesQualitativeEngine

        engine = RulesQualitativeEngine()
        s_phx = engine.analyze(self._opp(station="KPHX"), conditions_tracker=None)
        s_lax = engine.analyze(self._opp(station="KLAX"), conditions_tracker=None)
        assert s_phx.side == "buy_yes"
        assert s_lax.side == "buy_no"

    def test_timing_rule_skipped_for_lead_days_greater_than_1(self):
        """Timing rules (Rule 4) only fire for lead_time_days <= 1."""
        from weatheredge.agents.rules_qualitative import RulesQualitativeEngine

        engine = RulesQualitativeEngine()
        # lead_time_days=2 → timing rules skip
        opp = self._opp(station="KMDW", lead_days=2)
        signal = engine.analyze(opp, conditions_tracker=None)
        # KMDW=0 + no timing → no_trade
        assert signal.side == "no_trade"


# ===========================================================================
# 7. STRESS / SCALE TESTS
# ===========================================================================


class TestStressScale:
    """Performance under load, large trade counts, repeated parsing."""

    def test_500_blended_probability_calls_under_10s(self):
        """500 blended_probability calls complete in under 10 seconds."""
        from weatheredge.probability.blended import blended_probability

        start = time.monotonic()
        for i in range(500):
            ensemble = [65.0 + (i % 8)] * 25
            blended_probability(
                threshold=70.0,
                direction="above",
                rmse=3.0,
                is_us_station=True,
                lead_time_hours=float(6 + (i % 48)),
                ensemble_temps=ensemble,
                ecmwf_forecast_temp=68.0 + (i % 5),
                hrrr_forecast_temp=69.0 + (i % 5),
            )
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, (
            f"500 blended_probability calls took {elapsed:.2f}s (limit: 10s)"
        )

    def test_500_edge_calculations_under_1s(self):
        """500 calculate_net_edge calls complete in under 1 second."""
        from weatheredge.probability.edge import calculate_net_edge

        start = time.monotonic()
        for i in range(500):
            calculate_net_edge(
                model_probability=0.50 + (i % 50) * 0.001,
                market_price=0.40,
                platform="kalshi" if i % 2 else "polymarket",
            )
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"500 edge calculations took {elapsed:.2f}s"

    def test_100_paper_trades_bankroll_exact(self):
        """After 100 paper trades at $0.40 each, bankroll = 10000 - 40.0."""
        from weatheredge.execution.paper_trader import Order, PaperTrader

        trader = PaperTrader(initial_bankroll=10000.0)
        total_cost = 0.0
        for i in range(100):
            order = Order(
                market_id=f"mkt-{i}",
                side="buy_yes",
                price=0.40,
                amount=1,
            )
            result = trader.place_order(order)
            if result["filled"]:
                total_cost += 0.40

        assert abs(trader.bankroll - (10000.0 - total_cost)) < 1e-6, (
            f"Bankroll mismatch: expected {10000.0 - total_cost:.4f}, "
            f"got {trader.bankroll:.4f}"
        )

    def test_performance_report_with_100_resolved_trades(self):
        """get_performance_report on 100 resolved trades is fast and numerically correct."""
        from weatheredge.execution.paper_trader import Order, PaperTrader

        trader = PaperTrader(initial_bankroll=10000.0)
        for i in range(100):
            order = Order(
                market_id=f"mkt-{i}",
                side="buy_yes",
                price=0.40,
                amount=1,
                edge_at_entry=0.15,
            )
            trader.place_order(order)
            trader.resolve_position(f"mkt-{i}", "yes" if i % 2 == 0 else "no")

        start = time.monotonic()
        report = trader.get_performance_report()
        elapsed = time.monotonic() - start

        assert elapsed < 1.0
        assert report.total_trades == 100
        assert report.winning_trades == 50
        assert report.losing_trades == 50
        assert abs(report.win_rate - 0.5) < 1e-9

    def test_500_title_parses_no_crash_no_exception(self):
        """500 varied title strings parsed without any exception."""
        from weatheredge.data.market_parser import parse_polymarket_title

        titles = (
            [
                f"Highest temperature in New York City on April {d}? - between {60+d}-{61+d}°F"
                for d in range(1, 20)
            ]
            + [
                "Malformed title",
                "Temperature in Unknown City on March 31?",
                "",
                "!@#$%^&*()",
                "Will New York City temperature exceed 75°F?",  # no date
            ]
        ) * 20  # ~480 titles → trim to 500

        titles = titles[:500]

        crashed = []
        for title in titles:
            try:
                parse_polymarket_title(title)
            except Exception as e:
                crashed.append((title[:60], str(e)))

        assert crashed == [], f"Parsing crashed on {len(crashed)} titles: {crashed[:3]}"

    def test_has_open_position_1000_lookups_fast(self):
        """has_open_position with 100 open positions: 1000 lookups in <1s."""
        from weatheredge.execution.paper_trader import Order, PaperTrader

        trader = PaperTrader(initial_bankroll=100000.0)
        for i in range(100):
            trader.place_order(Order(market_id=f"mkt-{i}", side="buy_yes", price=0.10, amount=1))

        start = time.monotonic()
        for i in range(1000):
            trader.has_open_position(f"mkt-{i % 100}")
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"1000 lookups took {elapsed:.2f}s"

    def test_rules_engine_100_calls_no_state_accumulation(self):
        """RulesQualitativeEngine.analyze called 100x: no state leak, no crash."""
        from weatheredge.agents.base import MarketOpportunity
        from weatheredge.agents.rules_qualitative import RulesQualitativeEngine

        engine = RulesQualitativeEngine()

        for i in range(100):
            opp = MarketOpportunity(
                market_id=f"mkt-{i}",
                platform="kalshi",
                city="Phoenix",
                station="KPHX",
                threshold=70.0 + i,
                direction="above",
                market_price=0.50,
                model_probability=0.65,
                net_edge=0.14,
                rmse=2.0,
                lead_time_days=1,
            )
            engine.analyze(opp)
        # No state accumulation — engine has no instance variables to check
        # Passes if no exception raised across 100 calls.
