"""Integration tests for the WeatherEdge Phase 2 agent pipeline.

Covers:
- test_full_pipeline: mock data → strategies → portfolio manager → risk check → paper trade
- test_decision_loop_runs_strategies: verify all strategies are called in the loop
- test_cross_platform_arb_runs_in_loop: verify arb scanner is called each cycle
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import pytest

import weatheredge.main as main_module
from weatheredge.main import AppState, run_decision_loop
from weatheredge.agents.base import MarketOpportunity, Signal, TradeOrder
from weatheredge.agents.portfolio_manager import PortfolioManager
from weatheredge.agents.risk_manager import RiskManager
from weatheredge.agents.strategies.conservative import conservative_strategy
from weatheredge.agents.strategies.aggressive import aggressive_strategy
from weatheredge.agents.strategies.mean_reversion import mean_reversion_strategy
from weatheredge.config import Settings
from weatheredge.execution.paper_trader import Order, PaperTrader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_settings(**overrides) -> Settings:
    """Build a Settings instance with test-friendly defaults (no .env required)."""
    defaults = dict(
        polymarket_private_key="",
        polymarket_proxy_address="",
        kalshi_api_key="",
        kalshi_private_key="",
        anthropic_api_key="",
        telegram_bot_token="",
        telegram_chat_id="",
        initial_bankroll=1000.0,
        min_edge_percent=10.0,
        kelly_fraction=0.25,
        max_open_positions=20,
        max_single_market_exposure=0.08,
        target_volatility=0.10,
        max_leverage=2.0,
        min_allocation=0.20,
        vpin_danger_threshold=0.70,
        current_risk_tier=100,
        conservative_min_edge=0.20,
        aggressive_min_edge=0.08,
        mean_reversion_lookback_hours=168,
        mean_reversion_min_sigma=3.0,
        enable_conservative_agent=True,
        enable_aggressive_agent=True,
        enable_mean_reversion_agent=True,
        enable_qualitative_agent=True,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_opportunity_with_edge(market_id: str = "KXHIGHNY", net_edge: float = 0.25) -> MarketOpportunity:
    """Build a MarketOpportunity with a specified edge."""
    return MarketOpportunity(
        market_id=market_id,
        platform="kalshi",
        city="KNYC",
        station="KNYC",
        threshold=70.0,
        direction="above",
        market_price=0.40,
        model_probability=0.40 + net_edge + 0.02,  # ensure net_edge after fees
        net_edge=net_edge,
        ensemble_temps=None,
        forecast_temp=82.0,
        rmse=5.0,
        lead_time_days=1,
        gaussian_prob=0.40 + net_edge + 0.02,
        ensemble_prob=0.40 + net_edge + 0.01,
    )


def _make_app_state() -> AppState:
    """Return a freshly initialised AppState with mocked sub-components."""
    state = AppState()
    state.paper_trader = MagicMock(spec=PaperTrader)
    state.paper_trader.bankroll = 1000.0
    state.paper_trader.open_positions = []
    state.paper_trader.daily_pnl = {}
    state.paper_trader.trade_history = []
    state.paper_trader.cancel_expired_orders.return_value = 0
    state.paper_trader.check_limit_fills.return_value = 0
    state.paper_trader.has_open_position.return_value = False
    state.paper_trader.has_reached_daily_entry_limit.return_value = False
    state.paper_trader.place_order.return_value = {
        "filled": True,
        "trade": {"market_id": "KXHIGHNY"},
    }
    state.risk_manager = MagicMock(spec=RiskManager)
    state.risk_manager.calculate_position_size.return_value = 50.0
    state.risk_manager.pre_trade_check.return_value = {"blocked": False}
    state.market_scanner = MagicMock()
    state.portfolio_manager = MagicMock(spec=PortfolioManager)
    state.cross_platform_arb = MagicMock()
    state.cross_platform_arb.scan.return_value = []
    state.fill_monitor = None
    state._price_history = {}
    return state


def _patch_state(state: AppState):
    return patch.object(main_module, "_state", state)


# ---------------------------------------------------------------------------
# test_full_pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """test_full_pipeline: mock data → strategies → portfolio manager → risk check → paper trade."""

    def test_full_pipeline_25_percent_edge(self):
        """End-to-end: 25% edge opportunity flows through all 3 strategies,
        portfolio manager, risk check, and results in a paper trade order."""
        settings = make_settings()
        risk_manager = RiskManager(settings)
        portfolio_manager = PortfolioManager(settings, risk_manager)
        paper_trader = PaperTrader(initial_bankroll=1000.0)

        opp = make_opportunity_with_edge(net_edge=0.25)

        # --- Run all 3 strategies ---
        conservative_sig = conservative_strategy(opp, settings)
        aggressive_sigs = aggressive_strategy(opp, [], settings)
        # Conservative should accept (0.25 > 0.20 threshold)
        assert conservative_sig.side in ("buy_yes", "buy_no"), (
            f"Conservative should trade at 25% edge, got {conservative_sig.side}: {conservative_sig.reasoning}"
        )
        assert conservative_sig.confidence > 0.0

        # Aggressive should produce at least one signal (0.25 > 0.08 threshold)
        assert len(aggressive_sigs) >= 1
        assert aggressive_sigs[0].agent_name == "Aggressive"

        # MeanReversion: construct a price history with a clear 4-sigma crash so it votes
        import numpy as np
        rng = np.random.default_rng(0)
        price_history = list(0.4 + 0.05 * rng.standard_normal(200))
        recent = price_history[-168:]
        mu, sigma = float(np.mean(recent)), float(np.std(recent))
        price_history[-1] = mu - 4.0 * sigma  # 4-sigma crash → buy_yes
        mean_rev_sig = mean_reversion_strategy(opp.market_id, price_history, settings)
        assert mean_rev_sig.side == "buy_yes", (
            f"Mean reversion should vote buy_yes at 4-sigma crash, got {mean_rev_sig.side}"
        )

        # --- Feed signals through portfolio manager ---
        all_signals: list[Signal] = [conservative_sig] + aggressive_sigs + [mean_rev_sig]
        signals_by_market = {opp.market_id: all_signals}
        opportunities = {opp.market_id: opp}

        orders = portfolio_manager.aggregate_signals(
            signals_by_market=signals_by_market,
            opportunities=opportunities,
            bankroll=paper_trader.bankroll,
            open_positions=paper_trader.open_positions,
            daily_pnl=0.0,
            total_pnl=0.0,
        )

        # Portfolio manager should emit at least one order
        assert len(orders) >= 1, (
            f"Expected at least 1 order from portfolio manager, got {len(orders)}"
        )
        order = orders[0]
        assert order.market_id == opp.market_id
        assert order.side in ("buy_yes", "buy_no")
        assert order.amount > 0.0

        # --- Execute via paper trader ---
        paper_order = Order(
            market_id=order.market_id,
            side=order.side,
            price=order.price,
            amount=order.amount,
            agent_name=order.agent_name,
            model_probability=order.model_probability,
            edge_at_entry=order.edge_at_entry,
            city=order.city,
        )
        result = paper_trader.place_order(paper_order)

        # Verify paper trader received and filled the order
        assert result.get("filled") is True, f"Paper trader rejected order: {result.get('reason')}"
        assert len(paper_trader.open_positions) == 1
        assert paper_trader.open_positions[0]["market_id"] == opp.market_id


# ---------------------------------------------------------------------------
# test_decision_loop_runs_strategies
# ---------------------------------------------------------------------------

class TestDecisionLoopRunsStrategies:
    """test_decision_loop_runs_strategies: mock the decision loop, verify all strategies are called."""

    def _make_forecasts(self):
        return {
            "KNYC": {
                "temperature_forecast": 85.0,
                "ensemble_members": None,
                "model": "forecast",
                "forecast_timestamp": datetime.now(timezone.utc),
            }
        }

    def _make_market_prices(self):
        return {
            "KXHIGHNY": {
                "yes_price": 0.70,
                "no_price": 0.30,
                "volume": 5000.0,
                "spread": 0.02,
                "timestamp": datetime.now(timezone.utc),
            }
        }

    def test_all_strategies_called_when_enabled(self):
        """When all strategy toggles are enabled, each strategy function is invoked."""
        state = _make_app_state()
        # portfolio_manager is a mock — make it return a TradeOrder so we can track execution
        trade_order = TradeOrder(
            market_id="KXHIGHNY",
            side="buy_yes",
            price=0.35,
            amount=50.0,
            agent_name="PortfolioManager",
            model_probability=0.70,
            edge_at_entry=0.33,
            city="KNYC",
            platform="kalshi",
        )
        state.portfolio_manager.aggregate_signals.return_value = [trade_order]

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                with patch.object(
                    main_module, "_read_latest_forecasts", return_value=self._make_forecasts()
                ):
                    with patch.object(
                        main_module, "_read_latest_market_prices", return_value=self._make_market_prices()
                    ):
                        with patch.object(
                            main_module, "conservative_strategy", wraps=conservative_strategy
                        ) as mock_cons:
                            with patch.object(
                                main_module, "aggressive_strategy", wraps=aggressive_strategy
                            ) as mock_agg:
                                run_decision_loop()

        # Both conservative and aggressive strategies must have been called
        mock_cons.assert_called_once()
        mock_agg.assert_called_once()

        # Verify they were called with the correct market_id
        cons_opp_arg = mock_cons.call_args[0][0]
        assert cons_opp_arg.market_id == "KXHIGHNY"

        agg_opp_arg = mock_agg.call_args[0][0]
        assert agg_opp_arg.market_id == "KXHIGHNY"

    def test_portfolio_manager_receives_signals(self):
        """portfolio_manager.aggregate_signals is called with signals from the strategies."""
        state = _make_app_state()
        state.portfolio_manager.aggregate_signals.return_value = []

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                with patch.object(
                    main_module, "_read_latest_forecasts", return_value=self._make_forecasts()
                ):
                    with patch.object(
                        main_module, "_read_latest_market_prices", return_value=self._make_market_prices()
                    ):
                        run_decision_loop()

        # aggregate_signals must have been called
        state.portfolio_manager.aggregate_signals.assert_called_once()
        call_kwargs = state.portfolio_manager.aggregate_signals.call_args[1]

        # signals_by_market must contain KXHIGHNY with at least one signal
        signals_by_market = call_kwargs["signals_by_market"]
        assert "KXHIGHNY" in signals_by_market
        assert len(signals_by_market["KXHIGHNY"]) >= 1

    def test_disabled_strategies_not_called(self):
        """Strategies that are disabled via settings toggles must not be invoked."""
        state = _make_app_state()
        state.settings = MagicMock(spec=Settings)
        state.settings.weather_data_ttl_minutes = 120
        state.settings.min_edge_percent = 10.0
        state.settings.conservative_min_edge = 0.20
        state.settings.aggressive_min_edge = 0.08
        state.settings.enable_conservative_agent = False
        state.settings.enable_aggressive_agent = True
        state.settings.enable_mean_reversion_agent = False
        state.settings.mean_reversion_lookback_hours = 168
        state.settings.use_limit_orders = False
        import weatheredge.config as _cfg
        _cfg.MODULES["limit_orders"] = False
        state.portfolio_manager.aggregate_signals.return_value = []

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                with patch.object(
                    main_module, "_read_latest_forecasts", return_value=self._make_forecasts()
                ):
                    with patch.object(
                        main_module, "_read_latest_market_prices", return_value=self._make_market_prices()
                    ):
                        with patch.object(main_module, "conservative_strategy") as mock_cons:
                            with patch.object(
                                main_module, "aggressive_strategy", return_value=[]
                            ) as mock_agg:
                                with patch.object(
                                    main_module, "mean_reversion_strategy"
                                ) as mock_mr:
                                    run_decision_loop()

        mock_cons.assert_not_called()
        mock_agg.assert_called_once()
        mock_mr.assert_not_called()


# ---------------------------------------------------------------------------
# test_cross_platform_arb_runs_in_loop
# ---------------------------------------------------------------------------

class TestCrossPlatformArbRunsInLoop:
    """test_cross_platform_arb_runs_in_loop: verify arb scanner is called each cycle."""

    def test_arb_scanner_called_each_cycle(self):
        """cross_platform_arb.scan() must be called every decision loop iteration."""
        state = _make_app_state()
        state.portfolio_manager.aggregate_signals.return_value = []

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(main_module, "_read_latest_market_prices", return_value={}):
                        run_decision_loop()

        state.cross_platform_arb.scan.assert_called_once()

    def test_arb_scanner_called_with_list_args(self):
        """cross_platform_arb.scan() receives two list arguments (poly_markets, kalshi_markets)."""
        state = _make_app_state()
        state.portfolio_manager.aggregate_signals.return_value = []

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(main_module, "_read_latest_market_prices", return_value={}):
                        run_decision_loop()

        args = state.cross_platform_arb.scan.call_args[0]
        assert len(args) == 2
        poly_arg, kalshi_arg = args
        assert isinstance(poly_arg, list)
        assert isinstance(kalshi_arg, list)

    def test_arb_opportunities_logged(self, caplog):
        """When arb opportunities are found, they are logged at INFO level."""
        from weatheredge.execution.cross_platform import ArbOpportunity

        state = _make_app_state()
        state.portfolio_manager.aggregate_signals.return_value = []
        arb_opp = ArbOpportunity(
            city="NYC",
            date="2026-03-29",
            threshold=70.0,
            poly_price=0.45,
            kalshi_price=0.52,
            spread=0.07,
            direction="buy_poly_sell_kalshi",
        )
        state.cross_platform_arb.scan.return_value = [arb_opp]

        import logging
        with caplog.at_level(logging.INFO):
            with _patch_state(state):
                with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                    with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                        with patch.object(
                            main_module, "_read_latest_market_prices", return_value={}
                        ):
                            run_decision_loop()

        assert any(
            "cross-platform arb" in record.message and "1" in record.message
            for record in caplog.records
        ), "Expected arb opportunity count to be logged"

    def test_arb_not_called_when_stale_data(self):
        """cross_platform_arb.scan() must NOT be called when data is stale."""
        state = _make_app_state()
        stale_age = state.settings.weather_data_ttl_minutes + 1

        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=stale_age
            ):
                run_decision_loop()

        state.cross_platform_arb.scan.assert_not_called()
