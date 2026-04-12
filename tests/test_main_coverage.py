"""Coverage boost for main.py — targeting the 151 uncovered lines.

Tests cover: _scan_all_markets, _fetch_all_forecasts, decision loop body,
_build_opportunity, _evaluate_market, Phase 3/4.5 scheduler jobs with data,
and Phase 5 integration paths.
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from weatheredge.agents.base import MarketOpportunity, Signal, TradeOrder


# =========================================================================
# _scan_all_markets — async gather + error handling (lines 229-245)
# =========================================================================

class TestScanAllMarkets:
    @pytest.mark.asyncio
    async def test_both_return_lists(self):
        from weatheredge.main import _scan_all_markets, init_app
        state = init_app()
        state.market_scanner.scan_polymarket = AsyncMock(return_value=[{"ticker": "A"}])
        state.market_scanner.scan_kalshi = AsyncMock(return_value=[{"ticker": "B"}])
        results = await _scan_all_markets(state)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_polymarket_error(self):
        from weatheredge.main import _scan_all_markets, init_app
        state = init_app()
        state.market_scanner.scan_polymarket = AsyncMock(side_effect=Exception("poly down"))
        state.market_scanner.scan_kalshi = AsyncMock(return_value=[{"ticker": "B"}])
        results = await _scan_all_markets(state)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_kalshi_error(self):
        from weatheredge.main import _scan_all_markets, init_app
        state = init_app()
        state.market_scanner.scan_polymarket = AsyncMock(return_value=[{"ticker": "A"}])
        state.market_scanner.scan_kalshi = AsyncMock(side_effect=Exception("kalshi down"))
        results = await _scan_all_markets(state)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_both_error(self):
        from weatheredge.main import _scan_all_markets, init_app
        state = init_app()
        state.market_scanner.scan_polymarket = AsyncMock(side_effect=Exception("err"))
        state.market_scanner.scan_kalshi = AsyncMock(side_effect=Exception("err"))
        results = await _scan_all_markets(state)
        assert len(results) == 0


# =========================================================================
# _fetch_all_forecasts — forecast loop (lines 264-301)
# =========================================================================

class TestFetchAllForecasts:
    @pytest.mark.asyncio
    async def test_fetches_all_models(self):
        from weatheredge.main import _fetch_all_forecasts, init_app
        state = init_app()
        state.settings.target_cities = "NYC"

        with patch("weatheredge.main.fetch_forecast", new_callable=AsyncMock, return_value={"hourly": {}}), \
             patch("weatheredge.main.fetch_ensemble", new_callable=AsyncMock, return_value={"hourly": {}}), \
             patch("weatheredge.main.fetch_ecmwf", new_callable=AsyncMock, return_value={"hourly": {}}), \
             patch("weatheredge.main.store_forecast") as mock_store, \
             patch("weatheredge.main.get_session") as mock_session:
            mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            await _fetch_all_forecasts(state)
            # Should store forecast + ensemble (2 calls for NYC)
            assert mock_store.call_count >= 2

    @pytest.mark.asyncio
    async def test_unknown_city_skipped(self):
        from weatheredge.main import _fetch_all_forecasts, init_app
        state = init_app()
        state.settings.target_cities = "UnknownCity"

        with patch("weatheredge.main.fetch_forecast", new_callable=AsyncMock) as mock_fetch:
            await _fetch_all_forecasts(state)
            mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_forecast_error_continues(self):
        from weatheredge.main import _fetch_all_forecasts, init_app
        state = init_app()
        state.settings.target_cities = "NYC"

        with patch("weatheredge.main.fetch_forecast", new_callable=AsyncMock, side_effect=Exception("API down")), \
             patch("weatheredge.main.fetch_ensemble", new_callable=AsyncMock, side_effect=Exception("API down")), \
             patch("weatheredge.main.get_session") as mock_session:
            mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            # Should not raise — errors are caught per-city
            await _fetch_all_forecasts(state)


# =========================================================================
# fetch_market_data — store price snapshots (lines 209-215)
# =========================================================================

class TestFetchMarketDataStore:
    def test_stores_market_prices(self):
        from weatheredge.main import fetch_market_data, init_app
        state = init_app()
        markets = [{"market_id": "MKT1", "yes_price": 0.55, "volume_24h": 100}]

        with patch("weatheredge.main.asyncio.run", side_effect=[markets, None]), \
             patch("weatheredge.main.get_session") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            state.market_scanner.store_market_price = MagicMock()
            fetch_market_data()

    def test_stores_with_ticker_fallback(self):
        from weatheredge.main import fetch_market_data, init_app
        state = init_app()
        # No market_id, uses ticker
        markets = [{"ticker": "TKR1", "yes_price": 0.45}]

        with patch("weatheredge.main.asyncio.run", side_effect=[markets, None]), \
             patch("weatheredge.main.get_session") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            state.market_scanner.store_market_price = MagicMock()
            fetch_market_data()


# =========================================================================
# Decision loop body — strategies, signals, trades (lines 370-544)
# =========================================================================

class TestDecisionLoopFull:
    def _setup_state_with_market(self):
        from weatheredge.main import init_app
        state = init_app()
        # Fresh forecasts and market prices
        forecasts = {
            "KNYC": {
                "temperature_forecast": 75.0,
                "ensemble_members": [74.0 + i * 0.2 for i in range(31)],
                "model": "ensemble",
                "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        market_prices = {
            "KXHIGHNY": {
                "yes_price": 0.50,
                "no_price": 0.50,
                "volume": 1000,
                "spread": 0.02,
                "timestamp": datetime.now(timezone.utc),
            }
        }
        return state, forecasts, market_prices

    def test_full_decision_loop_with_market(self):
        from weatheredge.main import run_decision_loop, init_app

        state, forecasts, market_prices = self._setup_state_with_market()

        with patch("weatheredge.main._get_weather_data_age_minutes", return_value=5.0), \
             patch("weatheredge.main._read_latest_forecasts", return_value=forecasts), \
             patch("weatheredge.main._read_latest_market_prices", return_value=market_prices):
            run_decision_loop()

    def test_decision_loop_build_opportunity_exception(self):
        from weatheredge.main import run_decision_loop, init_app

        state, forecasts, market_prices = self._setup_state_with_market()

        with patch("weatheredge.main._get_weather_data_age_minutes", return_value=5.0), \
             patch("weatheredge.main._read_latest_forecasts", return_value=forecasts), \
             patch("weatheredge.main._read_latest_market_prices", return_value=market_prices), \
             patch("weatheredge.main._build_opportunity", side_effect=Exception("boom")):
            run_decision_loop()  # Should not raise

    def test_decision_loop_portfolio_aggregation_error(self):
        from weatheredge.main import run_decision_loop, init_app

        state, forecasts, market_prices = self._setup_state_with_market()

        with patch("weatheredge.main._get_weather_data_age_minutes", return_value=5.0), \
             patch("weatheredge.main._read_latest_forecasts", return_value=forecasts), \
             patch("weatheredge.main._read_latest_market_prices", return_value=market_prices):
            state.portfolio_manager.aggregate_signals = MagicMock(side_effect=Exception("agg fail"))
            run_decision_loop()

    def test_decision_loop_trade_execution(self):
        from weatheredge.main import run_decision_loop, init_app

        state, forecasts, market_prices = self._setup_state_with_market()

        trade_order = TradeOrder(
            market_id="KXHIGHNY", side="buy_yes", price=0.50, amount=10.0,
            agent_name="conservative", model_probability=0.65, edge_at_entry=0.15, city="KNYC",
        )

        with patch("weatheredge.main._get_weather_data_age_minutes", return_value=5.0), \
             patch("weatheredge.main._read_latest_forecasts", return_value=forecasts), \
             patch("weatheredge.main._read_latest_market_prices", return_value=market_prices):
            state.portfolio_manager.aggregate_signals = MagicMock(return_value=[trade_order])
            state.paper_trader.place_order = MagicMock(return_value={"filled": True, "trade": {}})
            run_decision_loop()

    def test_decision_loop_trade_rejected(self):
        from weatheredge.main import run_decision_loop, init_app

        state, forecasts, market_prices = self._setup_state_with_market()

        trade_order = TradeOrder(
            market_id="KXHIGHNY", side="buy_yes", price=0.50, amount=10.0,
            agent_name="conservative",
        )

        with patch("weatheredge.main._get_weather_data_age_minutes", return_value=5.0), \
             patch("weatheredge.main._read_latest_forecasts", return_value=forecasts), \
             patch("weatheredge.main._read_latest_market_prices", return_value=market_prices):
            state.portfolio_manager.aggregate_signals = MagicMock(return_value=[trade_order])
            state.paper_trader.place_order = MagicMock(return_value={"filled": False, "reason": "no bankroll"})
            run_decision_loop()

    def test_decision_loop_trade_exception(self):
        from weatheredge.main import run_decision_loop, init_app

        state, forecasts, market_prices = self._setup_state_with_market()

        trade_order = TradeOrder(
            market_id="KXHIGHNY", side="buy_yes", price=0.50, amount=10.0,
        )

        with patch("weatheredge.main._get_weather_data_age_minutes", return_value=5.0), \
             patch("weatheredge.main._read_latest_forecasts", return_value=forecasts), \
             patch("weatheredge.main._read_latest_market_prices", return_value=market_prices):
            state.portfolio_manager.aggregate_signals = MagicMock(return_value=[trade_order])
            state.paper_trader.place_order = MagicMock(side_effect=Exception("exec error"))
            run_decision_loop()


# =========================================================================
# _build_opportunity — blended probability path (lines 633-732)
# =========================================================================

class TestBuildOpportunity:
    def test_known_kalshi_market(self):
        from weatheredge.main import _build_opportunity, init_app
        init_app()
        forecasts = {
            "KNYC": {
                "temperature_forecast": 75.0,
                "ensemble_members": [74.0 + i * 0.2 for i in range(31)],
                "model": "ensemble",
                "ecmwf_temp": 76.0,
            }
        }
        price_data = {"yes_price": 0.55}
        opp = _build_opportunity("KXHIGHNY", price_data, forecasts)
        assert opp is not None
        assert opp.station == "KNYC"
        assert opp.platform == "kalshi"

    def test_unknown_market_returns_none(self):
        from weatheredge.main import _build_opportunity, init_app
        init_app()
        opp = _build_opportunity("UNKNOWN_TICKER", {"yes_price": 0.5}, {})
        assert opp is None

    def test_no_forecast_returns_none(self):
        from weatheredge.main import _build_opportunity, init_app
        init_app()
        opp = _build_opportunity("KXHIGHNY", {"yes_price": 0.5}, {})
        assert opp is None

    def test_bad_price_returns_none(self):
        from weatheredge.main import _build_opportunity, init_app
        init_app()
        forecasts = {"KNYC": {"temperature_forecast": 75.0, "ensemble_members": None, "model": "forecast"}}
        # Price 0.0 is invalid
        assert _build_opportunity("KXHIGHNY", {"yes_price": 0.0}, forecasts) is None
        # Price 1.0 is invalid
        assert _build_opportunity("KXHIGHNY", {"yes_price": 1.0}, forecasts) is None
        # Price None
        assert _build_opportunity("KXHIGHNY", {"yes_price": None}, forecasts) is None

    def test_no_temp_returns_none(self):
        from weatheredge.main import _build_opportunity, init_app
        init_app()
        forecasts = {"KNYC": {"temperature_forecast": None, "ensemble_members": None, "model": "forecast"}}
        opp = _build_opportunity("KXHIGHNY", {"yes_price": 0.5}, forecasts)
        assert opp is None

    def test_blended_probability_fallback(self):
        from weatheredge.main import _build_opportunity, init_app
        init_app()
        forecasts = {
            "KNYC": {
                "temperature_forecast": 75.0,
                "ensemble_members": None,  # No ensemble → Gaussian fallback
                "model": "forecast",
            }
        }
        with patch("weatheredge.main.calculate_blended_probability", side_effect=Exception("blended broke")):
            opp = _build_opportunity("KXHIGHNY", {"yes_price": 0.5}, forecasts)
            # Should fall back to calculate_probability
            assert opp is not None

    def test_all_probability_fails_returns_none(self):
        from weatheredge.main import _build_opportunity, init_app
        init_app()
        forecasts = {"KNYC": {"temperature_forecast": 75.0, "ensemble_members": None, "model": "forecast"}}
        with patch("weatheredge.main.calculate_blended_probability", side_effect=Exception("fail")), \
             patch("weatheredge.main.calculate_probability", side_effect=Exception("also fail")):
            opp = _build_opportunity("KXHIGHNY", {"yes_price": 0.5}, forecasts)
            assert opp is None


# =========================================================================
# _evaluate_market — full trade evaluation pipeline (lines 736-929)
# =========================================================================

@pytest.mark.skip(reason="_evaluate_market is deprecated (raises NotImplementedError)")
class TestEvaluateMarket:
    def test_unknown_market_returns_zero(self):
        from weatheredge.main import _evaluate_market, init_app
        state = init_app()
        result = _evaluate_market(state, "UNKNOWN", {}, {}, 0.10)
        assert result == 0

    def test_no_forecast_returns_zero(self):
        from weatheredge.main import _evaluate_market, init_app
        state = init_app()
        result = _evaluate_market(state, "KXHIGHNY", {"yes_price": 0.5}, {}, 0.10)
        assert result == 0

    def test_bad_price_returns_zero(self):
        from weatheredge.main import _evaluate_market, init_app
        state = init_app()
        forecasts = {"KNYC": {"temperature_forecast": 75.0, "ensemble_members": None}}
        result = _evaluate_market(state, "KXHIGHNY", {"yes_price": 0.0}, forecasts, 0.10)
        assert result == 0

    def test_no_temp_returns_zero(self):
        from weatheredge.main import _evaluate_market, init_app
        state = init_app()
        forecasts = {"KNYC": {"temperature_forecast": None, "ensemble_members": None}}
        result = _evaluate_market(state, "KXHIGHNY", {"yes_price": 0.5}, forecasts, 0.10)
        assert result == 0

    def test_low_edge_returns_zero(self):
        from weatheredge.main import _evaluate_market, init_app
        state = init_app()
        forecasts = {"KNYC": {"temperature_forecast": 71.0, "ensemble_members": None}}
        # Edge will be tiny since forecast is near threshold (70°F)
        result = _evaluate_market(state, "KXHIGHNY", {"yes_price": 0.55}, forecasts, 0.50)
        assert result == 0

    def test_successful_trade(self):
        from weatheredge.main import _evaluate_market, init_app
        state = init_app()
        forecasts = {"KNYC": {
            "temperature_forecast": 85.0,
            "ensemble_members": [85.0 + i * 0.1 for i in range(31)],
        }}
        state.risk_manager.calculate_position_size = MagicMock(return_value=10.0)
        state.risk_manager.pre_trade_check = MagicMock(return_value={"blocked": False})
        state.paper_trader.place_order = MagicMock(return_value={"filled": True})
        result = _evaluate_market(state, "KXHIGHNY", {"yes_price": 0.30}, forecasts, 0.05)
        assert result == 1

    def test_risk_blocked(self):
        from weatheredge.main import _evaluate_market, init_app
        state = init_app()
        forecasts = {"KNYC": {
            "temperature_forecast": 85.0,
            "ensemble_members": [85.0 + i * 0.1 for i in range(31)],
        }}
        state.risk_manager.calculate_position_size = MagicMock(return_value=10.0)
        state.risk_manager.pre_trade_check = MagicMock(return_value={"blocked": True, "reason": "circuit breaker"})
        result = _evaluate_market(state, "KXHIGHNY", {"yes_price": 0.30}, forecasts, 0.05)
        assert result == 0

    def test_vpin_reduce_size(self):
        from weatheredge.main import _evaluate_market, init_app
        state = init_app()
        forecasts = {"KNYC": {
            "temperature_forecast": 85.0,
            "ensemble_members": [85.0 + i * 0.1 for i in range(31)],
        }}
        state.risk_manager.calculate_position_size = MagicMock(return_value=10.0)
        state.risk_manager.pre_trade_check = MagicMock(return_value={"blocked": False, "reduce_size": True})
        state.paper_trader.place_order = MagicMock(return_value={"filled": True})
        result = _evaluate_market(state, "KXHIGHNY", {"yes_price": 0.30}, forecasts, 0.05)
        assert result == 1
        # Verify size was reduced
        order = state.paper_trader.place_order.call_args[0][0]
        assert order.amount == 5.0  # 10.0 * 0.5

    def test_zero_position_size(self):
        from weatheredge.main import _evaluate_market, init_app
        state = init_app()
        forecasts = {"KNYC": {
            "temperature_forecast": 85.0,
            "ensemble_members": [85.0 + i * 0.1 for i in range(31)],
        }}
        state.risk_manager.calculate_position_size = MagicMock(return_value=0.0)
        result = _evaluate_market(state, "KXHIGHNY", {"yes_price": 0.30}, forecasts, 0.05)
        assert result == 0

    def test_order_rejected(self):
        from weatheredge.main import _evaluate_market, init_app
        state = init_app()
        forecasts = {"KNYC": {
            "temperature_forecast": 85.0,
            "ensemble_members": [85.0 + i * 0.1 for i in range(31)],
        }}
        state.risk_manager.calculate_position_size = MagicMock(return_value=10.0)
        state.risk_manager.pre_trade_check = MagicMock(return_value={"blocked": False})
        state.paper_trader.place_order = MagicMock(return_value={"filled": False, "reason": "rejected"})
        result = _evaluate_market(state, "KXHIGHNY", {"yes_price": 0.30}, forecasts, 0.05)
        assert result == 0


# =========================================================================
# Phase 3 jobs with actual data (lines 960-1046)
# =========================================================================

class TestPhase3JobsWithData:
    def test_resolution_check_with_unresolved(self):
        from weatheredge.main import run_resolution_check, init_app
        state = init_app()
        state.resolution_tracker = MagicMock()
        state.resolution_tracker.get_unresolved_markets.return_value = ["MKT1", "MKT2"]
        run_resolution_check()
        state.resolution_tracker.get_unresolved_markets.assert_called_once()

    def test_resolution_check_none_unresolved(self):
        from weatheredge.main import run_resolution_check, init_app
        state = init_app()
        state.resolution_tracker = MagicMock()
        state.resolution_tracker.get_unresolved_markets.return_value = []
        run_resolution_check()

    def test_resolution_check_exception(self):
        from weatheredge.main import run_resolution_check, init_app
        state = init_app()
        state.resolution_tracker = MagicMock()
        state.resolution_tracker.get_unresolved_markets.side_effect = Exception("DB error")
        run_resolution_check()  # Should not raise

    def test_record_calibration_for_market_edge_stop_with_dedupe(self):
        from weatheredge.main import _record_calibration_for_market, init_app

        state = init_app()
        state.calibration_tracker = MagicMock()
        state.calibration_tracker.update_calibration.side_effect = [True, False]

        trade_row = SimpleNamespace(
            trade_id=123,
            model_probability=0.62,
            station="KNYC",
            status="edge_stop",
            side="buy_yes",
            market_date="2026-04-10",
            created_at=datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc),
            agent_name="aggressive",
        )

        with patch("weatheredge.main.get_session") as mock_session:
            mock_db = MagicMock()
            mock_db.execute.return_value.all.return_value = [trade_row]
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)

            first = _record_calibration_for_market(state, "MKT1", "yes")
            second = _record_calibration_for_market(state, "MKT1", "yes")

        assert first == 1
        assert second == 0
        assert state.calibration_tracker.update_calibration.call_count == 2
        first_call = state.calibration_tracker.update_calibration.call_args_list[0].kwargs
        assert first_call["trade_id"] == 123
        assert first_call["actual_outcome"] == 1
        assert first_call["predicted_prob"] == 0.62

    def test_rmse_recalculation_runs(self):
        from weatheredge.main import run_rmse_recalculation, init_app
        state = init_app()
        state.rmse_updater = MagicMock()
        run_rmse_recalculation()
        state.rmse_updater.update_all_rmse.assert_called_once()

    def test_rmse_recalculation_exception(self):
        from weatheredge.main import run_rmse_recalculation, init_app
        state = init_app()
        state.rmse_updater = MagicMock()
        state.rmse_updater.update_all_rmse.side_effect = Exception("fail")
        run_rmse_recalculation()

    def test_error_analysis_with_losses(self):
        from weatheredge.main import run_error_analysis, init_app
        state = init_app()
        state.error_analyzer = MagicMock()

        report = MagicMock()
        report.total_losses = 5
        report.model_errors = 2
        report.timing_errors = 1
        report.edge_compression = 1
        report.adverse_selection = 1
        report.worst_trades = []
        state.error_analyzer.analyze_errors.return_value = report
        state.settings.openrouter_api_key = ""  # No LLM analysis
        run_error_analysis()

    def test_error_analysis_no_losses(self):
        from weatheredge.main import run_error_analysis, init_app
        state = init_app()
        state.error_analyzer = MagicMock()
        report = MagicMock()
        report.total_losses = 0
        state.error_analyzer.analyze_errors.return_value = report
        run_error_analysis()

    def test_error_analysis_with_llm(self):
        from weatheredge.main import run_error_analysis, init_app
        state = init_app()
        state.error_analyzer = MagicMock()
        state.settings.openrouter_api_key = "fake-key"

        report = MagicMock()
        report.total_losses = 25
        report.model_errors = 10
        report.timing_errors = 5
        report.edge_compression = 5
        report.adverse_selection = 5
        report.worst_trades = [{"id": 1}]
        state.error_analyzer.analyze_errors.return_value = report

        llm_report = MagicMock()
        llm_report.findings = ["bias detected", "timing issue"]
        state.error_analyzer.analyze_with_llm.return_value = llm_report

        with patch("weatheredge.notifications.telegram.TelegramBot") as MockBot:
            mock_bot = MagicMock()
            mock_bot.send_alert = AsyncMock()
            MockBot.return_value = mock_bot
            run_error_analysis()

    def test_error_analysis_exception(self):
        from weatheredge.main import run_error_analysis, init_app
        state = init_app()
        state.error_analyzer = MagicMock()
        state.error_analyzer.analyze_errors.side_effect = Exception("analyzer crash")
        run_error_analysis()


# =========================================================================
# Phase 4.5 jobs with data (lines 1062-1107)
# =========================================================================

class TestPhase45JobsWithData:
    def test_wallet_profiling_runs(self):
        from weatheredge.main import run_wallet_profiling, init_app
        state = init_app()
        state.wallet_collector = MagicMock()
        state.wallet_profiler = MagicMock()

        state.wallet_collector.collect_weather_wallets.return_value = [
            {"address": "0xabc", "total_trades": 100, "win_rate": 0.6}
        ]
        state.wallet_profiler.profile_wallet.return_value = "Good trader"

        with patch("weatheredge.main.get_session") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            run_wallet_profiling()

        state.wallet_profiler.profile_wallet.assert_called_once()

    def test_wallet_profiling_no_wallets(self):
        from weatheredge.main import run_wallet_profiling, init_app
        state = init_app()
        state.wallet_collector = MagicMock()
        state.wallet_profiler = MagicMock()
        state.wallet_collector.collect_weather_wallets.return_value = []
        run_wallet_profiling()

    def test_wallet_profiling_exception(self):
        from weatheredge.main import run_wallet_profiling, init_app
        state = init_app()
        state.wallet_collector = MagicMock()
        state.wallet_profiler = MagicMock()
        state.wallet_collector.collect_weather_wallets.side_effect = Exception("CLI error")
        run_wallet_profiling()

    def test_edge_decay_check_runs(self):
        from weatheredge.main import run_edge_decay_check, init_app
        state = init_app()
        state.edge_decay_monitor = MagicMock()
        state.edge_decay_monitor.check_edge_decay.return_value = MagicMock(
            assessment="Edge stable"
        )
        run_edge_decay_check()

    def test_edge_decay_check_exception(self):
        from weatheredge.main import run_edge_decay_check, init_app
        state = init_app()
        state.edge_decay_monitor = MagicMock()
        state.edge_decay_monitor.check_edge_decay.side_effect = Exception("fail")
        run_edge_decay_check()

    def test_morning_briefing_runs(self):
        from weatheredge.main import run_morning_briefing, init_app
        state = init_app()
        state.wallet_query_engine = MagicMock()
        state.wallet_query_engine.query.return_value = "Markets are calm"

        with patch("weatheredge.notifications.formatters.format_morning_briefing", return_value="Briefing text"), \
             patch("weatheredge.notifications.telegram.TelegramBot") as MockBot:
            mock_bot = MagicMock()
            mock_bot.send_alert = AsyncMock()
            MockBot.return_value = mock_bot
            run_morning_briefing()

    def test_morning_briefing_exception(self):
        from weatheredge.main import run_morning_briefing, init_app
        state = init_app()
        state.wallet_query_engine = MagicMock()
        state.wallet_query_engine.query.side_effect = Exception("LLM down")
        run_morning_briefing()


# =========================================================================
# _get_weather_data_age_minutes and DB read helpers (lines 579-625)
# =========================================================================

class TestDBHelpers:
    def test_get_weather_data_age_with_data(self):
        from weatheredge.main import _get_weather_data_age_minutes, init_app
        init_app()

        mock_row = (datetime.now(timezone.utc),)
        with patch("weatheredge.main.get_session") as mock_session:
            mock_db = MagicMock()
            mock_db.query.return_value.order_by.return_value.first.return_value = mock_row
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            age = _get_weather_data_age_minutes()
            assert age is not None
            assert age < 1.0  # Just created

    def test_get_weather_data_age_no_data(self):
        from weatheredge.main import _get_weather_data_age_minutes, init_app
        init_app()

        with patch("weatheredge.main.get_session") as mock_session:
            mock_db = MagicMock()
            mock_db.query.return_value.order_by.return_value.first.return_value = None
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            age = _get_weather_data_age_minutes()
            assert age is None

    def test_get_weather_data_age_exception(self):
        from weatheredge.main import _get_weather_data_age_minutes, init_app
        init_app()

        with patch("weatheredge.main.get_session", side_effect=Exception("DB error")):
            age = _get_weather_data_age_minutes()
            assert age is None

    def test_read_latest_forecasts(self):
        from weatheredge.main import _read_latest_forecasts, init_app
        init_app()

        target_date = "2099-12-31"  # Far future — always passes the date filter

        # Simulate multiple model rows for the same station
        gfs_row = MagicMock()
        gfs_row.station = "KNYC"
        gfs_row.target_date = target_date
        gfs_row.temperature_forecast = 75.0
        gfs_row.ensemble_members = None
        gfs_row.model = "gfs"
        gfs_row.forecast_timestamp = datetime.now(timezone.utc)

        ensemble_row = MagicMock()
        ensemble_row.station = "KNYC"
        ensemble_row.target_date = target_date
        ensemble_row.temperature_forecast = 74.5
        ensemble_row.ensemble_members = json.dumps([74.0, 75.0, 76.0])
        ensemble_row.model = "ensemble"
        ensemble_row.forecast_timestamp = datetime.now(timezone.utc)

        ecmwf_row = MagicMock()
        ecmwf_row.station = "KNYC"
        ecmwf_row.target_date = target_date
        ecmwf_row.temperature_forecast = 76.0
        ecmwf_row.ensemble_members = None
        ecmwf_row.model = "ecmwf"
        ecmwf_row.forecast_timestamp = datetime.now(timezone.utc)

        with patch("weatheredge.main.get_session") as mock_session:
            mock_db = MagicMock()
            # _read_latest_forecasts uses .filter().filter().order_by().limit().all()
            mock_db.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
                gfs_row, ensemble_row, ecmwf_row,
            ]
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            result = _read_latest_forecasts()
            assert "KNYC" in result
            # New nested structure: {station: {date: {fields}}}
            assert target_date in result["KNYC"]
            entry = result["KNYC"][target_date]
            assert entry["temperature_forecast"] == 75.0
            assert entry["ensemble_members"] == [74.0, 75.0, 76.0]
            assert entry["ecmwf_temp"] == 76.0

    def test_read_latest_market_prices(self):
        from weatheredge.main import _read_latest_market_prices, init_app
        init_app()

        mock_row = MagicMock()
        mock_row.market_id = "KXHIGHNY"
        mock_row.yes_price = 0.55
        mock_row.no_price = 0.45
        mock_row.volume = 1000
        mock_row.spread = 0.02
        mock_row.timestamp = datetime.now(timezone.utc)

        with patch("weatheredge.main.get_session") as mock_session:
            mock_db = MagicMock()
            mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_row]
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            result = _read_latest_market_prices()
            assert "KXHIGHNY" in result
            assert result["KXHIGHNY"]["yes_price"] == 0.55


# =========================================================================
# Cross-platform arb in decision loop (lines 526-544)
# =========================================================================

class TestDecisionLoopArb:
    def test_arb_scan_with_opportunities(self):
        from weatheredge.main import run_decision_loop, init_app
        state = init_app()

        mock_arb = MagicMock()
        mock_arb.city = "NYC"
        mock_arb.threshold = 75.0
        mock_arb.poly_price = 0.55
        mock_arb.kalshi_price = 0.45
        mock_arb.spread = 0.10
        mock_arb.direction = "buy_poly_sell_kalshi"

        state.cross_platform_arb.scan = MagicMock(return_value=[mock_arb])

        with patch("weatheredge.main._get_weather_data_age_minutes", return_value=5.0), \
             patch("weatheredge.main._read_latest_forecasts", return_value={}), \
             patch("weatheredge.main._read_latest_market_prices", return_value={}):
            run_decision_loop()

    def test_arb_scan_exception(self):
        from weatheredge.main import run_decision_loop, init_app
        state = init_app()
        state.cross_platform_arb.scan = MagicMock(side_effect=Exception("arb error"))

        with patch("weatheredge.main._get_weather_data_age_minutes", return_value=5.0), \
             patch("weatheredge.main._read_latest_forecasts", return_value={}), \
             patch("weatheredge.main._read_latest_market_prices", return_value={}):
            run_decision_loop()
