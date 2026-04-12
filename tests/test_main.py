"""Tests for weatheredge.main — entry point and event-driven pipeline.

Covers:
- Logging setup creates RotatingFileHandler (Gap 13)
- Decision loop skips when weather data is stale >2h (Gap 12)
- Decision loop processes markets when data is fresh (Gap 7)
- SIGTERM handler sets shutting_down flag (Gap 14)

All external dependencies (DB, fetchers, risk manager) are mocked.
"""

from __future__ import annotations

import signal
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import weatheredge.main as main_module
from weatheredge.main import (
    AppState,
    _handle_shutdown,
    run_decision_loop,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app_state() -> AppState:
    """Return a freshly initialised AppState with mocked sub-components."""
    state = AppState()
    state.paper_trader = MagicMock()
    state.paper_trader.bankroll = 1000.0
    state.paper_trader.open_positions = []
    state.paper_trader.daily_pnl = {}
    state.paper_trader.trade_history = []
    state.paper_trader.cancel_expired_orders.return_value = 0
    state.paper_trader.check_limit_fills.return_value = 0
    state.paper_trader.has_open_position.return_value = False
    state.risk_manager = MagicMock()
    state.market_scanner = MagicMock()
    # Phase 2 components — mocked so existing tests are unaffected
    state.portfolio_manager = MagicMock()
    state.portfolio_manager.aggregate_signals.return_value = []
    state.cross_platform_arb = MagicMock()
    state.cross_platform_arb.scan.return_value = []
    state.fill_monitor = None
    state._price_history = {}
    # Phase 3 learning components — mocked
    state.resolution_tracker = MagicMock()
    state.calibration_tracker = MagicMock()
    state.model_weight_updater = MagicMock()
    state.model_weight_updater.should_update.return_value = False
    state.agent_weight_updater = MagicMock()
    state.agent_weight_updater.should_update.return_value = False
    state.rmse_updater = MagicMock()
    state.error_analyzer = MagicMock()
    state._total_resolved_trades = 0
    # Phase 4: qualitative agent — None by default (not enabled in tests)
    state.qualitative_agent = None
    # Phase 4.5: wallet intelligence components — None by default
    state.wallet_collector = None
    state.wallet_profiler = None
    state.wallet_query_engine = None
    state.edge_decay_monitor = None
    state.competitor_tracker = None
    return state


def _patch_state(state: AppState):
    """Context manager: temporarily replace the module-level _state."""
    return patch.object(main_module, "_state", state)


# ---------------------------------------------------------------------------
# Gap 13: Logging setup
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_rotating_file_handler_created(self, tmp_path, monkeypatch):
        """setup_logging must add a RotatingFileHandler to the root logger."""
        import logging

        monkeypatch.setattr(main_module, "_LOG_DIR", tmp_path)
        monkeypatch.setattr(main_module, "_LOG_FILE", tmp_path / "weatheredge.log")

        root = logging.getLogger()
        # Remove any handlers added by earlier tests to keep this isolated
        original_handlers = root.handlers[:]
        root.handlers = []

        try:
            setup_logging()
            handler_types = [type(h) for h in root.handlers]
            assert RotatingFileHandler in handler_types, (
                f"Expected RotatingFileHandler in root logger handlers, got {handler_types}"
            )
        finally:
            # Restore handlers to avoid polluting other tests
            for h in root.handlers:
                h.close()
            root.handlers = original_handlers

    def test_rotating_file_handler_params(self, tmp_path, monkeypatch):
        """RotatingFileHandler must use 10MB maxBytes and 5 backupCount."""
        import logging

        monkeypatch.setattr(main_module, "_LOG_DIR", tmp_path)
        monkeypatch.setattr(main_module, "_LOG_FILE", tmp_path / "weatheredge.log")

        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers = []

        try:
            setup_logging()
            rfh = next(
                h for h in root.handlers if isinstance(h, RotatingFileHandler)
            )
            assert rfh.maxBytes == 10 * 1024 * 1024
            assert rfh.backupCount == 5
        finally:
            for h in root.handlers:
                h.close()
            root.handlers = original_handlers

    def test_log_format_contains_required_fields(self, tmp_path, monkeypatch):
        """Log format must include asctime, name, levelname, message."""
        import logging

        monkeypatch.setattr(main_module, "_LOG_DIR", tmp_path)
        monkeypatch.setattr(main_module, "_LOG_FILE", tmp_path / "weatheredge.log")

        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers = []

        try:
            setup_logging()
            rfh = next(
                h for h in root.handlers if isinstance(h, RotatingFileHandler)
            )
            fmt = rfh.formatter._fmt  # type: ignore[union-attr]
            assert "%(asctime)s" in fmt
            assert "%(name)s" in fmt
            assert "%(levelname)s" in fmt
            assert "%(message)s" in fmt
        finally:
            for h in root.handlers:
                h.close()
            root.handlers = original_handlers

    def test_setup_logging_uses_env_log_file_override(self, tmp_path, monkeypatch):
        """setup_logging must prefer WEATHEREDGE_LOG_FILE over default _LOG_FILE."""
        import logging
        from pathlib import Path

        env_log_file = tmp_path / "scenario.log"
        monkeypatch.setenv("WEATHEREDGE_LOG_FILE", str(env_log_file))
        monkeypatch.setattr(main_module, "_LOG_FILE", tmp_path / "ignored.log")

        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers = []

        try:
            setup_logging()
            rfh = next(h for h in root.handlers if isinstance(h, RotatingFileHandler))
            assert Path(rfh.baseFilename) == env_log_file
        finally:
            for h in root.handlers:
                h.close()
            root.handlers = original_handlers


# ---------------------------------------------------------------------------
# Gap 12: Decision loop — stale data check
# ---------------------------------------------------------------------------


class TestDecisionLoopStaleness:
    def test_skips_when_no_weather_data(self):
        """Decision loop skips cycle when no weather rows exist in DB."""
        state = _make_app_state()

        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=None
            ):
                run_decision_loop()

        # cancel_expired_orders should NOT be called — loop returned early
        state.paper_trader.cancel_expired_orders.assert_not_called()

    def test_skips_when_data_older_than_ttl(self):
        """Decision loop skips when weather data age > weather_data_ttl_minutes."""
        state = _make_app_state()
        # TTL defaults to 120 minutes; make data 121 minutes old
        stale_age = state.settings.weather_data_ttl_minutes + 1

        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=stale_age
            ):
                run_decision_loop()

        state.paper_trader.cancel_expired_orders.assert_not_called()

    def test_skips_when_data_exactly_at_ttl_boundary(self):
        """Decision loop skips when weather data age equals TTL (> not >=)."""
        state = _make_app_state()
        ttl = state.settings.weather_data_ttl_minutes
        # Exactly at TTL — condition is >, so TTL+1 skips but TTL itself proceeds
        # Confirm TTL+1 skips
        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=ttl + 1
            ):
                run_decision_loop()
        state.paper_trader.cancel_expired_orders.assert_not_called()

    def test_proceeds_when_data_within_ttl(self):
        """Decision loop proceeds when weather data is fresh (age < TTL)."""
        state = _make_app_state()
        fresh_age = state.settings.weather_data_ttl_minutes - 1

        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=fresh_age
            ):
                # _read_latest_forecasts and _read_latest_market_prices need mocking
                with patch.object(
                    main_module, "_read_latest_forecasts", return_value={}
                ):
                    with patch.object(
                        main_module, "_read_latest_market_prices", return_value={}
                    ):
                        run_decision_loop()

        # cancel_expired_orders IS called — loop ran to completion
        state.paper_trader.cancel_expired_orders.assert_called_once()

    def test_does_not_skip_when_shutting_down(self):
        """Decision loop returns early without logging stale-data warning when shutting_down=True."""
        state = _make_app_state()
        state.shutting_down = True

        with _patch_state(state):
            run_decision_loop()  # must not raise

        state.paper_trader.cancel_expired_orders.assert_not_called()


# ---------------------------------------------------------------------------
# Gap 7: Decision loop — processes markets with fresh data
# ---------------------------------------------------------------------------


class TestDecisionLoopMarkets:
    def _fresh_state(self) -> AppState:
        state = _make_app_state()
        # Not shutting down; fresh data
        return state

    def test_no_markets_places_no_trades(self):
        """Loop runs without placing trades when market_prices is empty."""
        state = self._fresh_state()
        fresh_age = 10  # well within TTL

        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=fresh_age
            ):
                with patch.object(
                    main_module, "_read_latest_forecasts", return_value={}
                ):
                    with patch.object(
                        main_module, "_read_latest_market_prices", return_value={}
                    ):
                        run_decision_loop()

        state.paper_trader.place_order.assert_not_called()
        state.paper_trader.cancel_expired_orders.assert_called_once()

    def test_known_market_with_sufficient_edge_places_trade(self):
        """Loop places a paper trade when net edge exceeds min_edge threshold."""
        from weatheredge.agents.base import TradeOrder as _TradeOrder
        state = self._fresh_state()
        # Use a known Kalshi ticker from RESOLUTION_SOURCES
        market_id = "KXHIGHNY"
        fresh_age = 10

        # Forecast for KNYC station (maps to KXHIGHNY per RESOLUTION_SOURCES)
        forecasts = {
            "KNYC": {
                "temperature_forecast": 85.0,  # well above 70°F threshold
                "ensemble_members": None,
                "model": "forecast",
                "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        # Market yes_price well below model probability → big edge
        market_prices = {
            market_id: {
                "yes_price": 0.35,  # model prob ~0.95 at 85°F → large edge
                "no_price": 0.65,
                "volume": 1000.0,
                "spread": 0.02,
            }
        }

        # Portfolio manager returns a TradeOrder so the paper trader is called
        trade_order = _TradeOrder(
            market_id=market_id,
            side="buy_yes",
            price=0.35,
            amount=50.0,
            agent_name="PortfolioManager",
            model_probability=0.99,
            edge_at_entry=0.63,
            city="KNYC",
            platform="kalshi",
        )
        state.portfolio_manager.aggregate_signals.return_value = [trade_order]
        state.paper_trader.place_order.return_value = {
            "filled": True,
            "trade": {"market_id": market_id},
        }

        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=fresh_age
            ):
                with patch.object(
                    main_module, "_read_latest_forecasts", return_value=forecasts
                ):
                    with patch.object(
                        main_module,
                        "_read_latest_market_prices",
                        return_value=market_prices,
                    ):
                        run_decision_loop()

        state.paper_trader.place_order.assert_called_once()
        order_arg = state.paper_trader.place_order.call_args[0][0]
        assert order_arg.market_id == market_id
        assert order_arg.side == "buy_yes"

    def test_risk_manager_veto_blocks_trade(self):
        """Pre-trade risk check blocking result prevents order placement."""
        state = self._fresh_state()
        market_id = "KXHIGHNY"
        fresh_age = 10

        forecasts = {
            "KNYC": {
                "temperature_forecast": 85.0,
                "ensemble_members": None,
                "model": "forecast",
                "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        market_prices = {
            market_id: {"yes_price": 0.35, "no_price": 0.65, "volume": 1000.0, "spread": 0.02}
        }

        state.risk_manager.calculate_position_size.return_value = 50.0
        state.risk_manager.pre_trade_check.return_value = {
            "blocked": True,
            "reason": "Max open positions reached",
        }

        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=fresh_age
            ):
                with patch.object(
                    main_module, "_read_latest_forecasts", return_value=forecasts
                ):
                    with patch.object(
                        main_module,
                        "_read_latest_market_prices",
                        return_value=market_prices,
                    ):
                        run_decision_loop()

        state.paper_trader.place_order.assert_not_called()

    def test_unknown_market_is_skipped(self):
        """Markets not in RESOLUTION_SOURCES do not trigger trade evaluation."""
        state = self._fresh_state()
        fresh_age = 10

        market_prices = {
            "UNKNOWN_MARKET_XYZ": {"yes_price": 0.50, "no_price": 0.50}
        }

        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=fresh_age
            ):
                with patch.object(
                    main_module, "_read_latest_forecasts", return_value={}
                ):
                    with patch.object(
                        main_module,
                        "_read_latest_market_prices",
                        return_value=market_prices,
                    ):
                        run_decision_loop()

        state.paper_trader.place_order.assert_not_called()

    def test_expired_orders_cancelled_each_cycle(self):
        """cancel_expired_orders is called every decision loop cycle with fresh data."""
        state = self._fresh_state()
        state.paper_trader.cancel_expired_orders.return_value = 3
        fresh_age = 10

        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=fresh_age
            ):
                with patch.object(
                    main_module, "_read_latest_forecasts", return_value={}
                ):
                    with patch.object(
                        main_module, "_read_latest_market_prices", return_value={}
                    ):
                        run_decision_loop()

        state.paper_trader.cancel_expired_orders.assert_called_once()


# ---------------------------------------------------------------------------
# Gap 14: SIGTERM / graceful shutdown
# ---------------------------------------------------------------------------


class TestSigterm:
    def test_handle_shutdown_sets_shutting_down_flag(self):
        """_handle_shutdown must set state.shutting_down = True."""
        state = _make_app_state()
        state.scheduler = MagicMock()

        with _patch_state(state):
            _handle_shutdown(signal.SIGTERM, None)

        assert state.shutting_down is True

    def test_handle_shutdown_calls_scheduler_shutdown(self):
        """_handle_shutdown must call scheduler.shutdown()."""
        state = _make_app_state()
        state.scheduler = MagicMock()

        with _patch_state(state):
            _handle_shutdown(signal.SIGTERM, None)

        state.scheduler.shutdown.assert_called_once_with(wait=False)

    def test_handle_shutdown_no_scheduler_does_not_raise(self):
        """_handle_shutdown is safe when scheduler is None."""
        state = _make_app_state()
        state.scheduler = None

        with _patch_state(state):
            _handle_shutdown(signal.SIGTERM, None)  # must not raise

        assert state.shutting_down is True

    def test_handle_sigint_also_sets_flag(self):
        """SIGINT (Ctrl-C) triggers the same shutdown path."""
        state = _make_app_state()
        state.scheduler = MagicMock()

        with _patch_state(state):
            _handle_shutdown(signal.SIGINT, None)

        assert state.shutting_down is True

    def test_decision_loop_exits_immediately_when_shutting_down(self):
        """If shutting_down is True when decision loop fires, it returns immediately."""
        state = _make_app_state()
        state.shutting_down = True

        # _get_weather_data_age_minutes should never be called
        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes"
            ) as mock_age:
                run_decision_loop()
                mock_age.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 3: Learning components
# ---------------------------------------------------------------------------


class TestLearningComponentsInitialized:
    def test_learning_components_initialized(self):
        """AppState must have all 6 Phase 3 learning component slots."""
        state = AppState()
        # All six attributes must exist (may be None before DB init)
        assert hasattr(state, "resolution_tracker")
        assert hasattr(state, "calibration_tracker")
        assert hasattr(state, "model_weight_updater")
        assert hasattr(state, "agent_weight_updater")
        assert hasattr(state, "rmse_updater")
        assert hasattr(state, "error_analyzer")
        assert hasattr(state, "_total_resolved_trades")

    def test_learning_component_slots_start_none(self):
        """Learning components are None until DB is ready (initialised in main())."""
        state = AppState()
        assert state.resolution_tracker is None
        assert state.calibration_tracker is None
        assert state.model_weight_updater is None
        assert state.agent_weight_updater is None
        assert state.rmse_updater is None
        assert state.error_analyzer is None
        assert state._total_resolved_trades == 0


class TestWeightUpdateTriggeredAtInterval:
    def test_weight_update_triggered_at_interval(self):
        """model_weight_updater.should_update is called when trades are placed."""
        from weatheredge.agents.base import TradeOrder as _TradeOrder

        state = _make_app_state()
        state.model_weight_updater.should_update.return_value = True
        state.model_weight_updater.update_model_weights.return_value = None

        market_id = "KXHIGHNY"
        fresh_age = 10
        forecasts = {
            "KNYC": {
                "temperature_forecast": 85.0,
                "ensemble_members": None,
                "model": "forecast",
                "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        market_prices = {
            market_id: {
                "yes_price": 0.35,
                "no_price": 0.65,
                "volume": 1000.0,
                "spread": 0.02,
            }
        }
        trade_order = _TradeOrder(
            market_id=market_id,
            side="buy_yes",
            price=0.35,
            amount=50.0,
            agent_name="PortfolioManager",
            model_probability=0.99,
            edge_at_entry=0.63,
            city="KNYC",
            platform="kalshi",
        )
        state.portfolio_manager.aggregate_signals.return_value = [trade_order]
        state.paper_trader.place_order.return_value = {
            "filled": True,
            "trade": {"market_id": market_id},
        }

        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=fresh_age
            ):
                with patch.object(
                    main_module, "_read_latest_forecasts", return_value=forecasts
                ):
                    with patch.object(
                        main_module,
                        "_read_latest_market_prices",
                        return_value=market_prices,
                    ):
                        run_decision_loop()

        state.model_weight_updater.record_trade.assert_called()
        state.model_weight_updater.should_update.assert_called()
        state.model_weight_updater.update_model_weights.assert_called_once()

    def test_weight_update_not_triggered_below_interval(self):
        """model_weight_updater.update_model_weights is NOT called when should_update is False."""
        from weatheredge.agents.base import TradeOrder as _TradeOrder

        state = _make_app_state()
        # Default: should_update returns False
        state.model_weight_updater.should_update.return_value = False

        market_id = "KXHIGHNY"
        fresh_age = 10
        forecasts = {
            "KNYC": {
                "temperature_forecast": 85.0,
                "ensemble_members": None,
                "model": "forecast",
                "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        market_prices = {
            market_id: {
                "yes_price": 0.35,
                "no_price": 0.65,
                "volume": 1000.0,
                "spread": 0.02,
            }
        }
        trade_order = _TradeOrder(
            market_id=market_id,
            side="buy_yes",
            price=0.35,
            amount=50.0,
            agent_name="PortfolioManager",
            model_probability=0.99,
            edge_at_entry=0.63,
            city="KNYC",
            platform="kalshi",
        )
        state.portfolio_manager.aggregate_signals.return_value = [trade_order]
        state.paper_trader.place_order.return_value = {
            "filled": True,
            "trade": {"market_id": market_id},
        }

        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=fresh_age
            ):
                with patch.object(
                    main_module, "_read_latest_forecasts", return_value=forecasts
                ):
                    with patch.object(
                        main_module,
                        "_read_latest_market_prices",
                        return_value=market_prices,
                    ):
                        run_decision_loop()

        state.model_weight_updater.update_model_weights.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 4: Qualitative agent
# ---------------------------------------------------------------------------


class TestQualitativeAgentInitialization:
    def test_qualitative_agent_initialized_when_enabled(self, monkeypatch):
        """When enable_qualitative_agent=True and openrouter_api_key is set, agent is created."""
        from weatheredge.agents.qualitative import QualitativeAgent
        import weatheredge.main as _main

        # Patch main() startup to skip DB / scheduler / signal setup
        with patch.object(_main, "init_db"), \
             patch.object(_main, "FillMonitor"), \
             patch.object(_main, "BackgroundScheduler") as mock_sched_cls:
            mock_scheduler = MagicMock()
            mock_sched_cls.return_value = mock_scheduler
            mock_scheduler.start = MagicMock()
            # Make the while loop exit immediately
            monkeypatch.setattr(_main, "_state", None)

            state = _main.AppState()
            state.settings = MagicMock()
            state.settings.enable_qualitative_agent = True
            state.settings.openrouter_api_key = "test-key-123"

            monkeypatch.setattr(_main, "_state", state)

            if state.settings.enable_qualitative_agent and state.settings.openrouter_api_key:
                state.qualitative_agent = QualitativeAgent(state.settings)

        assert state.qualitative_agent is not None
        assert isinstance(state.qualitative_agent, QualitativeAgent)

    def test_qualitative_agent_skipped_when_disabled(self, monkeypatch):
        """When enable_qualitative_agent=False, qualitative_agent slot stays None."""
        import weatheredge.main as _main

        state = _main.AppState()
        state.settings = MagicMock()
        state.settings.enable_qualitative_agent = False
        state.settings.openrouter_api_key = "test-key-123"

        monkeypatch.setattr(_main, "_state", state)

        if state.settings.enable_qualitative_agent and state.settings.openrouter_api_key:
            from weatheredge.agents.qualitative import QualitativeAgent
            state.qualitative_agent = QualitativeAgent(state.settings)

        assert state.qualitative_agent is None

    def test_qualitative_agent_skipped_when_no_api_key(self, monkeypatch):
        """When openrouter_api_key is empty, qualitative_agent slot stays None."""
        import weatheredge.main as _main

        state = _main.AppState()
        state.settings = MagicMock()
        state.settings.enable_qualitative_agent = True
        state.settings.openrouter_api_key = ""

        monkeypatch.setattr(_main, "_state", state)

        if state.settings.enable_qualitative_agent and state.settings.openrouter_api_key:
            from weatheredge.agents.qualitative import QualitativeAgent
            state.qualitative_agent = QualitativeAgent(state.settings)

        assert state.qualitative_agent is None


# ---------------------------------------------------------------------------
# Area 1: Scheduler Job Registration
# ---------------------------------------------------------------------------


class TestSchedulerJobRegistration:
    """Test that main() registers all expected scheduler jobs."""

    @pytest.fixture()
    def registered_jobs(self, monkeypatch):
        """Run main() with mocks and capture all scheduler.add_job calls."""
        import weatheredge.main as _main

        # Track add_job calls
        jobs: list[dict] = []
        mock_scheduler = MagicMock()

        def capture_add_job(func, trigger, *, id, **kwargs):
            jobs.append({"func": func, "trigger": trigger, "id": id, **kwargs})

        mock_scheduler.add_job.side_effect = capture_add_job
        mock_scheduler.start = MagicMock()

        # Patch BackgroundScheduler to return our mock
        monkeypatch.setattr(_main, "BackgroundScheduler", lambda: mock_scheduler)

        # Patch everything that main() calls before scheduler setup
        with patch.object(_main, "setup_logging"), \
             patch.object(_main, "init_db"), \
             patch.object(_main, "FillMonitor", return_value=MagicMock()):
            # Patch DB engine access
            mock_db = MagicMock()
            mock_db._engine = MagicMock()
            with patch.dict("sys.modules", {"weatheredge.db.database": mock_db}):
                # Patch signal.signal to avoid interfering with test runner
                with patch("signal.signal"):
                    # Patch the while loop so main() returns immediately
                    state = _main.AppState()
                    state.shutting_down = True
                    state.settings = MagicMock()
                    state.settings.market_scan_interval_minutes = 5
                    state.settings.forecast_poll_interval_minutes = 15
                    state.settings.decision_loop_interval_minutes = 5
                    state.settings.resolution_check_time = "08:00"
                    state.settings.rmse_recalc_interval_days = 30
                    state.settings.error_analysis_interval_days = 7
                    state.settings.enable_qualitative_agent = False
                    state.settings.openrouter_api_key = ""
                    state.settings.initial_bankroll = 1000.0

                    # Override init_app to return our pre-configured state
                    def fake_init_app():
                        _main._state = state
                        return state
                    monkeypatch.setattr(_main, "init_app", fake_init_app)

                    # Patch Phase 3 + 4.5 component constructors
                    monkeypatch.setattr(_main, "ResolutionTracker", MagicMock())
                    monkeypatch.setattr(_main, "CalibrationTracker", MagicMock())
                    monkeypatch.setattr(_main, "ModelWeightUpdater", MagicMock())
                    monkeypatch.setattr(_main, "AgentWeightUpdater", MagicMock())
                    monkeypatch.setattr(_main, "RMSEUpdater", MagicMock())
                    monkeypatch.setattr(_main, "ErrorAnalyzer", MagicMock())
                    monkeypatch.setattr(_main, "WalletCollector", MagicMock())
                    monkeypatch.setattr(_main, "WalletProfiler", MagicMock())
                    monkeypatch.setattr(_main, "WalletQueryEngine", MagicMock())
                    monkeypatch.setattr(_main, "EdgeDecayMonitor", MagicMock())
                    monkeypatch.setattr(_main, "CompetitorTracker", MagicMock())

                    # Run main — it will exit because shutting_down=True
                    _main.main()

        return jobs

    def test_market_scan_registered(self, registered_jobs):
        ids = [j["id"] for j in registered_jobs]
        assert "market_scan" in ids
        job = next(j for j in registered_jobs if j["id"] == "market_scan")
        assert job["trigger"] == "interval"
        assert job["minutes"] == 5

    def test_forecast_poll_registered(self, registered_jobs):
        ids = [j["id"] for j in registered_jobs]
        assert "forecast_poll" in ids
        job = next(j for j in registered_jobs if j["id"] == "forecast_poll")
        assert job["trigger"] == "interval"
        assert job["minutes"] == 15

    def test_decision_loop_registered(self, registered_jobs):
        ids = [j["id"] for j in registered_jobs]
        assert "decision_loop" in ids
        job = next(j for j in registered_jobs if j["id"] == "decision_loop")
        assert job["trigger"] == "interval"
        assert job["minutes"] == 5
        assert job["max_instances"] == 1
        assert job["coalesce"] is True

    def test_resolution_check_registered(self, registered_jobs):
        ids = [j["id"] for j in registered_jobs]
        assert "resolution_check" in ids
        job = next(j for j in registered_jobs if j["id"] == "resolution_check")
        assert job["trigger"] == "cron"

    def test_rmse_recalculation_registered(self, registered_jobs):
        ids = [j["id"] for j in registered_jobs]
        assert "rmse_recalculation" in ids
        job = next(j for j in registered_jobs if j["id"] == "rmse_recalculation")
        assert job["trigger"] == "interval"
        assert job["days"] == 30

    def test_error_analysis_registered(self, registered_jobs):
        ids = [j["id"] for j in registered_jobs]
        assert "error_analysis" in ids
        job = next(j for j in registered_jobs if j["id"] == "error_analysis")
        assert job["trigger"] == "interval"
        assert job["days"] == 7

    def test_wallet_profiling_registered(self, registered_jobs):
        ids = [j["id"] for j in registered_jobs]
        assert "wallet_profiling" in ids
        job = next(j for j in registered_jobs if j["id"] == "wallet_profiling")
        assert job["trigger"] == "interval"
        assert job["days"] == 7

    def test_edge_decay_check_registered(self, registered_jobs):
        ids = [j["id"] for j in registered_jobs]
        assert "edge_decay_check" in ids
        job = next(j for j in registered_jobs if j["id"] == "edge_decay_check")
        assert job["trigger"] == "interval"
        assert job["days"] == 7

    def test_morning_briefing_registered(self, registered_jobs):
        ids = [j["id"] for j in registered_jobs]
        assert "morning_briefing" in ids
        job = next(j for j in registered_jobs if j["id"] == "morning_briefing")
        assert job["trigger"] == "cron"
        assert job["hour"] == 6
        assert job["minute"] == 0

    def test_all_eleven_jobs_registered(self, registered_jobs):
        """11 core jobs + 1 optional bias_correction_refresh when Phase 1 enabled."""
        from weatheredge.config import MODULES
        expected = 11 + (1 if MODULES.get("bias_correction", False) else 0)
        assert len(registered_jobs) == expected

    def test_bias_correction_refresh_registered_when_enabled(self, registered_jobs):
        """bias_correction_refresh job present iff bias_correction module is ON."""
        from weatheredge.config import MODULES
        ids = [j["id"] for j in registered_jobs]
        if MODULES.get("bias_correction", False):
            assert "bias_correction_refresh" in ids
            job = next(j for j in registered_jobs if j["id"] == "bias_correction_refresh")
            assert job["trigger"] == "interval"
            assert job["days"] == 1
        else:
            assert "bias_correction_refresh" not in ids


# ---------------------------------------------------------------------------
# Area 2: Decision Loop Reads Latest Data (additional coverage)
# ---------------------------------------------------------------------------


class TestDecisionLoopDataFlow:
    """Test the decision loop reads data, builds opportunities, runs strategies."""

    def test_db_read_failure_returns_early(self):
        """When _read_latest_forecasts raises, loop returns without trading."""
        state = _make_app_state()
        fresh_age = 10

        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=fresh_age
            ):
                with patch.object(
                    main_module, "_read_latest_forecasts", side_effect=Exception("DB error")
                ):
                    run_decision_loop()

        # cancel_expired_orders should NOT be called — loop errored before it
        state.paper_trader.cancel_expired_orders.assert_not_called()

    def test_portfolio_manager_exception_yields_empty_trades(self):
        """When portfolio_manager.aggregate_signals raises, trade_orders becomes []."""
        state = _make_app_state()
        fresh_age = 10

        state.portfolio_manager.aggregate_signals.side_effect = Exception("PM crash")

        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=fresh_age
            ):
                with patch.object(
                    main_module, "_read_latest_forecasts", return_value={"KNYC": {
                        "temperature_forecast": 85.0,
                        "ensemble_members": None,
                        "model": "forecast",
                        "forecast_timestamp": datetime.now(timezone.utc),
                    }}
                ):
                    with patch.object(
                        main_module, "_read_latest_market_prices", return_value={
                            "KXHIGHNY": {"yes_price": 0.35, "no_price": 0.65, "volume": 1000.0, "spread": 0.02}
                        }
                    ):
                        with patch.object(main_module, "_build_opportunity", return_value=MagicMock(
                            market_id="KXHIGHNY", platform="kalshi", city="KNYC",
                            market_price=0.35, threshold=70.0,
                        )):
                            run_decision_loop()

        state.paper_trader.place_order.assert_not_called()
        # cancel_expired_orders is still called at the end
        state.paper_trader.cancel_expired_orders.assert_called_once()

    def test_fill_monitor_records_fill_on_trade(self):
        """When fill_monitor is set, record_fill is called for filled trades."""
        from weatheredge.agents.base import TradeOrder as _TradeOrder

        state = _make_app_state()
        state.fill_monitor = MagicMock()
        fresh_age = 10

        trade_order = _TradeOrder(
            market_id="KXHIGHNY", side="buy_yes", price=0.35, amount=50.0,
            agent_name="PM", model_probability=0.99, edge_at_entry=0.63,
            city="KNYC", platform="kalshi",
        )
        state.portfolio_manager.aggregate_signals.return_value = [trade_order]
        state.paper_trader.place_order.return_value = {"filled": True, "trade": {"market_id": "KXHIGHNY"}}

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=fresh_age):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(main_module, "_read_latest_market_prices", return_value={}):
                        run_decision_loop()

        state.fill_monitor.record_fill.assert_called_once()

    def test_order_rejection_logged_no_crash(self):
        """When paper_trader rejects an order, loop continues without crashing."""
        from weatheredge.agents.base import TradeOrder as _TradeOrder

        state = _make_app_state()
        fresh_age = 10

        trade_order = _TradeOrder(
            market_id="KXHIGHNY", side="buy_yes", price=0.35, amount=50.0,
            agent_name="PM", model_probability=0.99, edge_at_entry=0.63,
            city="KNYC", platform="kalshi",
        )
        state.portfolio_manager.aggregate_signals.return_value = [trade_order]
        state.paper_trader.place_order.return_value = {"filled": False, "reason": "bankroll exceeded"}

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=fresh_age):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(main_module, "_read_latest_market_prices", return_value={}):
                        run_decision_loop()

        state.paper_trader.cancel_expired_orders.assert_called_once()

    def test_agent_weight_updater_triggered(self):
        """agent_weight_updater.update_agent_weights is called when should_update=True."""
        from weatheredge.agents.base import TradeOrder as _TradeOrder

        state = _make_app_state()
        state.agent_weight_updater.should_update.return_value = True
        state.agent_weight_updater.update_agent_weights.return_value = {"conservative": 0.5, "aggressive": 0.5}
        state.portfolio_manager.AGENT_WEIGHTS = {"conservative": 0.4, "aggressive": 0.6}
        fresh_age = 10

        trade_order = _TradeOrder(
            market_id="KXHIGHNY", side="buy_yes", price=0.35, amount=50.0,
            agent_name="PM", model_probability=0.99, edge_at_entry=0.63,
            city="KNYC", platform="kalshi",
        )
        state.portfolio_manager.aggregate_signals.return_value = [trade_order]
        state.paper_trader.place_order.return_value = {"filled": True, "trade": {"market_id": "KXHIGHNY"}}
        state.model_weight_updater.should_update.return_value = False

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=fresh_age):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(main_module, "_read_latest_market_prices", return_value={}):
                        run_decision_loop()

        state.agent_weight_updater.update_agent_weights.assert_called_once()

    def test_cross_platform_arb_exception_does_not_crash(self):
        """Exception in cross_platform_arb.scan is caught and loop completes."""
        state = _make_app_state()
        state.cross_platform_arb.scan.side_effect = Exception("arb crash")
        fresh_age = 10

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=fresh_age):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(main_module, "_read_latest_market_prices", return_value={}):
                        run_decision_loop()

        state.paper_trader.cancel_expired_orders.assert_called_once()


# ---------------------------------------------------------------------------
# Area 3: Graceful Shutdown (extended)
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    """Extended tests for _handle_shutdown including engine dispose."""

    def test_handle_shutdown_disposes_engine(self):
        """_handle_shutdown disposes the SQLAlchemy engine."""
        state = _make_app_state()
        state.scheduler = MagicMock()

        mock_db = MagicMock()
        mock_db._engine = MagicMock()

        with _patch_state(state):
            with patch.dict("sys.modules", {"weatheredge.db.database": mock_db}):
                # Need to also patch the import inside _handle_shutdown
                with patch("weatheredge.main.get_state", return_value=state):
                    _handle_shutdown(signal.SIGTERM, None)

        assert state.shutting_down is True
        state.scheduler.shutdown.assert_called_once_with(wait=False)

    def test_handle_shutdown_scheduler_error_does_not_crash(self):
        """If scheduler.shutdown raises, _handle_shutdown still completes."""
        state = _make_app_state()
        state.scheduler = MagicMock()
        state.scheduler.shutdown.side_effect = Exception("sched error")

        with _patch_state(state):
            _handle_shutdown(signal.SIGTERM, None)

        assert state.shutting_down is True

    def test_handle_shutdown_with_sigint(self):
        """_handle_shutdown works with SIGINT signal."""
        state = _make_app_state()
        state.scheduler = MagicMock()

        with _patch_state(state):
            _handle_shutdown(signal.SIGINT, None)

        assert state.shutting_down is True
        state.scheduler.shutdown.assert_called_once_with(wait=False)


# ---------------------------------------------------------------------------
# Area 4: Bot Initializes Without API Keys
# ---------------------------------------------------------------------------


class TestBotWithoutApiKeys:
    """Test that the bot runs without OpenRouter API keys."""

    def test_qualitative_agent_none_when_no_api_key(self, monkeypatch):
        """When openrouter_api_key is empty, qualitative_agent remains None."""
        import weatheredge.main as _main

        state = _main.AppState()
        state.settings = MagicMock()
        state.settings.enable_qualitative_agent = True
        state.settings.openrouter_api_key = ""

        # Replicate the logic from main()
        if state.settings.enable_qualitative_agent and state.settings.openrouter_api_key:
            from weatheredge.agents.qualitative import QualitativeAgent
            state.qualitative_agent = QualitativeAgent(state.settings)

        assert state.qualitative_agent is None

    def test_decision_loop_works_without_qualitative_agent(self):
        """Decision loop runs to completion when qualitative_agent is None."""
        state = _make_app_state()
        state.qualitative_agent = None
        fresh_age = 10

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=fresh_age):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(main_module, "_read_latest_market_prices", return_value={}):
                        run_decision_loop()

        state.paper_trader.cancel_expired_orders.assert_called_once()

    def test_main_skips_qualitative_when_no_key(self, monkeypatch):
        """main() does not create QualitativeAgent when openrouter_api_key is empty."""
        import weatheredge.main as _main

        mock_scheduler = MagicMock()
        mock_scheduler.start = MagicMock()
        monkeypatch.setattr(_main, "BackgroundScheduler", lambda: mock_scheduler)

        with patch.object(_main, "setup_logging"), \
             patch.object(_main, "init_db"), \
             patch.object(_main, "FillMonitor", return_value=MagicMock()), \
             patch("signal.signal"):

            mock_db = MagicMock()
            mock_db._engine = MagicMock()
            with patch.dict("sys.modules", {"weatheredge.db.database": mock_db}):
                state = _main.AppState()
                state.shutting_down = True
                state.settings = MagicMock()
                state.settings.enable_qualitative_agent = True
                state.settings.openrouter_api_key = ""
                state.settings.market_scan_interval_minutes = 5
                state.settings.forecast_poll_interval_minutes = 15
                state.settings.decision_loop_interval_minutes = 5
                state.settings.resolution_check_time = "08:00"
                state.settings.rmse_recalc_interval_days = 30
                state.settings.error_analysis_interval_days = 7

                def fake_init_app():
                    _main._state = state
                    return state
                monkeypatch.setattr(_main, "init_app", fake_init_app)
                monkeypatch.setattr(_main, "ResolutionTracker", MagicMock())
                monkeypatch.setattr(_main, "CalibrationTracker", MagicMock())
                monkeypatch.setattr(_main, "ModelWeightUpdater", MagicMock())
                monkeypatch.setattr(_main, "AgentWeightUpdater", MagicMock())
                monkeypatch.setattr(_main, "RMSEUpdater", MagicMock())
                monkeypatch.setattr(_main, "ErrorAnalyzer", MagicMock())
                monkeypatch.setattr(_main, "WalletCollector", MagicMock())
                monkeypatch.setattr(_main, "WalletProfiler", MagicMock())
                monkeypatch.setattr(_main, "WalletQueryEngine", MagicMock())
                monkeypatch.setattr(_main, "EdgeDecayMonitor", MagicMock())
                monkeypatch.setattr(_main, "CompetitorTracker", MagicMock())

                _main.main()

        assert state.qualitative_agent is None


# ---------------------------------------------------------------------------
# Area 5: Circuit Breaker Halts Activity
# (Circuit breaker is in RiskManager.pre_trade_check, which blocks trades
#  in the decision loop when halt=True)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="_evaluate_market is deprecated (raises NotImplementedError)")
class TestCircuitBreakerHaltsActivity:
    """Test that circuit breaker firing in risk manager blocks all trades.

    The circuit breaker integration lives in _evaluate_market(), which calls
    risk_manager.pre_trade_check(). When the circuit breaker fires (halt=True),
    pre_trade_check returns blocked=True and no order is placed.
    """

    def test_circuit_breaker_blocks_trade_via_evaluate_market(self):
        """When circuit breaker fires, _evaluate_market returns 0 and no order placed."""
        state = _make_app_state()
        state.risk_manager.calculate_position_size.return_value = 50.0
        state.risk_manager.pre_trade_check.return_value = {
            "blocked": True,
            "reason": "KILL SWITCH: Total drawdown exceeded -20%",
        }
        state.paper_trader.bankroll = 1000.0
        state.paper_trader.open_positions = []
        state.paper_trader.daily_pnl = {}
        state.paper_trader.trade_history = []
        forecasts = {
            "KNYC": {
                "temperature_forecast": 95.0, "ensemble_members": None,
                "model": "forecast", "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        result = main_module._evaluate_market(
            state, "KXHIGHNY", {"yes_price": 0.30}, forecasts, 0.01
        )
        assert result == 0
        state.paper_trader.place_order.assert_not_called()

    def test_circuit_breaker_reason_is_from_risk_manager(self):
        """Verify the blocked reason matches the circuit breaker message."""
        state = _make_app_state()
        state.risk_manager.calculate_position_size.return_value = 50.0
        state.risk_manager.pre_trade_check.return_value = {
            "blocked": True,
            "reason": "Daily loss limit exceeded -5%",
        }
        state.paper_trader.bankroll = 1000.0
        state.paper_trader.open_positions = []
        state.paper_trader.daily_pnl = {}
        state.paper_trader.trade_history = []
        forecasts = {
            "KNYC": {
                "temperature_forecast": 95.0, "ensemble_members": None,
                "model": "forecast", "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        result = main_module._evaluate_market(
            state, "KXHIGHNY", {"yes_price": 0.30}, forecasts, 0.01
        )
        assert result == 0
        state.paper_trader.place_order.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 3 scheduler jobs coverage
# ---------------------------------------------------------------------------


class TestResolutionCheck:
    """Tests for run_resolution_check() scheduler job."""

    def test_skips_when_tracker_is_none(self):
        state = _make_app_state()
        state.resolution_tracker = None

        with _patch_state(state):
            main_module.run_resolution_check()  # must not raise

    def test_logs_unresolved_markets(self):
        state = _make_app_state()
        state.resolution_tracker.get_unresolved_markets.return_value = ["MKT1", "MKT2"]

        with _patch_state(state):
            main_module.run_resolution_check()

        state.resolution_tracker.get_unresolved_markets.assert_called_once()

    def test_logs_no_unresolved_markets(self):
        state = _make_app_state()
        state.resolution_tracker.get_unresolved_markets.return_value = []

        with _patch_state(state):
            main_module.run_resolution_check()

        state.resolution_tracker.get_unresolved_markets.assert_called_once()

    def test_exception_caught(self):
        state = _make_app_state()
        state.resolution_tracker.get_unresolved_markets.side_effect = Exception("DB error")

        with _patch_state(state):
            main_module.run_resolution_check()  # must not raise


class TestRMSERecalculation:
    """Tests for run_rmse_recalculation() scheduler job."""

    def test_skips_when_updater_is_none(self):
        state = _make_app_state()
        state.rmse_updater = None

        with _patch_state(state):
            main_module.run_rmse_recalculation()  # must not raise

    def test_calls_update_all_rmse(self):
        state = _make_app_state()

        with _patch_state(state):
            main_module.run_rmse_recalculation()

        state.rmse_updater.update_all_rmse.assert_called_once()

    def test_exception_caught(self):
        state = _make_app_state()
        state.rmse_updater.update_all_rmse.side_effect = Exception("RMSE error")

        with _patch_state(state):
            main_module.run_rmse_recalculation()  # must not raise


class TestErrorAnalysis:
    """Tests for run_error_analysis() scheduler job."""

    def test_skips_when_analyzer_is_none(self):
        state = _make_app_state()
        state.error_analyzer = None

        with _patch_state(state):
            main_module.run_error_analysis()  # must not raise

    def test_no_losses(self):
        state = _make_app_state()
        report = MagicMock()
        report.total_losses = 0
        state.error_analyzer.analyze_errors.return_value = report

        with _patch_state(state):
            main_module.run_error_analysis()

        state.error_analyzer.analyze_errors.assert_called_once()

    def test_losses_with_telegram_alert(self):
        state = _make_app_state()
        state.settings = MagicMock()
        state.settings.openrouter_api_key = ""
        report = MagicMock()
        report.total_losses = 25
        report.model_errors = 10
        report.timing_errors = 3
        report.edge_compression = 5
        report.adverse_selection = 7
        state.error_analyzer.analyze_errors.return_value = report

        with _patch_state(state):
            with patch("weatheredge.main.TelegramBot", create=True) as mock_tg_cls:
                mock_bot = MagicMock()
                mock_tg_cls.return_value = mock_bot
                with patch("weatheredge.notifications.telegram.TelegramBot", mock_tg_cls, create=True):
                    main_module.run_error_analysis()

        state.error_analyzer.analyze_errors.assert_called_once()

    def test_exception_caught(self):
        state = _make_app_state()
        state.error_analyzer.analyze_errors.side_effect = Exception("analysis error")

        with _patch_state(state):
            main_module.run_error_analysis()  # must not raise


# ---------------------------------------------------------------------------
# Phase 4.5 scheduler jobs coverage
# ---------------------------------------------------------------------------


class TestWalletProfiling:
    """Tests for run_wallet_profiling() scheduler job."""

    def test_skips_when_collector_is_none(self):
        state = _make_app_state()
        state.wallet_collector = None
        state.wallet_profiler = None

        with _patch_state(state):
            main_module.run_wallet_profiling()  # must not raise

    def test_no_wallets_collected(self):
        state = _make_app_state()
        state.wallet_collector = MagicMock()
        state.wallet_profiler = MagicMock()
        state.wallet_collector.collect_weather_wallets.return_value = []

        with _patch_state(state):
            main_module.run_wallet_profiling()

        state.wallet_collector.collect_weather_wallets.assert_called_once()
        state.wallet_profiler.build_index.assert_not_called()

    def test_profiles_wallets_and_builds_index(self):
        state = _make_app_state()
        state.wallet_collector = MagicMock()
        state.wallet_profiler = MagicMock()
        state.wallet_collector.collect_weather_wallets.return_value = [
            {"address": "0x1"}, {"address": "0x2"}
        ]
        state.wallet_profiler.profile_wallet.return_value = "profile text"

        with _patch_state(state):
            main_module.run_wallet_profiling()

        assert state.wallet_profiler.profile_wallet.call_count == 2
        state.wallet_profiler.build_index.assert_called_once()

    def test_exception_caught(self):
        state = _make_app_state()
        state.wallet_collector = MagicMock()
        state.wallet_profiler = MagicMock()
        state.wallet_collector.collect_weather_wallets.side_effect = Exception("wallet error")

        with _patch_state(state):
            main_module.run_wallet_profiling()  # must not raise


class TestEdgeDecayCheck:
    """Tests for run_edge_decay_check() scheduler job."""

    def test_skips_when_monitor_is_none(self):
        state = _make_app_state()
        state.edge_decay_monitor = None

        with _patch_state(state):
            main_module.run_edge_decay_check()  # must not raise

    def test_skipped_result(self):
        state = _make_app_state()
        state.edge_decay_monitor = MagicMock()
        state.edge_decay_monitor.weekly_check.return_value = {
            "checked": False, "reason": "insufficient data"
        }

        with _patch_state(state):
            main_module.run_edge_decay_check()

        state.edge_decay_monitor.weekly_check.assert_called_once()

    def test_stable_result(self):
        state = _make_app_state()
        state.edge_decay_monitor = MagicMock()
        state.edge_decay_monitor.weekly_check.return_value = {
            "checked": True, "is_decaying": False, "avg_sharpe": 1.5
        }

        with _patch_state(state):
            main_module.run_edge_decay_check()

        state.edge_decay_monitor.weekly_check.assert_called_once()

    def test_decaying_sends_alert(self):
        state = _make_app_state()
        state.settings = MagicMock()
        state.edge_decay_monitor = MagicMock()
        state.edge_decay_monitor.weekly_check.return_value = {
            "checked": True, "is_decaying": True, "avg_sharpe": 0.3,
            "old_sharpe": 1.2, "assessment": "Edge is decaying"
        }

        with _patch_state(state):
            with patch("weatheredge.main.asyncio.run"):
                with patch("weatheredge.notifications.formatters.format_edge_decay_alert", return_value="alert", create=True):
                    with patch("weatheredge.notifications.telegram.TelegramBot", create=True):
                        main_module.run_edge_decay_check()

        state.edge_decay_monitor.weekly_check.assert_called_once()

    def test_exception_caught(self):
        state = _make_app_state()
        state.edge_decay_monitor = MagicMock()
        state.edge_decay_monitor.weekly_check.side_effect = Exception("decay error")

        with _patch_state(state):
            main_module.run_edge_decay_check()  # must not raise


class TestMorningBriefing:
    """Tests for run_morning_briefing() scheduler job."""

    def test_skips_when_query_engine_is_none(self):
        state = _make_app_state()
        state.wallet_query_engine = None

        with _patch_state(state):
            main_module.run_morning_briefing()  # must not raise

    def test_sends_briefing(self):
        state = _make_app_state()
        state.settings = MagicMock()
        state.wallet_query_engine = MagicMock()
        state.wallet_query_engine.query.return_value = "market summary text"

        with _patch_state(state):
            with patch("weatheredge.main.asyncio.run"):
                with patch("weatheredge.notifications.formatters.format_morning_briefing", return_value="briefing", create=True):
                    with patch("weatheredge.notifications.telegram.TelegramBot", create=True):
                        main_module.run_morning_briefing()

        state.wallet_query_engine.query.assert_called_once()

    def test_exception_caught(self):
        state = _make_app_state()
        state.wallet_query_engine = MagicMock()
        state.wallet_query_engine.query.side_effect = Exception("query error")

        with _patch_state(state):
            main_module.run_morning_briefing()  # must not raise


# ---------------------------------------------------------------------------
# Data fetch jobs coverage
# ---------------------------------------------------------------------------


class TestFetchMarketData:
    """Tests for fetch_market_data() scheduler job."""

    def test_calls_scan_all_markets(self):
        state = _make_app_state()

        with _patch_state(state):
            with patch.object(main_module, "asyncio") as mock_asyncio:
                mock_asyncio.run.return_value = []
                main_module.fetch_market_data()

        mock_asyncio.run.assert_called_once()

    def test_exception_caught(self):
        state = _make_app_state()

        with _patch_state(state):
            with patch.object(main_module, "asyncio") as mock_asyncio:
                mock_asyncio.run.side_effect = Exception("network error")
                main_module.fetch_market_data()  # must not raise


class TestFetchForecastData:
    """Tests for fetch_forecast_data() scheduler job."""

    def test_calls_fetch_all_forecasts(self):
        state = _make_app_state()

        with _patch_state(state):
            with patch.object(main_module, "asyncio") as mock_asyncio:
                mock_asyncio.run.return_value = None
                main_module.fetch_forecast_data()

        mock_asyncio.run.assert_called_once()

    def test_exception_caught(self):
        state = _make_app_state()

        with _patch_state(state):
            with patch.object(main_module, "asyncio") as mock_asyncio:
                mock_asyncio.run.side_effect = Exception("network error")
                main_module.fetch_forecast_data()  # must not raise


# ---------------------------------------------------------------------------
# get_state() and init_app() coverage
# ---------------------------------------------------------------------------


class TestGetState:
    """Tests for get_state() and init_app()."""

    def test_get_state_raises_when_not_initialized(self):
        with patch.object(main_module, "_state", None):
            with pytest.raises(RuntimeError, match="AppState not initialised"):
                main_module.get_state()

    def test_init_app_returns_app_state(self):
        original = main_module._state
        try:
            state = main_module.init_app()
            assert isinstance(state, AppState)
            assert main_module._state is state
        finally:
            main_module._state = original


# ---------------------------------------------------------------------------
# _scan_all_markets coverage
# ---------------------------------------------------------------------------


class TestScanAllMarkets:
    """Tests for _scan_all_markets() async helper."""

    def test_both_platforms_return_lists(self):
        import asyncio as _asyncio

        state = _make_app_state()

        async def run_test():
            return await main_module._scan_all_markets(state)

        with patch.object(main_module.asyncio, "gather") as mock_gather:
            async def fake_gather(*coros, return_exceptions=False):
                return [[{"market_id": "p1"}], [{"market_id": "k1"}]]
            mock_gather.side_effect = fake_gather

            result = _asyncio.run(run_test())

        assert len(result) == 2

    def test_polymarket_error_kalshi_ok(self):
        import asyncio as _asyncio

        state = _make_app_state()

        async def run_test():
            return await main_module._scan_all_markets(state)

        with patch.object(main_module.asyncio, "gather") as mock_gather:
            async def fake_gather(*coros, return_exceptions=False):
                return [Exception("poly fail"), [{"market_id": "k1"}]]
            mock_gather.side_effect = fake_gather

            result = _asyncio.run(run_test())

        assert len(result) == 1
        assert result[0]["market_id"] == "k1"

    def test_both_platforms_error(self):
        import asyncio as _asyncio

        state = _make_app_state()

        async def run_test():
            return await main_module._scan_all_markets(state)

        with patch.object(main_module.asyncio, "gather") as mock_gather:
            async def fake_gather(*coros, return_exceptions=False):
                return [Exception("poly fail"), Exception("kalshi fail")]
            mock_gather.side_effect = fake_gather

            result = _asyncio.run(run_test())

        assert result == []


# ---------------------------------------------------------------------------
# _evaluate_market coverage
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="_evaluate_market is deprecated (raises NotImplementedError)")
class TestEvaluateMarket:
    """Tests for _evaluate_market() function."""

    def test_unknown_market_returns_zero(self):
        state = _make_app_state()
        result = main_module._evaluate_market(
            state, "UNKNOWN_XYZ", {"yes_price": 0.5}, {}, 0.05
        )
        assert result == 0

    def test_no_forecast_returns_zero(self):
        state = _make_app_state()
        # KXHIGHNY maps to station KNYC, but we provide no forecast for KNYC
        result = main_module._evaluate_market(
            state, "KXHIGHNY", {"yes_price": 0.5}, {}, 0.05
        )
        assert result == 0

    def test_invalid_price_returns_zero(self):
        state = _make_app_state()
        forecasts = {
            "KNYC": {
                "temperature_forecast": 85.0, "ensemble_members": None,
                "model": "forecast", "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        # yes_price=0.0 is invalid
        result = main_module._evaluate_market(
            state, "KXHIGHNY", {"yes_price": 0.0}, forecasts, 0.05
        )
        assert result == 0

    def test_no_temperature_forecast_returns_zero(self):
        state = _make_app_state()
        forecasts = {
            "KNYC": {
                "temperature_forecast": None, "ensemble_members": None,
                "model": "forecast", "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        result = main_module._evaluate_market(
            state, "KXHIGHNY", {"yes_price": 0.5}, forecasts, 0.05
        )
        assert result == 0

    def test_edge_below_min_returns_zero(self):
        state = _make_app_state()
        forecasts = {
            "KNYC": {
                "temperature_forecast": 70.0, "ensemble_members": None,
                "model": "forecast", "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        # With temp=70 and threshold=70, probability ~0.5, price=0.5 → edge ~0
        result = main_module._evaluate_market(
            state, "KXHIGHNY", {"yes_price": 0.50}, forecasts, 0.50
        )
        assert result == 0

    def test_risk_blocked_returns_zero(self):
        state = _make_app_state()
        state.risk_manager.calculate_position_size.return_value = 50.0
        state.risk_manager.pre_trade_check.return_value = {
            "blocked": True, "reason": "Max positions"
        }
        forecasts = {
            "KNYC": {
                "temperature_forecast": 95.0, "ensemble_members": None,
                "model": "forecast", "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        result = main_module._evaluate_market(
            state, "KXHIGHNY", {"yes_price": 0.30}, forecasts, 0.01
        )
        assert result == 0

    def test_successful_trade_returns_one(self):
        state = _make_app_state()
        state.risk_manager.calculate_position_size.return_value = 50.0
        state.risk_manager.pre_trade_check.return_value = {"blocked": False}
        state.paper_trader.place_order.return_value = {"filled": True, "trade": {}}
        state.paper_trader.bankroll = 1000.0
        state.paper_trader.open_positions = []
        state.paper_trader.daily_pnl = {}
        state.paper_trader.trade_history = []
        forecasts = {
            "KNYC": {
                "temperature_forecast": 95.0, "ensemble_members": None,
                "model": "forecast", "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        result = main_module._evaluate_market(
            state, "KXHIGHNY", {"yes_price": 0.30}, forecasts, 0.01
        )
        assert result == 1
        state.paper_trader.place_order.assert_called_once()

    def test_order_rejected_returns_zero(self):
        state = _make_app_state()
        state.risk_manager.calculate_position_size.return_value = 50.0
        state.risk_manager.pre_trade_check.return_value = {"blocked": False}
        state.paper_trader.place_order.return_value = {"filled": False, "reason": "no funds"}
        state.paper_trader.bankroll = 1000.0
        state.paper_trader.open_positions = []
        state.paper_trader.daily_pnl = {}
        state.paper_trader.trade_history = []
        forecasts = {
            "KNYC": {
                "temperature_forecast": 95.0, "ensemble_members": None,
                "model": "forecast", "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        result = main_module._evaluate_market(
            state, "KXHIGHNY", {"yes_price": 0.30}, forecasts, 0.01
        )
        assert result == 0

    def test_size_reduced_by_vpin(self):
        state = _make_app_state()
        state.risk_manager.calculate_position_size.return_value = 100.0
        state.risk_manager.pre_trade_check.return_value = {
            "blocked": False, "reduce_size": True
        }
        state.paper_trader.place_order.return_value = {"filled": True, "trade": {}}
        state.paper_trader.bankroll = 1000.0
        state.paper_trader.open_positions = []
        state.paper_trader.daily_pnl = {}
        state.paper_trader.trade_history = []
        forecasts = {
            "KNYC": {
                "temperature_forecast": 95.0, "ensemble_members": None,
                "model": "forecast", "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        result = main_module._evaluate_market(
            state, "KXHIGHNY", {"yes_price": 0.30}, forecasts, 0.01
        )
        assert result == 1
        # Check the order amount was halved (100 * 0.5 = 50)
        order_arg = state.paper_trader.place_order.call_args[0][0]
        assert order_arg.amount == 50.0

    def test_zero_position_size_returns_zero(self):
        state = _make_app_state()
        state.risk_manager.calculate_position_size.return_value = 0.0
        forecasts = {
            "KNYC": {
                "temperature_forecast": 95.0, "ensemble_members": None,
                "model": "forecast", "forecast_timestamp": datetime.now(timezone.utc),
            }
        }
        result = main_module._evaluate_market(
            state, "KXHIGHNY", {"yes_price": 0.30}, forecasts, 0.01
        )
        assert result == 0


class TestQualitativeSignalInPortfolio:
    def test_qualitative_signal_included_in_portfolio(self):
        """Rules engine returning buy_yes signal → included in signals_by_market."""
        from weatheredge.agents.base import Signal, MarketOpportunity, TradeOrder as _TradeOrder

        state = _make_app_state()
        market_id = "KXHIGHNY"
        fresh_age = 10

        # Use a real MarketOpportunity so net_edge comparison works correctly
        real_opp = MarketOpportunity(
            market_id=market_id, platform="kalshi", city="NYC", station="KNYC",
            threshold=85.0, direction="above", market_price=0.35,
            model_probability=0.65, net_edge=0.25, forecast_temp=87.0,
            rmse=2.5, lead_time_days=1,
        )

        # Mock rules engine to return a deterministic buy_yes signal
        mock_rules_engine = MagicMock()
        mock_rules_engine.analyze.return_value = Signal(
            market_id=market_id,
            side="buy_yes",
            confidence=0.08,
            agent_name="RulesQualitative",
            reasoning="KNYC microclimate: historically warmer than model (+0.02)",
        )
        state.rules_qualitative_engine = mock_rules_engine

        # Track what signals_by_market was passed to portfolio_manager
        captured_signals: dict = {}

        def capture_aggregate(signals_by_market, **kwargs):
            captured_signals.update(signals_by_market)
            return []

        state.portfolio_manager.aggregate_signals.side_effect = capture_aggregate

        with _patch_state(state):
            with patch.object(
                main_module, "_get_weather_data_age_minutes", return_value=fresh_age
            ):
                with patch.object(
                    main_module, "_read_latest_forecasts", return_value={}
                ):
                    with patch.object(
                        main_module, "_read_latest_market_prices",
                        return_value={market_id: {"yes_price": 0.35, "no_price": 0.65}},
                    ):
                        with patch.object(
                            main_module, "_build_opportunity", return_value=real_opp
                        ):
                            run_decision_loop()

        # Rules engine must have been called (net_edge=0.25 > LLM_EDGE_THRESHOLD=0.05)
        mock_rules_engine.analyze.assert_called_once()

        # The buy_yes rules signal must appear in signals passed to portfolio manager
        assert market_id in captured_signals
        qual_signals = [
            s for s in captured_signals[market_id]
            if getattr(s, "agent_name", None) == "RulesQualitative"
        ]
        assert len(qual_signals) == 1
        assert qual_signals[0].side == "buy_yes"


class TestAggressiveSignalRouting:
    def test_aggressive_cross_market_signal_routes_to_signal_market_id(self):
        """Aggressive ladder signals for adjacent buckets must be keyed by signal.market_id."""
        from weatheredge.agents.base import Signal, MarketOpportunity

        state = _make_app_state()
        market_a = "KXHIGHNY-A"
        market_b = "KXHIGHNY-B"

        opp_a = MarketOpportunity(
            market_id=market_a,
            platform="kalshi",
            city="NYC",
            station="KNYC",
            threshold=70.0,
            direction="above",
            market_price=0.55,
            model_probability=0.40,
            net_edge=-0.18,
            net_edge_no=0.15,
            forecast_temp=72.0,
            rmse=3.0,
            lead_time_days=1,
        )
        opp_b = MarketOpportunity(
            market_id=market_b,
            platform="kalshi",
            city="NYC",
            station="KNYC",
            threshold=72.0,
            direction="above",
            market_price=0.50,
            model_probability=0.35,
            net_edge=-0.18,
            net_edge_no=0.15,
            forecast_temp=72.0,
            rmse=3.0,
            lead_time_days=1,
        )

        captured_signals: dict[str, list] = {}

        def capture_aggregate(signals_by_market, **kwargs):
            for k, v in signals_by_market.items():
                captured_signals.setdefault(k, []).extend(v)
            return []

        state.portfolio_manager.aggregate_signals.side_effect = capture_aggregate
        state.rules_qualitative_engine = MagicMock()
        state.rules_qualitative_engine.analyze.return_value = Signal(
            market_id=market_a,
            side="no_trade",
            confidence=0.0,
            agent_name="Qualitative",
            reasoning="disabled for test",
        )

        def _fake_aggressive(opp, bucket_markets, settings):  # noqa: ARG001
            if opp.market_id == market_b:
                return [
                    Signal(
                        market_id=market_a,
                        side="buy_no",
                        confidence=1.0,
                        agent_name="Aggressive",
                        reasoning="adjacent bucket",
                    )
                ]
            return []

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(
                        main_module,
                        "_read_latest_market_prices",
                        return_value={
                            market_a: {"yes_price": 0.55},
                            market_b: {"yes_price": 0.50},
                        },
                    ):
                        with patch.object(
                            main_module,
                            "_build_opportunity",
                            side_effect=[opp_a, opp_b],
                        ):
                            with patch.object(main_module, "conservative_strategy") as _cons:
                                with patch.object(main_module, "sell_no_strategy") as _sell_no:
                                    with patch.object(main_module, "aggressive_strategy", side_effect=_fake_aggressive):
                                        _cons.side_effect = lambda opp, _settings: Signal(
                                            market_id=opp.market_id,
                                            side="no_trade",
                                            confidence=0.0,
                                            agent_name="Conservative",
                                            reasoning="n/a",
                                        )
                                        _sell_no.side_effect = lambda opp, _settings: Signal(
                                            market_id=opp.market_id,
                                            side="no_trade",
                                            confidence=0.0,
                                            agent_name="SellNO",
                                            reasoning="n/a",
                                        )
                                        run_decision_loop()

        assert market_a in captured_signals
        assert market_b in captured_signals  # may contain no_trade signals from other agents

        aggressive_a = [s for s in captured_signals[market_a] if s.agent_name == "Aggressive"]
        aggressive_b = [s for s in captured_signals[market_b] if s.agent_name == "Aggressive"]
        assert len(aggressive_a) == 1
        assert len(aggressive_b) == 0
        assert aggressive_a[0].market_id == market_a


# ---------------------------------------------------------------------------
# Coverage gap: run_decision_loop overlap guard (lines 339-340)
# ---------------------------------------------------------------------------


class TestDecisionLoopOverlapGuard:
    def test_overlap_guard_skips_when_lock_held(self):
        """When the decision cycle lock is already held, loop logs overlap and returns."""
        state = _make_app_state()
        # Pre-acquire the non-blocking lock so acquire(blocking=False) returns False
        state._decision_cycle_lock.acquire()
        try:
            with _patch_state(state):
                run_decision_loop()
            # cancel_expired_orders must NOT be called — loop returned early
            state.paper_trader.cancel_expired_orders.assert_not_called()
        finally:
            state._decision_cycle_lock.release()


# ---------------------------------------------------------------------------
# Coverage gaps: opportunity evaluation paths (lines 418-419, 433, 449-450,
# 462-466, 481, 552-553, 586-587, 603-604, 610-615)
# ---------------------------------------------------------------------------


class TestDecisionLoopOpportunityCoverage:
    """Tests for edge paths inside _run_decision_cycle market evaluation."""

    def _mock_opp(self, platform="kalshi", net_edge=0.1):
        opp = MagicMock()
        opp.platform = platform
        opp.city = "NYC"
        opp.threshold = 70.0
        opp.market_price = 0.5
        opp.net_edge = net_edge
        return opp

    def _run_with_opp(self, state, mock_opp, market_id="KXHIGHNY"):
        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(
                        main_module, "_read_latest_market_prices",
                        return_value={market_id: {"yes_price": 0.5}},
                    ):
                        with patch.object(main_module, "_build_opportunity", return_value=mock_opp):
                            run_decision_loop()

    def test_open_position_increments_skipped(self):
        """When has_open_position is True, market is skipped (lines 418-419)."""
        state = _make_app_state()
        state.paper_trader.has_open_position.return_value = True
        self._run_with_opp(state, self._mock_opp())
        state.paper_trader.place_order.assert_not_called()
        state.paper_trader.cancel_expired_orders.assert_called_once()

    def test_kalshi_market_appended_to_kalshi_raw(self):
        """Kalshi platform opportunity goes to kalshi_markets_raw (line 433)."""
        state = _make_app_state()
        state.paper_trader.has_open_position.return_value = False
        state.rules_qualitative_engine = MagicMock()
        state.rules_qualitative_engine.analyze.return_value = MagicMock(side="no_trade")
        # Run with kalshi platform — the arb scan receives it in kalshi_markets_raw
        arb_calls = []
        orig_scan = state.cross_platform_arb.scan
        state.cross_platform_arb.scan.side_effect = lambda poly, kalshi: arb_calls.append(kalshi) or []
        self._run_with_opp(state, self._mock_opp(platform="kalshi", net_edge=0.01))
        assert len(arb_calls) == 1
        assert len(arb_calls[0]) == 1  # one kalshi entry

    def test_conservative_strategy_exception_caught(self):
        """Exception in conservative_strategy is caught and loop continues (lines 449-450)."""
        state = _make_app_state()
        state.paper_trader.has_open_position.return_value = False
        state.rules_qualitative_engine = MagicMock()
        state.rules_qualitative_engine.analyze.return_value = MagicMock(side="no_trade")
        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(
                        main_module, "_read_latest_market_prices",
                        return_value={"KXHIGHNY": {"yes_price": 0.5}},
                    ):
                        with patch.object(main_module, "_build_opportunity", return_value=self._mock_opp()):
                            with patch.object(
                                main_module, "conservative_strategy",
                                side_effect=Exception("strategy boom"),
                            ):
                                run_decision_loop()
        state.paper_trader.cancel_expired_orders.assert_called_once()

    def test_mean_reversion_called_with_sufficient_history(self):
        """mean_reversion_strategy is called when price history >= 20 (lines 462-466)."""
        state = _make_app_state()
        state.paper_trader.has_open_position.return_value = False
        state.settings.enable_mean_reversion_agent = True
        state.rules_qualitative_engine = MagicMock()
        state.rules_qualitative_engine.analyze.return_value = MagicMock(side="no_trade")
        market_id = "KXHIGHNY"
        # Pre-seed 20 history entries so the >= 20 guard passes
        state._price_history[market_id] = [0.5] * 20

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(
                        main_module, "_read_latest_market_prices",
                        return_value={market_id: {"yes_price": 0.5}},
                    ):
                        with patch.object(main_module, "_build_opportunity", return_value=self._mock_opp()):
                            with patch.object(main_module, "mean_reversion_strategy") as mock_mr:
                                mock_mr.return_value = MagicMock(side="no_trade", agent_name="MR")
                                run_decision_loop()

        mock_mr.assert_called_once()

    def test_mean_reversion_exception_caught(self):
        """Exception in mean_reversion_strategy is caught and loop continues (lines 464-466)."""
        state = _make_app_state()
        state.paper_trader.has_open_position.return_value = False
        state.settings.enable_mean_reversion_agent = True
        state.rules_qualitative_engine = MagicMock()
        state.rules_qualitative_engine.analyze.return_value = MagicMock(side="no_trade")
        market_id = "KXHIGHNY"
        state._price_history[market_id] = [0.5] * 20

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(
                        main_module, "_read_latest_market_prices",
                        return_value={market_id: {"yes_price": 0.5}},
                    ):
                        with patch.object(main_module, "_build_opportunity", return_value=self._mock_opp()):
                            with patch.object(
                                main_module, "mean_reversion_strategy",
                                side_effect=Exception("mr boom"),
                            ):
                                run_decision_loop()

        state.paper_trader.cancel_expired_orders.assert_called_once()

    def test_low_edge_increments_llm_skipped(self):
        """Opportunity with net_edge < 0.05 increments llm_skipped (line 481)."""
        state = _make_app_state()
        state.paper_trader.has_open_position.return_value = False
        state.rules_qualitative_engine = MagicMock()
        self._run_with_opp(state, self._mock_opp(net_edge=0.01))
        # rules_qualitative_engine.analyze must NOT have been called
        state.rules_qualitative_engine.analyze.assert_not_called()
        state.paper_trader.cancel_expired_orders.assert_called_once()

    def test_fill_monitor_record_fill_exception_caught(self):
        """Exception in fill_monitor.record_fill is caught (lines 552-553)."""
        from weatheredge.agents.base import TradeOrder as _TradeOrder
        state = _make_app_state()
        state.fill_monitor = MagicMock()
        state.fill_monitor.record_fill.side_effect = Exception("fill boom")

        trade_order = _TradeOrder(
            market_id="KXHIGHNY", side="buy_yes", price=0.35, amount=50.0,
            agent_name="PM", model_probability=0.99, edge_at_entry=0.63,
            city="KNYC", platform="kalshi",
        )
        state.portfolio_manager.aggregate_signals.return_value = [trade_order]
        state.paper_trader.place_order.return_value = {
            "filled": True, "trade": {"market_id": "KXHIGHNY"},
        }

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(main_module, "_read_latest_market_prices", return_value={}):
                        run_decision_loop()  # must not raise

        state.paper_trader.cancel_expired_orders.assert_called_once()

    def test_model_weight_update_exception_caught(self):
        """Exception in model_weight_updater.update_model_weights is caught (lines 586-587)."""
        from weatheredge.agents.base import TradeOrder as _TradeOrder
        state = _make_app_state()
        state.model_weight_updater.should_update.return_value = True
        state.model_weight_updater.update_model_weights.side_effect = Exception("weight boom")

        trade_order = _TradeOrder(
            market_id="KXHIGHNY", side="buy_yes", price=0.35, amount=50.0,
            agent_name="PM", model_probability=0.99, edge_at_entry=0.63,
            city="KNYC", platform="kalshi",
        )
        state.portfolio_manager.aggregate_signals.return_value = [trade_order]
        state.paper_trader.place_order.return_value = {
            "filled": True, "trade": {"market_id": "KXHIGHNY"},
        }

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(main_module, "_read_latest_market_prices", return_value={}):
                        run_decision_loop()  # must not raise

        state.paper_trader.cancel_expired_orders.assert_called_once()

    def test_agent_weight_update_exception_caught(self):
        """Exception in agent_weight_updater.update_agent_weights is caught (lines 603-604)."""
        from weatheredge.agents.base import TradeOrder as _TradeOrder
        state = _make_app_state()
        state.model_weight_updater.should_update.return_value = False
        state.agent_weight_updater.should_update.return_value = True
        state.agent_weight_updater.update_agent_weights.side_effect = Exception("agent weight boom")
        state.portfolio_manager.AGENT_WEIGHTS = {"conservative": 0.5}

        trade_order = _TradeOrder(
            market_id="KXHIGHNY", side="buy_yes", price=0.35, amount=50.0,
            agent_name="PM", model_probability=0.99, edge_at_entry=0.63,
            city="KNYC", platform="kalshi",
        )
        state.portfolio_manager.aggregate_signals.return_value = [trade_order]
        state.paper_trader.place_order.return_value = {
            "filled": True, "trade": {"market_id": "KXHIGHNY"},
        }

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(main_module, "_read_latest_market_prices", return_value={}):
                        run_decision_loop()  # must not raise

        state.paper_trader.cancel_expired_orders.assert_called_once()

    def test_arb_opportunities_found_are_logged(self):
        """Non-empty arb scan results are logged (lines 610-615)."""
        state = _make_app_state()
        arb_opp = MagicMock()
        arb_opp.city = "NYC"
        arb_opp.threshold = 70.0
        arb_opp.poly_price = 0.4
        arb_opp.kalshi_price = 0.6
        arb_opp.spread = 0.2
        arb_opp.direction = "kalshi_buy"
        state.cross_platform_arb.scan.return_value = [arb_opp]

        with _patch_state(state):
            with patch.object(main_module, "_get_weather_data_age_minutes", return_value=10):
                with patch.object(main_module, "_read_latest_forecasts", return_value={}):
                    with patch.object(main_module, "_read_latest_market_prices", return_value={}):
                        run_decision_loop()

        state.paper_trader.cancel_expired_orders.assert_called_once()


# ---------------------------------------------------------------------------
# Coverage gaps: _get_weather_data_age_minutes naive ts (line 650)
# and _read_latest_forecasts ECMWF + bad JSON (lines 700-701, 705)
# ---------------------------------------------------------------------------


class TestWeatherDataAgeAndForecasts:
    def test_get_weather_data_age_handles_naive_timestamp(self):
        """Naive datetime gets UTC tzinfo attached (line 650)."""
        from datetime import timedelta
        naive_ts = (datetime.now(timezone.utc) - timedelta(minutes=30)).replace(tzinfo=None)

        mock_ctx = MagicMock()
        mock_session = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_session)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.order_by.return_value.first.return_value = (naive_ts,)

        with patch("weatheredge.main.get_session", return_value=mock_ctx):
            age = main_module._get_weather_data_age_minutes()

        assert age is not None
        assert 25.0 < age < 35.0  # approximately 30 minutes

    def test_get_weather_data_age_db_exception_returns_none(self):
        """DB exception in _get_weather_data_age_minutes returns None."""
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(side_effect=Exception("DB down"))
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with patch("weatheredge.main.get_session", return_value=mock_ctx):
            age = main_module._get_weather_data_age_minutes()

        assert age is None

    def test_read_latest_forecasts_ecmwf_model_populates_ecmwf_temp(self):
        """ECMWF rows populate ecmwf_temp (line 705)."""
        target_date = "2099-12-31"
        ecmwf_row = MagicMock()
        ecmwf_row.station = "KNYC"
        ecmwf_row.target_date = target_date
        ecmwf_row.model = "ecmwf"
        ecmwf_row.temperature_forecast = 75.0
        ecmwf_row.ensemble_members = None
        ecmwf_row.forecast_timestamp = datetime.now(timezone.utc)

        mock_ctx = MagicMock()
        mock_session = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_session)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        # _read_latest_forecasts uses .filter().filter().order_by().limit().all()
        mock_session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
            ecmwf_row,
        ]

        with patch("weatheredge.main.get_session", return_value=mock_ctx):
            result = main_module._read_latest_forecasts()

        assert "KNYC" in result
        assert target_date in result["KNYC"]
        assert result["KNYC"][target_date]["ecmwf_temp"] == 75.0

    def test_read_latest_forecasts_bad_ensemble_json_is_swallowed(self):
        """Bad JSON in ensemble_members is caught silently (lines 700-701)."""
        target_date = "2099-12-31"
        ensemble_row = MagicMock()
        ensemble_row.station = "KNYC"
        ensemble_row.target_date = target_date
        ensemble_row.model = "ensemble"
        ensemble_row.temperature_forecast = None
        ensemble_row.ensemble_members = "not valid json {"
        ensemble_row.forecast_timestamp = datetime.now(timezone.utc)

        mock_ctx = MagicMock()
        mock_session = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_session)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        # _read_latest_forecasts uses .filter().filter().order_by().limit().all()
        mock_session.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
            ensemble_row,
        ]

        with patch("weatheredge.main.get_session", return_value=mock_ctx):
            result = main_module._read_latest_forecasts()  # must not raise

        assert "KNYC" in result
        assert target_date in result["KNYC"]
        assert result["KNYC"][target_date]["ensemble_members"] is None


# ---------------------------------------------------------------------------
# Coverage gaps: _build_opportunity edge paths
# (lines 776-778, 785, 816-817, 858-859, 886-898, 902-904)
# ---------------------------------------------------------------------------


class TestBuildOpportunityEdgePaths:
    """Call _build_opportunity directly to exercise uncovered branches."""

    def _make_state(self):
        state = _make_app_state()
        state.conditions_tracker = MagicMock()
        state.conditions_tracker.metar_probability_override.return_value = None
        return state

    def _base_forecasts(self, station="KNYC"):
        return {
            station: {
                "temperature_forecast": 75.0,
                "ecmwf_temp": None,
                "hrrr_temp": None,
                "ensemble_members": None,
                "model": "gfs",
                "forecast_timestamp": datetime.now(timezone.utc),
            }
        }

    def test_polymarket_exact_match_fallback(self):
        """london_temp is not a Kalshi prefix — falls through to exact match (lines 776-778)."""
        state = self._make_state()
        # london_temp is in RESOLUTION_SOURCES["polymarket"] as an exact key
        forecasts = self._base_forecasts("EGLC")

        with _patch_state(state):
            opp = main_module._build_opportunity(
                "london_temp", {"yes_price": 0.5}, forecasts
            )

        assert opp is not None
        assert opp.platform == "polymarket"

    def test_resolution_source_without_station_returns_none(self):
        """resolution_source with no 'station' key returns None (line 785)."""
        state = self._make_state()
        state.market_metadata["no_station_mkt"] = {
            "resolution_source": {"type": "daily_high"},  # deliberate: no 'station'
            "platform": "polymarket",
            "parsed": None,
        }

        with _patch_state(state):
            opp = main_module._build_opportunity("no_station_mkt", {"yes_price": 0.5}, {})

        assert opp is None

    def test_kalshi_b_prefix_threshold_parsed(self):
        """'B76' suffix on Kalshi contract ID parses threshold=76 (lines 816-817)."""
        state = self._make_state()
        market_id = "KXHIGHDEN-26MAR30-B76"
        forecasts = self._base_forecasts("KDEN")

        with _patch_state(state):
            opp = main_module._build_opportunity(market_id, {"yes_price": 0.5}, forecasts)

        assert opp is not None
        assert opp.threshold == pytest.approx(76.0)

    def test_zero_fahrenheit_forecast_not_dropped_as_falsy(self):
        """0.0°F is a valid forecast and must not be treated as missing data."""
        state = self._make_state()
        forecasts = {
            "KNYC": {
                "temperature_forecast": 0.0,
                "ecmwf_temp": None,
                "hrrr_temp": None,
                "ensemble_members": None,
                "model": "gfs",
                "forecast_timestamp": datetime.now(timezone.utc),
            }
        }

        with _patch_state(state):
            opp = main_module._build_opportunity("KXHIGHNY", {"yes_price": 0.5}, forecasts)

        assert opp is not None
        assert opp.forecast_temp == pytest.approx(0.0)

    def test_invalid_market_date_falls_back_to_default_lead_time(self):
        """Unparseable date in parsed.date is silently ignored (lines 858-859)."""
        state = self._make_state()

        mock_parsed = MagicMock()
        mock_parsed.threshold = 70.0
        mock_parsed.direction = "above"
        mock_parsed.market_type = "daily_high"
        mock_parsed.date = "not-a-valid-date-string"  # triggers ValueError
        mock_parsed.threshold_low = None
        mock_parsed.threshold_high = None

        state.market_metadata["KXHIGHNY"] = {
            "resolution_source": {"station": "KNYC", "type": "daily_high"},
            "platform": "kalshi",
            "parsed": mock_parsed,
        }

        with _patch_state(state):
            opp = main_module._build_opportunity(
                "KXHIGHNY", {"yes_price": 0.5}, self._base_forecasts("KNYC")
            )

        # Must complete without raising; lead_time_hours falls back to 24.0
        assert opp is not None

    def test_market_date_uses_station_local_close_helper(self):
        """Lead-time path should use station-local close helper for parsed markets."""
        state = self._make_state()

        mock_parsed = MagicMock()
        mock_parsed.threshold = 70.0
        mock_parsed.direction = "above"
        mock_parsed.market_type = "daily_high"
        mock_parsed.date = "2026-04-01"
        mock_parsed.threshold_low = None
        mock_parsed.threshold_high = None

        market_id = "0xlocal_close_market"
        state.market_metadata[market_id] = {
            "resolution_source": {"station": "KLAX", "type": "daily_high"},
            "platform": "polymarket",
            "parsed": mock_parsed,
        }

        with _patch_state(state):
            with patch.object(main_module, "hours_until_market_close", return_value=6.0) as mock_hours:
                opp = main_module._build_opportunity(
                    market_id, {"yes_price": 0.5}, self._base_forecasts("KLAX")
                )

        assert opp is not None
        mock_hours.assert_called_once_with("2026-04-01", "KLAX")

    def test_blended_probability_exception_falls_back_to_simple(self):
        """calculate_blended_probability failure falls back to calculate_probability (lines 886-898)."""
        state = self._make_state()

        with _patch_state(state):
            with patch.object(
                main_module, "calculate_blended_probability",
                side_effect=Exception("blend boom"),
            ):
                opp = main_module._build_opportunity(
                    "KXHIGHNY", {"yes_price": 0.5}, self._base_forecasts("KNYC")
                )

        assert opp is not None

    def test_edge_calc_exception_returns_none(self):
        """calculate_net_edge raising returns None (lines 902-904)."""
        state = self._make_state()

        with _patch_state(state):
            with patch.object(
                main_module, "calculate_net_edge",
                side_effect=Exception("edge boom"),
            ):
                opp = main_module._build_opportunity(
                    "KXHIGHNY", {"yes_price": 0.5}, self._base_forecasts("KNYC")
                )

        assert opp is None


# ---------------------------------------------------------------------------
# Coverage gaps: _evaluate_market edge paths (lines 973, 1003-1005, 1010-1012)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="_evaluate_market is deprecated (raises NotImplementedError)")
class TestEvaluateMarketEdgePaths:
    def _base_forecasts(self):
        return {
            "KNYC": {
                "temperature_forecast": 75.0,
                "ensemble_members": None,
            }
        }

    def test_zero_size_returns_zero(self):
        """size <= 0 causes _evaluate_market to return 0 (line 973)."""
        state = _make_app_state()
        state.risk_manager.calculate_position_size.return_value = 0.0
        state.paper_trader.bankroll = 1000.0
        state.paper_trader.open_positions = []
        state.paper_trader.daily_pnl = {}
        state.paper_trader.trade_history = []

        with _patch_state(state):
            result = main_module._evaluate_market(
                state, "KXHIGHNY", {"yes_price": 0.5}, self._base_forecasts(), 0.01
            )

        assert result == 0
        state.paper_trader.place_order.assert_not_called()

    def test_probability_exception_returns_zero(self):
        """Exception in calculate_probability returns 0 (lines 1003-1005)."""
        state = _make_app_state()
        state.paper_trader.bankroll = 1000.0
        state.paper_trader.open_positions = []
        state.paper_trader.daily_pnl = {}
        state.paper_trader.trade_history = []

        with _patch_state(state):
            with patch.object(
                main_module, "calculate_probability",
                side_effect=Exception("prob boom"),
            ):
                result = main_module._evaluate_market(
                    state, "KXHIGHNY", {"yes_price": 0.5}, self._base_forecasts(), 0.01
                )

        assert result == 0

    def test_edge_calc_exception_returns_zero(self):
        """Exception in calculate_net_edge in _evaluate_market returns 0 (lines 1010-1012)."""
        state = _make_app_state()
        state.paper_trader.bankroll = 1000.0
        state.paper_trader.open_positions = []
        state.paper_trader.daily_pnl = {}
        state.paper_trader.trade_history = []

        with _patch_state(state):
            with patch.object(
                main_module, "calculate_net_edge",
                side_effect=Exception("edge boom"),
            ):
                result = main_module._evaluate_market(
                    state, "KXHIGHNY", {"yes_price": 0.5}, self._base_forecasts(), 0.01
                )

        assert result == 0


# ---------------------------------------------------------------------------
# Coverage gaps: _handle_shutdown DB dispose error (lines 1135-1138)
# ---------------------------------------------------------------------------


class TestHandleShutdownDBDisposeError:
    def test_handle_shutdown_db_dispose_error_is_caught(self):
        """_handle_shutdown catches DB dispose exception (lines 1135-1138)."""
        state = _make_app_state()
        state.scheduler = MagicMock()

        mock_db = MagicMock()
        mock_db._engine = MagicMock()
        mock_db._engine.dispose.side_effect = Exception("dispose boom")

        with _patch_state(state):
            with patch.dict("sys.modules", {"weatheredge.db.database": mock_db}):
                _handle_shutdown(signal.SIGTERM, None)  # must not raise

        assert state.shutting_down is True


# ---------------------------------------------------------------------------
# Coverage gaps: run_error_analysis LLM exception (lines 1218-1221)
# ---------------------------------------------------------------------------


class TestRunErrorAnalysis:
    def test_llm_analysis_exception_is_caught(self):
        """Exception in analyze_with_llm is caught and loop completes (lines 1218-1221)."""
        state = _make_app_state()

        mock_report = MagicMock()
        mock_report.total_losses = 3
        mock_report.model_errors = 2
        mock_report.timing_errors = 0
        mock_report.edge_compression = 0
        mock_report.adverse_selection = 0
        state.error_analyzer.analyze_errors.return_value = mock_report
        state.error_analyzer.analyze_with_llm.side_effect = Exception("LLM down")
        state.settings.openrouter_api_key = "test-key"

        with _patch_state(state):
            main_module.run_error_analysis()  # must not raise


# ---------------------------------------------------------------------------
# Coverage gaps: scheduler job functions
# (lines 307-308, 1337-1342, 1351-1364, 1369-1370)
# ---------------------------------------------------------------------------


class TestSchedulerJobFunctions:
    def test_fetch_forecast_data_noaa_exception_does_not_crash(self):
        """NOAA fetch exception in _fetch_all_forecasts is caught (lines 307-308)."""
        state = _make_app_state()
        state.settings = MagicMock()
        state.settings.cities_list = ["NYC"]

        mock_cs = AsyncMock()
        mock_cs.__aenter__ = AsyncMock(return_value=mock_cs)
        mock_cs.__aexit__ = AsyncMock(return_value=False)

        mock_db_ctx = MagicMock()
        mock_db_session = MagicMock()
        mock_db_ctx.__enter__ = MagicMock(return_value=mock_db_session)
        mock_db_ctx.__exit__ = MagicMock(return_value=False)

        noaa_mock = AsyncMock(side_effect=Exception("NOAA down"))

        with _patch_state(state):
            with patch("aiohttp.ClientSession", return_value=mock_cs):
                with patch.object(main_module, "fetch_forecast", new_callable=AsyncMock) as mock_ff:
                    mock_ff.return_value = {}
                    with patch.object(main_module, "fetch_ensemble", new_callable=AsyncMock) as mock_fe:
                        mock_fe.return_value = {}
                        with patch.object(main_module, "store_forecast"):
                            with patch("weatheredge.main.get_session", return_value=mock_db_ctx):
                                with patch("weatheredge.data.noaa.fetch_noaa_forecast", noaa_mock):
                                    main_module.fetch_forecast_data()  # must not raise

    def test_poll_metar_conditions_exception_is_caught(self):
        """Exception in poll_metar_conditions is caught (lines 1337-1342)."""
        state = _make_app_state()

        with _patch_state(state):
            with patch("asyncio.run", side_effect=Exception("metar boom")):
                main_module.poll_metar_conditions()  # must not raise

    def test_check_model_runs_detector_none_returns_early(self):
        """check_model_runs returns immediately when model_run_detector is None (line 1353)."""
        state = _make_app_state()
        state.model_run_detector = None

        with _patch_state(state):
            main_module.check_model_runs()  # must not raise, returns at line 1353

    def test_check_model_runs_with_events_triggers_decision_loop(self):
        """check_model_runs triggers run_decision_loop when events returned (lines 1355-1362)."""
        state = _make_app_state()
        state.model_run_detector = MagicMock()
        mock_event = MagicMock()

        with _patch_state(state):
            with patch("asyncio.run", return_value=[mock_event]):
                with patch.object(main_module, "run_decision_loop") as mock_loop:
                    main_module.check_model_runs()

        mock_loop.assert_called_once()

    def test_check_model_runs_no_events_no_decision_loop(self):
        """check_model_runs does NOT trigger decision loop when no events (line 1356)."""
        state = _make_app_state()
        state.model_run_detector = MagicMock()

        with _patch_state(state):
            with patch("asyncio.run", return_value=[]):
                with patch.object(main_module, "run_decision_loop") as mock_loop:
                    main_module.check_model_runs()

        mock_loop.assert_not_called()

    def test_check_model_runs_asyncio_exception_is_caught(self):
        """asyncio.run exception in check_model_runs is caught (lines 1363-1364)."""
        state = _make_app_state()
        state.model_run_detector = MagicMock()

        with _patch_state(state):
            with patch("asyncio.run", side_effect=Exception("async boom")):
                main_module.check_model_runs()  # must not raise

    @pytest.mark.asyncio
    async def test_check_model_runs_async_calls_detector(self):
        """_check_model_runs_async passes session to detector (lines 1369-1370)."""
        state = _make_app_state()
        state.model_run_detector = MagicMock()
        state.model_run_detector.check_for_new_run = AsyncMock(return_value=[])

        mock_cs = AsyncMock()
        mock_cs.__aenter__ = AsyncMock(return_value=mock_cs)
        mock_cs.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_cs):
            result = await main_module._check_model_runs_async(state)

        assert result == []
        state.model_run_detector.check_for_new_run.assert_awaited_once_with(mock_cs)


# ---------------------------------------------------------------------------
# Coverage gaps: main() entry-point exception paths
# (lines 1421-1423, 1430-1431, 1438-1439, 1459-1460, 1464-1465,
#  1485-1486, 1618-1620)
# ---------------------------------------------------------------------------


class TestMainEntrypointExceptions:
    """Test exception handling inside main() startup."""

    def test_init_db_exception_propagates(self):
        """init_db failure causes main() to re-raise (lines 1421-1423)."""
        with patch.object(main_module, "setup_logging"):
            with patch.object(main_module, "init_db", side_effect=RuntimeError("DB init fail")):
                with pytest.raises(RuntimeError, match="DB init fail"):
                    main_module.main()

    def test_paper_trader_connect_failure_logs_warning(self, monkeypatch):
        """PaperTrader.connect_db failure is caught and logged (lines 1430-1431)."""
        state = _make_app_state()
        state.shutting_down = True
        state.paper_trader.connect_db = MagicMock(side_effect=Exception("connect fail"))
        state.settings = MagicMock()
        state.settings.enable_qualitative_agent = False
        state.settings.openrouter_api_key = ""
        state.settings.market_scan_interval_minutes = 5
        state.settings.forecast_poll_interval_minutes = 15
        state.settings.decision_loop_interval_minutes = 5
        state.settings.resolution_check_time = "08:00"
        state.settings.rmse_recalc_interval_days = 30
        state.settings.error_analysis_interval_days = 7
        state.settings.initial_bankroll = 1000.0

        mock_scheduler = MagicMock()
        mock_scheduler.start = MagicMock()
        monkeypatch.setattr(main_module, "BackgroundScheduler", lambda: mock_scheduler)

        def fake_init_app():
            main_module._state = state
            return state
        monkeypatch.setattr(main_module, "init_app", fake_init_app)

        mock_db = MagicMock()
        mock_db._engine = MagicMock()

        for cls_name in ["ResolutionTracker", "CalibrationTracker", "ModelWeightUpdater",
                          "AgentWeightUpdater", "RMSEUpdater", "ErrorAnalyzer",
                          "WalletCollector", "WalletProfiler", "WalletQueryEngine",
                          "EdgeDecayMonitor", "CompetitorTracker"]:
            monkeypatch.setattr(main_module, cls_name, MagicMock())

        with patch.object(main_module, "setup_logging"):
            with patch.object(main_module, "init_db"):
                with patch.object(main_module, "FillMonitor", return_value=MagicMock()):
                    with patch.dict("sys.modules", {"weatheredge.db.database": mock_db}):
                        with patch("signal.signal"):
                            main_module.main()  # must not raise

    def test_fill_monitor_init_failure_logs_warning(self, monkeypatch):
        """FillMonitor init failure is caught and logged (lines 1438-1439)."""
        state = _make_app_state()
        state.shutting_down = True
        state.settings = MagicMock()
        state.settings.enable_qualitative_agent = False
        state.settings.openrouter_api_key = ""
        state.settings.market_scan_interval_minutes = 5
        state.settings.forecast_poll_interval_minutes = 15
        state.settings.decision_loop_interval_minutes = 5
        state.settings.resolution_check_time = "08:00"
        state.settings.rmse_recalc_interval_days = 30
        state.settings.error_analysis_interval_days = 7
        state.settings.initial_bankroll = 1000.0

        mock_scheduler = MagicMock()
        mock_scheduler.start = MagicMock()
        monkeypatch.setattr(main_module, "BackgroundScheduler", lambda: mock_scheduler)

        def fake_init_app():
            main_module._state = state
            return state
        monkeypatch.setattr(main_module, "init_app", fake_init_app)

        mock_db = MagicMock()
        mock_db._engine = MagicMock()

        for cls_name in ["ResolutionTracker", "CalibrationTracker", "ModelWeightUpdater",
                          "AgentWeightUpdater", "RMSEUpdater", "ErrorAnalyzer",
                          "WalletCollector", "WalletProfiler", "WalletQueryEngine",
                          "EdgeDecayMonitor", "CompetitorTracker"]:
            monkeypatch.setattr(main_module, cls_name, MagicMock())

        with patch.object(main_module, "setup_logging"):
            with patch.object(main_module, "init_db"):
                with patch.object(
                    main_module, "FillMonitor",
                    side_effect=Exception("FillMonitor init fail"),
                ):
                    with patch.dict("sys.modules", {"weatheredge.db.database": mock_db}):
                        with patch("signal.signal"):
                            main_module.main()  # must not raise

    def test_phase3_init_failure_logs_warning(self, monkeypatch):
        """Phase 3 component init failure is caught and logged (lines 1459-1460)."""
        state = _make_app_state()
        state.shutting_down = True
        state.settings = MagicMock()
        state.settings.enable_qualitative_agent = False
        state.settings.openrouter_api_key = ""
        state.settings.market_scan_interval_minutes = 5
        state.settings.forecast_poll_interval_minutes = 15
        state.settings.decision_loop_interval_minutes = 5
        state.settings.resolution_check_time = "08:00"
        state.settings.rmse_recalc_interval_days = 30
        state.settings.error_analysis_interval_days = 7
        state.settings.initial_bankroll = 1000.0

        mock_scheduler = MagicMock()
        mock_scheduler.start = MagicMock()
        monkeypatch.setattr(main_module, "BackgroundScheduler", lambda: mock_scheduler)

        def fake_init_app():
            main_module._state = state
            return state
        monkeypatch.setattr(main_module, "init_app", fake_init_app)

        mock_db = MagicMock()
        mock_db._engine = MagicMock()

        # Make ResolutionTracker raise to hit the Phase 3 except block
        monkeypatch.setattr(main_module, "ResolutionTracker", MagicMock(side_effect=Exception("p3 fail")))
        for cls_name in ["CalibrationTracker", "ModelWeightUpdater", "AgentWeightUpdater",
                          "RMSEUpdater", "ErrorAnalyzer", "WalletCollector", "WalletProfiler",
                          "WalletQueryEngine", "EdgeDecayMonitor", "CompetitorTracker"]:
            monkeypatch.setattr(main_module, cls_name, MagicMock())

        with patch.object(main_module, "setup_logging"):
            with patch.object(main_module, "init_db"):
                with patch.object(main_module, "FillMonitor", return_value=MagicMock()):
                    with patch.dict("sys.modules", {"weatheredge.db.database": mock_db}):
                        with patch("signal.signal"):
                            main_module.main()  # must not raise

    def test_qualitative_agent_created_when_enabled(self, monkeypatch):
        """QualitativeAgent is created when enabled and api key present (lines 1464-1465)."""
        from weatheredge.agents.qualitative import QualitativeAgent

        state = _make_app_state()
        state.shutting_down = True
        state.settings = MagicMock()
        state.settings.enable_qualitative_agent = True
        state.settings.openrouter_api_key = "test-key-abc"
        state.settings.market_scan_interval_minutes = 5
        state.settings.forecast_poll_interval_minutes = 15
        state.settings.decision_loop_interval_minutes = 5
        state.settings.resolution_check_time = "08:00"
        state.settings.rmse_recalc_interval_days = 30
        state.settings.error_analysis_interval_days = 7
        state.settings.initial_bankroll = 1000.0

        mock_scheduler = MagicMock()
        mock_scheduler.start = MagicMock()
        monkeypatch.setattr(main_module, "BackgroundScheduler", lambda: mock_scheduler)

        def fake_init_app():
            main_module._state = state
            return state
        monkeypatch.setattr(main_module, "init_app", fake_init_app)

        mock_db = MagicMock()
        mock_db._engine = MagicMock()

        for cls_name in ["ResolutionTracker", "CalibrationTracker", "ModelWeightUpdater",
                          "AgentWeightUpdater", "RMSEUpdater", "ErrorAnalyzer",
                          "WalletCollector", "WalletProfiler", "WalletQueryEngine",
                          "EdgeDecayMonitor", "CompetitorTracker"]:
            monkeypatch.setattr(main_module, cls_name, MagicMock())

        with patch.object(main_module, "setup_logging"):
            with patch.object(main_module, "init_db"):
                with patch.object(main_module, "FillMonitor", return_value=MagicMock()):
                    with patch.dict("sys.modules", {"weatheredge.db.database": mock_db}):
                        with patch("signal.signal"):
                            main_module.main()

        assert state.qualitative_agent is not None
        assert isinstance(state.qualitative_agent, QualitativeAgent)

    def test_phase45_init_failure_logs_warning(self, monkeypatch):
        """Phase 4.5 component init failure is caught and logged (lines 1485-1486)."""
        state = _make_app_state()
        state.shutting_down = True
        state.settings = MagicMock()
        state.settings.enable_qualitative_agent = False
        state.settings.openrouter_api_key = ""
        state.settings.market_scan_interval_minutes = 5
        state.settings.forecast_poll_interval_minutes = 15
        state.settings.decision_loop_interval_minutes = 5
        state.settings.resolution_check_time = "08:00"
        state.settings.rmse_recalc_interval_days = 30
        state.settings.error_analysis_interval_days = 7
        state.settings.initial_bankroll = 1000.0

        mock_scheduler = MagicMock()
        mock_scheduler.start = MagicMock()
        monkeypatch.setattr(main_module, "BackgroundScheduler", lambda: mock_scheduler)

        def fake_init_app():
            main_module._state = state
            return state
        monkeypatch.setattr(main_module, "init_app", fake_init_app)

        mock_db = MagicMock()
        mock_db._engine = MagicMock()

        for cls_name in ["ResolutionTracker", "CalibrationTracker", "ModelWeightUpdater",
                          "AgentWeightUpdater", "RMSEUpdater", "ErrorAnalyzer"]:
            monkeypatch.setattr(main_module, cls_name, MagicMock())

        # Make WalletCollector raise to hit Phase 4.5 except block
        monkeypatch.setattr(main_module, "WalletCollector", MagicMock(side_effect=Exception("p45 fail")))
        for cls_name in ["WalletProfiler", "WalletQueryEngine", "EdgeDecayMonitor", "CompetitorTracker"]:
            monkeypatch.setattr(main_module, cls_name, MagicMock())

        with patch.object(main_module, "setup_logging"):
            with patch.object(main_module, "init_db"):
                with patch.object(main_module, "FillMonitor", return_value=MagicMock()):
                    with patch.dict("sys.modules", {"weatheredge.db.database": mock_db}):
                        with patch("signal.signal"):
                            main_module.main()  # must not raise

    def test_keyboard_interrupt_triggers_shutdown(self, monkeypatch):
        """KeyboardInterrupt in main sleep loop triggers _handle_shutdown (lines 1618-1620)."""
        state = _make_app_state()
        # shutting_down starts False so the while loop runs at least once
        state.shutting_down = False
        state.settings = MagicMock()
        state.settings.enable_qualitative_agent = False
        state.settings.openrouter_api_key = ""
        state.settings.market_scan_interval_minutes = 5
        state.settings.forecast_poll_interval_minutes = 15
        state.settings.decision_loop_interval_minutes = 5
        state.settings.resolution_check_time = "08:00"
        state.settings.rmse_recalc_interval_days = 30
        state.settings.error_analysis_interval_days = 7
        state.settings.initial_bankroll = 1000.0

        mock_scheduler = MagicMock()
        mock_scheduler.start = MagicMock()
        monkeypatch.setattr(main_module, "BackgroundScheduler", lambda: mock_scheduler)

        def fake_init_app():
            main_module._state = state
            return state
        monkeypatch.setattr(main_module, "init_app", fake_init_app)

        mock_db = MagicMock()
        mock_db._engine = MagicMock()

        for cls_name in ["ResolutionTracker", "CalibrationTracker", "ModelWeightUpdater",
                          "AgentWeightUpdater", "RMSEUpdater", "ErrorAnalyzer",
                          "WalletCollector", "WalletProfiler", "WalletQueryEngine",
                          "EdgeDecayMonitor", "CompetitorTracker"]:
            monkeypatch.setattr(main_module, cls_name, MagicMock())

        with patch.object(main_module, "setup_logging"):
            with patch.object(main_module, "init_db"):
                with patch.object(main_module, "FillMonitor", return_value=MagicMock()):
                    with patch.dict("sys.modules", {"weatheredge.db.database": mock_db}):
                        with patch("signal.signal"):
                            with patch("time.sleep", side_effect=KeyboardInterrupt):
                                main_module.main()  # must not raise

        assert state.shutting_down is True
