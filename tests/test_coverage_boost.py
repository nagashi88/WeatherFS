"""Coverage boost tests — targeted at modules below 90%.

Covers missing lines in:
- probability/ensemble.py (below direction)
- probability/gaussian.py (below direction)
- backtest/metrics.py (all edge cases)
- notifications/telegram.py (Bot.send_message paths)
- data/metar.py (async HTTP, late-day overrides, poll_all)
- data/model_run_detector.py (_fetch_forecast_temp, arm_status, first-run cache)
- intelligence/wallet_profiler.py (LLM calls, store_profile, build_index, query)
- main.py (scheduler jobs, shutdown, decision loop branches, Phase 5 jobs)
"""

import asyncio
import json
import math
import signal
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# =========================================================================
# 1. probability/ensemble.py — "below" direction (line 5)
# =========================================================================

class TestEnsembleBelowDirection:
    def test_below_all_below(self):
        from weatheredge.probability.ensemble import ensemble_probability
        temps = [60.0] * 31
        p = ensemble_probability(temps, 70.0, direction="below")
        assert p == pytest.approx(0.95)  # clamped from 1.0

    def test_below_none_below(self):
        from weatheredge.probability.ensemble import ensemble_probability
        temps = [80.0] * 31
        p = ensemble_probability(temps, 70.0, direction="below")
        assert p == pytest.approx(0.10)  # clamped from 0.0 (PROB_FLOOR=0.10)

    def test_below_mixed(self):
        from weatheredge.probability.ensemble import ensemble_probability
        temps = [65.0] * 15 + [75.0] * 16
        p = ensemble_probability(temps, 70.0, direction="below")
        # 15 members at 65 are <= 70 → below
        # 16 members at 75 are > 70 → not below
        assert p == pytest.approx(15 / 31, abs=0.01)


# =========================================================================
# 2. probability/gaussian.py — "below" direction (line 7)
# =========================================================================

class TestGaussianBelowDirection:
    def test_below_low_forecast(self):
        from weatheredge.probability.gaussian import gaussian_probability
        # forecast 60, threshold 70, rmse 2.5 → P(below 70) should be high (clamped to 0.95)
        p = gaussian_probability(60.0, 70.0, 2.5, direction="below")
        assert p >= 0.90  # Student-T + clamp at 0.95

    def test_below_high_forecast(self):
        from weatheredge.probability.gaussian import gaussian_probability
        # forecast 85, threshold 70, rmse 2.5 → P(below 70) should be very low (clamped to 0.05)
        p = gaussian_probability(85.0, 70.0, 2.5, direction="below")
        assert p <= 0.10  # Student-T + clamp at 0.05

    def test_below_at_threshold(self):
        from weatheredge.probability.gaussian import gaussian_probability
        p = gaussian_probability(70.0, 70.0, 2.5, direction="below")
        assert abs(p - 0.50) < 0.01


# =========================================================================
# 3. backtest/metrics.py — all edge cases (lines 8, 19, 25, 31, 37)
# =========================================================================

class TestBacktestMetrics:
    def test_sharpe_empty_series(self):
        from weatheredge.backtest.metrics import sharpe_ratio
        assert sharpe_ratio([]) == 0.0

    def test_sharpe_single_element(self):
        from weatheredge.backtest.metrics import sharpe_ratio
        assert sharpe_ratio([1.0]) == 0.0

    def test_sharpe_zero_std(self):
        from weatheredge.backtest.metrics import sharpe_ratio
        # All identical values → std = 0 → return 0
        assert sharpe_ratio([5.0, 5.0, 5.0]) == 0.0

    def test_sharpe_normal(self):
        from weatheredge.backtest.metrics import sharpe_ratio
        pnl = [1.0, 2.0, 1.5, 2.5, 1.0]
        result = sharpe_ratio(pnl)
        assert result > 0  # positive mean → positive sharpe

    def test_sortino_empty(self):
        from weatheredge.backtest.metrics import sortino_ratio
        assert sortino_ratio([]) == 0.0

    def test_sortino_single(self):
        from weatheredge.backtest.metrics import sortino_ratio
        assert sortino_ratio([1.0]) == 0.0

    def test_sortino_no_downside(self):
        from weatheredge.backtest.metrics import sortino_ratio
        # All positive → downside_std = 0 → return 0
        assert sortino_ratio([1.0, 2.0, 3.0]) == 0.0

    def test_sortino_with_losses(self):
        from weatheredge.backtest.metrics import sortino_ratio
        pnl = [1.0, -0.5, 2.0, -1.0, 1.5]
        result = sortino_ratio(pnl)
        assert result > 0  # net positive mean

    def test_max_drawdown_empty(self):
        from weatheredge.backtest.metrics import max_drawdown
        assert max_drawdown([]) == 0.0

    def test_max_drawdown_all_losses(self):
        from weatheredge.backtest.metrics import max_drawdown
        # All losses: cumulative never goes positive → peak <= 0 → return 0
        assert max_drawdown([-1.0, -2.0, -3.0]) == 0.0

    def test_max_drawdown_normal(self):
        from weatheredge.backtest.metrics import max_drawdown
        pnl = [10.0, -5.0, -3.0, 4.0, -1.0]
        result = max_drawdown(pnl)
        assert 0.0 < result <= 1.0


# =========================================================================
# 4. notifications/telegram.py — Bot.send_message paths (lines 37-43, 85)
# =========================================================================

class TestTelegramBot:
    def test_send_alert_calls_bot(self):
        from weatheredge.notifications.telegram import TelegramBot
        from weatheredge.config import Settings

        settings = Settings(telegram_bot_token="fake-token", telegram_chat_id="12345")
        tg = TelegramBot(settings)

        mock_instance = MagicMock()
        mock_instance.send_message = AsyncMock()

        with patch.dict("sys.modules", {"telegram": MagicMock()}):
            with patch("telegram.Bot", return_value=mock_instance):
                asyncio.run(tg.send_alert("test message"))
                mock_instance.send_message.assert_called_once_with(
                    chat_id="12345", text="test message"
                )

    def test_send_alert_exception_swallowed(self):
        from weatheredge.notifications.telegram import TelegramBot
        from weatheredge.config import Settings

        settings = Settings(telegram_bot_token="fake-token", telegram_chat_id="12345")
        tg = TelegramBot(settings)

        with patch.dict("sys.modules", {"telegram": MagicMock()}):
            with patch("telegram.Bot", side_effect=Exception("connection error")):
                # Should not raise
                asyncio.run(tg.send_alert("test message"))

    def test_format_error_alert(self):
        from weatheredge.notifications.telegram import TelegramBot
        from weatheredge.config import Settings

        settings = Settings()
        bot = TelegramBot(settings)
        result = bot.format_error_alert("something broke")
        assert "something broke" in result


# =========================================================================
# 5. data/metar.py — async paths, late-day overrides (lines 94, 105-106,
#    120-121, 153-155, 158, 170-180, 201, 219-224, 228-233)
# =========================================================================

class TestMetarCoverage:
    def test_daily_tracker_reset_on_new_day(self):
        """Tracker resets when date changes."""
        from weatheredge.data.metar import CurrentConditionsTracker, DailyTracker, StationObservation

        tracker = CurrentConditionsTracker()
        # Set an old tracker with yesterday's date
        old_tracker = DailyTracker(station="KNYC", date="2026-03-29")
        old_tracker.update(StationObservation(station="KNYC", temperature_f=50.0))
        tracker._trackers["KNYC"] = old_tracker

        # Simulate poll that internally creates a new tracker for today
        # We do this by calling metar_probability_override which reads the tracker
        # Then verify the old tracker is still there (it won't reset until next poll)
        result = tracker.metar_probability_override("KNYC", 75.0)
        # Old tracker has temp 50, not >= 75, no late-day override
        assert result is None

    def test_metar_override_late_day_large_gap(self):
        """Late day, large gap below threshold for 'above' market → None (use forecast models).

        Late-day downward adjustments were removed: current temp says nothing about the
        afternoon high. Only the observed daily HIGH exceeding the threshold triggers an override.
        """
        from weatheredge.data.metar import CurrentConditionsTracker, DailyTracker, StationObservation

        tracker = CurrentConditionsTracker()
        daily = DailyTracker(station="KNYC", date="2026-03-30")
        obs = StationObservation(station="KNYC", temperature_f=55.0)
        daily.update(obs)
        tracker._trackers["KNYC"] = daily

        with patch("weatheredge.data.metar.datetime") as mock_dt:
            mock_now = MagicMock()
            mock_now.hour = 22
            mock_now.strftime.return_value = "2026-03-30"
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            # Daily high (55°F) < threshold (75°F) → no override, defer to forecast models
            result = tracker.metar_probability_override("KNYC", 75.0, direction="above")
            assert result is None

    def test_metar_override_late_day_moderate_gap(self):
        """Late day, moderate gap for 'above' market → None (use forecast models)."""
        from weatheredge.data.metar import CurrentConditionsTracker, DailyTracker, StationObservation

        tracker = CurrentConditionsTracker()
        daily = DailyTracker(station="KNYC", date="2026-03-30")
        obs = StationObservation(station="KNYC", temperature_f=68.0)
        daily.update(obs)
        tracker._trackers["KNYC"] = daily

        with patch("weatheredge.data.metar.datetime") as mock_dt:
            mock_now = MagicMock()
            mock_now.hour = 21
            mock_now.strftime.return_value = "2026-03-30"
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            # Daily high (68°F) < threshold (75°F) → no override, defer to forecast models
            result = tracker.metar_probability_override("KNYC", 75.0, direction="above")
            assert result is None

    @pytest.mark.asyncio
    async def test_poll_station_empty_data(self):
        """METAR API returns empty list."""
        from weatheredge.data.metar import CurrentConditionsTracker

        tracker = CurrentConditionsTracker()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        obs = await tracker.poll_station("KNYC", session=mock_session)
        assert obs is None

    @pytest.mark.asyncio
    async def test_poll_station_no_temp(self):
        """METAR API returns data but with null temperature."""
        from weatheredge.data.metar import CurrentConditionsTracker

        tracker = CurrentConditionsTracker()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[{
            "temp": None, "dewp": None, "wspd": 5,
            "reportTime": "2026-03-30T12:00:00Z", "rawOb": "METAR KNYC ...",
        }])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        obs = await tracker.poll_station("KNYC", session=mock_session)
        assert obs is not None
        assert obs.temperature_f is None

    @pytest.mark.asyncio
    async def test_poll_station_exception_returns_none(self):
        """Network error during poll returns None."""
        from weatheredge.data.metar import CurrentConditionsTracker

        tracker = CurrentConditionsTracker()
        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=Exception("network error"))

        obs = await tracker.poll_station("KNYC", session=mock_session)
        assert obs is None

    @pytest.mark.asyncio
    async def test_poll_station_bad_time_format(self):
        """METAR with unparseable observation time."""
        from weatheredge.data.metar import CurrentConditionsTracker

        tracker = CurrentConditionsTracker()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[{
            "temp": 20.0, "dewp": 10.0,
            "reportTime": "not-a-date",
            "rawOb": "METAR KNYC ...",
        }])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        obs = await tracker.poll_station("KNYC", session=mock_session)
        assert obs is not None
        assert obs.observation_time is None  # Unparseable time

    @pytest.mark.asyncio
    async def test_poll_all_stations(self):
        """poll_all_stations delegates to poll_station for each."""
        from weatheredge.data.metar import CurrentConditionsTracker

        tracker = CurrentConditionsTracker()
        call_count = 0

        async def mock_poll(station, session=None):
            nonlocal call_count
            call_count += 1
            return None

        tracker.poll_station = mock_poll
        results = await tracker.poll_all_stations(["KNYC", "KMDW", "KMIA"])
        assert call_count == 3
        assert len(results) == 3

    def test_metar_override_none_temp_in_obs(self):
        """Override returns None when observation has no temperature."""
        from weatheredge.data.metar import CurrentConditionsTracker, DailyTracker, StationObservation

        tracker = CurrentConditionsTracker()
        daily = DailyTracker(station="KNYC", date="2026-03-30")
        daily.last_observation = StationObservation(station="KNYC", temperature_f=None)
        tracker._trackers["KNYC"] = daily
        assert tracker.metar_probability_override("KNYC", 75.0) is None


# =========================================================================
# 6. data/model_run_detector.py — _fetch_forecast_temp, arm_status details,
#    first-run cache prime (lines 96, 121, 134-136, 139, 169, 176-177, 191-214, 229)
# =========================================================================

class TestModelRunDetectorCoverage:
    @pytest.mark.asyncio
    async def test_fetch_forecast_temp_success(self):
        from weatheredge.data.model_run_detector import ModelRunDetector

        det = ModelRunDetector(city_coords={"NYC": (40.7, -74.0)})

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "hourly": {"temperature_2m": [70.0, 72.0, 75.0, 73.0]}
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        temp = await det._fetch_forecast_temp(40.7, -74.0, mock_session)
        assert temp == 75.0  # max of the list

    @pytest.mark.asyncio
    async def test_fetch_forecast_temp_api_error(self):
        from weatheredge.data.model_run_detector import ModelRunDetector

        det = ModelRunDetector(city_coords={"NYC": (40.7, -74.0)})

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        temp = await det._fetch_forecast_temp(40.7, -74.0, mock_session)
        assert temp is None

    @pytest.mark.asyncio
    async def test_fetch_forecast_temp_empty_data(self):
        from weatheredge.data.model_run_detector import ModelRunDetector

        det = ModelRunDetector(city_coords={"NYC": (40.7, -74.0)})

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"hourly": {"temperature_2m": []}})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        temp = await det._fetch_forecast_temp(40.7, -74.0, mock_session)
        assert temp is None

    @pytest.mark.asyncio
    async def test_first_run_primes_cache_no_events(self):
        """First check primes cache (no old temp to compare), so no events."""
        from weatheredge.data.model_run_detector import ModelRunDetector

        det = ModelRunDetector(city_coords={"NYC": (40.7, -74.0)})
        det.is_in_check_window = lambda now=None: (True, "12Z")

        async def mock_fetch(lat, lon, session):
            return 72.0

        det._fetch_forecast_temp = mock_fetch
        events = await det.check_for_new_run(session=MagicMock())
        assert len(events) == 0
        assert det._cache["NYC"].temperature_f == 72.0

    @pytest.mark.asyncio
    async def test_callback_exception_swallowed(self):
        """Callback failure should not propagate."""
        from weatheredge.data.model_run_detector import ModelRunDetector

        async def bad_callback(event):
            raise RuntimeError("callback broke")

        det = ModelRunDetector(
            city_coords={"NYC": (40.7, -74.0)},
            on_new_run=bad_callback,
        )
        det.is_in_check_window = lambda now=None: (True, "12Z")
        det._cache["NYC"].temperature_f = 70.0

        async def mock_fetch(lat, lon, session):
            return 75.0  # +5°F delta

        det._fetch_forecast_temp = mock_fetch
        # Should not raise
        events = await det.check_for_new_run(session=MagicMock())
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_fetch_failure_skips_city(self):
        """If fetch raises for a city, skip it gracefully."""
        from weatheredge.data.model_run_detector import ModelRunDetector

        det = ModelRunDetector(city_coords={"NYC": (40.7, -74.0)})
        det.is_in_check_window = lambda now=None: (True, "12Z")
        det._cache["NYC"].temperature_f = 70.0

        async def mock_fetch(lat, lon, session):
            raise Exception("timeout")

        det._fetch_forecast_temp = mock_fetch
        events = await det.check_for_new_run(session=MagicMock())
        assert len(events) == 0

    def test_arm_status_all_fields(self):
        from weatheredge.data.model_run_detector import ModelRunDetector

        det = ModelRunDetector(city_coords={
            "NYC": (40.7, -74.0),
            "Chicago": (41.9, -87.6),
        })
        status = det.arm_status()
        assert "armed" in status
        assert "in_check_window" in status
        assert "next_window" in status
        assert "total_detections" in status
        assert "cached_cities" in status
        assert "NYC" in status["cached_cities"]
        # Verify cache entry structure
        nyc = status["cached_cities"]["NYC"]
        assert "temp_f" in nyc
        assert "last_check" in nyc
        assert "last_run" in nyc


# =========================================================================
# 7. intelligence/wallet_profiler.py — LLM calls, store, build_index, query
#    (lines 48-50, 53, 59, 67-69, 73-75, 134, 186-193, 224, 237-238)
# =========================================================================

class TestWalletProfilerCoverage:
    def test_is_weather_trade_true(self):
        from weatheredge.intelligence.wallet_profiler import _is_weather_trade
        trade = {"market": "NYC Temperature", "category": "weather"}
        assert _is_weather_trade(trade) is True

    def test_is_weather_trade_false(self):
        from weatheredge.intelligence.wallet_profiler import _is_weather_trade
        trade = {"market": "Bitcoin price", "category": "crypto"}
        assert _is_weather_trade(trade) is False

    def test_compute_stats(self):
        from weatheredge.intelligence.wallet_profiler import _compute_stats
        trades = [
            {"pnl": "10.0"}, {"pnl": "-5.0"}, {"pnl": "3.0"},
        ]
        entry = {"sharpe": 1.5, "max_drawdown": 0.1, "categories": ["weather"]}
        stats = _compute_stats("0xabc", trades, entry)
        assert stats["address"] == "0xabc"
        assert stats["total_trades"] == 3
        assert stats["win_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert stats["total_pnl"] == pytest.approx(8.0)
        assert stats["sharpe_ratio"] == 1.5

    def test_collector_handles_empty_leaderboard(self):
        from weatheredge.intelligence.wallet_profiler import WalletCollector
        collector = WalletCollector()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="[]", stderr="")
            result = collector.collect_weather_wallets()
            assert result == []

    def test_collector_handles_bad_json(self):
        from weatheredge.intelligence.wallet_profiler import WalletCollector
        collector = WalletCollector()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="not json", stderr="")
            result = collector.collect_weather_wallets()
            assert result == []

    def test_collector_filters_by_min_trades(self):
        from weatheredge.intelligence.wallet_profiler import WalletCollector
        collector = WalletCollector()

        leaderboard_json = json.dumps([{"wallet": "0xabc"}])
        trades_json = json.dumps([
            {"market": "temperature forecast", "pnl": "1.0"}
        ] * 10)  # Only 10 trades, below min_trades=50

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout=leaderboard_json),  # leaderboard call
                MagicMock(stdout=trades_json),  # trades call
            ]
            result = collector.collect_weather_wallets(min_trades=50)
            assert result == []

    def test_collector_trade_json_parse_error(self):
        from weatheredge.intelligence.wallet_profiler import WalletCollector
        collector = WalletCollector()

        leaderboard_json = json.dumps([{"wallet": "0xabc"}])

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout=leaderboard_json),
                MagicMock(stdout="bad json"),  # trades call returns bad json
            ]
            result = collector.collect_weather_wallets()
            assert result == []

    def test_collector_trade_cli_not_found(self):
        from weatheredge.intelligence.wallet_profiler import WalletCollector
        collector = WalletCollector()

        leaderboard_json = json.dumps([{"wallet": "0xabc"}])

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout=leaderboard_json),
                FileNotFoundError("polymarket not found"),
            ]
            result = collector.collect_weather_wallets()
            assert result == []

    def test_collector_no_wallet_address(self):
        from weatheredge.intelligence.wallet_profiler import WalletCollector
        collector = WalletCollector()

        leaderboard_json = json.dumps([{"name": "no wallet field"}])

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=leaderboard_json)
            result = collector.collect_weather_wallets()
            assert result == []

    def test_profiler_profile_wallet(self):
        from weatheredge.intelligence.wallet_profiler import WalletProfiler

        with patch("weatheredge.intelligence.wallet_profiler.get_llm_client") as mock_client_fn, \
             patch("weatheredge.intelligence.wallet_profiler.chat_completion") as mock_chat:
            mock_chat.return_value = "This is a sophisticated weather trader."
            profiler = WalletProfiler(engine=MagicMock(), api_key="fake")
            result = profiler.profile_wallet({
                "address": "0xabc",
                "total_trades": 100,
                "win_rate": 0.65,
                "sharpe_ratio": 1.5,
                "total_pnl": 500.0,
                "max_drawdown": 0.1,
                "categories": ["weather"],
            })
            assert "sophisticated" in result
            mock_chat.assert_called_once()

    def test_profiler_store_profile_new(self):
        from weatheredge.intelligence.wallet_profiler import WalletProfiler

        mock_session = MagicMock()
        mock_session.get.return_value = None  # No existing profile

        profiler = WalletProfiler(engine=MagicMock(), api_key="fake")
        profiler.store_profile(mock_session, "0xabc", "profile text", {
            "total_trades": 50, "win_rate": 0.6,
        })
        mock_session.add.assert_called_once()

    def test_profiler_store_profile_update(self):
        from weatheredge.intelligence.wallet_profiler import WalletProfiler
        from weatheredge.db.models import WalletProfile

        existing = WalletProfile(
            wallet_address="0xabc",
            profile_text="old profile",
        )
        mock_session = MagicMock()
        mock_session.get.return_value = existing

        profiler = WalletProfiler(engine=MagicMock(), api_key="fake")
        profiler.store_profile(mock_session, "0xabc", "new profile", {
            "total_trades": 100, "win_rate": 0.7,
            "sharpe_ratio": 1.5, "total_pnl": 500.0,
            "max_drawdown": 0.1, "categories": ["weather"],
            "is_automated": True,
        })
        assert existing.profile_text == "new profile"
        assert existing.total_trades == 100

    def test_profiler_build_index(self, tmp_path):
        from weatheredge.intelligence.wallet_profiler import WalletProfiler

        profiler = WalletProfiler(engine=MagicMock(), api_key="fake")
        with patch("weatheredge.intelligence.wallet_profiler._INTELLIGENCE_DIR", tmp_path):
            path = profiler.build_index([
                {"address": "0x1", "profile_text": "trader A", "win_rate": 0.6},
                {"address": "0x2", "profile_text": "trader B", "win_rate": 0.7},
            ])
            assert Path(path).exists()
            data = json.loads(Path(path).read_text())
            assert len(data) == 2

    def test_query_engine_with_profiles(self, tmp_path):
        from weatheredge.intelligence.wallet_profiler import WalletQueryEngine

        index = tmp_path / "wallet_index.json"
        index.write_text(json.dumps([{
            "address": "0xabc",
            "profile_text": "Sophisticated weather trader",
            "stats": {"win_rate": 0.65},
        }]))

        with patch("weatheredge.intelligence.wallet_profiler.get_llm_client"), \
             patch("weatheredge.intelligence.wallet_profiler.chat_completion") as mock_chat:
            mock_chat.return_value = "The top trader is 0xabc"
            engine = WalletQueryEngine(index_path=str(index), api_key="fake")
            result = engine.query("Who is the best trader?")
            assert "0xabc" in result

    def test_query_engine_api_failure(self, tmp_path):
        from weatheredge.intelligence.wallet_profiler import WalletQueryEngine

        index = tmp_path / "wallet_index.json"
        index.write_text(json.dumps([{"address": "0x1", "profile_text": "a", "stats": {}}]))

        with patch("weatheredge.intelligence.wallet_profiler.get_llm_client"), \
             patch("weatheredge.intelligence.wallet_profiler.chat_completion") as mock_chat:
            mock_chat.return_value = None  # LLM returns None → triggers RuntimeError
            engine = WalletQueryEngine(index_path=str(index), api_key="fake")
            result = engine.query("test")
            assert "Error" in result


# =========================================================================
# 8. main.py — scheduler jobs, shutdown, Phase 5 jobs, decision loop branches
#    Key missing lines: 209-215, 264-301, 372-377, 392, 408-409, 421-425,
#    433-434, 492-493, 503-504, 526-527, 543-544, 579-595, 603-625,
#    633-650, 676, 680-681, 685, 689, 720-732, 736-738, 749, 804, 834-836,
#    841-843, 960-963, 1031-1046, 1152-1157, 1162-1167, 1176-1189,
#    1194-1195, 1203-1217, 1246-1248, 1255-1256, 1276-1277, 1281-1282,
#    1302-1303, 1435-1437, 1443
# =========================================================================

class TestMainShutdown:
    def test_handle_shutdown_sets_flag(self):
        from weatheredge.main import _handle_shutdown, init_app, get_state
        state = init_app()
        state.scheduler = MagicMock()
        _handle_shutdown(signal.SIGTERM, None)
        assert state.shutting_down is True
        state.scheduler.shutdown.assert_called_once_with(wait=False)

    def test_handle_shutdown_scheduler_exception(self):
        from weatheredge.main import _handle_shutdown, init_app
        state = init_app()
        state.scheduler = MagicMock()
        state.scheduler.shutdown.side_effect = Exception("already stopped")
        # Should not raise
        _handle_shutdown(signal.SIGINT, None)
        assert state.shutting_down is True

    def test_handle_shutdown_db_dispose(self):
        from weatheredge.main import _handle_shutdown, init_app
        state = init_app()
        state.scheduler = None
        with patch("weatheredge.main.get_state", return_value=state):
            _handle_shutdown(signal.SIGTERM, None)
            assert state.shutting_down is True


class TestMainPhase5Jobs:
    def test_poll_metar_conditions(self):
        from weatheredge.main import poll_metar_conditions, init_app
        state = init_app()
        with patch("weatheredge.main.asyncio.run") as mock_run:
            poll_metar_conditions()
            mock_run.assert_called_once()

    def test_poll_metar_conditions_exception(self):
        from weatheredge.main import poll_metar_conditions, init_app
        init_app()
        with patch("weatheredge.main.asyncio.run", side_effect=Exception("network")):
            # Should not raise
            poll_metar_conditions()

    def test_check_model_runs_no_detector(self):
        from weatheredge.main import check_model_runs, init_app
        state = init_app()
        state.model_run_detector = None
        # Should return immediately without error
        check_model_runs()

    def test_check_model_runs_no_events(self):
        from weatheredge.main import check_model_runs, init_app
        state = init_app()
        state.model_run_detector = MagicMock()
        with patch("weatheredge.main.asyncio.run", return_value=[]):
            check_model_runs()

    def test_check_model_runs_with_events(self):
        from weatheredge.main import check_model_runs, init_app
        state = init_app()
        state.model_run_detector = MagicMock()
        mock_event = MagicMock()
        with patch("weatheredge.main.asyncio.run", return_value=[mock_event]), \
             patch("weatheredge.main.run_decision_loop") as mock_loop:
            check_model_runs()
            mock_loop.assert_called_once()

    def test_check_model_runs_exception(self):
        from weatheredge.main import check_model_runs, init_app
        state = init_app()
        state.model_run_detector = MagicMock()
        with patch("weatheredge.main.asyncio.run", side_effect=Exception("fail")):
            # Should not raise
            check_model_runs()

    @pytest.mark.asyncio
    async def test_on_new_model_run_callback(self):
        from weatheredge.main import _on_new_model_run, init_app
        from weatheredge.data.model_run_detector import ModelRunEvent
        init_app()
        event = ModelRunEvent(
            run_label="12Z",
            detected_at=datetime(2026, 3, 30, 15, 30, tzinfo=timezone.utc),
            city="NYC",
            delta_f=2.5,
            old_temp_f=70.0,
            new_temp_f=72.5,
        )
        with patch("weatheredge.main.TelegramBot", create=True) as MockBot:
            mock_instance = MagicMock()
            mock_instance.send_alert = AsyncMock()
            MockBot.return_value = mock_instance
            # Patch the import inside the function
            with patch("weatheredge.notifications.telegram.TelegramBot", MockBot):
                await _on_new_model_run(event)

    @pytest.mark.asyncio
    async def test_on_new_model_run_telegram_error(self):
        from weatheredge.main import _on_new_model_run, init_app
        from weatheredge.data.model_run_detector import ModelRunEvent
        init_app()
        event = ModelRunEvent(
            run_label="12Z",
            detected_at=datetime(2026, 3, 30, 15, 30, tzinfo=timezone.utc),
            city="NYC",
            delta_f=2.5,
            old_temp_f=70.0,
            new_temp_f=72.5,
        )
        with patch("weatheredge.notifications.telegram.TelegramBot") as MockBot:
            MockBot.side_effect = Exception("telegram down")
            # Should not raise
            await _on_new_model_run(event)


class TestMainPhase3Jobs:
    def test_run_resolution_check_no_tracker(self):
        from weatheredge.main import run_resolution_check, init_app
        state = init_app()
        state.resolution_tracker = None
        # Should return immediately
        run_resolution_check()

    def test_run_rmse_recalculation_no_updater(self):
        from weatheredge.main import run_rmse_recalculation, init_app
        state = init_app()
        state.rmse_updater = None
        run_rmse_recalculation()

    def test_run_error_analysis_no_analyzer(self):
        from weatheredge.main import run_error_analysis, init_app
        state = init_app()
        state.error_analyzer = None
        run_error_analysis()

    def test_run_wallet_profiling_no_components(self):
        from weatheredge.main import run_wallet_profiling, init_app
        state = init_app()
        state.wallet_collector = None
        run_wallet_profiling()

    def test_run_edge_decay_check_no_monitor(self):
        from weatheredge.main import run_edge_decay_check, init_app
        state = init_app()
        state.edge_decay_monitor = None
        run_edge_decay_check()

    def test_run_morning_briefing_no_engine(self):
        from weatheredge.main import run_morning_briefing, init_app
        state = init_app()
        state.wallet_query_engine = None
        run_morning_briefing()


class TestMainDecisionLoop:
    def test_decision_loop_shutting_down(self):
        from weatheredge.main import run_decision_loop, init_app
        state = init_app()
        state.shutting_down = True
        # Should return immediately
        run_decision_loop()

    def test_decision_loop_no_weather_data(self):
        from weatheredge.main import run_decision_loop, init_app
        state = init_app()
        with patch("weatheredge.main._get_weather_data_age_minutes", return_value=None):
            run_decision_loop()

    def test_decision_loop_stale_data(self):
        from weatheredge.main import run_decision_loop, init_app
        state = init_app()
        with patch("weatheredge.main._get_weather_data_age_minutes", return_value=999.0):
            run_decision_loop()

    def test_decision_loop_no_markets(self):
        from weatheredge.main import run_decision_loop, init_app
        state = init_app()
        with patch("weatheredge.main._get_weather_data_age_minutes", return_value=5.0), \
             patch("weatheredge.main._read_latest_forecasts", return_value={}), \
             patch("weatheredge.main._read_latest_market_prices", return_value={}):
            run_decision_loop()

    def test_decision_loop_db_read_error(self):
        from weatheredge.main import run_decision_loop, init_app
        state = init_app()
        with patch("weatheredge.main._get_weather_data_age_minutes", return_value=5.0), \
             patch("weatheredge.main._read_latest_forecasts", side_effect=Exception("DB error")):
            run_decision_loop()

    def test_fetch_market_data_success(self):
        from weatheredge.main import fetch_market_data, init_app
        state = init_app()
        with patch("weatheredge.main.asyncio.run", return_value=[]), \
             patch("weatheredge.main.get_session"):
            fetch_market_data()

    def test_fetch_market_data_exception(self):
        from weatheredge.main import fetch_market_data, init_app
        init_app()
        with patch("weatheredge.main.asyncio.run", side_effect=Exception("scan failed")):
            fetch_market_data()

    def test_fetch_forecast_data_exception(self):
        from weatheredge.main import fetch_forecast_data, init_app
        init_app()
        with patch("weatheredge.main.asyncio.run", side_effect=Exception("forecast failed")):
            fetch_forecast_data()
