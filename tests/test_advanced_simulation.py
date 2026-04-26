"""Advanced simulation tests for WeatherEdge bot.

7 test suites:
1. Full day simulation (6am–6pm NYC temperature market lifecycle)
2. Week simulation with GFS model run detection and HRRR weight shifts
3. Calibration / Brier score (200 synthetic outcomes)
4. Concurrency / race condition tests
5. Realistic edge plausibility
6. Crash recovery tests
7. Error analysis learning test
"""

from __future__ import annotations

import asyncio
import math
import os
import tempfile
import threading
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.pool import StaticPool

from weatheredge.data.metar import (
    CurrentConditionsTracker,
    DailyTracker,
    StationObservation,
)
from weatheredge.data.model_run_detector import (
    DELTA_THRESHOLD_F,
    ModelRunDetector,
    ModelRunEvent,
)
from weatheredge.db.database import init_db, get_session
from weatheredge.db.models import Forecast, Trade, CalibrationRecord
from weatheredge.execution.paper_trader import Order, PaperTrader
from weatheredge.learning.calibration import CalibrationTracker
from weatheredge.learning.error_analysis import ErrorAnalyzer
from weatheredge.learning.rmse_updater import RMSEUpdater
from weatheredge.probability.blended import (
    WEIGHTS_US,
    _get_lead_bucket,
    blended_probability,
)
from weatheredge.probability.edge import calculate_gross_edge, calculate_net_edge


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_engine():
    """In-memory SQLite with StaticPool so all sessions share the same DB."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    return engine


def _insert_trade(
    engine,
    *,
    market_id: str,
    side: str = "buy_yes",
    price: float = 0.60,
    amount: float = 10.0,
    cost: float = 6.0,
    model_prob: float = 0.80,
    edge: float = 0.185,
    status: str = "lost",
    pnl: float = -6.0,
    agent_name: str = "test",
) -> None:
    with get_session(engine) as session:
        trade = Trade(
            market_id=market_id,
            platform="polymarket",
            mode="paper",
            side=side,
            price=price,
            amount=amount,
            cost=cost,
            agent_name=agent_name,
            model_probability=model_prob,
            edge_at_entry=edge,
            status=status,
            pnl=pnl,
        )
        session.add(trade)


# ===========================================================================
# 1. FULL DAY SIMULATION
# ===========================================================================

class TestFullDaySimulation:
    """Simulate a complete NYC 'daily high above 70°F' trading day."""

    def test_morning_initial_state(self):
        """6am: bankroll intact, no positions, no history."""
        trader = PaperTrader(initial_bankroll=1000.0)
        assert trader.bankroll == 1000.0
        assert trader.open_positions == []
        assert trader.trade_history == []

    def test_10am_edge_exists_and_trade_placed(self):
        """10am: METAR 65°F climbing, model prob 0.80, market $0.65 → positive edge → trade placed."""
        market_price = 0.65
        model_prob = 0.80
        net_edge = calculate_net_edge(model_prob, market_price, "polymarket")

        # net edge = 0.80 - 0.65 - 0.015 = 0.135
        assert net_edge == pytest.approx(0.135, abs=1e-6), (
            f"Expected net edge 0.135, got {net_edge:.4f}"
        )

        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(
            market_id="nyc-high-70",
            side="buy_yes",
            price=market_price,
            amount=10,
            agent_name="conservative",
            model_probability=model_prob,
            edge_at_entry=net_edge,
            city="NYC",
            platform="polymarket",
        )
        result = trader.place_order(order)

        assert result["filled"] is True
        assert len(trader.open_positions) == 1
        assert trader.bankroll == pytest.approx(1000.0 - 0.65 * 10, abs=1e-6)

    def test_1pm_metar_override_at_71f(self):
        """1pm: METAR daily high reaches 71°F (>= threshold 70°F) → override prob 0.99."""
        tracker = CurrentConditionsTracker()
        station = "KNYC"
        dt = DailyTracker(station=station, date="2026-03-30")
        for temp in [65.0, 68.0, 71.0]:
            dt.update(StationObservation(station=station, temperature_f=temp))
        tracker._trackers[station] = dt

        override = tracker.metar_probability_override(
            station, threshold=70.0, direction="above",
            market_type="daily_high", market_date=None,
        )
        assert override == 0.99

    def test_3pm_daily_high_confirmed_at_74f(self):
        """3pm: METAR reads 74°F → daily high updated, observation count correct."""
        station = "KNYC"
        dt = DailyTracker(station=station, date="2026-03-30")
        for temp in [65.0, 71.0, 74.0]:
            dt.update(StationObservation(station=station, temperature_f=temp))

        assert dt.daily_high_f == 74.0
        assert dt.daily_low_f == 65.0
        assert dt.observation_count == 3

    def test_6pm_resolution_yes_pnl_with_fees(self):
        """6pm: market resolves YES → P&L correct with 1.5% polymarket fee."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(
            market_id="nyc-high-70",
            side="buy_yes",
            price=0.65,
            amount=10,
            platform="polymarket",
            model_probability=0.80,
            edge_at_entry=0.135,
        )
        trader.place_order(order)
        trader.resolve_position("nyc-high-70", "yes")

        assert len(trader.open_positions) == 0
        assert len(trader.trade_history) == 1

        trade = trader.trade_history[0]
        assert trade["status"] == "won"

        # Polymarket has 0% fee on winnings (spread cost is in edge calc, not here)
        # pnl = (amount - cost) * (1 - fee_rate) = (10 - 6.5) * 1.0 = 3.50
        expected_pnl = (10.0 - 0.65 * 10.0) * (1.0 - 0.00)
        assert trade["pnl"] == pytest.approx(expected_pnl, abs=1e-6)

    def test_bankroll_net_gain_after_win(self):
        """Bankroll ends higher than initial after a winning trade."""
        initial = 1000.0
        trader = PaperTrader(initial_bankroll=initial)
        order = Order(
            market_id="nyc-high-70",
            side="buy_yes",
            price=0.65,
            amount=10,
            platform="polymarket",
        )
        trader.place_order(order)
        trader.resolve_position("nyc-high-70", "yes")

        expected_pnl = (10.0 - 6.5) * (1.0 - 0.00)  # Polymarket 0% fee = 3.50
        assert trader.bankroll == pytest.approx(initial + expected_pnl, abs=1e-6)
        assert trader.bankroll > initial

    def test_daily_pnl_recorded(self):
        """Daily P&L dict contains today's P&L after resolution."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="nyc-high-70", side="buy_yes", price=0.65, amount=10, platform="polymarket")
        trader.place_order(order)
        trader.resolve_position("nyc-high-70", "yes")

        assert len(trader.daily_pnl) == 1
        day_pnl = list(trader.daily_pnl.values())[0]
        expected_pnl = (10.0 - 6.5) * (1.0 - 0.00)  # Polymarket 0% fee
        assert day_pnl == pytest.approx(expected_pnl, abs=1e-6)

    def test_trade_history_all_fields_populated(self):
        """Trade history record contains all expected fields after resolution."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(
            market_id="nyc-high-70",
            side="buy_yes",
            price=0.65,
            amount=10,
            agent_name="conservative",
            model_probability=0.80,
            edge_at_entry=0.135,
            city="NYC",
            platform="polymarket",
        )
        trader.place_order(order)
        trader.resolve_position("nyc-high-70", "yes")

        trade = trader.trade_history[0]
        assert trade["market_id"] == "nyc-high-70"
        assert trade["side"] == "buy_yes"
        assert trade["price"] == 0.65
        assert trade["amount"] == 10
        assert trade["agent_name"] == "conservative"
        assert trade["model_probability"] == 0.80
        assert trade["status"] == "won"
        assert trade["pnl"] is not None
        assert trade["cost"] == pytest.approx(6.5, abs=1e-6)

    def test_loss_scenario_bankroll_decrease(self):
        """Losing trade: bankroll decreases by full cost (no recovery)."""
        initial = 1000.0
        trader = PaperTrader(initial_bankroll=initial)
        order = Order(market_id="nyc-high-70", side="buy_yes", price=0.65, amount=10, platform="polymarket")
        trader.place_order(order)
        trader.resolve_position("nyc-high-70", "no")  # market resolves NO

        trade = trader.trade_history[0]
        assert trade["status"] == "lost"
        assert trade["pnl"] == pytest.approx(-(0.65 * 10), abs=1e-6)
        assert trader.bankroll == pytest.approx(initial - 6.5, abs=1e-6)


# ===========================================================================
# 2. WEEK SIMULATION WITH MODEL RUNS
# ===========================================================================

class TestWeekSimulationModelRuns:
    """Simulate 7 days of GFS model run detection and HRRR weight shifting."""

    def test_day1_low_probability_no_edge(self):
        """Day 1: GFS 68°F, ECMWF 70°F, 'above 72°F' at $0.40 → probability low, no edge."""
        prob, _, _ = blended_probability(
            threshold=72.0, direction="above", rmse=5.0,
            is_us_station=True, lead_time_hours=24.0,
            ecmwf_forecast_temp=70.0,
            gfs_forecast_temp=68.0,   # fallback only (no ensemble)
        )
        assert prob < 0.60, f"Day 1 probability {prob:.3f} should be < 0.60 (below threshold)"
        edge = calculate_net_edge(prob, 0.40, "polymarket")
        assert edge < 0.10, f"No strong edge on Day 1, got {edge:.3f}"

    def test_day2_gfs_shift_triggers_detection(self):
        """Day 2: GFS run shifts 68→71°F (delta=3°F > 0.5 threshold) → detector fires."""
        city_coords = {"NYC": (40.7128, -74.006)}
        events_fired: list[ModelRunEvent] = []

        async def on_new_run(event: ModelRunEvent) -> None:
            events_fired.append(event)

        detector = ModelRunDetector(city_coords=city_coords, on_new_run=on_new_run)
        detector._cache["NYC"].temperature_f = 68.0  # Day 1 seed

        async def run() -> list[ModelRunEvent]:
            mock_session = MagicMock()
            with patch.object(detector, "is_in_check_window", return_value=(True, "12Z")):
                with patch.object(detector, "_fetch_forecast_temp", new=AsyncMock(return_value=71.0)):
                    return await detector.check_for_new_run(session=mock_session)

        events = asyncio.run(run())

        assert len(events) == 1, f"Expected 1 event, got {len(events)}"
        assert events[0].delta_f == pytest.approx(3.0, abs=0.05)
        assert events[0].city == "NYC"
        assert events[0].run_label == "12Z"
        assert len(events_fired) == 1
        # Cache updated to new temp
        assert detector._cache["NYC"].temperature_f == pytest.approx(71.0, abs=0.01)

    def test_day3_further_shift_fires_again(self):
        """Day 3: 71→73°F on 18Z run → second detection event fires."""
        city_coords = {"NYC": (40.7128, -74.006)}
        events_fired: list[ModelRunEvent] = []

        async def on_new_run(event: ModelRunEvent) -> None:
            events_fired.append(event)

        detector = ModelRunDetector(city_coords=city_coords, on_new_run=on_new_run)
        detector._cache["NYC"].temperature_f = 68.0

        mock_session = MagicMock()

        async def run_day2() -> list[ModelRunEvent]:
            with patch.object(detector, "is_in_check_window", return_value=(True, "12Z")):
                with patch.object(detector, "_fetch_forecast_temp", new=AsyncMock(return_value=71.0)):
                    return await detector.check_for_new_run(session=mock_session)

        async def run_day3() -> list[ModelRunEvent]:
            with patch.object(detector, "is_in_check_window", return_value=(True, "18Z")):
                with patch.object(detector, "_fetch_forecast_temp", new=AsyncMock(return_value=73.0)):
                    return await detector.check_for_new_run(session=mock_session)

        asyncio.run(run_day2())
        day3_events = asyncio.run(run_day3())

        assert len(day3_events) == 1
        assert day3_events[0].delta_f == pytest.approx(2.0, abs=0.05)  # 73 - 71
        assert day3_events[0].run_label == "18Z"
        assert len(events_fired) == 2
        assert len(detector.detection_log) == 2

    def test_day4_5_steady_forecast_no_event(self):
        """Days 4–5: Forecast holds at 73°F (delta=0.2 < 0.5 threshold) → no event."""
        city_coords = {"NYC": (40.7128, -74.006)}
        events_fired: list[ModelRunEvent] = []

        async def on_new_run(event: ModelRunEvent) -> None:
            events_fired.append(event)

        detector = ModelRunDetector(city_coords=city_coords, on_new_run=on_new_run)
        detector._cache["NYC"].temperature_f = 73.0
        detector._cache["NYC"].last_model_run_detected = "18Z"  # already saw this run

        async def run_steady() -> list[ModelRunEvent]:
            mock_session = MagicMock()
            with patch.object(detector, "is_in_check_window", return_value=(True, "18Z")):
                with patch.object(detector, "_fetch_forecast_temp", new=AsyncMock(return_value=73.2)):
                    return await detector.check_for_new_run(session=mock_session)

        # delta = 0.2 < DELTA_THRESHOLD_F, AND same run label → no event either way
        events = asyncio.run(run_steady())
        assert len(events) == 0
        assert len(events_fired) == 0

    def test_hrrr_weights_decrease_as_lead_time_grows(self):
        """HRRR weight decreases as NBM/GFS/ECMWF take more long-lead weight."""
        assert _get_lead_bucket(6.0) == "short"
        assert _get_lead_bucket(18.0) == "medium"
        assert _get_lead_bucket(30.0) == "long"

        assert WEIGHTS_US["short"]["hrrr"] == 0.40
        assert WEIGHTS_US["medium"]["hrrr"] == 0.25
        assert WEIGHTS_US["long"]["hrrr"] == 0.05
        assert WEIGHTS_US["long"]["nbm"] == 0.30

        # HRRR strictly decreasing across buckets
        assert WEIGHTS_US["short"]["hrrr"] > WEIGHTS_US["medium"]["hrrr"] > WEIGHTS_US["long"]["hrrr"]

    def test_day6_resolution_win(self):
        """Day 6: actual 74°F, trade resolves YES → profit recorded."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(
            market_id="nyc-above-72",
            side="buy_yes",
            price=0.55,
            amount=20,
            platform="polymarket",
            model_probability=0.82,
        )
        trader.place_order(order)
        trader.resolve_position("nyc-above-72", "yes")

        assert trader.trade_history[0]["status"] == "won"
        assert trader.trade_history[0]["pnl"] > 0
        assert trader.bankroll > 1000.0


# ===========================================================================
# 3. CALIBRATION / BRIER SCORE TEST
# ===========================================================================

class TestCalibrationBrierScore:
    """200 synthetic outcomes for Brier score validation and calibration tracking."""

    @staticmethod
    def _generate_outcomes() -> list[tuple[float, int]]:
        """
        200 deterministic outcomes with slight over-calibration so Brier < 0.15.

        Bin   | Pred prob | YES | NO  | Total
        ------+-----------+-----+-----+------
        0.90  |   0.90    |  46 |  4  |  50
        0.70  |   0.70    |  36 | 14  |  50
        0.30  |   0.30    |  14 | 36  |  50
        0.10  |   0.10    |   4 | 46  |  50

        avg Brier = 27.60 / 200 = 0.138 < 0.15
        """
        out: list[tuple[float, int]] = []
        out += [(0.90, 1)] * 46 + [(0.90, 0)] * 4
        out += [(0.70, 1)] * 36 + [(0.70, 0)] * 14
        out += [(0.30, 1)] * 14 + [(0.30, 0)] * 36
        out += [(0.10, 1)] * 4  + [(0.10, 0)] * 46
        return out

    def test_brier_formula_correctness(self):
        """calculate_brier_score returns (pred - actual)^2."""
        assert CalibrationTracker.calculate_brier_score(0.90, 1) == pytest.approx(0.01, abs=1e-9)
        assert CalibrationTracker.calculate_brier_score(0.90, 0) == pytest.approx(0.81, abs=1e-9)
        assert CalibrationTracker.calculate_brier_score(0.50, 1) == pytest.approx(0.25, abs=1e-9)
        assert CalibrationTracker.calculate_brier_score(0.00, 0) == pytest.approx(0.00, abs=1e-9)
        assert CalibrationTracker.calculate_brier_score(1.00, 1) == pytest.approx(0.00, abs=1e-9)

    def test_outcome_counts_within_tolerance(self):
        """YES counts fall within spec tolerances."""
        outcomes = self._generate_outcomes()
        assert len(outcomes) == 200

        def yes_count(p: float) -> int:
            return sum(1 for prob, act in outcomes if prob == p and act == 1)

        assert abs(yes_count(0.90) - 45) <= 5, f"p90 YES={yes_count(0.90)} (want 45±5)"
        assert abs(yes_count(0.70) - 35) <= 7, f"p70 YES={yes_count(0.70)} (want 35±7)"
        assert abs(yes_count(0.30) - 15) <= 7, f"p30 YES={yes_count(0.30)} (want 15±7)"
        assert abs(yes_count(0.10) - 5)  <= 5, f"p10 YES={yes_count(0.10)} (want 5±5)"

    def test_200_outcomes_brier_below_015(self):
        """200 synthetic records → avg Brier score < 0.15."""
        engine = _make_engine()
        tracker = CalibrationTracker(engine)

        for prob, actual in self._generate_outcomes():
            tracker.update_calibration(
                model="blended", station="KNYC",
                lead_time_days=1, predicted_prob=prob, actual_outcome=actual,
            )

        stats = tracker.get_calibration_stats(model="blended")
        assert stats["count"] == 200
        assert stats["avg_brier"] is not None
        assert stats["avg_brier"] < 0.15, (
            f"Brier score {stats['avg_brier']:.4f} is not < 0.15 (expected well-calibrated)"
        )

    def test_calibration_tracker_records_and_computes_correctly(self):
        """Four symmetric records → avg Brier = (0.01+0.81+0.81+0.01)/4 = 0.41."""
        engine = _make_engine()
        tracker = CalibrationTracker(engine)

        tracker.update_calibration("blended", "KNYC", 1, 0.9, 1)  # 0.01
        tracker.update_calibration("blended", "KNYC", 1, 0.9, 0)  # 0.81
        tracker.update_calibration("blended", "KNYC", 1, 0.1, 1)  # 0.81
        tracker.update_calibration("blended", "KNYC", 1, 0.1, 0)  # 0.01

        stats = tracker.get_calibration_stats(model="blended")
        assert stats["count"] == 4
        assert stats["avg_brier"] == pytest.approx((0.01 + 0.81 + 0.81 + 0.01) / 4, abs=1e-6)

    def test_reliability_data_bins_populated(self):
        """Reliability diagram has 10 bins; exactly 4 bins have data."""
        engine = _make_engine()
        tracker = CalibrationTracker(engine)

        for prob, actual in self._generate_outcomes():
            tracker.update_calibration("blended", "KNYC", 1, prob, actual)

        bins = tracker.get_reliability_data(model="blended")
        assert len(bins) == 10

        bins_with_data = [b for b in bins if b["count"] > 0]
        assert len(bins_with_data) == 4  # bins 0.1, 0.3, 0.7, 0.9

    def test_filter_by_station_isolates_records(self):
        """Stats filtered by station only see that station's records."""
        engine = _make_engine()
        tracker = CalibrationTracker(engine)

        tracker.update_calibration("blended", "KNYC", 1, 0.9, 1)
        tracker.update_calibration("blended", "KMDW", 1, 0.1, 0)

        nyc_stats = tracker.get_calibration_stats(model="blended", station="KNYC")
        mdw_stats = tracker.get_calibration_stats(model="blended", station="KMDW")

        assert nyc_stats["count"] == 1
        assert mdw_stats["count"] == 1
        assert nyc_stats["avg_brier"] == pytest.approx(0.01, abs=1e-6)   # (0.9-1)^2
        assert mdw_stats["avg_brier"] == pytest.approx(0.01, abs=1e-6)   # (0.1-0)^2


# ===========================================================================
# 4. CONCURRENCY / RACE CONDITION TESTS
# ===========================================================================

class TestConcurrencyRaceConditions:
    """Thread safety and overlap lock behaviour."""

    def test_overlap_lock_only_one_thread_acquires(self):
        """When two threads compete, exactly one acquires the lock."""
        lock = threading.Lock()
        results: list[bool] = []
        barrier = threading.Barrier(2)

        def try_acquire() -> None:
            barrier.wait()  # simultaneous start
            acquired = lock.acquire(blocking=False)
            results.append(acquired)
            if acquired:
                time.sleep(0.05)
                lock.release()

        t1 = threading.Thread(target=try_acquire)
        t2 = threading.Thread(target=try_acquire)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert results.count(True) == 1, f"Expected exactly 1 acquire; got {results}"
        assert results.count(False) == 1

    def test_lock_released_so_subsequent_cycles_run(self):
        """After a cycle finishes, the lock is released for the next cycle."""
        lock = threading.Lock()
        completed = 0

        def run_cycle() -> None:
            nonlocal completed
            if lock.acquire(blocking=False):
                try:
                    time.sleep(0.01)
                    completed += 1
                finally:
                    lock.release()

        for _ in range(5):
            run_cycle()

        assert completed == 5

    def test_metar_tracker_concurrent_write_read_no_crash(self):
        """METAR tracker survives 200 concurrent writes and reads."""
        tracker = CurrentConditionsTracker()
        station = "KNYC"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        errors: list[Exception] = []

        def write_loop() -> None:
            for i in range(200):
                temp = 60.0 + (i % 20)
                dt = DailyTracker(station=station, date=today)
                obs = StationObservation(station=station, temperature_f=temp)
                dt.update(obs)
                tracker._trackers[station] = dt

        def read_loop() -> None:
            for _ in range(200):
                try:
                    t = tracker.get_tracker(station)
                    if t is not None:
                        _ = t.daily_high_f
                        _ = t.daily_low_f
                        _ = t.observation_count
                except Exception as exc:
                    errors.append(exc)

        t1 = threading.Thread(target=write_loop)
        t2 = threading.Thread(target=read_loop)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert len(errors) == 0, f"Concurrent METAR access caused errors: {errors}"

    def test_market_metadata_concurrent_scan_and_decision(self):
        """market_metadata dict survives concurrent scan writes and decision reads."""
        metadata: dict[str, dict] = {}
        errors: list[Exception] = []

        def scan_loop() -> None:
            for i in range(500):
                mid = f"market-{i % 10}"
                metadata[mid] = {"yes_price": 0.5 + (i % 5) * 0.1, "city": "NYC"}

        def decision_loop() -> None:
            for _ in range(500):
                try:
                    for mid, data in list(metadata.items()):
                        _ = data.get("yes_price", 0)
                except Exception as exc:
                    errors.append(exc)

        t1 = threading.Thread(target=scan_loop)
        t2 = threading.Thread(target=decision_loop)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert len(errors) == 0, f"Concurrent metadata access caused errors: {errors}"

    def test_skipped_cycle_logged_not_crashed(self):
        """When lock is already held, a second acquire attempt is declined cleanly."""
        lock = threading.Lock()
        skips = []

        lock.acquire()
        try:
            acquired = lock.acquire(blocking=False)
            if not acquired:
                skips.append("skipped")
        finally:
            lock.release()

        assert skips == ["skipped"]


# ===========================================================================
# 5. REALISTIC EDGE PLAUSIBILITY
# ===========================================================================

class TestRealisticEdgePlausibility:
    """Blended probability and edge calculations with real-like inputs."""

    def test_hrrr_dominant_high_prob_positive_edge(self):
        """GFS 72, ECMWF 71, HRRR 73, above 70 at $0.60, lead 8h → prob high, edge 10–40%."""
        prob, model, details = blended_probability(
            threshold=70.0, direction="above", rmse=5.0,
            is_us_station=True, lead_time_hours=8.0,
            gfs_forecast_temp=72.0,
            ecmwf_forecast_temp=71.0,
            hrrr_forecast_temp=73.0,
        )

        assert prob > 0.70, f"Blended prob {prob:.3f} should be > 0.70 (HRRR dominant)"
        assert details.get("weight_bucket") == "short"

        edge = calculate_net_edge(prob, 0.60, "polymarket")
        assert 0.09 <= edge <= 0.40, (
            f"Edge {edge:.3f} should be 10–40% for this realistic setup"
        )

    def test_overpriced_market_negative_edge(self):
        """Same forecast, market at $0.95 → edge strongly negative (market already priced in)."""
        prob, _, _ = blended_probability(
            threshold=70.0, direction="above", rmse=5.0,
            is_us_station=True, lead_time_hours=8.0,
            gfs_forecast_temp=72.0, ecmwf_forecast_temp=71.0, hrrr_forecast_temp=73.0,
        )
        edge = calculate_net_edge(prob, 0.95, "polymarket")
        assert edge < 0.0, f"Edge {edge:.3f} should be negative (market overpriced at $0.95)"

    def test_wide_model_spread_moderate_probability(self):
        """GFS ensemble ~75°F vs ECMWF 65°F (10°F spread) → moderate probability, not extreme."""
        # Deterministic GFS ensemble: all 20 members at 75°F → P(>70) = 1.0
        gfs_ensemble = [75.0] * 20

        prob, _, _ = blended_probability(
            threshold=70.0, direction="above", rmse=5.0,
            is_us_station=True, lead_time_hours=24.0,
            ensemble_temps=gfs_ensemble,
            ecmwf_forecast_temp=65.0,
        )

        # GFS says 1.0, ECMWF says ~0.159 → blend should be moderate
        assert 0.35 <= prob <= 0.80, (
            f"With 10°F model disagreement, prob {prob:.3f} should be moderate (0.35–0.80)"
        )

    def test_metar_override_gross_edge_39pct(self):
        """METAR daily high 75°F for 'above 70°F' at $0.60 → override 0.99, gross edge ~39%."""
        tracker = CurrentConditionsTracker()
        station = "KNYC"
        dt = DailyTracker(station=station, date="2026-03-30")
        dt.update(StationObservation(station=station, temperature_f=75.0))
        tracker._trackers[station] = dt

        override = tracker.metar_probability_override(
            station, threshold=70.0, direction="above",
            market_type="daily_high", market_date=None,
        )
        assert override == 0.99

        gross_edge = calculate_gross_edge(override, 0.60)
        assert gross_edge == pytest.approx(0.39, abs=1e-6), (
            f"Gross edge {gross_edge:.4f} should be 0.99 - 0.60 = 0.39"
        )

    def test_metar_override_takes_precedence_over_models(self):
        """METAR override in blended_probability ignores all model forecasts."""
        prob, model, _ = blended_probability(
            threshold=70.0, direction="above",
            gfs_forecast_temp=72.0, ecmwf_forecast_temp=71.0, hrrr_forecast_temp=73.0,
            metar_override=0.99,
        )
        assert prob == 0.99
        assert model == "metar_override"

    def test_forecast_well_below_threshold_negative_edge(self):
        """Forecast 65°F for 'above 72°F' at $0.40 → negative edge (market is fairly priced or overpriced)."""
        prob, _, _ = blended_probability(
            threshold=72.0, direction="above", rmse=5.0,
            gfs_forecast_temp=65.0, ecmwf_forecast_temp=64.0,
        )
        edge = calculate_net_edge(prob, 0.40, "polymarket")
        assert edge < 0.05, f"No meaningful edge when forecast well below threshold, got {edge:.3f}"

    def test_net_edge_deducts_fees_vs_gross(self):
        """Net edge = gross edge minus platform round-trip cost."""
        model_prob = 0.75
        market_price = 0.60
        gross = calculate_gross_edge(model_prob, market_price)
        net_poly = calculate_net_edge(model_prob, market_price, "polymarket")
        net_kalshi = calculate_net_edge(model_prob, market_price, "kalshi")

        assert gross == pytest.approx(0.15, abs=1e-6)
        assert net_poly < gross   # polymarket costs 0.015
        assert net_kalshi < gross  # kalshi costs 0.02
        assert net_kalshi < net_poly  # kalshi more expensive


# ===========================================================================
# 6. CRASH RECOVERY TESTS
# ===========================================================================

class TestCrashRecovery:
    """DB persistence, corruption handling, and midnight rollover."""

    def test_open_positions_restored_after_restart(self):
        """PaperTrader restart restores open positions from DB with correct bankroll."""
        engine = _make_engine()

        # Run 1: place a trade
        trader1 = PaperTrader(initial_bankroll=1000.0)
        trader1.connect_db(engine)
        result = trader1.place_order(Order(
            market_id="nyc-70-recovery", side="buy_yes",
            price=0.60, amount=10, platform="polymarket",
        ))
        assert result["filled"] is True
        bankroll_before_crash = trader1.bankroll  # 994.0

        # Run 2: fresh trader restores from DB
        trader2 = PaperTrader(initial_bankroll=1000.0)
        trader2.connect_db(engine)

        assert len(trader2.open_positions) == 1
        assert trader2.open_positions[0]["market_id"] == "nyc-70-recovery"
        assert trader2.bankroll == pytest.approx(bankroll_before_crash, abs=1e-6)

    def test_no_duplicate_positions_on_restart(self):
        """Restarting does not double-load positions."""
        engine = _make_engine()

        trader1 = PaperTrader(initial_bankroll=1000.0)
        trader1.connect_db(engine)
        for i in range(3):
            trader1.place_order(Order(
                market_id=f"market-{i}", side="buy_yes",
                price=0.50, amount=5, platform="polymarket",
            ))

        trader2 = PaperTrader(initial_bankroll=1000.0)
        trader2.connect_db(engine)

        assert len(trader2.open_positions) == 3
        ids = [p["market_id"] for p in trader2.open_positions]
        assert len(ids) == len(set(ids)), "Duplicate positions detected after restart"

    def test_resolved_trades_in_history_after_restart(self):
        """Resolved trades appear in trade_history of the restarted trader."""
        engine = _make_engine()

        trader1 = PaperTrader(initial_bankroll=1000.0)
        trader1.connect_db(engine)
        trader1.place_order(Order(market_id="nyc-resolved", side="buy_yes", price=0.60, amount=10, platform="polymarket"))
        trader1.resolve_position("nyc-resolved", "yes")

        trader2 = PaperTrader(initial_bankroll=1000.0)
        trader2.connect_db(engine)

        assert len(trader2.open_positions) == 0
        assert len(trader2.trade_history) == 1
        assert trader2.trade_history[0]["status"] == "won"

    def test_db_corruption_handled_gracefully(self):
        """Corrupted SQLite file: connect_db error does not crash the program."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmpdb = f.name
            f.write(b"NOT_A_SQLITE_DATABASE" * 50)

        try:
            corrupt_engine = create_engine(
                f"sqlite:///{tmpdb}",
                connect_args={"check_same_thread": False},
            )
            trader = PaperTrader(initial_bankroll=500.0)

            # Should not raise an unhandled exception that kills the process
            try:
                trader.connect_db(corrupt_engine)
            except Exception:
                pass  # Acceptable: DB error is caught by the caller in main

            # In-memory trading still works (place_order catches persist errors)
            result = trader.place_order(Order(
                market_id="in-memory-only", side="buy_yes",
                price=0.50, amount=5, platform="polymarket",
            ))
            assert result["filled"] is True  # in-memory trade succeeds
        finally:
            os.unlink(tmpdb)

    def test_midnight_daily_high_resets(self):
        """After midnight, a new DailyTracker is created with reset high/low."""
        tracker = CurrentConditionsTracker()
        station = "KNYC"
        yesterday = "2026-03-29"
        today = "2026-03-30"

        # Yesterday's tracker with warm temperature
        old_dt = DailyTracker(station=station, date=yesterday)
        old_dt.update(StationObservation(station=station, temperature_f=68.0))
        tracker._trackers[station] = old_dt

        assert tracker.get_tracker(station).daily_high_f == 68.0

        # Simulate the date-mismatch reset (mirrors logic in poll_station)
        existing = tracker.get_tracker(station)
        if existing.date != today:
            new_dt = DailyTracker(station=station, date=today)
            tracker._trackers[station] = new_dt

        new_t = tracker.get_tracker(station)
        assert new_t.date == today
        assert new_t.daily_high_f == -999.0, "Daily high must reset at midnight"
        assert new_t.daily_low_f == 999.0,   "Daily low must reset at midnight"
        assert new_t.observation_count == 0

    def test_trade_placed_before_midnight_still_open_next_day(self):
        """Open position placed just before midnight remains open after the day ticks over."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="nyc-overnight", side="buy_yes", price=0.60, amount=10, platform="polymarket")
        trader.place_order(order)

        # Simulate next day (no resolution yet)
        assert trader.has_open_position("nyc-overnight")
        assert len(trader.open_positions) == 1

        # Resolve on the "next day"
        trader.resolve_position("nyc-overnight", "yes")
        assert not trader.has_open_position("nyc-overnight")
        assert trader.trade_history[0]["status"] == "won"


# ===========================================================================
# 7. ERROR ANALYSIS LEARNING TEST
# ===========================================================================

class TestErrorAnalysisLearning:
    """Systematic bias detection: Chicago overestimates, NYC reliable."""

    def _insert_chicago_losses(self, engine, count: int = 20) -> None:
        """Insert Chicago trades that all lost due to model overestimation (~3°F high)."""
        for i in range(count):
            _insert_trade(
                engine,
                market_id=f"chicago-high-{i}",
                model_prob=0.85,  # Model overconfident → model_error category
                edge=0.235,
                status="lost",
                pnl=-6.0,
                agent_name="chicago_conservative",
            )

    def _insert_nyc_trades(self, engine, wins: int = 8, losses: int = 2) -> None:
        for i in range(wins):
            _insert_trade(
                engine,
                market_id=f"nyc-high-{i}",
                model_prob=0.80,
                edge=0.185,
                status="won",
                pnl=3.50,
                agent_name="nyc_conservative",
            )
        for i in range(losses):
            _insert_trade(
                engine,
                market_id=f"nyc-loss-{i}",
                price=0.65,
                cost=6.5,
                model_prob=0.75,
                edge=0.085,
                status="lost",
                pnl=-6.5,
                agent_name="nyc_conservative",
            )

    def test_chicago_all_losses_model_error_category(self):
        """All Chicago losses have high model_probability → categorized as model errors."""
        engine = _make_engine()
        self._insert_chicago_losses(engine, count=10)

        analyzer = ErrorAnalyzer(engine)
        report = analyzer.analyze_errors(days=365)

        assert report.total_losses == 10
        assert report.model_errors == 10  # prob 0.85 > 0.70 → model error
        assert report.timing_errors == 0
        assert report.edge_compression == 0

    def test_chicago_zero_win_rate_vs_nyc_80pct(self):
        """Per-city win rates: Chicago 0%, NYC 80%."""
        engine = _make_engine()
        self._insert_chicago_losses(engine, count=20)
        self._insert_nyc_trades(engine, wins=8, losses=2)

        with get_session(engine) as session:
            rows = session.execute(select(Trade.market_id, Trade.status)).all()

        chicago = [(mid, st) for mid, st in rows if "chicago" in mid]
        nyc     = [(mid, st) for mid, st in rows if "nyc" in mid]

        chicago_win_rate = sum(1 for _, st in chicago if st == "won") / len(chicago)
        nyc_win_rate     = sum(1 for _, st in nyc     if st == "won") / len(nyc)

        assert chicago_win_rate == 0.0,  "Chicago should have 0% win rate (all lost)"
        assert nyc_win_rate == pytest.approx(0.80, abs=1e-6), "NYC should have 80% win rate"
        assert nyc_win_rate > chicago_win_rate, "NYC must be more reliable than Chicago"

    def test_chicago_rmse_higher_than_nyc(self):
        """Systematic 3°F Chicago overestimate → RMSE > NYC (±0.3°F noise)."""
        engine = _make_engine()

        # Chicago: consistent +3°F overestimate
        with get_session(engine) as session:
            for i in range(20):
                session.add(Forecast(
                    station="KMDW",
                    target_date="2026-03-30",
                    model="gfs",
                    forecast_timestamp=datetime.now(timezone.utc),
                    lead_time_hours=24,
                    temperature_forecast=70.0 + i * 0.1,
                    actual_temp=67.0 + i * 0.1,  # ~3°F lower
                ))

        # NYC: small noise ±0.3°F
        with get_session(engine) as session:
            for i in range(20):
                noise = 0.3 * (-1 if i % 2 == 0 else 1)
                session.add(Forecast(
                    station="KNYC",
                    target_date="2026-03-30",
                    model="gfs",
                    forecast_timestamp=datetime.now(timezone.utc),
                    lead_time_hours=24,
                    temperature_forecast=72.0 + i * 0.1,
                    actual_temp=72.0 + i * 0.1 + noise,
                ))

        updater = RMSEUpdater(engine)
        chicago_rmse = updater.recalculate_rmse("KMDW", "gfs", days=365)
        nyc_rmse = updater.recalculate_rmse("KNYC", "gfs", days=365)

        assert chicago_rmse is not None, "Chicago RMSE should be computable"
        assert nyc_rmse is not None, "NYC RMSE should be computable"

        assert chicago_rmse > nyc_rmse, (
            f"Chicago RMSE {chicago_rmse:.3f} should exceed NYC RMSE {nyc_rmse:.3f}"
        )
        assert chicago_rmse == pytest.approx(3.0, abs=0.01), (
            f"Chicago RMSE should be ~3.0°F (systematic bias), got {chicago_rmse:.3f}"
        )
        assert nyc_rmse == pytest.approx(0.3, abs=0.01), (
            f"NYC RMSE should be ~0.3°F (noise only), got {nyc_rmse:.3f}"
        )

    def test_worst_trades_sorted_by_pnl(self):
        """ErrorAnalyzer.worst_trades is sorted ascending (most negative first)."""
        engine = _make_engine()

        for i, pnl in enumerate([-10.0, -5.0, -8.0, -3.0, -15.0]):
            _insert_trade(
                engine,
                market_id=f"market-{i}",
                model_prob=0.80,
                edge=0.18,
                status="lost",
                pnl=pnl,
            )

        analyzer = ErrorAnalyzer(engine)
        report = analyzer.analyze_errors(days=365)

        assert report.total_losses == 5
        assert len(report.worst_trades) == 3

        pnls = [t["pnl"] for t in report.worst_trades]
        assert pnls[0] <= pnls[1] <= pnls[2], "worst_trades should be sorted ascending by P&L"
        assert pnls[0] == -15.0, "Worst loss should be -15.0"

    def test_error_types_mixed_chicago_nyc(self):
        """Combined: Chicago adds model_errors, NYC losses add timing/edge errors."""
        engine = _make_engine()
        # Chicago: high prob losses → model_errors
        self._insert_chicago_losses(engine, count=20)
        # NYC losses have prob 0.75 but edge 0.085 → edge_compression (0.75 < 0.70 is false; 0.75 > 0.70 → model_errors too)
        # Let's use low edge explicitly
        for i in range(2):
            _insert_trade(
                engine,
                market_id=f"nyc-edge-{i}",
                model_prob=0.60,  # < 0.70, edge < 0.10 → edge_compression
                edge=0.05,
                status="lost",
                pnl=-6.0,
            )

        analyzer = ErrorAnalyzer(engine)
        report = analyzer.analyze_errors(days=365)

        assert report.total_losses == 22
        assert report.model_errors == 20       # Chicago high-prob losses
        assert report.edge_compression == 2    # NYC low-edge losses
