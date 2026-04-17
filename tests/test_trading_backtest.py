"""Tests for the trading P&L backtester (STEP 4).

Uses synthetic data to verify:
- DailyRecord building from forecast/observation dicts
- Trade simulation logic (edge filtering, side selection, P&L)
- Metrics calculation (win rate, ROI, Sharpe, drawdown, per-city)
- CSV export
"""

from __future__ import annotations

import math
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from weatheredge.backtesting.backtester import (
    BacktestMetrics,
    DailyRecord,
    SimulatedTrade,
    TradingBacktester,
)
from weatheredge.backtesting.market_prices import HistoricalMarketPriceBook


# ---------------------------------------------------------------------------
# Fixtures: synthetic data
# ---------------------------------------------------------------------------

def _make_records(station: str, n_days: int = 90, base_temp: float = 70.0) -> list[DailyRecord]:
    """Generate n_days of synthetic daily records.

    Forecast oscillates around base_temp with a small bias (+1°F).
    Actual temp oscillates with slightly different phase.
    """
    start = date(2025, 7, 1)
    records = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        # Forecast: base + seasonal wave + small positive bias
        forecast = base_temp + 10 * math.sin(2 * math.pi * i / 30) + 1.0
        # Actual: base + seasonal wave (no bias)
        actual = base_temp + 10 * math.sin(2 * math.pi * i / 30 + 0.3)
        records.append(DailyRecord(
            station=station,
            date=d,
            forecast_high_f=round(forecast, 1),
            actual_high_f=round(actual, 1),
        ))
    return records


# ---------------------------------------------------------------------------
# DailyRecord building
# ---------------------------------------------------------------------------

class TestBuildDailyRecords:
    def test_join_matching_dates(self):
        bt = TradingBacktester()
        d1 = date(2025, 7, 1)
        d2 = date(2025, 7, 2)
        d3 = date(2025, 7, 3)

        forecasts = {d1: 75.0, d2: 78.0, d3: 72.0}
        observations = {d1: 74.0, d3: 73.0}  # d2 missing

        records = bt.build_daily_records(forecasts, observations, "KNYC")
        assert len(records) == 2  # Only d1 and d3 have both
        assert records[0].date == d1
        assert records[0].forecast_high_f == 75.0
        assert records[0].actual_high_f == 74.0
        assert records[1].date == d3

    def test_empty_inputs(self):
        bt = TradingBacktester()
        assert bt.build_daily_records({}, {}, "KNYC") == []

    def test_no_overlap(self):
        bt = TradingBacktester()
        forecasts = {date(2025, 7, 1): 75.0}
        observations = {date(2025, 7, 2): 74.0}
        assert bt.build_daily_records(forecasts, observations, "KNYC") == []


# ---------------------------------------------------------------------------
# Climatological probability
# ---------------------------------------------------------------------------

class TestClimatologicalProb:
    def test_all_above(self):
        """If all historical days exceeded threshold, prob should be ~1.0."""
        bt = TradingBacktester()
        records = [
            DailyRecord("KNYC", date(2025, 7, 1) + timedelta(days=i), 80.0, 85.0)
            for i in range(60)
        ]
        prob = bt.compute_climatological_prob(records, 70.0, "above", window_days=30, current_idx=50)
        assert prob > 0.95

    def test_none_above(self):
        """If no historical days exceeded threshold, prob should be ~0.0."""
        bt = TradingBacktester()
        records = [
            DailyRecord("KNYC", date(2025, 7, 1) + timedelta(days=i), 60.0, 55.0)
            for i in range(60)
        ]
        prob = bt.compute_climatological_prob(records, 70.0, "above", window_days=30, current_idx=50)
        assert prob < 0.05

    def test_boundary_equals_threshold_matches_live_resolution(self):
        """At exact threshold: above should count as YES, below as NO."""
        bt = TradingBacktester()
        records = [
            DailyRecord("KNYC", date(2025, 7, 1) + timedelta(days=i), 70.0, 70.0)
            for i in range(60)
        ]
        prob_above = bt.compute_climatological_prob(
            records, 70.0, "above", window_days=30, current_idx=50
        )
        prob_below = bt.compute_climatological_prob(
            records, 70.0, "below", window_days=30, current_idx=50
        )
        assert prob_above > 0.95
        assert prob_below < 0.05

    def test_too_few_records(self):
        """With fewer than 5 records, should return 0.50."""
        bt = TradingBacktester()
        records = [
            DailyRecord("KNYC", date(2025, 7, 1) + timedelta(days=i), 80.0, 85.0)
            for i in range(3)
        ]
        prob = bt.compute_climatological_prob(records, 70.0, "above", window_days=30, current_idx=2)
        assert prob == 0.50


# ---------------------------------------------------------------------------
# Trading simulation
# ---------------------------------------------------------------------------

class TestSimulateTrading:
    def test_default_thresholds_use_forecasts_not_actuals(self):
        """Threshold ladder should be derived from forecast range (no lookahead on actuals)."""
        bt = TradingBacktester()
        records = []
        start = date(2025, 7, 1)
        for i in range(60):
            records.append(DailyRecord(
                station="KNYC",
                date=start + timedelta(days=i),
                forecast_high_f=70.0 + (i % 2),  # tight forecast band: 70-71
                actual_high_f=40.0 if i % 2 == 0 else 110.0,  # wide realised extremes
            ))

        thresholds = bt._default_thresholds(records)
        # Default ladder pads ±5°F around the forecast range and snaps to the
        # 5°F grid used by market builders, so tight 70–71°F forecasts expand
        # to [65, 80].
        assert min(thresholds) == 65
        assert max(thresholds) == 80

    def test_returns_trades(self):
        """Simulate should produce at least some trades on 90 days of data."""
        bt = TradingBacktester(min_edge=0.10, bet_size=10.0)
        records = _make_records("KNYC", 90)
        trades = bt.simulate_trading(records)
        assert len(trades) > 0

    def test_no_trades_below_30_days(self):
        """First 30 days should be skipped (climatological warm-up)."""
        bt = TradingBacktester()
        records = _make_records("KNYC", 35)
        trades = bt.simulate_trading(records)
        for t in trades:
            assert t.trade_date >= records[30].date

    def test_edge_cap_respected(self):
        """No trade should have edge > max_edge."""
        bt = TradingBacktester(min_edge=0.10, max_edge=0.50)
        records = _make_records("KNYC", 90)
        trades = bt.simulate_trading(records)
        for t in trades:
            edge = abs(t.model_prob - t.market_proxy)
            assert edge <= 0.50 + 0.01  # small float tolerance

    def test_pnl_sign_correct(self):
        """Winners should have positive P&L, losers negative (before fees can flip small wins)."""
        bt = TradingBacktester(min_edge=0.10, bet_size=10.0, fee_rate=0.0)
        records = _make_records("KNYC", 90)
        trades = bt.simulate_trading(records)
        for t in trades:
            if t.won:
                assert t.pnl > 0, f"Won trade has negative pnl: {t}"
            else:
                assert t.pnl < 0, f"Lost trade has positive pnl: {t}"

    def test_empty_records(self):
        bt = TradingBacktester()
        assert bt.simulate_trading([]) == []

    def test_one_trade_per_threshold_per_day_after_complement_dedup(self):
        """Complementary above/below representations should not double-count the same threshold/day."""
        bt = TradingBacktester(min_edge=0.05, max_trades_per_station_day=50)
        records = _make_records("KNYC", 90)
        trades = bt.simulate_trading(records, thresholds=[70.0])
        keys = [(t.trade_date, t.threshold) for t in trades]
        assert len(keys) == len(set(keys))

    def test_daily_trade_cap_enforced(self):
        """No day should exceed max_trades_per_station_day."""
        bt = TradingBacktester(min_edge=0.05, max_trades_per_station_day=1)
        records = _make_records("KNYC", 90)
        trades = bt.simulate_trading(records, thresholds=[60.0, 65.0, 70.0, 75.0, 80.0])
        counts: dict[date, int] = {}
        for t in trades:
            counts[t.trade_date] = counts.get(t.trade_date, 0) + 1
        assert counts, "Expected at least one simulated trade day"
        assert max(counts.values()) <= 1

    def test_uses_real_market_price_book_when_available(self):
        """When price book has data, it should override climatological proxy."""
        bt = TradingBacktester(min_edge=0.10, max_edge=0.50, max_trades_per_station_day=5)
        records = _make_records("KNYC", 60)

        # Force climatological fallback to produce trades.
        bt.compute_climatological_prob = lambda *args, **kwargs: 0.50  # type: ignore[assignment]
        baseline = bt.simulate_trading(records, thresholds=[70.0])
        assert len(baseline) > 0

        # Real market says YES already very expensive (95c), compressing edge.
        rows = []
        for r in records:
            rows.append({
                "station": "KNYC",
                "date": r.date.isoformat(),
                "threshold": 70.0,
                "direction": "above",
                "yes_price": 0.95,
            })
        price_book = HistoricalMarketPriceBook.from_rows(rows)
        with_real_prices = bt.simulate_trading(
            records, thresholds=[70.0], market_price_book=price_book
        )
        assert len(with_real_prices) < len(baseline)

    def test_price_floor_applies_to_selected_side_not_just_yes_side(self):
        """buy_no entries below min_price should be filtered out."""
        bt = TradingBacktester(
            min_edge=0.10,
            max_edge=1.0,
            min_price=0.10,
            max_trades_per_station_day=5,
        )
        records = _make_records("KNYC", 60, base_temp=60.0)
        bt.compute_climatological_prob = lambda *args, **kwargs: 0.95  # type: ignore[assignment]
        trades = bt.simulate_trading(records, thresholds=[70.0])
        assert trades == []


# ---------------------------------------------------------------------------
# Metrics calculation
# ---------------------------------------------------------------------------

class TestCalculateMetrics:
    def _make_trades(self) -> list[SimulatedTrade]:
        """Create a known set of trades for deterministic metrics testing."""
        d1 = date(2025, 7, 1)
        d2 = date(2025, 7, 2)
        return [
            SimulatedTrade("KNYC", d1, 75.0, "above", 0.70, 0.50, "buy_yes", 0.50, True, 4.80),
            SimulatedTrade("KNYC", d1, 80.0, "above", 0.60, 0.45, "buy_yes", 0.45, False, -4.70),
            SimulatedTrade("KMDW", d2, 70.0, "above", 0.65, 0.50, "buy_yes", 0.50, True, 4.80),
            SimulatedTrade("KMDW", d2, 65.0, "below", 0.70, 0.55, "buy_yes", 0.55, True, 4.30),
        ]

    def test_win_rate(self):
        bt = TradingBacktester()
        trades = self._make_trades()
        metrics = bt.calculate_metrics(trades)
        assert metrics.total_trades == 4
        assert metrics.wins == 3
        assert metrics.losses == 1
        assert metrics.win_rate == pytest.approx(0.75, abs=0.01)

    def test_net_pnl(self):
        bt = TradingBacktester()
        trades = self._make_trades()
        metrics = bt.calculate_metrics(trades)
        # gross_pnl from actual entry prices and bet_size=10
        assert metrics.net_pnl != 0  # Should have some P&L

    def test_metrics_fee_consistency_with_trade_pnls(self):
        """total_fees should reconcile gross_pnl and sum of per-trade net pnl."""
        bt = TradingBacktester()
        trades = self._make_trades()
        metrics = bt.calculate_metrics(trades)

        expected_net = sum(t.pnl for t in trades)
        expected_gross = sum(
            ((1.0 - t.entry_price) if t.won else -t.entry_price) * bt.bet_size
            for t in trades
        )
        expected_fees = max(0.0, expected_gross - expected_net)

        assert metrics.net_pnl == pytest.approx(expected_net, abs=1e-9)
        assert metrics.total_fees == pytest.approx(expected_fees, abs=1e-9)

    def test_per_city_breakdown(self):
        bt = TradingBacktester()
        trades = self._make_trades()
        metrics = bt.calculate_metrics(trades)
        assert "KNYC" in metrics.per_city
        assert "KMDW" in metrics.per_city
        assert metrics.per_city["KNYC"]["trades"] == 2
        assert metrics.per_city["KMDW"]["trades"] == 2

    def test_empty_trades(self):
        bt = TradingBacktester()
        metrics = bt.calculate_metrics([])
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.net_pnl == 0.0

    def test_max_drawdown_nonnegative(self):
        bt = TradingBacktester()
        records = _make_records("KNYC", 90)
        trades = bt.simulate_trading(records)
        metrics = bt.calculate_metrics(trades)
        assert metrics.max_drawdown >= 0.0

    def test_edge_distribution(self):
        bt = TradingBacktester()
        trades = self._make_trades()
        metrics = bt.calculate_metrics(trades)
        assert len(metrics.edge_distribution) == 4
        for e in metrics.edge_distribution:
            assert 0.0 <= e <= 1.0


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

class TestCSVExport:
    def test_save_and_read(self):
        bt = TradingBacktester()
        trades = [
            SimulatedTrade("KNYC", date(2025, 7, 1), 75.0, "above", 0.70, 0.50, "buy_yes", 0.50, True, 4.80),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            bt.save_trades_csv(trades, path)
            assert path.exists()
            content = path.read_text()
            assert "KNYC" in content
            assert "2025-07-01" in content
            lines = content.strip().split("\n")
            assert len(lines) == 2  # header + 1 trade


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

class TestPrintReport:
    def test_report_contains_key_metrics(self, capsys):
        bt = TradingBacktester()
        metrics = BacktestMetrics(
            total_trades=100,
            wins=60,
            losses=40,
            win_rate=0.60,
            gross_pnl=120.0,
            total_fees=20.0,
            net_pnl=100.0,
            roi_pct=10.0,
            sharpe_ratio=1.5,
            max_drawdown=30.0,
            per_city={"KNYC": {"trades": 50, "wins": 30, "pnl": 60.0, "win_rate": 0.60}},
            edge_distribution=[0.12, 0.15, 0.20],
        )
        report = bt.print_report(metrics)
        assert "Win rate:" in report
        assert "60.0%" in report
        assert "Sharpe ratio:" in report
        assert "KNYC" in report
