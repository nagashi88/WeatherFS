"""Tests for PaperTrader and PositionTracker.

PRD Section 6.1, Addendum Gap 9.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from weatheredge.execution.paper_trader import Order, PaperTrader, ORDER_TTL_HOURS
from weatheredge.execution.position_tracker import PositionTracker


# ---------------------------------------------------------------------------
# PaperTrader.place_order
# ---------------------------------------------------------------------------


class TestPlaceOrder:
    def test_deducts_cost_from_bankroll(self):
        """Placing an order deducts price * amount from bankroll."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_yes", price=0.40, amount=100.0)
        result = trader.place_order(order)
        assert result["filled"] is True
        assert abs(trader.bankroll - (1000.0 - 0.40 * 100.0)) < 1e-9

    def test_filled_position_added_to_open_positions(self):
        """Filled order appears in open_positions with status 'open'."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m2", side="buy_no", price=0.30, amount=50.0)
        trader.place_order(order)
        assert len(trader.open_positions) == 1
        assert trader.open_positions[0]["status"] == "open"
        assert trader.open_positions[0]["market_id"] == "m2"

    def test_rejects_order_when_bankroll_insufficient(self):
        """Order is rejected when cost exceeds available bankroll."""
        trader = PaperTrader(initial_bankroll=10.0)
        order = Order(market_id="m3", side="buy_yes", price=0.50, amount=100.0)
        result = trader.place_order(order)
        assert result["filled"] is False
        assert "reason" in result
        # Bankroll unchanged
        assert trader.bankroll == 10.0
        assert len(trader.open_positions) == 0

    def test_order_exactly_equal_to_bankroll_is_accepted(self):
        """An order whose cost equals the full bankroll is accepted."""
        trader = PaperTrader(initial_bankroll=50.0)
        order = Order(market_id="m4", side="buy_yes", price=0.50, amount=100.0)
        result = trader.place_order(order)
        assert result["filled"] is True
        assert abs(trader.bankroll) < 1e-9


# ---------------------------------------------------------------------------
# Market id normalization (int/str robustness)
# ---------------------------------------------------------------------------


class TestMarketIdNormalization:
    def test_has_open_position_matches_int_and_str_market_id(self):
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(market_id=1814995, side="buy_no", price=0.22, amount=10.0))
        assert trader.has_open_position("1814995") is True
        assert trader.has_open_position(1814995) is True

    def test_close_position_matches_int_and_str_market_id(self):
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(market_id=42, side="buy_yes", price=0.40, amount=10.0))
        closed = trader.close_position("42", current_yes_price=0.45, reason="edge_stop")
        assert closed is True
        assert len(trader.open_positions) == 0


# ---------------------------------------------------------------------------
# PaperTrader.resolve_position
# ---------------------------------------------------------------------------


class TestResolvePosition:
    def _place(self, trader: PaperTrader, market_id: str, side: str, price: float, amount: float) -> dict:
        order = Order(market_id=market_id, side=side, price=price, amount=amount, edge_at_entry=0.20)
        result = trader.place_order(order)
        return result["trade"]

    def test_winning_trade_buy_yes_outcome_yes(self):
        """buy_yes + yes outcome → pnl positive, status 'won'."""
        trader = PaperTrader(initial_bankroll=1000.0)
        self._place(trader, "m1", "buy_yes", price=0.40, amount=100.0)
        bankroll_before_resolve = trader.bankroll

        trader.resolve_position("m1", "yes")

        assert len(trader.open_positions) == 0
        assert len(trader.trade_history) == 1
        trade = trader.trade_history[0]
        assert trade["status"] == "won"
        # Polymarket 0% fee: pnl = (amount - cost) * (1 - 0.00) = 100 - 40 = 60.0
        expected_pnl = (100.0 - 40.0) * (1.0 - 0.00)
        assert abs(trade["pnl"] - expected_pnl) < 1e-9
        assert trader.bankroll > bankroll_before_resolve

    def test_losing_trade_buy_yes_outcome_no(self):
        """buy_yes + no outcome → pnl negative, status 'lost'."""
        trader = PaperTrader(initial_bankroll=1000.0)
        self._place(trader, "m2", "buy_yes", price=0.40, amount=100.0)
        cost = 0.40 * 100.0

        trader.resolve_position("m2", "no")

        trade = trader.trade_history[0]
        assert trade["status"] == "lost"
        assert abs(trade["pnl"] - (-cost)) < 1e-9
        # Bankroll: started 1000, paid 40, lost 40 → ends at 960
        assert abs(trader.bankroll - (1000.0 - cost)) < 1e-9

    def test_winning_trade_buy_no_outcome_no(self):
        """buy_no + no outcome → pnl positive, status 'won'."""
        trader = PaperTrader(initial_bankroll=1000.0)
        self._place(trader, "m3", "buy_no", price=0.35, amount=50.0)

        trader.resolve_position("m3", "no")

        trade = trader.trade_history[0]
        assert trade["status"] == "won"
        # BugFix C1: buy_no cost = (1 - price) * amount = 0.65 * 50 = 32.5
        # Polymarket 0% fee: pnl = (amount - cost) * (1 - 0.00) = 50 - 32.5 = 17.5
        expected_pnl = (50.0 - (1.0 - 0.35) * 50.0) * (1.0 - 0.00)
        assert abs(trade["pnl"] - expected_pnl) < 1e-9

    def test_losing_trade_buy_no_outcome_yes(self):
        """buy_no + yes outcome → pnl negative, status 'lost'."""
        trader = PaperTrader(initial_bankroll=1000.0)
        self._place(trader, "m4", "buy_no", price=0.35, amount=50.0)

        trader.resolve_position("m4", "yes")

        trade = trader.trade_history[0]
        assert trade["status"] == "lost"
        assert trade["pnl"] < 0.0

    def test_unrelated_market_not_resolved(self):
        """Resolving market X does not affect positions in market Y."""
        trader = PaperTrader(initial_bankroll=1000.0)
        self._place(trader, "m_keep", "buy_yes", price=0.50, amount=10.0)
        trader.resolve_position("m_other", "yes")
        assert len(trader.open_positions) == 1
        assert len(trader.trade_history) == 0

    def test_bankroll_increases_on_win(self):
        """After a win the bankroll is higher than after paying cost."""
        trader = PaperTrader(initial_bankroll=1000.0)
        self._place(trader, "m5", "buy_yes", price=0.40, amount=100.0)
        bankroll_after_fill = trader.bankroll  # 960

        trader.resolve_position("m5", "yes")

        # Win: get back 100 (amount), net gain 60
        assert trader.bankroll > bankroll_after_fill


# ---------------------------------------------------------------------------
# PaperTrader.cancel_expired_orders
# ---------------------------------------------------------------------------


class TestCancelExpiredOrders:
    def _add_open_order(self, trader: PaperTrader, market_id: str, age_hours: float) -> dict:
        """Directly insert a fake open order with a controlled timestamp."""
        order = {
            "market_id": market_id,
            "side": "buy_yes",
            "price": 0.40,
            "amount": 10.0,
            "cost": 4.0,
            "status": "open",
            "created_at": datetime.now(timezone.utc) - timedelta(hours=age_hours),
        }
        trader.open_orders.append(order)
        return order

    def test_order_older_than_ttl_gets_cancelled(self):
        """An order older than ORDER_TTL_HOURS is cancelled and moved to history."""
        trader = PaperTrader(initial_bankroll=1000.0)
        self._add_open_order(trader, "expired_market", age_hours=ORDER_TTL_HOURS + 0.1)

        cancelled = trader.cancel_expired_orders()

        assert cancelled == 1
        assert len(trader.open_orders) == 0
        assert len(trader.trade_history) == 1
        assert trader.trade_history[0]["status"] == "cancelled"

    def test_fresh_order_survives_ttl_check(self):
        """An order younger than ORDER_TTL_HOURS is not cancelled."""
        trader = PaperTrader(initial_bankroll=1000.0)
        self._add_open_order(trader, "fresh_market", age_hours=ORDER_TTL_HOURS - 0.1)

        cancelled = trader.cancel_expired_orders()

        assert cancelled == 0
        assert len(trader.open_orders) == 1
        assert len(trader.trade_history) == 0

    def test_mixed_orders_only_expired_cancelled(self):
        """Only expired orders are cancelled; fresh orders remain."""
        trader = PaperTrader(initial_bankroll=1000.0)
        self._add_open_order(trader, "old", age_hours=ORDER_TTL_HOURS + 1.0)
        self._add_open_order(trader, "new", age_hours=0.5)

        cancelled = trader.cancel_expired_orders()

        assert cancelled == 1
        assert len(trader.open_orders) == 1
        assert trader.open_orders[0]["market_id"] == "new"

    def test_no_orders_returns_zero(self):
        """No open orders → cancel_expired_orders returns 0."""
        trader = PaperTrader(initial_bankroll=1000.0)
        assert trader.cancel_expired_orders() == 0


# ---------------------------------------------------------------------------
# PaperTrader.get_performance_report
# ---------------------------------------------------------------------------


class TestPerformanceReport:
    def _resolve(self, trader: PaperTrader, market_id: str, side: str, price: float, amount: float, outcome: str) -> None:
        order = Order(market_id=market_id, side=side, price=price, amount=amount, edge_at_entry=0.15)
        trader.place_order(order)
        trader.resolve_position(market_id, outcome)

    def test_no_trades_returns_zeros(self):
        """No resolved trades → all metrics are zero."""
        trader = PaperTrader(initial_bankroll=1000.0)
        report = trader.get_performance_report()
        assert report.total_trades == 0
        assert report.winning_trades == 0
        assert report.losing_trades == 0
        assert report.win_rate == 0.0
        assert report.total_pnl == 0.0
        assert report.roi == 0.0
        assert report.sharpe_ratio == 0.0
        assert report.max_drawdown == 0.0
        assert report.avg_edge_at_entry == 0.0

    def test_win_rate_two_wins_one_loss(self):
        """2 wins, 1 loss → win_rate == 2/3."""
        trader = PaperTrader(initial_bankroll=1000.0)
        self._resolve(trader, "a", "buy_yes", 0.40, 10.0, "yes")
        self._resolve(trader, "b", "buy_yes", 0.40, 10.0, "yes")
        self._resolve(trader, "c", "buy_yes", 0.40, 10.0, "no")
        report = trader.get_performance_report()
        assert report.total_trades == 3
        assert report.winning_trades == 2
        assert report.losing_trades == 1
        assert abs(report.win_rate - 2 / 3) < 1e-9

    def test_total_pnl_correct(self):
        """total_pnl = sum of individual trade PnLs."""
        trader = PaperTrader(initial_bankroll=1000.0)
        # Win: pnl = 10 - 4 = 6
        self._resolve(trader, "w", "buy_yes", 0.40, 10.0, "yes")
        # Loss: pnl = -4
        self._resolve(trader, "l", "buy_yes", 0.40, 10.0, "no")
        report = trader.get_performance_report()
        # Polymarket 0% fee: win pnl = (amount - cost) * (1 - 0.00) = 10 - 4 = 6.0
        expected_win_pnl = (10.0 - 0.40 * 10.0) * (1.0 - 0.00)
        assert abs(report.total_pnl - (expected_win_pnl + (-4.0))) < 1e-9

    def test_roi_equals_pnl_over_initial_bankroll(self):
        """roi == total_pnl / initial_bankroll."""
        trader = PaperTrader(initial_bankroll=1000.0)
        self._resolve(trader, "r", "buy_yes", 0.40, 10.0, "yes")
        report = trader.get_performance_report()
        assert abs(report.roi - report.total_pnl / 1000.0) < 1e-9

    def test_sharpe_single_trade_is_zero(self):
        """Single trade → std is 0 → sharpe returns 0."""
        trader = PaperTrader(initial_bankroll=1000.0)
        self._resolve(trader, "s", "buy_yes", 0.40, 10.0, "yes")
        report = trader.get_performance_report()
        assert report.sharpe_ratio == 0.0

    def test_sharpe_multiple_trades_calculated(self):
        """Multiple trades with variance → sharpe != 0."""
        trader = PaperTrader(initial_bankroll=1000.0)
        self._resolve(trader, "t1", "buy_yes", 0.40, 10.0, "yes")   # pnl = +6
        self._resolve(trader, "t2", "buy_yes", 0.40, 10.0, "no")    # pnl = -4
        self._resolve(trader, "t3", "buy_yes", 0.40, 10.0, "yes")   # pnl = +6
        report = trader.get_performance_report()
        assert report.sharpe_ratio != 0.0

    def test_avg_edge_at_entry(self):
        """avg_edge_at_entry reflects edge_at_entry stored on each trade."""
        trader = PaperTrader(initial_bankroll=1000.0)
        # edge_at_entry=0.15 set by helper
        self._resolve(trader, "e1", "buy_yes", 0.40, 10.0, "yes")
        self._resolve(trader, "e2", "buy_yes", 0.40, 10.0, "no")
        report = trader.get_performance_report()
        assert abs(report.avg_edge_at_entry - 0.15) < 1e-9

    def test_cancelled_trades_excluded_from_report(self):
        """Cancelled orders do not count toward performance metrics."""
        trader = PaperTrader(initial_bankroll=1000.0)
        # Add a cancelled order directly to trade_history
        trader.trade_history.append({
            "market_id": "cx",
            "status": "cancelled",
            "pnl": None,
            "edge_at_entry": 0.10,
            "created_at": datetime.now(timezone.utc),
        })
        report = trader.get_performance_report()
        assert report.total_trades == 0

    def test_early_exit_trade_included_in_report(self):
        """Early exits with realised pnl must be included in performance metrics."""
        trader = PaperTrader(initial_bankroll=1000.0)

        # Open a position then close it early via risk/exit manager path.
        trader.place_order(Order(market_id="x1", side="buy_yes", price=0.40, amount=10.0, edge_at_entry=0.15))
        closed = trader.close_position("x1", current_yes_price=0.30, reason="capital_protect")
        assert closed is True

        report = trader.get_performance_report()
        assert report.total_trades == 1
        assert report.winning_trades == 0
        assert report.losing_trades == 1
        assert report.total_pnl < 0


# ---------------------------------------------------------------------------
# PositionTracker
# ---------------------------------------------------------------------------


class TestPositionTracker:
    def _make_pos(self, market_id: str, city: str, cost: float, status: str = "open") -> dict:
        return {
            "market_id": market_id,
            "city": city,
            "cost": cost,
            "status": status,
        }

    def test_get_open_positions_returns_only_open(self):
        """Only positions with status='open' are returned."""
        tracker = PositionTracker()
        tracker.add_position(self._make_pos("m1", "NYC", 40.0, status="open"))
        tracker.add_position(self._make_pos("m2", "NYC", 20.0, status="won"))
        assert len(tracker.get_open_positions()) == 1

    def test_get_city_exposure_sums_costs(self):
        """get_city_exposure sums cost of all open positions in a city."""
        tracker = PositionTracker()
        tracker.add_position(self._make_pos("m1", "NYC", 40.0))
        tracker.add_position(self._make_pos("m2", "NYC", 20.0))
        tracker.add_position(self._make_pos("m3", "Chicago", 30.0))
        assert abs(tracker.get_city_exposure("NYC") - 60.0) < 1e-9
        assert abs(tracker.get_city_exposure("Chicago") - 30.0) < 1e-9

    def test_get_city_exposure_unknown_city_returns_zero(self):
        """City with no positions returns 0.0 exposure."""
        tracker = PositionTracker()
        assert tracker.get_city_exposure("Denver") == 0.0

    def test_get_positions_by_city_groups_correctly(self):
        """Positions are grouped by city key."""
        tracker = PositionTracker()
        tracker.add_position(self._make_pos("m1", "NYC", 40.0))
        tracker.add_position(self._make_pos("m2", "NYC", 20.0))
        tracker.add_position(self._make_pos("m3", "Miami", 50.0))
        by_city = tracker.get_positions_by_city()
        assert set(by_city.keys()) == {"NYC", "Miami"}
        assert len(by_city["NYC"]) == 2
        assert len(by_city["Miami"]) == 1

    def test_remove_position_clears_market(self):
        """remove_position removes all positions for the given market_id."""
        tracker = PositionTracker()
        tracker.add_position(self._make_pos("m1", "NYC", 40.0))
        tracker.add_position(self._make_pos("m2", "NYC", 20.0))
        tracker.remove_position("m1")
        assert all(p["market_id"] != "m1" for p in tracker._positions)
        assert len(tracker._positions) == 1
