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


# ---------------------------------------------------------------------------
# Re-entry guard (daily entry limit)
# ---------------------------------------------------------------------------


class TestDailyEntryLimit:
    def _order(self, market_id: str) -> Order:
        return Order(market_id=market_id, side="buy_yes", price=0.30, amount=10.0)

    def test_first_entry_allowed(self):
        """First entry on a market is always allowed."""
        trader = PaperTrader(initial_bankroll=1000.0)
        assert trader.has_reached_daily_entry_limit("m1") is False

    def test_second_entry_allowed(self):
        """Second entry is still within the default limit of 2."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(self._order("m1"))
        assert trader.has_reached_daily_entry_limit("m1") is False

    def test_third_entry_blocked(self):
        """Third entry on the same market in the same day is blocked."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(self._order("m1"))
        trader.place_order(self._order("m1"))
        assert trader.has_reached_daily_entry_limit("m1") is True

    def test_different_markets_independent(self):
        """Entry limit on m1 does not affect m2."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(self._order("m1"))
        trader.place_order(self._order("m1"))
        assert trader.has_reached_daily_entry_limit("m2") is False

    def test_custom_max_entries(self):
        """max_entries parameter is respected."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(self._order("m1"))
        assert trader.has_reached_daily_entry_limit("m1", max_entries=1) is True
        assert trader.has_reached_daily_entry_limit("m1", max_entries=2) is False

    def test_int_and_str_market_id_treated_same(self):
        """Market IDs '1953479' and 1953479 count against the same limit."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(market_id="1953479", side="buy_no", price=0.70, amount=5.0))
        trader.place_order(Order(market_id=1953479, side="buy_no", price=0.68, amount=5.0))
        assert trader.has_reached_daily_entry_limit("1953479") is True
        assert trader.has_reached_daily_entry_limit(1953479) is True


# ---------------------------------------------------------------------------
# BugFix P0-7: market_date-scoped re-entry guard
# (UTC-midnight rollover must not reset the counter within one market day)
# ---------------------------------------------------------------------------


class TestDailyEntryLimitMarketDate:
    """Re-entry guard must key on station-local market_date, not UTC today.

    Profitability impact: a market_date-blind counter resets at UTC midnight while
    Kalshi's market is still live (local evening for western stations). That was
    the 1953479 pattern that produced 4 losing entries in one local day.
    """

    def _order(self, market_id, market_date="2026-04-16"):
        return Order(
            market_id=market_id, side="buy_yes", price=0.30, amount=10.0,
            market_date=market_date,
        )

    def test_market_date_keeps_separate_buckets(self):
        """Different market_dates for same market_id keep independent counters."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader._record_daily_entry("m1", market_date="2026-04-16")
        trader._record_daily_entry("m1", market_date="2026-04-16")
        assert trader.has_reached_daily_entry_limit("m1", market_date="2026-04-16") is True
        # Next day's same market id is a different contract → fresh counter.
        assert trader.has_reached_daily_entry_limit("m1", market_date="2026-04-17") is False

    def test_place_order_records_on_market_date_bucket(self):
        """place_order routes to the order's market_date, not UTC today."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(self._order("m1", market_date="2026-04-16"))
        trader.place_order(self._order("m1", market_date="2026-04-16"))
        assert trader.has_reached_daily_entry_limit(
            "m1", market_date="2026-04-16",
        ) is True
        # Different day's contract → still fresh, prevents false-positive block.
        assert trader.has_reached_daily_entry_limit(
            "m1", market_date="2026-04-17",
        ) is False

    def test_utc_fallback_when_market_date_none(self):
        """Orders without market_date fall back to UTC today (legacy behavior)."""
        trader = PaperTrader(initial_bankroll=1000.0)
        # Order with empty market_date uses UTC fallback path.
        trader.place_order(Order(
            market_id="legacy", side="buy_yes", price=0.30, amount=10.0,
        ))
        trader.place_order(Order(
            market_id="legacy", side="buy_yes", price=0.30, amount=10.0,
        ))
        assert trader.has_reached_daily_entry_limit("legacy") is True

    def test_utc_midnight_rollover_does_not_reset(self):
        """A market trading past UTC midnight keeps counting against the same
        station-local day (the whole point of BUG #3).
        """
        trader = PaperTrader(initial_bankroll=1000.0)
        market_date = "2026-04-16"
        # Two entries recorded during station-local Apr 16 (regardless of UTC wall clock).
        trader._record_daily_entry("m1", market_date=market_date)
        trader._record_daily_entry("m1", market_date=market_date)
        # Caller (main.py) always passes market_date → the UTC date of "now"
        # cannot accidentally open a third slot.
        assert trader.has_reached_daily_entry_limit(
            "m1", market_date=market_date,
        ) is True

    def test_check_limit_fills_uses_pending_market_date(self):
        """check_limit_fills must record the pending order's market_date."""
        trader = PaperTrader(initial_bankroll=1000.0)
        pending = Order(
            market_id="lim1", side="buy_yes", price=0.30, amount=10.0,
            market_date="2026-04-16", order_type="limit",
        )
        trader.place_order(pending)
        trader.check_limit_fills({"lim1": 0.10})  # well below limit — triggers
        assert trader.has_reached_daily_entry_limit(
            "lim1", market_date="2026-04-16",
        ) is False  # only 1 entry
        # Re-fill with a second pending order to push to the limit.
        trader.place_order(pending)
        trader.check_limit_fills({"lim1": 0.10})
        assert trader.has_reached_daily_entry_limit(
            "lim1", market_date="2026-04-16",
        ) is True

    def test_custom_max_entries_with_market_date(self):
        """max_entries parameter still respected when market_date is scoped."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader._record_daily_entry("m1", market_date="2026-04-16")
        assert trader.has_reached_daily_entry_limit(
            "m1", max_entries=1, market_date="2026-04-16",
        ) is True
        assert trader.has_reached_daily_entry_limit(
            "m1", max_entries=3, market_date="2026-04-16",
        ) is False


# ---------------------------------------------------------------------------
# _load_from_db restores counters keyed on market_date
# ---------------------------------------------------------------------------


class TestLoadFromDbDailyEntries:
    """_load_from_db must rebuild _daily_entries using market_date so the
    re-entry guard survives restarts mid-session without losing track of
    how many times we already entered a market this station-local day.
    """

    def _engine(self):
        from weatheredge.db.database import init_db, make_engine
        eng = make_engine("sqlite:///:memory:")
        init_db(engine=eng)
        return eng

    def _insert_trade(
        self, engine, *, market_id, market_date, created_at=None,
        status="open", pnl=None,
    ):
        from sqlalchemy.orm import Session
        from weatheredge.db.models import Trade
        with Session(engine) as session:
            t = Trade(
                market_id=market_id,
                platform="polymarket",
                mode="paper",
                side="buy_yes",
                price=0.30,
                amount=10.0,
                cost=3.0,
                status=status,
                market_date=market_date,
                created_at=created_at or datetime.now(timezone.utc),
                pnl=pnl,
            )
            session.add(t)
            session.commit()

    def test_restores_counter_on_market_date(self):
        """Two open trades with same market_date restore counter=2."""
        engine = self._engine()
        md = "2026-04-16"
        self._insert_trade(engine, market_id="m1", market_date=md)
        self._insert_trade(engine, market_id="m1", market_date=md)

        trader = PaperTrader(initial_bankroll=1000.0)
        trader.connect_db(engine)

        assert trader.has_reached_daily_entry_limit("m1", market_date=md) is True

    def test_different_market_dates_do_not_collide_on_restore(self):
        """Trades on different market_dates build independent counters."""
        engine = self._engine()
        self._insert_trade(engine, market_id="m1", market_date="2026-04-16")
        self._insert_trade(engine, market_id="m1", market_date="2026-04-17")

        trader = PaperTrader(initial_bankroll=1000.0)
        trader.connect_db(engine)

        assert trader.has_reached_daily_entry_limit(
            "m1", market_date="2026-04-16",
        ) is False
        assert trader.has_reached_daily_entry_limit(
            "m1", market_date="2026-04-17",
        ) is False

    def test_legacy_trade_without_market_date_uses_utc_fallback(self):
        """Legacy rows (market_date=None) still count against UTC-today bucket."""
        engine = self._engine()
        self._insert_trade(
            engine, market_id="legacy", market_date=None,
            created_at=datetime.now(timezone.utc),
        )

        trader = PaperTrader(initial_bankroll=1000.0)
        trader.connect_db(engine)

        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        # counter restored against UTC today.
        assert trader._daily_entries.get("legacy", {}).get(today_utc, 0) == 1

    def test_legacy_trade_from_prior_utc_day_not_counted(self):
        """Legacy row with old created_at must not inflate today's counter."""
        engine = self._engine()
        old = datetime.now(timezone.utc) - timedelta(days=5)
        self._insert_trade(
            engine, market_id="legacy_old", market_date=None, created_at=old,
        )

        trader = PaperTrader(initial_bankroll=1000.0)
        trader.connect_db(engine)

        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert trader._daily_entries.get("legacy_old", {}).get(today_utc, 0) == 0


# ---------------------------------------------------------------------------
# Small coverage wins for non-DB edge cases
# ---------------------------------------------------------------------------


class TestPaperTraderEdgeCases:
    def test_market_id_key_none_returns_empty_string(self):
        """_market_id_key(None) → '' (defensive)."""
        assert PaperTrader._market_id_key(None) == ""

    def test_close_position_buy_no_uses_no_leg_value(self):
        """close_position on buy_no computes current_value from (1 - yes_price)."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="no_side", side="buy_no", price=0.30, amount=10.0,
        ))
        # yes_price 0.40 → no_value = 0.60; we paid 0.30 → pnl ≈ +3.0 on 10 contracts
        closed = trader.close_position(
            "no_side", current_yes_price=0.40, reason="edge_stop",
        )
        assert closed is True
        assert len(trader.open_positions) == 0

    def test_check_limit_fills_skips_non_pending_orders(self):
        """Orders without status=='pending' are skipped by check_limit_fills."""
        trader = PaperTrader(initial_bankroll=1000.0)
        # Manually inject a limit order whose status is not 'pending'.
        trader.open_orders.append({
            "market_id": "ghost", "side": "buy_yes", "price": 0.30,
            "amount": 10.0, "limit_price": 0.25, "agent_name": "",
            "model_probability": 0.0, "edge_at_entry": 0.0, "city": "",
            "platform": "polymarket", "station": "", "market_date": "",
            "gaussian_prob": None, "ensemble_prob": None, "blended_prob": None,
            "raw_model_prob": None, "status": "cancelled",
            "created_at": datetime.now(timezone.utc), "pnl": None,
        })
        filled = trader.check_limit_fills({"ghost": 0.10})
        assert filled == 0
        assert len(trader.open_orders) == 1  # kept, still non-pending

    def test_check_limit_fills_insufficient_bankroll_keeps_pending(self):
        """A limit order whose fill cost exceeds bankroll stays pending, not lost."""
        trader = PaperTrader(initial_bankroll=1.0)  # tight bankroll
        trader.place_order(Order(
            market_id="big", side="buy_yes", price=0.50, amount=100.0,
            market_date="2026-04-16", order_type="limit",
        ))
        # Price falls to well below limit → would trigger a fill (cost ~$50).
        filled = trader.check_limit_fills({"big": 0.01})
        assert filled == 0
        assert len(trader.open_orders) == 1
        assert trader.open_orders[0]["status"] == "pending"

    def test_check_limit_fills_persist_exception_keeps_inmemory(self):
        """If _persist_new_trade raises, fill still completes in-memory (line 422-427)."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="persist_fail", side="buy_yes", price=0.50, amount=10.0,
            market_date="2026-04-16", order_type="limit",
        ))
        # Force _persist_new_trade to raise, simulating DB failure mid-fill.
        from unittest.mock import patch
        with patch.object(trader, "_persist_new_trade", side_effect=RuntimeError("db down")):
            filled = trader.check_limit_fills({"persist_fail": 0.10})
        assert filled == 1
        assert len(trader.open_positions) == 1
        assert trader.open_positions[0]["db_id"] is None

    def test_load_from_db_corrupt_inmem_created_at_string_ignored(self):
        """The defensive ``isinstance(created, str)`` branch in
        ``_load_from_db`` only fires for corrupted/legacy in-memory state — DB
        ``created_at`` is a typed ``DateTime`` column, so a real row never
        produces a string here. We exercise the branch via direct injection;
        the test asserts the parser silently skips rather than crashing.
        """
        from weatheredge.db.database import init_db, make_engine
        eng = make_engine("sqlite:///:memory:")
        init_db(engine=eng)
        trader = PaperTrader(initial_bankroll=1000.0)
        trader._engine = eng
        trader.open_positions.append({
            "market_id": "corrupt", "market_date": None,
            "created_at": "not-an-iso-date", "cost": 0.0,
        })
        trader._daily_entries.clear()
        trader.connect_db(eng)
        assert trader._daily_entries.get("corrupt", {}) == {}

    def test_load_from_db_real_row_with_null_dates_skipped(self):
        """End-to-end via DB: a persisted Trade with NULL ``created_at`` AND
        NULL ``market_date`` must produce no counter entry. Exercises the
        real DB-load path (not just in-memory walking). We use a raw SQL UPDATE
        because the ORM column default would otherwise auto-fill ``created_at``
        with ``datetime.now()`` even when the constructor receives ``None``.
        """
        from sqlalchemy import text
        from sqlalchemy.orm import Session
        from weatheredge.db.database import init_db, make_engine
        from weatheredge.db.models import Trade
        eng = make_engine("sqlite:///:memory:")
        init_db(engine=eng)
        with Session(eng) as session:
            session.add(Trade(
                market_id="nodate", platform="polymarket", side="buy_yes",
                price=0.30, amount=10.0, cost=3.0, mode="paper", status="open",
                market_date=None,
            ))
            session.commit()
            session.execute(text(
                "UPDATE trades SET created_at = NULL, market_date = NULL "
                "WHERE market_id = 'nodate'"
            ))
            session.commit()
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.connect_db(eng)
        assert trader._daily_entries.get("nodate", {}) == {}

    def test_load_from_db_real_row_with_market_date_increments_counter(self):
        """End-to-end via DB: a persisted open Trade with a populated
        ``market_date`` must show up in the daily-entry counter under that
        market_date — confirming the DB-load path (not the in-memory walk)
        feeds the re-entry guard correctly.
        """
        from sqlalchemy.orm import Session
        from weatheredge.db.database import init_db, make_engine
        from weatheredge.db.models import Trade
        eng = make_engine("sqlite:///:memory:")
        init_db(engine=eng)
        with Session(eng) as session:
            session.add(Trade(
                market_id="m99", platform="polymarket", side="buy_yes",
                price=0.30, amount=10.0, cost=3.0, mode="paper", status="open",
                market_date="2026-04-16",
            ))
            session.commit()
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.connect_db(eng)
        assert trader._daily_entries.get("m99", {}).get("2026-04-16") == 1


# ---------------------------------------------------------------------------
# DB persistence coverage (place_order/resolve/close with engine attached)
# ---------------------------------------------------------------------------


def _fresh_engine():
    """Return an in-memory SQLite engine with the trades schema initialized."""
    from weatheredge.db.database import init_db, make_engine
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    return eng


class TestDbPersistencePaths:
    """Exercises _persist_new_trade / _update_trade_in_db / resolved-loop so the
    DB-backed code paths are actually covered.
    """

    def test_place_order_persists_new_trade_and_stamps_db_id(self):
        """place_order with engine attached calls _persist_new_trade and gets a db_id."""
        eng = _fresh_engine()
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.connect_db(eng)
        result = trader.place_order(Order(
            market_id="persist_ok", side="buy_yes", price=0.40,
            amount=10.0, market_date="2030-05-01",
        ))
        assert result["filled"] is True
        assert trader.open_positions[0]["db_id"] is not None

    def test_resolve_position_persists_status_and_pnl(self):
        """resolve_position with engine attached updates DB via _update_trade_in_db."""
        eng = _fresh_engine()
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.connect_db(eng)
        trader.place_order(Order(
            market_id="resolve_db", side="buy_yes", price=0.40,
            amount=10.0, market_date="2030-05-01",
        ))
        db_id = trader.open_positions[0]["db_id"]
        trader.resolve_position("resolve_db", "yes")

        from sqlalchemy.orm import Session
        from weatheredge.db.models import Trade
        with Session(eng) as session:
            row = session.get(Trade, db_id)
            assert row.status == "won"
            assert row.pnl is not None
            assert row.resolved_at is not None

    def test_close_position_persists_reason_and_pnl(self):
        """close_position with engine attached persists the exit reason to DB."""
        eng = _fresh_engine()
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.connect_db(eng)
        trader.place_order(Order(
            market_id="close_db", side="buy_yes", price=0.40,
            amount=10.0, market_date="2030-05-01",
        ))
        db_id = trader.open_positions[0]["db_id"]
        trader.close_position("close_db", current_yes_price=0.50, reason="edge_stop")

        from sqlalchemy.orm import Session
        from weatheredge.db.models import Trade
        with Session(eng) as session:
            row = session.get(Trade, db_id)
            assert row.status == "edge_stop"
            assert row.pnl is not None

    def test_load_from_db_rebuilds_daily_pnl_from_resolved_trades(self):
        """Resolved rows with pnl+resolved_at repopulate daily_pnl on restart (129-133)."""
        eng = _fresh_engine()
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.connect_db(eng)
        # Place + resolve a winning trade, then spin up a second trader on the
        # same engine. The resolved-trades loop must rebuild daily_pnl.
        trader.place_order(Order(
            market_id="rebuild_pnl", side="buy_yes", price=0.40,
            amount=10.0, market_date="2030-05-01",
        ))
        trader.resolve_position("rebuild_pnl", "yes")
        # Second trader restores from DB.
        trader2 = PaperTrader(initial_bankroll=1000.0)
        trader2.connect_db(eng)
        assert len(trader2.trade_history) == 1
        # daily_pnl keyed by resolved_at date; has exactly one entry.
        assert sum(trader2.daily_pnl.values()) > 0.0

    def test_update_trade_in_db_no_engine_is_noop(self):
        """_update_trade_in_db early-returns when no engine is attached."""
        trader = PaperTrader(initial_bankroll=1000.0)
        # Should not raise even though db_id is set but no engine exists.
        trader._update_trade_in_db(db_id=42, status="won", pnl=1.0)

    def test_update_trade_in_db_missing_row_is_noop(self):
        """_update_trade_in_db silently skips when the db_id doesn't exist."""
        eng = _fresh_engine()
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.connect_db(eng)
        # db_id 9999 doesn't exist → session.get returns None → no-op branch.
        trader._update_trade_in_db(db_id=9999, status="won", pnl=1.0)

    def test_persist_new_trade_no_engine_returns_none(self):
        """_persist_new_trade early-returns None when no engine is attached."""
        trader = PaperTrader(initial_bankroll=1000.0)
        assert trader._persist_new_trade({"market_id": "x", "side": "buy_yes",
                                          "price": 0.4, "amount": 1.0, "cost": 0.4}) is None

    def test_place_order_persist_exception_keeps_position_in_memory(self):
        """If _persist_new_trade raises, place_order still fills in-memory (345-350)."""
        eng = _fresh_engine()
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.connect_db(eng)
        from unittest.mock import patch
        with patch.object(trader, "_persist_new_trade", side_effect=RuntimeError("boom")):
            result = trader.place_order(Order(
                market_id="persist_boom", side="buy_yes", price=0.40,
                amount=10.0, market_date="2030-05-01",
            ))
        assert result["filled"] is True
        assert trader.open_positions[0]["db_id"] is None


# ---------------------------------------------------------------------------
# Limit-order branches (buy_no leg, missing price, not-triggered)
# ---------------------------------------------------------------------------


class TestLimitOrderBranches:
    def test_buy_no_limit_price_uses_no_leg_offset(self):
        """buy_no limit orders compute limit_price = (1 - yes_price) - offset (line 287)."""
        trader = PaperTrader(initial_bankroll=1000.0)
        result = trader.place_order(Order(
            market_id="lim_no", side="buy_no", price=0.30, amount=10.0,
            market_date="2030-05-01", order_type="limit",
        ))
        assert result["filled"] is False
        assert result["pending"] is True
        # (1 - 0.30) - offset; offset default is tiny. Must be < 0.70.
        assert result["limit_price"] < 0.70

    def test_check_limit_fills_missing_market_price_keeps_pending(self):
        """check_limit_fills skips (keeps pending) when price absent (377-378)."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="no_price", side="buy_yes", price=0.40, amount=10.0,
            market_date="2030-05-01", order_type="limit",
        ))
        filled = trader.check_limit_fills({})  # no price for "no_price"
        assert filled == 0
        assert len(trader.open_orders) == 1
        assert trader.open_orders[0]["status"] == "pending"

    def test_check_limit_fills_not_triggered_keeps_pending(self):
        """Price that doesn't cross limit keeps the order pending (387-388)."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="far_price", side="buy_yes", price=0.40, amount=10.0,
            market_date="2030-05-01", order_type="limit",
        ))
        # current yes price well above the limit → not triggered.
        filled = trader.check_limit_fills({"far_price": 0.95})
        assert filled == 0
        assert len(trader.open_orders) == 1

    def test_check_limit_fills_buy_no_triggers_on_low_no_leg(self):
        """buy_no limit fills when (1 - yes_price) <= limit_price (384, 393)."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="bn_fill", side="buy_no", price=0.80, amount=10.0,
            market_date="2030-05-01", order_type="limit",
        ))
        # Pending limit_price is ~ (1 - 0.80) = 0.20. Need yes >= 0.80 so no-leg <= 0.20.
        filled = trader.check_limit_fills({"bn_fill": 0.85})
        assert filled == 1
        assert len(trader.open_positions) == 1
        # Cost == fill_price * amount (for buy_no fill_price is no-leg price).
        pos = trader.open_positions[0]
        assert pos["side"] == "buy_no"
        assert pos["cost"] > 0.0

    def test_check_limit_fills_buy_no_not_triggered(self):
        """buy_no limit stays pending when yes_price keeps no-leg above limit."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="bn_nofill", side="buy_no", price=0.80, amount=10.0,
            market_date="2030-05-01", order_type="limit",
        ))
        filled = trader.check_limit_fills({"bn_nofill": 0.50})  # no-leg = 0.50 > limit
        assert filled == 0
        assert len(trader.open_orders) == 1

    def test_buy_no_limit_fill_stores_canonical_yes_price(self):
        # STEP 1 fix: limit fill must normalize buy_no trade["price"] to YES-side
        # so downstream readers (exit_manager entry_price, main.py close fallback)
        # see the same semantic as a market-filled buy_no (which stores YES-price).
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="bn_canonical", side="buy_no", price=0.80, amount=10.0,
            market_date="2030-05-01", order_type="limit",
        ))
        filled = trader.check_limit_fills({"bn_canonical": 0.85})
        assert filled == 1
        pos = trader.open_positions[0]
        assert pos["side"] == "buy_no"
        # limit_price ≈ 0.20 - offset (no-leg). YES-canonical ≈ 0.80 + offset.
        assert pos["price"] > 0.5, f"expected YES-canonical > 0.5, got {pos['price']}"
        assert abs(pos["price"] + pos["fill_no_price"] - 1.0) < 1e-9
        # Cost still on the NO leg (fill_no_price * amount).
        assert abs(pos["cost"] - pos["fill_no_price"] * pos["amount"]) < 1e-9

    def test_buy_no_limit_fill_pnl_matches_market_fill_equivalent(self):
        # STEP 1 fix regression: a buy_no limit fill closed at the same YES-price
        # must produce the same PnL as a buy_no market fill closed at that price.
        from weatheredge import config
        # Disable temporal guard so tests don't need timestamp plumbing.
        trader_limit = PaperTrader(initial_bankroll=1000.0)
        trader_market = PaperTrader(initial_bankroll=1000.0)

        # Limit fill path: place limit, trigger at yes=0.85 → no-leg fills ~0.15.
        trader_limit.place_order(Order(
            market_id="pnl_limit", side="buy_no", price=0.80, amount=10.0,
            market_date="2030-05-01", order_type="limit",
        ))
        trader_limit.check_limit_fills({"pnl_limit": 0.85})
        pos_limit = trader_limit.open_positions[0]
        yes_price_at_fill = pos_limit["price"]  # should be YES-canonical ~0.85

        # Market fill path: place market order at the SAME YES-price.
        trader_market.place_order(Order(
            market_id="pnl_market", side="buy_no",
            price=yes_price_at_fill, amount=10.0,
            market_date="2030-05-01",
        ))
        pos_market = trader_market.open_positions[0]

        # Cost should match within floating tolerance.
        assert abs(pos_limit["cost"] - pos_market["cost"]) < 1e-9

        # Close both at yes=0.60 — buy_no profits when YES falls.
        trader_limit.close_position("pnl_limit", current_yes_price=0.60, reason="edge_stop")
        trader_market.close_position("pnl_market", current_yes_price=0.60, reason="edge_stop")

        limit_pnl = trader_limit.trade_history[-1]["pnl"]
        market_pnl = trader_market.trade_history[-1]["pnl"]
        assert abs(limit_pnl - market_pnl) < 1e-9, (
            f"buy_no limit PnL {limit_pnl} != market PnL {market_pnl} — semantics diverged"
        )


# ---------------------------------------------------------------------------
# Temporal guard (lookahead leakage protection)
# ---------------------------------------------------------------------------


class TestTemporalGuardPath:
    def test_temporal_guard_off_allows_order_with_lookahead_timestamps(self, monkeypatch):
        """Temporal guard disabled → timestamps do not gate the order."""
        from weatheredge import config
        monkeypatch.setitem(config.MODULES, "temporal_guard", False)
        trader = PaperTrader(initial_bankroll=1000.0)
        # forecast_timestamp AFTER market_price_timestamp = lookahead, but guard is off.
        later = datetime.now(timezone.utc)
        earlier = later - timedelta(hours=1)
        result = trader.place_order(Order(
            market_id="tg_off", side="buy_yes", price=0.40, amount=10.0,
            market_date="2030-05-01",
            forecast_timestamp=later,
            market_price_timestamp=earlier,
        ))
        assert result["filled"] is True

    def test_temporal_guard_clean_timestamps_passes(self, monkeypatch):
        """Temporal guard on + forecast<market_price → assert_no_lookahead passes."""
        from weatheredge import config
        monkeypatch.setitem(config.MODULES, "temporal_guard", True)
        trader = PaperTrader(initial_bankroll=1000.0)
        earlier = datetime.now(timezone.utc) - timedelta(hours=1)
        later = datetime.now(timezone.utc)
        result = trader.place_order(Order(
            market_id="tg_ok", side="buy_yes", price=0.40, amount=10.0,
            market_date="2030-05-01",
            forecast_timestamp=earlier,
            market_price_timestamp=later,
        ))
        assert result["filled"] is True

    def test_temporal_guard_lookahead_blocks_order(self, monkeypatch):
        """Temporal guard on + lookahead timestamps → TemporalLeakageError blocks fill."""
        from weatheredge import config
        monkeypatch.setitem(config.MODULES, "temporal_guard", True)
        trader = PaperTrader(initial_bankroll=1000.0)
        # forecast_timestamp strictly AFTER market_price_timestamp → leakage.
        forecast_ts = datetime.now(timezone.utc)
        market_ts = forecast_ts - timedelta(hours=2)
        result = trader.place_order(Order(
            market_id="tg_leak", side="buy_yes", price=0.40, amount=10.0,
            market_date="2030-05-01",
            forecast_timestamp=forecast_ts,
            market_price_timestamp=market_ts,
        ))
        # Either the order is rejected outright or the leakage shows up as a
        # failed fill — in both cases the position must not be open.
        assert result["filled"] is False
        assert result.get("reason") == "temporal_leakage"
        assert len(trader.open_positions) == 0


# ---------------------------------------------------------------------------
# Profitability-damage scenarios (silent-loss regression guards)
# ---------------------------------------------------------------------------


class TestProfitabilityGuards:
    """Scenarios that previously cost money or would cost money if logic silently
    regressed. These are blast-radius tests — each one targets a specific way the
    bot can bleed bankroll without any exception being raised."""

    def test_buy_no_cost_uses_one_minus_price(self):
        """BUG C1 regression: buy_no cost must be (1-price)*amount, not price*amount."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="bnc", side="buy_no", price=0.30, amount=100.0,
        ))
        # buy_no at yes_price=0.30 means no-leg = 0.70 → cost = 70.0.
        assert abs(trader.open_positions[0]["cost"] - 70.0) < 1e-9
        assert abs(trader.bankroll - 930.0) < 1e-9

    def test_buy_no_winning_resolution_pays_correctly(self):
        """buy_no + outcome=no → payout = amount, pnl = amount - cost."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="bnw", side="buy_no", price=0.30, amount=100.0,
            platform="polymarket",
        ))
        # cost should be 70, payout should be 100, pnl = 30
        trader.resolve_position("bnw", "no")
        trade = trader.trade_history[0]
        assert trade["status"] == "won"
        assert abs(trade["pnl"] - 30.0) < 1e-9
        assert abs(trader.bankroll - 1030.0) < 1e-9

    def test_losing_trade_does_not_refund_cost(self):
        """A loss must burn the full cost — bankroll stays down."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="loser", side="buy_yes", price=0.40, amount=100.0,
        ))
        assert trader.bankroll == 960.0  # 1000 - 40
        trader.resolve_position("loser", "no")
        # pnl = -cost = -40; bankroll += cost + pnl = 960 + 40 + (-40) = 960
        assert abs(trader.bankroll - 960.0) < 1e-9
        assert trader.trade_history[0]["status"] == "lost"

    def test_kalshi_win_fee_is_applied(self):
        """Kalshi entry fee is booked in cost; win pnl = amount - cost.

        Fee formula is ceil(0.07·p·(1-p)·100)/100 per contract at entry
        (symmetric in p, $0/$1 settlement is fee-free). A silent fallback
        to gross stake would overstate P&L by the entry fee.
        """
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="k1", side="buy_yes", price=0.40, amount=100.0,
            platform="kalshi",
        ))
        # Entry fee = ceil(0.07*0.40*0.60*100)/100 * 100 = 0.02 * 100 = 2.00.
        # Bankroll after entry = 1000 - (40 + 2) = 958.
        assert abs(trader.bankroll - 958.0) < 1e-9
        trader.resolve_position("k1", "yes")
        # pnl = amount - cost = 100 - 42 = 58; bankroll = 958 + 42 + 58 = 1058.
        trade = trader.trade_history[0]
        assert abs(trade["pnl"] - 58.0) < 1e-3
        assert abs(trader.bankroll - 1058.0) < 1e-3

    def test_polymarket_win_has_no_fee(self):
        """Polymarket has no winnings fee — fee regression would under-pay."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="p1", side="buy_yes", price=0.40, amount=100.0,
            platform="polymarket",
        ))
        trader.resolve_position("p1", "yes")
        trade = trader.trade_history[0]
        # Gross winnings = 60; no fee → pnl = 60.
        assert abs(trade["pnl"] - 60.0) < 1e-9
        assert abs(trader.bankroll - 1060.0) < 1e-9

    def test_kalshi_buy_no_entry_fee_symmetric(self):
        """Kalshi fee is symmetric in price — buy_no at p=0.60 pays the same
        entry fee as buy_yes at p=0.40 (both see fee(0.4)=fee(0.6)=0.02)."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="kno", side="buy_no", price=0.60, amount=100.0,
            platform="kalshi",
        ))
        # stake = (1-0.60)*100 = 40; entry_fee = 0.02*100 = 2; cost = 42.
        assert abs(trader.bankroll - 958.0) < 1e-9
        trader.resolve_position("kno", "no")
        assert abs(trader.trade_history[0]["pnl"] - 58.0) < 1e-3

    def test_kalshi_loss_burns_cost_including_entry_fee(self):
        """Loss must burn the full cost (stake + entry fee), not just stake.
        Skipping this means a losing streak under-reports the real drawdown."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="klos", side="buy_yes", price=0.40, amount=100.0,
            platform="kalshi",
        ))
        trader.resolve_position("klos", "no")
        # pnl = -cost = -42 (full $42 lost, including the $2 entry fee).
        assert abs(trader.trade_history[0]["pnl"] + 42.0) < 1e-3
        assert abs(trader.bankroll - 958.0) < 1e-3

    def test_kalshi_early_close_at_flat_price_pays_two_fees(self):
        """On Kalshi, closing early at entry price is NOT a flat exit.
        Taker fees on entry + exit make the round-trip a net loss."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="krt", side="buy_yes", price=0.40, amount=100.0,
            platform="kalshi",
        ))
        # Close at the same price: expected pnl ≈ -(entry_fee + exit_fee) = -4.
        trader.close_position("krt", current_yes_price=0.40, reason="edge_stop")
        pnl = trader.trade_history[0]["pnl"]
        assert -4.1 < pnl < -3.9, f"expected ~-4 (two fees), got {pnl}"

    def test_polymarket_unchanged_by_kalshi_fee_fix(self):
        """Polymarket entries stay fee-free: cost equals stake."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="p2", side="buy_yes", price=0.40, amount=100.0,
            platform="polymarket",
        ))
        assert abs(trader.bankroll - 960.0) < 1e-9  # no fee deducted

    def test_daily_pnl_booked_to_market_date_not_today(self):
        """PRD M3: P&L must book to market_date, not UTC today.

        Cross-midnight resolution distorts daily circuit breakers if mis-booked.
        """
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="mdpl", side="buy_yes", price=0.40, amount=100.0,
            market_date="2030-05-01",
        ))
        trader.resolve_position("mdpl", "yes")
        assert "2030-05-01" in trader.daily_pnl
        assert trader.daily_pnl["2030-05-01"] > 0.0

    def test_close_position_early_exit_books_pnl_to_market_date(self):
        """PRD M3 consistency: close_position must book to market_date too.

        Without this, an early exit across UTC midnight contaminates the next
        trading day's circuit-breaker budget instead of the day the position
        was actually carried on.
        """
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="ee", side="buy_yes", price=0.40, amount=100.0,
            market_date="2030-05-01",
        ))
        trader.close_position("ee", current_yes_price=0.55, reason="edge_stop")
        assert "2030-05-01" in trader.daily_pnl
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today_utc != "2030-05-01":
            assert today_utc not in trader.daily_pnl or trader.daily_pnl[today_utc] == 0.0

    def test_close_position_matches_all_positions_not_just_first(self):
        """BugFix N8: close_position must close ALL matching positions."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="multi", side="buy_yes", price=0.40, amount=10.0,
        ))
        trader.place_order(Order(
            market_id="multi", side="buy_yes", price=0.45, amount=10.0,
        ))
        assert len(trader.open_positions) == 2
        trader.close_position("multi", current_yes_price=0.50, reason="edge_stop")
        assert len(trader.open_positions) == 0
        assert len(trader.trade_history) == 2

    def test_expired_orders_removed_and_status_cancelled(self):
        """Stale limit orders must be auto-cancelled — a leak here ties up bankroll budgeting."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="stale", side="buy_yes", price=0.40, amount=10.0,
            order_type="limit",
        ))
        # Backdate created_at past the TTL.
        trader.open_orders[0]["created_at"] = (
            datetime.now(timezone.utc) - timedelta(hours=ORDER_TTL_HOURS + 1)
        )
        n = trader.cancel_expired_orders()
        assert n == 1
        assert len(trader.open_orders) == 0
        assert trader.trade_history[-1]["status"] == "cancelled"

    def test_zero_resolved_trades_report_returns_zero(self):
        """Empty trade history → get_performance_report must return a zero report.

        A numpy error here would brick the daily reporting pipeline.
        """
        trader = PaperTrader(initial_bankroll=1000.0)
        report = trader.get_performance_report()
        assert report.total_trades == 0
        assert report.total_pnl == 0.0
        assert report.sharpe_ratio == 0.0

    def test_has_reached_daily_entry_limit_counts_per_market_date(self):
        """Two entries on market_date=D must gate a third entry on the same D."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.place_order(Order(
            market_id="cap", side="buy_yes", price=0.40, amount=10.0,
            market_date="2030-05-01",
        ))
        trader.place_order(Order(
            market_id="cap", side="buy_yes", price=0.40, amount=10.0,
            market_date="2030-05-01",
        ))
        assert trader.has_reached_daily_entry_limit(
            "cap", market_date="2030-05-01",
        ) is True
        # Different date → counter fresh.
        assert trader.has_reached_daily_entry_limit(
            "cap", market_date="2030-05-02",
        ) is False

    def test_resolve_position_on_unknown_market_is_noop(self):
        """resolve_position on a non-existent market must not crash or mutate bankroll."""
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.resolve_position("never_placed", "yes")
        assert trader.bankroll == 1000.0
        assert trader.trade_history == []

    def test_close_position_returns_false_when_no_match(self):
        """close_position on a market with no open position must return False."""
        trader = PaperTrader(initial_bankroll=1000.0)
        assert trader.close_position(
            "nothing", current_yes_price=0.5, reason="edge_stop",
        ) is False
        assert trader.bankroll == 1000.0

    def test_bankroll_conservation_across_round_trip(self):
        """place → resolve must conserve: bankroll_end = bankroll_start + pnl."""
        trader = PaperTrader(initial_bankroll=500.0)
        start = trader.bankroll
        trader.place_order(Order(
            market_id="rt", side="buy_yes", price=0.30, amount=50.0,
        ))
        trader.resolve_position("rt", "yes")
        pnl = trader.trade_history[0]["pnl"]
        assert abs(trader.bankroll - (start + pnl)) < 1e-9
