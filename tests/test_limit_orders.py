"""Tests for passive limit order support in PaperTrader.

Covers:
- Limit order goes to open_orders (not filled immediately)
- check_limit_fills fills when price drops to limit
- check_limit_fills does NOT fill when price stays above limit
- buy_no limit order fill logic
- Expired limit orders still cancelled by cancel_expired_orders
- Market orders still work (backwards compatible)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from weatheredge.execution.paper_trader import Order, PaperTrader, ORDER_TTL_HOURS


class TestLimitOrderNotFilledImmediately:
    def test_limit_order_goes_to_open_orders_not_positions(self):
        """A limit order must NOT be filled immediately — it goes to open_orders."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_yes", price=0.50, amount=100.0, order_type="limit")
        result = trader.place_order(order)

        assert result["filled"] is False
        assert result["pending"] is True
        assert "limit_price" in result
        assert len(trader.open_orders) == 1
        assert len(trader.open_positions) == 0
        # Bankroll must NOT be deducted yet
        assert trader.bankroll == 1000.0

    def test_limit_order_has_correct_limit_price_buy_yes(self):
        """buy_yes limit_price = price - limit_order_offset (default 0.02)."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_yes", price=0.50, amount=100.0, order_type="limit")
        result = trader.place_order(order)

        expected_limit = 0.50 - 0.02  # 0.48
        assert abs(result["limit_price"] - expected_limit) < 1e-9
        assert abs(trader.open_orders[0]["limit_price"] - expected_limit) < 1e-9

    def test_limit_order_status_is_pending(self):
        """Limit order in open_orders has status 'pending'."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m2", side="buy_yes", price=0.40, amount=50.0, order_type="limit")
        trader.place_order(order)

        assert trader.open_orders[0]["status"] == "pending"


class TestCheckLimitFillsBuyYes:
    def test_fills_when_price_drops_to_limit(self):
        """buy_yes order fills when current_price <= limit_price."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_yes", price=0.50, amount=100.0, order_type="limit")
        trader.place_order(order)  # limit_price = 0.48

        # Price drops exactly to limit
        fills = trader.check_limit_fills({"m1": 0.48})

        assert fills == 1
        assert len(trader.open_orders) == 0
        assert len(trader.open_positions) == 1
        # Cost deducted: limit_price * amount = 0.48 * 100 = 48
        assert abs(trader.bankroll - (1000.0 - 0.48 * 100.0)) < 1e-9

    def test_fills_when_price_drops_below_limit(self):
        """buy_yes order fills when current_price < limit_price."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_yes", price=0.50, amount=100.0, order_type="limit")
        trader.place_order(order)  # limit_price = 0.48

        fills = trader.check_limit_fills({"m1": 0.45})

        assert fills == 1
        assert len(trader.open_positions) == 1

    def test_does_not_fill_when_price_above_limit(self):
        """buy_yes order does NOT fill when current_price > limit_price."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_yes", price=0.50, amount=100.0, order_type="limit")
        trader.place_order(order)  # limit_price = 0.48

        fills = trader.check_limit_fills({"m1": 0.55})

        assert fills == 0
        assert len(trader.open_orders) == 1
        assert len(trader.open_positions) == 0
        assert trader.bankroll == 1000.0  # unchanged

    def test_does_not_fill_when_price_just_above_limit(self):
        """buy_yes order stays pending when price is just above limit_price."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_yes", price=0.50, amount=100.0, order_type="limit")
        trader.place_order(order)  # limit_price = 0.48

        fills = trader.check_limit_fills({"m1": 0.49})

        assert fills == 0
        assert len(trader.open_orders) == 1

    def test_missing_market_in_prices_leaves_order_pending(self):
        """If market not in current_prices dict, order stays pending."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_yes", price=0.50, amount=100.0, order_type="limit")
        trader.place_order(order)

        fills = trader.check_limit_fills({"other_market": 0.30})

        assert fills == 0
        assert len(trader.open_orders) == 1

    def test_filled_position_price_is_limit_price(self):
        """Filled position records the limit_price as fill price, not original price."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_yes", price=0.50, amount=100.0, order_type="limit")
        trader.place_order(order)  # limit_price = 0.48

        trader.check_limit_fills({"m1": 0.48})

        pos = trader.open_positions[0]
        assert abs(pos["price"] - 0.48) < 1e-9


class TestCheckLimitFillsBuyNo:
    def test_buy_no_limit_price_calculation(self):
        """buy_no limit_price = (1 - yes_price) - limit_order_offset."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_no", price=0.30, amount=100.0, order_type="limit")
        result = trader.place_order(order)

        # (1 - 0.30) - 0.02 = 0.70 - 0.02 = 0.68
        expected_limit = (1.0 - 0.30) - 0.02
        assert abs(result["limit_price"] - expected_limit) < 1e-9

    def test_buy_no_fills_when_no_price_drops_to_limit(self):
        """buy_no fills when (1 - current_yes) <= limit_price."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_no", price=0.30, amount=100.0, order_type="limit")
        trader.place_order(order)  # limit_price = 0.68

        # (1 - current_yes) <= 0.68 → current_yes >= 0.32
        # Use current_yes = 0.32: (1 - 0.32) = 0.68 <= 0.68 → fill
        fills = trader.check_limit_fills({"m1": 0.32})

        assert fills == 1
        assert len(trader.open_positions) == 1

    def test_buy_no_does_not_fill_when_no_price_above_limit(self):
        """buy_no does NOT fill when (1 - current_yes) > limit_price."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_no", price=0.30, amount=100.0, order_type="limit")
        trader.place_order(order)  # limit_price = 0.68

        # (1 - 0.20) = 0.80 > 0.68 → no fill
        fills = trader.check_limit_fills({"m1": 0.20})

        assert fills == 0
        assert len(trader.open_orders) == 1


class TestExpiredLimitOrders:
    def test_expired_pending_order_cancelled_by_cancel_expired_orders(self):
        """Pending limit orders older than ORDER_TTL_HOURS are cancelled."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_yes", price=0.50, amount=100.0, order_type="limit")
        trader.place_order(order)

        # Manually age the order beyond TTL
        trader.open_orders[0]["created_at"] = (
            datetime.now(timezone.utc) - timedelta(hours=ORDER_TTL_HOURS + 0.1)
        )

        cancelled = trader.cancel_expired_orders()

        assert cancelled == 1
        assert len(trader.open_orders) == 0
        assert len(trader.trade_history) == 1
        assert trader.trade_history[0]["status"] == "cancelled"
        # Bankroll not deducted (order was never filled)
        assert trader.bankroll == 1000.0

    def test_fresh_pending_order_not_cancelled(self):
        """A fresh pending limit order is not cancelled by cancel_expired_orders."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_yes", price=0.50, amount=100.0, order_type="limit")
        trader.place_order(order)

        cancelled = trader.cancel_expired_orders()

        assert cancelled == 0
        assert len(trader.open_orders) == 1


class TestMarketOrderBackwardsCompatibility:
    def test_market_order_still_fills_immediately(self):
        """Default (market) orders fill immediately — backwards compatible."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_yes", price=0.40, amount=100.0)
        result = trader.place_order(order)

        assert result["filled"] is True
        assert len(trader.open_positions) == 1
        assert len(trader.open_orders) == 0
        assert abs(trader.bankroll - (1000.0 - 0.40 * 100.0)) < 1e-9

    def test_explicit_market_order_type_fills_immediately(self):
        """order_type='market' fills immediately."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_yes", price=0.40, amount=100.0, order_type="market")
        result = trader.place_order(order)

        assert result["filled"] is True
        assert len(trader.open_positions) == 1

    def test_check_limit_fills_returns_zero_when_no_pending_orders(self):
        """check_limit_fills returns 0 when there are no pending limit orders."""
        trader = PaperTrader(initial_bankroll=1000.0)
        fills = trader.check_limit_fills({"m1": 0.40})
        assert fills == 0

    def test_market_order_buy_no_cost_still_correct(self):
        """buy_no market order cost = (1 - price) * amount (BugFix C1 still applies)."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="m1", side="buy_no", price=0.30, amount=100.0)
        trader.place_order(order)

        expected_cost = (1.0 - 0.30) * 100.0
        assert abs(trader.bankroll - (1000.0 - expected_cost)) < 1e-9
