"""Tests for CrossPlatformArbitrage.execute_arb()."""

from __future__ import annotations

import pytest

from weatheredge.execution.cross_platform import ArbOpportunity, CrossPlatformArbitrage
from weatheredge.execution.paper_trader import PaperTrader


def _make_opp(direction: str = "buy_poly_sell_kalshi") -> ArbOpportunity:
    return ArbOpportunity(
        city="NYC",
        date="2024-07-01",
        threshold=85.0,
        poly_price=0.40,
        kalshi_price=0.55,
        spread=0.11,
        direction=direction,
    )


class TestExecuteArbPlacesTwoTrades:
    def test_places_two_trades(self):
        arb = CrossPlatformArbitrage()
        trader = PaperTrader(initial_bankroll=1000.0)
        opp = _make_opp("buy_poly_sell_kalshi")

        fills = arb.execute_arb(opp, trader, trader.bankroll)

        assert len(fills) == 2

    def test_fills_are_dicts_with_platform(self):
        arb = CrossPlatformArbitrage()
        trader = PaperTrader(initial_bankroll=1000.0)
        opp = _make_opp("buy_poly_sell_kalshi")

        fills = arb.execute_arb(opp, trader, trader.bankroll)

        platforms = {f["platform"] for f in fills}
        assert platforms == {"polymarket", "kalshi"}


class TestExecuteArbSkipsWhenInsufficientBankroll:
    def test_skips_when_bankroll_too_low(self):
        arb = CrossPlatformArbitrage()
        trader = PaperTrader(initial_bankroll=1000.0)
        opp = _make_opp("buy_poly_sell_kalshi")

        # bankroll=0 cannot cover any cost
        fills = arb.execute_arb(opp, trader, 0.0)

        assert fills == []

    def test_skips_when_paper_trader_bankroll_insufficient(self):
        """paper_trader.place_order rejects the order when its own bankroll is too low."""
        arb = CrossPlatformArbitrage()
        # Give paper_trader almost no money so place_order fails for both legs
        trader = PaperTrader(initial_bankroll=0.01)
        opp = _make_opp("buy_poly_sell_kalshi")

        # bankroll arg is large so execute_arb doesn't early-exit, but trader has no funds
        fills = arb.execute_arb(opp, trader, 1000.0)

        # Both legs are rejected by paper_trader due to insufficient bankroll
        assert len(fills) == 0


class TestExecuteArbCorrectSides:
    def test_buy_poly_sell_kalshi_sides(self):
        """buy_poly_sell_kalshi => buy YES on poly, buy NO on kalshi."""
        arb = CrossPlatformArbitrage()
        trader = PaperTrader(initial_bankroll=1000.0)
        opp = _make_opp("buy_poly_sell_kalshi")

        fills = arb.execute_arb(opp, trader, trader.bankroll)

        poly_fill = next(f for f in fills if f["platform"] == "polymarket")
        kalshi_fill = next(f for f in fills if f["platform"] == "kalshi")

        assert poly_fill["side"] == "buy_yes"
        assert kalshi_fill["side"] == "buy_no"

    def test_buy_kalshi_sell_poly_sides(self):
        """buy_kalshi_sell_poly => buy YES on kalshi, buy NO on poly."""
        arb = CrossPlatformArbitrage()
        trader = PaperTrader(initial_bankroll=1000.0)
        opp = _make_opp("buy_kalshi_sell_poly")

        fills = arb.execute_arb(opp, trader, trader.bankroll)

        poly_fill = next(f for f in fills if f["platform"] == "polymarket")
        kalshi_fill = next(f for f in fills if f["platform"] == "kalshi")

        assert poly_fill["side"] == "buy_no"
        assert kalshi_fill["side"] == "buy_yes"


class TestExecuteArbPositionSizing:
    def test_leg_size_capped_at_50(self):
        """With large bankroll, each leg amount is capped at $50."""
        arb = CrossPlatformArbitrage()
        trader = PaperTrader(initial_bankroll=10_000.0)
        opp = _make_opp("buy_poly_sell_kalshi")

        fills = arb.execute_arb(opp, trader, trader.bankroll)

        for fill in fills:
            assert fill["amount"] == 50.0

    def test_leg_size_is_5pct_of_small_bankroll(self):
        """With small bankroll, each leg amount is 5% of bankroll."""
        arb = CrossPlatformArbitrage()
        bankroll = 200.0
        trader = PaperTrader(initial_bankroll=bankroll)
        opp = _make_opp("buy_poly_sell_kalshi")

        fills = arb.execute_arb(opp, trader, bankroll)

        expected_leg = 0.05 * bankroll  # 10.0, which is < 50
        for fill in fills:
            assert fill["amount"] == pytest.approx(expected_leg)
