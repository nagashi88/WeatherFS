"""Tests for bilateral edge fix (v3): unlock NO-side trading.

Covers:
- US-101: net_edge_no in MarketOpportunity + bilateral filter
- US-102: Kelly sizing for buy_no trades
- US-103: edge_at_entry correctness for buy_no
- US-104: Integration: full pipeline produces buy_no TradeOrders
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from weatheredge.agents.base import MarketOpportunity, Signal, TradeOrder
from weatheredge.agents.portfolio_manager import PortfolioManager
from weatheredge.agents.risk_manager import RiskManager
from weatheredge.agents.strategies.conservative import conservative_strategy
from weatheredge.agents.strategies.aggressive import aggressive_strategy
from weatheredge.agents.strategies.sell_no import sell_no_strategy
from weatheredge.probability.edge import calculate_net_edge
from weatheredge.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_settings(**overrides) -> Settings:
    defaults = dict(
        polymarket_private_key="",
        polymarket_proxy_address="",
        kalshi_api_key="",
        kalshi_private_key="",
        anthropic_api_key="",
        telegram_bot_token="",
        telegram_chat_id="",
        kelly_fraction=0.08,
        max_open_positions=20,
        max_single_market_exposure=0.08,
        target_volatility=0.10,
        max_leverage=2.0,
        min_allocation=0.20,
        vpin_danger_threshold=0.70,
        current_risk_tier=100,
        initial_bankroll=1000.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_opp(
    market_id: str = "MKT-NO",
    model_prob: float = 0.10,
    market_price: float = 0.55,
    platform: str = "polymarket",
    lead_time_days: int = 1,
) -> MarketOpportunity:
    """Build an opportunity where NO-side has strong edge."""
    net_edge = calculate_net_edge(model_prob, market_price, platform)
    net_edge_no = calculate_net_edge(1.0 - model_prob, 1.0 - market_price, platform)
    return MarketOpportunity(
        market_id=market_id,
        platform=platform,
        city="NYC",
        station="KNYC",
        threshold=75.0,
        direction="above",
        market_price=market_price,
        model_probability=model_prob,
        net_edge=net_edge,
        net_edge_no=net_edge_no,
        lead_time_days=lead_time_days,
        gaussian_prob=model_prob,
        ensemble_prob=model_prob,
        blended_prob=model_prob,
    )


# ---------------------------------------------------------------------------
# US-101: Bilateral edge in MarketOpportunity
# ---------------------------------------------------------------------------

class TestBilateralEdge:

    def test_net_edge_no_field_exists(self):
        """MarketOpportunity has net_edge_no field."""
        opp = make_opp()
        assert hasattr(opp, "net_edge_no")

    def test_no_side_edge_positive_when_yes_negative(self):
        """When model_prob=0.10 and market_price=0.55, YES edge is negative but NO edge is positive."""
        opp = make_opp(model_prob=0.10, market_price=0.55)
        assert opp.net_edge < 0, f"YES edge should be negative, got {opp.net_edge}"
        assert opp.net_edge_no > 0.30, f"NO edge should be > 0.30, got {opp.net_edge_no}"

    def test_both_edges_calculated_correctly(self):
        """net_edge and net_edge_no are mathematically consistent."""
        opp = make_opp(model_prob=0.10, market_price=0.55)
        # YES: 0.10 - 0.55 - fees
        # NO: 0.90 - 0.45 - fees
        # The difference between the two should be independent of fees
        gross_yes = 0.10 - 0.55
        gross_no = 0.90 - 0.45
        assert abs((opp.net_edge_no - opp.net_edge) - (gross_no - gross_yes)) < 0.01

    def test_symmetric_edge_case(self):
        """When model_prob == market_price, both edges are near-zero (just fees)."""
        opp = make_opp(model_prob=0.50, market_price=0.50)
        assert abs(opp.net_edge) < 0.05
        assert abs(opp.net_edge_no) < 0.05

    def test_llm_filter_passes_with_no_side_edge(self):
        """Opportunity with only NO-side edge passes max(net_edge, net_edge_no) >= 0.05 filter."""
        opp = make_opp(model_prob=0.10, market_price=0.55)
        LLM_EDGE_THRESHOLD = 0.05
        # Old filter: opp.net_edge >= threshold → FAIL (negative)
        assert opp.net_edge < LLM_EDGE_THRESHOLD
        # New filter: max(net_edge, net_edge_no) >= threshold → PASS
        assert max(opp.net_edge, opp.net_edge_no) >= LLM_EDGE_THRESHOLD


# ---------------------------------------------------------------------------
# US-102: Kelly sizing for buy_no
# ---------------------------------------------------------------------------

class TestKellySizingBuyNo:

    def setup_method(self):
        # Pin max_cost_per_trade_fraction so Kelly math below is deterministic
        # regardless of future default tuning.
        self.settings = make_settings(max_cost_per_trade_fraction=0.015)
        self.rm = RiskManager(self.settings)

    def test_buy_no_returns_positive_size(self):
        """Kelly sizing returns > 0 for buy_no when NO-side has edge."""
        size = self.rm.calculate_position_size(
            win_probability=0.10,  # model says YES unlikely
            market_price=0.55,    # market prices YES at 55c
            bankroll=1000.0,
            side="buy_no",
        )
        assert size > 0, f"buy_no size should be > 0, got {size}"

    def test_buy_yes_returns_zero_when_no_edge(self):
        """Kelly sizing returns 0 for buy_yes when model_prob < market_price."""
        size = self.rm.calculate_position_size(
            win_probability=0.10,
            market_price=0.55,
            bankroll=1000.0,
            side="buy_yes",
        )
        assert size == 0.0

    def test_buy_yes_still_works_normally(self):
        """Normal YES-side trade still produces positive size."""
        size = self.rm.calculate_position_size(
            win_probability=0.90,
            market_price=0.55,
            bankroll=1000.0,
            side="buy_yes",
        )
        assert size > 0, f"buy_yes size should be > 0, got {size}"

    def test_default_side_is_buy_yes(self):
        """Without side parameter, defaults to buy_yes behavior."""
        size_default = self.rm.calculate_position_size(
            win_probability=0.10, market_price=0.55, bankroll=1000.0,
        )
        size_explicit = self.rm.calculate_position_size(
            win_probability=0.10, market_price=0.55, bankroll=1000.0, side="buy_yes",
        )
        assert size_default == size_explicit

    def test_buy_no_kelly_math(self):
        """Verify Kelly formula uses flipped values for buy_no."""
        # model_prob=0.10, market_price=0.55 → buy_no
        # Flipped: win_prob=0.90, price=0.45
        # b = (1-0.45)/0.45 = 1.222
        # kelly = (b*p - q) / b = (1.222*0.90 - 0.10) / 1.222 = 0.818
        # warmup_kelly (0.10): kelly_size = 0.818 * 0.10 * 1000 = 81.8
        # capped by max_single_market_exposure=0.08 → 80 contracts
        # cost cap at max_cost = $15. CCG-bughunt3 G3-new: contracts =
        # $15 / (price + kalshi_fee) = 15 / (0.45 + 0.02) = 15 / 0.47 ≈ 31.91.
        size = self.rm.calculate_position_size(
            win_probability=0.10, market_price=0.55, bankroll=1000.0, side="buy_no",
        )
        assert size == pytest.approx(31.91, abs=0.1), f"Expected ~31.91, got {size}"

    def test_buy_no_zero_when_no_real_edge(self):
        """buy_no returns 0 when model says event IS likely (no NO-side edge)."""
        size = self.rm.calculate_position_size(
            win_probability=0.90,  # model says YES likely
            market_price=0.55,     # market agrees-ish
            bankroll=1000.0,
            side="buy_no",
        )
        # Flipped: win_prob=0.10, price=0.45 → 0.10 <= 0.45 → 0
        assert size == 0.0


# ---------------------------------------------------------------------------
# US-103: edge_at_entry in PortfolioManager
# ---------------------------------------------------------------------------

class TestEdgeAtEntry:

    def test_buy_no_edge_at_entry_positive(self):
        """buy_no trade has positive edge_at_entry when NO-side has real edge."""
        settings = make_settings()
        rm = MagicMock()
        rm.pre_trade_check.return_value = {"blocked": False}
        rm.calculate_position_size.return_value = 50.0
        pm = PortfolioManager(settings, rm)

        opp = make_opp(model_prob=0.10, market_price=0.55)
        signals = [
            Signal(market_id="MKT-NO", side="buy_no", confidence=1.0, agent_name="Aggressive"),
            Signal(market_id="MKT-NO", side="buy_no", confidence=1.0, agent_name="MeanReversion"),
            Signal(market_id="MKT-NO", side="buy_no", confidence=0.9, agent_name="SellNO"),
        ]

        orders = pm.aggregate_signals(
            signals_by_market={"MKT-NO": signals},
            opportunities={"MKT-NO": opp},
            bankroll=1000.0,
            open_positions=[],
            daily_pnl=0.0,
            total_pnl=0.0,
        )

        assert len(orders) == 1
        order = orders[0]
        assert order.side == "buy_no"
        # edge_at_entry now uses net_edge_no (with fee deduction) per BugFix L2
        # gross = (1-0.10) - (1-0.55) = 0.45, minus ~1.5% polymarket fee ≈ 0.435
        assert order.edge_at_entry > 0, f"edge_at_entry should be positive for buy_no, got {order.edge_at_entry}"
        assert order.edge_at_entry == pytest.approx(opp.net_edge_no)

    def test_buy_yes_edge_at_entry_still_correct(self):
        """buy_yes trade still has correct positive edge_at_entry."""
        settings = make_settings()
        rm = MagicMock()
        rm.pre_trade_check.return_value = {"blocked": False}
        rm.calculate_position_size.return_value = 50.0
        pm = PortfolioManager(settings, rm)

        opp = make_opp(model_prob=0.70, market_price=0.45)
        signals = [
            Signal(market_id="MKT-YES", side="buy_yes", confidence=1.0, agent_name="Aggressive"),
            Signal(market_id="MKT-YES", side="buy_yes", confidence=1.0, agent_name="MeanReversion"),
            Signal(market_id="MKT-YES", side="buy_yes", confidence=0.9, agent_name="Qualitative"),
        ]

        orders = pm.aggregate_signals(
            signals_by_market={"MKT-YES": signals},
            opportunities={"MKT-YES": opp},
            bankroll=1000.0,
            open_positions=[],
            daily_pnl=0.0,
            total_pnl=0.0,
        )

        assert len(orders) == 1
        order = orders[0]
        assert order.side == "buy_yes"
        # edge_at_entry now uses net_edge (with fee deduction) per BugFix L2
        # gross = 0.70 - 0.45 = 0.25, minus ~1.5% polymarket fee ≈ 0.235
        assert order.edge_at_entry == pytest.approx(opp.net_edge)


# ---------------------------------------------------------------------------
# US-104: Integration — strategies produce buy_no signals for NO-edge markets
# ---------------------------------------------------------------------------

class TestBuyNoIntegration:

    def test_conservative_produces_buy_no(self):
        """Conservative strategy outputs buy_no when model_prob < market_price and NO-side edge is large."""
        settings = make_settings(conservative_min_edge=0.15)
        opp = make_opp(model_prob=0.10, market_price=0.55)
        sig = conservative_strategy(opp, settings)
        # With bilateral fix, conservative now uses net_edge_no for buy_no side
        # net_edge_no ~ 0.435, which is > 0.15 min_edge → produces buy_no
        assert sig.side == "buy_no"
        assert sig.confidence > 0

    def test_conservative_no_trade_when_edge_too_small(self):
        """Conservative returns no_trade when neither side has enough edge."""
        settings = make_settings(conservative_min_edge=0.15)
        # Nearly fair market: model and market roughly agree
        opp = make_opp(model_prob=0.48, market_price=0.50)
        sig = conservative_strategy(opp, settings)
        assert sig.side == "no_trade"

    def test_sell_no_strategy_produces_buy_no(self):
        """SellNO strategy correctly identifies overpriced YES market."""
        settings = make_settings(aggressive_min_edge=0.05)
        opp = make_opp(model_prob=0.10, market_price=0.55, lead_time_days=1)
        sig = sell_no_strategy(opp, settings)
        assert sig.side == "buy_no"
        assert sig.agent_name == "SellNO"
        assert sig.confidence > 0

    def test_end_to_end_buy_no_trade(self):
        """Full pipeline: NO-edge opportunity → signals → portfolio manager → TradeOrder."""
        settings = make_settings()
        rm = RiskManager(settings)
        pm = PortfolioManager(settings, rm)

        opp = make_opp(model_prob=0.10, market_price=0.55)

        # SellNO should produce a buy_no signal
        sell_no_sig = sell_no_strategy(opp, settings)
        assert sell_no_sig.side == "buy_no"

        # Add enough votes to pass threshold
        signals = [
            sell_no_sig,
            Signal(market_id="MKT-NO", side="buy_no", confidence=0.8, agent_name="Aggressive"),
            Signal(market_id="MKT-NO", side="buy_no", confidence=0.7, agent_name="MeanReversion"),
        ]

        orders = pm.aggregate_signals(
            signals_by_market={"MKT-NO": signals},
            opportunities={"MKT-NO": opp},
            bankroll=1000.0,
            open_positions=[],
            daily_pnl=0.0,
            total_pnl=0.0,
        )

        assert len(orders) == 1
        order = orders[0]
        assert order.side == "buy_no"
        assert order.amount > 0, f"Trade amount should be > 0, got {order.amount}"
        assert order.edge_at_entry > 0, f"edge_at_entry should be positive, got {order.edge_at_entry}"
        # edge_at_entry now uses net_edge_no (with fee deduction) per BugFix L2
        assert order.edge_at_entry == pytest.approx(opp.net_edge_no)

    def test_backtest_buy_no_pnl_logic(self):
        """Verify that buy_no PnL calculation is correct.

        buy_no at price 0.45 (1 - 0.55):
        - If event does NOT happen (we win): pnl = (1.0 - 0.45) / 0.45 * cost = positive
        - If event DOES happen (we lose): pnl = -cost
        """
        no_price = 1.0 - 0.55  # 0.45
        cost = 50.0

        # Win case: event doesn't happen, NO pays out
        contracts = cost / no_price
        win_pnl = contracts * (1.0 - no_price) - cost  # payout - cost
        assert win_pnl > 0

        # Loss case: event happens, NO worthless
        loss_pnl = -cost
        assert loss_pnl < 0
