"""Tests for PortfolioManager weighted voting (Addendum Gap 7a)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from weatheredge.agents.base import MarketOpportunity, Signal
from weatheredge.agents.portfolio_manager import PortfolioManager
from weatheredge.agents.risk_manager import RiskManager
from weatheredge.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_settings(**overrides) -> Settings:
    """Build a Settings instance with test-friendly defaults (no .env required)."""
    defaults = dict(
        polymarket_private_key="",
        polymarket_proxy_address="",
        kalshi_api_key="",
        kalshi_private_key="",
        anthropic_api_key="",
        telegram_bot_token="",
        telegram_chat_id="",
        kelly_fraction=0.15,
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


def make_opportunity(market_id: str = "MKT-1", city: str = "NYC") -> MarketOpportunity:
    return MarketOpportunity(
        market_id=market_id,
        platform="polymarket",
        city=city,
        station="KNYC",
        threshold=75.0,
        direction="above",
        market_price=0.45,
        model_probability=0.65,
        net_edge=0.18,
    )


def make_pm(rm: MagicMock | None = None) -> PortfolioManager:
    settings = make_settings()
    if rm is None:
        rm = MagicMock()
        rm.pre_trade_check.return_value = {"blocked": False}
        rm.calculate_position_size.return_value = 50.0
    return PortfolioManager(settings, rm)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_weighted_voting_above_threshold():
    """Conservative=no_trade, others vote buy_yes → weighted=0.66 > 0.50 → order generated.

    Conservative (0.30) * 0.0 (no_trade excluded) = 0.00
    Aggressive   (0.30) * 1.0 = 0.30
    MeanReversion(0.20) * 1.0 = 0.20
    Qualitative  (0.20) * 0.8 = 0.16
    Total = 0.66
    """
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 50.0
    pm = make_pm(rm)

    signals = [
        Signal(market_id="MKT-1", side="no_trade",  confidence=1.0, agent_name="Conservative"),
        Signal(market_id="MKT-1", side="buy_yes",   confidence=1.0, agent_name="Aggressive"),
        Signal(market_id="MKT-1", side="buy_yes",   confidence=1.0, agent_name="MeanReversion"),
        Signal(market_id="MKT-1", side="buy_yes",   confidence=0.8, agent_name="Qualitative"),
    ]
    opp = make_opportunity("MKT-1")
    # Voting-logic test — mark as bucket market so the YES calibration haircut
    # (threshold-markets only) does not alter the asserted size.
    opp.bucket_low = 72.0
    opp.bucket_high = 78.0

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-1": signals},
        opportunities={"MKT-1": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 1
    assert orders[0].market_id == "MKT-1"
    assert orders[0].side == "buy_yes"
    assert orders[0].amount == 50.0


def test_weighted_voting_below_threshold():
    """Only Conservative=buy_yes(0.5) → weighted=0.25*0.5=0.125 < 0.20 → no order."""
    rm = MagicMock()
    pm = make_pm(rm)

    signals = [
        Signal(market_id="MKT-2", side="buy_yes", confidence=0.5, agent_name="Conservative"),
    ]
    opp = make_opportunity("MKT-2")

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-2": signals},
        opportunities={"MKT-2": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert orders == []
    rm.pre_trade_check.assert_not_called()


def test_opposing_weak_votes_do_not_inflate_past_threshold():
    """Regression for CCG-bughunt2 F-new-7: the gate must use the MAJORITY
    side's score, not sum(side_scores.values()). Two weak opposing votes
    (neither side reaching TRADE_THRESHOLD=0.25) used to clear the gate
    because their combined total exceeded threshold, trading on the winning
    side even though that winning side had sub-threshold conviction.

    Setup: Conservative(0.30) × 0.5 = 0.15 for buy_yes,
           Aggressive(0.25)   × 0.5 = 0.125 for buy_no.
    sum = 0.275 > 0.25 (old gate would pass); majority_side_score = 0.15 <
    0.25 (new gate correctly skips).
    """
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 50.0
    pm = make_pm(rm)

    signals = [
        Signal(market_id="MKT-WEAK", side="buy_yes", confidence=0.5, agent_name="Conservative"),
        Signal(market_id="MKT-WEAK", side="buy_no",  confidence=0.5, agent_name="Aggressive"),
    ]
    opp = make_opportunity("MKT-WEAK")

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-WEAK": signals},
        opportunities={"MKT-WEAK": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert orders == [], (
        "Two weak opposing votes (0.15 vs 0.125) must not inflate past "
        "TRADE_THRESHOLD — neither side has conviction"
    )
    rm.pre_trade_check.assert_not_called()


def test_risk_veto_blocks_trade():
    """Weighted score above threshold but risk_manager blocks → no order."""
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": True, "reason": "daily loss limit"}
    rm.calculate_position_size.return_value = 50.0
    pm = make_pm(rm)

    signals = [
        Signal(market_id="MKT-3", side="buy_yes", confidence=1.0, agent_name="Aggressive"),
        Signal(market_id="MKT-3", side="buy_yes", confidence=1.0, agent_name="MeanReversion"),
        Signal(market_id="MKT-3", side="buy_yes", confidence=1.0, agent_name="Qualitative"),
    ]
    opp = make_opportunity("MKT-3")

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-3": signals},
        opportunities={"MKT-3": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert orders == []


def test_vpin_reduces_size():
    """risk_manager returns reduce_size=True → order amount is halved."""
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False, "reduce_size": True}
    rm.calculate_position_size.return_value = 50.0
    pm = make_pm(rm)

    signals = [
        Signal(market_id="MKT-4", side="buy_yes", confidence=1.0, agent_name="Aggressive"),
        Signal(market_id="MKT-4", side="buy_yes", confidence=1.0, agent_name="MeanReversion"),
        Signal(market_id="MKT-4", side="buy_yes", confidence=1.0, agent_name="Qualitative"),
    ]
    opp = make_opportunity("MKT-4")
    # VPIN-reduction test — mark as bucket market so the YES calibration haircut
    # (threshold-markets only) does not compound with the VPIN halving.
    opp.bucket_low = 72.0
    opp.bucket_high = 78.0

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-4": signals},
        opportunities={"MKT-4": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 1
    assert orders[0].amount == pytest.approx(25.0)


def test_empty_signals():
    """No signals at all → empty list returned."""
    pm = make_pm()

    orders = pm.aggregate_signals(
        signals_by_market={},
        opportunities={},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert orders == []


def test_multiple_markets():
    """Signals for 2 markets both above threshold → up to 2 orders."""
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 40.0
    pm = make_pm(rm)

    # Both markets have Aggressive + MeanReversion + Qualitative voting → 0.70 each
    def voting_signals(mid: str) -> list[Signal]:
        return [
            Signal(market_id=mid, side="buy_yes", confidence=1.0, agent_name="Aggressive"),
            Signal(market_id=mid, side="buy_yes", confidence=1.0, agent_name="MeanReversion"),
            Signal(market_id=mid, side="buy_yes", confidence=1.0, agent_name="Qualitative"),
        ]

    opp_a = make_opportunity("MKT-A", city="NYC")
    opp_b = make_opportunity("MKT-B", city="Chicago")

    orders = pm.aggregate_signals(
        signals_by_market={
            "MKT-A": voting_signals("MKT-A"),
            "MKT-B": voting_signals("MKT-B"),
        },
        opportunities={"MKT-A": opp_a, "MKT-B": opp_b},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 2
    market_ids = {o.market_id for o in orders}
    assert "MKT-A" in market_ids
    assert "MKT-B" in market_ids


def test_risk_check_uses_cost_and_total_capital_base():
    """pre_trade_check should receive dollar cost (not contracts) and cash+open-cost capital."""
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 80.0
    pm = make_pm(rm)

    market_id = "MKT-NO"
    opp = make_opportunity(market_id=market_id, city="NYC")
    opp.market_price = 0.80
    opp.model_probability = 0.20  # Favors buy_no

    signals = [
        Signal(market_id=market_id, side="buy_no", confidence=1.0, agent_name="Aggressive"),
        Signal(market_id=market_id, side="buy_no", confidence=1.0, agent_name="SellNO"),
    ]

    orders = pm.aggregate_signals(
        signals_by_market={market_id: signals},
        opportunities={market_id: opp},
        bankroll=1000.0,
        open_positions=[{"cost": 120.0, "city": "NYC", "status": "open"}],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 1
    rm.pre_trade_check.assert_called_once()
    call_kwargs = rm.pre_trade_check.call_args.kwargs
    # buy_no cost = (1 - yes_price) * contracts = 0.2 * 80 = 16
    assert call_kwargs["proposed_size"] == pytest.approx(16.0)
    # bankroll passed to risk checks should include open position cost
    assert call_kwargs["bankroll"] == pytest.approx(1120.0)


def test_max_open_positions_respected_with_multiple_orders_same_cycle():
    """Accepted orders in the same cycle must consume projected open slots."""
    settings = make_settings(max_open_positions=1)
    rm = RiskManager(settings)
    pm = PortfolioManager(settings, rm)

    def voting_signals(mid: str) -> list[Signal]:
        return [
            Signal(market_id=mid, side="buy_yes", confidence=1.0, agent_name="Aggressive"),
            Signal(market_id=mid, side="buy_yes", confidence=1.0, agent_name="SellNO"),
        ]

    opp_a = make_opportunity("MKT-A", city="NYC")
    opp_b = make_opportunity("MKT-B", city="Chicago")

    orders = pm.aggregate_signals(
        signals_by_market={
            "MKT-A": voting_signals("MKT-A"),
            "MKT-B": voting_signals("MKT-B"),
        },
        opportunities={"MKT-A": opp_a, "MKT-B": opp_b},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 1


def test_candidates_prioritized_by_score_when_slots_are_limited():
    """When only one slot is available, highest-score candidate should be selected first."""
    settings = make_settings(max_open_positions=1)
    rm = RiskManager(settings)
    pm = PortfolioManager(settings, rm)

    low_score_market = "MKT-LOW"
    high_score_market = "MKT-HIGH"

    low_signals = [
        Signal(market_id=low_score_market, side="buy_no", confidence=0.60, agent_name="SellNO"),  # 0.15
        Signal(market_id=low_score_market, side="buy_no", confidence=0.90, agent_name="Qualitative"),  # 0.135
        Signal(market_id=low_score_market, side="buy_no", confidence=0.30, agent_name="Aggressive"),  # 0.075
    ]  # total 0.36

    high_signals = [
        Signal(market_id=high_score_market, side="buy_yes", confidence=1.00, agent_name="Conservative"),  # 0.30
        Signal(market_id=high_score_market, side="buy_yes", confidence=1.00, agent_name="Aggressive"),    # 0.25
        Signal(market_id=high_score_market, side="buy_yes", confidence=1.00, agent_name="Qualitative"),   # 0.15
    ]  # total 0.70

    opp_low = make_opportunity(market_id=low_score_market, city="NYC")
    opp_low.market_price = 0.70
    opp_low.model_probability = 0.10
    opp_low.net_edge = -0.60
    opp_low.net_edge_no = 0.57

    opp_high = make_opportunity(market_id=high_score_market, city="Chicago")
    opp_high.market_price = 0.20
    opp_high.model_probability = 0.85
    opp_high.net_edge = 0.62
    opp_high.net_edge_no = -0.65
    # Bucket market so the YES guard-rail (threshold-markets only) does not fire
    # here — this test focuses on slot-prioritisation by weighted_score, not on
    # the April-2026 calibration audit.
    opp_high.bucket_low = 70.0
    opp_high.bucket_high = 75.0

    orders = pm.aggregate_signals(
        signals_by_market={
            low_score_market: low_signals,   # Intentionally first insertion order
            high_score_market: high_signals,
        },
        opportunities={
            low_score_market: opp_low,
            high_score_market: opp_high,
        },
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 1
    assert orders[0].market_id == high_score_market
    assert orders[0].side == "buy_yes"


# ---------------------------------------------------------------------------
# YES calibration guard-rail (April-2026 audit)
# ---------------------------------------------------------------------------

def test_yes_guard_rail_blocks_high_prob_yes_on_threshold_market():
    """buy_yes with model_prob > yes_guard_rail_max_prob on a threshold market
    (bucket_low/high None) must be rejected — historical calibration on
    daily_high/daily_low showed 33 YES trades with 3% actual win rate."""
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 50.0
    pm = make_pm(rm)

    signals = [
        Signal(market_id="MKT-Y", side="buy_yes", confidence=1.0, agent_name="Aggressive"),
        Signal(market_id="MKT-Y", side="buy_yes", confidence=1.0, agent_name="MeanReversion"),
        Signal(market_id="MKT-Y", side="buy_yes", confidence=0.9, agent_name="Qualitative"),
    ]
    opp = make_opportunity("MKT-Y")
    opp.model_probability = 0.95  # Above 0.70 default — mirrors the 2011167 trade
    opp.bucket_low = None
    opp.bucket_high = None

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-Y": signals},
        opportunities={"MKT-Y": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert orders == [], "YES guard-rail must block high-prob buy_yes on threshold markets"


def test_yes_guard_rail_allows_buy_no_on_threshold_market():
    """Guard-rail targets YES only — buy_no on same market type must pass through."""
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 50.0
    pm = make_pm(rm)

    signals = [
        Signal(market_id="MKT-N", side="buy_no", confidence=1.0, agent_name="Aggressive"),
        Signal(market_id="MKT-N", side="buy_no", confidence=1.0, agent_name="MeanReversion"),
        Signal(market_id="MKT-N", side="buy_no", confidence=0.9, agent_name="Qualitative"),
    ]
    opp = make_opportunity("MKT-N")
    opp.model_probability = 0.10
    opp.market_price = 0.60
    opp.net_edge = -0.52
    opp.net_edge_no = 0.28
    opp.bucket_low = None
    opp.bucket_high = None

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-N": signals},
        opportunities={"MKT-N": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 1
    assert orders[0].side == "buy_no"


def test_yes_guard_rail_allows_low_prob_buy_yes():
    """Guard-rail only fires above yes_guard_rail_max_prob (0.70 default).
    A buy_yes with model_prob=0.40 must trade normally."""
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 50.0
    pm = make_pm(rm)

    signals = [
        Signal(market_id="MKT-Y2", side="buy_yes", confidence=1.0, agent_name="Aggressive"),
        Signal(market_id="MKT-Y2", side="buy_yes", confidence=1.0, agent_name="MeanReversion"),
        Signal(market_id="MKT-Y2", side="buy_yes", confidence=0.9, agent_name="Qualitative"),
    ]
    opp = make_opportunity("MKT-Y2")
    opp.model_probability = 0.40
    opp.market_price = 0.22
    opp.net_edge = 0.16
    opp.net_edge_no = -0.20

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-Y2": signals},
        opportunities={"MKT-Y2": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 1
    assert orders[0].side == "buy_yes"


def test_yes_guard_rail_disabled_flag_lets_trade_through():
    """When yes_guard_rail_enabled=False, high-prob buy_yes on threshold market passes."""
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 50.0
    settings = make_settings(yes_guard_rail_enabled=False)
    pm = PortfolioManager(settings, rm)

    signals = [
        Signal(market_id="MKT-Y3", side="buy_yes", confidence=1.0, agent_name="Aggressive"),
        Signal(market_id="MKT-Y3", side="buy_yes", confidence=1.0, agent_name="MeanReversion"),
        Signal(market_id="MKT-Y3", side="buy_yes", confidence=0.9, agent_name="Qualitative"),
    ]
    opp = make_opportunity("MKT-Y3")
    opp.model_probability = 0.95

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-Y3": signals},
        opportunities={"MKT-Y3": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 1


def test_yes_guard_rail_does_not_block_bucket_market():
    """Bucket markets (bucket_low/high set) are NOT threshold markets.
    Guard-rail targets threshold markets only — bucket YES must still be evaluated
    on its own merits (bucket-correlation and standard risk checks)."""
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 50.0
    pm = make_pm(rm)

    signals = [
        Signal(market_id="MKT-B", side="buy_yes", confidence=1.0, agent_name="Aggressive"),
        Signal(market_id="MKT-B", side="buy_yes", confidence=1.0, agent_name="MeanReversion"),
        Signal(market_id="MKT-B", side="buy_yes", confidence=0.9, agent_name="Qualitative"),
    ]
    opp = make_opportunity("MKT-B")
    opp.model_probability = 0.85
    opp.bucket_low = 70.0
    opp.bucket_high = 75.0

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-B": signals},
        opportunities={"MKT-B": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 1
    assert orders[0].side == "buy_yes"


# ---------------------------------------------------------------------------
# YES guard-rail: safety net + haircut tier (April-2026 iteration)
# ---------------------------------------------------------------------------

def test_yes_guard_rail_never_blocks_buy_no_even_at_extreme_prob():
    """Regression guard: buy_no with model_probability=0.95 on a threshold
    market must pass through untouched. The guard-rail is side-specific —
    NO trades have calibrated win rates and must never be penalised."""
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 40.0
    pm = make_pm(rm)

    signals = [
        Signal(market_id="MKT-NO-EXTREME", side="buy_no", confidence=1.0, agent_name="Aggressive"),
        Signal(market_id="MKT-NO-EXTREME", side="buy_no", confidence=1.0, agent_name="MeanReversion"),
        Signal(market_id="MKT-NO-EXTREME", side="buy_no", confidence=0.9, agent_name="Qualitative"),
    ]
    opp = make_opportunity("MKT-NO-EXTREME")
    opp.model_probability = 0.95
    opp.market_price = 0.99  # YES=0.99 → NO=0.01, edge for NO when P(YES)=0.95
    opp.net_edge = 0.0
    opp.net_edge_no = 0.04
    opp.bucket_low = None
    opp.bucket_high = None

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-NO-EXTREME": signals},
        opportunities={"MKT-NO-EXTREME": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 1
    assert orders[0].side == "buy_no"
    # Full size — no haircut applied to NO
    assert orders[0].amount == 40.0


def test_yes_guard_rail_haircut_reduces_size_in_gray_zone():
    """buy_yes on threshold market with model_p in (haircut_prob, max_prob]
    (default 0.50 < p ≤ 0.70) must trade but at haircut_multiplier × size."""
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 100.0
    pm = make_pm(rm)

    signals = [
        Signal(market_id="MKT-HAIRCUT", side="buy_yes", confidence=1.0, agent_name="Aggressive"),
        Signal(market_id="MKT-HAIRCUT", side="buy_yes", confidence=1.0, agent_name="MeanReversion"),
        Signal(market_id="MKT-HAIRCUT", side="buy_yes", confidence=0.9, agent_name="Qualitative"),
    ]
    opp = make_opportunity("MKT-HAIRCUT")
    opp.model_probability = 0.60  # In gray zone (0.50, 0.70]
    opp.market_price = 0.40
    opp.net_edge = 0.18
    opp.bucket_low = None
    opp.bucket_high = None

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-HAIRCUT": signals},
        opportunities={"MKT-HAIRCUT": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 1
    assert orders[0].side == "buy_yes"
    # Default haircut multiplier 0.15 → 100.0 × 0.15 = 15.0
    assert orders[0].amount == pytest.approx(15.0, rel=1e-3), (
        f"Expected haircut to reduce 100.0→15.0, got {orders[0].amount}"
    )


def test_yes_guard_rail_haircut_does_not_apply_to_low_prob_yes():
    """buy_yes with model_p ≤ haircut_prob (0.50) must trade at full size —
    the haircut is surgical, not a blanket penalty on all YES."""
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 100.0
    pm = make_pm(rm)

    signals = [
        Signal(market_id="MKT-LOW", side="buy_yes", confidence=1.0, agent_name="Aggressive"),
        Signal(market_id="MKT-LOW", side="buy_yes", confidence=1.0, agent_name="MeanReversion"),
        Signal(market_id="MKT-LOW", side="buy_yes", confidence=0.9, agent_name="Qualitative"),
    ]
    opp = make_opportunity("MKT-LOW")
    opp.model_probability = 0.40  # Below haircut_prob
    opp.market_price = 0.22
    opp.net_edge = 0.16
    opp.bucket_low = None
    opp.bucket_high = None

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-LOW": signals},
        opportunities={"MKT-LOW": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 1
    assert orders[0].amount == 100.0  # No haircut


def test_yes_guard_rail_haircut_does_not_apply_to_bucket_market():
    """Bucket markets are exempt from the YES guard-rail at every tier
    (block AND haircut). A gray-zone YES on a bucket market trades full size."""
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 100.0
    pm = make_pm(rm)

    signals = [
        Signal(market_id="MKT-BUCKET-GRAY", side="buy_yes", confidence=1.0, agent_name="Aggressive"),
        Signal(market_id="MKT-BUCKET-GRAY", side="buy_yes", confidence=1.0, agent_name="MeanReversion"),
        Signal(market_id="MKT-BUCKET-GRAY", side="buy_yes", confidence=0.9, agent_name="Qualitative"),
    ]
    opp = make_opportunity("MKT-BUCKET-GRAY")
    opp.model_probability = 0.60  # Gray zone, but bucket market
    opp.bucket_low = 72.0
    opp.bucket_high = 76.0

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-BUCKET-GRAY": signals},
        opportunities={"MKT-BUCKET-GRAY": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 1
    assert orders[0].amount == 100.0  # No haircut — bucket exempt


def test_yes_guard_rail_disabled_also_skips_haircut():
    """When yes_guard_rail_enabled=False, neither block nor haircut applies.
    A high-prob buy_yes trades full size even in the gray zone."""
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 100.0
    settings = make_settings(yes_guard_rail_enabled=False)
    pm = PortfolioManager(settings, rm)

    signals = [
        Signal(market_id="MKT-OFF", side="buy_yes", confidence=1.0, agent_name="Aggressive"),
        Signal(market_id="MKT-OFF", side="buy_yes", confidence=1.0, agent_name="MeanReversion"),
        Signal(market_id="MKT-OFF", side="buy_yes", confidence=0.9, agent_name="Qualitative"),
    ]
    opp = make_opportunity("MKT-OFF")
    opp.model_probability = 0.60  # Gray zone
    opp.bucket_low = None
    opp.bucket_high = None

    orders = pm.aggregate_signals(
        signals_by_market={"MKT-OFF": signals},
        opportunities={"MKT-OFF": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    assert len(orders) == 1
    assert orders[0].amount == 100.0  # Full size, no haircut


def test_cash_cap_reservation_is_fee_inclusive_on_kalshi():
    """CCG-bughunt3 G4-new regression: cross-cycle cash reservation must include
    Kalshi per-contract fee, otherwise the second candidate gets an inflated
    budget and paper_trader rejects it at fill time (cost = stake + entry_fee).

    Scenario: bankroll=$20, two Kalshi bucket markets at price=$0.25
    (fee=$0.02 ⇒ per_contract_cost=$0.27). RiskManager mocked to return 80
    contracts each. First order is cash-capped to 74.07 contracts (=$20.00
    fee-incl). With fix: remaining_cash drops to $0, second cash-capped to 0
    → skipped, total cost = $20. Without fix: first reserves only $18.52
    (raw 0.25), remaining=$1.48, second cash-capped to 5.48 contracts →
    emitted, total fee-incl cost = $21.48, breaching bankroll.
    """
    from weatheredge.probability.edge import kalshi_taker_fee

    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 80.0  # large; cash cap must bind
    pm = make_pm(rm)

    bankroll = 20.0
    price = 0.25
    fee = kalshi_taker_fee(price)

    signals_by_market: dict[str, list[Signal]] = {}
    opportunities: dict[str, MarketOpportunity] = {}
    for i, (city, lo, hi) in enumerate([("NYC", 70.0, 75.0), ("Chicago", 80.0, 85.0)]):
        mid = f"KALSHI-{i}"
        signals_by_market[mid] = [
            Signal(market_id=mid, side="buy_yes", confidence=1.0, agent_name="Aggressive"),
            Signal(market_id=mid, side="buy_yes", confidence=1.0, agent_name="MeanReversion"),
        ]
        opportunities[mid] = MarketOpportunity(
            market_id=mid,
            platform="kalshi",
            city=city,
            station=f"KC{i}",
            threshold=75.0,
            direction="above",
            market_price=price,
            model_probability=0.40,  # sub-haircut gate so YES guard does not fire; bucket markets are exempt anyway
            net_edge=0.15,
            bucket_low=lo,
            bucket_high=hi,
        )

    orders = pm.aggregate_signals(
        signals_by_market=signals_by_market,
        opportunities=opportunities,
        bankroll=bankroll,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )

    total_fee_incl_cost = sum(o.amount * (o.price + fee) for o in orders)
    assert total_fee_incl_cost <= bankroll + 0.01, (
        f"Total fee-inclusive cost {total_fee_incl_cost:.2f} exceeds bankroll "
        f"{bankroll:.2f} — cash reservation is not fee-aware"
    )
    # With the fix, first order fully consumes the bankroll; second is skipped.
    assert len(orders) == 1, f"Expected second order to be skipped (0-capped), got {len(orders)}"
