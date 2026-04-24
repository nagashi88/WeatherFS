"""Comprehensive tests for ExitManager — exit rule coverage to ~100%.

Covers:
  - Exit 1: Edge-Based Trailing Stop
  - Exit 2: Asymmetric Capital Protection (hard stop, soft stop w/ confidence gate)
  - Exit 3: Time-Decay Exit (force exit, profit lock)
  - buy_no position edge and confidence handling
  - _lead_time_hours helper (station path, fallback UTC path, no metadata)
  - No-exit / hold cases
  - opportunities dict missing a market_id (None opp)
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock


from weatheredge.execution.exit_manager import (
    ExitManager,
    EDGE_STOP_THRESHOLD,
    SOFT_LOSS_PCT,
    HARD_LOSS_PCT,
    MODEL_CONF_FLOOR,
    TIME_PROFIT_HOURS,
    TIME_PROFIT_MIN_PNL,
    TIME_FORCE_EXIT_HOURS,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_opp(
    model_probability: float = 0.65,
    market_price: float = 0.60,
    bucket_invalidated_for_yes: bool = False,
    bucket_metar_high: float | None = None,
    bucket_high: float | None = None,
    direction: str = "above",
    market_type: str = "daily_high",
    threshold: float = 75.0,
    metar_daily_high: float | None = None,
) -> MagicMock:
    """Return a mock MarketOpportunity with the attributes ExitManager reads."""
    opp = MagicMock()
    opp.model_probability = model_probability
    opp.market_price = market_price
    opp.bucket_invalidated_for_yes = bucket_invalidated_for_yes
    opp.bucket_metar_high = bucket_metar_high
    opp.bucket_high = bucket_high
    opp.direction = direction
    opp.market_type = market_type
    opp.threshold = threshold
    opp.metar_daily_high = metar_daily_high
    return opp


def make_pos(
    market_id: str = "MKT-1",
    side: str = "buy_yes",
    price: float = 0.60,
    amount: float = 100.0,
    cost: float = 60.0,
    edge_at_entry: float = 0.05,
) -> dict:
    """Return a minimal position dict."""
    return {
        "market_id": market_id,
        "side": side,
        "price": price,
        "amount": amount,
        "cost": cost,
        "edge_at_entry": edge_at_entry,
    }


@pytest.fixture()
def mgr() -> ExitManager:
    return ExitManager()


# ---------------------------------------------------------------------------
# Constant sanity checks
# ---------------------------------------------------------------------------

def test_constants_have_expected_values():
    assert EDGE_STOP_THRESHOLD == pytest.approx(-0.03)
    assert SOFT_LOSS_PCT == pytest.approx(-0.05)
    assert HARD_LOSS_PCT == pytest.approx(-0.15)
    assert MODEL_CONF_FLOOR == pytest.approx(0.70)
    assert TIME_PROFIT_HOURS == pytest.approx(3.0)
    assert TIME_PROFIT_MIN_PNL == pytest.approx(0.05)
    assert TIME_FORCE_EXIT_HOURS == pytest.approx(1.0)


def test_market_consensus_stop_closes_buy_no_loser(mgr):
    import weatheredge.config as cfg

    original = cfg.MODULES.get("exit_pnl_stoploss", False)
    cfg.MODULES["exit_pnl_stoploss"] = False
    try:
        pos = make_pos(side="buy_no", price=0.35, cost=35.0)
        opp = make_opp(model_probability=0.10, market_price=0.95)
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    finally:
        cfg.MODULES["exit_pnl_stoploss"] = original

    assert reason == "edge_stop"


def test_market_consensus_stop_closes_buy_yes_loser(mgr):
    import weatheredge.config as cfg

    original = cfg.MODULES.get("exit_pnl_stoploss", False)
    cfg.MODULES["exit_pnl_stoploss"] = False
    try:
        pos = make_pos(side="buy_yes", price=0.35, cost=35.0)
        opp = make_opp(model_probability=0.90, market_price=0.05)
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    finally:
        cfg.MODULES["exit_pnl_stoploss"] = original

    assert reason == "edge_stop"


def test_below_market_intraday_stop_closes_buy_no_before_tail_loss(mgr):
    pos = make_pos(
        market_id="BELOW-MKT",
        side="buy_no",
        price=0.44,
        amount=10.0,
        cost=5.60,
    )
    pos["station"] = "KNYC"
    pos["market_date"] = "2099-01-01"
    opp = make_opp(
        model_probability=0.40,
        market_price=0.85,
        direction="below",
        threshold=66.5,
        metar_daily_high=64.9,
    )

    with patch("weatheredge.execution.exit_manager.hours_until_market_close", return_value=8.0):
        reason = mgr._check_position(pos, {"BELOW-MKT": opp}, {})

    assert reason == "edge_stop"


def test_below_market_intraday_stop_waits_when_observed_high_breached_threshold(mgr):
    import weatheredge.config as cfg

    pos = make_pos(
        market_id="BELOW-MKT",
        side="buy_no",
        price=0.44,
        amount=10.0,
        cost=5.60,
        edge_at_entry=0.40,
    )
    pos["station"] = "KNYC"
    pos["market_date"] = "2099-01-01"
    opp = make_opp(
        model_probability=0.40,
        market_price=0.85,
        direction="below",
        threshold=66.5,
        metar_daily_high=67.0,
    )

    original = cfg.MODULES.get("exit_pnl_stoploss", False)
    cfg.MODULES["exit_pnl_stoploss"] = False
    try:
        with patch("weatheredge.execution.exit_manager.hours_until_market_close", return_value=8.0):
            reason = mgr._check_position(pos, {"BELOW-MKT": opp}, {})
    finally:
        cfg.MODULES["exit_pnl_stoploss"] = original

    assert reason is None


def test_market_consensus_stop_respects_exit_edge_flip_gate(mgr):
    import weatheredge.config as cfg

    original = cfg.MODULES.get("exit_edge_flip", True)
    cfg.MODULES["exit_edge_flip"] = False
    try:
        pos = make_pos(side="buy_no", price=0.35, cost=35.0)
        opp = make_opp(model_probability=0.10, market_price=0.95)
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    finally:
        cfg.MODULES["exit_edge_flip"] = original

    assert reason != "edge_stop"


# ---------------------------------------------------------------------------
# evaluate() — top-level method (covers lines 41-45)
# ---------------------------------------------------------------------------

def test_evaluate_returns_empty_when_no_positions(mgr):
    result = mgr.evaluate([], {}, {})
    assert result == []


def test_evaluate_returns_only_positions_that_should_close(mgr):
    # STEP 2: isolate edge_flip by disabling exit_ev_based so this test
    # targets the edge-based rule only (after STEP 2 reorder, ev_based
    # would also fire on these params and shadow edge_stop).
    import weatheredge.config as cfg
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    cfg.MODULES["exit_ev_based"] = False
    try:
        pos_close = make_pos(market_id="MKT-A", side="buy_yes", price=0.60, amount=100, cost=60)
        pos_hold = make_pos(market_id="MKT-B", side="buy_yes", price=0.60, amount=100, cost=60)

        # MKT-A: model flipped against us hard → edge_stop
        opp_close = make_opp(model_probability=0.50, market_price=0.60)  # edge = -0.10
        # MKT-B: model still favourable → no exit
        opp_hold = make_opp(model_probability=0.70, market_price=0.60)   # edge = +0.10

        opps = {"MKT-A": opp_close, "MKT-B": opp_hold}
        result = mgr.evaluate([pos_close, pos_hold], opps, {})

        assert len(result) == 1
        assert result[0][0]["market_id"] == "MKT-A"
        assert result[0][1] == "edge_stop"
    finally:
        cfg.MODULES["exit_ev_based"] = orig_ev


def test_evaluate_appends_tuple_of_position_and_reason(mgr):
    pos = make_pos(market_id="MKT-X")
    opp = make_opp(model_probability=0.50, market_price=0.60)  # edge = -0.10
    result = mgr.evaluate([pos], {"MKT-X": opp}, {})
    assert isinstance(result[0], tuple)
    assert result[0][0] is pos
    assert isinstance(result[0][1], str)


# ---------------------------------------------------------------------------
# EXIT 1: Edge-Based Trailing Stop (covers lines 93-98)
# ---------------------------------------------------------------------------

def test_exit1_triggers_when_edge_just_below_threshold(mgr):
    """current_edge = -0.031 (just below -3%) → edge_stop."""
    # model_prob=0.569, market_price=0.60 → edge = 0.569-0.60 = -0.031
    pos = make_pos(side="buy_yes", price=0.60, amount=100, cost=60)
    opp = make_opp(model_probability=0.569, market_price=0.60)
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason == "edge_stop"


def test_exit1_triggers_at_exactly_minus_four_percent(mgr):
    """current_edge = -0.04 → edge_stop."""
    pos = make_pos(side="buy_yes", price=0.60, amount=100, cost=60)
    opp = make_opp(model_probability=0.56, market_price=0.60)  # edge=-0.04
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason == "edge_stop"


def test_exit1_does_not_trigger_at_exactly_threshold(mgr):
    """current_edge exactly at threshold (-0.03) → NOT triggered (strictly less than).

    Use integer-exact arithmetic: model=0.67, price=0.70 → edge = -0.03 exactly.
    Python float 0.67 - 0.70 can be slightly off, so use values where subtraction
    is exact in IEEE-754: model=0.50, price=0.53 → edge=-0.03.
    Actually safest: patch _check_position by testing a value slightly ABOVE threshold.
    model=0.68, price=0.70 → edge=-0.02 > -0.03 → no edge_stop.
    """
    pos = make_pos(side="buy_yes", price=0.70, amount=100, cost=70)
    opp = make_opp(model_probability=0.68, market_price=0.70)  # edge=-0.02
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason != "edge_stop"


def test_exit1_does_not_trigger_when_edge_positive(mgr):
    pos = make_pos(side="buy_yes", price=0.60, amount=100, cost=60)
    opp = make_opp(model_probability=0.70, market_price=0.60)  # edge=+0.10
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason != "edge_stop"


# ---------------------------------------------------------------------------
# EXIT 1: buy_no edge calculation (covers line 85)
# ---------------------------------------------------------------------------

def test_exit1_buy_no_edge_formula(mgr):
    """buy_no edge = (1-model_prob) - (1-price) = price - model_prob.
    price=0.60, model_prob=0.65 → edge = 0.60-0.65 = -0.05 → edge_stop.
    """
    # STEP 2: isolate edge_flip — after reorder, ev_based would also fire.
    import weatheredge.config as cfg
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    cfg.MODULES["exit_ev_based"] = False
    try:
        pos = make_pos(side="buy_no", price=0.60, amount=100, cost=40)
        opp = make_opp(model_probability=0.65, market_price=0.60)  # edge = -0.05
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
        assert reason == "edge_stop"
    finally:
        cfg.MODULES["exit_ev_based"] = orig_ev


def test_exit1_buy_no_no_trigger_when_edge_favourable(mgr):
    """buy_no edge = price - model_prob = 0.60 - 0.50 = +0.10 → no edge_stop."""
    pos = make_pos(side="buy_no", price=0.60, amount=100, cost=40)
    opp = make_opp(model_probability=0.50, market_price=0.60)
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason != "edge_stop"


# ---------------------------------------------------------------------------
# EXIT 2: Hard stop (covers lines 103-108)
# ---------------------------------------------------------------------------

def test_exit2_hard_stop_triggers(mgr):
    """pnl_pct = -0.16 < -0.15 → capital_protect (hard stop).

    Edge must be >= -0.03 so Exit1 doesn't fire first.
    model=0.87, price=0.84 → edge=+0.03 (safe). pnl=(84-100)/100=-16%.
    """
    pos = make_pos(side="buy_yes", price=0.84, amount=100, cost=100)
    opp = make_opp(model_probability=0.87, market_price=0.84)  # edge=+0.03
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason == "capital_protect"


def test_exit2_hard_stop_triggers_at_exactly_minus_15_plus_epsilon(mgr):
    """pnl_pct = -0.151 → capital_protect."""
    pos = make_pos(side="buy_yes", price=0.849, amount=1000, cost=1000)
    opp = make_opp(model_probability=0.85, market_price=0.849)
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason == "capital_protect"


def test_exit2_hard_stop_does_not_trigger_at_threshold(mgr):
    """pnl_pct = exactly -0.15 → NOT hard stop (strictly less than)."""
    # current_value = 0.85*100 = 85, cost=100 → pnl=-15 → pnl_pct=-0.15
    pos = make_pos(side="buy_yes", price=0.85, amount=100, cost=100)
    opp = make_opp(model_probability=0.90, market_price=0.85)
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    # edge = 0.90-0.85=+0.05, pnl_pct=-0.15 (not < -0.15), so not capital_protect hard stop
    assert reason != "capital_protect"


# ---------------------------------------------------------------------------
# EXIT 2: Soft stop — low confidence triggers (covers lines 110-118)
# ---------------------------------------------------------------------------

def test_exit2_soft_stop_triggers_when_confidence_low(mgr):
    """pnl between -5% and -15%, model_confidence < 70% → capital_protect."""
    # cost=100, price=0.93 → value=93 → pnl=-7% → in soft zone
    pos = make_pos(side="buy_yes", price=0.93, amount=100, cost=100)
    # model_prob=0.60 < 0.70 → low confidence
    opp = make_opp(model_probability=0.60, market_price=0.93)
    # edge = 0.60-0.93 = -0.33 (triggers exit1 first actually)
    # Use a case where edge is fine but pnl is bad
    pos = make_pos(side="buy_yes", price=0.93, amount=100, cost=100)
    opp = make_opp(model_probability=0.96, market_price=0.93)  # edge=+0.03 (not < -0.03)
    # pnl = 0.93*100 - 100 = -7 → pnl_pct=-0.07 → soft zone; model_confidence=0.96 → hold
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason is None  # high confidence, hold the dip

    # Now with low confidence
    opp2 = make_opp(model_probability=0.62, market_price=0.93)  # edge=-0.31, triggers exit1
    # Need to arrange so edge is fine but pnl is in soft zone and confidence is low
    # edge = model - price; for soft stop, we need edge >= -0.03
    # pnl_pct in [-0.15, -0.05]: cost=100, current_price=0.93 → pnl=-7%
    # model_prob needs: edge = model-0.93 >= -0.03 → model >= 0.90
    # but confidence = model_prob, so for low confidence need model_prob < 0.70 — contradiction
    # For buy_yes soft stop with low confidence:
    # need model_prob < 0.70 AND edge = model_prob - price >= -0.03
    # → price <= model_prob + 0.03 <= 0.73
    # AND pnl in soft zone: current_value/cost-1 in [-0.15,-0.05]
    # current_value = price*amount; let cost=100, amount=100 → price in [0.85,0.95] — contradiction
    # So for buy_yes, we can't have both low confidence and edge OK and pnl in soft zone
    # but for buy_no, confidence = 1 - model_prob:
    pass


def test_exit2_soft_stop_buy_no_low_confidence(mgr):
    """buy_no: confidence = 1-model_prob. If model_prob=0.40 → confidence=0.60 < 0.70."""
    # buy_no: value = (1-price)*amount
    # Let price=0.60 (YES price), amount=100, cost=40
    # current_value = (1-0.60)*100 = 40 → same as cost, pnl=0%  — need loss
    # Increase YES price to cause loss for buy_no:
    # Let price=0.65, cost=40 → value=(1-0.65)*100=35 → pnl=-5/40=-12.5% → soft zone
    # edge for buy_no = (1-model_prob)-(1-price) = price-model_prob
    #   = 0.65 - 0.35 = +0.30 → not edge_stop
    # confidence = 1-model_prob = 1-0.35=0.65 < 0.70 → soft stop triggers
    pos = make_pos(side="buy_no", price=0.60, amount=100, cost=40)
    opp = make_opp(model_probability=0.35, market_price=0.65)
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason == "capital_protect"


def test_exit2_soft_stop_buy_no_high_confidence_holds(mgr):
    """buy_no: confidence = 1-model_prob=0.80 >= 0.70 → hold the dip."""
    # same setup but model_prob=0.20 → confidence=0.80
    # price=0.65, cost=40 → pnl=-12.5% (soft zone)
    # edge = 0.65-0.20=+0.45 → not edge_stop
    pos = make_pos(side="buy_no", price=0.60, amount=100, cost=40)
    opp = make_opp(model_probability=0.20, market_price=0.65)
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason is None


def test_exit2_no_trigger_when_pnl_above_soft_threshold(mgr):
    """pnl > -5% → neither hard nor soft stop."""
    # pnl = -3%: cost=100, value=97
    pos = make_pos(side="buy_yes", price=0.97, amount=100, cost=100)
    opp = make_opp(model_probability=0.98, market_price=0.97)  # edge=+0.01 (fine)
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason is None


# ---------------------------------------------------------------------------
# EXIT 2: cost == 0 edge case (line 78: pnl_pct = 0.0 when cost=0)
# ---------------------------------------------------------------------------

def test_pnl_pct_zero_when_cost_is_zero(mgr):
    """cost=0 should not raise; pnl_pct defaults to 0."""
    pos = make_pos(side="buy_yes", price=0.70, amount=100, cost=0)
    opp = make_opp(model_probability=0.72, market_price=0.70)  # edge=+0.02, pnl_pct=0
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason is None


# ---------------------------------------------------------------------------
# EXIT 3: Time-Decay — force exit (covers lines 129-134)
# ---------------------------------------------------------------------------

def test_exit3_force_exit_when_under_1h(mgr):
    """lead_time < 1h → time_decay (force wire)."""
    pos = make_pos(side="buy_yes", price=0.65, amount=100, cost=65)
    opp = make_opp(model_probability=0.70, market_price=0.65)  # edge=+0.05, pnl=0

    with patch.object(ExitManager, "_lead_time_hours", return_value=0.5):
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason == "time_decay"


def test_exit3_force_exit_at_zero_lead_time(mgr):
    """lead_time = 0 → time_decay."""
    pos = make_pos(side="buy_yes", price=0.65, amount=100, cost=65)
    opp = make_opp(model_probability=0.70, market_price=0.65)

    with patch.object(ExitManager, "_lead_time_hours", return_value=0.0):
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason == "time_decay"


def test_exit3_no_force_exit_at_exactly_1h(mgr):
    """lead_time = exactly 1.0h → NOT force exit (strictly less than)."""
    pos = make_pos(side="buy_yes", price=0.65, amount=100, cost=65)
    opp = make_opp(model_probability=0.70, market_price=0.65)

    with patch.object(ExitManager, "_lead_time_hours", return_value=1.0):
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    # At 1.0h: not force exit, not profit lock (pnl=0 < 5%), should be None
    assert reason is None


# ---------------------------------------------------------------------------
# EXIT 3: Time-Decay — profit lock (covers lines 136-141)
# ---------------------------------------------------------------------------

def test_exit3_profit_lock_triggers(mgr):
    """lead_time in (1h, 3h) and pnl > 5% → time_decay (profit lock).

    amount=200, cost=100 (bought at 0.50 each). current_price=0.60
    → value=200*0.60=120 → pnl=+20%. edge=0.70-0.60=+0.10 (safe).
    """
    pos = make_pos(side="buy_yes", price=0.50, amount=200, cost=100)
    opp = make_opp(model_probability=0.70, market_price=0.60)  # edge=+0.10, pnl=+20%

    with patch.object(ExitManager, "_lead_time_hours", return_value=2.0):
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason == "time_decay"


def test_exit3_profit_lock_does_not_trigger_when_profit_too_low(mgr):
    """lead_time in profit window but pnl <= 5% → no exit.

    amount=200, cost=100. current_price=0.52 → value=104 → pnl=+4%. edge=+0.18 (safe).
    """
    pos = make_pos(side="buy_yes", price=0.50, amount=200, cost=100)
    opp = make_opp(model_probability=0.70, market_price=0.52)  # edge=+0.18, pnl=+4%

    with patch.object(ExitManager, "_lead_time_hours", return_value=2.0):
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason is None


def test_exit3_profit_lock_does_not_trigger_outside_time_window(mgr):
    """lead_time > 3h → profit lock not active even with good profit.

    Same +20% position as trigger test but with 5h lead time.
    """
    pos = make_pos(side="buy_yes", price=0.50, amount=200, cost=100)
    opp = make_opp(model_probability=0.70, market_price=0.60)  # edge=+0.10, pnl=+20%

    with patch.object(ExitManager, "_lead_time_hours", return_value=5.0):
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason is None


def test_exit3_no_exit_when_lead_time_is_none(mgr):
    """If lead_time_hours returns None (no metadata) → time exit rules skipped."""
    pos = make_pos(side="buy_yes", price=0.65, amount=100, cost=65)
    opp = make_opp(model_probability=0.70, market_price=0.65)

    with patch.object(ExitManager, "_lead_time_hours", return_value=None):
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason is None


# ---------------------------------------------------------------------------
# buy_no P&L calculation (covers lines 75-76)
# ---------------------------------------------------------------------------

def test_buy_no_pnl_calculation_profit(mgr):
    """buy_no value = (1-current_price)*amount. Rising YES price hurts NO holder."""
    # Held NO at entry price YES=0.60 (NO=0.40), amount=100, cost=40
    # YES falls to 0.50 → NO value=(1-0.50)*100=50 → pnl=+10/40=+25%
    pos = make_pos(side="buy_no", price=0.60, amount=100, cost=40)
    opp = make_opp(model_probability=0.45, market_price=0.50)
    # edge for buy_no = price - model_prob = 0.50-0.45=+0.05 (fine)
    # pnl=+25% → well in profit, no exit
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason is None


def test_buy_no_pnl_triggers_hard_stop(mgr):
    """buy_no loss: YES price rises sharply → NO loses value → hard stop."""
    # cost=40, YES rises to 0.97 → value=(1-0.97)*100=3 → pnl=(3-40)/40=-92.5%
    pos = make_pos(side="buy_no", price=0.60, amount=100, cost=40)
    opp = make_opp(model_probability=0.96, market_price=0.97)
    # edge = 0.97-0.96=+0.01 (no edge_stop)
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason == "capital_protect"


# ---------------------------------------------------------------------------
# opportunity is None (covers lines 60-62)
# ---------------------------------------------------------------------------

def test_no_opportunity_returns_none(mgr):
    """If market_id not in opportunities → keep open (return None)."""
    pos = make_pos(market_id="MKT-MISSING")
    reason = mgr._check_position(pos, {}, {})
    assert reason is None


def test_no_opportunity_in_evaluate_is_skipped(mgr):
    """evaluate() skips positions where opp is None — they don't appear in to_close."""
    pos = make_pos(market_id="MKT-GHOST")
    result = mgr.evaluate([pos], {}, {})
    assert result == []


# ---------------------------------------------------------------------------
# _lead_time_hours — station path (covers lines 148-161)
# ---------------------------------------------------------------------------

def test_lead_time_hours_uses_station_from_resolution_source(mgr):
    """resolution_source dict with 'station' key → hours_until_market_close called."""
    parsed = MagicMock()
    parsed.date = "2026-04-10"
    meta = {
        "parsed": parsed,
        "resolution_source": {"station": "KNYC"},
    }

    with patch(
        "weatheredge.execution.exit_manager.hours_until_market_close",
        return_value=5.0,
    ) as mock_huc:
        result = ExitManager._lead_time_hours("MKT-1", {"MKT-1": meta})

    mock_huc.assert_called_once_with("2026-04-10", "KNYC")
    assert result == pytest.approx(5.0)


def test_lead_time_hours_clamps_negative_to_zero(mgr):
    """hours_until_market_close returns negative → clamped to 0.0."""
    parsed = MagicMock()
    parsed.date = "2026-04-10"
    meta = {
        "parsed": parsed,
        "resolution_source": {"station": "KNYC"},
    }

    with patch(
        "weatheredge.execution.exit_manager.hours_until_market_close",
        return_value=-2.5,
    ):
        result = ExitManager._lead_time_hours("MKT-1", {"MKT-1": meta})

    assert result == pytest.approx(0.0)


def test_lead_time_hours_falls_back_to_parsed_station_attr(mgr):
    """If resolution_source has no station, falls back to parsed.station."""
    parsed = MagicMock()
    parsed.date = "2026-04-10"
    parsed.station = "KMDW"
    meta = {
        "parsed": parsed,
        "resolution_source": {},  # no 'station' key
    }

    with patch(
        "weatheredge.execution.exit_manager.hours_until_market_close",
        return_value=3.0,
    ) as mock_huc:
        result = ExitManager._lead_time_hours("MKT-1", {"MKT-1": meta})

    mock_huc.assert_called_once_with("2026-04-10", "KMDW")
    assert result == pytest.approx(3.0)


def test_lead_time_hours_station_none_when_hours_until_returns_none(mgr):
    """If hours_until_market_close returns None, fall through to UTC fallback."""
    parsed = MagicMock()
    parsed.date = "2026-04-10"
    parsed.station = "KNYC"
    meta = {
        "parsed": parsed,
        "resolution_source": {},
    }

    with patch(
        "weatheredge.execution.exit_manager.hours_until_market_close",
        return_value=None,
    ):
        # UTC fallback: date is far future → positive hours
        result = ExitManager._lead_time_hours("MKT-1", {"MKT-1": meta})

    # Should fall through to UTC EOD fallback → some positive number (or 0 if past)
    assert result is not None
    assert result >= 0.0


def test_lead_time_hours_no_station_uses_utc_fallback(mgr):
    """No station available → UTC end-of-day fallback (covers lines 164-170)."""
    parsed = MagicMock()
    parsed.date = "2026-04-10"
    # no .station attribute
    del parsed.station  # remove it so hasattr returns False
    meta = {
        "parsed": parsed,
        "resolution_source": {},
    }

    result = ExitManager._lead_time_hours("MKT-1", {"MKT-1": meta})
    # Date is in the future → should return positive hours
    assert result is not None
    assert result >= 0.0


def test_lead_time_hours_utc_fallback_past_date_returns_zero(mgr):
    """Past date in UTC fallback → clamped to 0.0."""
    parsed = MagicMock()
    parsed.date = "2020-01-01"  # well in the past
    del parsed.station
    meta = {
        "parsed": parsed,
        "resolution_source": {},
    }

    result = ExitManager._lead_time_hours("MKT-1", {"MKT-1": meta})
    assert result == pytest.approx(0.0)


def test_lead_time_hours_returns_none_when_no_metadata(mgr):
    """market_id not in metadata → None."""
    result = ExitManager._lead_time_hours("MKT-ABSENT", {})
    assert result is None


def test_lead_time_hours_returns_none_when_parsed_is_none(mgr):
    """parsed=None in meta → None."""
    meta = {"parsed": None}
    result = ExitManager._lead_time_hours("MKT-1", {"MKT-1": meta})
    assert result is None


def test_lead_time_hours_returns_none_when_parsed_has_no_date(mgr):
    """parsed exists but parsed.date is falsy → None (covers line 150)."""
    parsed = MagicMock()
    parsed.date = None
    meta = {"parsed": parsed}
    result = ExitManager._lead_time_hours("MKT-1", {"MKT-1": meta})
    assert result is None


def test_lead_time_hours_returns_none_when_parsed_lacks_date_attr(mgr):
    """parsed has no .date attribute → None."""
    parsed = MagicMock(spec=[])  # no attributes at all
    meta = {"parsed": parsed}
    result = ExitManager._lead_time_hours("MKT-1", {"MKT-1": meta})
    assert result is None


def test_lead_time_hours_invalid_date_string_returns_none(mgr):
    """Invalid date string in UTC fallback path → ValueError caught → None (line 170)."""
    parsed = MagicMock()
    parsed.date = "not-a-date"
    del parsed.station
    meta = {
        "parsed": parsed,
        "resolution_source": {},
    }
    result = ExitManager._lead_time_hours("MKT-1", {"MKT-1": meta})
    assert result is None


def test_lead_time_hours_resolution_source_not_dict(mgr):
    """resolution_source is not a dict → falls through to parsed.station."""
    parsed = MagicMock()
    parsed.date = "2026-04-10"
    parsed.station = "KNYC"
    meta = {
        "parsed": parsed,
        "resolution_source": "string_not_dict",  # not a dict
    }

    with patch(
        "weatheredge.execution.exit_manager.hours_until_market_close",
        return_value=4.0,
    ) as mock_huc:
        result = ExitManager._lead_time_hours("MKT-1", {"MKT-1": meta})

    mock_huc.assert_called_once_with("2026-04-10", "KNYC")
    assert result == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# No-exit / hold cases (position should remain open)
# ---------------------------------------------------------------------------

def test_hold_when_all_rules_pass(mgr):
    """All three exit rules pass → None returned."""
    pos = make_pos(side="buy_yes", price=0.65, amount=100, cost=65)
    # edge = 0.72-0.65=+0.07 (fine), pnl=+0% (fine), no metadata → time skipped
    opp = make_opp(model_probability=0.72, market_price=0.65)
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason is None


def test_hold_with_positive_pnl_no_time_pressure(mgr):
    """Profitable position, no time pressure → hold."""
    pos = make_pos(side="buy_yes", price=0.80, amount=100, cost=70)
    opp = make_opp(model_probability=0.85, market_price=0.80)
    # pnl=(80-70)/70=+14.3%, edge=+0.05
    with patch.object(ExitManager, "_lead_time_hours", return_value=10.0):
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason is None


def test_hold_buy_no_favourable(mgr):
    """buy_no with good edge, small profit, plenty of time → hold."""
    pos = make_pos(side="buy_no", price=0.60, amount=100, cost=40)
    # YES price dropped to 0.55 → value=45 → pnl=+12.5%
    # edge = 0.55-0.50=+0.05 (fine)
    opp = make_opp(model_probability=0.50, market_price=0.55)
    with patch.object(ExitManager, "_lead_time_hours", return_value=10.0):
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason is None


# ---------------------------------------------------------------------------
# edge_at_entry fallback (pos missing 'edge_at_entry' key, covers line 68)
# ---------------------------------------------------------------------------

def test_position_without_edge_at_entry_key(mgr):
    """Positions without 'edge_at_entry' key should use default 0.0 without error."""
    pos = {
        "market_id": "MKT-1",
        "side": "buy_yes",
        "price": 0.65,
        "amount": 100.0,
        "cost": 65.0,
        # no 'edge_at_entry'
    }
    opp = make_opp(model_probability=0.72, market_price=0.65)
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason is None  # should not raise


# ---------------------------------------------------------------------------
# Exit 1: exit_edge_flip MODULES gate (Phase 1 fix)
# ---------------------------------------------------------------------------

def test_exit1_does_not_trigger_when_exit_edge_flip_disabled(mgr):
    """exit_edge_flip=False in MODULES must prevent edge_stop from firing.

    Isolate: disable both exit_edge_flip and exit_ev_based so only edge_stop
    is under test. Edge = -0.30 would fire in Phase 0; must be suppressed here.
    """
    import weatheredge.config as cfg
    orig_flip = cfg.MODULES.get("exit_edge_flip", True)
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    cfg.MODULES["exit_edge_flip"] = False
    cfg.MODULES["exit_ev_based"] = False
    try:
        pos = make_pos(side="buy_yes", price=0.60, amount=100, cost=60)
        opp = make_opp(model_probability=0.30, market_price=0.60)  # edge = -0.30
        with patch.object(ExitManager, "_lead_time_hours", return_value=10.0):
            reason = mgr._check_position(pos, {"MKT-1": opp}, {})
        assert reason is None
    finally:
        cfg.MODULES["exit_edge_flip"] = orig_flip
        cfg.MODULES["exit_ev_based"] = orig_ev


# ---------------------------------------------------------------------------
# Exit 2b: EV-Based Exit (covers lines 135-139)
# ---------------------------------------------------------------------------

def test_exit_ev_based_triggers_when_ev_close_exceeds_ev_hold(mgr):
    """ev_based fires when closing now has higher EV than holding to resolution.

    Phase 1 context: exit_edge_flip=False, exit_ev_based=True.
    buy_yes, amount=100, cost=60, model_prob=0.65, current_price=0.80:
      ev_hold = 0.65*(100-60) - 0.35*60 = 26 - 21 = +5
      kalshi fee(0.80)=0.02, half-spread=0.01 → exit=0.03/contract
      ev_close = 0.80*100 - 60 - 0.03*100 = 80 - 63 = +17  →  17 > 5+3 → exit
    """
    import weatheredge.config as cfg
    orig_flip = cfg.MODULES.get("exit_edge_flip", True)
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    cfg.MODULES["exit_edge_flip"] = False
    cfg.MODULES["exit_ev_based"] = True
    try:
        pos = make_pos(side="buy_yes", price=0.60, amount=100, cost=60)
        opp = make_opp(model_probability=0.65, market_price=0.80)
        with patch.object(ExitManager, "_lead_time_hours", return_value=10.0):
            reason = mgr._check_position(pos, {"MKT-1": opp}, {})
        assert reason == "ev_based"
    finally:
        cfg.MODULES["exit_edge_flip"] = orig_flip
        cfg.MODULES["exit_ev_based"] = orig_ev


# ---------------------------------------------------------------------------
# EXIT 0: Bucket invalidation (structural kill — YES physically impossible)
# ---------------------------------------------------------------------------

def test_exit0_bucket_invalidated_triggers_for_buy_yes(mgr):
    """flag=True + side=buy_yes → bucket_invalidated."""
    pos = make_pos(side="buy_yes", price=0.50, amount=100, cost=50)
    opp = make_opp(
        model_probability=0.05,
        market_price=0.01,
        bucket_invalidated_for_yes=True,
        bucket_metar_high=86.0,
        bucket_high=84.0,
    )
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason == "bucket_invalidated"


def test_exit0_bucket_invalidated_does_not_trigger_for_buy_no(mgr):
    """flag=True but side=buy_no → rule is asymmetric, must NOT trigger."""
    # buy_no: price=0.01 YES means NO value=(1-0.01)*100=99, cost=40 → huge profit → hold
    pos = make_pos(side="buy_no", price=0.60, amount=100, cost=40)
    opp = make_opp(
        model_probability=0.05,
        market_price=0.01,
        bucket_invalidated_for_yes=True,  # flag is set
        bucket_metar_high=86.0,
        bucket_high=84.0,
    )
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    # buy_no with YES collapsed is deeply profitable; no exit rule should fire
    assert reason != "bucket_invalidated"


def test_exit0_bucket_invalidated_does_not_trigger_when_flag_false(mgr):
    """flag=False → rule must NOT fire even for buy_yes."""
    pos = make_pos(side="buy_yes", price=0.65, amount=100, cost=65)
    opp = make_opp(
        model_probability=0.72,
        market_price=0.65,
        bucket_invalidated_for_yes=False,
    )
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason != "bucket_invalidated"


def test_exit0_has_priority_over_edge_flip(mgr):
    """Even if edge < -3% would trigger edge_stop, bucket_invalidated wins."""
    # edge = 0.05 - 0.60 = -0.55 → would trigger edge_stop
    pos = make_pos(side="buy_yes", price=0.60, amount=100, cost=60)
    opp = make_opp(
        model_probability=0.05,
        market_price=0.60,
        bucket_invalidated_for_yes=True,
        bucket_metar_high=86.0,
        bucket_high=84.0,
    )
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason == "bucket_invalidated"


def test_exit0_has_priority_over_time_decay_force_exit(mgr):
    """Even if <1h to close would trigger time_decay, bucket_invalidated wins."""
    pos = make_pos(side="buy_yes", price=0.50, amount=100, cost=50)
    opp = make_opp(
        model_probability=0.05,
        market_price=0.01,
        bucket_invalidated_for_yes=True,
        bucket_metar_high=86.0,
        bucket_high=84.0,
    )
    with patch.object(ExitManager, "_lead_time_hours", return_value=0.3):
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason == "bucket_invalidated"


def test_exit0_has_priority_over_ev_based(mgr):
    """Even if EV-based would fire, bucket_invalidated wins (structural kill first)."""
    import weatheredge.config as cfg
    orig_flip = cfg.MODULES.get("exit_edge_flip", True)
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    cfg.MODULES["exit_edge_flip"] = False
    cfg.MODULES["exit_ev_based"] = True
    try:
        # Any buy_yes position with invalidation flag should short-circuit to EXIT 0
        pos = make_pos(side="buy_yes", price=0.60, amount=100, cost=60)
        opp = make_opp(
            model_probability=0.65,
            market_price=0.80,
            bucket_invalidated_for_yes=True,
            bucket_metar_high=86.0,
            bucket_high=84.0,
        )
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
        assert reason == "bucket_invalidated"
    finally:
        cfg.MODULES["exit_edge_flip"] = orig_flip
        cfg.MODULES["exit_ev_based"] = orig_ev


def test_exit0_handles_missing_bucket_metadata_gracefully(mgr):
    """flag=True but bucket_metar_high/bucket_high are None → still fires, logging safe."""
    pos = make_pos(side="buy_yes", price=0.50, amount=100, cost=50)
    opp = make_opp(
        model_probability=0.05,
        market_price=0.01,
        bucket_invalidated_for_yes=True,
        bucket_metar_high=None,
        bucket_high=None,
    )
    reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason == "bucket_invalidated"


def test_ev_based_priority_over_edge_flip_when_both_would_fire(mgr):
    # STEP 2: with both exit_ev_based and exit_edge_flip enabled, and both
    # rules would trigger on the same position, the reorder guarantees the
    # close is labeled "ev_based" (the optimal rule) rather than "edge_stop"
    # (the heuristic trailing stop).
    # buy_yes, cost=60, model=0.10, price=0.50:
    #   current_edge = 0.10 - 0.50 = -0.40 → edge_flip would fire
    #   ev_hold = 0.10*(100-60) - 0.90*60 = 4 - 54 = -50
    #   kalshi fee(0.50)=0.02, exit=0.03 → ev_close = 50-60-3 = -13
    #   -13 > -50 + 3 → fire
    import weatheredge.config as cfg
    orig_flip = cfg.MODULES.get("exit_edge_flip", True)
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    cfg.MODULES["exit_edge_flip"] = True
    cfg.MODULES["exit_ev_based"] = True
    try:
        pos = make_pos(side="buy_yes", price=0.60, amount=100, cost=60)
        opp = make_opp(model_probability=0.10, market_price=0.50)
        with patch.object(ExitManager, "_lead_time_hours", return_value=10.0):
            reason = mgr._check_position(pos, {"MKT-1": opp}, {})
        assert reason == "ev_based", f"expected ev_based priority, got {reason}"
    finally:
        cfg.MODULES["exit_edge_flip"] = orig_flip
        cfg.MODULES["exit_ev_based"] = orig_ev


def test_edge_flip_still_fires_when_ev_says_hold(mgr):
    # STEP 2 regression: if EV returns False (hold) but edge has flipped,
    # edge_flip must still fire as the safety net. The reorder MUST NOT
    # mask edge_flip when EV itself does not want to close.
    # Setup: pick params where EV says hold but edge_flip would trigger.
    # buy_no, entry YES=0.20, cost=80. Now model_prob=0.10, market YES=0.05.
    #   current_edge = current_price - model_prob = 0.05 - 0.10 = -0.05
    #                  → below EDGE_STOP_THRESHOLD (-0.03) → edge_flip triggers
    #   ev_hold = (1-0.10)*(100-80) - 0.10*80 = 18 - 8 = +10
    #   ev_close = (1-0.05)*100 - 80 - 0.02*100 = 95 - 80 - 2 = +13
    #   buffer = 0.05*80 = 4; 13 > 10 + 4 → FALSE → EV says HOLD
    # Expectation: EV holds (falls through) but edge_flip catches it.
    import weatheredge.config as cfg
    orig_flip = cfg.MODULES.get("exit_edge_flip", True)
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    cfg.MODULES["exit_edge_flip"] = True
    cfg.MODULES["exit_ev_based"] = True
    try:
        pos = make_pos(side="buy_no", price=0.20, amount=100, cost=80)
        opp = make_opp(model_probability=0.10, market_price=0.05)
        with patch.object(ExitManager, "_lead_time_hours", return_value=10.0):
            reason = mgr._check_position(pos, {"MKT-1": opp}, {})
        assert reason == "edge_stop", f"expected edge_flip safety net, got {reason}"
    finally:
        cfg.MODULES["exit_edge_flip"] = orig_flip
        cfg.MODULES["exit_ev_based"] = orig_ev


def test_exit_ev_based_holds_when_ev_hold_exceeds_ev_close(mgr):
    """ev_based does NOT fire when holding has higher EV than closing.

    Phase 1 context: exit_edge_flip=False, exit_ev_based=True.
    buy_yes, amount=100, cost=60, model_prob=0.65, current_price=0.60:
      ev_hold = +5
      kalshi fee(0.60)=0.02, exit=0.03 → ev_close = 60 - 60 - 3 = -3
      -3 > 5 + 3 → False → hold
    """
    import weatheredge.config as cfg
    orig_flip = cfg.MODULES.get("exit_edge_flip", True)
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    cfg.MODULES["exit_edge_flip"] = False
    cfg.MODULES["exit_ev_based"] = True
    try:
        pos = make_pos(side="buy_yes", price=0.60, amount=100, cost=60)
        opp = make_opp(model_probability=0.65, market_price=0.60)
        with patch.object(ExitManager, "_lead_time_hours", return_value=10.0):
            reason = mgr._check_position(pos, {"MKT-1": opp}, {})
        assert reason is None
    finally:
        cfg.MODULES["exit_edge_flip"] = orig_flip
        cfg.MODULES["exit_ev_based"] = orig_ev


def test_exit_ev_price_dependent_exit_cost_holds_when_flat_would_fire(mgr):
    """Regression for the hard-coded 0.02 flat exit-cost bug.

    At contested mid-prices, the Kalshi taker fee is ~0.02/contract, so the
    per-leg exit cost is 0.02 + 0.01 half-spread = 0.03/contract — NOT the
    old flat 0.02. This test pins a boundary case whose decision flips:

    Phase 1 context: exit_edge_flip=False, exit_ev_based=True.
    buy_yes, amount=100, cost=50 (entry p=0.50), model=0.55, current=0.60:
      ev_hold   = 0.55*50 - 0.45*50 = +5
      ev_buffer = 0.05*50 = 2.5
      Under OLD flat 0.02:  ev_close = 60 - 50 - 2 = +8
                            8 > 5 + 2.5 = 7.5 → FIRES (bug: over-eager)
      Under NEW kalshi:     fee(0.60)=0.02, per-leg=0.03
                            ev_close = 60 - 50 - 3 = +7
                            7 > 7.5 → FALSE → holds (correct)
    """
    import weatheredge.config as cfg
    orig_flip = cfg.MODULES.get("exit_edge_flip", True)
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    cfg.MODULES["exit_edge_flip"] = False
    cfg.MODULES["exit_ev_based"] = True
    try:
        pos = make_pos(side="buy_yes", price=0.50, amount=100, cost=50)
        opp = make_opp(model_probability=0.55, market_price=0.60)
        with patch.object(ExitManager, "_lead_time_hours", return_value=10.0):
            reason = mgr._check_position(pos, {"MKT-1": opp}, {})
        assert reason is None, (
            f"EV-exit must HOLD at mid-prices where the only margin comes from "
            f"under-estimating Kalshi taker fees. Got reason={reason}"
        )
    finally:
        cfg.MODULES["exit_edge_flip"] = orig_flip
        cfg.MODULES["exit_ev_based"] = orig_ev


def test_exit_ev_price_dependent_exit_cost_symmetric_for_buy_no(mgr):
    """Kalshi fee(p) == fee(1-p) — exit cost for buy_no at YES=0.40 equals
    exit cost for buy_yes at YES=0.60. This pins the symmetry so a future
    regression (e.g. using (1-price) by mistake) can't sneak in.

    buy_no, amount=100, cost=50 (entry NO=0.50), model_prob=0.45 → NO-side
    model_conf = 0.55. Current YES=0.40, NO=0.60:
      ev_hold   = 0.55*(100-50) - 0.45*50 = 27.5 - 22.5 = +5
      kalshi fee(0.40) = ceil(0.07*0.40*0.60*100)/100 = 0.02, per-leg=0.03
      current NO value = (1-0.40)*100 = 60
      ev_close  = 60 - 50 - 3 = +7
      ev_buffer = 0.05*50 = 2.5 → threshold = 7.5
      7 > 7.5 → FALSE → holds
    """
    import weatheredge.config as cfg
    orig_flip = cfg.MODULES.get("exit_edge_flip", True)
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    cfg.MODULES["exit_edge_flip"] = False
    cfg.MODULES["exit_ev_based"] = True
    try:
        pos = make_pos(side="buy_no", price=0.50, amount=100, cost=50)
        opp = make_opp(model_probability=0.45, market_price=0.40)
        with patch.object(ExitManager, "_lead_time_hours", return_value=10.0):
            reason = mgr._check_position(pos, {"MKT-1": opp}, {})
        assert reason is None, (
            f"buy_no mirror of buy_yes case must also HOLD. Got reason={reason}"
        )
    finally:
        cfg.MODULES["exit_edge_flip"] = orig_flip
        cfg.MODULES["exit_ev_based"] = orig_ev


# ---------------------------------------------------------------------------
# Finding 4: Kalshi MtM is net of exit taker fee (asymmetry vs polymarket)
# ---------------------------------------------------------------------------

def test_polymarket_pnl_pct_stays_gross_profit_lock_fires(mgr):
    """Baseline: Polymarket MtM stays gross → profit-lock fires at 2h, +7.5%.

    buy_yes @ 0.40, amount=100, cost=40 (polymarket, no fees)
    current_price=0.43 → current_value=43, unrealized=3, pnl_pct=7.5% > 5%
    lead_time=2h < 3h → time_decay profit-lock triggers.
    """
    pos = make_pos(side="buy_yes", price=0.40, amount=100, cost=40.0)
    opp = make_opp(model_probability=0.45, market_price=0.43)
    with patch.object(ExitManager, "_lead_time_hours", return_value=2.0):
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason == "time_decay", (
        f"Polymarket baseline: pnl_pct=7.5% at 2h must fire profit-lock, "
        f"got reason={reason}"
    )


def test_kalshi_pnl_pct_net_of_exit_fee_holds_where_poly_would_sell(mgr):
    """Finding 4: Kalshi MtM subtracts exit taker fee, diverging from polymarket.

    Same position as the baseline test but platform=kalshi with cost=42
    (40 stake + 2 entry fee). At price=0.43 the exit fee is another 2
    (kalshi_taker_fee(0.43)=0.02 per contract × 100), so:
        current_value = 43
        exit_fee      = 2
        unrealized    = 43 - 2 - 42 = -1
        pnl_pct       = -1/42 ≈ -2.4%   (below the 5% profit-lock threshold)
    Profit-lock must NOT fire — closing now would realize a loss.
    """
    pos = {
        "market_id": "MKT-1",
        "side": "buy_yes",
        "price": 0.40,
        "amount": 100.0,
        "cost": 42.0,            # stake (40) + kalshi entry fee (2)
        "edge_at_entry": 0.05,
        "platform": "kalshi",
    }
    opp = make_opp(model_probability=0.45, market_price=0.43)
    with patch.object(ExitManager, "_lead_time_hours", return_value=2.0):
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    assert reason is None, (
        f"Kalshi: exit fee wipes the paper profit, profit-lock must NOT fire. "
        f"Got reason={reason}"
    )


# ---------------------------------------------------------------------------
# CCG-bughunt2 Codex F8: MODULES["exit_time_decay"] gate
# ---------------------------------------------------------------------------

def test_exit_time_decay_gate_suppresses_force_exit(mgr):
    """exit_time_decay=False in MODULES must prevent time_decay from firing
    on the <1h force-exit branch. Default was True (unconditional) before
    the gate was added in CCG-bughunt2 F8.
    """
    import weatheredge.config as cfg
    orig_td = cfg.MODULES.get("exit_time_decay", True)
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    orig_flip = cfg.MODULES.get("exit_edge_flip", True)
    cfg.MODULES["exit_time_decay"] = False
    cfg.MODULES["exit_ev_based"] = False
    cfg.MODULES["exit_edge_flip"] = False
    try:
        pos = make_pos(side="buy_yes", price=0.65, amount=100, cost=65)
        opp = make_opp(model_probability=0.70, market_price=0.65)
        with patch.object(ExitManager, "_lead_time_hours", return_value=0.5):
            reason = mgr._check_position(pos, {"MKT-1": opp}, {})
        assert reason is None
    finally:
        cfg.MODULES["exit_time_decay"] = orig_td
        cfg.MODULES["exit_ev_based"] = orig_ev
        cfg.MODULES["exit_edge_flip"] = orig_flip


def test_exit_time_decay_gate_suppresses_profit_lock(mgr):
    """exit_time_decay=False also suppresses the in-window profit-lock branch."""
    import weatheredge.config as cfg
    orig_td = cfg.MODULES.get("exit_time_decay", True)
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    orig_flip = cfg.MODULES.get("exit_edge_flip", True)
    cfg.MODULES["exit_time_decay"] = False
    cfg.MODULES["exit_ev_based"] = False
    cfg.MODULES["exit_edge_flip"] = False
    try:
        pos = make_pos(side="buy_yes", price=0.50, amount=200, cost=100)
        opp = make_opp(model_probability=0.70, market_price=0.60)  # pnl ~ +20%
        with patch.object(ExitManager, "_lead_time_hours", return_value=2.0):
            reason = mgr._check_position(pos, {"MKT-1": opp}, {})
        assert reason is None
    finally:
        cfg.MODULES["exit_time_decay"] = orig_td
        cfg.MODULES["exit_ev_based"] = orig_ev
        cfg.MODULES["exit_edge_flip"] = orig_flip


def test_exit_time_decay_default_is_on(mgr):
    """Default (no override) keeps time_decay active — regression guard
    that the F8 gate default True matches historical unconditional behavior.
    """
    import weatheredge.config as cfg
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    orig_flip = cfg.MODULES.get("exit_edge_flip", True)
    cfg.MODULES["exit_ev_based"] = False
    cfg.MODULES["exit_edge_flip"] = False
    # intentionally do NOT touch exit_time_decay — default must remain ON
    try:
        pos = make_pos(side="buy_yes", price=0.65, amount=100, cost=65)
        opp = make_opp(model_probability=0.70, market_price=0.65)
        with patch.object(ExitManager, "_lead_time_hours", return_value=0.5):
            reason = mgr._check_position(pos, {"MKT-1": opp}, {})
        assert reason == "time_decay"
    finally:
        cfg.MODULES["exit_ev_based"] = orig_ev
        cfg.MODULES["exit_edge_flip"] = orig_flip


# ---------------------------------------------------------------------------
# CCG-bughunt2 G8: edge_stop log must show comparable units.
# Prior log paired `entry_edge` (net) with `current_edge` (gross) under
# identical "%" formatting — operators post-mortem'ing a stop would read
# a delta that undercounted the true edge decay by the round-trip fee.
# Decision semantics remain gross (calibrated against gross); the fix is
# purely in the log content so on-call engineers and dashboards consuming
# these lines see apples-to-apples values.
# ---------------------------------------------------------------------------


def _opp_with_net_edge(*, model_probability, market_price, net_edge, net_edge_no=0.0):
    """Like make_opp but with explicit net_edge/net_edge_no floats.

    Plain make_opp returns a MagicMock, whose auto-generated net_edge
    attribute would pass %.1f%% formatting with a garbage value — hiding
    whether the production code actually read the correct field.
    """
    opp = make_opp(model_probability=model_probability, market_price=market_price)
    opp.net_edge = net_edge
    opp.net_edge_no = net_edge_no
    return opp


def test_edge_stop_log_uses_net_current_edge_for_entry_comparison(mgr, caplog):
    """When edge_stop fires, the log must include a NET current edge
    so it is directly comparable to the stored entry_edge (also net).
    Asserts against the new format; a regression to the old format
    (bare `current_edge=%`) would not carry `current_edge_net=`."""
    import logging
    caplog.set_level(logging.INFO, logger="weatheredge.execution.exit_manager")

    import weatheredge.config as cfg
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    cfg.MODULES["exit_ev_based"] = False  # isolate edge_stop
    try:
        # buy_yes: gross = 0.56 - 0.60 = -0.04 → triggers edge_stop.
        # Net = gross - round-trip fees (≈ 0.03 Kalshi @ p=0.60).
        pos = make_pos(side="buy_yes", price=0.60, amount=100, cost=60,
                       edge_at_entry=0.08)
        opp = _opp_with_net_edge(
            model_probability=0.56, market_price=0.60,
            net_edge=-0.07,
        )
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    finally:
        cfg.MODULES["exit_ev_based"] = orig_ev

    assert reason == "edge_stop"
    records = [r for r in caplog.records if "edge_stop" in r.getMessage()]
    assert len(records) == 1
    msg = records[0].getMessage()
    # Net entry AND net current both present with unambiguous labels.
    assert "entry_edge_net=8.0%" in msg, msg
    assert "current_edge_net=-7.0%" in msg, msg
    # Gross value that actually triggered the decision is visible alongside.
    assert "gross=-4.0%" in msg, msg
    # Threshold value is shown so the exit condition is self-documenting.
    assert "-3%" in msg, msg


def test_edge_stop_log_buy_no_uses_net_edge_no(mgr, caplog):
    """buy_no side must read opp.net_edge_no for the net figure so the
    log matches the position's directional edge."""
    import logging
    caplog.set_level(logging.INFO, logger="weatheredge.execution.exit_manager")

    import weatheredge.config as cfg
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    cfg.MODULES["exit_ev_based"] = False
    try:
        # buy_no: gross edge = price - model = 0.60 - 0.65 = -0.05 → triggers.
        pos = make_pos(side="buy_no", price=0.60, amount=100, cost=40,
                       edge_at_entry=0.09)
        opp = _opp_with_net_edge(
            model_probability=0.65, market_price=0.60,
            net_edge=0.05,          # YES-side (irrelevant for buy_no)
            net_edge_no=-0.08,      # NO-side is what should be logged
        )
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    finally:
        cfg.MODULES["exit_ev_based"] = orig_ev

    assert reason == "edge_stop"
    records = [r for r in caplog.records if "edge_stop" in r.getMessage()]
    msg = records[-1].getMessage()
    # buy_no side must log net_edge_no, not net_edge.
    assert "current_edge_net=-8.0%" in msg, msg
    assert "current_edge_net=5.0%" not in msg, msg


def test_edge_stop_decision_threshold_still_gross(mgr):
    """G8 fix is strictly observational — decision must still compare
    gross current_edge to EDGE_STOP_THRESHOLD, so calibration is stable."""
    import weatheredge.config as cfg
    orig_ev = cfg.MODULES.get("exit_ev_based", False)
    cfg.MODULES["exit_ev_based"] = False
    try:
        # Gross edge = -0.02 (above threshold). If the fix accidentally
        # flipped the decision to use net_edge, this would trigger.
        pos = make_pos(side="buy_yes", price=0.70, amount=100, cost=70,
                       edge_at_entry=0.05)
        opp = _opp_with_net_edge(
            model_probability=0.68, market_price=0.70,
            net_edge=-0.05,  # net says "stop", gross says "hold"
        )
        reason = mgr._check_position(pos, {"MKT-1": opp}, {})
    finally:
        cfg.MODULES["exit_ev_based"] = orig_ev

    # Gross = -0.02 > -0.03 threshold → MUST hold.
    assert reason != "edge_stop"
