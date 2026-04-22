from weatheredge.agents.base import Signal, MarketOpportunity
from weatheredge.agents.portfolio_manager import PortfolioManager
from weatheredge.config import Settings

# CCG-bughunt3 G5-new: Conservative has two distinct edge thresholds and the
# docstring used to mention only one, which was counterintuitive for callers
# reading `min_edge: 20%` and expecting that to trigger trades on its own.
#
#   * vote floor = settings.conservative_min_edge (default 0.20)
#       Below this, the agent abstains (no_trade). Above, it contributes a
#       weighted vote but confidence is partial — the vote needs a partner
#       agent on the same side for the market to trade.
#   * solo-trade floor = SOLO_CONFIDENCE_EDGE (0.40)
#       Edge at which confidence saturates to 1.0. A saturated Conservative
#       vote alone clears PortfolioManager.CONVICTION_THRESHOLD because
#       AGENT_WEIGHTS["Conservative"] * 1.0 ≥ CONVICTION_THRESHOLD.
#
# The invariant `AGENT_WEIGHTS["Conservative"] >= CONVICTION_THRESHOLD` is
# what makes a solo Conservative vote possible at all; if CONVICTION_THRESHOLD
# is raised above the Conservative weight, SOLO_CONFIDENCE_EDGE becomes
# unreachable. Guarded by a unit test so silent drift is caught.
SOLO_CONFIDENCE_EDGE: float = 0.40


def conservative_strategy(opportunity: MarketOpportunity, settings: Settings) -> Signal:
    """
    Conservative "Ben Graham" strategy. Only trade when edge is overwhelming.
    - vote floor: `conservative_min_edge` (default 20%) — emits a partial-
      confidence vote; requires a second agent on the same side to trade.
    - solo-trade floor: 40% edge — confidence saturates at 1.0, letting a
      single Conservative vote bypass PortfolioManager.MIN_DISTINCT_AGENTS
      via the CONVICTION_THRESHOLD override.
    - model_agreement: Gaussian and ensemble must agree within 10%.
    - lead_time: same-day (with edge bonus) and 1-day forecasts only.
    - Single bucket (not laddering).
    """
    market_id = opportunity.market_id

    # C3: Allow same-day (0) and 1-day forecasts. The previous `!= 1` gate
    # silently disabled Conservative on all same-day markets (where
    # `int(delta_hours / 24.0) == 0`), pushing the portfolio toward
    # Aggressive + SellNO and producing all-buy_no concentration.
    if opportunity.lead_time_days > 1:
        return Signal(market_id=market_id, side="no_trade", confidence=0.0, agent_name="Conservative", reasoning="Lead time > 1 day")

    # Determine side and use the corresponding edge
    side = "buy_yes" if opportunity.model_probability > opportunity.market_price else "buy_no"
    effective_edge = opportunity.net_edge if side == "buy_yes" else opportunity.net_edge_no

    # CCG-bug-5: same-day markets (lead_time_days == 0) often have predatory
    # spreads, thinner books, and faster decay. Require an extra edge bonus
    # before Conservative will trade them.
    edge_gate = settings.conservative_min_edge
    if opportunity.lead_time_days == 0:
        edge_gate += getattr(settings, "conservative_same_day_edge_bonus", 0.05)

    # Check net edge (using the correct side)
    if effective_edge < edge_gate:
        return Signal(market_id=market_id, side="no_trade", confidence=0.0, agent_name="Conservative", reasoning=f"Net edge {effective_edge:.2f} < gate {edge_gate:.2f}")

    # Check model agreement (if both available)
    if opportunity.gaussian_prob is not None and opportunity.ensemble_prob is not None:
        agreement = 1.0 - abs(opportunity.gaussian_prob - opportunity.ensemble_prob)
        if agreement < 0.90:
            return Signal(market_id=market_id, side="no_trade", confidence=0.0, agent_name="Conservative", reasoning=f"Model agreement {agreement:.2f} < 0.90")

    # Confidence ramps linearly from the vote floor up to SOLO_CONFIDENCE_EDGE:
    # at the vote floor (0.20) confidence is ≈0.5 → weighted score 0.15
    # (below CONVICTION_THRESHOLD, partner required); at SOLO_CONFIDENCE_EDGE
    # (0.40) confidence saturates at 1.0 → weighted score 0.30 (solo OK).
    confidence = min(effective_edge / SOLO_CONFIDENCE_EDGE, 1.0)

    return Signal(market_id=market_id, side=side, confidence=confidence, agent_name="Conservative", reasoning=f"Strong edge {effective_edge:.2f}, agreement OK")
