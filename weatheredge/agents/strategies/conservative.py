from weatheredge.agents.base import Signal, MarketOpportunity
from weatheredge.config import Settings

def conservative_strategy(opportunity: MarketOpportunity, settings: Settings) -> Signal:
    """
    Conservative "Ben Graham" strategy. Only trade when edge is overwhelming.
    - min_edge: 20% net edge
    - model_agreement: Gaussian and ensemble must agree within 10%
    - lead_time: 1-day forecasts only
    - Single bucket (not laddering)
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

    # Confidence scales with edge (0.20 edge -> 0.5 confidence, 0.40 edge -> 1.0)
    confidence = min(effective_edge / 0.40, 1.0)

    return Signal(market_id=market_id, side=side, confidence=confidence, agent_name="Conservative", reasoning=f"Strong edge {effective_edge:.2f}, agreement OK")
