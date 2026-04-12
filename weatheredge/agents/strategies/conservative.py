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

    # Only 1-day forecasts
    if opportunity.lead_time_days != 1:
        return Signal(market_id=market_id, side="no_trade", confidence=0.0, agent_name="Conservative", reasoning="Lead time > 1 day")

    # Determine side and use the corresponding edge
    side = "buy_yes" if opportunity.model_probability > opportunity.market_price else "buy_no"
    effective_edge = opportunity.net_edge if side == "buy_yes" else opportunity.net_edge_no

    # Check net edge (using the correct side)
    if effective_edge < settings.conservative_min_edge:
        return Signal(market_id=market_id, side="no_trade", confidence=0.0, agent_name="Conservative", reasoning=f"Net edge {effective_edge:.2f} < {settings.conservative_min_edge}")

    # Check model agreement (if both available)
    if opportunity.gaussian_prob is not None and opportunity.ensemble_prob is not None:
        agreement = 1.0 - abs(opportunity.gaussian_prob - opportunity.ensemble_prob)
        if agreement < 0.90:
            return Signal(market_id=market_id, side="no_trade", confidence=0.0, agent_name="Conservative", reasoning=f"Model agreement {agreement:.2f} < 0.90")

    # Confidence scales with edge (0.20 edge -> 0.5 confidence, 0.40 edge -> 1.0)
    confidence = min(effective_edge / 0.40, 1.0)

    return Signal(market_id=market_id, side=side, confidence=confidence, agent_name="Conservative", reasoning=f"Strong edge {effective_edge:.2f}, agreement OK")
