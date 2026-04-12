from weatheredge.agents.base import Signal, MarketOpportunity
from weatheredge.config import Settings

def aggressive_strategy(
    opportunity: MarketOpportunity,
    bucket_markets: list[MarketOpportunity],
    settings: Settings,
) -> list[Signal]:
    """
    Aggressive "Cathie Wood" strategy. Temperature laddering across adjacent buckets.
    Buys up to 3 adjacent buckets around the forecast. Center bucket gets highest
    confidence, adjacent get 50%.
    """
    # Check minimum edge on primary opportunity (best of YES or NO side)
    best_edge = max(opportunity.net_edge, opportunity.net_edge_no)
    if best_edge < settings.aggressive_min_edge:
        return []

    # Only 1-2 day forecasts
    if opportunity.lead_time_days > 2:
        return []

    # If no bucket markets provided, generate single signal for the primary opportunity
    if not bucket_markets:
        side = "buy_yes" if opportunity.model_probability > opportunity.market_price else "buy_no"
        effective_edge = opportunity.net_edge if side == "buy_yes" else opportunity.net_edge_no
        confidence = min(effective_edge / 0.20, 1.0)
        return [Signal(
            market_id=opportunity.market_id,
            side=side,
            confidence=confidence,
            agent_name="Aggressive",
            reasoning=f"Single market, edge {effective_edge:.2f}"
        )]

    # Sort buckets by effective net edge (highest first) — NOT by probability.
    # A bucket with 90% prob at 92% market price has NEGATIVE edge.
    scored = sorted(
        bucket_markets,
        key=lambda m: max(m.net_edge, getattr(m, "net_edge_no", 0.0)),
        reverse=True,
    )[:3]

    signals = []
    for i, bucket in enumerate(scored):
        bucket_best_edge = max(bucket.net_edge, getattr(bucket, "net_edge_no", 0.0))
        if bucket_best_edge < settings.aggressive_min_edge:
            continue
        # Center bucket (index 0) gets full confidence, others get 50%
        is_center = (i == 0)
        side = "buy_yes" if bucket.model_probability > bucket.market_price else "buy_no"
        effective_edge = bucket.net_edge if side == "buy_yes" else getattr(bucket, "net_edge_no", 0.0)
        base_confidence = min(effective_edge / 0.20, 1.0)
        confidence = base_confidence if is_center else base_confidence * 0.5

        signals.append(Signal(
            market_id=bucket.market_id,
            side=side,
            confidence=confidence,
            agent_name="Aggressive",
            reasoning=f"Ladder {'center' if is_center else 'adjacent'}, edge {effective_edge:.2f}"
        ))

    return signals
