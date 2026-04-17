from weatheredge.agents.base import Signal, MarketOpportunity
from weatheredge.config import Settings


def _emit_signal(opp: MarketOpportunity, role: str) -> Signal | None:
    """Build a single Aggressive signal for ``opp`` if it clears the edge gate."""
    side = "buy_yes" if opp.model_probability > opp.market_price else "buy_no"
    effective_edge = opp.net_edge if side == "buy_yes" else getattr(opp, "net_edge_no", 0.0)
    if effective_edge <= 0:
        return None
    base_confidence = min(effective_edge / 0.20, 1.0)
    confidence = base_confidence if role == "center" else base_confidence * 0.5
    return Signal(
        market_id=opp.market_id,
        side=side,
        confidence=confidence,
        agent_name="Aggressive",
        reasoning=f"Ladder {role}, edge {effective_edge:.2f}",
    )


def aggressive_strategy(
    opportunity: MarketOpportunity,
    bucket_markets: list[MarketOpportunity],
    settings: Settings,
) -> list[Signal]:
    """
    Aggressive "Cathie Wood" strategy. Always emits at most one signal for the
    center opportunity, plus up to 2 STRICTLY adjacent bucket neighbors
    (sharing a bucket boundary). Previous behaviour ranked the entire city by
    edge and emitted top-3 signals on neighbors only — this caused the same
    market to receive an Aggressive vote from every other market in its city
    during a single decision cycle (the "14 trades in 8ms" burst).
    """
    # Edge gate on center
    best_edge = max(opportunity.net_edge, getattr(opportunity, "net_edge_no", 0.0))
    if best_edge < settings.aggressive_min_edge:
        return []

    # Only 1-2 day forecasts
    if opportunity.lead_time_days > 2:
        return []

    signals: list[Signal] = []
    center_sig = _emit_signal(opportunity, role="center")
    if center_sig is not None:
        signals.append(center_sig)

    # Laddering only applies when the center IS a bucket market.
    center_low = opportunity.bucket_low
    center_high = opportunity.bucket_high
    if center_low is None or center_high is None or not bucket_markets:
        return signals

    # CCG-bug-4: scope candidates to same date+station+platform — bucket_markets
    # arrives as the full same-city list, which would let us ladder into a bucket
    # for next week or a different reporting station. Independent contracts.
    center_date = getattr(opportunity, "market_date", "")
    center_station = getattr(opportunity, "station", "")
    center_platform = getattr(opportunity, "platform", "")
    same_contract_buckets = [
        nb for nb in bucket_markets
        if getattr(nb, "market_date", "") == center_date
        and getattr(nb, "station", "") == center_station
        and getattr(nb, "platform", "") == center_platform
    ]

    # CCG-bug-7: pick the highest-edge adjacent neighbours, not the first 2 in
    # iteration order, so ladder quality doesn't depend on dict ordering.
    # CCG-bug-5 (audit-2): rank by the SAME edge that `_emit_signal` will use —
    # the side `_emit_signal` picks (`buy_yes` if model_probability > market_price
    # else `buy_no`). Using `max(net_edge, net_edge_no)` could prioritize a
    # neighbor whose best-edge side is the opposite of the side actually emitted.
    def _signal_side_edge(nb: MarketOpportunity) -> float:
        side = "buy_yes" if nb.model_probability > nb.market_price else "buy_no"
        return nb.net_edge if side == "buy_yes" else getattr(nb, "net_edge_no", 0.0)

    same_contract_buckets.sort(key=_signal_side_edge, reverse=True)

    # Strict adjacency: neighbor shares a boundary with the center bucket.
    adjacent: list[MarketOpportunity] = []
    for nb in same_contract_buckets:
        nb_low = nb.bucket_low
        nb_high = nb.bucket_high
        if nb_low is None or nb_high is None:
            continue
        if abs(nb_low - center_high) < 1e-6 or abs(nb_high - center_low) < 1e-6:
            adjacent.append(nb)
        if len(adjacent) >= 2:
            break

    for nb in adjacent:
        nb_best_edge = max(nb.net_edge, getattr(nb, "net_edge_no", 0.0))
        if nb_best_edge < settings.aggressive_min_edge:
            continue
        sig = _emit_signal(nb, role="adjacent")
        if sig is not None:
            signals.append(sig)

    return signals
