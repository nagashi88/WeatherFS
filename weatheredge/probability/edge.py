# Platform fee estimates (Addendum Gap 2, as of March 2026 — verify before go-live)
PLATFORM_FEES: dict[str, dict[str, float]] = {
    "polymarket": {
        "explicit_fee": 0.00,           # No explicit trading fee
        "avg_spread": 0.03,             # ~3% typical bid-ask on weather markets
        "estimated_round_trip": 0.015,  # Half-spread as cost estimate
    },
    "kalshi": {
        "fee_on_winnings": 0.02,        # ~2% fee on winning trades
        "avg_spread": 0.02,             # ~2% typical spread
        "estimated_round_trip": 0.02,   # Combined estimate
    },
}


def calculate_gross_edge(model_probability: float, market_price: float) -> float:
    """Gross edge = model probability - market implied probability."""
    return model_probability - market_price


def calculate_net_edge(model_probability: float, market_price: float, platform: str) -> float:
    """
    Net edge = gross edge - estimated trading costs (Addendum Gap 2).
    Only trade when NET edge exceeds threshold.
    """
    # BugFix N1: Use .get() to avoid KeyError for unknown platforms
    fees = PLATFORM_FEES.get(platform, {"estimated_round_trip": 0.02})
    estimated_cost = fees["estimated_round_trip"]
    gross_edge = calculate_gross_edge(model_probability, market_price)
    return gross_edge - estimated_cost
