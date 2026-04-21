"""Edge calculation and platform fee model.

The Kalshi fee is a per-contract fee charged at trade execution (both
buy and sell), NOT a winnings tax. Formula per the Kalshi fee schedule
(Feb 2026): ``ceil(coef * price * (1 - price) * 100) / 100``.

- Taker coefficient: 0.07
- Maker coefficient: 0.0175
- The fee is symmetric in price (p*(1-p) is symmetric), so YES and NO
  sides pay the same fee at the same contract price.
"""

from __future__ import annotations

import math


# Kalshi fee coefficients (per Kalshi fee schedule Feb 2026).
KALSHI_TAKER_COEF: float = 0.07
KALSHI_MAKER_COEF: float = 0.0175


# Platform fee estimates (verify before go-live).
# Kalshi's `estimated_round_trip` is retained for legacy callers and
# fallback paths; the authoritative cost for calculate_net_edge is
# kalshi_round_trip_cost(price), which is price-dependent.
PLATFORM_FEES: dict[str, dict[str, float]] = {
    "polymarket": {
        "explicit_fee": 0.00,           # No explicit trading fee
        "avg_spread": 0.03,             # ~3% typical bid-ask on weather markets
        "estimated_round_trip": 0.015,  # Half-spread as cost estimate
    },
    "kalshi": {
        "taker_coef": KALSHI_TAKER_COEF,
        "maker_coef": KALSHI_MAKER_COEF,
        "avg_spread": 0.02,             # ~2% typical spread
        # Flat fallback used only for code paths that need a scalar; the
        # price-dependent cost is kalshi_round_trip_cost(price).
        "estimated_round_trip": 0.02,
    },
}


def kalshi_taker_fee(price: float) -> float:
    """Kalshi taker fee per contract.

    Formula: ``ceil(0.07 * p * (1 - p) * 100) / 100``. Symmetric in p,
    so buying YES at p costs the same fee as buying NO at (1-p).
    Returns 0.0 for degenerate prices (<=0 or >=1).
    """
    if price <= 0.0 or price >= 1.0:
        return 0.0
    return math.ceil(KALSHI_TAKER_COEF * price * (1.0 - price) * 100.0) / 100.0


def kalshi_round_trip_cost(price: float) -> float:
    """Estimated round-trip cost for a Kalshi taker trade at ``price``.

    Composition:
      - Entry taker fee (price-dependent).
      - Exit taker fee (same formula; assumes exit at or near entry price).
      - Half-spread slippage on one crossing (taker takes the worse side
        once; the other leg may settle at $1/$0 without spread cost).
    """
    fee_round_trip = 2.0 * kalshi_taker_fee(price)
    half_spread = PLATFORM_FEES["kalshi"]["avg_spread"] / 2.0
    return fee_round_trip + half_spread


def calculate_gross_edge(model_probability: float, market_price: float) -> float:
    """Gross edge = model probability - market implied probability."""
    return model_probability - market_price


def calculate_net_edge(model_probability: float, market_price: float, platform: str) -> float:
    """
    Net edge = gross edge - estimated trading costs (Addendum Gap 2).
    Only trade when NET edge exceeds threshold.
    """
    gross_edge = calculate_gross_edge(model_probability, market_price)
    if platform == "kalshi":
        cost = kalshi_round_trip_cost(market_price)
    else:
        # BugFix N1: Use .get() to avoid KeyError for unknown platforms
        fees = PLATFORM_FEES.get(platform, {"estimated_round_trip": 0.02})
        cost = fees["estimated_round_trip"]
    return gross_edge - cost
