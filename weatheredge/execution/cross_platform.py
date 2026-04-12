"""Cross-platform arbitrage detection between Polymarket and Kalshi.

Implements PRD Section 6.3: scans for price discrepancies on matching
city+date+threshold markets across both platforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from weatheredge.execution.paper_trader import Order

log = logging.getLogger(__name__)


@dataclass
class ArbOpportunity:
    city: str
    date: str
    threshold: float
    poly_price: float
    kalshi_price: float
    spread: float
    direction: str  # Which side to trade


class CrossPlatformArbitrage:
    MIN_SPREAD = 0.05  # 5% minimum spread to flag
    # BugFix M6: Account for platform fees in arbitrage spread calculation
    _POLY_ROUNDTRIP_FEE = 0.015   # ~1.5% round-trip on Polymarket
    _KALSHI_ROUNDTRIP_FEE = 0.02  # ~2% round-trip on Kalshi

    def scan(self, poly_markets: list[dict], kalshi_markets: list[dict]) -> list[ArbOpportunity]:
        """Match markets by city+date+threshold, flag spreads > 5% net of fees."""
        opportunities = []

        # Index kalshi markets by (city, date, threshold)
        kalshi_index = {}
        for m in kalshi_markets:
            key = (m.get("city", "").lower(), m.get("date", ""), m.get("threshold"))
            kalshi_index[key] = m

        # Find matching polymarket markets
        for pm in poly_markets:
            key = (pm.get("city", "").lower(), pm.get("date", ""), pm.get("threshold"))
            km = kalshi_index.get(key)
            if km is None:
                continue

            poly_price = pm.get("yes_price", 0)
            kalshi_price = km.get("yes_price", 0)
            gross_spread = abs(poly_price - kalshi_price)
            # Deduct round-trip fees from both platforms
            spread = gross_spread - self._POLY_ROUNDTRIP_FEE - self._KALSHI_ROUNDTRIP_FEE

            if spread >= self.MIN_SPREAD:
                direction = "buy_poly_sell_kalshi" if poly_price < kalshi_price else "buy_kalshi_sell_poly"
                opportunities.append(ArbOpportunity(
                    city=pm.get("city", ""),
                    date=pm.get("date", ""),
                    threshold=pm.get("threshold", 0),
                    poly_price=poly_price,
                    kalshi_price=kalshi_price,
                    spread=spread,
                    direction=direction,
                ))

        return opportunities

    def execute_arb(self, opp: ArbOpportunity, paper_trader, bankroll: float) -> list[dict]:
        """Execute both legs of an arbitrage opportunity via paper_trader.

        Sizes each leg at min(5% of bankroll, $50). Skips if bankroll is
        insufficient to cover both legs.

        Returns a list of filled trade dicts (one per leg placed).
        """
        leg_size = min(0.05 * bankroll, 50.0)

        # Determine which side to trade on each platform
        if opp.direction == "buy_poly_sell_kalshi":
            # poly YES is cheaper: buy YES on poly, buy NO on kalshi
            poly_side = "buy_yes"
            kalshi_side = "buy_no"
        else:
            # kalshi YES is cheaper: buy YES on kalshi, buy NO on poly
            poly_side = "buy_no"
            kalshi_side = "buy_yes"

        # Compute cost for each leg
        if poly_side == "buy_yes":
            poly_cost = opp.poly_price * leg_size
        else:
            poly_cost = (1.0 - opp.poly_price) * leg_size

        if kalshi_side == "buy_yes":
            kalshi_cost = opp.kalshi_price * leg_size
        else:
            kalshi_cost = (1.0 - opp.kalshi_price) * leg_size

        total_cost = poly_cost + kalshi_cost
        if leg_size <= 0 or bankroll < total_cost:
            log.warning(
                "execute_arb: insufficient bankroll %.2f for arb cost %.2f — skipping %s",
                bankroll, total_cost, opp.city,
            )
            return []

        market_id = f"arb_{opp.city}_{opp.date}_{opp.threshold}"

        poly_order = Order(
            market_id=market_id,
            side=poly_side,
            price=opp.poly_price,
            amount=leg_size,
            agent_name="cross_platform_arb",
            platform="polymarket",
            city=opp.city,
        )
        kalshi_order = Order(
            market_id=market_id,
            side=kalshi_side,
            price=opp.kalshi_price,
            amount=leg_size,
            agent_name="cross_platform_arb",
            platform="kalshi",
            city=opp.city,
        )

        fills: list[dict] = []
        for order in (poly_order, kalshi_order):
            result = paper_trader.place_order(order)
            if result.get("filled"):
                fills.append(result["trade"])
            else:
                log.warning(
                    "execute_arb: order rejected platform=%s reason=%s",
                    order.platform, result.get("reason"),
                )

        return fills
