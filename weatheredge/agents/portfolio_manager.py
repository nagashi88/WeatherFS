"""Portfolio Manager for WeatherEdge.

Implements weighted voting across agents (Addendum Gap 7a):
- AGENT_WEIGHTS defines per-agent contribution to the consensus signal
- TRADE_THRESHOLD gates whether the weighted score triggers a trade
- Delegates final sizing and risk checks to RiskManager
"""

from __future__ import annotations

import logging

from weatheredge.agents.base import MarketOpportunity, Signal, TradeOrder
from weatheredge.agents.risk_manager import RiskManager
from weatheredge.config import Settings

log = logging.getLogger(__name__)


class PortfolioManager:
    AGENT_WEIGHTS: dict[str, float] = {
        "Conservative": 0.30,
        "Aggressive": 0.25,
        "MeanReversion": 0.05,  # Disabled by default; kept at min weight for learning loop
        "Qualitative": 0.15,
        "SellNO": 0.25,
    }
    TRADE_THRESHOLD: float = 0.25  # Option A €100 bankroll: lowered from 0.35 — Phase 1 rmse_floor suppresses multi-agent consensus, allow single strong-edge vote

    def __init__(self, settings: Settings, risk_manager: RiskManager) -> None:
        self.settings = settings
        self.risk_manager = risk_manager

    def aggregate_signals(
        self,
        signals_by_market: dict[str, list[Signal]],
        opportunities: dict[str, MarketOpportunity],
        bankroll: float,
        open_positions: list,
        daily_pnl: float,
        total_pnl: float,
        recent_fills: list[dict] | None = None,
    ) -> list[TradeOrder]:
        """Aggregate per-agent signals into TradeOrders via weighted voting.

        Algorithm (Addendum Gap 7a):
        1. For each market, sum weighted confidences for non-no_trade signals.
        2. If weighted_score > TRADE_THRESHOLD, determine majority side.
        3. Run risk pre-check; skip market if blocked.
        4. Size position via Kelly; halve if VPIN reduce_size flag is set.
        5. Emit a TradeOrder.
        """
        orders: list[TradeOrder] = []
        candidates: list[dict] = []
        # Track projected open positions during this decision cycle so risk checks
        # (especially max_open_positions) account for orders already accepted.
        projected_open_positions = list(open_positions)

        for market_id, signals in signals_by_market.items():
            if not signals:
                continue

            # Step 1: weighted score — only count non-no_trade votes
            weighted_score = sum(
                self.AGENT_WEIGHTS.get(sig.agent_name, 0.0) * sig.confidence
                for sig in signals
                if sig.side != "no_trade"
            )

            opp = opportunities.get(market_id)
            log.info(
                "Voting: market=%s edge_yes=%s edge_no=%s score=%.2f threshold=%.2f",
                market_id,
                f"{opp.net_edge:.1%}" if opp else "n/a",
                f"{opp.net_edge_no:.1%}" if opp else "n/a",
                weighted_score,
                self.TRADE_THRESHOLD,
            )

            # BugFix N11: Use < instead of <= so SellNO (weight=0.20) can trade at threshold
            if weighted_score < self.TRADE_THRESHOLD:
                continue

            # Step 2: majority side among voting signals
            side_scores: dict[str, float] = {}
            for sig in signals:
                if sig.side == "no_trade":
                    continue
                weight = self.AGENT_WEIGHTS.get(sig.agent_name, 0.0)
                side_scores[sig.side] = side_scores.get(sig.side, 0.0) + weight * sig.confidence

            if not side_scores:
                continue

            majority_side = max(side_scores, key=lambda s: side_scores[s])
            majority_side_score = side_scores[majority_side]
            minority_side_score = max(
                (score for side, score in side_scores.items() if side != majority_side),
                default=0.0,
            )

            # Build candidate and defer risk/sizing checks until after prioritization.
            market_price = opp.market_price if opp else 0.50
            model_probability = opp.model_probability if opp else 0.50
            city = opp.city if opp else ""
            station = opp.station if opp else ""
            platform = opp.platform if opp else ""
            market_date = opp.market_date if (opp and hasattr(opp, "market_date")) else ""
            volume = opp.volume if (opp and hasattr(opp, "volume")) else None
            effective_edge = (
                opp.net_edge_no if (opp and majority_side == "buy_no")
                else (opp.net_edge if opp else 0.0)
            )
            candidates.append(
                {
                    "market_id": market_id,
                    "majority_side": majority_side,
                    "market_price": market_price,
                    "model_probability": model_probability,
                    "city": city,
                    "station": station,
                    "platform": platform,
                    "market_date": market_date,
                    "volume": volume,
                    "effective_edge": effective_edge,
                    "weighted_score": weighted_score,
                    "side_margin": majority_side_score - minority_side_score,
                }
            )

        # Prioritize strongest opportunities first; this matters when risk caps
        # stop processing before all candidates can be accepted.
        candidates.sort(
            key=lambda c: (
                c["weighted_score"],
                c["effective_edge"],
                c["side_margin"],
            ),
            reverse=True,
        )

        for candidate in candidates:
            market_id = candidate["market_id"]
            opp = opportunities.get(market_id)
            majority_side = candidate["majority_side"]
            market_price = candidate["market_price"]
            model_probability = candidate["model_probability"]
            city = candidate["city"]
            station = candidate.get("station", "")
            platform = candidate["platform"]
            market_date = candidate.get("market_date", "")
            volume = candidate.get("volume")
            effective_edge = candidate["effective_edge"]

            # C4 fix: compute capital_base (free cash + open cost) before sizing so
            # both Kelly sizing and risk limits use the same total-capital base.
            # Previously Kelly used free bankroll while risk used capital_base → asymmetry.
            open_cost = sum(
                (p.get("cost", 0.0) if isinstance(p, dict) else getattr(p, "cost", 0.0))
                for p in projected_open_positions
            )
            capital_base = bankroll + open_cost

            preliminary_size = self.risk_manager.calculate_position_size(
                win_probability=model_probability,
                market_price=market_price,
                bankroll=capital_base,
                side=majority_side,
                volume=volume,
            )

            # Risk checks consume exposure in dollars (cost), while position sizing
            # returns contract amount. Convert contracts -> cost for correct limits.
            if majority_side == "buy_no":
                proposed_cost = (1.0 - market_price) * preliminary_size
            else:
                proposed_cost = market_price * preliminary_size

            risk_result = self.risk_manager.pre_trade_check(
                city=city,
                proposed_size=proposed_cost,
                open_positions=projected_open_positions,
                bankroll=capital_base,
                daily_pnl=daily_pnl,
                total_pnl=total_pnl,
                recent_fills=recent_fills,
                market_date=market_date,
                station=station,
            )

            if risk_result.get("blocked"):
                log.info(
                    "Risk blocked: market=%s side=%s reason=%s",
                    market_id,
                    majority_side,
                    risk_result.get("reason", "blocked"),
                )
                continue

            # Step 4: final position size (uses same capital_base as risk checks)
            amount = self.risk_manager.calculate_position_size(
                win_probability=model_probability,
                market_price=market_price,
                bankroll=capital_base,
                side=majority_side,
                volume=volume,
            )

            # Step 5: VPIN reduction
            if risk_result.get("reduce_size"):
                amount = round(amount / 2, 2)

            if amount <= 0:
                continue

            order = TradeOrder(
                market_id=market_id,
                side=majority_side,
                price=market_price,
                amount=amount,
                agent_name="PortfolioManager",
                model_probability=model_probability,
                # BugFix L2: Use pre-computed net edge (with fee deduction) from opportunity
                edge_at_entry=effective_edge,
                city=city,
                platform=platform,
                station=station,
                market_date=market_date,
                gaussian_prob=opp.gaussian_prob if opp else None,
                ensemble_prob=opp.ensemble_prob if opp else None,
                blended_prob=opp.blended_prob if opp else None,
                raw_model_prob=opp.raw_model_prob if opp else None,
            )
            orders.append(order)

            # Reserve this order in projected exposure so subsequent markets in the
            # same cycle cannot exceed position-count or exposure caps.
            projected_cost = (
                (1.0 - market_price) * amount if majority_side == "buy_no"
                else market_price * amount
            )
            projected_open_positions.append(
                {"market_id": market_id, "city": city, "station": station, "market_date": market_date, "cost": projected_cost, "status": "open"}
            )

        return orders
