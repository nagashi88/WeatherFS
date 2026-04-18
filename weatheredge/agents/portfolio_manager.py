"""Portfolio Manager for WeatherEdge.

Implements weighted voting across agents (Addendum Gap 7a):
- AGENT_WEIGHTS defines per-agent contribution to the consensus signal
- TRADE_THRESHOLD gates whether the weighted score triggers a trade
- Delegates final sizing and risk checks to RiskManager
"""

from __future__ import annotations

import logging
import math

from weatheredge.agents.base import MarketOpportunity, Signal, TradeOrder
from weatheredge.agents.risk_manager import RiskManager
from weatheredge.config import Settings

log = logging.getLogger(__name__)


def _dedupe_agent_votes(signals: list[Signal]) -> list[Signal]:
    """Keep at most one Signal per agent_name — the one with highest confidence.

    Defensive net for strategies that may emit the same market_id from multiple
    outer-loop iterations within a single decision cycle.
    """
    best: dict[str, Signal] = {}
    no_trade_keep: list[Signal] = []
    for sig in signals:
        if sig.side == "no_trade":
            # Preserve no_trade votes unchanged — they don't affect weighted_score
            no_trade_keep.append(sig)
            continue
        prev = best.get(sig.agent_name)
        if prev is None or sig.confidence > prev.confidence:
            best[sig.agent_name] = sig
    return list(best.values()) + no_trade_keep


class PortfolioManager:
    AGENT_WEIGHTS: dict[str, float] = {
        "Conservative": 0.30,
        "Aggressive": 0.25,
        "MeanReversion": 0.05,  # Disabled by default; kept at min weight for learning loop
        "Qualitative": 0.15,
        "SellNO": 0.25,
    }
    TRADE_THRESHOLD: float = 0.25  # Option A €100 bankroll: lowered from 0.35 — Phase 1 rmse_floor suppresses multi-agent consensus, allow single strong-edge vote
    MIN_DISTINCT_AGENTS: int = 2  # A2: require >=2 distinct non-no_trade agents on the MAJORITY side
    # CCG-bug-1 (audit-2): single-agent override threshold MUST be reachable.
    # Max single-agent weighted score = max(AGENT_WEIGHTS) = 0.30 (Conservative).
    # Setting threshold = 0.30 lets Conservative's full-confidence vote bypass
    # the distinct-agent gate; lower-weight agents still need a partner.
    CONVICTION_THRESHOLD: float = 0.30

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

            # C2: dedupe per agent — keep only the highest-confidence signal per
            # agent_name. The Aggressive strategy in particular can be invoked
            # from multiple outer-loop iterations within the same cycle (each
            # neighbour's call ladders back onto this market). Without dedupe,
            # a single agent contributes multiple weighted votes for one market.
            signals = _dedupe_agent_votes(signals)

            # Compute per-side weighted scores AND per-side distinct-agent sets first;
            # CCG-bug-1 requires the A2 gate to be evaluated on the majority side, not
            # on the union (which previously let conflicting buy_yes+buy_no pairs pass).
            side_scores: dict[str, float] = {}
            side_distinct_agents: dict[str, set[str]] = {}
            for sig in signals:
                if sig.side == "no_trade":
                    continue
                weight = self.AGENT_WEIGHTS.get(sig.agent_name, 0.0)
                side_scores[sig.side] = side_scores.get(sig.side, 0.0) + weight * sig.confidence
                side_distinct_agents.setdefault(sig.side, set()).add(sig.agent_name)

            if not side_scores:
                continue

            majority_side = max(side_scores, key=lambda s: side_scores[s])
            majority_side_score = side_scores[majority_side]
            minority_side_score = max(
                (score for side, score in side_scores.items() if side != majority_side),
                default=0.0,
            )
            # CCG-bug-1 (audit-2): when buy_yes and buy_no scores tie, max() picks
            # arbitrarily (dict-insertion order). Skip rather than trade an
            # ambiguous side — there's no clear majority to act on.
            if (
                len(side_scores) >= 2
                and abs(majority_side_score - minority_side_score) < 1e-9
            ):
                log.info(
                    "Voting: market=%s skipped — side_scores tied at %.4f (no clear majority)",
                    market_id, majority_side_score,
                )
                continue
            majority_distinct = len(side_distinct_agents.get(majority_side, set()))
            weighted_score = sum(side_scores.values())

            opp = opportunities.get(market_id)
            log.info(
                "Voting: market=%s edge_yes=%s edge_no=%s score=%.2f majority=%s majority_score=%.2f distinct_majority=%d threshold=%.2f",
                market_id,
                f"{opp.net_edge:.1%}" if opp else "n/a",
                f"{opp.net_edge_no:.1%}" if opp else "n/a",
                weighted_score,
                majority_side,
                majority_side_score,
                majority_distinct,
                self.TRADE_THRESHOLD,
            )

            # BugFix N11: Use < instead of <= so SellNO (weight=0.20) can trade at threshold
            if weighted_score < self.TRADE_THRESHOLD:
                continue

            # CCG-bug-1: A2 distinct-agent gate is now SIDE-AWARE.
            # Pass condition: majority side has >= MIN_DISTINCT_AGENTS distinct agents
            # OR the majority-side weighted score exceeds CONVICTION_THRESHOLD
            # (single-agent override for high-conviction trades).
            if (
                majority_distinct < self.MIN_DISTINCT_AGENTS
                and majority_side_score < self.CONVICTION_THRESHOLD
            ):
                log.info(
                    "Voting: market=%s skipped — majority side %s has %d agent(s)<%d and score %.2f<conviction %.2f",
                    market_id, majority_side, majority_distinct, self.MIN_DISTINCT_AGENTS,
                    majority_side_score, self.CONVICTION_THRESHOLD,
                )
                continue

            # Build candidate and defer risk/sizing checks until after prioritization.
            market_price = opp.market_price if opp else 0.50
            model_probability = opp.model_probability if opp else 0.50
            city = opp.city if opp else ""
            station = opp.station if opp else ""
            platform = opp.platform if opp else ""
            market_date = opp.market_date if (opp and hasattr(opp, "market_date")) else ""
            volume = opp.volume if (opp and hasattr(opp, "volume")) else None
            bucket_low = getattr(opp, "bucket_low", None) if opp else None
            bucket_high = getattr(opp, "bucket_high", None) if opp else None
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
                    "bucket_low": bucket_low,
                    "bucket_high": bucket_high,
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

        # C1 fix: compute equity_base ONCE before the candidate loop using only
        # pre-existing open positions. Recomputing inside the loop with
        # projected_open_positions caused capital_base to grow with each accepted
        # order, inflating per-trade caps and bypassing max_cost_per_trade_fraction.
        existing_open_cost = sum(
            (p.get("cost", 0.0) if isinstance(p, dict) else getattr(p, "cost", 0.0))
            for p in open_positions
        )
        equity_base = bankroll + existing_open_cost
        # CCG-bug-8: track free cash decreasing as orders accumulate within the
        # same decision cycle, so multiple sized-against-equity-base orders cannot
        # collectively exceed available cash.
        remaining_cash = max(0.0, float(bankroll))

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
            bucket_low = candidate.get("bucket_low")
            bucket_high = candidate.get("bucket_high")
            effective_edge = candidate["effective_edge"]

            # A3: in-cycle bucket correlation guard. Adjacent/overlapping buckets in the
            # same city+date are highly correlated (one outcome pays both) — taking both
            # in one cycle doubles directional risk without adding diversification.
            if bucket_low is not None and bucket_high is not None:
                correlated = False
                for proj in projected_open_positions:
                    proj_city = proj.get("city") if isinstance(proj, dict) else getattr(proj, "city", "")
                    proj_date = proj.get("market_date") if isinstance(proj, dict) else getattr(proj, "market_date", "")
                    if proj_city != city or proj_date != market_date:
                        continue
                    proj_low = proj.get("bucket_low") if isinstance(proj, dict) else getattr(proj, "bucket_low", None)
                    proj_high = proj.get("bucket_high") if isinstance(proj, dict) else getattr(proj, "bucket_high", None)
                    if proj_low is None or proj_high is None:
                        continue
                    # CCG-bug-2: STRICT overlap only — adjacent buckets sharing a single
                    # boundary point (e.g., [70,75] vs [75,80]) are NOT blocked, because
                    # the Aggressive ladder strategy intentionally emits them together.
                    # We block only when intervals share an interior region.
                    if (
                        bucket_low < proj_high - 1e-6
                        and bucket_high > proj_low + 1e-6
                    ):
                        log.info(
                            "Bucket-correlation skip: market=%s [%.1f,%.1f] overlaps reserved %s [%.1f,%.1f] (city=%s date=%s)",
                            market_id, bucket_low, bucket_high,
                            proj.get("market_id") if isinstance(proj, dict) else getattr(proj, "market_id", "?"),
                            proj_low, proj_high, city, market_date,
                        )
                        correlated = True
                        break
                if correlated:
                    continue

            capital_base = equity_base

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

            # CCG-bug-8: hard-cap by remaining free cash so cumulative orders within
            # one cycle cannot deploy more than the user actually has. Round DOWN
            # to nearest 0.01 contract to avoid floating-point overshoot.
            per_contract_cost = (1.0 - market_price) if majority_side == "buy_no" else market_price
            cash_capped = False
            if per_contract_cost > 0 and remaining_cash < amount * per_contract_cost:
                cap_amount = remaining_cash / per_contract_cost
                amount = math.floor(cap_amount * 100) / 100
                cash_capped = True
                log.info(
                    "Cash-cap: market=%s amount reduced to %.2f (remaining_cash=%.2f, per_contract=%.4f)",
                    market_id, amount, remaining_cash, per_contract_cost,
                )

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
                cash_capped=cash_capped,
            )
            orders.append(order)

            # Reserve this order in projected exposure so subsequent markets in the
            # same cycle cannot exceed position-count or exposure caps.
            projected_cost = (
                (1.0 - market_price) * amount if majority_side == "buy_no"
                else market_price * amount
            )
            projected_open_positions.append(
                {
                    "market_id": market_id,
                    "city": city,
                    "station": station,
                    "market_date": market_date,
                    "cost": projected_cost,
                    "status": "open",
                    "bucket_low": bucket_low,
                    "bucket_high": bucket_high,
                }
            )
            # CCG-bug-8: decrement free-cash budget for subsequent candidates.
            remaining_cash = max(0.0, remaining_cash - projected_cost)

        return orders
