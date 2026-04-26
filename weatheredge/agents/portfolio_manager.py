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
from weatheredge.data.station_time import hours_until_market_close
from weatheredge.regime import classify_regime

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


def _is_opposing_signal(signal_side: str, majority_side: str) -> bool:
    return (
        signal_side in {"buy_yes", "buy_no"}
        and majority_side in {"buy_yes", "buy_no"}
        and signal_side != majority_side
    )


def _is_fragile_entry_context(opp: MarketOpportunity, majority_side: str) -> bool:
    """Contexts where disagreement should reduce size instead of being ignored."""
    market_type = getattr(opp, "market_type", "daily_high")
    bucket_low = getattr(opp, "bucket_low", None)
    bucket_high = getattr(opp, "bucket_high", None)

    if majority_side == "buy_yes":
        return True
    if market_type == "daily_low":
        return True
    if bucket_low is not None and bucket_high is not None and (bucket_high - bucket_low) <= 2.0:
        return True
    if opp.market_date and opp.station:
        lead_hours = hours_until_market_close(opp.market_date, opp.station)
        if lead_hours is not None and lead_hours <= 12.0:
            return True
    return False


def _same_cycle_duplicate_key(candidate: dict) -> tuple[str, str, str, str] | None:
    """Group same-cycle candidates that express the same directional exposure."""
    station_or_city = str(candidate.get("station") or candidate.get("city") or "").strip()
    market_date = str(candidate.get("market_date") or "").strip()
    majority_side = str(candidate.get("majority_side") or "").strip()
    if not station_or_city or not market_date or majority_side not in {"buy_yes", "buy_no"}:
        return None
    market_type = str(candidate.get("market_type") or "daily_high").strip()
    return (station_or_city, market_date, market_type, majority_side)


def _same_cycle_duplicate_rank(candidate: dict) -> tuple[float, float, float]:
    """Tie-break same-exposure candidates by real edge first, then conviction."""
    return (
        float(candidate.get("effective_edge") or 0.0),
        float(candidate.get("majority_score") or 0.0),
        float(candidate.get("side_margin") or 0.0),
    )


def _filter_same_cycle_duplicates(candidates: list[dict]) -> list[dict]:
    """Keep the best same-cycle candidate per station/date/type/side.

    This is intentionally narrower than a "max one trade per event" rule. It
    only dedupes candidates produced in the same aggregate call and keeps the
    candidate with the strongest signal-side edge, preserving future cycles if
    prices or forecasts move.
    """
    passthrough: list[dict] = []
    best_by_key: dict[tuple[str, str, str, str], dict] = {}
    removed: list[dict] = []

    for candidate in candidates:
        key = _same_cycle_duplicate_key(candidate)
        if key is None:
            passthrough.append(candidate)
            continue

        current_best = best_by_key.get(key)
        if current_best is None:
            best_by_key[key] = candidate
            continue

        if _same_cycle_duplicate_rank(candidate) > _same_cycle_duplicate_rank(current_best):
            removed.append(current_best)
            best_by_key[key] = candidate
        else:
            removed.append(candidate)

    for candidate in removed:
        log.info(
            "Same-cycle duplicate BLOCK: market=%s skipped (station=%s date=%s type=%s side=%s edge=%.3f)",
            candidate.get("market_id"),
            candidate.get("station") or candidate.get("city"),
            candidate.get("market_date"),
            candidate.get("market_type"),
            candidate.get("majority_side"),
            float(candidate.get("effective_edge") or 0.0),
        )

    return passthrough + list(best_by_key.values())


class PortfolioManager:
    AGENT_WEIGHTS: dict[str, float] = {
        "Baseline": 1.00,
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
            # total_score kept for telemetry only — the gate below uses
            # majority_side_score (see CCG-bughunt2 F-new-7).
            total_score = sum(side_scores.values())

            opp = opportunities.get(market_id)
            log.info(
                "Voting: market=%s edge_yes=%s edge_no=%s total=%.2f majority=%s majority_score=%.2f distinct_majority=%d threshold=%.2f",
                market_id,
                f"{opp.net_edge:.1%}" if opp else "n/a",
                f"{opp.net_edge_no:.1%}" if opp else "n/a",
                total_score,
                majority_side,
                majority_side_score,
                majority_distinct,
                self.TRADE_THRESHOLD,
            )

            # CCG-bughunt2 F-new-7: gate on the MAJORITY side's score, not the
            # sum of both sides. The prior `weighted_score = sum(side_scores)`
            # counted opposing votes as additional evidence for *trading*,
            # letting two weak disagreeing signals clear the threshold even
            # though neither side had conviction (e.g. Conservative buy_yes
            # at 0.15 + Aggressive buy_no at 0.125 → sum 0.275 > 0.25, but
            # the winning side was only 0.15). Preserving `<` (not `<=`) so
            # a single full-weight vote exactly at the threshold still trades.
            if majority_side_score < self.TRADE_THRESHOLD:
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
            market_type = getattr(opp, "market_type", "daily_high") if opp else "daily_high"
            volume = opp.volume if (opp and hasattr(opp, "volume")) else None
            bucket_low = getattr(opp, "bucket_low", None) if opp else None
            bucket_high = getattr(opp, "bucket_high", None) if opp else None
            effective_edge = (
                opp.net_edge_no if (opp and majority_side == "buy_no")
                else (opp.net_edge if opp else 0.0)
            )
            caution_haircut = 1.0
            if (
                opp is not None
                and getattr(self.settings, "model_disagreement_guard_enabled", True)
                and opp.gaussian_prob is not None
                and opp.ensemble_prob is not None
            ):
                disagreement = abs(float(opp.gaussian_prob) - float(opp.ensemble_prob))
                block_threshold = getattr(self.settings, "model_disagreement_block_threshold", 0.25)
                haircut_threshold = getattr(self.settings, "model_disagreement_haircut_threshold", 0.15)
                if disagreement >= block_threshold:
                    log.warning(
                        "Model-disagreement BLOCK: market=%s skipped (gaussian=%.3f ensemble=%.3f diff=%.3f >= %.2f)",
                        market_id,
                        opp.gaussian_prob,
                        opp.ensemble_prob,
                        disagreement,
                        block_threshold,
                    )
                    continue
                if disagreement >= haircut_threshold:
                    haircut = getattr(self.settings, "model_disagreement_haircut_multiplier", 0.50)
                    caution_haircut = min(caution_haircut, haircut)
                    log.warning(
                        "Model-disagreement HAIRCUT: market=%s side=%s diff=%.3f >= %.2f (x%.2f)",
                        market_id,
                        majority_side,
                        disagreement,
                        haircut_threshold,
                        haircut,
                    )
            if (
                opp is not None
                and getattr(self.settings, "qualitative_caution_guard_enabled", True)
            ):
                conservative_abstained = any(
                    sig.agent_name == "Conservative" and sig.side == "no_trade"
                    for sig in signals
                )
                qualitative_against = any(
                    sig.agent_name == "Qualitative"
                    and _is_opposing_signal(sig.side, majority_side)
                    for sig in signals
                )
                if (
                    conservative_abstained
                    and qualitative_against
                    and _is_fragile_entry_context(opp, majority_side)
                ):
                    caution_haircut = getattr(
                        self.settings,
                        "qualitative_caution_haircut_multiplier",
                        0.50,
                    )
            if (
                opp is not None
                and getattr(self.settings, "weak_long_buy_yes_guard_enabled", True)
                and majority_side == "buy_yes"
            ):
                is_threshold_market = bucket_low is None and bucket_high is None
                threshold_only = getattr(self.settings, "weak_long_buy_yes_threshold_only", True)
                min_lead_days = getattr(self.settings, "weak_long_buy_yes_min_lead_days", 1)
                min_probability = getattr(self.settings, "weak_long_buy_yes_min_probability", 0.50)
                if (
                    (not threshold_only or is_threshold_market)
                    and getattr(opp, "lead_time_days", 0) > min_lead_days
                    and model_probability < min_probability
                ):
                    log.warning(
                        "Weak-long-YES BLOCK: market=%s skipped buy_yes (lead_days=%s, model_p=%.3f < %.2f)",
                        market_id,
                        getattr(opp, "lead_time_days", None),
                        model_probability,
                        min_probability,
                    )
                    continue
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
                    "market_type": market_type,
                    "volume": volume,
                    "bucket_low": bucket_low,
                    "bucket_high": bucket_high,
                    "effective_edge": effective_edge,
                    "majority_score": majority_side_score,
                    "side_margin": majority_side_score - minority_side_score,
                    "caution_haircut": caution_haircut,
                    "raw_model_prob": getattr(opp, "raw_model_prob", None) if opp else None,
                }
            )

        if getattr(self.settings, "same_cycle_duplicate_guard_enabled", True):
            candidates = _filter_same_cycle_duplicates(candidates)

        # Prioritize strongest opportunities first; this matters when risk caps
        # stop processing before all candidates can be accepted. Rank by the
        # majority side's score (what the gate above also checks) so the
        # strongest-conviction market wins the slot, not the noisiest one.
        candidates.sort(
            key=lambda c: (
                c["majority_score"],
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
            market_type = candidate.get("market_type", "daily_high")
            volume = candidate.get("volume")
            bucket_low = candidate.get("bucket_low")
            bucket_high = candidate.get("bucket_high")
            effective_edge = candidate["effective_edge"]
            caution_haircut = float(candidate.get("caution_haircut", 1.0) or 1.0)
            raw_model_prob = candidate.get("raw_model_prob")

            if (
                opp is not None
                and getattr(self.settings, "bucket_raw_prob_floor_guard_enabled", True)
                and majority_side == "buy_no"
                and bucket_low is not None
                and bucket_high is not None
                and raw_model_prob is not None
                and float(raw_model_prob)
                < getattr(self.settings, "bucket_raw_prob_floor_guard_min_raw_yes", 0.01)
            ):
                log.warning(
                    "Raw-prob floor BLOCK: market=%s skipped buy_no bucket "
                    "(raw_yes=%.4f < %.4f, model_p=%.3f, market_price=%.3f, edge=%.3f, "
                    "station=%s date=%s type=%s)",
                    market_id,
                    float(raw_model_prob),
                    getattr(self.settings, "bucket_raw_prob_floor_guard_min_raw_yes", 0.01),
                    model_probability,
                    market_price,
                    effective_edge,
                    station,
                    market_date,
                    market_type,
                )
                continue

            if (
                market_type == "daily_low"
                and not getattr(self.settings, "daily_low_entry_enabled", False)
            ):
                log.warning(
                    "Daily-low entry BLOCK: market=%s skipped until true daily-min forecasts are available",
                    market_id,
                )
                continue

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

            # YES calibration guard-rail (April-2026 audit) — HARD BLOCK tier.
            # Historical daily_high/daily_low buy_yes: n=33, mean predicted 0.556,
            # actual 0.030 — effectively zero wins above predicted 0.50. Extreme
            # conviction (model_p > max_prob, default 0.70) is nearly always a
            # data/calibration artefact, not a real edge. Applies only to
            # threshold markets (bucket_low/high both None); bucket markets and
            # buy_no are always exempt.
            is_threshold_market = bucket_low is None and bucket_high is None
            guard_enabled = getattr(self.settings, "yes_guard_rail_enabled", True)
            guard_max_prob = getattr(self.settings, "yes_guard_rail_max_prob", 0.70)
            guard_haircut_prob = getattr(self.settings, "yes_guard_rail_haircut_prob", 0.50)
            yes_on_threshold = (
                guard_enabled
                and majority_side == "buy_yes"
                and is_threshold_market
            )
            if yes_on_threshold and model_probability > guard_max_prob:
                log.warning(
                    "YES guard-rail BLOCK: market=%s skipped buy_yes (model_p=%.3f > %.2f on threshold market)",
                    market_id,
                    model_probability,
                    guard_max_prob,
                )
                continue

            capital_base = equity_base

            preliminary_size = self.risk_manager.calculate_position_size(
                win_probability=model_probability,
                market_price=market_price,
                bankroll=capital_base,
                side=majority_side,
                volume=volume,
                platform=platform or "kalshi",
            )
            if caution_haircut < 1.0:
                pre_haircut = preliminary_size
                preliminary_size = round(preliminary_size * caution_haircut, 2)
                log.warning(
                    "Qualitative caution HAIRCUT: market=%s side=%s preliminary size %.2f→%.2f "
                    "(x%.2f; Conservative=no_trade, Qualitative opposed, fragile context)",
                    market_id,
                    majority_side,
                    pre_haircut,
                    preliminary_size,
                    caution_haircut,
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
                platform=platform or "kalshi",
            )
            if caution_haircut < 1.0:
                amount = round(amount * caution_haircut, 2)

            # Step 5: VPIN reduction
            if risk_result.get("reduce_size"):
                amount = round(amount / 2, 2)

            # YES calibration guard-rail (April-2026 audit) — HAIRCUT tier.
            # For buy_yes on threshold markets with model_p in (haircut_prob, max_prob]
            # (default 0.50 < p ≤ 0.70), apply a Kelly haircut (default 0.15x) rather
            # than blocking outright. Rationale: we keep collecting calibration
            # data in the gray zone while capping downside until the YES curve
            # is re-validated. buy_no, bucket markets, and low-conviction YES
            # (p ≤ 0.50) are never touched.
            if yes_on_threshold and model_probability > guard_haircut_prob:
                haircut = getattr(self.settings, "yes_guard_rail_haircut_multiplier", 0.15)
                pre_haircut = amount
                amount = round(amount * haircut, 2)
                log.warning(
                    "YES guard-rail HAIRCUT: market=%s buy_yes size %.2f→%.2f (x%.2f, model_p=%.3f in (%.2f, %.2f])",
                    market_id,
                    pre_haircut,
                    amount,
                    haircut,
                    model_probability,
                    guard_haircut_prob,
                    guard_max_prob,
                )

            # CCG-bug-8: hard-cap by remaining free cash so cumulative orders within
            # one cycle cannot deploy more than the user actually has. Round DOWN
            # to nearest 0.01 contract to avoid floating-point overshoot.
            # CCG-bughunt2 Codex F9: PaperTrader charges Kalshi taker fees on top
            # of the raw contract price, so the cap must include that fee per
            # contract, otherwise the cash-capped order still overshoots and
            # gets rejected downstream.
            base_contract_cost = (1.0 - market_price) if majority_side == "buy_no" else market_price
            fee_per_contract = 0.0
            if (platform or "kalshi") == "kalshi":
                from weatheredge.probability.edge import kalshi_taker_fee
                fee_per_contract = kalshi_taker_fee(market_price)
            per_contract_cost = base_contract_cost + fee_per_contract
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

            is_bucket_market = bool(
                opp
                and opp.bucket_low is not None
                and opp.bucket_high is not None
            )
            regime = classify_regime(
                side=majority_side,
                is_bucket=is_bucket_market,
                raw_model_prob=opp.raw_model_prob if opp else None,
                metar_override=getattr(opp, "metar_override", None) if opp else None,
            )

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
                forecast_timestamp=getattr(opp, "forecast_timestamp", None) if opp else None,
                market_price_timestamp=getattr(opp, "market_price_timestamp", None) if opp else None,
                regime=regime,
            )
            orders.append(order)

            # Reserve this order in projected exposure so subsequent markets in the
            # same cycle cannot exceed position-count or exposure caps.
            # CCG-bughunt3 G4-new: the free-cash cap (line 395) uses per_contract_cost
            # (fee-inclusive) but this reservation used only raw contract cost. On
            # multi-candidate cycles the second+ order got an inflated budget and
            # was rejected downstream by paper_trader, whose `cost = stake + entry_fee`
            # matches per_contract_cost. Use the same fee-inclusive value here.
            projected_cost = per_contract_cost * amount
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
