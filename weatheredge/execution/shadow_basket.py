"""Basket-level shadow simulation for mutually exclusive bucket-NO trades.

The basket ledger is deliberately isolated from PaperTrader. It records a
synthetic full-basket buy_no entry only when several daily-high bucket markets
share the same station/date event and the terminal payoff surface remains
positive after conservative fill adjustments.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from typing import Iterable

from sqlalchemy.exc import IntegrityError

from weatheredge.agents.base import MarketOpportunity
from weatheredge.config import Settings
from weatheredge.db.database import get_session
from weatheredge.db.models import ShadowBasketLeg, ShadowBasketTrade
from weatheredge.execution.shadow_twin import (
    _as_utc_aware,
    _has_causal_timestamps,
    expanded_shadow_stations,
    is_floor_bound_buy_no_candidate,
)
from weatheredge.probability.edge import kalshi_taker_fee

log = logging.getLogger(__name__)

SHADOW_BASKET_POLICY = "expanded_no_basket_v1"
SIZING_EQUAL_CONTRACT = "equal_contract_minimax_v1"
EXECUTION_MODEL = "buy_no_quote_plus_adverse_tick_v1"


@dataclass(frozen=True)
class BasketLegPlan:
    opp: MarketOpportunity
    entry_no_price: float
    fill_no_price: float
    fee_per_contract: float
    per_contract_cost: float


@dataclass(frozen=True)
class BasketPlan:
    policy_name: str
    basket_id: str
    event_identity: str
    legs: tuple[BasketLegPlan, ...]
    amount_per_leg: float
    cost: float
    payoff_none: float
    worst_case_payoff: float
    worst_case_threshold: float
    expected_pnl: float
    stressed_expected_pnl: float
    market_yes_mass: float
    model_yes_mass: float
    scenario_payoffs: dict[str, float]
    score: float


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _event_identity(opp: MarketOpportunity) -> str:
    return "|".join(
        [
            str(opp.platform or "unknown").lower(),
            str(opp.station or "").upper(),
            str(opp.market_date or ""),
            str(opp.market_type or ""),
            "unit=F",
            "bucket_bounds=display",
        ]
    )


def _basket_id(event_identity: str) -> str:
    digest = hashlib.sha1(f"{SHADOW_BASKET_POLICY}|{event_identity}".encode("utf-8")).hexdigest()[:12]
    readable = event_identity.replace("|", ":").replace("=", "-")
    return f"{SHADOW_BASKET_POLICY}:{readable}:{digest}"


def _entry_no_quote(opp: MarketOpportunity, settings: Settings) -> tuple[float, float, float] | None:
    yes_price = _optional_float(getattr(opp, "market_price", None))
    if yes_price is None or yes_price <= 0.0 or yes_price >= 1.0:
        return None

    candidates = [1.0 - yes_price]

    no_quote = _optional_float(getattr(opp, "no_market_price", None))
    if no_quote is not None and 0.0 < no_quote < 1.0:
        candidates.append(no_quote)

    # On binary books, an executable NO buy is conservatively approximated by
    # the complement of the YES bid when only YES-side order-book data exists.
    best_bid = _optional_float(getattr(opp, "best_bid", None))
    if best_bid is not None and 0.0 < best_bid < 1.0:
        candidates.append(1.0 - best_bid)

    entry_no_price = max(candidates)
    adverse_tick = max(0.0, float(getattr(settings, "shadow_basket_adverse_tick", 0.01)))
    fill_no_price = min(0.999, entry_no_price + adverse_tick)
    fee_per_contract = 0.0
    if str(opp.platform or "").lower() == "kalshi":
        fee_per_contract = kalshi_taker_fee(fill_no_price)
    per_contract_cost = fill_no_price + fee_per_contract
    if per_contract_cost <= 0.0 or per_contract_cost >= 1.0:
        return None
    return entry_no_price, fill_no_price, fee_per_contract


def _candidate_leg(opp: MarketOpportunity, settings: Settings) -> BasketLegPlan | None:
    if str(opp.market_type or "") != "daily_high":
        return None
    if str(opp.station or "").upper() not in expanded_shadow_stations(settings):
        return None
    if getattr(opp, "bucket_invalidated_for_yes", False):
        return None
    if not _has_causal_timestamps(opp):
        return None
    if not is_floor_bound_buy_no_candidate(opp, settings):
        return None

    min_edge = float(getattr(settings, "shadow_basket_min_net_edge", 0.10))
    if _optional_float(getattr(opp, "net_edge_no", None)) is None or float(opp.net_edge_no) < min_edge:
        return None

    forecast_temp = _optional_float(getattr(opp, "forecast_temp", None))
    if forecast_temp is None:
        return None

    volume = _optional_float(getattr(opp, "volume", None))
    spread = _optional_float(getattr(opp, "spread", None))
    if volume is None or spread is None:
        return None
    if volume < float(getattr(settings, "shadow_basket_min_volume", 300.0)):
        return None
    if spread > float(getattr(settings, "shadow_basket_max_spread_pct", 0.05)):
        return None

    quote = _entry_no_quote(opp, settings)
    if quote is None:
        return None
    entry_no_price, fill_no_price, fee_per_contract = quote
    return BasketLegPlan(
        opp=opp,
        entry_no_price=entry_no_price,
        fill_no_price=fill_no_price,
        fee_per_contract=fee_per_contract,
        per_contract_cost=fill_no_price + fee_per_contract,
    )


def _sort_key(leg: BasketLegPlan) -> tuple[float, str]:
    low = _optional_float(getattr(leg.opp, "bucket_low", None))
    return (low if low is not None else 9999.0, str(leg.opp.market_id))


def _is_contiguous(legs: Iterable[BasketLegPlan]) -> bool:
    ordered = sorted(legs, key=_sort_key)
    for prev, cur in zip(ordered, ordered[1:]):
        prev_high = getattr(prev.opp, "bucket_high_int", None)
        cur_low = getattr(cur.opp, "bucket_low_int", None)
        if prev_high is None or cur_low is None:
            continue
        if int(cur_low) > int(prev_high) + 1:
            return False
    return True


def _candidate_windows(legs: list[BasketLegPlan], settings: Settings) -> Iterable[tuple[BasketLegPlan, ...]]:
    ordered = sorted(legs, key=_sort_key)
    min_legs = max(3, int(getattr(settings, "shadow_basket_min_legs", 3)))
    max_legs = max(min_legs, int(getattr(settings, "shadow_basket_max_legs", 6)))
    max_legs = min(max_legs, len(ordered))
    require_contiguous = bool(getattr(settings, "shadow_basket_require_contiguous", True))

    if require_contiguous:
        for size in range(min_legs, max_legs + 1):
            for start in range(0, len(ordered) - size + 1):
                window = tuple(ordered[start:start + size])
                if _is_contiguous(window):
                    yield window
        return

    for size in range(min_legs, max_legs + 1):
        for combo in combinations(ordered, size):
            yield tuple(combo)


def _plan_for_legs(legs: tuple[BasketLegPlan, ...], settings: Settings) -> BasketPlan | None:
    if len(legs) < 3:
        return None

    sum_contract_cost = sum(leg.per_contract_cost for leg in legs)
    if sum_contract_cost <= 0.0:
        return None

    target_cost = min(
        float(getattr(settings, "shadow_basket_max_cost", 6.0)),
        float(getattr(settings, "shadow_basket_cost_per_leg", 1.0)) * len(legs),
    )
    if target_cost <= 0.0:
        return None

    amount = target_cost / sum_contract_cost
    cost = amount * sum_contract_cost
    payoff_none = amount * len(legs) - cost
    scenario_payoffs = {
        str(leg.opp.market_id): payoff_none - amount
        for leg in legs
    }
    worst_case = min([payoff_none, *scenario_payoffs.values()])
    threshold = max(
        float(getattr(settings, "shadow_basket_min_worst_case_abs", 0.20)),
        float(getattr(settings, "shadow_basket_min_worst_case_fraction", 0.05)) * cost,
    )
    if worst_case < threshold:
        return None

    no_hit_threshold = float(getattr(settings, "shadow_basket_min_no_hit_fraction", 0.20)) * cost
    if payoff_none < no_hit_threshold:
        return None

    stress_add = max(0.0, float(getattr(settings, "shadow_basket_stress_yes_prob_add", 0.05)))
    expected = 0.0
    stressed = 0.0
    market_yes_mass = 0.0
    model_yes_mass = 0.0
    min_volume = math.inf
    max_spread = 0.0
    for leg in legs:
        raw_prob = _optional_float(getattr(leg.opp, "raw_model_prob", None))
        if raw_prob is None:
            raw_prob = _optional_float(getattr(leg.opp, "model_probability", None)) or 0.0
        raw_prob = min(max(raw_prob, 0.0), 1.0)
        stressed_prob = min(1.0, raw_prob + stress_add)
        expected += amount * ((1.0 - raw_prob) - leg.per_contract_cost)
        stressed += amount * ((1.0 - stressed_prob) - leg.per_contract_cost)
        market_yes_mass += float(leg.opp.market_price)
        model_yes_mass += raw_prob
        volume = _optional_float(getattr(leg.opp, "volume", None))
        spread = _optional_float(getattr(leg.opp, "spread", None))
        if volume is not None:
            min_volume = min(min_volume, volume)
        if spread is not None:
            max_spread = max(max_spread, spread)

    ev_threshold = float(getattr(settings, "shadow_basket_min_ev_fraction", 0.08)) * cost
    if expected < ev_threshold:
        return None
    if stressed < 0.0:
        return None

    liquidity_quality = 0.0
    if min_volume < math.inf:
        liquidity_quality = min(1.0, min_volume / max(1.0, 20.0 * cost))
    if max_spread:
        liquidity_quality = max(0.0, liquidity_quality - max_spread)

    score = (
        0.35 * (worst_case / cost)
        + 0.25 * (payoff_none / cost)
        + 0.25 * (stressed / cost)
        + 0.15 * liquidity_quality
    )

    event_identity = _event_identity(legs[0].opp)
    return BasketPlan(
        policy_name=SHADOW_BASKET_POLICY,
        basket_id=_basket_id(event_identity),
        event_identity=event_identity,
        legs=legs,
        amount_per_leg=amount,
        cost=cost,
        payoff_none=payoff_none,
        worst_case_payoff=worst_case,
        worst_case_threshold=threshold,
        expected_pnl=expected,
        stressed_expected_pnl=stressed,
        market_yes_mass=market_yes_mass,
        model_yes_mass=model_yes_mass,
        scenario_payoffs=scenario_payoffs,
        score=score,
    )


def build_shadow_basket_plan(
    opportunities: dict[str, MarketOpportunity],
    settings: Settings,
) -> list[BasketPlan]:
    """Build at most one passing basket per station/date event identity."""
    grouped: dict[str, list[BasketLegPlan]] = {}
    for opp in opportunities.values():
        leg = _candidate_leg(opp, settings)
        if leg is None:
            continue
        grouped.setdefault(_event_identity(opp), []).append(leg)

    plans: list[BasketPlan] = []
    for legs in grouped.values():
        best: BasketPlan | None = None
        for window in _candidate_windows(legs, settings):
            plan = _plan_for_legs(window, settings)
            if plan is None:
                continue
            if best is None or plan.score > best.score:
                best = plan
        if best is not None:
            plans.append(best)
    return plans


def _min_datetime(values: Iterable[datetime | None]) -> datetime | None:
    parsed = [_as_utc_aware(v) for v in values if isinstance(v, datetime)]
    return min(parsed) if parsed else None


def _max_datetime(values: Iterable[datetime | None]) -> datetime | None:
    parsed = [_as_utc_aware(v) for v in values if isinstance(v, datetime)]
    return max(parsed) if parsed else None


def _insert_plan(plan: BasketPlan) -> bool:
    first = plan.legs[0].opp
    forecast_ts = _max_datetime(leg.opp.forecast_timestamp for leg in plan.legs)
    price_ts = _min_datetime(leg.opp.market_price_timestamp for leg in plan.legs)
    if forecast_ts is not None and price_ts is not None and forecast_ts > price_ts:
        log.info(
            "shadow_basket: skipped %s — basket forecast_ts=%s newer than earliest price_ts=%s",
            plan.event_identity,
            forecast_ts.isoformat(),
            price_ts.isoformat(),
        )
        return False

    lows = [_optional_float(getattr(leg.opp, "bucket_low", None)) for leg in plan.legs]
    highs = [_optional_float(getattr(leg.opp, "bucket_high", None)) for leg in plan.legs]
    volumes = [_optional_float(getattr(leg.opp, "volume", None)) for leg in plan.legs]
    spreads = [_optional_float(getattr(leg.opp, "spread", None)) for leg in plan.legs]
    scenario_json = json.dumps(plan.scenario_payoffs, sort_keys=True, ensure_ascii=True)

    try:
        with get_session() as session:
            existing = session.query(ShadowBasketTrade.id).filter_by(
                policy_name=plan.policy_name,
                event_identity=plan.event_identity,
            ).first()
            if existing is not None:
                return False

            session.add(ShadowBasketTrade(
                basket_id=plan.basket_id,
                policy_name=plan.policy_name,
                event_identity=plan.event_identity,
                platform=str(first.platform or "unknown"),
                station=str(first.station or ""),
                market_date=str(first.market_date or ""),
                market_type=str(first.market_type or ""),
                sizing_method=SIZING_EQUAL_CONTRACT,
                execution_model=EXECUTION_MODEL,
                leg_count=len(plan.legs),
                amount_per_leg=round(plan.amount_per_leg, 6),
                cost=round(plan.cost, 6),
                payoff_none=round(plan.payoff_none, 6),
                worst_case_payoff=round(plan.worst_case_payoff, 6),
                worst_case_threshold=round(plan.worst_case_threshold, 6),
                expected_pnl=round(plan.expected_pnl, 6),
                stressed_expected_pnl=round(plan.stressed_expected_pnl, 6),
                market_yes_mass=round(plan.market_yes_mass, 6),
                model_yes_mass=round(plan.model_yes_mass, 6),
                min_bucket_low=min(v for v in lows if v is not None) if any(v is not None for v in lows) else None,
                max_bucket_high=max(v for v in highs if v is not None) if any(v is not None for v in highs) else None,
                min_entry_volume=min(v for v in volumes if v is not None) if any(v is not None for v in volumes) else None,
                max_entry_spread=max(v for v in spreads if v is not None) if any(v is not None for v in spreads) else None,
                scenario_payoffs=scenario_json,
                forecast_timestamp=forecast_ts,
                market_price_timestamp=price_ts,
                status="open",
            ))

            for leg in plan.legs:
                opp = leg.opp
                leg_cost = plan.amount_per_leg * leg.per_contract_cost
                session.add(ShadowBasketLeg(
                    basket_id=plan.basket_id,
                    market_id=str(opp.market_id),
                    platform=str(opp.platform or "unknown"),
                    side="buy_no",
                    entry_yes_price=float(opp.market_price),
                    entry_no_price=round(leg.entry_no_price, 6),
                    fill_no_price=round(leg.fill_no_price, 6),
                    fee_per_contract=round(leg.fee_per_contract, 6),
                    amount=round(plan.amount_per_leg, 6),
                    cost=round(leg_cost, 6),
                    model_probability=float(opp.model_probability) if opp.model_probability is not None else None,
                    raw_model_prob=float(opp.raw_model_prob) if opp.raw_model_prob is not None else None,
                    net_edge_at_entry=float(opp.net_edge_no),
                    station=str(opp.station or ""),
                    market_date=str(opp.market_date or ""),
                    bucket_low=float(opp.bucket_low) if opp.bucket_low is not None else None,
                    bucket_high=float(opp.bucket_high) if opp.bucket_high is not None else None,
                    entry_volume=_optional_float(opp.volume),
                    entry_spread=_optional_float(getattr(opp, "spread", None)),
                    forecast_timestamp=opp.forecast_timestamp,
                    market_price_timestamp=opp.market_price_timestamp,
                    status="open",
                ))
        log.info(
            "shadow_basket: recorded %s station=%s date=%s legs=%d cost=%.2f worst=%.3f none=%.3f ev=%.3f",
            plan.policy_name,
            first.station,
            first.market_date,
            len(plan.legs),
            plan.cost,
            plan.worst_case_payoff,
            plan.payoff_none,
            plan.expected_pnl,
        )
        return True
    except IntegrityError:
        return False
    except Exception as exc:  # noqa: BLE001 - shadow must never block main trading
        log.warning("shadow_basket: record failed for %s: %s", plan.event_identity, exc)
        return False


def record_shadow_baskets(
    opportunities: dict[str, MarketOpportunity],
    settings: Settings,
) -> dict[str, int]:
    """Record passing Expanded bucket-NO baskets from built opportunities."""
    counts = {SHADOW_BASKET_POLICY: 0}
    if not getattr(settings, "shadow_twin_enabled", True):
        return counts
    if not getattr(settings, "shadow_basket_enabled", True):
        return counts
    for plan in build_shadow_basket_plan(opportunities, settings):
        if _insert_plan(plan):
            counts[SHADOW_BASKET_POLICY] += 1
    return counts


def resolve_shadow_baskets_for_market(market_id: str, outcome: str) -> int:
    """Resolve open basket legs for one market and close complete baskets."""
    if outcome not in {"yes", "no"}:
        return 0

    now = datetime.now(timezone.utc)
    resolved = 0
    affected_baskets: set[str] = set()
    try:
        with get_session() as session:
            rows = session.query(ShadowBasketLeg).filter_by(
                market_id=str(market_id),
                status="open",
            ).all()
            for row in rows:
                won = (row.side == "buy_no" and outcome == "no") or (
                    row.side == "buy_yes" and outcome == "yes"
                )
                row.status = "won" if won else "lost"
                row.outcome = outcome
                row.pnl = round((float(row.amount) - float(row.cost)) if won else -float(row.cost), 6)
                row.resolved_at = now
                affected_baskets.add(str(row.basket_id))
                resolved += 1

            for basket_id in affected_baskets:
                legs = session.query(ShadowBasketLeg).filter_by(basket_id=basket_id).all()
                if not legs or any(leg.status == "open" for leg in legs):
                    continue
                basket = session.query(ShadowBasketTrade).filter_by(
                    basket_id=basket_id,
                    status="open",
                ).one_or_none()
                if basket is None:
                    continue
                total_pnl = round(sum(float(leg.pnl or 0.0) for leg in legs), 6)
                basket.status = "won" if total_pnl >= 0.0 else "lost"
                basket.pnl = total_pnl
                basket.resolved_at = now
                losing_leg = next((leg for leg in legs if leg.outcome == "yes"), None)
                basket.outcome_market_id = str(losing_leg.market_id) if losing_leg is not None else None
                resolved += 1

        if resolved:
            log.info("shadow_basket: resolved %d row(s) for %s -> %s", resolved, market_id, outcome)
        return resolved
    except Exception as exc:  # noqa: BLE001 - resolution should not break main job
        log.warning("shadow_basket: resolve failed for %s: %s", market_id, exc)
        return 0
