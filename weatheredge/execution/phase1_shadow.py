"""Phase-1 shadow-only rejection ledger.

Records main-bot candidates intentionally blocked by Phase-1 guards so their
counterfactual performance can be audited without routing them through
PaperTrader or the shadow-twin floor-only filter.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError

from weatheredge.agents.base import MarketOpportunity
from weatheredge.db.database import get_session
from weatheredge.db.models import Phase1ShadowRejection
from weatheredge.probability.edge import kalshi_taker_fee

log = logging.getLogger(__name__)

PHASE1_SHADOW_POLICY = "phase1_same_day_narrow_bucket_no_v1"
PHASE1_SHADOW_BUDGET = 1.0


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _entry_amount_and_cost(opp: MarketOpportunity) -> tuple[float, float, float, float] | None:
    yes_price = _optional_float(getattr(opp, "market_price", None))
    if yes_price is None or yes_price <= 0.0 or yes_price >= 1.0:
        return None

    no_price = 1.0 - yes_price
    fee_per_contract = 0.0
    if str(getattr(opp, "platform", "") or "").lower() == "kalshi":
        fee_per_contract = kalshi_taker_fee(no_price)
    per_contract_cost = no_price + fee_per_contract
    if per_contract_cost <= 0.0:
        return None

    amount = PHASE1_SHADOW_BUDGET / per_contract_cost
    cost = amount * per_contract_cost
    return round(no_price, 6), round(fee_per_contract, 6), round(amount, 6), round(cost, 6)


def record_phase1_shadow_only_rejection(
    opp: MarketOpportunity,
    *,
    reason: str,
    majority_side: str,
    majority_score: float,
    side_margin: float,
    distinct_majority: int,
) -> bool:
    """Record or refresh one intentionally skipped Phase-1 shadow-only entry.

    Duplicate cycles update observation metadata only. The first executable
    price remains the counterfactual entry price.
    """
    if majority_side != "buy_no":
        return False

    quote = _entry_amount_and_cost(opp)
    if quote is None:
        return False
    entry_no_price, fee_per_contract, amount, cost = quote

    now = datetime.now(timezone.utc)
    try:
        with get_session() as session:
            existing = session.query(Phase1ShadowRejection).filter_by(
                policy_name=PHASE1_SHADOW_POLICY,
                market_id=str(opp.market_id),
            ).one_or_none()
            if existing is not None:
                existing.last_seen_at = now
                existing.observation_count = int(existing.observation_count or 0) + 1
                existing.majority_score = float(majority_score)
                existing.side_margin = float(side_margin)
                existing.distinct_majority = int(distinct_majority)
                return False

            session.add(Phase1ShadowRejection(
                policy_name=PHASE1_SHADOW_POLICY,
                market_id=str(opp.market_id),
                platform=str(opp.platform or "unknown"),
                side="buy_no",
                reason=str(reason),
                entry_yes_price=float(opp.market_price),
                entry_no_price=entry_no_price,
                fee_per_contract=fee_per_contract,
                amount=amount,
                cost=cost,
                model_probability=_optional_float(getattr(opp, "model_probability", None)),
                raw_model_prob=_optional_float(getattr(opp, "raw_model_prob", None)),
                net_edge_at_entry=_optional_float(getattr(opp, "net_edge_no", None)),
                net_edge_yes=_optional_float(getattr(opp, "net_edge", None)),
                station=str(opp.station or ""),
                market_date=str(opp.market_date or ""),
                market_type=str(getattr(opp, "market_type", "") or ""),
                bucket_low=_optional_float(getattr(opp, "bucket_low", None)),
                bucket_high=_optional_float(getattr(opp, "bucket_high", None)),
                forecast_temp=_optional_float(getattr(opp, "forecast_temp", None)),
                entry_volume=_optional_float(getattr(opp, "volume", None)),
                entry_spread=_optional_float(getattr(opp, "spread", None)),
                majority_score=float(majority_score),
                side_margin=float(side_margin),
                distinct_majority=int(distinct_majority),
                forecast_timestamp=getattr(opp, "forecast_timestamp", None),
                market_price_timestamp=getattr(opp, "market_price_timestamp", None),
                status="open",
                observation_count=1,
                first_seen_at=now,
                last_seen_at=now,
                created_at=now,
            ))
        log.info(
            "phase1_shadow: recorded rejection market=%s station=%s date=%s yes=%.3f edge_no=%.3f reason=%s",
            opp.market_id,
            opp.station,
            opp.market_date,
            float(opp.market_price),
            float(getattr(opp, "net_edge_no", 0.0) or 0.0),
            reason,
        )
        return True
    except IntegrityError:
        return False
    except Exception as exc:  # noqa: BLE001 - shadow audit must never block trading
        log.warning("phase1_shadow: record failed for %s: %s", getattr(opp, "market_id", ""), exc)
        return False


def resolve_phase1_shadow_rejections_for_market(market_id: str, outcome: str) -> int:
    """Resolve open Phase-1 shadow-only rejections for one market outcome."""
    if outcome not in {"yes", "no"}:
        return 0

    now = datetime.now(timezone.utc)
    resolved = 0
    try:
        with get_session() as session:
            rows = session.query(Phase1ShadowRejection).filter_by(
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
                resolved += 1

        if resolved:
            log.info("phase1_shadow: resolved %d row(s) for %s -> %s", resolved, market_id, outcome)
        return resolved
    except Exception as exc:  # noqa: BLE001 - resolution should not break main job
        log.warning("phase1_shadow: resolve failed for %s: %s", market_id, exc)
        return 0
