"""Isolated paper-variant ledger for main-like counterfactual trades."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError

from weatheredge.agents.base import MarketOpportunity, TradeOrder
from weatheredge.db.database import get_session
from weatheredge.db.models import PaperVariantTrade
from weatheredge.probability.edge import kalshi_taker_fee

log = logging.getLogger(__name__)

VARIANT_MAIN_ADAPTIVE_CORE = "main_adaptive_core"
VARIANT_MAIN_ADAPTIVE_EXPANDED = "main_adaptive_expanded"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _order_cost(order: TradeOrder) -> float:
    if order.side == "buy_no":
        stake = (1.0 - float(order.price)) * float(order.amount)
    else:
        stake = float(order.price) * float(order.amount)
    fee = 0.0
    if (order.platform or "").lower() == "kalshi":
        fee = kalshi_taker_fee(float(order.price)) * float(order.amount)
    return round(stake + fee, 6)


def record_paper_variant_orders(
    variant_name: str,
    trade_orders: list[TradeOrder],
    opportunities: dict[str, MarketOpportunity],
    *,
    liquidity_min_volume: float | None = None,
) -> int:
    """Persist paper-variant market fills without touching PaperTrader state."""
    if not trade_orders:
        return 0

    inserted = 0
    try:
        with get_session() as session:
            for order in trade_orders:
                existing = session.query(PaperVariantTrade.id).filter_by(
                    variant_name=variant_name,
                    market_id=str(order.market_id),
                ).first()
                if existing is not None:
                    continue

                opp = opportunities.get(str(order.market_id))
                cost = _order_cost(order)
                if cost <= 0.0 or float(order.amount) <= 0.0:
                    continue

                session.add(PaperVariantTrade(
                    variant_name=variant_name,
                    market_id=str(order.market_id),
                    platform=str(order.platform or "unknown"),
                    side=str(order.side),
                    entry_yes_price=float(order.price),
                    amount=float(order.amount),
                    cost=cost,
                    agent_name=str(order.agent_name or ""),
                    model_probability=_optional_float(order.model_probability),
                    gaussian_prob=_optional_float(order.gaussian_prob),
                    ensemble_prob=_optional_float(order.ensemble_prob),
                    blended_prob=_optional_float(order.blended_prob),
                    raw_model_prob=_optional_float(order.raw_model_prob),
                    edge_at_entry=_optional_float(order.edge_at_entry),
                    station=str(order.station or ""),
                    market_date=str(order.market_date or ""),
                    bucket_low=_optional_float(getattr(opp, "bucket_low", None)),
                    bucket_high=_optional_float(getattr(opp, "bucket_high", None)),
                    entry_volume=_optional_float(getattr(opp, "volume", None)),
                    entry_spread=_optional_float(getattr(opp, "spread", None)),
                    liquidity_min_volume=liquidity_min_volume,
                    forecast_timestamp=order.forecast_timestamp,
                    market_price_timestamp=order.market_price_timestamp,
                    status="open",
                ))
                inserted += 1
        if inserted:
            log.info("paper_variant: recorded %s %d order(s)", variant_name, inserted)
        return inserted
    except IntegrityError:
        return inserted
    except Exception as exc:  # noqa: BLE001 - variants must never block main
        log.warning("paper_variant: record failed for %s: %s", variant_name, exc)
        return inserted


def load_open_variant_positions(variant_name: str) -> list[dict]:
    """Return open variant rows in the dict shape PortfolioManager expects."""
    try:
        with get_session() as session:
            rows = (
                session.query(PaperVariantTrade)
                .filter_by(variant_name=variant_name, status="open")
                .all()
            )
            return [
                {
                    "market_id": str(row.market_id),
                    "side": str(row.side),
                    "price": float(row.entry_yes_price),
                    "amount": float(row.amount),
                    "cost": float(row.cost),
                    "agent_name": row.agent_name,
                    "model_probability": row.model_probability,
                    "edge_at_entry": row.edge_at_entry,
                    "platform": row.platform,
                    "station": row.station,
                    "market_date": row.market_date,
                    "bucket_low": row.bucket_low,
                    "bucket_high": row.bucket_high,
                    "status": row.status,
                }
                for row in rows
            ]
    except Exception as exc:  # noqa: BLE001
        log.debug("paper_variant: open position load failed for %s: %s", variant_name, exc)
        return []


def paper_variant_pnl_state(variant_name: str) -> tuple[float, float]:
    """Return realized total P&L and current paper cash estimate for a variant."""
    try:
        with get_session() as session:
            rows = (
                session.query(PaperVariantTrade)
                .filter(PaperVariantTrade.variant_name == variant_name)
                .all()
            )
            realized = sum(float(row.pnl or 0.0) for row in rows if row.status != "open")
            open_cost = sum(float(row.cost or 0.0) for row in rows if row.status == "open")
            return realized, open_cost
    except Exception as exc:  # noqa: BLE001
        log.debug("paper_variant: pnl state failed for %s: %s", variant_name, exc)
        return 0.0, 0.0


def resolve_paper_variant_trades_for_market(market_id: str, outcome: str) -> int:
    """Resolve open paper-variant rows for one YES-side market outcome."""
    if outcome not in {"yes", "no"}:
        return 0
    now = datetime.now(timezone.utc)
    resolved = 0
    try:
        with get_session() as session:
            rows = session.query(PaperVariantTrade).filter_by(
                market_id=str(market_id),
                status="open",
            ).all()
            for row in rows:
                won = (row.side == "buy_no" and outcome == "no") or (
                    row.side == "buy_yes" and outcome == "yes"
                )
                row.status = "won" if won else "lost"
                row.outcome = outcome
                row.pnl = (float(row.amount) - float(row.cost)) if won else -float(row.cost)
                row.resolved_at = now
                resolved += 1
        if resolved:
            log.info("paper_variant: resolved %d trade(s) for %s -> %s", resolved, market_id, outcome)
        return resolved
    except Exception as exc:  # noqa: BLE001
        log.warning("paper_variant: resolve failed for %s: %s", market_id, exc)
        return 0
