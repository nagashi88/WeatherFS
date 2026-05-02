"""Shadow twin simulations for floor-bound bucket NO trades.

The shadow twins are deliberately separate from PaperTrader. They simulate a
fixed-cost buy_no hold-to-resolution policy and write only to
``shadow_twin_trades`` so the main Phase 1 baseline remains uncontaminated.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy.exc import IntegrityError

from weatheredge.agents.base import MarketOpportunity
from weatheredge.config import Settings
from weatheredge.data.market_parser import CITY_TO_STATION
from weatheredge.db.database import get_session
from weatheredge.db.models import ShadowPolicyTrade, ShadowTwinTrade
from weatheredge.probability.edge import kalshi_taker_fee
from weatheredge.probability.gaussian import PROB_FLOOR

log = logging.getLogger(__name__)

TWIN_CORE = "floor_no_core"
TWIN_EXPANDED = "floor_no_expanded"
TWIN_BROAD = "floor_no_broad"
TWIN_CORE_ADAPTIVE_LIQUIDITY = "floor_no_core_adaptive_liquidity"
TWIN_EXPANDED_ADAPTIVE_LIQUIDITY = "floor_no_expanded_adaptive_liquidity"
TWIN_BROAD_ADAPTIVE_LIQUIDITY = "floor_no_broad_adaptive_liquidity"
OPPORTUNITY_BASKET_FLOOR_NO = "floor_bound_bucket_no"
OPPORTUNITY_BASKET_FLOOR_NO_ADAPTIVE_LIQUIDITY = "floor_bound_bucket_no_adaptive_liquidity"
POLICY_HOLD = "hold_to_resolution"
POLICY_MAIN_EXIT = "main_exit_clone"
POLICY_PRICE_BAND = "price_band_25_45"
POLICY_HIGH_EDGE = "high_edge_15"
SHADOW_POLICIES = (
    POLICY_HOLD,
    POLICY_MAIN_EXIT,
    POLICY_PRICE_BAND,
    POLICY_HIGH_EDGE,
)
PRICE_BAND_MIN_YES = 0.25
PRICE_BAND_MAX_YES = 0.45
HIGH_EDGE_MIN_NO = 0.15


def _station_set(cities: Iterable[str]) -> set[str]:
    stations: set[str] = set()
    for city in cities:
        raw = str(city or "").strip()
        if not raw:
            continue
        upper = raw.upper()
        station = CITY_TO_STATION.get(raw.lower())
        if station:
            stations.add(station)
            continue
        if len(upper) == 4 and upper[0].isalpha():
            stations.add(upper)
    return stations


def core_shadow_stations(settings: Settings) -> set[str]:
    """Stations in the main bot universe."""
    return _station_set(settings.cities_list)


def expanded_shadow_stations(settings: Settings) -> set[str]:
    """Stations in the expanded twin: main universe plus extra cities."""
    return core_shadow_stations(settings) | _station_set(settings.shadow_twin_extended_cities_list)


def broad_shadow_stations(settings: Settings) -> set[str]:
    """Stations in the broad twin: expanded universe plus broad city list."""
    if not getattr(settings, "shadow_twin_broad_enabled", True):
        return set()
    return expanded_shadow_stations(settings) | _station_set(settings.shadow_twin_broad_cities_list)


def is_main_trading_station(station: str, settings: Settings) -> bool:
    """Return True when a station belongs to the main bot target universe."""
    return str(station or "").upper() in core_shadow_stations(settings)


def _as_utc_aware(ts: datetime) -> datetime:
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _has_causal_timestamps(opp: MarketOpportunity) -> bool:
    """Defense-in-depth: shadow ledgers only accept causally timestamped rows."""
    forecast_ts = getattr(opp, "forecast_timestamp", None)
    price_ts = getattr(opp, "market_price_timestamp", None)
    if not isinstance(forecast_ts, datetime) or not isinstance(price_ts, datetime):
        log.info(
            "shadow_twin: skipped %s — missing causal timestamp(s) forecast_ts=%s price_ts=%s",
            getattr(opp, "market_id", ""),
            forecast_ts,
            price_ts,
        )
        return False
    if _as_utc_aware(forecast_ts) > _as_utc_aware(price_ts):
        log.info(
            "shadow_twin: skipped %s — forecast_ts=%s newer than price_ts=%s",
            getattr(opp, "market_id", ""),
            _as_utc_aware(forecast_ts).isoformat(),
            _as_utc_aware(price_ts).isoformat(),
        )
        return False
    return True


def is_floor_bound_buy_no_candidate(opp: MarketOpportunity, settings: Settings) -> bool:
    """Detect the structural floor-bound bucket-NO setup for shadow simulation."""
    if opp.bucket_low is None or opp.bucket_high is None:
        return False
    if opp.metar_override is not None:
        return False
    if opp.raw_model_prob is None:
        return False
    try:
        raw_yes = float(opp.raw_model_prob)
        net_edge_no = float(opp.net_edge_no)
    except (TypeError, ValueError):
        return False
    if raw_yes >= PROB_FLOOR:
        return False
    min_edge = float(getattr(settings, "shadow_twin_min_net_edge", 0.10))
    return net_edge_no >= min_edge


def _shadow_amount_and_cost(opp: MarketOpportunity, settings: Settings) -> tuple[float, float]:
    """Return fixed-budget synthetic NO contracts and fee-inclusive cost."""
    budget = max(0.0, float(getattr(settings, "shadow_twin_cost_per_trade", 1.0)))
    yes_price = float(opp.market_price)
    no_price = 1.0 - yes_price
    if no_price <= 0.0 or budget <= 0.0:
        return 0.0, 0.0
    fee_per_contract = 0.0
    if (opp.platform or "").lower() == "kalshi":
        fee_per_contract = kalshi_taker_fee(yes_price)
    per_contract_cost = no_price + fee_per_contract
    if per_contract_cost <= 0.0:
        return 0.0, 0.0
    amount = budget / per_contract_cost
    cost = amount * per_contract_cost
    return round(amount, 6), round(cost, 6)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def adaptive_liquidity_min_volume(expected_cost: float, settings: Settings) -> float:
    """Return the adaptive shadow-only minimum volume for a synthetic entry."""
    floor = max(0.0, float(getattr(settings, "shadow_twin_adaptive_liquidity_min_volume", 250.0)))
    multiple = max(0.0, float(getattr(settings, "shadow_twin_adaptive_liquidity_volume_multiple", 100.0)))
    return max(floor, max(0.0, float(expected_cost)) * multiple)


def adaptive_liquidity_passes(
    volume: float,
    spread_pct: float,
    expected_cost: float,
    settings: Settings,
) -> tuple[bool, str, float]:
    """Evaluate the shadow-only adaptive liquidity gate.

    This intentionally does not replace the main liquidity filter. It only
    decides whether a low-volume market is worth recording in the isolated
    adaptive shadow experiment.
    """
    try:
        vol = float(volume)
        spread = float(spread_pct)
    except (TypeError, ValueError):
        return False, "adaptive_liquidity_missing_inputs", adaptive_liquidity_min_volume(expected_cost, settings)
    if not math.isfinite(vol) or not math.isfinite(spread):
        return False, "adaptive_liquidity_missing_inputs", adaptive_liquidity_min_volume(expected_cost, settings)
    min_volume = adaptive_liquidity_min_volume(expected_cost, settings)
    if vol < min_volume:
        return False, "adaptive_liquidity_low_volume", min_volume
    max_spread = float(getattr(settings, "shadow_twin_adaptive_liquidity_max_spread_pct", 0.08))
    if spread > max_spread:
        return False, "adaptive_liquidity_wide_spread", min_volume
    return True, "", min_volume


def _policy_allows_entry(policy_name: str, opp: MarketOpportunity) -> bool:
    if policy_name in {POLICY_HOLD, POLICY_MAIN_EXIT}:
        return True
    if policy_name == POLICY_PRICE_BAND:
        yes_price = float(opp.market_price)
        return PRICE_BAND_MIN_YES <= yes_price <= PRICE_BAND_MAX_YES
    if policy_name == POLICY_HIGH_EDGE:
        return float(opp.net_edge_no) >= HIGH_EDGE_MIN_NO
    return False


def _mark_to_market_pnl(row: ShadowPolicyTrade, current_yes_price: float) -> float:
    from weatheredge.probability.edge import kalshi_taker_fee

    if row.side == "buy_yes":
        current_value = current_yes_price * float(row.amount)
    else:
        current_value = (1.0 - current_yes_price) * float(row.amount)
    exit_fee = 0.0
    if (row.platform or "").lower() == "kalshi":
        exit_fee = kalshi_taker_fee(current_yes_price) * float(row.amount)
    return round(current_value - exit_fee - float(row.cost), 6)


def _position_from_policy_row(row: ShadowPolicyTrade) -> dict:
    return {
        "market_id": str(row.market_id),
        "side": str(row.side),
        "price": float(row.entry_yes_price),
        "amount": float(row.amount),
        "cost": float(row.cost),
        "edge_at_entry": float(row.net_edge_at_entry or 0.0),
        "platform": str(row.platform or "unknown"),
        "station": str(row.station or ""),
        "market_date": str(row.market_date or ""),
    }


def record_shadow_twins(
    opportunities: dict[str, MarketOpportunity],
    settings: Settings,
) -> dict[str, int]:
    """Record new shadow twin trades from built opportunities.

    Returns per-twin insert counts. Duplicate markets are ignored so repeated
    decision cycles do not create re-entry noise.
    """
    if not getattr(settings, "shadow_twin_enabled", True):
        return {TWIN_CORE: 0, TWIN_EXPANDED: 0, TWIN_BROAD: 0}

    return _record_shadow_twins_for_universes(
        opportunities,
        settings,
        core_twin_name=TWIN_CORE,
        expanded_twin_name=TWIN_EXPANDED,
        broad_twin_name=TWIN_BROAD,
        opportunity_basket=OPPORTUNITY_BASKET_FLOOR_NO,
    )


def record_adaptive_liquidity_shadow_twins(
    opportunities: dict[str, MarketOpportunity],
    settings: Settings,
) -> dict[str, int]:
    """Record low-volume markets that pass the shadow-only adaptive gate."""
    counts = {
        TWIN_CORE_ADAPTIVE_LIQUIDITY: 0,
        TWIN_EXPANDED_ADAPTIVE_LIQUIDITY: 0,
        TWIN_BROAD_ADAPTIVE_LIQUIDITY: 0,
    }
    if not getattr(settings, "shadow_twin_enabled", True):
        return counts
    if not getattr(settings, "shadow_twin_adaptive_liquidity_enabled", True):
        return counts

    return _record_shadow_twins_for_universes(
        opportunities,
        settings,
        core_twin_name=TWIN_CORE_ADAPTIVE_LIQUIDITY,
        expanded_twin_name=TWIN_EXPANDED_ADAPTIVE_LIQUIDITY,
        broad_twin_name=TWIN_BROAD_ADAPTIVE_LIQUIDITY,
        opportunity_basket=OPPORTUNITY_BASKET_FLOOR_NO_ADAPTIVE_LIQUIDITY,
    )


def _record_shadow_twins_for_universes(
    opportunities: dict[str, MarketOpportunity],
    settings: Settings,
    *,
    core_twin_name: str,
    expanded_twin_name: str,
    broad_twin_name: str,
    opportunity_basket: str,
) -> dict[str, int]:
    counts = {core_twin_name: 0, expanded_twin_name: 0, broad_twin_name: 0}

    core = core_shadow_stations(settings)
    expanded = expanded_shadow_stations(settings)
    broad = broad_shadow_stations(settings)

    for opp in opportunities.values():
        if not _has_causal_timestamps(opp):
            continue
        if not is_floor_bound_buy_no_candidate(opp, settings):
            continue
        station = str(opp.station or "").upper()
        twin_names: list[str] = []
        if station in core:
            twin_names.append(core_twin_name)
        if station in expanded:
            twin_names.append(expanded_twin_name)
        if station in broad:
            twin_names.append(broad_twin_name)
        if not twin_names:
            continue

        liquidity_min_volume = None
        if opportunity_basket == OPPORTUNITY_BASKET_FLOOR_NO_ADAPTIVE_LIQUIDITY:
            expected_cost = float(getattr(settings, "shadow_twin_cost_per_trade", 1.0))
            liquidity_min_volume = adaptive_liquidity_min_volume(expected_cost, settings)

        for twin_name in twin_names:
            if _record_shadow_trade(twin_name, opp, settings, liquidity_min_volume=liquidity_min_volume):
                counts[twin_name] += 1
            _record_policy_trades(twin_name, opp, settings, opportunity_basket, liquidity_min_volume=liquidity_min_volume)

    return counts


def _record_shadow_trade(
    twin_name: str,
    opp: MarketOpportunity,
    settings: Settings,
    *,
    liquidity_min_volume: float | None = None,
) -> bool:
    amount, cost = _shadow_amount_and_cost(opp, settings)
    if amount <= 0.0 or cost <= 0.0:
        return False

    try:
        with get_session() as session:
            existing = session.query(ShadowTwinTrade.id).filter_by(
                twin_name=twin_name,
                market_id=str(opp.market_id),
            ).first()
            if existing is not None:
                return False
            session.add(ShadowTwinTrade(
                twin_name=twin_name,
                market_id=str(opp.market_id),
                platform=str(opp.platform or "unknown"),
                side="buy_no",
                entry_yes_price=float(opp.market_price),
                amount=amount,
                cost=cost,
                model_probability=float(opp.model_probability),
                raw_model_prob=float(opp.raw_model_prob) if opp.raw_model_prob is not None else None,
                net_edge_at_entry=float(opp.net_edge_no),
                station=str(opp.station or ""),
                market_date=str(opp.market_date or ""),
                bucket_low=float(opp.bucket_low) if opp.bucket_low is not None else None,
                bucket_high=float(opp.bucket_high) if opp.bucket_high is not None else None,
                entry_volume=_optional_float(opp.volume),
                entry_spread=_optional_float(getattr(opp, "spread", None)),
                liquidity_min_volume=liquidity_min_volume,
                forecast_timestamp=opp.forecast_timestamp,
                market_price_timestamp=opp.market_price_timestamp,
                status="open",
            ))
        log.info(
            "shadow_twin: recorded %s market=%s station=%s date=%s yes=%.3f raw=%.4f edge_no=%.3f",
            twin_name,
            opp.market_id,
            opp.station,
            opp.market_date,
            opp.market_price,
            float(opp.raw_model_prob or 0.0),
            opp.net_edge_no,
        )
        return True
    except IntegrityError:
        return False
    except Exception as exc:  # noqa: BLE001 - shadow must never block main trading
        log.warning("shadow_twin: record failed for %s/%s: %s", twin_name, opp.market_id, exc)
        return False


def _record_policy_trades(
    twin_name: str,
    opp: MarketOpportunity,
    settings: Settings,
    opportunity_basket: str = OPPORTUNITY_BASKET_FLOOR_NO,
    *,
    liquidity_min_volume: float | None = None,
) -> int:
    amount, cost = _shadow_amount_and_cost(opp, settings)
    if amount <= 0.0 or cost <= 0.0:
        return 0

    inserted = 0
    for policy_name in SHADOW_POLICIES:
        if not _policy_allows_entry(policy_name, opp):
            continue
        try:
            with get_session() as session:
                existing = session.query(ShadowPolicyTrade.id).filter_by(
                    universe_name=twin_name,
                    policy_name=policy_name,
                    market_id=str(opp.market_id),
                ).first()
                if existing is not None:
                    continue
                session.add(ShadowPolicyTrade(
                    universe_name=twin_name,
                    policy_name=policy_name,
                    opportunity_basket=opportunity_basket,
                    market_id=str(opp.market_id),
                    platform=str(opp.platform or "unknown"),
                    side="buy_no",
                    entry_yes_price=float(opp.market_price),
                    amount=amount,
                    cost=cost,
                    model_probability=float(opp.model_probability),
                    raw_model_prob=float(opp.raw_model_prob) if opp.raw_model_prob is not None else None,
                    net_edge_at_entry=float(opp.net_edge_no),
                    station=str(opp.station or ""),
                    market_date=str(opp.market_date or ""),
                    bucket_low=float(opp.bucket_low) if opp.bucket_low is not None else None,
                    bucket_high=float(opp.bucket_high) if opp.bucket_high is not None else None,
                    entry_volume=_optional_float(opp.volume),
                    entry_spread=_optional_float(getattr(opp, "spread", None)),
                    liquidity_min_volume=liquidity_min_volume,
                    forecast_timestamp=opp.forecast_timestamp,
                    market_price_timestamp=opp.market_price_timestamp,
                    status="open",
                ))
            inserted += 1
        except IntegrityError:
            continue
        except Exception as exc:  # noqa: BLE001 - policy ledger must never block main
            log.warning(
                "shadow_policy: record failed for %s/%s/%s: %s",
                twin_name,
                policy_name,
                opp.market_id,
                exc,
            )
    return inserted


def update_shadow_policy_exits(
    opportunities: dict[str, MarketOpportunity],
    market_metadata: dict[str, dict],
) -> int:
    """Apply non-resolution counterfactual exits to open policy rows."""
    if not opportunities:
        return 0
    from weatheredge.execution.exit_manager import ExitManager

    now = datetime.now(timezone.utc)
    closed = 0
    exit_manager = ExitManager()
    try:
        with get_session() as session:
            rows = (
                session.query(ShadowPolicyTrade)
                .filter_by(status="open", policy_name=POLICY_MAIN_EXIT)
                .all()
            )
            for row in rows:
                opp = opportunities.get(str(row.market_id))
                if opp is None:
                    continue
                position = _position_from_policy_row(row)
                exits = exit_manager.evaluate([position], {str(row.market_id): opp}, market_metadata)
                if not exits:
                    continue
                _, reason = exits[0]
                row.status = "exited"
                row.exit_reason = reason
                row.pnl = _mark_to_market_pnl(row, float(opp.market_price))
                row.resolved_at = now
                closed += 1
        if closed:
            log.info("shadow_policy: main_exit_clone closed %d trade(s)", closed)
        return closed
    except Exception as exc:  # noqa: BLE001 - shadow exits must never block main
        log.warning("shadow_policy: exit update failed: %s", exc)
        return 0


def resolve_shadow_trades_for_market(market_id: str, outcome: str) -> int:
    """Resolve open shadow trades for one market outcome.

    Outcome is YES-side: ``yes`` or ``no``. The current shadow policy is buy_no,
    so it wins on ``no`` and loses on ``yes``.
    """
    if outcome not in {"yes", "no"}:
        return 0
    now = datetime.now(timezone.utc)
    resolved = 0
    try:
        with get_session() as session:
            rows = session.query(ShadowTwinTrade).filter_by(
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
            policy_rows = session.query(ShadowPolicyTrade).filter_by(
                market_id=str(market_id),
                status="open",
            ).all()
            for row in policy_rows:
                won = (row.side == "buy_no" and outcome == "no") or (
                    row.side == "buy_yes" and outcome == "yes"
                )
                row.status = "won" if won else "lost"
                row.outcome = outcome
                row.pnl = (float(row.amount) - float(row.cost)) if won else -float(row.cost)
                row.resolved_at = now
                resolved += 1
        if resolved:
            log.info("shadow_twin: resolved %d trade(s) for %s -> %s", resolved, market_id, outcome)
        return resolved
    except Exception as exc:  # noqa: BLE001 - resolution should not break main job
        log.warning("shadow_twin: resolve failed for %s: %s", market_id, exc)
        return 0
