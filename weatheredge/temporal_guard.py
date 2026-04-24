"""Anti-leakage temporal guard for WeatherEdge.

PRD v3.1 Section 0.4 — prevents lookahead bias in paper trading.

This is a hard system halt, not a soft warning. Any violation means forecast
or price data arrived from the future relative to the decision point, which
contaminates every downstream metric (Brier, calibration, PnL).

Usage (in paper_trader.py or any execution path)::

    from weatheredge.temporal_guard import assert_no_lookahead, TemporalLeakageError

    try:
        assert_no_lookahead(
            forecast_timestamp=order.forecast_timestamp,
            market_price_timestamp=order.market_price_timestamp,
            trade_decision_timestamp=datetime.now(timezone.utc),
            forecast_source=order.primary_model,
            market_id=order.market_id,
        )
    except TemporalLeakageError as e:
        log.critical(str(e))
        record_leakage_violation(order.market_id, str(e), order.__dict__)
        return None  # Trade rejected — do NOT execute
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class TemporalLeakageError(Exception):
    """Raised when forecast or price data is newer than the decision timestamp.

    This is a hard system halt — not a warning, not a skip. Any caller that
    catches this must record the violation and abort the trade.
    """


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Market price must be at the same instant or NEWER than the forecast.
# Zero tolerance: even 1 second of staleness is a protocol violation.
MAX_PRICE_STALENESS = timedelta(minutes=0)

# ---------------------------------------------------------------------------
# Core guard
# ---------------------------------------------------------------------------


def _as_utc_aware(ts: datetime) -> datetime:
    """Normalize SQLite-naive datetimes to UTC-aware for safe comparisons."""
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def assert_no_lookahead(
    forecast_timestamp: datetime,
    market_price_timestamp: datetime,
    trade_decision_timestamp: datetime,
    forecast_source: str = "",
    market_id: str = "",
) -> None:
    """Assert that no lookahead bias exists before executing a trade.

    Must be called BEFORE every trade execution without exception.

    Temporal ordering rules enforced:
      1. forecast_timestamp <= market_price_timestamp
         A forecast that arrived after the price snapshot cannot be used to
         trade at that price — that would be using future information.
      2. forecast_timestamp <= trade_decision_timestamp
         The forecast cannot be from the future relative to now.
      3. market_price_timestamp <= trade_decision_timestamp
         The price cannot be from the future relative to now.

    Args:
        forecast_timestamp: When the forecast data became available to the
            system (``available_time``, not ``run_init_time``).
        market_price_timestamp: When the market price was fetched
            (``fetched_at`` on the price record).
        trade_decision_timestamp: The current wall-clock time at the moment
            the trade decision is being evaluated. Use
            ``datetime.now(timezone.utc)`` at the call site.
        forecast_source: Human-readable model name (e.g. ``"gfs"``).
            Used only for error messages.
        market_id: Market identifier. Used only for error messages.

    Raises:
        TemporalLeakageError: If any of the three ordering rules is violated.
    """
    src = f"({forecast_source})" if forecast_source else ""
    forecast_timestamp = _as_utc_aware(forecast_timestamp)
    market_price_timestamp = _as_utc_aware(market_price_timestamp)
    trade_decision_timestamp = _as_utc_aware(trade_decision_timestamp)

    # Rule 1: forecast must not be newer than the price being traded at.
    if forecast_timestamp > market_price_timestamp:
        delta = forecast_timestamp - market_price_timestamp
        raise TemporalLeakageError(
            f"LOOKAHEAD DETECTED: forecast {src} at {forecast_timestamp.isoformat()} "
            f"is NEWER than market price at {market_price_timestamp.isoformat()} "
            f"for market '{market_id}'. "
            f"Delta: {delta}. "
            f"THIS TRADE WOULD USE FUTURE INFORMATION. Halting."
        )

    # Rule 2: forecast must not be newer than the decision moment.
    if forecast_timestamp > trade_decision_timestamp:
        delta = forecast_timestamp - trade_decision_timestamp
        raise TemporalLeakageError(
            f"LOOKAHEAD DETECTED: forecast {src} at {forecast_timestamp.isoformat()} "
            f"is NEWER than decision time {trade_decision_timestamp.isoformat()} "
            f"for market '{market_id}'. "
            f"Delta: {delta}. "
            f"THIS IS A TIME TRAVEL BUG. Halting."
        )

    # Rule 3: price must not be newer than the decision moment.
    if market_price_timestamp > trade_decision_timestamp:
        delta = market_price_timestamp - trade_decision_timestamp
        raise TemporalLeakageError(
            f"LOOKAHEAD DETECTED: market price at {market_price_timestamp.isoformat()} "
            f"is NEWER than decision time {trade_decision_timestamp.isoformat()} "
            f"for market '{market_id}'. "
            f"Delta: {delta}. "
            f"THIS IS A TIME TRAVEL BUG. Halting."
        )


# ---------------------------------------------------------------------------
# Violation recorder
# ---------------------------------------------------------------------------


def record_leakage_violation(
    market_id: str,
    error_msg: str,
    order_dict: dict[str, Any] | None = None,
) -> None:
    """Persist a leakage violation to the database and log at CRITICAL level.

    This is a best-effort helper. A failure here must never propagate — the
    trade has already been halted by the TemporalLeakageError. The violation
    is logged regardless of whether the DB write succeeds.

    Args:
        market_id: Market identifier for the rejected trade.
        error_msg: The full TemporalLeakageError message.
        order_dict: Optional serialisable representation of the order that
            triggered the violation (e.g. ``order.__dict__``).
    """
    log.critical(
        "TEMPORAL LEAKAGE VIOLATION | market=%s | %s | order=%s",
        market_id,
        error_msg,
        order_dict,
    )

    try:
        from sqlalchemy import text

        from weatheredge.db.database import get_session

        order_json: str | None = None
        if order_dict is not None:
            try:
                order_json = json.dumps(order_dict, default=str)
            except (TypeError, ValueError):
                order_json = str(order_dict)

        violated_at = datetime.now(timezone.utc).isoformat()

        with get_session() as session:
            session.execute(
                text(
                    "INSERT INTO leakage_violations "
                    "(market_id, error_msg, order_json, violated_at) "
                    "VALUES (:market_id, :error_msg, :order_json, :violated_at)"
                ),
                {
                    "market_id": market_id,
                    "error_msg": error_msg,
                    "order_json": order_json,
                    "violated_at": violated_at,
                },
            )
    except Exception:  # noqa: BLE001
        log.exception(
            "Failed to persist leakage violation for market '%s' — "
            "violation is already logged above.",
            market_id,
        )
