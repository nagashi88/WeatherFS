"""Phase 2 market drift tracking — records entry prices and tracks convergence.

Measures whether the market price drifts toward the model's implied probability
after entry.  A drift_rate > 0.5 indicates the market is converging to the
model's view, which is a positive signal for the edge being real rather than
noise.

Uses the project SQLite database via weatheredge.db.database.get_session.
Tables are created lazily via raw SQL (no new ORM model needed — keeps the
schema addition contained to this module).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import text

from weatheredge.db.database import get_session

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema (created lazily on first use)
# ---------------------------------------------------------------------------

_DDL_DRIFT_RECORDS = """
CREATE TABLE IF NOT EXISTS drift_records (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id      INTEGER NOT NULL,
    entry_price   REAL    NOT NULL,
    model_prob    REAL    NOT NULL,
    entry_ts      TEXT    NOT NULL,
    price_snapshots TEXT  NOT NULL DEFAULT '[]'
)
"""


def _ensure_table() -> None:
    """Create the drift_records table if it does not already exist."""
    with get_session() as session:
        session.execute(text(_DDL_DRIFT_RECORDS))


class MarketDriftTracker:
    """Record trade entries and snapshot prices to measure market convergence.

    All data is persisted to the ``drift_records`` SQLite table.  The table is
    created automatically on first use so no migration step is required.

    Drift rate definition:

        For each trade, the *direction* is determined by whether entry_price is
        below (long YES) or above (long NO) the model_prob.  A snapshot is
        counted as converging if the market moved in the model's direction
        relative to entry.  drift_rate = converging_snapshots / total_snapshots.

        An aggregate drift_rate > 0.5 across all trades indicates the market
        systematically converges to model estimates after entry — a good signal.
    """

    def __init__(self) -> None:
        _ensure_table()
        log.info("MarketDriftTracker initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_entry(
        self,
        trade_id: int,
        entry_price: float,
        model_prob: float,
        timestamp: datetime,
    ) -> None:
        """Store a new drift record for a trade entry.

        Args:
            trade_id:    ID of the trade (matches trades.id).
            entry_price: Market price at which the trade was entered.
            model_prob:  Model-implied probability at entry.
            timestamp:   UTC datetime of entry.
        """
        ts_str = timestamp.isoformat()
        with get_session() as session:
            session.execute(
                text(
                    """
                    INSERT INTO drift_records
                        (trade_id, entry_price, model_prob, entry_ts, price_snapshots)
                    VALUES
                        (:trade_id, :entry_price, :model_prob, :entry_ts, :snapshots)
                    """
                ),
                {
                    "trade_id": trade_id,
                    "entry_price": entry_price,
                    "model_prob": model_prob,
                    "entry_ts": ts_str,
                    "snapshots": "[]",
                },
            )
        log.debug(
            "DriftRecord created: trade_id=%s entry=%.3f model_prob=%.3f",
            trade_id,
            entry_price,
            model_prob,
        )

    def snapshot_price(
        self,
        trade_id: int,
        current_price: float,
        timestamp: datetime,
    ) -> None:
        """Append a price snapshot to an existing drift record.

        The snapshot list is stored as a JSON array in the ``price_snapshots``
        column.  Each element is ``[iso_timestamp, price]``.

        Args:
            trade_id:      ID of the trade to update.
            current_price: Current market mid-price.
            timestamp:     UTC datetime of the snapshot.
        """
        with get_session() as session:
            row = session.execute(
                text(
                    "SELECT id, price_snapshots FROM drift_records "
                    "WHERE trade_id = :trade_id "
                    "ORDER BY id DESC LIMIT 1"
                ),
                {"trade_id": trade_id},
            ).fetchone()

            if row is None:
                log.warning(
                    "snapshot_price: no drift record found for trade_id=%s", trade_id
                )
                return

            record_id, snapshots_json = row
            snapshots: list = json.loads(snapshots_json or "[]")
            snapshots.append([timestamp.isoformat(), current_price])

            session.execute(
                text(
                    "UPDATE drift_records SET price_snapshots = :snapshots "
                    "WHERE id = :record_id"
                ),
                {"snapshots": json.dumps(snapshots), "record_id": record_id},
            )
        log.debug(
            "Price snapshot appended: trade_id=%s price=%.3f", trade_id, current_price
        )

    def compute_drift_stats(self) -> dict:
        """Compute aggregate drift statistics across all recorded trades.

        Returns:
            A dict with:

            * ``n`` (int): number of drift records evaluated.
            * ``drift_rate`` (float | None): fraction of price snapshots that
              moved in the model's direction.  ``None`` when there are no
              snapshots to evaluate.  A value > 0.5 means the market
              systematically converges to the model estimate.
        """
        with get_session() as session:
            rows = session.execute(
                text(
                    "SELECT entry_price, model_prob, price_snapshots "
                    "FROM drift_records"
                )
            ).fetchall()

        n = len(rows)
        if n == 0:
            return {"n": 0, "drift_rate": None}

        converging = 0
        total_snapshots = 0

        for entry_price, model_prob, snapshots_json in rows:
            snapshots: list = json.loads(snapshots_json or "[]")
            if not snapshots:
                continue

            # Model says price should be above entry_price when model_prob > entry_price
            model_direction_up = model_prob > entry_price

            for _ts, snap_price in snapshots:
                total_snapshots += 1
                moved_up = snap_price > entry_price
                if model_direction_up == moved_up:
                    converging += 1

        drift_rate: float | None = (
            converging / total_snapshots if total_snapshots > 0 else None
        )

        log.info(
            "DriftStats: n=%d total_snapshots=%d drift_rate=%s",
            n,
            total_snapshots,
            f"{drift_rate:.3f}" if drift_rate is not None else "None",
        )
        return {"n": n, "drift_rate": drift_rate}
