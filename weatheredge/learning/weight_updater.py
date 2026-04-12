from __future__ import annotations
import logging
from collections import defaultdict
from sqlalchemy import select, func
from weatheredge.db.models import CalibrationRecord
from weatheredge.db.database import get_session

log = logging.getLogger(__name__)


class ModelWeightUpdater:
    """Updates forecast model weights based on Brier score performance (every 50 trades)."""

    def __init__(self, engine):
        self.engine = engine
        self._model_weights: dict[tuple[str, str, int], dict[str, float]] = {}
        self._trade_count = 0
        self._last_update_count = 0

    def record_trade(self):
        self._trade_count += 1

    def should_update(self, interval: int = 50) -> bool:
        return (self._trade_count - self._last_update_count) >= interval

    def update_model_weights(self):
        """Recalculate model weights from Brier scores. Lower Brier -> higher weight."""
        with get_session(self.engine) as session:
            # Get avg Brier per (station, model, lead_time_hours)
            rows = session.execute(
                select(
                    CalibrationRecord.station,
                    CalibrationRecord.model,
                    CalibrationRecord.lead_time_hours,
                    func.avg(CalibrationRecord.brier_score).label("avg_brier"),
                    func.count().label("n"),
                ).group_by(
                    CalibrationRecord.station,
                    CalibrationRecord.model,
                    CalibrationRecord.lead_time_hours,
                )
            ).all()

        # Group by (station, lead_time_hours)
        groups: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
        for row in rows:
            key = (row.station, row.lead_time_hours)
            groups[key][row.model] = row.avg_brier

        # Inverse-Brier weighting per group
        for key, model_briers in groups.items():
            if not model_briers:
                continue
            # Inverse: lower Brier -> higher weight
            # Add small epsilon to avoid division by zero
            inv = {m: 1.0 / (b + 0.001) for m, b in model_briers.items()}
            total = sum(inv.values())
            weights = {m: v / total for m, v in inv.items()}

            station, lead_time_hours = key
            lead_days = lead_time_hours // 24
            self._model_weights[(station, "all", lead_days)] = weights
            log.info("Updated model weights for %s/%dd: %s", station, lead_days, weights)

        self._last_update_count = self._trade_count

    def get_model_weight(self, station: str, model: str, lead_time_days: int) -> float:
        key = (station, "all", lead_time_days)
        weights = self._model_weights.get(key, {})
        return weights.get(model, 1.0)  # Default equal weight
