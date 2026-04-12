from __future__ import annotations

import logging
from typing import Any
from sqlalchemy import select, func
from weatheredge.db.models import CalibrationRecord
from weatheredge.db.database import get_session

log = logging.getLogger(__name__)


def compute_reliability_data(
    records: list[dict[str, Any]],
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Compute reliability diagram data from calibration records.

    Each record must have ``predicted_probability`` (float) and ``actual_outcome`` (int 0/1).
    Returns *n_bins* dicts with bin_low, bin_high, predicted_mean, empirical_freq, n.
    """
    bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        in_bin = [r for r in records if low <= r["predicted_probability"] < high]
        if not in_bin:
            bins.append({"bin_low": low, "bin_high": high, "predicted_mean": (low + high) / 2, "empirical_freq": None, "n": 0})
            continue
        predicted_mean = sum(r["predicted_probability"] for r in in_bin) / len(in_bin)
        empirical_freq = sum(r["actual_outcome"] for r in in_bin) / len(in_bin)
        bins.append({"bin_low": low, "bin_high": high, "predicted_mean": predicted_mean, "empirical_freq": empirical_freq, "n": len(in_bin)})
    return bins


class CalibrationTracker:
    def __init__(self, engine):
        self.engine = engine

    @staticmethod
    def calculate_brier_score(predicted_prob: float, actual_outcome: int) -> float:
        """Brier score = (predicted - actual)^2. Lower is better. 0=perfect, 1=worst."""
        return (predicted_prob - actual_outcome) ** 2

    def update_calibration(
        self,
        model: str,
        station: str,
        lead_time_days: int,
        predicted_prob: float,
        actual_outcome: int,
        trade_id: int | None = None,
    ) -> bool:
        """Record a calibration data point after trade resolution."""
        brier = self.calculate_brier_score(predicted_prob, actual_outcome)
        with get_session(self.engine) as session:
            if trade_id is not None:
                existing = session.execute(
                    select(CalibrationRecord.id).where(CalibrationRecord.trade_id == trade_id)
                ).scalar_one_or_none()
                if existing is not None:
                    return False

            record = CalibrationRecord(
                trade_id=trade_id,
                model=model,
                station=station,
                lead_time_hours=lead_time_days * 24,
                predicted_probability=predicted_prob,
                actual_outcome=actual_outcome,
                brier_score=brier,
            )
            session.add(record)
        log.info(
            "Calibration: trade_id=%s %s/%s/%dd - predicted=%.2f, actual=%d, brier=%.4f",
            str(trade_id) if trade_id is not None else "N/A",
            model,
            station,
            lead_time_days,
            predicted_prob,
            actual_outcome,
            brier,
        )
        return True

    def get_calibration_stats(self, model: str | None = None, station: str | None = None,
                               lead_time_days: int | None = None) -> dict:
        """Get average Brier score with optional filters."""
        with get_session(self.engine) as session:
            query = select(
                func.avg(CalibrationRecord.brier_score).label("avg_brier"),
                func.count(CalibrationRecord.id).label("count"),
            )
            if model:
                query = query.where(CalibrationRecord.model == model)
            if station:
                query = query.where(CalibrationRecord.station == station)
            if lead_time_days is not None:
                query = query.where(CalibrationRecord.lead_time_hours == lead_time_days * 24)

            row = session.execute(query).one()
            return {
                "avg_brier": float(row.avg_brier) if row.avg_brier is not None else None,
                "count": row.count,
            }

    def get_agent_brier_scores(self) -> dict[str, float]:
        """Return average Brier score per agent (keyed by model/agent name).

        Groups CalibrationRecord rows by the ``model`` column (which stores the
        agent name) and computes mean((predicted_prob - outcome)^2) per group.
        Returns an empty dict if any group has fewer than 20 records.
        """
        with get_session(self.engine) as session:
            rows = session.execute(
                select(
                    CalibrationRecord.model.label("agent_name"),
                    func.avg(CalibrationRecord.brier_score).label("avg_brier"),
                    func.count(CalibrationRecord.id).label("count"),
                ).group_by(CalibrationRecord.model)
            ).all()

        if not rows:
            return {}

        result: dict[str, float] = {}
        for row in rows:
            if row.count < 20:
                return {}
            result[row.agent_name] = float(row.avg_brier)

        return result

    def get_reliability_data(self, model: str | None = None, station: str | None = None) -> list[dict]:
        """
        Get reliability diagram data: 10 bins (0-0.1, 0.1-0.2, ..., 0.9-1.0).
        Returns list of dicts with bin_center, predicted_avg, actual_freq, count.
        """
        with get_session(self.engine) as session:
            query = select(
                CalibrationRecord.predicted_probability,
                CalibrationRecord.actual_outcome,
            )
            if model:
                query = query.where(CalibrationRecord.model == model)
            if station:
                query = query.where(CalibrationRecord.station == station)
            all_rows = session.execute(query).all()

        bins = []
        for i in range(10):
            low = i / 10.0
            high = (i + 1) / 10.0
            center = (low + high) / 2

            in_bin = [(p, a) for p, a in all_rows if low <= p < high]
            if not in_bin:
                bins.append({
                    "bin_center": center,
                    "predicted_avg": center,
                    "actual_freq": None,
                    "count": 0,
                })
                continue

            predicted_avg = sum(p for p, _ in in_bin) / len(in_bin)
            actual_freq = sum(a for _, a in in_bin) / len(in_bin)
            bins.append({
                "bin_center": center,
                "predicted_avg": predicted_avg,
                "actual_freq": actual_freq,
                "count": len(in_bin),
            })

        return bins
