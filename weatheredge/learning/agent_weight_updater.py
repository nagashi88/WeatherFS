from __future__ import annotations
import logging
from sqlalchemy import select, func
from weatheredge.db.models import AgentPerformance
from weatheredge.db.database import get_session

log = logging.getLogger(__name__)

MIN_WEIGHT = 0.05  # Addendum Gap 6
MAX_WEIGHT = 0.50


class AgentWeightUpdater:
    """Updates portfolio manager agent weights based on PnL (every 100 trades)."""

    def __init__(self, engine):
        self.engine = engine
        self._trade_count = 0
        self._last_update_count = 0

    def record_trade(self):
        self._trade_count += 1

    def should_update(self, interval: int = 100) -> bool:
        return (self._trade_count - self._last_update_count) >= interval

    def update_agent_weights(
        self,
        current_weights: dict[str, float],
        calibration_tracker=None,  # CalibrationTracker | None
    ) -> dict[str, float]:
        """
        Adjust weights based on calibration (Brier score) when available,
        falling back to per-agent total PnL.

        Inverse-Brier weighting: weight = 1/brier_score per agent, normalized
        to sum=1.0. Lower Brier score (more accurate) → higher weight.

        Falls back to PnL-based weighting when calibration_tracker is None or
        returns insufficient data (<20 records per agent).

        Enforces min=0.05, max=0.50, sum=1.0.
        """
        # --- Attempt inverse-Brier weighting ---
        if calibration_tracker is not None:
            try:
                brier_scores = calibration_tracker.get_agent_brier_scores()
            except Exception as exc:
                log.warning("update_agent_weights: brier score fetch failed: %s", exc)
                brier_scores = {}

            agents_with_brier = [a for a in current_weights if a in brier_scores]
            if agents_with_brier:
                # Compute inverse-Brier scores; guard against zero with small epsilon
                inv_brier: dict[str, float] = {
                    a: 1.0 / max(brier_scores[a], 1e-9) for a in agents_with_brier
                }
                inv_total = sum(inv_brier.values())

                scores: dict[str, float] = {}
                for a in current_weights:
                    if a in inv_brier:
                        scores[a] = inv_brier[a] / inv_total
                    else:
                        scores[a] = current_weights[a]

                score_total = sum(scores.values())
                new_weights = {a: s / score_total for a, s in scores.items()}
                new_weights = self._enforce_bounds(new_weights)
                self._last_update_count = self._trade_count
                log.info("Updated agent weights (Brier): %s", new_weights)
                return new_weights

        # --- Fallback: PnL-based weighting ---
        with get_session(self.engine) as session:
            rows = session.execute(
                select(
                    AgentPerformance.agent_name,
                    func.sum(AgentPerformance.total_pnl).label("total_pnl"),
                ).group_by(AgentPerformance.agent_name)
            ).all()

        if not rows:
            return current_weights

        # Build pnl map per agent
        pnl_map: dict[str, float] = {}
        for row in rows:
            pnl_map[row.agent_name] = row.total_pnl or 0.0

        # Only update agents we know about
        agents = [a for a in current_weights if a in pnl_map]
        if not agents:
            return current_weights

        # Shift PnLs to be non-negative (add abs(min) + small offset)
        min_pnl = min(pnl_map[a] for a in agents)
        shifted = {a: pnl_map[a] - min_pnl + 0.01 for a in agents}
        shifted_total = sum(shifted.values())

        # Build full weight dict: updated agents get PnL-proportional scores,
        # agents not in DB keep their current weight as a score.
        # Then normalize everything together so the result always sums to 1.0.
        pnl_scores: dict[str, float] = {}
        for a in current_weights:
            if a in shifted:
                pnl_scores[a] = shifted[a] / shifted_total
            else:
                pnl_scores[a] = current_weights[a]

        score_total = sum(pnl_scores.values())
        new_weights = {a: s / score_total for a, s in pnl_scores.items()}

        # Enforce floor/ceiling
        new_weights = self._enforce_bounds(new_weights)

        self._last_update_count = self._trade_count
        log.info("Updated agent weights (PnL): %s", new_weights)
        return new_weights

    @staticmethod
    def _enforce_bounds(weights: dict[str, float]) -> dict[str, float]:
        """Clamp to [MIN_WEIGHT, MAX_WEIGHT] and renormalize to sum=1.0.

        Uses water-filling: repeatedly normalize then clamp until the
        normalized values no longer change (fixed point = valid solution).
        At the fixed point, sum=1.0 and all values are within bounds.
        """
        keys = list(weights)
        vals = list(weights.values())

        for _ in range(100):
            total = sum(vals)
            normalized = [v / total for v in vals]
            clamped = [max(MIN_WEIGHT, min(MAX_WEIGHT, v)) for v in normalized]

            # Fixed point reached when clamping changes nothing
            if all(abs(c - n) < 1e-10 for c, n in zip(clamped, normalized)):
                return dict(zip(keys, normalized))

            vals = clamped

        # Converged (hit limit): return last normalized values
        total = sum(vals)
        return dict(zip(keys, [v / total for v in vals]))
