"""Trade regime classification for per-strategy P&L tracking.

A trade's regime captures the structural category that produced the entry
decision and is frozen at place_order time. Splitting metrics by regime keeps
semantically distinct strategies separated: floor-bound NO insurance (H3) vs
METAR override deterministic edge vs directional threshold trades.

Regimes are mutually exclusive and assigned in priority order:
    1. METAR override (any side) — pre-empts everything else; the override
       forces p>=0.85 regardless of the model's raw output.
    2. Floor-bound buy_no on bucket markets — raw_p_yes < PROB_FLOOR (0.10);
       the floor manufactures the edge, not the model (H3).
    3. Normal bucket — bucket markets not in case 2.
    4. Directional threshold (buy_yes / buy_no) — non-bucket markets.
    5. Unknown — missing inputs or unsupported side.
"""
from __future__ import annotations

from weatheredge.probability.gaussian import PROB_FLOOR

REGIME_FLOOR_BOUND_NO = "floor_bound_buy_no"
REGIME_METAR_OVERRIDE_YES = "metar_override_yes"
REGIME_METAR_OVERRIDE_NO = "metar_override_no"
REGIME_NORMAL_BUCKET = "normal_bucket"
REGIME_THRESHOLD_YES = "threshold_yes"
REGIME_THRESHOLD_NO = "threshold_no"
REGIME_UNKNOWN = "unknown"


def classify_regime(
    side: str,
    is_bucket: bool,
    raw_model_prob: float | None,
    metar_override: float | None,
) -> str:
    """Classify a trade entry into a strategy regime."""
    if metar_override is not None:
        if side == "buy_yes":
            return REGIME_METAR_OVERRIDE_YES
        if side == "buy_no":
            return REGIME_METAR_OVERRIDE_NO
        return REGIME_UNKNOWN
    if is_bucket:
        if (
            side == "buy_no"
            and raw_model_prob is not None
            and raw_model_prob < PROB_FLOOR
        ):
            return REGIME_FLOOR_BOUND_NO
        return REGIME_NORMAL_BUCKET
    if side == "buy_yes":
        return REGIME_THRESHOLD_YES
    if side == "buy_no":
        return REGIME_THRESHOLD_NO
    return REGIME_UNKNOWN
