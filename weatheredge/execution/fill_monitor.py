"""Fill quality monitoring and adversarial fill detection.

Implements PRD Section 5.7 adversarial fill monitoring: tracks slippage
and post-fill price movement to detect adverse selection patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

from weatheredge.db.models import FillQuality
from weatheredge.db.database import get_session


@dataclass
class FillRecord:
    trade_id: int
    fill_price: float
    market_mid_at_fill: float
    side: str = "buy_yes"  # BugFix M3: track side for correct adverse_move direction
    price_5s_after: float | None = None


class FillMonitor:
    def __init__(self, db_engine):
        self.engine = db_engine

    def record_fill(self, record: FillRecord):
        """Store fill quality data for adversarial detection."""
        slippage = record.fill_price - record.market_mid_at_fill
        adverse_move = None
        if record.price_5s_after is not None:
            # BugFix M3: For buy_no, price UP is adverse; for buy_yes, price DOWN is adverse
            if record.side == "buy_no":
                adverse_move = record.price_5s_after - record.market_mid_at_fill
            else:
                adverse_move = record.market_mid_at_fill - record.price_5s_after

        with get_session(self.engine) as session:
            fq = FillQuality(
                trade_id=record.trade_id,
                fill_price=record.fill_price,
                market_mid_at_fill=record.market_mid_at_fill,
                slippage=slippage,
                price_5s_after=record.price_5s_after,
                adverse_move=adverse_move,
            )
            session.add(fq)

    def check_adverse_selection(self, last_n: int = 20) -> dict:
        """Check if recent fills show adverse selection pattern."""
        with get_session(self.engine) as session:
            fills = session.query(FillQuality).order_by(FillQuality.id.desc()).limit(last_n).all()
            if len(fills) < 5:
                return {"adverse": False, "reason": "Insufficient fills"}

            adverse_count = sum(1 for f in fills if f.adverse_move and f.adverse_move > 0)
            adverse_pct = adverse_count / len(fills)

            return {
                "adverse": adverse_pct > 0.70,
                "adverse_pct": adverse_pct,
                "fills_checked": len(fills),
                "reason": f"{adverse_pct:.0%} of last {len(fills)} fills moved against us"
            }
