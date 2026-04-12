from __future__ import annotations
import logging
from datetime import date, datetime, timezone
from sqlalchemy import select
from weatheredge.db.models import EdgeDecay, WalletProfile
from weatheredge.db.database import get_session

log = logging.getLogger(__name__)

SHARPE_ALERT_THRESHOLD = 1.5  # Alert if avg top-10 Sharpe drops below this

class EdgeDecayMonitor:
    """Track whether the overall weather market edge is compressing."""

    def __init__(self, engine):
        self.engine = engine

    def weekly_check(self) -> dict:
        """Compare this week's top weather wallet performance vs 4 weeks ago."""
        with get_session(self.engine) as session:
            # Get top 10 weather wallets by Sharpe
            top_wallets = session.execute(
                select(WalletProfile)
                .where(WalletProfile.sharpe_ratio.is_not(None))
                .order_by(WalletProfile.sharpe_ratio.desc())
                .limit(10)
            ).scalars().all()

            if len(top_wallets) < 3:
                return {"checked": False, "reason": "Insufficient wallet data (<3 profiled wallets)"}

            # BugFix L5: Guard against potential None sharpe_ratio
            avg_sharpe = sum(w.sharpe_ratio or 0 for w in top_wallets) / len(top_wallets)
            avg_winrate = sum(w.win_rate or 0 for w in top_wallets) / len(top_wallets)

            # Compare to last check
            last_check = session.execute(
                select(EdgeDecay).order_by(EdgeDecay.id.desc()).limit(1)
            ).scalar_one_or_none()

            old_sharpe = last_check.top10_avg_sharpe if last_check else None

            # Determine assessment
            is_decaying = avg_sharpe < SHARPE_ALERT_THRESHOLD
            assessment = "ALERT: Edge may be eroding" if is_decaying else "Stable"
            if old_sharpe and avg_sharpe < old_sharpe * 0.85:
                assessment = "WARNING: Significant Sharpe decline (>15% drop)"

            # Store result
            decay = EdgeDecay(
                check_date=date.today().isoformat(),
                top10_avg_sharpe=avg_sharpe,
                top10_avg_winrate=avg_winrate,
                assessment=assessment,
            )
            session.add(decay)

        result = {
            "checked": True,
            "avg_sharpe": avg_sharpe,
            "avg_winrate": avg_winrate,
            "old_sharpe": old_sharpe,
            "is_decaying": is_decaying,
            "assessment": assessment,
        }
        log.info("Edge decay check: avg_sharpe=%.2f, assessment=%s", avg_sharpe, assessment)
        return result
