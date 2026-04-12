from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, func
from weatheredge.db.models import WalletProfile
from weatheredge.db.database import get_session

log = logging.getLogger(__name__)

class CompetitorTracker:
    """Monitor top weather wallets for new entrants and strategy changes."""

    def __init__(self, engine):
        self.engine = engine

    def track_new_wallets(self, min_trades: int = 50) -> list[dict]:
        """Detect new high-volume weather wallets (profiled in last 7 days)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        with get_session(self.engine) as session:
            new_wallets = session.execute(
                select(WalletProfile).where(
                    WalletProfile.profiled_at >= cutoff,
                    WalletProfile.total_trades >= min_trades,
                )
            ).scalars().all()

            results = [{
                "address": w.wallet_address,
                "trades": w.total_trades,
                "win_rate": w.win_rate,
                "sharpe": w.sharpe_ratio,
                "pnl": w.total_pnl,
                "strategy": w.strategy_type,
            } for w in new_wallets]

        if results:
            log.info("Found %d new weather wallets (>%d trades)", len(results), min_trades)
        return results

    def get_top_competitors(self, n: int = 10) -> list[dict]:
        """Get top N weather wallets by PnL."""
        with get_session(self.engine) as session:
            top = session.execute(
                select(WalletProfile)
                .where(WalletProfile.total_pnl.is_not(None))
                .order_by(WalletProfile.total_pnl.desc())
                .limit(n)
            ).scalars().all()

            return [{
                "address": w.wallet_address,
                "trades": w.total_trades,
                "win_rate": w.win_rate,
                "sharpe": w.sharpe_ratio,
                "pnl": w.total_pnl,
                "strategy": w.strategy_type,
            } for w in top]
