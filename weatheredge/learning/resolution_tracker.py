from __future__ import annotations
import logging
from datetime import datetime, timezone, date
from sqlalchemy import select
from weatheredge.db.models import Trade, DailyPerformance, AgentPerformance
from weatheredge.db.database import get_session

log = logging.getLogger(__name__)


class ResolutionTracker:
    def __init__(self, engine):
        self.engine = engine

    def check_resolution(self, market_id: str, actual_outcome: str) -> str:
        """
        Check the actual outcome for a market.
        actual_outcome: 'yes' or 'no'
        Returns the outcome string.

        In production, this would query NOAA NWS CLI data or market resolution API.
        For now, accepts the outcome as a parameter (called by resolution check job).
        """
        return actual_outcome

    def resolve_trades(self, market_id: str, actual_outcome: str) -> list[dict]:
        """
        Resolve all open trades for a given market.
        Returns list of resolved trade summaries.
        """
        resolved = []
        with get_session(self.engine) as session:
            open_trades = session.execute(
                select(Trade).where(Trade.market_id == market_id, Trade.status == "open")
            ).scalars().all()

            for trade in open_trades:
                won = (trade.side == "buy_yes" and actual_outcome == "yes") or \
                      (trade.side == "buy_no" and actual_outcome == "no")

                if won:
                    # Kalshi/Polymarket settle $0/$1 fee-free; entry fee is
                    # already booked into trade.cost, so win pnl is the flat
                    # payout minus cost (matches paper_trader.resolve_position).
                    pnl = trade.amount - trade.cost
                    trade.status = "won"
                else:
                    pnl = -trade.cost  # Lose: lose entire cost (fee included)
                    trade.status = "lost"

                trade.pnl = pnl
                trade.resolved_at = datetime.now(timezone.utc)

                resolved.append({
                    "trade_id": trade.id,
                    "market_id": market_id,
                    "status": trade.status,
                    "pnl": pnl,
                    "cost": trade.cost,
                    "model_probability": trade.model_probability,
                    "agent_name": trade.agent_name,
                    "station": trade.station,
                    "market_date": trade.market_date,
                    "created_at": trade.created_at,
                })
                log.info("Resolved trade %s: %s, P&L: $%.2f", trade.id, trade.status, pnl)

            # Update daily performance and agent performance
            if resolved:
                self._update_daily_performance(session, resolved)
                self._update_agent_performance(session, open_trades, resolved)

        return resolved

    def _update_daily_performance(self, session, resolved_trades: list[dict]):
        """Aggregate resolved trades into daily performance."""
        today = date.today().isoformat()
        total_pnl = sum(t["pnl"] for t in resolved_trades)
        wins = sum(1 for t in resolved_trades if t["status"] == "won")
        losses = sum(1 for t in resolved_trades if t["status"] == "lost")

        # Check if today's row exists
        existing = session.execute(
            select(DailyPerformance).where(DailyPerformance.date == today)
        ).scalar_one_or_none()

        if existing:
            existing.total_pnl = (existing.total_pnl or 0) + total_pnl
            existing.winning_trades = (existing.winning_trades or 0) + wins
            existing.losing_trades = (existing.losing_trades or 0) + losses
            existing.total_trades = (existing.total_trades or 0) + len(resolved_trades)
        else:
            dp = DailyPerformance(
                date=today,
                total_pnl=total_pnl,
                winning_trades=wins,
                losing_trades=losses,
                total_trades=len(resolved_trades),
            )
            session.add(dp)

    def _update_agent_performance(self, session, trades, resolved_trades: list[dict]):
        """Update per-agent performance metrics after resolution.

        Groups resolved trades by agent_name and upserts AgentPerformance rows
        so the AgentWeightUpdater can read real performance data.
        """
        # Build agent -> trade mapping from the ORM trade objects
        agent_trades: dict[str, list[dict]] = {}
        for trade in trades:
            agent = trade.agent_name or "Unknown"
            if agent not in agent_trades:
                agent_trades[agent] = []
            # Find matching resolved summary
            for rt in resolved_trades:
                if rt["trade_id"] == trade.id:
                    agent_trades[agent].append(rt)
                    break

        for agent_name, agent_resolved in agent_trades.items():
            if not agent_resolved:
                continue
            wins = sum(1 for t in agent_resolved if t["status"] == "won")
            total = len(agent_resolved)
            pnl = sum(t["pnl"] for t in agent_resolved)
            win_rate = wins / total if total > 0 else 0.0

            # Upsert: update existing all_time row or create new one
            existing = session.execute(
                select(AgentPerformance).where(
                    AgentPerformance.agent_name == agent_name,
                    AgentPerformance.period == "all_time",
                )
            ).scalar_one_or_none()

            if existing:
                existing.total_trades = (existing.total_trades or 0) + total
                existing.total_pnl = (existing.total_pnl or 0.0) + pnl
                all_wins = (existing.win_rate or 0.0) * ((existing.total_trades or total) - total) + wins
                existing.win_rate = all_wins / existing.total_trades if existing.total_trades else 0.0
            else:
                session.add(AgentPerformance(
                    agent_name=agent_name,
                    period="all_time",
                    total_trades=total,
                    win_rate=win_rate,
                    total_pnl=pnl,
                ))
            log.info("AgentPerformance updated: %s — trades=%d pnl=$%.2f", agent_name, total, pnl)

    def get_unresolved_markets(self) -> list[str]:
        """Get distinct market IDs with open trades."""
        with get_session(self.engine) as session:
            result = session.execute(
                select(Trade.market_id).where(Trade.status == "open").distinct()
            ).scalars().all()
            return list(result)
