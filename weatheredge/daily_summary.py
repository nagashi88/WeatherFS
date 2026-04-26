"""Daily aggregation job — produces a one-shot health snapshot for the bot.

Surfaces the things that silently bleed the model: which agent is making/losing
money, how often the parser is failing (title drift), and the day's win/loss
counts. Designed to run from the APScheduler at ~23:55 UTC; also usable
ad-hoc from a REPL or script.

Output is a plain dict so callers can: log it, format it for Telegram, or
serialize to JSON for a dashboard.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from sqlalchemy import select

from weatheredge.data.market_parser import get_parse_failure_stats
from weatheredge.db.database import get_session
from weatheredge.db.models import Signal as SignalRow
from weatheredge.db.models import Trade as TradeRow
from weatheredge.probability.edge import kalshi_taker_fee

log = logging.getLogger(__name__)

# Considered "won" or "lost" — early exits are also wins/losses but with a
# different status string. Anything still "open" is excluded from win-rate
# arithmetic so an open position cannot artificially deflate the day's stats.
_RESOLVED_STATUSES = {
    "won", "lost", "edge_stop", "capital_protect", "time_decay",
    "ev_based", "bucket_invalidated", "abandoned",
}


def _round_trip_fee(price: float | None, amount: float | None, platform: str | None) -> float:
    """Estimated round-trip taker fee for a resolved trade.

    Kalshi charges ceil(0.07·p·(1-p)·100)/100 per contract per fill, so a
    round-trip is 2× that — entry + exit at the same canonical YES price.
    Exit slippage is ignored: the goal is to surface the fee burn that
    edge_stop / ev_based exits pay even when their net P&L looks flat.
    Polymarket has no taker fee in our model → 0.
    """
    if not amount or price is None:
        return 0.0
    if (platform or "polymarket").lower() != "kalshi":
        return 0.0
    return 2.0 * kalshi_taker_fee(float(price)) * float(amount)


def _utc_day_window(target_day: date | None = None) -> tuple[datetime, datetime]:
    """Return (start_utc, end_utc) for the UTC day containing `target_day`.

    If `target_day` is None, uses today (UTC). End is exclusive.
    """
    if target_day is None:
        target_day = datetime.now(timezone.utc).date()
    start = datetime.combine(target_day, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def compute_daily_summary(target_day: date | None = None) -> dict[str, Any]:
    """Aggregate the day's trades, signals, and observability metrics.

    Returns a dict with:
      - day: ISO date string (UTC)
      - trades: total trades placed (resolved + open)
      - resolved_trades: trades that settled
      - winning_trades / losing_trades / win_rate
      - realized_pnl: sum over resolved trades
      - per_agent_pnl: dict[agent_name -> {trades, resolved, wins, losses, pnl}]
      - parser_failures: rolling 24h failure stats from market_parser
      - signal_count / agent_disagreement_rate
      - cash_cap_hits: trades where the amount column is fractional
        (< 1.0 contract), heuristic for cash-cap reductions

    All numeric fields are 0 if the day has no data — callers can format
    safely without None-checks.
    """
    start_utc, end_utc = _utc_day_window(target_day)
    out: dict[str, Any] = {
        "day": start_utc.date().isoformat(),
        "trades": 0,
        "resolved_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "realized_pnl": 0.0,
        "fee_burn_total": 0.0,
        "per_agent_pnl": {},
        "per_regime_pnl": {},
        "parser_failures": {},
        "signal_count": 0,
        "agent_disagreement_rate": 0.0,
        "cash_cap_hits": 0,
    }

    # Two-window query: trades placed today (created_at) are a different set
    # from trades that *settled* today (resolved_at). A trade placed at 23:50
    # on day N-1 that settles at 00:10 on day N must contribute its P&L to
    # day N (when the cash actually moved), but its "placed" count belongs to
    # day N-1. Mixing the two — as the previous COALESCE-style query did —
    # produces ghost P&L: positions show up resolved on a day they weren't
    # placed, or never appear at all if placed near the boundary.
    #
    # Snapshot rows into plain tuples inside the session — caller code may
    # iterate after the session closes, which would otherwise raise
    # DetachedInstanceError on lazy attribute access.
    with get_session() as session:
        # Trades placed in this UTC day (operational metrics: cash-cap,
        # per-agent placement count, signal count alignment)
        placed_rows = [
            # getattr-with-default keeps this resilient if the cash_capped
            # column hasn't been added yet (Fix #7 rollout) — pre-column rows
            # fall back to the legacy amount<1.0 heuristic below.
            (tr.agent_name, tr.amount, getattr(tr, "cash_capped", None),
             getattr(tr, "regime", None))
            for tr in session.execute(
                select(TradeRow).where(
                    TradeRow.created_at >= start_utc,
                    TradeRow.created_at < end_utc,
                )
            ).scalars().all()
        ]
        # Trades that *settled* in this UTC day (financial metrics: P&L,
        # win-rate, resolved counts). Filtered by resolved_at, not created_at,
        # so cross-day positions land on the correct settlement day.
        resolved_rows = [
            (
                tr.agent_name, tr.status, tr.pnl,
                getattr(tr, "regime", None),
                tr.price, tr.amount, tr.platform,
            )
            for tr in session.execute(
                select(TradeRow).where(
                    TradeRow.resolved_at >= start_utc,
                    TradeRow.resolved_at < end_utc,
                    TradeRow.status.in_(_RESOLVED_STATUSES),
                )
            ).scalars().all()
        ]
        signal_rows = [
            (sig.market_id, sig.signal_type)
            for sig in session.execute(
                select(SignalRow).where(
                    SignalRow.created_at >= start_utc,
                    SignalRow.created_at < end_utc,
                )
            ).scalars().all()
        ]

    out["trades"] = len(placed_rows)
    out["signal_count"] = len(signal_rows)

    # --- Per-agent + per-regime buckets ---
    def _new_agent_bucket() -> dict[str, float]:
        return {"trades": 0, "resolved": 0, "wins": 0, "losses": 0, "pnl": 0.0}

    def _new_regime_bucket() -> dict[str, float]:
        # edge_stop / ev_based exposed as first-class counters: they're the
        # primary indicators for H7 (edge_flip too reactive) and Phase 1
        # success (EV-based exit competing against edge_flip).
        # fee_burn tracks taker fees paid on the round-trip — most punishing
        # on edge_stop trades that exit near break-even but still pay both
        # legs of the Kalshi taker fee.
        return {
            "trades": 0, "resolved": 0, "wins": 0, "losses": 0,
            "edge_stop": 0, "ev_based": 0, "pnl": 0.0, "fee_burn": 0.0,
        }

    agent_buckets: dict[str, dict[str, float]] = defaultdict(_new_agent_bucket)
    regime_buckets: dict[str, dict[str, float]] = defaultdict(_new_regime_bucket)

    # Pass 1: placed trades → per-agent + per-regime placement counts +
    # cash-cap detection
    for agent_name, _amount, cash_capped, regime in placed_rows:
        agent = agent_name or "unknown"
        agent_buckets[agent]["trades"] += 1
        regime_buckets[regime or "unknown"]["trades"] += 1
        # cash_capped is the authoritative flag written by portfolio_manager
        # when the cap path reduced size. Legacy rows pre-dating the column
        # read as False (ALTER TABLE DEFAULT 0) and simply don't count —
        # accept the small blind spot in historical summaries rather than
        # reintroducing the amount<1.0 heuristic, which produced false
        # positives for legitimately small orders.
        if cash_capped:
            out["cash_cap_hits"] += 1

    # Pass 2: trades that settled today → P&L, wins/losses, fee burn
    for agent_name, status, pnl, regime, price, amount, platform in resolved_rows:
        agent = agent_name or "unknown"
        bucket = agent_buckets[agent]
        rbucket = regime_buckets[regime or "unknown"]
        bucket["resolved"] += 1
        rbucket["resolved"] += 1
        out["resolved_trades"] += 1
        pnl_val = float(pnl or 0.0)
        bucket["pnl"] += pnl_val
        rbucket["pnl"] += pnl_val
        out["realized_pnl"] += pnl_val
        fee_val = _round_trip_fee(price, amount, platform)
        rbucket["fee_burn"] += fee_val
        out["fee_burn_total"] += fee_val
        if status == "won":
            bucket["wins"] += 1
            rbucket["wins"] += 1
            out["winning_trades"] += 1
        elif status == "lost":
            bucket["losses"] += 1
            rbucket["losses"] += 1
            out["losing_trades"] += 1
        elif status == "edge_stop":
            rbucket["edge_stop"] += 1
        elif status == "ev_based":
            rbucket["ev_based"] += 1

    if out["resolved_trades"] > 0:
        out["win_rate"] = out["winning_trades"] / out["resolved_trades"]
    out["fee_burn_total"] = round(out["fee_burn_total"], 4)

    out["per_agent_pnl"] = {
        agent: {
            "trades": int(bucket["trades"]),
            "resolved": int(bucket["resolved"]),
            "wins": int(bucket["wins"]),
            "losses": int(bucket["losses"]),
            "pnl": round(bucket["pnl"], 4),
        }
        for agent, bucket in sorted(agent_buckets.items())
    }
    out["per_regime_pnl"] = {
        regime: {
            "trades": int(rbucket["trades"]),
            "resolved": int(rbucket["resolved"]),
            "wins": int(rbucket["wins"]),
            "losses": int(rbucket["losses"]),
            "edge_stop": int(rbucket["edge_stop"]),
            "ev_based": int(rbucket["ev_based"]),
            "pnl": round(rbucket["pnl"], 4),
            "fee_burn": round(rbucket["fee_burn"], 4),
        }
        for regime, rbucket in sorted(regime_buckets.items())
    }

    # --- Agent disagreement: % of multi-agent markets with conflicting BUY sides ---
    # Counts markets where at least two agents took an *opposite* tradeable
    # position (e.g. one buy_yes + one buy_no). 'no_trade' votes are dropped
    # before grouping because "I didn't see edge" is not symmetric to a buy
    # — it's an abstention. The denominator is markets with ≥1 BUY vote
    # (so a market where every agent abstained doesn't show up at all),
    # which keeps the rate interpretable as "% of actively-voted markets
    # where agents disagree on direction".
    by_market: dict[str, set[str]] = defaultdict(set)
    for market_id, signal_type in signal_rows:
        if not market_id or signal_type == "no_trade":
            continue
        by_market[market_id].add(signal_type)

    multi_side_markets = sum(1 for sides in by_market.values() if len(sides) > 1)
    total_voted_markets = len(by_market)
    if total_voted_markets > 0:
        out["agent_disagreement_rate"] = multi_side_markets / total_voted_markets

    # --- Parser failure rate ---
    # Window must match _PARSE_WINDOW_SECONDS in market_parser (3600s = 60min).
    # Asking for a wider window produces an empty result silently because the
    # parser prunes outside its own retention window — we'd report "0 attempts"
    # while the parser was actually busy. Keep these in sync.
    out["parser_failures"] = get_parse_failure_stats(window_minutes=60)

    return out


def format_daily_summary_text(summary: dict[str, Any]) -> str:
    """Render the summary as a multi-line ASCII string for log/Telegram.

    Designed to read cleanly in both monospace fonts and Telegram's HTML mode.
    """
    lines = [
        f"Daily summary — {summary['day']} (UTC)",
        f"  Trades: {summary['trades']} placed | {summary['resolved_trades']} resolved",
        f"  W/L: {summary['winning_trades']}/{summary['losing_trades']} "
        f"(win rate {summary['win_rate'] * 100:.1f}%)",
        f"  Realized P&L: ${summary['realized_pnl']:+,.2f} "
        f"(fee burn ${summary.get('fee_burn_total', 0.0):,.2f})",
        f"  Cash-cap hits: {summary['cash_cap_hits']}",
        f"  Signals: {summary['signal_count']} | "
        f"agent disagreement {summary['agent_disagreement_rate'] * 100:.1f}%",
    ]
    pf = summary.get("parser_failures") or {}
    if pf.get("attempts"):
        lines.append(
            f"  Parser: {pf['failures']}/{pf['attempts']} failed "
            f"({pf['failure_rate'] * 100:.1f}%) over last 1h"
        )
    else:
        lines.append("  Parser: 0 attempts in last 1h")
    if summary.get("per_agent_pnl"):
        lines.append("  Per-agent P&L:")
        for agent, stats in summary["per_agent_pnl"].items():
            lines.append(
                f"    {agent}: {stats['wins']}W/{stats['losses']}L "
                f"of {stats['resolved']} resolved → ${stats['pnl']:+,.2f}"
            )
    if summary.get("per_regime_pnl"):
        lines.append("  Per-regime P&L:")
        for regime, stats in summary["per_regime_pnl"].items():
            lines.append(
                f"    {regime}: {stats['wins']}W/{stats['losses']}L "
                f"({stats['edge_stop']} edge_stop, {stats['ev_based']} ev_based) "
                f"of {stats['resolved']} resolved → ${stats['pnl']:+,.2f} "
                f"(fee ${stats.get('fee_burn', 0.0):,.2f})"
            )
    return "\n".join(lines)
