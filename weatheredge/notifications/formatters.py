"""Format helpers for Telegram bot commands (/status, /positions, /performance)."""

from __future__ import annotations

from weatheredge.execution.paper_trader import PerformanceReport


def format_performance_report(report: PerformanceReport) -> str:
    """Format a PerformanceReport for the /performance command."""
    return (
        f"Performance Report\n"
        f"  Trades     : {report.total_trades} "
        f"(W:{report.winning_trades} / L:{report.losing_trades})\n"
        f"  Win rate   : {report.win_rate * 100:.1f}%\n"
        f"  Total P&L  : ${report.total_pnl:.2f}\n"
        f"  ROI        : {report.roi * 100:.2f}%\n"
        f"  Sharpe     : {report.sharpe_ratio:.2f}\n"
        f"  Max DD     : {report.max_drawdown * 100:.1f}%\n"
        f"  Avg edge   : {report.avg_edge_at_entry * 100:.1f}%"
    )


def format_positions_list(positions: list) -> str:
    """Format a list of open positions for the /positions command."""
    if not positions:
        return "No open positions."
    lines = ["Open Positions:"]
    for pos in positions:
        market = pos.get("market", "?")
        side = pos.get("side", "?")
        amount = pos.get("amount", "?")
        entry = pos.get("entry_price", "?")
        mark = pos.get("mark_price", "?")
        pnl = pos.get("unrealized_pnl", "?")
        lines.append(
            f"  {market} | {side} {amount} @ ${entry} | mark ${mark} | uPnL ${pnl}"
        )
    return "\n".join(lines)


def format_status(bankroll: float, open_count: int, daily_pnl: float) -> str:
    """Format bot status for the /status command."""
    pnl_sign = "+" if daily_pnl >= 0 else ""
    return (
        f"WeatherEdge Status\n"
        f"  Bankroll     : ${bankroll:.2f}\n"
        f"  Open positions: {open_count}\n"
        f"  Daily P&L    : {pnl_sign}${daily_pnl:.2f}"
    )


def format_agent_performance(agent_stats: dict) -> str:
    """Format per-agent performance for /agents command.

    agent_stats example: {
        "Conservative": {"trades": 50, "win_rate": 0.62, "pnl": 45.0, "weight": 0.30},
        "Aggressive": {"trades": 80, "win_rate": 0.58, "pnl": 120.0, "weight": 0.30},
        ...
    }
    """
    lines = ["Agent Performance:"]
    for name, stats in agent_stats.items():
        lines.append(
            f"  {name}: {stats['trades']} trades, "
            f"WR {stats['win_rate']:.0%}, "
            f"P&L ${stats['pnl']:.2f}, "
            f"Weight {stats['weight']:.2f}"
        )
    return "\n".join(lines)


def format_vpin_status(vpin: float, is_toxic: bool) -> str:
    """Format VPIN status for /vpin command."""
    status = "TOXIC - Reducing position sizes 50%" if is_toxic else "Normal"
    return f"VPIN Score: {vpin:.2f}\nStatus: {status}\nThreshold: 0.70"


def format_kill_confirmation() -> str:
    """Format kill switch confirmation for /kill command."""
    return (
        "KILL SWITCH ACTIVATED\n"
        "- All unfilled orders cancelled\n"
        "- Trading halted\n"
        "- Open positions marked as abandoned-pending-resolution\n"
        "- Manual restart required: /restart"
    )


def format_performance_dashboard(
    bankroll: float,
    total_pnl: float,
    win_rate: float,
    sharpe: float,
    max_drawdown: float,
    brier_score: float,
    agent_stats: dict,
) -> str:
    """Format a full performance dashboard for Telegram display.

    agent_stats example: {
        "Conservative": {"trades": 50, "win_rate": 0.62, "pnl": 45.0, "weight": 0.30},
    }
    """
    pnl_sign = "+" if total_pnl >= 0 else ""
    lines = [
        "Performance Dashboard",
        f"  Bankroll     : ${bankroll:.2f}",
        f"  Total P&L    : {pnl_sign}${total_pnl:.2f}",
        f"  Win rate     : {win_rate * 100:.1f}%",
        f"  Sharpe       : {sharpe:.2f}",
        f"  Max DD       : {max_drawdown * 100:.1f}%",
        f"  Brier score  : {brier_score:.4f}",
    ]
    if agent_stats:
        lines.append("Agent Breakdown:")
        for name, stats in agent_stats.items():
            lines.append(
                f"  {name}: {stats['trades']} trades, "
                f"WR {stats['win_rate']:.0%}, "
                f"P&L ${stats['pnl']:.2f}, "
                f"Weight {stats['weight']:.2f}"
            )
    return "\n".join(lines)


def format_morning_briefing(briefing_text: str) -> str:
    return f"MORNING BRIEFING:\n{briefing_text}"


def format_edge_decay_alert(old_sharpe: float | None, new_sharpe: float, assessment: str) -> str:
    old = f"{old_sharpe:.2f}" if old_sharpe else "N/A"
    return f"EDGE DECAY CHECK:\nPrev avg Sharpe: {old}\nCurrent avg Sharpe: {new_sharpe:.2f}\nAssessment: {assessment}"


def format_competitor_alert(new_wallets: list[dict]) -> str:
    if not new_wallets:
        return "No new weather wallets detected this week."
    lines = [f"NEW WEATHER WALLETS ({len(new_wallets)}):"]
    for w in new_wallets:
        lines.append(f"  {w['address'][:10]}... - {w.get('trades', 0)} trades, Sharpe {w.get('sharpe', 0):.2f}")
    return "\n".join(lines)
