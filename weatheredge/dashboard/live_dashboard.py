#!/usr/bin/env python3
"""WeatherEdge live terminal dashboard using rich.live.Live for real-time auto-refresh.

Run standalone:
    python -m weatheredge.dashboard.live_dashboard
    python -m weatheredge.dashboard.live_dashboard --refresh 5
    python -m weatheredge.dashboard.live_dashboard --db /path/to/weatheredge.db
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "weatheredge.db"
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "weatheredge.log"
BOT_FRESHNESS_SECONDS = 60
SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"
INITIAL_BANKROLL_DEFAULT = 1000.0


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _open_db(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    except sqlite3.OperationalError:
        return None


def _safe_fetchall(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.DatabaseError:
        return []


def _safe_fetchone(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> sqlite3.Row | None:
    try:
        return conn.execute(sql, params).fetchone()
    except sqlite3.DatabaseError:
        return None


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = (), default: float = 0.0) -> float:
    row = _safe_fetchone(conn, sql, params)
    if row is None or row[0] is None:
        return default
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Data queries
# ---------------------------------------------------------------------------

def query_open_positions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return _safe_fetchall(
        conn,
        """
        SELECT market_id, side, price, amount, cost, agent_name,
               created_at, model_probability, edge_at_entry
        FROM trades
        WHERE status = 'open'
        ORDER BY created_at DESC
        """,
    )


def query_recent_trades(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return _safe_fetchall(
        conn,
        """
        SELECT created_at, market_id, side, price, edge_at_entry,
               agent_name, status, pnl
        FROM trades
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )


def query_performance(conn: sqlite3.Connection) -> dict:
    # All non-open statuses count as resolved (includes edge_stop, won, lost, etc.)
    CLOSED_STATUSES = "('won','lost','edge_stop','capital_protect','time_decay','ev_based')"
    won_final = int(_scalar(conn, "SELECT COUNT(*) FROM trades WHERE status='won'"))
    lost_final = int(_scalar(conn, "SELECT COUNT(*) FROM trades WHERE status='lost'"))
    total_pnl = _scalar(conn, f"SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status IN {CLOSED_STATUSES}")
    open_cost = _scalar(conn, "SELECT COALESCE(SUM(cost), 0) FROM trades WHERE status='open'")

    # Resolved = all closed trades (edge_stop counts as resolved)
    resolved = int(_scalar(conn, f"SELECT COUNT(*) FROM trades WHERE status IN {CLOSED_STATUSES}"))
    # Win = any closed trade with pnl > 0
    wins_all = int(_scalar(conn, f"SELECT COUNT(*) FROM trades WHERE status IN {CLOSED_STATUSES} AND pnl > 0"))
    win_rate = (wins_all / resolved * 100.0) if resolved > 0 else None

    # best agent by win rate (min 3 trades, pnl > 0 = win)
    agent_rows = _safe_fetchall(
        conn,
        f"""
        SELECT agent_name,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
               COUNT(*) AS total
        FROM trades
        WHERE status IN {CLOSED_STATUSES}
          AND agent_name IS NOT NULL
        GROUP BY agent_name
        HAVING total >= 3
        ORDER BY CAST(wins AS FLOAT) / total DESC
        LIMIT 1
        """,
    )
    best_agent: str | None = None
    best_agent_wr: float | None = None
    if agent_rows:
        row = agent_rows[0]
        best_agent = str(row["agent_name"])
        if row["total"]:
            best_agent_wr = row["wins"] / row["total"] * 100.0

    return {
        "won": won_final,
        "lost": lost_final,
        "wins_all": wins_all,
        "resolved": resolved,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "open_cost": open_cost,
        "best_agent": best_agent,
        "best_agent_wr": best_agent_wr,
    }


def query_unrealized_pnl(conn: sqlite3.Connection) -> dict:
    """Calculate unrealized P&L by comparing open positions to current market prices."""
    rows = _safe_fetchall(
        conn,
        """
        SELECT t.market_id, t.side, t.price, t.amount, t.cost,
               lp.yes_price AS current_yes
        FROM trades t
        LEFT JOIN (
            SELECT market_id, yes_price,
                   ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC, id DESC) AS rn
            FROM market_prices WHERE market_id IS NOT NULL
        ) lp ON t.market_id = lp.market_id AND lp.rn = 1
        WHERE t.status = 'open'
        """,
    )
    total_cost = 0.0
    total_value = 0.0
    positions = 0
    for r in rows:
        cost = float(r["cost"] or 0)
        amount = float(r["amount"] or 0)
        current_yes = float(r["current_yes"]) if r["current_yes"] is not None else float(r["price"] or 0)
        total_cost += cost
        if r["side"] == "buy_yes":
            total_value += current_yes * amount
        else:
            total_value += (1.0 - current_yes) * amount
        positions += 1
    unrealized = total_value - total_cost
    pct = (unrealized / total_cost * 100.0) if total_cost > 0 else 0.0
    return {
        "unrealized_pnl": unrealized,
        "unrealized_pct": pct,
        "total_cost": total_cost,
        "total_value": total_value,
        "open_positions": positions,
    }


def query_brier_score(conn: sqlite3.Connection) -> float | None:
    """Compute Brier score from won/lost trades using model_probability vs actual outcome.

    outcome_yes = 1 if (buy_yes + won) OR (buy_no + lost), else 0.
    Brier = mean((model_probability - outcome_yes)^2).
    edge_stop trades are excluded — we don't know the actual market outcome.
    """
    rows = _safe_fetchall(
        conn,
        """
        SELECT side, status, model_probability
        FROM trades
        WHERE status IN ('won', 'lost')
          AND model_probability IS NOT NULL
        """,
    )
    if not rows:
        return None
    total = 0.0
    for row in rows:
        side = row["side"]
        status = row["status"]
        prob = float(row["model_probability"])
        # outcome_yes: YES resolved = 1, NO resolved = 0
        if (side == "buy_yes" and status == "won") or (side == "buy_no" and status == "lost"):
            outcome_yes = 1.0
        else:
            outcome_yes = 0.0
        total += (prob - outcome_yes) ** 2
    return total / len(rows)


def query_timesfm_corrections(conn: sqlite3.Connection) -> int:
    # Count trades that used TimesFM (agent_name contains 'timesfm' or 'TimesFM')
    return int(_scalar(
        conn,
        "SELECT COUNT(*) FROM trades WHERE LOWER(agent_name) LIKE '%timesfm%'",
    ))


def query_active_markets(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return _safe_fetchall(
        conn,
        """
        WITH latest_price AS (
            SELECT market_id, yes_price, no_price, volume, spread,
                   timestamp,
                   ROW_NUMBER() OVER (
                       PARTITION BY market_id ORDER BY timestamp DESC, id DESC
                   ) AS rn
            FROM market_prices
            WHERE market_id IS NOT NULL
        ),
        latest_forecast AS (
            SELECT station,
                   temperature_forecast,
                   ensemble_members,
                   forecast_timestamp,
                   ROW_NUMBER() OVER (
                       PARTITION BY station ORDER BY forecast_timestamp DESC, id DESC
                   ) AS rn
            FROM forecasts
        )
        SELECT m.market_id, m.city, m.threshold_low, m.threshold_high,
               lp.yes_price, lp.no_price, lp.volume,
               lf.temperature_forecast, lf.ensemble_members
        FROM markets m
        LEFT JOIN latest_price lp ON lp.market_id = m.market_id AND lp.rn = 1
        LEFT JOIN latest_forecast lf ON lf.station = m.station AND lf.rn = 1
        ORDER BY m.market_date ASC, m.city ASC
        LIMIT 20
        """,
    )


def query_daily_pnl(conn: sqlite3.Connection) -> list[tuple[str, float]]:
    rows = _safe_fetchall(
        conn,
        """
        SELECT DATE(resolved_at) AS day, SUM(pnl) AS daily_pnl
        FROM trades
        WHERE pnl IS NOT NULL
          AND resolved_at IS NOT NULL
          AND status IN ('won','lost','edge_stop','capital_protect','time_decay','ev_based')
        GROUP BY day
        ORDER BY day ASC
        LIMIT 30
        """,
    )
    return [(str(r["day"]), float(r["daily_pnl"])) for r in rows]


# ---------------------------------------------------------------------------
# Bot status helpers
# ---------------------------------------------------------------------------

def _detect_bot_running(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        mtime = log_path.stat().st_mtime
        return (time.time() - mtime) <= BOT_FRESHNESS_SECONDS
    except OSError:
        return False


def _timesfm_active(log_path: Path) -> bool:
    """Check if Phase 6 (TimesFM) appears in last 500 lines of log."""
    if not log_path.exists():
        return False
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = list(deque(fh, maxlen=500))
        return any("Phase 6" in line or "TimesFM" in line for line in lines)
    except OSError:
        return False


def _bot_uptime(log_path: Path) -> str:
    if not log_path.exists():
        return "-"
    try:
        mtime = log_path.stat().st_mtime
        age_seconds = time.time() - mtime
        if age_seconds > BOT_FRESHNESS_SECONDS:
            return "-"
        # Try to read start timestamp from first log line
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline()
        # Try to parse ISO timestamp at start of line
        parts = first_line.strip().split()
        if len(parts) >= 2:
            try:
                ts = datetime.fromisoformat(f"{parts[0]}T{parts[1][:8]}")
                elapsed = datetime.now() - ts
                h, rem = divmod(int(elapsed.total_seconds()), 3600)
                m, s = divmod(rem, 60)
                return f"{h:02d}:{m:02d}:{s:02d}"
            except (ValueError, IndexError):
                pass
        return "running"
    except OSError:
        return "-"


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------

def _sparkline(values: list[float], width: int = 20) -> str:
    if not values:
        return " " * width
    if len(values) == 1:
        return SPARKLINE_CHARS[4] * width

    mn, mx = min(values), max(values)
    span = mx - mn
    result: list[str] = []
    for v in values[-width:]:
        if span == 0:
            idx = 4
        else:
            idx = round((v - mn) / span * (len(SPARKLINE_CHARS) - 1))
        result.append(SPARKLINE_CHARS[idx])
    # Pad left if shorter than width
    return " " * max(0, width - len(result)) + "".join(result)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_price(val) -> str:
    if val is None:
        return "-"
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pnl(val) -> Text:
    if val is None:
        return Text("-", style="dim")
    try:
        f = float(val)
    except (TypeError, ValueError):
        return Text("-", style="dim")
    sign = "+" if f >= 0 else ""
    style = "green" if f >= 0 else "red"
    return Text(f"{sign}{f:.2f}", style=style)


def _fmt_edge(val) -> str:
    if val is None:
        return "-"
    try:
        return f"{float(val)*100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_ts(val) -> str:
    if not val:
        return "-"
    return str(val).replace("T", " ")[:16]


def _status_style(status: str) -> str:
    mapping = {
        "open": "yellow",
        "won": "green",
        "lost": "red",
        "edge_stop": "cyan",
        "capital_protect": "magenta",
        "time_decay": "blue",
        "ev_based": "blue",
        "cancelled": "dim",
    }
    return mapping.get(status.lower(), "white")


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def build_header_panel(
    bot_running: bool,
    uptime: str,
    perf: dict,
    initial_bankroll: float,
    refresh_ts: str,
    unrealized: dict | None = None,
) -> Panel:
    open_cost = perf.get("open_cost", 0.0)
    total_pnl = perf.get("total_pnl", 0.0)

    # Unrealized P&L from mark-to-market
    unr_pnl = unrealized.get("unrealized_pnl", 0.0) if unrealized else 0.0
    unr_pct = unrealized.get("unrealized_pct", 0.0) if unrealized else 0.0
    open_positions = unrealized.get("open_positions", 0) if unrealized else 0

    # Bankroll = initial + realized P&L + unrealized P&L
    bankroll = initial_bankroll + total_pnl + unr_pnl

    pnl_pct = (total_pnl / initial_bankroll * 100.0) if initial_bankroll else 0.0
    win_rate = perf.get("win_rate")

    pnl_sign = "+" if total_pnl >= 0 else ""
    pnl_color = "green" if total_pnl >= 0 else "red"
    pnl_pct_sign = "+" if pnl_pct >= 0 else ""

    unr_sign = "+" if unr_pnl >= 0 else ""
    unr_color = "green" if unr_pnl >= 0 else "red"
    unr_pct_sign = "+" if unr_pct >= 0 else ""

    wr_str = f"{win_rate:.1f}%" if win_rate is not None else "N/A"

    line = Text()
    line.append("  WEATHEREDGE LIVE DASHBOARD", style="bold white")
    line.append("          Status: ")
    line.append("RUNNING" if bot_running else "STOPPED", style="bold green" if bot_running else "bold red")
    line.append(f"  Uptime: {uptime}")
    line.append(f"  Refresh: {refresh_ts}\n", style="dim")
    line.append(
        f"  Bankroll: ${bankroll:,.2f}",
        style="bold cyan",
    )
    line.append(
        f"  Realized: {pnl_sign}${total_pnl:,.2f} ({pnl_pct_sign}{pnl_pct:.1f}%)",
        style=pnl_color,
    )
    line.append(
        f"  Unrealized: {unr_sign}${unr_pnl:,.2f} ({unr_pct_sign}{unr_pct:.1f}%)",
        style=f"bold {unr_color}",
    )
    line.append(f"  Positions: {open_positions}", style="bold yellow")
    line.append(f"  Win Rate: {wr_str}", style="bold")

    return Panel(line, style="bold blue", padding=(0, 1))


def build_open_positions_panel(rows: list[sqlite3.Row]) -> Panel:
    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Market", style="white", no_wrap=True, max_width=22)
    table.add_column("Side", style="cyan", width=8)
    table.add_column("Price", justify="right", width=6)
    table.add_column("Amount", justify="right", width=7)
    table.add_column("Cost", justify="right", width=7)
    table.add_column("Agent", style="dim", max_width=14)
    table.add_column("Entered", style="dim", width=12)

    if not rows:
        table.add_row(Text("Waiting for data...", style="dim"), "", "", "", "", "", "")
    else:
        for row in rows:
            side_text = Text(str(row["side"] or "-"))
            side_style = "green" if "yes" in str(row["side"] or "").lower() else "yellow"
            side_text.stylize(side_style)
            table.add_row(
                str(row["market_id"] or "-")[-22:],
                side_text,
                _fmt_price(row["price"]),
                _fmt_price(row["amount"]),
                _fmt_price(row["cost"]),
                str(row["agent_name"] or "-")[:14],
                _fmt_ts(row["created_at"])[-12:],
            )

    return Panel(
        table,
        title=f"[bold]OPEN POSITIONS ({len(rows)})[/bold]",
        border_style="cyan",
        padding=(0, 0),
    )


def build_model_performance_panel(
    perf: dict,
    brier: float | None,
    timesfm_active: bool,
    timesfm_corrections: int,
) -> Panel:
    won_final = perf.get("won", 0)
    lost_final = perf.get("lost", 0)
    wins_all = perf.get("wins_all", 0)
    resolved = perf.get("resolved", 0)
    win_rate = perf.get("win_rate")
    best_agent = perf.get("best_agent")
    best_wr = perf.get("best_agent_wr")

    lines = Text()
    brier_str = f"{brier:.4f}" if brier is not None else "N/A (need won/lost)"
    wr_str = f"{win_rate:.1f}%" if win_rate is not None else "N/A"

    lines.append("Brier Score:  ", style="dim")
    lines.append(f"{brier_str}\n", style="bold")

    lines.append("Win Rate:     ", style="dim")
    lines.append(f"{wr_str}", style="bold green" if (win_rate or 0) >= 55 else "bold yellow")
    lines.append(f"  ({wins_all}/{resolved} closed)\n", style="dim")

    lines.append("Final:        ", style="dim")
    lines.append(f"{won_final}", style="green")
    lines.append(" won / ", style="dim")
    lines.append(f"{lost_final}", style="red")
    lines.append(" lost (market-resolved)\n", style="dim")

    if best_agent and best_wr is not None:
        lines.append("Best Agent:   ", style="dim")
        lines.append(f"{best_agent} ({best_wr:.1f}%)\n", style="bold magenta")
    else:
        lines.append("Best Agent:   N/A\n", style="dim")

    lines.append("TimesFM:      ", style="dim")
    if timesfm_active:
        lines.append("Active ", style="bold green")
        lines.append("\u2713", style="bold green")
    else:
        lines.append("Inactive \u2717", style="dim red")
    lines.append(f"  Corrections: {timesfm_corrections}\n", style="dim")

    return Panel(
        lines,
        title="[bold]MODEL PERFORMANCE[/bold]",
        border_style="magenta",
        padding=(0, 1),
    )


def build_recent_trades_panel(rows: list[sqlite3.Row]) -> Panel:
    table = Table(
        show_header=True,
        header_style="bold yellow",
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Time", style="dim", width=14)
    table.add_column("Market", no_wrap=True, max_width=22)
    table.add_column("Side", width=8)
    table.add_column("Price", justify="right", width=6)
    table.add_column("Edge", justify="right", width=7)
    table.add_column("Agent", style="dim", max_width=14)
    table.add_column("Status", width=14)
    table.add_column("P&L", justify="right", width=8)

    if not rows:
        table.add_row(Text("Waiting for data...", style="dim"), "", "", "", "", "", "", "")
    else:
        for row in rows:
            status = str(row["status"] or "")
            st = _status_style(status)
            side_str = str(row["side"] or "-")
            side_style = "green" if "yes" in side_str.lower() else "yellow"

            table.add_row(
                _fmt_ts(row["created_at"])[-14:],
                str(row["market_id"] or "-")[-22:],
                Text(side_str, style=side_style),
                _fmt_price(row["price"]),
                _fmt_edge(row["edge_at_entry"]),
                str(row["agent_name"] or "-")[:14],
                Text(status, style=st),
                _fmt_pnl(row["pnl"]),
            )

    return Panel(
        table,
        title="[bold]RECENT TRADES (last 20)[/bold]",
        border_style="yellow",
        padding=(0, 0),
    )


def build_active_markets_panel(rows: list[sqlite3.Row]) -> Panel:
    table = Table(
        show_header=True,
        header_style="bold green",
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("City", width=10)
    table.add_column("Threshold", justify="right", width=10)
    table.add_column("Yes Px", justify="right", width=7)
    table.add_column("No Px", justify="right", width=7)
    table.add_column("Volume", justify="right", width=8)
    table.add_column("Model Tmp", justify="right", width=10)

    if not rows:
        table.add_row(Text("Waiting for data...", style="dim"), "", "", "", "", "")
    else:
        for row in rows:
            thr_low = row["threshold_low"]
            thr_high = row["threshold_high"]
            if thr_low is not None and thr_high is not None:
                thr_str = f"{thr_low:.0f}-{thr_high:.0f}F"
            elif thr_low is not None:
                thr_str = f">{thr_low:.0f}F"
            elif thr_high is not None:
                thr_str = f"<{thr_high:.0f}F"
            else:
                thr_str = "-"

            model_tmp = row["temperature_forecast"]
            model_str = f"{float(model_tmp):.1f}F" if model_tmp is not None else "-"

            table.add_row(
                str(row["city"] or "-")[:10],
                thr_str,
                _fmt_price(row["yes_price"]),
                _fmt_price(row["no_price"]),
                _fmt_price(row["volume"]),
                model_str,
            )

    return Panel(
        table,
        title=f"[bold]ACTIVE MARKETS ({len(rows)})[/bold]",
        border_style="green",
        padding=(0, 0),
    )


def build_pnl_history_panel(daily_pnl: list[tuple[str, float]]) -> Panel:
    if not daily_pnl:
        return Panel(
            Text("Waiting for data...", style="dim"),
            title="[bold]P&L HISTORY[/bold]",
            border_style="blue",
            padding=(0, 1),
        )

    values = [v for _, v in daily_pnl]
    spark = _sparkline(values, width=min(30, len(values)))

    lines = Text()
    lines.append(spark + "\n", style="cyan")

    # Last 7 days bar
    recent = daily_pnl[-7:]
    for day, pnl in recent:
        sign = "+" if pnl >= 0 else ""
        color = "green" if pnl >= 0 else "red"
        bar_len = min(abs(int(pnl / 5)), 12)
        bar = ("+" if pnl >= 0 else "-") * max(bar_len, 1)
        lines.append(f"{day[-5:]}: ", style="dim")
        lines.append(f"{sign}{pnl:+.2f} ", style=color)
        lines.append(bar + "\n", style=color)

    total = sum(values)
    sign = "+" if total >= 0 else ""
    lines.append(f"\nTotal: {sign}${total:.2f}", style="bold green" if total >= 0 else "bold red")

    return Panel(
        lines,
        title="[bold]P&L HISTORY[/bold]",
        border_style="blue",
        padding=(0, 1),
    )


# ---------------------------------------------------------------------------
# Layout builder
# ---------------------------------------------------------------------------

def build_layout(
    header: Panel,
    open_positions: Panel,
    model_perf: Panel,
    recent_trades: Panel,
    active_markets: Panel,
    pnl_history: Panel,
) -> Layout:
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="middle", size=12),
        Layout(name="recent", size=14),
        Layout(name="bottom", size=12),
    )

    layout["header"].update(header)

    layout["middle"].split_row(
        Layout(name="open_pos", ratio=1),
        Layout(name="model_perf", ratio=1),
    )
    layout["middle"]["open_pos"].update(open_positions)
    layout["middle"]["model_perf"].update(model_perf)

    layout["recent"].update(recent_trades)

    layout["bottom"].split_row(
        Layout(name="active_markets", ratio=3),
        Layout(name="pnl_history", ratio=2),
    )
    layout["bottom"]["active_markets"].update(active_markets)
    layout["bottom"]["pnl_history"].update(pnl_history)

    return layout


# ---------------------------------------------------------------------------
# Dashboard class
# ---------------------------------------------------------------------------

class LiveDashboard:
    def __init__(self, db_path: Path, log_path: Path, refresh_seconds: int = 3) -> None:
        self.db_path = db_path
        self.log_path = log_path
        self.refresh_seconds = refresh_seconds
        self._initial_bankroll = INITIAL_BANKROLL_DEFAULT
        self._load_initial_bankroll()

    def _load_initial_bankroll(self) -> None:
        try:
            from weatheredge.config import Settings
            self._initial_bankroll = Settings().initial_bankroll
        except Exception:
            self._initial_bankroll = INITIAL_BANKROLL_DEFAULT

    def render(self) -> Layout:
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        bot_running = _detect_bot_running(self.log_path)
        uptime = _bot_uptime(self.log_path)
        tf_active = _timesfm_active(self.log_path)

        conn = _open_db(self.db_path)
        try:
            if conn is not None:
                open_rows = query_open_positions(conn)
                recent_rows = query_recent_trades(conn)
                perf = query_performance(conn)
                brier = query_brier_score(conn)
                tf_corrections = query_timesfm_corrections(conn)
                market_rows = query_active_markets(conn)
                daily_pnl = query_daily_pnl(conn)
                unrealized = query_unrealized_pnl(conn)
            else:
                open_rows = []
                recent_rows = []
                perf = {
                    "won": 0, "lost": 0, "resolved": 0,
                    "win_rate": None, "total_pnl": 0.0, "open_cost": 0.0,
                    "best_agent": None, "best_agent_wr": None,
                }
                brier = None
                tf_corrections = 0
                market_rows = []
                daily_pnl = []
                unrealized = None
        finally:
            if conn is not None:
                conn.close()

        header = build_header_panel(bot_running, uptime, perf, self._initial_bankroll, now_str, unrealized)
        open_positions = build_open_positions_panel(open_rows)
        model_perf = build_model_performance_panel(perf, brier, tf_active, tf_corrections)
        recent_trades = build_recent_trades_panel(recent_rows)
        active_markets = build_active_markets_panel(market_rows)
        pnl_history = build_pnl_history_panel(daily_pnl)

        return build_layout(header, open_positions, model_perf, recent_trades, active_markets, pnl_history)

    def run(self) -> None:
        console = Console()
        with Live(
            self.render(),
            console=console,
            refresh_per_second=1,
            screen=True,
        ) as live:
            while True:
                time.sleep(self.refresh_seconds)
                live.update(self.render())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WeatherEdge live terminal dashboard (auto-refreshes every N seconds)."
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=3,
        metavar="SECONDS",
        help="Refresh interval in seconds (default: 3)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to weatheredge.db (overrides WEATHEREDGE_DB_PATH env var)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to bot log file for status detection",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Resolve DB path: CLI arg > env var > default
    if args.db:
        db_path = Path(args.db).expanduser().resolve()
    elif "WEATHEREDGE_DB_PATH" in os.environ:
        db_path = Path(os.environ["WEATHEREDGE_DB_PATH"]).expanduser().resolve()
    else:
        db_path = DEFAULT_DB_PATH

    log_path = Path(args.log).expanduser().resolve() if args.log else DEFAULT_LOG_PATH

    dashboard = LiveDashboard(
        db_path=db_path,
        log_path=log_path,
        refresh_seconds=max(1, args.refresh),
    )
    try:
        dashboard.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
