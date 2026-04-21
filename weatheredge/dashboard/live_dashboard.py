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
BOT_FRESHNESS_SECONDS = 120
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
    """Open positions joined with the latest YES price for unrealized P&L.

    `current_yes` is NULL when no recent non-NULL price exists for the market
    (e.g. scraper outage). The previous COALESCE(lp.yes_price, t.price) hid
    the outage by silently substituting the entry price, making positions
    look perfectly flat ($0 unrealized) when in reality the bot was flying
    blind. Returning NULL forces the renderer to show STALE so the operator
    can react.
    """
    return _safe_fetchall(
        conn,
        """
        SELECT t.market_id, t.side, t.price, t.amount, t.cost, t.agent_name,
               t.created_at, t.model_probability, t.edge_at_entry,
               lp.yes_price AS current_yes
        FROM trades t
        LEFT JOIN (
            SELECT market_id, yes_price,
                   ROW_NUMBER() OVER (
                       PARTITION BY market_id ORDER BY timestamp DESC, id DESC
                   ) AS rn
            FROM market_prices
            WHERE market_id IS NOT NULL
              AND yes_price IS NOT NULL
        ) lp ON t.market_id = lp.market_id AND lp.rn = 1
        WHERE t.status = 'open'
        ORDER BY t.created_at DESC
        """,
    )


def query_per_agent_pnl(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Per-agent realized P&L over all resolved trades.

    Sorted by P&L descending so the table reads top-down "best agent first".
    Includes early-exit statuses since they realize P&L too.
    """
    return _safe_fetchall(
        conn,
        """
        SELECT
            COALESCE(agent_name, 'unknown')                         AS agent,
            COUNT(*)                                                AS resolved,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)                AS wins,
            SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END)               AS losses,
            COALESCE(SUM(pnl), 0)                                   AS total_pnl
        FROM trades
        WHERE status IN ('won','lost','edge_stop','capital_protect','time_decay','ev_based','bucket_invalidated','abandoned')
          AND pnl IS NOT NULL
        GROUP BY COALESCE(agent_name, 'unknown')
        ORDER BY total_pnl DESC
        """,
    )


def query_recent_trades(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return _safe_fetchall(
        conn,
        """
        SELECT created_at, market_id, side, price, edge_at_entry,
               agent_name, status, pnl, cost
        FROM trades
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )


def query_performance(conn: sqlite3.Connection) -> dict:
    # All non-open statuses count as resolved (includes edge_stop, won, lost, etc.)
    CLOSED_STATUSES = "('won','lost','edge_stop','capital_protect','time_decay','ev_based','bucket_invalidated')"
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
    """Calculate unrealized P&L by comparing open positions to current market prices.

    Positions whose latest non-NULL `yes_price` is unavailable (scraper outage,
    market just opened, etc.) are excluded from the totals AND counted in
    `stale_positions`. The previous fallback to entry price made these read as
    perfectly flat $0 P&L, which masked outages and lied to the operator.
    """
    rows = _safe_fetchall(
        conn,
        """
        SELECT t.market_id, t.side, t.price, t.amount, t.cost,
               lp.yes_price AS current_yes
        FROM trades t
        LEFT JOIN (
            SELECT market_id, yes_price,
                   ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC, id DESC) AS rn
            FROM market_prices
            WHERE market_id IS NOT NULL
              AND yes_price IS NOT NULL
        ) lp ON t.market_id = lp.market_id AND lp.rn = 1
        WHERE t.status = 'open'
        """,
    )
    total_cost = 0.0
    total_value = 0.0
    positions = 0
    stale = 0
    for r in rows:
        positions += 1
        if r["current_yes"] is None:
            # No recent price → exclude from P&L totals so we don't print a
            # confident-looking $0; surface the count so operators see how
            # many positions are unmarkable.
            stale += 1
            continue
        cost = float(r["cost"] or 0)
        amount = float(r["amount"] or 0)
        current_yes = float(r["current_yes"])
        total_cost += cost
        if r["side"] == "buy_yes":
            total_value += current_yes * amount
        else:
            total_value += (1.0 - current_yes) * amount
    unrealized = total_value - total_cost
    pct = (unrealized / total_cost * 100.0) if total_cost > 0 else 0.0
    return {
        "unrealized_pnl": unrealized,
        "unrealized_pct": pct,
        "total_cost": total_cost,
        "total_value": total_value,
        "open_positions": positions,
        "stale_positions": stale,
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
          AND status IN ('won','lost','edge_stop','capital_protect','time_decay','ev_based','bucket_invalidated')
        GROUP BY day
        ORDER BY day ASC
        LIMIT 30
        """,
    )
    return [(str(r["day"]), float(r["daily_pnl"])) for r in rows]


# ---------------------------------------------------------------------------
# New observability queries (dashboard-only, no algorithm change)
# ---------------------------------------------------------------------------

PROB_FLOOR_VALUE = 0.10  # mirrors weatheredge.probability.gaussian.PROB_FLOOR


def query_open_positions_enriched(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Open positions joined with market thresholds, latest station forecast,
    and calibration RMSE so the dashboard can compute distance-to-band.

    Columns returned add: station, market_date, threshold_low, threshold_high,
    forecast_temp, station_rmse, raw_model_prob on top of the basic trade data.
    LEFT JOINs so missing forecast / calibration do not drop the row.
    """
    return _safe_fetchall(
        conn,
        """
        WITH latest_forecast AS (
            SELECT station, target_date, temperature_forecast,
                   ROW_NUMBER() OVER (
                       PARTITION BY station, target_date
                       ORDER BY forecast_timestamp DESC, id DESC
                   ) AS rn
            FROM forecasts
        ),
        station_rmse AS (
            SELECT station, AVG(rmse) AS rmse
            FROM calibration
            WHERE rmse IS NOT NULL
            GROUP BY station
        ),
        latest_price AS (
            SELECT market_id, yes_price,
                   ROW_NUMBER() OVER (
                       PARTITION BY market_id ORDER BY timestamp DESC, id DESC
                   ) AS rn
            FROM market_prices
            WHERE market_id IS NOT NULL
              AND yes_price IS NOT NULL
        )
        SELECT t.id AS trade_id, t.market_id, t.side, t.price, t.amount, t.cost,
               t.agent_name, t.created_at, t.model_probability, t.edge_at_entry,
               t.raw_model_prob, t.station, t.market_date,
               m.threshold_low, m.threshold_high,
               lf.temperature_forecast AS forecast_temp,
               sr.rmse AS station_rmse,
               lp.yes_price AS current_yes
        FROM trades t
        LEFT JOIN markets m ON t.market_id = m.id
        LEFT JOIN latest_forecast lf
            ON lf.station = t.station
           AND lf.target_date = t.market_date
           AND lf.rn = 1
        LEFT JOIN station_rmse sr ON sr.station = t.station
        LEFT JOIN latest_price lp ON lp.market_id = t.market_id AND lp.rn = 1
        WHERE t.status = 'open'
        ORDER BY t.market_date ASC, t.created_at DESC
        """,
    )


def query_exposure_by_date(conn: sqlite3.Connection) -> list[tuple[str, int, float]]:
    """Per-date exposure summary for open positions.

    Returns a list of (market_date, position_count, total_cost) sorted by date.
    Used by the Risk Cluster gauge.
    """
    rows = _safe_fetchall(
        conn,
        """
        SELECT market_date, COUNT(*) AS n, COALESCE(SUM(cost), 0) AS total_cost
        FROM trades
        WHERE status = 'open'
          AND market_date IS NOT NULL
        GROUP BY market_date
        ORDER BY market_date ASC
        """,
    )
    return [(str(r["market_date"]), int(r["n"] or 0), float(r["total_cost"] or 0.0)) for r in rows]


def query_edge_flip_stats(conn: sqlite3.Connection) -> dict:
    """Lifetime edge_stop statistics for the flip-saved counter.

    Estimates money saved vs a worst-case full-loss (cost) for each
    edge_stop trade: saved = cost - |pnl|. Shown as an approximate figure so
    the operator sees the mechanism at work, not a strict counterfactual.
    """
    rows = _safe_fetchall(
        conn,
        """
        SELECT COALESCE(pnl, 0) AS pnl, COALESCE(cost, 0) AS cost
        FROM trades
        WHERE status = 'edge_stop'
        """,
    )
    total_saved = 0.0
    realised_loss = 0.0
    for r in rows:
        pnl = float(r["pnl"] or 0.0)
        cost = float(r["cost"] or 0.0)
        realised_loss += pnl  # usually negative for edge_stop
        # Saved = cost - |loss|. Clamp at 0 so profitable flips don't distort.
        total_saved += max(0.0, cost + pnl)
    return {
        "count": len(rows),
        "total_saved": total_saved,
        "realised_loss": realised_loss,
    }


def query_source_of_edge(conn: sqlite3.Connection) -> dict:
    """Breakdown of open trades by whether the model signal or the floor is
    driving the pricing probability.

    floor_saturated: raw_model_prob < PROB_FLOOR (the floor is effectively the
    model). model_signal: raw_model_prob >= PROB_FLOOR (the model actually
    contributes).
    """
    total = int(_scalar(conn, "SELECT COUNT(*) FROM trades WHERE status='open'"))
    floor_saturated = int(_scalar(
        conn,
        "SELECT COUNT(*) FROM trades WHERE status='open' AND raw_model_prob IS NOT NULL AND raw_model_prob < ?",
        (PROB_FLOOR_VALUE,),
    ))
    model_signal = int(_scalar(
        conn,
        "SELECT COUNT(*) FROM trades WHERE status='open' AND raw_model_prob IS NOT NULL AND raw_model_prob >= ?",
        (PROB_FLOOR_VALUE,),
    ))
    unknown = max(0, total - floor_saturated - model_signal)
    return {
        "total_open": total,
        "floor_saturated": floor_saturated,
        "model_signal": model_signal,
        "unknown": unknown,
    }


def query_decisive_counts(conn: sqlite3.Connection) -> dict:
    """Decisive market-resolved counts (only won/lost, excludes early-exits)
    used to compute a defensible win-rate with a binomial CI.
    """
    won = int(_scalar(conn, "SELECT COUNT(*) FROM trades WHERE status='won'"))
    lost = int(_scalar(conn, "SELECT COUNT(*) FROM trades WHERE status='lost'"))
    return {"wins": won, "losses": lost, "decisive": won + lost}


def query_delta_24h(conn: sqlite3.Connection) -> float:
    """Net realised P&L over the last 24h (for the header delta ribbon)."""
    return _scalar(
        conn,
        """
        SELECT COALESCE(SUM(pnl), 0)
        FROM trades
        WHERE resolved_at >= datetime('now', '-1 day')
          AND pnl IS NOT NULL
          AND status IN ('won','lost','edge_stop','capital_protect','time_decay','ev_based','bucket_invalidated')
        """,
    )


def query_data_freshness(conn: sqlite3.Connection) -> dict:
    """Last-update seconds-ago for forecasts, market_prices and signals.

    Used by the data-freshness banner. Values are seconds since last write;
    None means no row seen at all (scraper never ran).
    """
    def _age_seconds(sql: str) -> float | None:
        row = _safe_fetchone(conn, sql)
        if row is None or row[0] is None:
            return None
        try:
            return float(row[0])
        except (TypeError, ValueError):
            return None

    return {
        "forecast_age_s": _age_seconds(
            "SELECT (julianday('now') - julianday(MAX(forecast_timestamp))) * 86400.0 FROM forecasts"
        ),
        "market_age_s": _age_seconds(
            "SELECT (julianday('now') - julianday(MAX(timestamp))) * 86400.0 FROM market_prices"
        ),
    }


def query_next_resolution(conn: sqlite3.Connection) -> dict | None:
    """Earliest upcoming market_date among open trades, for the countdown.

    Returns {date, positions, hours_to_resolution} or None.
    Resolution is assumed at 23:59Z of the market_date (end-of-day UTC),
    matching WeatherEdge's daily_high resolution convention. Clamps to 0 if
    past (which means resolution is happening now).
    """
    row = _safe_fetchone(
        conn,
        """
        SELECT market_date, COUNT(*) AS n
        FROM trades
        WHERE status='open' AND market_date IS NOT NULL
        GROUP BY market_date
        ORDER BY market_date ASC
        LIMIT 1
        """,
    )
    if row is None or row["market_date"] is None:
        return None
    date_str = str(row["market_date"])
    n = int(row["n"] or 0)
    try:
        # End-of-day UTC approximation for resolution time.
        dt = datetime.fromisoformat(f"{date_str}T23:59:00+00:00")
        delta = dt - datetime.now(timezone.utc)
        hours = max(0.0, delta.total_seconds() / 3600.0)
    except ValueError:
        hours = None
    return {"date": date_str, "positions": n, "hours_to_resolution": hours}


# ---------------------------------------------------------------------------
# Computation helpers (pure functions; unit-testable)
# ---------------------------------------------------------------------------

def compute_binomial_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson exact binomial CI. Returns (lo, hi) in [0,1].

    Falls back to Wilson if scipy is unavailable so the dashboard keeps
    rendering in stripped-down environments.
    """
    if n <= 0:
        return (0.0, 1.0)
    alpha = 1.0 - conf
    try:
        from scipy.stats import beta  # type: ignore
        lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
        hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
        return (lo, hi)
    except Exception:
        # Wilson score interval fallback (z=1.96 for 95%).
        import math
        z = 1.959963984540054 if conf >= 0.95 else 1.6448536269514722
        phat = k / n
        denom = 1 + z * z / n
        centre = (phat + z * z / (2 * n)) / denom
        half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))


def compute_distance_to_band(
    forecast_temp: float | None,
    low: float | None,
    high: float | None,
) -> float | None:
    """Minimum distance (in °F) from forecast to the nearer band edge.

    For a range market [low, high]:
      - forecast inside band → distance = 0 (dangerous zone)
      - forecast outside band → min(|forecast-low|, |forecast-high|)
    For one-sided thresholds (only one of low/high set) we use that edge.
    Returns None if forecast or both edges are missing.
    """
    if forecast_temp is None:
        return None
    try:
        f = float(forecast_temp)
    except (TypeError, ValueError):
        return None
    edges: list[float] = []
    if low is not None:
        try:
            edges.append(float(low))
        except (TypeError, ValueError):
            pass
    if high is not None:
        try:
            edges.append(float(high))
        except (TypeError, ValueError):
            pass
    if not edges:
        return None
    lo = min(edges)
    hi = max(edges)
    if lo <= f <= hi and len(edges) == 2:
        return 0.0
    return min(abs(f - e) for e in edges)


def compute_system_status(
    exposure_by_date: list[tuple[str, int, float]],
    delta_24h: float,
    initial_bankroll: float,
    freshness: dict,
    bot_running: bool,
) -> tuple[str, str, str]:
    """Aggregate portfolio-level status → (label, color, hint).

    CRITICAL when: bot halted, daily loss >10% bankroll, forecast stale >30min
    CONCENTRATED when: any single date >20% bankroll, total open >35% bankroll
    NORMAL otherwise.
    """
    if not bot_running:
        return ("CRITICAL", "bold red", "Bot not running")
    if initial_bankroll > 0 and delta_24h < -0.10 * initial_bankroll:
        return ("CRITICAL", "bold red", f"24h loss >10% bankroll ({delta_24h:+.2f})")
    fc_age = freshness.get("forecast_age_s")
    if fc_age is not None and fc_age > 1800:  # 30 min
        return ("CRITICAL", "bold red", f"Forecast stale ({fc_age/60:.0f}min)")
    total_open = sum(c for _, _, c in exposure_by_date)
    if initial_bankroll > 0:
        if any(c > 0.20 * initial_bankroll for _, _, c in exposure_by_date):
            offender = max(exposure_by_date, key=lambda r: r[2])
            return (
                "CONCENTRATED",
                "bold yellow",
                f"{offender[0]} holds ${offender[2]:.0f} ({offender[2]/initial_bankroll*100:.0f}%)",
            )
        if total_open > 0.35 * initial_bankroll:
            return (
                "CONCENTRATED",
                "bold yellow",
                f"Total open ${total_open:.0f} ({total_open/initial_bankroll*100:.0f}%)",
            )
    return ("NORMAL", "bold green", "All gauges green")


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
        "bucket_invalidated": "bright_cyan",
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
    system_status: tuple[str, str, str] | None = None,
    delta_24h: float | None = None,
    freshness: dict | None = None,
    next_resolution: dict | None = None,
) -> Panel:
    open_cost = perf.get("open_cost", 0.0)
    total_pnl = perf.get("total_pnl", 0.0)

    # Unrealized P&L from mark-to-market
    unr_pnl = unrealized.get("unrealized_pnl", 0.0) if unrealized else 0.0
    unr_pct = unrealized.get("unrealized_pct", 0.0) if unrealized else 0.0
    open_positions = unrealized.get("open_positions", 0) if unrealized else 0
    stale_positions = unrealized.get("stale_positions", 0) if unrealized else 0

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
    line.append("  ")
    # System status semaforo (NORMAL/CONCENTRATED/CRITICAL)
    if system_status:
        label, style, hint = system_status
        line.append(f"[{label}]", style=style)
        if hint:
            line.append(f" {hint}", style="dim")
    line.append("\n")
    line.append("  Status: ")
    line.append("RUNNING" if bot_running else "STOPPED", style="bold green" if bot_running else "bold red")
    line.append(f"  Uptime: {uptime}")
    line.append(f"  Refresh: {refresh_ts}", style="dim")
    # Data freshness inline: forecast/market price ages
    if freshness is not None:
        fc_age = freshness.get("forecast_age_s")
        mk_age = freshness.get("market_age_s")

        def _fmt_age(age: float | None) -> tuple[str, str]:
            if age is None:
                return ("none", "dim red")
            if age < 60:
                return (f"{age:.0f}s", "dim green")
            if age < 600:
                return (f"{age/60:.0f}m", "dim green")
            if age < 1800:
                return (f"{age/60:.0f}m", "dim yellow")
            return (f"{age/60:.0f}m", "dim red")

        fc_txt, fc_style = _fmt_age(fc_age)
        mk_txt, mk_style = _fmt_age(mk_age)
        line.append("  Data: ", style="dim")
        line.append(f"fc {fc_txt}", style=fc_style)
        line.append(" | ", style="dim")
        line.append(f"mkt {mk_txt}", style=mk_style)
    line.append("\n")
    line.append(
        f"  Bankroll: ${bankroll:,.2f}",
        style="bold cyan",
    )
    # 24h delta ribbon
    if delta_24h is not None:
        d_sign = "+" if delta_24h >= 0 else ""
        d_color = "green" if delta_24h >= 0 else "red"
        d_pct = (delta_24h / initial_bankroll * 100.0) if initial_bankroll else 0.0
        d_pct_sign = "+" if d_pct >= 0 else ""
        line.append(
            f"  24h: {d_sign}${delta_24h:,.2f} ({d_pct_sign}{d_pct:.1f}%)",
            style=d_color,
        )
    line.append(
        f"  Realized: {pnl_sign}${total_pnl:,.2f} ({pnl_pct_sign}{pnl_pct:.1f}%)",
        style=pnl_color,
    )
    line.append(
        f"  Unrealized: {unr_sign}${unr_pnl:,.2f} ({unr_pct_sign}{unr_pct:.1f}%)",
        style=f"bold {unr_color}",
    )
    if stale_positions:
        line.append(
            f"  Positions: {open_positions} ({stale_positions} STALE)",
            style="bold yellow",
        )
    else:
        line.append(f"  Positions: {open_positions}", style="bold yellow")
    line.append(f"  Win Rate: {wr_str}", style="bold")
    # Next resolution countdown
    if next_resolution:
        hrs = next_resolution.get("hours_to_resolution")
        date_str = next_resolution.get("date", "-")
        n = next_resolution.get("positions", 0)
        if hrs is not None:
            if hrs < 1:
                cd = f"{int(hrs * 60)}m"
            elif hrs < 48:
                cd = f"{hrs:.1f}h"
            else:
                cd = f"{hrs/24:.1f}d"
            line.append(f"  Next: {date_str} ({n} pos, in {cd})", style="dim cyan")
        else:
            line.append(f"  Next: {date_str} ({n} pos)", style="dim cyan")

    return Panel(line, style="bold blue", padding=(0, 1))


def _row_unrealized_pnl(row: sqlite3.Row) -> float | None:
    """Per-position mark-to-market P&L using the latest YES price.

    Returns None when the row lacks the joined `current_yes` column or any of
    the price/amount/cost values are unusable, so the table cell can render
    as '-' without crashing.
    """
    try:
        side = str(row["side"] or "")
        amount = float(row["amount"] or 0)
        cost = float(row["cost"] or 0)
        current_yes = row["current_yes"]
        if current_yes is None:
            return None
        current_yes = float(current_yes)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if side == "buy_yes":
        return current_yes * amount - cost
    if side == "buy_no":
        return (1.0 - current_yes) * amount - cost
    return None


def _row_band_label(row: sqlite3.Row) -> str:
    """Format the threshold band into a short label like '64-65F' or '>55F'."""
    try:
        lo = row["threshold_low"]
        hi = row["threshold_high"]
    except (KeyError, IndexError):
        return "-"
    if lo is not None and hi is not None:
        try:
            return f"{float(lo):.0f}-{float(hi):.0f}F"
        except (TypeError, ValueError):
            return "-"
    if lo is not None:
        try:
            return f">{float(lo):.0f}F"
        except (TypeError, ValueError):
            return "-"
    if hi is not None:
        try:
            return f"<{float(hi):.0f}F"
        except (TypeError, ValueError):
            return "-"
    return "-"


def _row_distance_cell(row: sqlite3.Row) -> tuple[Text, Text]:
    """Return (distance, d/rmse+tag) cells for a position.

    Colors encode the "distance / RMSE" ratio:
      <1.0  → red   (lottery zone, trade 12 KDEN territory)
      1-1.25→ yellow (marginal safety margin)
      ≥1.25 → green (safe)
    """
    try:
        forecast = row["forecast_temp"]
        rmse = row["station_rmse"]
        lo = row["threshold_low"]
        hi = row["threshold_high"]
    except (KeyError, IndexError):
        return (Text("-", style="dim"), Text("-", style="dim"))

    dist = compute_distance_to_band(forecast, lo, hi)
    if dist is None:
        return (Text("-", style="dim"), Text("-", style="dim"))
    dist_cell = Text(f"{dist:.1f}F")

    try:
        rmse_f = float(rmse) if rmse is not None else None
    except (TypeError, ValueError):
        rmse_f = None
    if rmse_f is None or rmse_f <= 0:
        return (dist_cell, Text("?", style="dim"))

    ratio = dist / rmse_f
    if ratio < 1.0:
        style = "bold red"
        tag = f"{ratio:.2f}x ZONE"
    elif ratio < 1.25:
        style = "bold yellow"
        tag = f"{ratio:.2f}x MARGIN"
    else:
        style = "bold green"
        tag = f"{ratio:.2f}x SAFE"
    return (dist_cell, Text(tag, style=style))


def build_open_positions_panel(rows: list[sqlite3.Row]) -> Panel:
    if not rows:
        return Panel(
            Text("Waiting for data...", style="dim"),
            title="[bold]OPEN POSITIONS (0)[/bold]",
            border_style="cyan",
            padding=(0, 1),
        )

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Stn", style="white", width=5)
    table.add_column("Date", style="dim", width=10)
    table.add_column("Band", style="white", width=8)
    table.add_column("Side", style="cyan", width=8)
    table.add_column("Px", justify="right", width=5)
    table.add_column("Cost", justify="right", width=6)
    table.add_column("Unr P&L", justify="right", width=8)
    table.add_column("Dist", justify="right", width=6)
    table.add_column("D/RMSE", justify="right", width=12)

    for row in rows:
        side_text = Text(str(row["side"] or "-"))
        side_style = "green" if "yes" in str(row["side"] or "").lower() else "yellow"
        side_text.stylize(side_style)
        unrealized = _row_unrealized_pnl(row)
        if unrealized is not None:
            unr_cell = _fmt_pnl(unrealized)
        else:
            try:
                is_stale = row["current_yes"] is None
            except (KeyError, IndexError):
                is_stale = False
            unr_cell = Text("STALE" if is_stale else "-", style="bold yellow" if is_stale else "dim")

        dist_cell, ratio_cell = _row_distance_cell(row)
        try:
            station = str(row["station"] or "-")[:5]
        except (KeyError, IndexError):
            station = "-"
        try:
            mkt_date = str(row["market_date"] or "-")[-10:]
        except (KeyError, IndexError):
            mkt_date = "-"

        table.add_row(
            station,
            mkt_date,
            _row_band_label(row),
            side_text,
            _fmt_price(row["price"]),
            _fmt_price(row["cost"]),
            unr_cell,
            dist_cell,
            ratio_cell,
        )

    return Panel(
        table,
        title=f"[bold]OPEN POSITIONS ({len(rows)})[/bold]",
        border_style="cyan",
        padding=(0, 0),
    )


def build_per_agent_pnl_panel(rows: list[sqlite3.Row]) -> Panel:
    """Per-agent realized P&L panel — surfaces which agent is making/losing money."""
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Agent", style="white", max_width=14)
    table.add_column("W/L", justify="right", width=8)
    table.add_column("Resolved", justify="right", width=9)
    table.add_column("P&L", justify="right", width=10)

    if not rows:
        table.add_row(Text("Waiting for data...", style="dim"), "", "", "")
    else:
        for row in rows:
            wins = int(row["wins"] or 0)
            losses = int(row["losses"] or 0)
            resolved = int(row["resolved"] or 0)
            pnl = float(row["total_pnl"] or 0)
            table.add_row(
                str(row["agent"] or "-")[:14],
                f"{wins}/{losses}",
                str(resolved),
                _fmt_pnl(pnl),
            )

    return Panel(
        table,
        title="[bold]PER-AGENT P&L[/bold]",
        border_style="magenta",
        padding=(0, 0),
    )


def build_model_performance_panel(
    perf: dict,
    brier: float | None,
    timesfm_active: bool,
    timesfm_corrections: int,
    decisive: dict | None = None,
    edge_source: dict | None = None,
    flip_stats: dict | None = None,
) -> Panel:
    won_final = perf.get("won", 0)
    lost_final = perf.get("lost", 0)
    wins_all = perf.get("wins_all", 0)
    resolved = perf.get("resolved", 0)
    win_rate = perf.get("win_rate")
    best_agent = perf.get("best_agent")
    best_wr = perf.get("best_agent_wr")

    lines = Text()
    brier_str = f"{brier:.4f}" if brier is not None else "N/A"
    wr_str = f"{win_rate:.1f}%" if win_rate is not None else "N/A"

    lines.append("Brier:    ", style="dim")
    lines.append(f"{brier_str}\n", style="bold")

    lines.append("Win Rate: ", style="dim")
    lines.append(f"{wr_str}", style="bold green" if (win_rate or 0) >= 55 else "bold yellow")
    lines.append(f"  ({wins_all}/{resolved} closed)\n", style="dim")

    # Decisive win rate with binomial 95% CI — makes sample-size fragility visible.
    if decisive is not None:
        wins = int(decisive.get("wins", 0))
        losses = int(decisive.get("losses", 0))
        n = wins + losses
        if n > 0:
            phat = wins / n
            lo, hi = compute_binomial_ci(wins, n, conf=0.95)
            ci_width = hi - lo
            ci_color = "bold red" if ci_width > 0.50 else ("bold yellow" if ci_width > 0.25 else "bold green")
            lines.append("Decisive: ", style="dim")
            lines.append(f"{wins}/{n} = {phat*100:.0f}%", style="bold")
            lines.append(f"  95%CI [{lo*100:.0f}%, {hi*100:.0f}%]\n", style=ci_color)
        else:
            lines.append("Decisive: ", style="dim")
            lines.append("no won/lost yet\n", style="dim")

    lines.append("Final:    ", style="dim")
    lines.append(f"{won_final}", style="green")
    lines.append(" won / ", style="dim")
    lines.append(f"{lost_final}", style="red")
    lines.append(" lost\n", style="dim")

    # Source of edge: floor-saturated vs genuine model signal among open trades.
    if edge_source is not None and edge_source.get("total_open", 0) > 0:
        total = int(edge_source["total_open"])
        floor = int(edge_source.get("floor_saturated", 0))
        model = int(edge_source.get("model_signal", 0))
        floor_pct = floor / total * 100 if total else 0.0
        model_pct = model / total * 100 if total else 0.0
        lines.append("Edge Src: ", style="dim")
        lines.append(f"Arb {floor_pct:.0f}%", style="yellow")
        lines.append(" | ", style="dim")
        lines.append(f"Model {model_pct:.0f}%", style="cyan")
        lines.append(f"  ({floor}+{model}/{total})\n", style="dim")

    # Edge Flip lifetime saver: shows the exit mechanism paying off.
    if flip_stats is not None and flip_stats.get("count", 0) > 0:
        count = int(flip_stats["count"])
        saved = float(flip_stats.get("total_saved", 0.0))
        lines.append("Flip Save:", style="dim")
        lines.append(f" {count}x ", style="bold cyan")
        lines.append(f"~${saved:.2f}\n", style="cyan")

    if best_agent and best_wr is not None:
        lines.append("Best Ag.: ", style="dim")
        lines.append(f"{best_agent[:12]} ({best_wr:.0f}%)\n", style="bold magenta")

    lines.append("TimesFM:  ", style="dim")
    if timesfm_active:
        lines.append("On ", style="bold green")
        lines.append("\u2713", style="bold green")
    else:
        lines.append("Off \u2717", style="dim red")
    lines.append(f"  ({timesfm_corrections})", style="dim")

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

            # Edge-flip badge: status=edge_stop means the exit manager
            # bailed out early. The "saved" estimate is cost + pnl when
            # pnl is negative (the remaining capital we recovered vs a
            # hypothetical total-loss outcome).
            if status == "edge_stop":
                cost = row["cost"]
                pnl = row["pnl"]
                saved = None
                try:
                    if cost is not None and pnl is not None and float(pnl) < 0:
                        saved = float(cost) + float(pnl)
                except (TypeError, ValueError):
                    saved = None
                status_cell = Text()
                status_cell.append("EDGE-FLIP ", style="bold magenta")
                if saved is not None and saved > 0:
                    status_cell.append(f"+${saved:.2f}", style="dim green")
            else:
                status_cell = Text(status, style=st)

            table.add_row(
                _fmt_ts(row["created_at"])[-14:],
                str(row["market_id"] or "-")[-22:],
                Text(side_str, style=side_style),
                _fmt_price(row["price"]),
                _fmt_edge(row["edge_at_entry"]),
                str(row["agent_name"] or "-")[:14],
                status_cell,
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


def build_risk_cluster_panel(
    exposure_by_date: list[tuple[str, int, float]],
    initial_bankroll: float,
) -> Panel:
    """Bar gauge showing concentration of open cost per resolution date.

    Highlights correlation risk: multiple positions on the same date share
    the same synoptic regime, so a single regime miss can take them all.
    Color thresholds: >20% red, >10% yellow, else green.
    """
    table = Table(
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
        expand=True,
    )
    table.add_column("Date", style="dim", width=10)
    table.add_column("#", justify="right", width=3)
    table.add_column("Cost", justify="right", width=9)
    table.add_column("% BR", justify="right", width=6)
    table.add_column("Gauge", no_wrap=True)

    if not exposure_by_date:
        table.add_row(Text("No open positions", style="dim"), "", "", "", "")
        return Panel(
            table,
            title="[bold]RISK CLUSTER (per date)[/bold]",
            border_style="magenta",
            padding=(0, 0),
        )

    bankroll = initial_bankroll if initial_bankroll else 1.0

    for date_str, count, total_cost in exposure_by_date:
        pct = (total_cost / bankroll * 100.0) if bankroll else 0.0
        if pct >= 20.0:
            color = "red"
        elif pct >= 10.0:
            color = "yellow"
        else:
            color = "green"

        bar_len = min(int(pct / 2), 20)
        bar = "█" * max(bar_len, 1)

        table.add_row(
            str(date_str or "-")[:10],
            str(count),
            f"${total_cost:,.2f}",
            f"{pct:.1f}%",
            Text(bar, style=color),
        )

    return Panel(
        table,
        title="[bold]RISK CLUSTER (per date)[/bold]",
        border_style="magenta",
        padding=(0, 0),
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
    per_agent_pnl: Panel,
    risk_cluster: Panel | None = None,
) -> Layout:
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=5),
        Layout(name="middle", size=12),
        Layout(name="recent", size=14),
        Layout(name="bottom", size=12),
    )

    layout["header"].update(header)

    # Middle row now has three columns: open positions on the left, model
    # performance in the middle, per-agent P&L on the right. Ratios chosen so
    # the agent table gets enough width for $-amounts without crowding the
    # market column in the open-positions table.
    layout["middle"].split_row(
        Layout(name="open_pos", ratio=2),
        Layout(name="model_perf", ratio=1),
        Layout(name="per_agent", ratio=1),
    )
    layout["middle"]["open_pos"].update(open_positions)
    layout["middle"]["model_perf"].update(model_perf)
    layout["middle"]["per_agent"].update(per_agent_pnl)

    layout["recent"].update(recent_trades)

    if risk_cluster is not None:
        layout["bottom"].split_row(
            Layout(name="active_markets", ratio=3),
            Layout(name="risk_cluster", ratio=2),
            Layout(name="pnl_history", ratio=2),
        )
        layout["bottom"]["active_markets"].update(active_markets)
        layout["bottom"]["risk_cluster"].update(risk_cluster)
        layout["bottom"]["pnl_history"].update(pnl_history)
    else:
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
                # Enriched query replaces the legacy open_positions call — it
                # carries the columns the distance/RMSE column needs.
                open_rows = query_open_positions_enriched(conn)
                recent_rows = query_recent_trades(conn)
                perf = query_performance(conn)
                brier = query_brier_score(conn)
                tf_corrections = query_timesfm_corrections(conn)
                market_rows = query_active_markets(conn)
                daily_pnl = query_daily_pnl(conn)
                unrealized = query_unrealized_pnl(conn)
                per_agent_rows = query_per_agent_pnl(conn)
                exposure_by_date = query_exposure_by_date(conn)
                flip_stats = query_edge_flip_stats(conn)
                edge_source = query_source_of_edge(conn)
                decisive = query_decisive_counts(conn)
                delta_24h = query_delta_24h(conn)
                freshness = query_data_freshness(conn)
                next_resolution = query_next_resolution(conn)
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
                per_agent_rows = []
                exposure_by_date = []
                flip_stats = {"count": 0, "total_saved": 0.0, "realised_loss": 0.0}
                edge_source = {"total_open": 0, "floor_saturated": 0, "model_signal": 0, "unknown": 0}
                decisive = {"wins": 0, "losses": 0, "decisive": 0}
                delta_24h = 0.0
                freshness = {"forecast_age_s": None, "market_age_s": None}
                next_resolution = None
        finally:
            if conn is not None:
                conn.close()

        system_status = compute_system_status(
            exposure_by_date,
            delta_24h,
            self._initial_bankroll,
            freshness,
            bot_running,
        )

        header = build_header_panel(
            bot_running,
            uptime,
            perf,
            self._initial_bankroll,
            now_str,
            unrealized,
            system_status=system_status,
            delta_24h=delta_24h,
            freshness=freshness,
            next_resolution=next_resolution,
        )
        open_positions = build_open_positions_panel(open_rows)
        model_perf = build_model_performance_panel(
            perf,
            brier,
            tf_active,
            tf_corrections,
            decisive=decisive,
            edge_source=edge_source,
            flip_stats=flip_stats,
        )
        recent_trades = build_recent_trades_panel(recent_rows)
        active_markets = build_active_markets_panel(market_rows)
        pnl_history = build_pnl_history_panel(daily_pnl)
        per_agent_pnl = build_per_agent_pnl_panel(per_agent_rows)
        risk_cluster = build_risk_cluster_panel(exposure_by_date, self._initial_bankroll)

        return build_layout(
            header, open_positions, model_perf, recent_trades,
            active_markets, pnl_history, per_agent_pnl,
            risk_cluster=risk_cluster,
        )

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
