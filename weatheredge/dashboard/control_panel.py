"""WeatherEdge Control Panel — Streamlit web UI with Phase Selector.

Launch: streamlit run weatheredge/dashboard/control_panel.py
"""

from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Ensure project root is on sys.path so weatheredge package is importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
DB_PATH = PROJECT_ROOT / "weatheredge.db"
LOG_PATH = PROJECT_ROOT / "logs" / "weatheredge.log"
PID_FILE = PROJECT_ROOT / ".weatheredge.pid"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

st.set_page_config(
    page_title="WeatherEdge",
    page_icon="\U0001f324",
    layout="wide",
)

# ── Dark Theme CSS ───────────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; }
    .module-on { color: #00ff88; font-weight: bold; }
    .module-off { color: #ff4444; }
    .module-kill { color: #666666; text-decoration: line-through; }
    .gate-pass { color: #00ff88; }
    .gate-fail { color: #ffaa00; }
    .metric-label { font-size: 0.8rem; color: #888; }
    div[data-testid="stMetric"] label { font-family: monospace; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────

def _is_running() -> tuple[bool, int | None]:
    if not PID_FILE.exists():
        return False, None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True, pid
    except (ValueError, ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return False, None


def _start_bot(phase: str = "phase_0"):
    if _is_running()[0]:
        return
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    proc = subprocess.Popen(
        [str(VENV_PYTHON), str(PROJECT_ROOT / "launch.py"), "--phase", phase],
        cwd=str(PROJECT_ROOT),
        stdout=open(PROJECT_ROOT / "logs" / "stdout.log", "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))


def _stop_bot():
    running, pid = _is_running()
    if not running or pid is None:
        PID_FILE.unlink(missing_ok=True)
        return
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass
    PID_FILE.unlink(missing_ok=True)


def _query_db(sql: str, params: tuple = ()) -> list:
    if not DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def _scalar(sql: str, default=0):
    rows = _query_db(sql)
    if rows and rows[0][0] is not None:
        return rows[0][0]
    return default


def _tail_log(n: int = 20) -> list[str]:
    if not LOG_PATH.exists():
        return []
    try:
        lines = LOG_PATH.read_text(errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


def _get_phase_config():
    from weatheredge.phase_config import (
        PHASE_CONFIGS, PHASE_ORDER, get_current_phase, get_phase_modules, get_gate_status,
    )
    return PHASE_CONFIGS, PHASE_ORDER, get_current_phase, get_phase_modules, get_gate_status


# Structural kills — never re-enable
STRUCTURAL_KILLS = {"agent_mean_reversion", "vpin", "exit_pnl_stoploss"}


# ── Header ───────────────────────────────────────────────────────────────

running, pid = _is_running()
PHASE_CONFIGS, PHASE_ORDER, get_current_phase, get_phase_modules, get_gate_status = _get_phase_config()
current_phase = get_current_phase()
current_config = PHASE_CONFIGS[current_phase]

st.markdown(
    f"""
    <div style="text-align:center; padding: 0.5rem 0 0.5rem 0;">
        <h1 style="margin:0;">\U0001f324 WeatherEdge</h1>
        <p style="color:#888; margin:0;">{current_config['name']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Controls ─────────────────────────────────────────────────────────────

col_status, col_start, col_stop, col_refresh = st.columns([2, 1, 1, 1])

with col_status:
    if running:
        st.success(f"BOT RUNNING  (PID {pid})")
    else:
        st.error("BOT STOPPED")

with col_start:
    if st.button("\u25b6  Start", use_container_width=True, type="primary", disabled=running):
        _start_bot(current_phase)
        time.sleep(2)
        st.rerun()

with col_stop:
    if st.button("\u23f9  Stop", use_container_width=True, type="secondary", disabled=not running):
        _stop_bot()
        time.sleep(1)
        st.rerun()

with col_refresh:
    if st.button("\U0001f504  Refresh", use_container_width=True):
        st.rerun()

st.divider()

# ── Sidebar: Phase Selector ─────────────────────────────────────────────

st.sidebar.markdown("### Phase Selector")

for phase_key in PHASE_ORDER:
    config = PHASE_CONFIGS[phase_key]
    is_active = (phase_key == current_phase)
    icon = "\U0001f7e2" if is_active else "\u26aa"

    with st.sidebar.expander(f"{icon} {config['name']}", expanded=is_active):
        st.caption(config["description"])
        st.markdown("**Details:**")
        for d in config["details"]:
            st.markdown(f"- {d}")
        st.markdown(f"**Gate:** {config.get('gate', 'N/A')}")

        if not is_active:
            if st.button(f"ATTIVA {phase_key.upper()}", key=f"activate_{phase_key}"):
                from weatheredge.phase_config import activate_phase
                activate_phase(phase_key)
                # Bot reads current_phase.json on each cycle — no restart needed.
                # Do NOT call _start_bot here: it would spawn a second instance on the main DB.
                st.rerun()
        else:
            st.info("Fase attiva")

st.sidebar.divider()
st.sidebar.markdown("### Settings")
auto = st.sidebar.toggle("Auto-refresh", value=True)
interval = st.sidebar.slider("Refresh (sec)", 5, 60, 15)

# ── Stats ────────────────────────────────────────────────────────────────

_CLOSED = "('won','lost','edge_stop','capital_protect','time_decay','ev_based','abandoned')"
total_trades = _scalar("SELECT COUNT(*) FROM trades")
open_trades = _scalar("SELECT COUNT(*) FROM trades WHERE status='open'")
won = _scalar("SELECT COUNT(*) FROM trades WHERE status='won'")
lost = _scalar("SELECT COUNT(*) FROM trades WHERE status='lost'")
# resolved = all closed statuses, not just won/lost
resolved = _scalar(f"SELECT COUNT(*) FROM trades WHERE status IN {_CLOSED}")
# wins_all = any closed trade with pnl > 0
wins_all = _scalar(f"SELECT COUNT(*) FROM trades WHERE status IN {_CLOSED} AND pnl > 0")
total_pnl = _scalar(f"SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status IN {_CLOSED}")
forecasts = _scalar("SELECT COUNT(*) FROM forecasts")
markets = _scalar("SELECT COUNT(DISTINCT id) FROM markets")
win_rate = (wins_all / resolved * 100) if resolved > 0 else 0.0

# Unrealized P&L for open positions
unrealized_pnl = 0.0
_unrealized_rows = _query_db("""
    SELECT t.side, t.price, t.amount, t.cost, mp.yes_price
    FROM trades t
    LEFT JOIN (
        SELECT market_id, yes_price,
               ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC) as rn
        FROM market_prices
    ) mp ON t.market_id = mp.market_id AND mp.rn = 1
    WHERE t.status = 'open' AND mp.yes_price IS NOT NULL
""")
for r in _unrealized_rows:
    if r["side"] == "buy_yes":
        unrealized_pnl += (r["yes_price"] * r["amount"]) - r["cost"]
    else:
        unrealized_pnl += ((1.0 - r["yes_price"]) * r["amount"]) - r["cost"]

# Brier score — only market-resolved trades (won/lost); outcome_yes depends on side
brier = 0.0
brier_rows = _query_db(
    "SELECT model_probability, status, side FROM trades WHERE status IN ('won','lost') AND model_probability IS NOT NULL"
)
if brier_rows:
    brier_sum = 0.0
    for r in brier_rows:
        # outcome_yes=1 if YES resolved, 0 if NO resolved
        if (r["side"] == "buy_yes" and r["status"] == "won") or \
           (r["side"] == "buy_no" and r["status"] == "lost"):
            outcome_yes = 1.0
        else:
            outcome_yes = 0.0
        brier_sum += (r["model_probability"] - outcome_yes) ** 2
    brier = brier_sum / len(brier_rows)

# Total equity = initial bankroll + realized P&L (all exited trades) + unrealized P&L (open positions).
# This is the headline number — the actual portfolio value right now.
_initial_bankroll = 100.0
try:
    from weatheredge.config import Settings as _Settings
    _initial_bankroll = float(_Settings().initial_bankroll)
except Exception:
    pass
total_equity = _initial_bankroll + total_pnl + unrealized_pnl
equity_delta = total_equity - _initial_bankroll

eq1, eq2, eq3 = st.columns([2, 1, 1])
eq1.metric(
    "Total Equity",
    f"${total_equity:.2f}",
    delta=f"{equity_delta:+.2f} ({100*equity_delta/_initial_bankroll:+.1f}%)",
    help=f"Initial ${_initial_bankroll:.2f} + realized ${total_pnl:.2f} + unrealized ${unrealized_pnl:.2f}",
)
eq2.metric("Realized P&L", f"${total_pnl:.2f}", help="Sum of pnl across ALL exited trades (won, lost, abandoned, edge_stop, etc.)")
eq3.metric("Unrealized P&L", f"${unrealized_pnl:.2f}", help="Mark-to-market on currently open positions")

c1, c2, c3, c4, c4b = st.columns(5)
c1.metric("Total Trades", total_trades)
c2.metric("Win Rate", f"{win_rate:.1f}%", help=f"{wins_all}/{resolved} closed trades with pnl>0")
c3.metric("Open Positions", open_trades)
c4.metric("Resolved", resolved, help="All exited trades (won, lost, abandoned, edge_stop, capital_protect, time_decay, ev_based)")
c4b.metric("Brier Score", f"{brier:.4f}" if brier_rows else "N/A", help="Only market-resolved (won/lost) trades")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Won (final)", won)
c6.metric("Lost (final)", lost)
c7.metric("Forecasts", forecasts)
c8.metric("Markets", markets)

st.divider()

# ── Module Status Grid ───────────────────────────────────────────────────

st.subheader("Module Status")
current_modules = get_phase_modules(current_phase)

col_a, col_b, col_c = st.columns(3)
sorted_mods = sorted(current_modules.items())
third = len(sorted_mods) // 3 + 1

for idx, (col, chunk) in enumerate(zip(
    [col_a, col_b, col_c],
    [sorted_mods[:third], sorted_mods[third:2*third], sorted_mods[2*third:]],
)):
    with col:
        for mod_name, enabled in chunk:
            if mod_name in STRUCTURAL_KILLS:
                st.markdown(f"<span class='module-kill'>\u2620 {mod_name}</span>", unsafe_allow_html=True)
            elif enabled:
                st.markdown(f"<span class='module-on'>\u2705 {mod_name}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='module-off'>\u274c {mod_name}</span>", unsafe_allow_html=True)

st.divider()

# ── Gate Status ──────────────────────────────────────────────────────────

st.subheader("Gate Status")
gate_cols = st.columns(len(PHASE_ORDER))
for i, phase_key in enumerate(PHASE_ORDER):
    config = PHASE_CONFIGS[phase_key]
    gate_result = get_gate_status(phase_key)
    with gate_cols[i]:
        is_active = (phase_key == current_phase)
        st.caption(phase_key.replace("_", " ").title())
        if is_active:
            st.markdown(f"**{config.get('gate', 'N/A')}**")
        else:
            st.markdown(f"{config.get('gate', 'N/A')}")
        passed = gate_result.get("passed")
        if passed is True:
            st.markdown("<span class='gate-pass'>PASS</span>", unsafe_allow_html=True)
        elif passed is False:
            st.markdown("<span class='gate-fail'>FAIL</span>", unsafe_allow_html=True)
            if gate_result.get("reason"):
                st.caption(str(gate_result["reason"]))
        else:
            st.caption("N/A")

st.divider()

# ── Equity Curve ─────────────────────────────────────────────────────────

st.subheader("Equity Curve")

import altair as alt
import pandas as pd

equity_rows = _query_db(
    f"SELECT resolved_at, pnl FROM trades "
    f"WHERE status IN {_CLOSED} AND pnl IS NOT NULL AND resolved_at IS NOT NULL "
    f"ORDER BY resolved_at ASC"
)

if not equity_rows:
    st.info("Nessun trade risolto ancora — equity curve disponibile dopo la prima chiusura.")
else:
    df = pd.DataFrame([dict(r) for r in equity_rows])
    df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["resolved_at"]).sort_values("resolved_at").reset_index(drop=True)
    df["day"] = df["resolved_at"].dt.tz_convert("UTC").dt.date

    initial_bankroll = _initial_bankroll  # Reuse value loaded for Total Equity metric

    min_day = df["day"].min()
    max_day = df["day"].max()

    eq_col1, eq_col2, eq_col3 = st.columns([2, 1, 1])
    with eq_col1:
        date_range = st.date_input(
            "Intervallo date",
            value=(min_day, max_day),
            min_value=min_day,
            max_value=max_day,
            key="equity_date_range",
        )
    with eq_col2:
        granularity = st.selectbox(
            "Granularità", ["Per trade", "Daily"], index=0, key="equity_granularity"
        )
    with eq_col3:
        min_threshold = st.number_input(
            "Soglia minima ($)",
            value=float(initial_bankroll * 0.9),
            step=1.0,
            key="equity_min_threshold",
            help="Linea di riferimento — eventi sotto soglia sono evidenziati.",
        )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        d_from, d_to = date_range
    else:
        d_from = d_to = date_range if not isinstance(date_range, tuple) else min_day

    mask = (df["day"] >= d_from) & (df["day"] <= d_to)
    sub = df.loc[mask].copy()

    if sub.empty:
        st.warning("Nessun trade nell'intervallo selezionato.")
    else:
        if granularity == "Daily":
            daily = sub.groupby("day", as_index=False)["pnl"].sum()
            daily["cum_pnl"] = daily["pnl"].cumsum()
            daily["equity"] = initial_bankroll + daily["cum_pnl"]
            daily["timestamp"] = pd.to_datetime(daily["day"])
            plot_df = daily[["timestamp", "equity", "pnl"]].rename(columns={"pnl": "delta"})
        else:
            sub["cum_pnl"] = sub["pnl"].cumsum()
            sub["equity"] = initial_bankroll + sub["cum_pnl"]
            plot_df = sub[["resolved_at", "equity", "pnl"]].rename(
                columns={"resolved_at": "timestamp", "pnl": "delta"}
            )

        plot_df["running_max"] = plot_df["equity"].cummax()
        plot_df["drawdown"] = plot_df["equity"] - plot_df["running_max"]

        max_idx = plot_df["equity"].idxmax()
        min_idx = plot_df["equity"].idxmin()
        max_dd = float(plot_df["drawdown"].min())
        max_dd_pct = (max_dd / float(plot_df.loc[plot_df["drawdown"].idxmin(), "running_max"]) * 100.0) \
            if plot_df.loc[plot_df["drawdown"].idxmin(), "running_max"] else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Equity finale", f"${float(plot_df['equity'].iloc[-1]):.2f}")
        m2.metric("Max equity", f"${float(plot_df['equity'].max()):.2f}")
        m3.metric("Min equity", f"${float(plot_df['equity'].min()):.2f}")
        m4.metric("Max drawdown", f"${max_dd:.2f}", delta=f"{max_dd_pct:.1f}%")

        line = alt.Chart(plot_df).mark_line(color="#00ccff", strokeWidth=2).encode(
            x=alt.X("timestamp:T", title="Tempo"),
            y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Quando"),
                alt.Tooltip("equity:Q", title="Equity", format="$.2f"),
                alt.Tooltip("delta:Q", title="ΔP&L", format="$.2f"),
                alt.Tooltip("drawdown:Q", title="Drawdown", format="$.2f"),
            ],
        )

        threshold_line = alt.Chart(pd.DataFrame({"y": [float(min_threshold)]})).mark_rule(
            color="#ffaa00", strokeDash=[6, 4]
        ).encode(y="y:Q")

        bankroll_line = alt.Chart(pd.DataFrame({"y": [initial_bankroll]})).mark_rule(
            color="#666", strokeDash=[2, 4]
        ).encode(y="y:Q")

        markers_df = plot_df.loc[[max_idx, min_idx]].copy()
        markers_df["label"] = ["MAX", "MIN"]
        markers = alt.Chart(markers_df).mark_point(size=120, filled=True).encode(
            x="timestamp:T",
            y="equity:Q",
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=["MAX", "MIN"], range=["#00ff88", "#ff4444"]),
                legend=alt.Legend(title="Estremi"),
            ),
            tooltip=["label:N", alt.Tooltip("equity:Q", format="$.2f")],
        )

        chart = (bankroll_line + threshold_line + line + markers).properties(height=320)
        st.altair_chart(chart, use_container_width=True)

        with st.expander("Drawdown nel tempo"):
            dd_chart = alt.Chart(plot_df).mark_area(
                color="#ff4444", opacity=0.4
            ).encode(
                x=alt.X("timestamp:T", title="Tempo"),
                y=alt.Y("drawdown:Q", title="Drawdown ($)"),
                tooltip=[
                    alt.Tooltip("timestamp:T"),
                    alt.Tooltip("drawdown:Q", format="$.2f"),
                ],
            ).properties(height=180)
            st.altair_chart(dd_chart, use_container_width=True)

st.divider()

# ── P&L by Lead Time ────────────────────────────────────────────────────

st.subheader("P&L by Lead Time")
# Approximate lead time from created_at vs market_date
pnl_lt_rows = _query_db("""
    SELECT
        CASE
            WHEN t.market_date IS NOT NULL AND t.created_at IS NOT NULL THEN
                CAST((julianday(t.market_date) - julianday(t.created_at)) * 24 AS INTEGER)
            ELSE -1
        END as lead_hours,
        t.pnl, t.status
    FROM trades t WHERE t.status IN ('won','lost') AND t.pnl IS NOT NULL
""")

if pnl_lt_rows:
    buckets = {"<12h": 0.0, "12-24h": 0.0, ">24h": 0.0, "unknown": 0.0}
    counts = {"<12h": 0, "12-24h": 0, ">24h": 0, "unknown": 0}
    for r in pnl_lt_rows:
        lh = r["lead_hours"]
        if lh < 0:
            k = "unknown"
        elif lh < 12:
            k = "<12h"
        elif lh < 24:
            k = "12-24h"
        else:
            k = ">24h"
        buckets[k] += r["pnl"]
        counts[k] += 1

    lt_cols = st.columns(4)
    for i, (k, v) in enumerate(buckets.items()):
        lt_cols[i].metric(f"P&L {k}", f"${v:.2f}", help=f"{counts[k]} trades")
else:
    st.info("No resolved trades with lead time data")

st.divider()

# ── Open Positions ───────────────────────────────────────────────────────

st.subheader("Open Positions")
open_rows = _query_db("""
    SELECT t.market_id, t.side, t.price as entry_yes_price, t.amount, t.cost,
           t.agent_name, t.platform, t.station, t.market_date, t.created_at,
           mp.yes_price as current_yes_price,
           CASE
               WHEN t.side = 'buy_yes' THEN ROUND((mp.yes_price * t.amount) - t.cost, 2)
               ELSE ROUND(((1.0 - mp.yes_price) * t.amount) - t.cost, 2)
           END as unrealized_pnl,
           CASE
               WHEN t.cost > 0 THEN ROUND(
                   CASE
                       WHEN t.side = 'buy_yes' THEN ((mp.yes_price * t.amount) - t.cost) / t.cost * 100
                       ELSE (((1.0 - mp.yes_price) * t.amount) - t.cost) / t.cost * 100
                   END, 1)
               ELSE 0
           END as pnl_pct
    FROM trades t
    LEFT JOIN (
        SELECT market_id, yes_price,
               ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC) as rn
        FROM market_prices
    ) mp ON t.market_id = mp.market_id AND mp.rn = 1
    WHERE t.status = 'open'
    ORDER BY t.created_at DESC LIMIT 20
""")
if open_rows:
    st.dataframe([dict(r) for r in open_rows], use_container_width=True, hide_index=True)
else:
    st.info("No open positions")

# ── Recent Trades ────────────────────────────────────────────────────────

st.subheader("Recent Resolved Trades")
recent = _query_db(
    "SELECT market_id, side, price, amount, cost, pnl, status, platform, agent_name, resolved_at "
    "FROM trades WHERE status IN ('won','lost','edge_stop','capital_protect','time_decay','ev_based') "
    "ORDER BY resolved_at DESC LIMIT 15"
)
if recent:
    st.dataframe([dict(r) for r in recent], use_container_width=True, hide_index=True)
else:
    st.info("No resolved trades yet")

# ── Agent Performance ────────────────────────────────────────────────────

st.subheader("Agent Performance")
agents = _query_db(
    "SELECT agent_name, COUNT(*) as trades, "
    "SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins, "
    "SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) as losses, "
    "COALESCE(SUM(pnl), 0) as total_pnl "
    "FROM trades WHERE status IN ('won','lost') "
    "GROUP BY agent_name ORDER BY total_pnl DESC"
)
if agents:
    agent_data = []
    for a in agents:
        d = dict(a)
        t = d["wins"] + d["losses"]
        d["win_rate"] = f"{d['wins'] / t * 100:.1f}%" if t > 0 else "\u2014"
        d["total_pnl"] = f"${d['total_pnl']:.2f}"
        agent_data.append(d)
    st.dataframe(agent_data, use_container_width=True, hide_index=True)
else:
    st.info("No agent performance data yet")

# ── Log Tail ─────────────────────────────────────────────────────────────

st.subheader("Recent Logs")
log_lines = _tail_log(25)
if log_lines:
    st.code("\n".join(log_lines), language="log")
else:
    st.info("No log file found")

# ── Auto-refresh ─────────────────────────────────────────────────────────

if auto:
    time.sleep(interval)
    st.rerun()
