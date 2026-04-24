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
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,650&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg: #f6f1e8;
        --ink: #18201b;
        --muted: #667167;
        --card: rgba(255, 252, 245, 0.88);
        --line: rgba(24, 32, 27, 0.12);
        --green: #0c7a43;
        --red: #b63b2e;
        --amber: #b66a00;
        --blue: #235c8f;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 191, 105, 0.22), transparent 28rem),
            radial-gradient(circle at 80% 10%, rgba(45, 109, 150, 0.16), transparent 24rem),
            linear-gradient(135deg, #f6f1e8 0%, #ece3d4 100%);
        color: var(--ink);
        font-family: 'IBM Plex Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Fraunces', serif;
        letter-spacing: -0.03em;
    }

    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 30px rgba(53, 43, 27, 0.08);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        border: 1px solid var(--line);
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(24, 32, 27, 0.94), rgba(36, 55, 44, 0.90)),
            radial-gradient(circle at right, rgba(255, 191, 105, 0.22), transparent 18rem);
        color: #fffaf0;
        box-shadow: 0 18px 50px rgba(24, 32, 27, 0.22);
    }

    .hero-title {
        font-family: 'Fraunces', serif;
        font-size: clamp(2rem, 4vw, 4rem);
        line-height: 0.95;
        margin: 0 0 0.4rem 0;
    }

    .hero-subtitle {
        color: rgba(255, 250, 240, 0.78);
        font-size: 1rem;
        margin: 0;
    }

    .card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: 0 12px 30px rgba(53, 43, 27, 0.08);
        min-height: 7.2rem;
    }

    .card-label {
        color: var(--muted);
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .card-value {
        color: var(--ink);
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }

    .card-note {
        color: var(--muted);
        font-size: 0.86rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .pill-ok { background: rgba(12, 122, 67, 0.13); color: var(--green); }
    .pill-bad { background: rgba(182, 59, 46, 0.13); color: var(--red); }
    .pill-warn { background: rgba(182, 106, 0, 0.13); color: var(--amber); }
    .pill-neutral { background: rgba(35, 92, 143, 0.12); color: var(--blue); }

    .module-on { color: var(--green); font-weight: 700; }
    .module-off { color: var(--red); }
    .module-kill { color: #8b8b7f; text-decoration: line-through; }
    .gate-pass { color: var(--green); font-weight: 800; }
    .gate-fail { color: var(--amber); font-weight: 800; }

    .section-note {
        color: var(--muted);
        margin-top: -0.4rem;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 0.55rem 1rem;
        background: rgba(255, 252, 245, 0.55);
    }

    @media (max-width: 900px) {
        .hero { padding: 1.1rem; border-radius: 22px; }
        .card-value { font-size: 1.4rem; }
    }
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


def _start_bot():
    if _is_running()[0]:
        return
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    proc = subprocess.Popen(
        [str(VENV_PYTHON), str(PROJECT_ROOT / "launch.py")],
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


def _scalar(sql: str, default=0, params: tuple = ()):
    rows = _query_db(sql, params)
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


def _fmt_money(value: float | int | None) -> str:
    value = float(value or 0.0)
    return f"${value:,.2f}"


def _fmt_signed_money(value: float | int | None) -> str:
    value = float(value or 0.0)
    return f"{value:+,.2f}"


def _fmt_age(ts: str | None) -> str:
    if not ts:
        return "n/a"
    try:
        parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        minutes = max(0.0, (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds() / 60)
        if minutes < 1:
            return "<1m"
        if minutes < 90:
            return f"{minutes:.0f}m"
        return f"{minutes / 60:.1f}h"
    except Exception:
        return "n/a"


def _card(label: str, value: str, note: str = "", tone: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _pill(text: str, tone: str = "neutral") -> str:
    class_name = {
        "ok": "pill-ok",
        "bad": "pill-bad",
        "warn": "pill-warn",
        "neutral": "pill-neutral",
    }.get(tone, "pill-neutral")
    return f'<span class="status-pill {class_name}">{text}</span>'


def _phase_number(phase: str) -> int:
    try:
        return int(str(phase).split("_")[-1])
    except Exception:
        return 0


def _gate_trade_target(phase: str) -> int | None:
    return {
        "phase_0": 20,
        "phase_1": 40,
        "phase_2": 50,
    }.get(phase)


# ── Header + Controls ────────────────────────────────────────────────────

import altair as alt
import pandas as pd

running, pid = _is_running()
PHASE_CONFIGS, PHASE_ORDER, get_current_phase, get_phase_modules, get_gate_status = _get_phase_config()
current_phase = get_current_phase()
current_config = PHASE_CONFIGS[current_phase]
current_modules = get_phase_modules(current_phase)

try:
    from weatheredge.phase_config import get_phase_activated_at
    phase_activated_at = get_phase_activated_at()
except Exception:
    phase_activated_at = None

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">WeatherEdge</div>
        <p class="hero-subtitle">{current_config['name']} · {current_config['description']}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

control_left, control_mid, control_start, control_stop, control_refresh = st.columns([2.4, 1.8, 1, 1, 1])
with control_left:
    st.markdown(
        _pill(f"Bot running · PID {pid}", "ok") if running else _pill("Bot stopped", "bad"),
        unsafe_allow_html=True,
    )
with control_mid:
    st.markdown(
        _pill(f"{current_phase.replace('_', ' ').title()}", "neutral"),
        unsafe_allow_html=True,
    )
with control_start:
    if st.button("Start", use_container_width=True, type="primary", disabled=running):
        _start_bot()
        time.sleep(2)
        st.rerun()
with control_stop:
    if st.button("Stop", use_container_width=True, type="secondary", disabled=not running):
        _stop_bot()
        time.sleep(1)
        st.rerun()
with control_refresh:
    if st.button("Refresh", use_container_width=True):
        st.rerun()

# ── Sidebar ──────────────────────────────────────────────────────────────

st.sidebar.title("WeatherEdge")
st.sidebar.caption("Solo i controlli necessari. Il resto sta in Advanced.")
st.sidebar.divider()
st.sidebar.markdown("### Phase")

for phase_key in PHASE_ORDER:
    config = PHASE_CONFIGS[phase_key]
    is_active = phase_key == current_phase
    title = f"{phase_key.replace('_', ' ').title()} · {'active' if is_active else 'available'}"
    with st.sidebar.expander(title, expanded=is_active):
        st.caption(config["description"])
        st.markdown(f"**Gate:** {config.get('gate', 'N/A')}")
        if is_active:
            st.info("Fase attiva")
        elif st.button(f"Switch to {phase_key}", key=f"activate_{phase_key}", use_container_width=True):
            from weatheredge.phase_config import activate_phase

            activate_phase(phase_key)
            st.rerun()

st.sidebar.divider()
st.sidebar.markdown("### Refresh")
auto = st.sidebar.toggle("Auto-refresh", value=True)
interval = st.sidebar.slider("Seconds", 5, 60, 15)

# ── Data Snapshot ────────────────────────────────────────────────────────

_CLOSED = "('won','lost','edge_stop','capital_protect','time_decay','ev_based','bucket_invalidated','abandoned')"
_GATE_CLOSED = "('won','lost','edge_stop','capital_protect','time_decay','ev_based','bucket_invalidated')"

total_trades = int(_scalar("SELECT COUNT(*) FROM trades"))
open_trades = int(_scalar("SELECT COUNT(*) FROM trades WHERE status='open'"))
open_cost = float(_scalar("SELECT COALESCE(SUM(cost), 0) FROM trades WHERE status='open'"))
won = int(_scalar("SELECT COUNT(*) FROM trades WHERE status='won'"))
lost = int(_scalar("SELECT COUNT(*) FROM trades WHERE status='lost'"))
resolved = int(_scalar(f"SELECT COUNT(*) FROM trades WHERE status IN {_CLOSED}"))
wins_all = int(_scalar(f"SELECT COUNT(*) FROM trades WHERE status IN {_CLOSED} AND pnl > 0"))
total_pnl = float(_scalar(f"SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status IN {_CLOSED}"))
forecasts = int(_scalar("SELECT COUNT(*) FROM forecasts"))
markets = int(_scalar("SELECT COUNT(DISTINCT id) FROM markets"))
signals = int(_scalar("SELECT COUNT(*) FROM signals"))
linked_signals = int(_scalar("SELECT COUNT(*) FROM signals WHERE trade_id IS NOT NULL"))
leakage_total = int(_scalar("SELECT COUNT(*) FROM leakage_violations"))
win_rate = (wins_all / resolved * 100) if resolved > 0 else 0.0

phase_filter = ""
phase_params: tuple = ()
if phase_activated_at:
    phase_filter = " AND julianday(created_at) >= julianday(?)"
    phase_params = (phase_activated_at,)
phase_closed = int(_scalar(
    f"SELECT COUNT(*) FROM trades WHERE mode='paper' AND status IN {_GATE_CLOSED}{phase_filter}",
    params=phase_params,
))
phase_leakage = int(_scalar(
    "SELECT COUNT(*) FROM leakage_violations"
    + (" WHERE julianday(violated_at) >= julianday(?)" if phase_activated_at else ""),
    params=(phase_activated_at,) if phase_activated_at else (),
))
gate_target = _gate_trade_target(current_phase)
gate_progress = min(1.0, phase_closed / gate_target) if gate_target else 0.0
gate_result = get_gate_status(current_phase)
gate_passed = gate_result.get("passed")

last_forecast_ts = _scalar("SELECT MAX(created_at) FROM forecasts", default=None)
last_market_ts = _scalar("SELECT MAX(timestamp) FROM market_prices", default=None)
last_signal_ts = _scalar("SELECT MAX(created_at) FROM signals", default=None)
last_trade_ts = _scalar("SELECT MAX(created_at) FROM trades", default=None)

unrealized_pnl = 0.0
stale_open_positions = 0
_unrealized_rows = _query_db("""
    SELECT t.side, t.amount, t.cost, mp.yes_price
    FROM trades t
    LEFT JOIN (
        SELECT market_id, yes_price,
               ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC, id DESC) as rn
        FROM market_prices
        WHERE yes_price IS NOT NULL
    ) mp ON t.market_id = mp.market_id AND mp.rn = 1
    WHERE t.status = 'open'
""")
for r in _unrealized_rows:
    if r["yes_price"] is None:
        stale_open_positions += 1
        continue
    if r["side"] == "buy_yes":
        unrealized_pnl += (r["yes_price"] * r["amount"]) - r["cost"]
    else:
        unrealized_pnl += ((1.0 - r["yes_price"]) * r["amount"]) - r["cost"]

brier = 0.0
brier_rows = _query_db(
    "SELECT model_probability, status, side FROM trades WHERE status IN ('won','lost') AND model_probability IS NOT NULL"
)
if brier_rows:
    brier_sum = 0.0
    for r in brier_rows:
        if (r["side"] == "buy_yes" and r["status"] == "won") or (
            r["side"] == "buy_no" and r["status"] == "lost"
        ):
            outcome_yes = 1.0
        else:
            outcome_yes = 0.0
        brier_sum += (r["model_probability"] - outcome_yes) ** 2
    brier = brier_sum / len(brier_rows)

_initial_bankroll = 100.0
try:
    from weatheredge.config import Settings as _Settings

    _initial_bankroll = float(_Settings().initial_bankroll)
except Exception:
    pass
total_equity = _initial_bankroll + total_pnl + unrealized_pnl
equity_delta = total_equity - _initial_bankroll

health_items = []
if not running:
    health_items.append(("Bot fermo", "bad"))
if phase_leakage > 0:
    health_items.append((f"{phase_leakage} leakage fase", "bad"))
if stale_open_positions > 0:
    health_items.append((f"{stale_open_positions} posizioni stale", "warn"))
if gate_target and phase_closed < gate_target:
    health_items.append((f"{gate_target - phase_closed} trade al gate", "warn"))
if not health_items:
    health_items.append(("Operativo pulito", "ok"))

# ── Main Summary ────────────────────────────────────────────────────────

summary_cols = st.columns([1.6, 1, 1, 1, 1])
with summary_cols[0]:
    _card(
        "Total equity",
        _fmt_money(total_equity),
        f"{_fmt_signed_money(equity_delta)} vs bankroll · realized {_fmt_money(total_pnl)}",
        "ok" if equity_delta >= 0 else "bad",
    )
with summary_cols[1]:
    _card("Open risk", _fmt_money(open_cost), f"{open_trades} open · MTM {_fmt_signed_money(unrealized_pnl)}")
with summary_cols[2]:
    _card("Closed", str(resolved), f"{wins_all} profitable · win {win_rate:.1f}%")
with summary_cols[3]:
    brier_text = f"{brier:.4f}" if brier_rows else "n/a"
    _card("Brier", brier_text, f"{len(brier_rows)} market-resolved rows")
with summary_cols[4]:
    _card("Freshness", _fmt_age(last_market_ts), f"market · forecast {_fmt_age(last_forecast_ts)}")

st.markdown(
    " ".join(_pill(text, tone) for text, tone in health_items),
    unsafe_allow_html=True,
)

gate_col, freshness_col = st.columns([1.2, 1])
with gate_col:
    st.markdown("### Phase gate")
    st.caption(current_config.get("gate", "No gate configured"))
    if gate_target:
        st.progress(gate_progress)
        st.caption(f"{phase_closed}/{gate_target} valid closed paper trades since activation")
    elif gate_passed is None:
        st.caption("No numeric trade target for this phase.")
    if gate_passed is True:
        st.markdown('<span class="gate-pass">PASS</span>', unsafe_allow_html=True)
    elif gate_passed is False:
        st.markdown('<span class="gate-fail">FAIL</span>', unsafe_allow_html=True)
        if gate_result.get("reason"):
            st.caption(str(gate_result["reason"]))
with freshness_col:
    st.markdown("### Data heartbeat")
    hb1, hb2, hb3 = st.columns(3)
    hb1.metric("Market", _fmt_age(last_market_ts))
    hb2.metric("Forecast", _fmt_age(last_forecast_ts))
    hb3.metric("Signal", _fmt_age(last_signal_ts))
    st.caption(f"Last trade: {_fmt_age(last_trade_ts)} ago · DB: {DB_PATH.name}")

overview_tab, positions_tab, performance_tab, system_tab = st.tabs(
    ["Overview", "Positions", "Performance", "Advanced"]
)

with overview_tab:
    st.subheader("Equity curve")
    st.markdown('<p class="section-note">Total equity includes mark-to-market; realized equity shows closed P&L only.</p>', unsafe_allow_html=True)

    trades_all_q = (
        "SELECT id, market_id, side, amount, cost, status, pnl, "
        "created_at, resolved_at FROM trades"
    )
    prices_q = (
        "SELECT market_id, yes_price, timestamp FROM market_prices "
        "WHERE yes_price IS NOT NULL ORDER BY timestamp"
    )
    if DB_PATH.exists():
        _eq_conn = sqlite3.connect(str(DB_PATH), timeout=5)
        try:
            trades_df = pd.read_sql_query(trades_all_q, _eq_conn)
            prices_df = pd.read_sql_query(prices_q, _eq_conn)
        finally:
            _eq_conn.close()
    else:
        trades_df = pd.DataFrame()
        prices_df = pd.DataFrame()

    if trades_df.empty:
        st.info("Nessun trade nel DB. La curva appare dopo il primo trade.")
    else:
        trades_df["created_at"] = pd.to_datetime(trades_df["created_at"], utc=True, errors="coerce")
        trades_df["resolved_at"] = pd.to_datetime(trades_df["resolved_at"], utc=True, errors="coerce")
        trades_df["pnl"] = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0)
        if not prices_df.empty:
            prices_df["timestamp"] = pd.to_datetime(prices_df["timestamp"], utc=True, errors="coerce")
            prices_df = prices_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        closed_set = {"won", "lost", "edge_stop", "capital_protect", "time_decay", "ev_based", "bucket_invalidated", "abandoned"}
        start_ts = trades_df["created_at"].min()
        if pd.isna(start_ts):
            st.warning("Nessun trade con timestamp valido.")
        else:
            hours = pd.date_range(start_ts.floor("h"), pd.Timestamp.now(tz="UTC").ceil("h"), freq="h")
            prices_by_market: dict[str, pd.DataFrame] = {}
            if not prices_df.empty:
                for mid, grp in prices_df.groupby("market_id"):
                    prices_by_market[str(mid)] = grp[["timestamp", "yes_price"]].reset_index(drop=True)

            def _yes_price_at(mid: str, tick: pd.Timestamp) -> float | None:
                grp = prices_by_market.get(str(mid))
                if grp is None or grp.empty:
                    return None
                idx = grp["timestamp"].searchsorted(tick, side="right") - 1
                if idx < 0:
                    return None
                return float(grp.iloc[idx]["yes_price"])

            records = []
            for tick in hours:
                closed_mask = (
                    trades_df["resolved_at"].notna()
                    & (trades_df["resolved_at"] <= tick)
                    & (trades_df["status"].isin(closed_set))
                )
                realized_pnl = float(trades_df.loc[closed_mask, "pnl"].sum())
                open_mask = (
                    trades_df["created_at"].notna()
                    & (trades_df["created_at"] <= tick)
                    & (trades_df["resolved_at"].isna() | (trades_df["resolved_at"] > tick))
                )
                unrealized = 0.0
                for _, t in trades_df.loc[open_mask].iterrows():
                    yp = _yes_price_at(t["market_id"], tick)
                    if yp is None:
                        entry_yes = (
                            (t["cost"] / t["amount"]) if t["side"] == "buy_yes"
                            else (1.0 - t["cost"] / t["amount"])
                        ) if t["amount"] else 0.0
                        yp = entry_yes
                    value = (1.0 - yp) * t["amount"] if t["side"] == "buy_no" else yp * t["amount"]
                    unrealized += value - t["cost"]
                records.append({
                    "timestamp": tick,
                    "realized_equity": _initial_bankroll + realized_pnl,
                    "total_equity": _initial_bankroll + realized_pnl + unrealized,
                    "unrealized": unrealized,
                    "realized_pnl": realized_pnl,
                })

            equity_ts = pd.DataFrame(records)
            min_day = equity_ts["timestamp"].dt.date.min()
            max_day = equity_ts["timestamp"].dt.date.max()
            eq_col1, eq_col2 = st.columns([2, 1])
            with eq_col1:
                date_range = st.date_input(
                    "Date range",
                    value=(min_day, max_day),
                    min_value=min_day,
                    max_value=max_day,
                    key="equity_date_range",
                )
            with eq_col2:
                show_drawdown = st.toggle("Show drawdown", value=False, key="equity_show_dd")

            if isinstance(date_range, tuple) and len(date_range) == 2:
                d_from, d_to = date_range
            else:
                d_from = d_to = date_range if not isinstance(date_range, tuple) else min_day
            plot_df = equity_ts.loc[
                (equity_ts["timestamp"].dt.date >= d_from)
                & (equity_ts["timestamp"].dt.date <= d_to)
            ].copy().reset_index(drop=True)

            if plot_df.empty:
                st.warning("Nessun dato nell'intervallo selezionato.")
            else:
                long_df = pd.concat([
                    plot_df[["timestamp", "total_equity"]].rename(columns={"total_equity": "equity"}).assign(series="Total"),
                    plot_df[["timestamp", "realized_equity"]].rename(columns={"realized_equity": "equity"}).assign(series="Realized"),
                ], ignore_index=True)

                bankroll_rule = alt.Chart(pd.DataFrame({"y": [_initial_bankroll]})).mark_rule(
                    color="#7b715f", strokeDash=[3, 4]
                ).encode(y=alt.Y("y:Q", title="Equity ($)"))
                total_area = alt.Chart(plot_df).mark_area(
                    color="#0c7a43", opacity=0.14,
                ).encode(
                    x="timestamp:T",
                    y=alt.Y("total_equity:Q", scale=alt.Scale(zero=False)),
                    y2=alt.Y2(datum=_initial_bankroll),
                )
                line = alt.Chart(long_df).mark_line(strokeWidth=3).encode(
                    x=alt.X("timestamp:T", title="UTC"),
                    y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        "series:N",
                        scale=alt.Scale(domain=["Total", "Realized"], range=["#0c7a43", "#235c8f"]),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Time"),
                        alt.Tooltip("series:N", title="Series"),
                        alt.Tooltip("equity:Q", title="Equity", format="$.2f"),
                    ],
                )
                st.altair_chart((total_area + bankroll_rule + line).properties(height=370), use_container_width=True)

                if show_drawdown:
                    plot_df["running_max"] = plot_df["total_equity"].cummax()
                    plot_df["drawdown"] = plot_df["total_equity"] - plot_df["running_max"]
                    dd_chart = alt.Chart(plot_df).mark_area(color="#b63b2e", opacity=0.35).encode(
                        x=alt.X("timestamp:T", title="UTC"),
                        y=alt.Y("drawdown:Q", title="Drawdown ($)"),
                        tooltip=[alt.Tooltip("timestamp:T"), alt.Tooltip("drawdown:Q", format="$.2f")],
                    ).properties(height=180)
                    st.altair_chart(dd_chart, use_container_width=True)

with positions_tab:
    st.subheader("Open positions")
    st.markdown('<p class="section-note">Only live exposure is shown here. Closed trades are below.</p>', unsafe_allow_html=True)
    open_rows = _query_db("""
        SELECT t.market_id, t.side, t.price as entry_yes_price, t.amount, t.cost,
               t.agent_name, t.platform, t.station, t.market_date, t.created_at,
               mp.yes_price as current_yes_price,
               CASE
                   WHEN mp.yes_price IS NULL THEN NULL
                   WHEN t.side = 'buy_yes' THEN ROUND((mp.yes_price * t.amount) - t.cost, 2)
                   ELSE ROUND(((1.0 - mp.yes_price) * t.amount) - t.cost, 2)
               END as unrealized_pnl,
               CASE
                   WHEN mp.yes_price IS NULL OR t.cost <= 0 THEN NULL
                   WHEN t.side = 'buy_yes' THEN ROUND(((mp.yes_price * t.amount) - t.cost) / t.cost * 100, 1)
                   ELSE ROUND((((1.0 - mp.yes_price) * t.amount) - t.cost) / t.cost * 100, 1)
               END as pnl_pct
        FROM trades t
        LEFT JOIN (
            SELECT market_id, yes_price,
                   ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC, id DESC) as rn
            FROM market_prices
            WHERE yes_price IS NOT NULL
        ) mp ON t.market_id = mp.market_id AND mp.rn = 1
        WHERE t.status = 'open'
        ORDER BY t.created_at DESC LIMIT 30
    """)
    if open_rows:
        open_df = pd.DataFrame([dict(r) for r in open_rows])
        st.dataframe(open_df, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")

    st.subheader("Recent resolved")
    recent = _query_db(
        "SELECT market_id, side, price, amount, cost, pnl, status, platform, agent_name, resolved_at "
        "FROM trades WHERE status IN ('won','lost','edge_stop','capital_protect','time_decay','ev_based','bucket_invalidated') "
        "ORDER BY resolved_at DESC LIMIT 20"
    )
    if recent:
        recent_df = pd.DataFrame([dict(r) for r in recent])
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
    else:
        st.info("No resolved trades yet.")

with performance_tab:
    st.subheader("What is making or losing money")
    status_rows = _query_db(
        f"SELECT status, COUNT(*) as trades, COALESCE(SUM(pnl), 0) as pnl, COALESCE(SUM(cost), 0) as cost "
        f"FROM trades GROUP BY status ORDER BY pnl DESC"
    )
    perf_col1, perf_col2 = st.columns([1, 1])
    with perf_col1:
        if status_rows:
            status_df = pd.DataFrame([dict(r) for r in status_rows])
            status_chart = alt.Chart(status_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X("status:N", sort="-y", title="Status"),
                y=alt.Y("pnl:Q", title="P&L"),
                color=alt.condition("datum.pnl >= 0", alt.value("#0c7a43"), alt.value("#b63b2e")),
                tooltip=["status:N", "trades:Q", alt.Tooltip("pnl:Q", format="$.2f")],
            ).properties(height=280)
            st.altair_chart(status_chart, use_container_width=True)
        else:
            st.info("No trade status data.")
    with perf_col2:
        agent_rows = _query_db(
            "SELECT agent_name, COUNT(*) as trades, "
            "SUM(CASE WHEN status='won' THEN 1 ELSE 0 END) as wins, "
            "SUM(CASE WHEN status='lost' THEN 1 ELSE 0 END) as losses, "
            "COALESCE(SUM(pnl), 0) as total_pnl "
            "FROM trades WHERE status IN ('won','lost') "
            "GROUP BY agent_name ORDER BY total_pnl DESC"
        )
        if agent_rows:
            agent_data = []
            for a in agent_rows:
                d = dict(a)
                t = int(d["wins"] or 0) + int(d["losses"] or 0)
                d["win_rate"] = f"{d['wins'] / t * 100:.1f}%" if t > 0 else "-"
                d["total_pnl"] = f"${float(d['total_pnl'] or 0):.2f}"
                agent_data.append(d)
            st.dataframe(agent_data, use_container_width=True, hide_index=True)
        else:
            st.info("No agent performance data yet.")

    st.subheader("P&L by lead time")
    pnl_lt_rows = _query_db("""
        SELECT
            CASE
                WHEN t.market_date IS NOT NULL AND t.created_at IS NOT NULL THEN
                    CAST((julianday(t.market_date) - julianday(t.created_at)) * 24 AS INTEGER)
                ELSE -1
            END as lead_hours,
            t.pnl
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
            buckets[k] += float(r["pnl"] or 0)
            counts[k] += 1
        lead_df = pd.DataFrame([
            {"bucket": k, "pnl": v, "trades": counts[k]} for k, v in buckets.items()
        ])
        lead_chart = alt.Chart(lead_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("bucket:N", sort=["<12h", "12-24h", ">24h", "unknown"], title="Lead time"),
            y=alt.Y("pnl:Q", title="P&L"),
            color=alt.condition("datum.pnl >= 0", alt.value("#0c7a43"), alt.value("#b63b2e")),
            tooltip=["bucket:N", "trades:Q", alt.Tooltip("pnl:Q", format="$.2f")],
        ).properties(height=260)
        st.altair_chart(lead_chart, use_container_width=True)
    else:
        st.info("No resolved trades with lead-time data.")

with system_tab:
    st.subheader("Phase gates")
    gate_cols = st.columns(len(PHASE_ORDER))
    for i, phase_key in enumerate(PHASE_ORDER):
        config = PHASE_CONFIGS[phase_key]
        gate_result_i = get_gate_status(phase_key)
        with gate_cols[i]:
            st.caption(phase_key.replace("_", " ").title())
            st.markdown(f"**{config.get('gate', 'N/A')}**" if phase_key == current_phase else config.get("gate", "N/A"))
            passed = gate_result_i.get("passed")
            if passed is True:
                st.markdown('<span class="gate-pass">PASS</span>', unsafe_allow_html=True)
            elif passed is False:
                st.markdown('<span class="gate-fail">FAIL</span>', unsafe_allow_html=True)
                if gate_result_i.get("reason"):
                    st.caption(str(gate_result_i["reason"]))
            else:
                st.caption("N/A")

    st.subheader("Modules")
    sorted_mods = sorted(current_modules.items())
    third = len(sorted_mods) // 3 + 1
    mod_cols = st.columns(3)
    for col, chunk in zip(mod_cols, [sorted_mods[:third], sorted_mods[third:2 * third], sorted_mods[2 * third:]]):
        with col:
            for mod_name, enabled in chunk:
                if mod_name in STRUCTURAL_KILLS:
                    st.markdown(f'<span class="module-kill">killed · {mod_name}</span>', unsafe_allow_html=True)
                elif enabled:
                    st.markdown(f'<span class="module-on">on · {mod_name}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="module-off">off · {mod_name}</span>', unsafe_allow_html=True)

    st.subheader("Audit")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Signals", signals)
    a2.metric("Linked", linked_signals)
    a3.metric("Leakage total", leakage_total)
    a4.metric("Phase leakage", phase_leakage)

    st.subheader("Recent logs")
    log_lines = _tail_log(35)
    if log_lines:
        st.code("\n".join(log_lines), language="log")
    else:
        st.info("No log file found.")

# ── Auto-refresh ─────────────────────────────────────────────────────────

if auto:
    time.sleep(interval)
    st.rerun()
