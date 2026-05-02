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
AUDIT_TEMPORAL_FIX_CUTOFF = os.environ.get("WEATHEREDGE_TEMPORAL_FIX_CUTOFF", "2026-04-26T21:44:03Z")
DECISION_GRADE_CUTOFF = os.environ.get("WEATHEREDGE_DECISION_GRADE_CUTOFF", "2026-04-29T08:36:23Z")

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
    launch_pid = _find_running_launch_pid()
    if not PID_FILE.exists():
        if launch_pid is not None:
            PID_FILE.write_text(str(launch_pid))
            return True, launch_pid
        return False, None
    try:
        pid = int(PID_FILE.read_text().strip())
    except ValueError:
        PID_FILE.unlink(missing_ok=True)
        if launch_pid is not None:
            PID_FILE.write_text(str(launch_pid))
            return True, launch_pid
        return False, None
    try:
        os.kill(pid, 0)
        return True, pid
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        if launch_pid is not None:
            PID_FILE.write_text(str(launch_pid))
            return True, launch_pid
        return False, None
    except PermissionError:
        # Sandboxed dashboard sessions may be unable to signal a live bot
        # process. Treat this as running to avoid starting a duplicate.
        return True, pid


def _find_running_launch_pid() -> int | None:
    """Detect a WeatherEdge launch.py process even when the PID file is stale."""
    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None

    project = str(PROJECT_ROOT)
    current_pid = os.getpid()
    candidates: list[int] = []
    for line in result.stdout.splitlines():
        raw = line.strip()
        if not raw:
            continue
        pid_text, _, command = raw.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        if "launch.py" not in command:
            continue
        if "python" not in command:
            continue
        if "tmux new-session" in command or " rg " in command:
            continue
        if project not in command and "Weather FS" not in command and not _process_cwd_matches_project(pid):
            continue
        candidates.append(pid)
    return min(candidates) if candidates else None


def _process_cwd_matches_project(pid: int) -> bool:
    """Return True when a relative launch.py process is running from this project."""
    proc_cwd = Path("/proc") / str(pid) / "cwd"
    try:
        if proc_cwd.exists() and proc_cwd.resolve() == PROJECT_ROOT:
            return True
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except Exception:
        return False
    return result.returncode == 0 and str(PROJECT_ROOT) in result.stdout


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
    except PermissionError:
        return
    PID_FILE.unlink(missing_ok=True)


def _query_db(sql: str, params: tuple = ()) -> list:
    if not DB_PATH.exists():
        return []
    conn = None
    try:
        conn = _connect_db(readonly=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        return rows
    except Exception:
        return []
    finally:
        if conn is not None:
            conn.close()


def _connect_db(*, readonly: bool = True) -> sqlite3.Connection:
    """Open the dashboard DB defensively.

    The dashboard is read-heavy and must never block trading writes. Read-only
    URI mode plus in-memory temp storage avoids SQLite temp-file failures when
    charts sort larger result sets.
    """
    if readonly:
        uri = f"{DB_PATH.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, timeout=10, uri=True)
        conn.execute("PRAGMA query_only=ON")
    else:
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


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


def _parse_utc_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _max_timestamp_for_sql(*values: object) -> str | None:
    parsed = [ts for ts in (_parse_utc_datetime(value) for value in values) if ts is not None]
    return max(parsed).isoformat() if parsed else None


def _first_available(candidates: tuple[str, ...], choices: list[str]) -> list[str]:
    for candidate in candidates:
        if candidate in choices:
            return [candidate]
    return choices


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


def _station_set(cities: list[str], city_to_station: dict[str, str]) -> set[str]:
    stations: set[str] = set()
    for city in cities:
        raw = str(city or "").strip()
        if not raw:
            continue
        upper = raw.upper()
        station = city_to_station.get(raw.lower())
        if station:
            stations.add(station)
            continue
        if len(upper) == 4 and upper[0].isalpha():
            stations.add(upper)
    return stations


def _shadow_station_groups() -> tuple[set[str], set[str], set[str]]:
    try:
        from weatheredge.config import Settings
        from weatheredge.data.market_parser import CITY_TO_STATION

        settings = Settings()
        core = _station_set(settings.cities_list, CITY_TO_STATION)
        extra = _station_set(settings.shadow_twin_extended_cities_list, CITY_TO_STATION) - core
        broad = _station_set(settings.shadow_twin_broad_cities_list, CITY_TO_STATION) - core - extra
        return core, extra, broad
    except Exception:
        return set(), set(), set()


def _strategy_class_case(trade_alias: str = "t", market_alias: str = "m") -> str:
    """Return SQL that classifies trades into coarse audit strategy buckets."""
    return f"""
        CASE
            WHEN {trade_alias}.side='buy_no'
                 AND {market_alias}.threshold_low IS NOT NULL
                 AND {market_alias}.threshold_high IS NOT NULL
                 AND {trade_alias}.raw_model_prob IS NOT NULL
                 AND {trade_alias}.raw_model_prob < 0.10
                 AND {trade_alias}.edge_at_entry >= 0.10
                THEN 'floor_bound_buy_no'
            WHEN {trade_alias}.side='buy_no'
                 AND {market_alias}.threshold_low IS NOT NULL
                 AND {market_alias}.threshold_high IS NOT NULL
                THEN 'other_bucket_buy_no'
            WHEN {trade_alias}.side='buy_no'
                THEN 'other_buy_no'
            WHEN {trade_alias}.side='buy_yes'
                 AND {market_alias}.threshold_low IS NOT NULL
                 AND {market_alias}.threshold_high IS NOT NULL
                THEN 'bucket_buy_yes'
            ELSE 'other_buy_yes'
        END
    """


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
_DECISION_GRADE_STATUSES = "('open','won','lost','edge_stop','capital_protect','time_decay','ev_based','bucket_invalidated','abandoned')"
_SETTLED = "('won','lost')"
_EXITED = "('edge_stop','capital_protect','time_decay','ev_based','bucket_invalidated','abandoned')"

total_trades = int(_scalar("SELECT COUNT(*) FROM trades"))
open_trades = int(_scalar("SELECT COUNT(*) FROM trades WHERE status='open'"))
open_cost = float(_scalar("SELECT COALESCE(SUM(cost), 0) FROM trades WHERE status='open'"))
won = int(_scalar("SELECT COUNT(*) FROM trades WHERE status='won'"))
lost = int(_scalar("SELECT COUNT(*) FROM trades WHERE status='lost'"))
resolved = int(_scalar(f"SELECT COUNT(*) FROM trades WHERE status IN {_CLOSED}"))
wins_all = int(_scalar(f"SELECT COUNT(*) FROM trades WHERE status IN {_CLOSED} AND pnl > 0"))
total_pnl = float(_scalar(f"SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status IN {_CLOSED}"))
settled = int(_scalar(f"SELECT COUNT(*) FROM trades WHERE status IN {_SETTLED}"))
settled_pnl = float(_scalar(f"SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status IN {_SETTLED}"))
exited = int(_scalar(f"SELECT COUNT(*) FROM trades WHERE status IN {_EXITED}"))
exited_pnl = float(_scalar(f"SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status IN {_EXITED}"))
forecasts = int(_scalar("SELECT COUNT(*) FROM forecasts"))
markets = int(_scalar("SELECT COUNT(DISTINCT id) FROM markets"))
signals = int(_scalar("SELECT COUNT(*) FROM signals"))
linked_signals = int(_scalar("SELECT COUNT(*) FROM signals WHERE trade_id IS NOT NULL"))
leakage_total = int(_scalar("SELECT COUNT(*) FROM leakage_violations"))
win_rate = (wins_all / resolved * 100) if resolved > 0 else 0.0

shadow_summary_rows = _query_db("""
    WITH latest_prices AS (
        SELECT market_id, yes_price
        FROM (
            SELECT market_id, yes_price,
                   ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC, id DESC) AS rn
            FROM market_prices
            WHERE yes_price IS NOT NULL
        )
        WHERE rn = 1
    )
    SELECT s.twin_name,
           COUNT(*) AS trades,
           SUM(CASE WHEN s.status='open' THEN 1 ELSE 0 END) AS open_trades,
           SUM(CASE WHEN s.status IN ('won','lost') THEN 1 ELSE 0 END) AS resolved,
           SUM(CASE WHEN s.status='won' THEN 1 ELSE 0 END) AS wins,
           SUM(CASE WHEN s.status='lost' THEN 1 ELSE 0 END) AS losses,
           COALESCE(SUM(CASE WHEN s.status IN ('won','lost') THEN s.pnl ELSE 0 END), 0) AS realized_pnl,
           COALESCE(SUM(
               CASE
                   WHEN s.status='open' AND lp.yes_price IS NOT NULL AND s.side='buy_yes'
                       THEN lp.yes_price * s.amount - s.cost
                   WHEN s.status='open' AND lp.yes_price IS NOT NULL
                       THEN (1.0 - lp.yes_price) * s.amount - s.cost
                   ELSE 0
               END
            ), 0) AS unrealized_pnl,
            SUM(CASE WHEN s.status='open' AND lp.yes_price IS NULL THEN 1 ELSE 0 END) AS stale_open,
            SUM(CASE
                    WHEN s.forecast_timestamp IS NOT NULL
                         AND s.market_price_timestamp IS NOT NULL
                         AND julianday(s.forecast_timestamp) > julianday(s.market_price_timestamp)
                    THEN 1 ELSE 0
                END) AS non_causal_rows,
             SUM(CASE WHEN julianday(s.created_at) >= julianday(?) THEN 1 ELSE 0 END) AS post_fix_rows,
             SUM(CASE
                     WHEN julianday(s.created_at) >= julianday(?)
                          AND s.forecast_timestamp IS NOT NULL
                          AND s.market_price_timestamp IS NOT NULL
                          AND julianday(s.forecast_timestamp) <= julianday(s.market_price_timestamp)
                     THEN 1 ELSE 0
                 END) AS clean_post_fix_rows,
             SUM(CASE
                     WHEN julianday(s.created_at) >= julianday(?)
                          AND s.forecast_timestamp IS NOT NULL
                          AND s.market_price_timestamp IS NOT NULL
                          AND julianday(s.forecast_timestamp) <= julianday(s.market_price_timestamp)
                     THEN 1 ELSE 0
                 END) AS decision_grade_clean_rows,
             SUM(CASE
                     WHEN julianday(s.created_at) >= julianday(?)
                          AND s.forecast_timestamp IS NOT NULL
                          AND s.market_price_timestamp IS NOT NULL
                          AND julianday(s.forecast_timestamp) <= julianday(s.market_price_timestamp)
                          AND s.entry_spread IS NOT NULL
                          AND s.entry_volume IS NOT NULL
                     THEN 1 ELSE 0
                 END) AS decision_grade_real_spread_rows,
             COALESCE(SUM(s.cost), 0) AS cost
      FROM shadow_twin_trades s
      LEFT JOIN latest_prices lp ON s.market_id = lp.market_id
      GROUP BY s.twin_name
""", params=(
    AUDIT_TEMPORAL_FIX_CUTOFF,
    AUDIT_TEMPORAL_FIX_CUTOFF,
    DECISION_GRADE_CUTOFF,
    DECISION_GRADE_CUTOFF,
))
shadow_summary = {str(r["twin_name"]): dict(r) for r in shadow_summary_rows}

shadow_basket_rows = _query_db("""
    WITH latest_prices AS (
        SELECT market_id, yes_price
        FROM (
            SELECT market_id, yes_price,
                   ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC, id DESC) AS rn
            FROM market_prices
            WHERE yes_price IS NOT NULL
        )
        WHERE rn = 1
    ),
    open_leg_mtm AS (
        SELECT l.basket_id,
               COALESCE(SUM(
                   CASE
                       WHEN lp.yes_price IS NOT NULL AND l.side='buy_yes'
                           THEN lp.yes_price * l.amount - l.cost
                       WHEN lp.yes_price IS NOT NULL
                           THEN (1.0 - lp.yes_price) * l.amount - l.cost
                       ELSE 0
                   END
               ), 0) AS mtm,
               SUM(CASE WHEN lp.yes_price IS NULL THEN 1 ELSE 0 END) AS stale_legs
        FROM shadow_basket_legs l
        LEFT JOIN latest_prices lp ON l.market_id = lp.market_id
        WHERE l.status='open'
        GROUP BY l.basket_id
    )
    SELECT b.policy_name,
           COUNT(*) AS baskets,
           COALESCE(SUM(b.leg_count), 0) AS legs,
           SUM(CASE WHEN b.status='open' THEN 1 ELSE 0 END) AS open_baskets,
           SUM(CASE WHEN b.status IN ('won','lost') THEN 1 ELSE 0 END) AS resolved,
           SUM(CASE WHEN b.status='won' THEN 1 ELSE 0 END) AS wins,
           SUM(CASE WHEN b.status='lost' THEN 1 ELSE 0 END) AS losses,
           COALESCE(SUM(CASE WHEN b.status IN ('won','lost') THEN b.pnl ELSE 0 END), 0) AS realized_pnl,
           COALESCE(SUM(CASE WHEN b.status='open' THEN o.mtm ELSE 0 END), 0) AS unrealized_pnl,
           COALESCE(SUM(CASE WHEN b.status='open' THEN o.stale_legs ELSE 0 END), 0) AS stale_legs,
           COALESCE(AVG(CASE WHEN b.cost > 0 THEN b.worst_case_payoff / b.cost ELSE NULL END), 0) AS avg_worst_roi,
           COALESCE(AVG(CASE WHEN b.cost > 0 THEN b.payoff_none / b.cost ELSE NULL END), 0) AS avg_none_roi,
           COALESCE(SUM(b.cost), 0) AS cost
      FROM shadow_basket_trades b
      LEFT JOIN open_leg_mtm o ON b.basket_id = o.basket_id
      GROUP BY b.policy_name
""")
shadow_basket_summary = {str(r["policy_name"]): dict(r) for r in shadow_basket_rows}

paper_variant_summary_rows = _query_db("""
    WITH latest_prices AS (
        SELECT market_id, yes_price
        FROM (
            SELECT market_id, yes_price,
                   ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC, id DESC) AS rn
            FROM market_prices
            WHERE yes_price IS NOT NULL
        )
        WHERE rn = 1
    )
    SELECT p.variant_name,
           COUNT(*) AS trades,
           SUM(CASE WHEN p.status='open' THEN 1 ELSE 0 END) AS open_trades,
           SUM(CASE WHEN p.status IN ('won','lost') THEN 1 ELSE 0 END) AS resolved,
           SUM(CASE WHEN p.status='won' THEN 1 ELSE 0 END) AS wins,
           SUM(CASE WHEN p.status='lost' THEN 1 ELSE 0 END) AS losses,
           COALESCE(SUM(CASE WHEN p.status IN ('won','lost') THEN p.pnl ELSE 0 END), 0) AS realized_pnl,
           COALESCE(SUM(
               CASE
                   WHEN p.status='open' AND lp.yes_price IS NOT NULL AND p.side='buy_yes'
                       THEN lp.yes_price * p.amount - p.cost
                   WHEN p.status='open' AND lp.yes_price IS NOT NULL
                       THEN (1.0 - lp.yes_price) * p.amount - p.cost
                   ELSE 0
               END
           ), 0) AS unrealized_pnl,
           COALESCE(SUM(p.cost), 0) AS cost
    FROM paper_variant_trades p
    LEFT JOIN latest_prices lp ON p.market_id = lp.market_id
    GROUP BY p.variant_name
""")
paper_variant_summary = {str(r["variant_name"]): dict(r) for r in paper_variant_summary_rows}

phase_gate_since = _max_timestamp_for_sql(phase_activated_at, DECISION_GRADE_CUTOFF)
phase_filter = ""
phase_params: tuple = ()
if phase_gate_since:
    phase_filter = " AND julianday(created_at) >= julianday(?)"
    phase_params = (phase_gate_since,)
phase_decision_grade_entries = int(_scalar(
    f"""
    SELECT COUNT(*)
    FROM trades
    WHERE mode='paper'
      AND status IN {_DECISION_GRADE_STATUSES}
      AND COALESCE(order_type, 'market')='market'
      {phase_filter}
    """,
    params=phase_params,
))
phase_closed = int(_scalar(
    f"""
    SELECT COUNT(*)
    FROM trades
    WHERE mode='paper'
      AND status IN {_GATE_CLOSED}
      AND COALESCE(order_type, 'market')='market'
      {phase_filter}
    """,
    params=phase_params,
))
phase_leakage = int(_scalar(
    "SELECT COUNT(*) FROM leakage_violations"
    + (" WHERE julianday(violated_at) >= julianday(?)" if phase_gate_since else ""),
    params=(phase_gate_since,) if phase_gate_since else (),
))
gate_target = _gate_trade_target(current_phase)
gate_progress = min(1.0, phase_decision_grade_entries / gate_target) if gate_target else 0.0
gate_result = get_gate_status(current_phase)
gate_passed = gate_result.get("passed")

last_forecast_ts = _scalar("SELECT MAX(created_at) FROM forecasts", default=None)
last_market_ts = _scalar("SELECT MAX(timestamp) FROM market_prices", default=None)
last_signal_ts = _scalar("SELECT MAX(created_at) FROM signals", default=None)
last_trade_ts = _scalar("SELECT MAX(created_at) FROM trades", default=None)

unrealized_pnl = 0.0
open_liquidation_value = 0.0
conservative_unrealized_pnl = 0.0
stale_open_positions = 0
_unrealized_rows = _query_db("""
    SELECT t.side, t.amount, t.cost, mp.yes_price, mp.best_bid, mp.best_ask
    FROM trades t
    LEFT JOIN (
        SELECT market_id, yes_price, best_bid, best_ask,
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
        liquidation_price = r["best_bid"] if r["best_bid"] is not None else r["yes_price"]
    else:
        unrealized_pnl += ((1.0 - r["yes_price"]) * r["amount"]) - r["cost"]
        liquidation_price = (1.0 - r["best_ask"]) if r["best_ask"] is not None else (1.0 - r["yes_price"])
    liquidation_price = min(1.0, max(0.0, float(liquidation_price)))
    liquidation_value = liquidation_price * float(r["amount"] or 0.0)
    open_liquidation_value += liquidation_value
    conservative_unrealized_pnl += liquidation_value - float(r["cost"] or 0.0)

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
cash_bankroll_est = _initial_bankroll + total_pnl - open_cost
conservative_equity = cash_bankroll_est + open_liquidation_value
conservative_delta = conservative_equity - _initial_bankroll

floor_probation = {
    "accepted_trades": 0,
    "open_trades": 0,
    "closed_trades": 0,
    "realized_pnl": 0.0,
    "unrealized_pnl": 0.0,
    "open_cost": 0.0,
    "total_pnl": 0.0,
}
floor_rows = _query_db(f"""
    WITH latest_prices AS (
        SELECT market_id, yes_price
        FROM (
            SELECT market_id, yes_price,
                   ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC, id DESC) AS rn
            FROM market_prices
            WHERE yes_price IS NOT NULL
        )
        WHERE rn = 1
    ), valued AS (
        SELECT {_strategy_class_case('t', 'm')} AS strategy_class,
               t.status, t.cost, t.amount, t.side, t.pnl,
               CASE
                   WHEN t.status='open' AND lp.yes_price IS NOT NULL AND t.side='buy_yes'
                       THEN lp.yes_price * t.amount - t.cost
                   WHEN t.status='open' AND lp.yes_price IS NOT NULL
                       THEN (1.0 - lp.yes_price) * t.amount - t.cost
                   ELSE 0
               END AS unrealized_pnl
        FROM trades t
        LEFT JOIN markets m ON m.id = t.market_id
        LEFT JOIN latest_prices lp ON lp.market_id = t.market_id
        WHERE t.mode='paper' AND t.status != 'cancelled'
    )
    SELECT COUNT(*) AS accepted_trades,
           SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) AS open_trades,
           SUM(CASE WHEN status!='open' THEN 1 ELSE 0 END) AS closed_trades,
           COALESCE(SUM(COALESCE(pnl, 0)), 0) AS realized_pnl,
           COALESCE(SUM(CASE WHEN status='open' THEN unrealized_pnl ELSE 0 END), 0) AS unrealized_pnl,
           COALESCE(SUM(CASE WHEN status='open' THEN cost ELSE 0 END), 0) AS open_cost,
           COALESCE(SUM(COALESCE(pnl, 0) + unrealized_pnl), 0) AS total_pnl
    FROM valued
    WHERE strategy_class='floor_bound_buy_no'
""")
if floor_rows:
    floor_probation.update(dict(floor_rows[0]))

floor_top_rows = _query_db(f"""
    SELECT t.market_id, COALESCE(t.pnl, 0) AS pnl
    FROM trades t
    LEFT JOIN markets m ON m.id = t.market_id
    WHERE t.mode='paper'
      AND t.status IN {_CLOSED}
      AND {_strategy_class_case('t', 'm')} = 'floor_bound_buy_no'
    ORDER BY pnl DESC
    LIMIT 5
""")
floor_top_pnls = [float(r["pnl"] or 0.0) for r in floor_top_rows]
floor_realized = float(floor_probation.get("realized_pnl") or 0.0)
floor_top1_share = (floor_top_pnls[0] / floor_realized * 100.0) if floor_realized and floor_top_pnls else 0.0
floor_top5_share = (sum(floor_top_pnls) / floor_realized * 100.0) if floor_realized and floor_top_pnls else 0.0
floor_realized_share = (floor_realized / total_pnl * 100.0) if total_pnl else 0.0
floor_open_risk_share = (float(floor_probation.get("open_cost") or 0.0) / open_cost * 100.0) if open_cost else 0.0

def _paper_variant_stats(variant_name: str) -> dict[str, float | int]:
    stats = paper_variant_summary.get(variant_name, {})
    trades = int(stats.get("trades") or 0)
    open_count = int(stats.get("open_trades") or 0)
    resolved_count = int(stats.get("resolved") or 0)
    wins_count = int(stats.get("wins") or 0)
    realized_value = float(stats.get("realized_pnl") or 0.0)
    unrealized_value = float(stats.get("unrealized_pnl") or 0.0)
    return {
        "trades": trades,
        "open": open_count,
        "resolved": resolved_count,
        "wins": wins_count,
        "realized": realized_value,
        "unrealized": unrealized_value,
        "total": realized_value + unrealized_value,
        "win_rate": (wins_count / resolved_count * 100.0) if resolved_count else 0.0,
    }


paper_variant_rows = [
    {
        "variant": "main_base",
        "liquidity_rule": "volume >= 2000",
        "entries": int(open_trades + resolved),
        "open": int(open_trades),
        "resolved": int(resolved),
        "realized_pnl": float(total_pnl),
        "unrealized_pnl": float(unrealized_pnl),
        "total_pnl": float(total_pnl + unrealized_pnl),
        "win_rate": (wins_all / resolved * 100.0) if resolved else 0.0,
    }
]
for _label, _twin_name in (
    ("main_adaptive_core", "main_adaptive_core"),
    ("main_adaptive_expanded", "main_adaptive_expanded"),
):
    _stats = _paper_variant_stats(_twin_name)
    paper_variant_rows.append({
        "variant": _label,
        "liquidity_rule": "adaptive volume",
        "entries": int(_stats["trades"]),
        "open": int(_stats["open"]),
        "resolved": int(_stats["resolved"]),
        "realized_pnl": float(_stats["realized"]),
        "unrealized_pnl": float(_stats["unrealized"]),
        "total_pnl": float(_stats["total"]),
        "win_rate": float(_stats["win_rate"]),
    })

health_items = []
if not running:
    health_items.append(("Bot fermo", "bad"))
if phase_leakage > 0:
    health_items.append((f"{phase_leakage} leakage fase", "bad"))
if stale_open_positions > 0:
    health_items.append((f"{stale_open_positions} posizioni stale", "warn"))
if floor_realized_share >= 50.0:
    health_items.append((f"floor NO {floor_realized_share:.0f}% P&L", "warn"))
if floor_open_risk_share >= 30.0:
    health_items.append((f"floor NO {floor_open_risk_share:.0f}% open risk", "warn"))
if gate_target and phase_decision_grade_entries < gate_target:
    health_items.append((f"{gate_target - phase_decision_grade_entries} trade al gate", "warn"))
if not health_items:
    health_items.append(("Operativo pulito", "ok"))

# ── Main Summary ────────────────────────────────────────────────────────

summary_cols = st.columns([1.6, 1, 1, 1, 1])
with summary_cols[0]:
    _card(
        "Conservative equity",
        _fmt_money(conservative_equity),
        f"{_fmt_signed_money(conservative_delta)} vs bankroll · mid {_fmt_money(total_equity)}",
        "ok" if conservative_delta >= 0 else "bad",
    )
with summary_cols[1]:
    _card(
        "Open risk",
        _fmt_money(open_cost),
        f"{open_trades} open · mid {_fmt_signed_money(unrealized_pnl)} · liquidation {_fmt_signed_money(conservative_unrealized_pnl)}",
    )
with summary_cols[2]:
    _card(
        "Closed",
        str(resolved),
        f"settled {settled} ({_fmt_signed_money(settled_pnl)}) · exited {exited} ({_fmt_signed_money(exited_pnl)})",
    )
with summary_cols[3]:
    brier_text = f"{brier:.4f}" if brier_rows else "n/a"
    _card("Brier", brier_text, f"{len(brier_rows)} market-resolved rows")
with summary_cols[4]:
    _card("Freshness", _fmt_age(last_market_ts), f"market · forecast {_fmt_age(last_forecast_ts)}")

st.markdown(
    " ".join(_pill(text, tone) for text, tone in health_items),
    unsafe_allow_html=True,
)

st.markdown("### Phase 1 probation")
probation_cols = st.columns(3)
with probation_cols[0]:
    _card(
        "floor_bound_buy_no",
        _fmt_signed_money(float(floor_probation.get("total_pnl") or 0.0)),
        f"realized {_fmt_signed_money(floor_realized)} · MTM {_fmt_signed_money(float(floor_probation.get('unrealized_pnl') or 0.0))} · "
        f"{int(floor_probation.get('closed_trades') or 0)} closed / {int(floor_probation.get('open_trades') or 0)} open",
        "warn",
    )
with probation_cols[1]:
    _card(
        "Concentration",
        f"{floor_realized_share:.0f}%",
        f"of realized P&L · {floor_open_risk_share:.0f}% of open risk",
        "warn" if floor_realized_share >= 50.0 or floor_open_risk_share >= 30.0 else "neutral",
    )
with probation_cols[2]:
    _card(
        "Top-trade dependency",
        f"{floor_top1_share:.0f}%",
        f"top 1 of floor realized · top 5 {floor_top5_share:.0f}%",
        "warn" if floor_top1_share >= 25.0 or floor_top5_share >= 60.0 else "neutral",
    )

st.markdown("### Paper variants")
variant_cols = st.columns(3)
for col, row, label in [
    (variant_cols[0], paper_variant_rows[0], "Main base"),
    (variant_cols[1], paper_variant_rows[1], "Main adaptive · core"),
    (variant_cols[2], paper_variant_rows[2], "Main adaptive · expanded"),
]:
    with col:
        _card(
            label,
            _fmt_signed_money(row["total_pnl"]),
            f"{row['entries']} entries · {row['open']} open · {row['resolved']} resolved · "
            f"realized {_fmt_signed_money(row['realized_pnl'])} · MTM {_fmt_signed_money(row['unrealized_pnl'])} · "
            f"win {float(row['win_rate']):.1f}% · {row['liquidity_rule']}",
            "ok" if float(row["total_pnl"]) >= 0 else "bad",
        )

st.markdown("### Shadow twins")
shadow_cols = st.columns(6)
for col, twin_name, label in [
    (shadow_cols[0], "floor_no_core", "Core · main cities"),
    (shadow_cols[1], "floor_no_expanded", "Expanded · main + extra"),
    (shadow_cols[2], "floor_no_broad", "Broad · all sensible"),
    (shadow_cols[3], "floor_no_core_adaptive_liquidity", "Adaptive liq · core"),
    (shadow_cols[4], "floor_no_expanded_adaptive_liquidity", "Adaptive liq · expanded"),
    (shadow_cols[5], "floor_no_broad_adaptive_liquidity", "Adaptive liq · broad"),
]:
    stats = shadow_summary.get(twin_name, {})
    trades = int(stats.get("trades") or 0)
    resolved_shadow = int(stats.get("resolved") or 0)
    wins_shadow = int(stats.get("wins") or 0)
    open_shadow = int(stats.get("open_trades") or 0)
    realized_shadow = float(stats.get("realized_pnl") or 0.0)
    unrealized_shadow = float(stats.get("unrealized_pnl") or 0.0)
    total_shadow = realized_shadow + unrealized_shadow
    stale_shadow = int(stats.get("stale_open") or 0)
    non_causal_shadow = int(stats.get("non_causal_rows") or 0)
    clean_post_fix_shadow = int(stats.get("clean_post_fix_rows") or 0)
    decision_grade_clean_shadow = int(stats.get("decision_grade_clean_rows") or 0)
    decision_grade_real_spread_shadow = int(stats.get("decision_grade_real_spread_rows") or 0)
    win_shadow = (wins_shadow / resolved_shadow * 100.0) if resolved_shadow else 0.0
    stale_note = f" · {stale_shadow} stale" if stale_shadow else ""
    causal_note = (
        f" · clean post-fix {clean_post_fix_shadow}/{trades}"
        f" · decision-grade {decision_grade_clean_shadow}"
        f" · real-spread {decision_grade_real_spread_shadow}"
        f" · non-causal {non_causal_shadow}"
    )
    with col:
        _card(
            label,
            _fmt_signed_money(total_shadow),
            f"{trades} entries · {open_shadow} open · {resolved_shadow} resolved · "
            f"realized {_fmt_signed_money(realized_shadow)} · MTM {_fmt_signed_money(unrealized_shadow)}{stale_note} · "
            f"win {win_shadow:.1f}%{causal_note}",
            "ok" if total_shadow >= 0 else "bad",
        )

st.markdown("### Shadow NO baskets")
basket_stats = shadow_basket_summary.get("expanded_no_basket_v1", {})
basket_count = int(basket_stats.get("baskets") or 0)
basket_open = int(basket_stats.get("open_baskets") or 0)
basket_resolved = int(basket_stats.get("resolved") or 0)
basket_wins = int(basket_stats.get("wins") or 0)
basket_legs = int(basket_stats.get("legs") or 0)
basket_realized = float(basket_stats.get("realized_pnl") or 0.0)
basket_unrealized = float(basket_stats.get("unrealized_pnl") or 0.0)
basket_total = basket_realized + basket_unrealized
basket_win = (basket_wins / basket_resolved * 100.0) if basket_resolved else 0.0
basket_avg_worst = float(basket_stats.get("avg_worst_roi") or 0.0) * 100.0
basket_avg_none = float(basket_stats.get("avg_none_roi") or 0.0) * 100.0
basket_stale = int(basket_stats.get("stale_legs") or 0)
basket_cols = st.columns(3)
with basket_cols[0]:
    _card(
        "Expanded basket v1",
        _fmt_signed_money(basket_total),
        f"{basket_count} baskets · {basket_legs} legs · {basket_open} open · {basket_resolved} resolved · "
        f"realized {_fmt_signed_money(basket_realized)} · MTM {_fmt_signed_money(basket_unrealized)}",
        "ok" if basket_total >= 0 else "bad",
    )
with basket_cols[1]:
    _card(
        "Basket quality",
        f"{basket_avg_worst:.1f}%",
        f"avg worst-case ROI · no-hit {basket_avg_none:.1f}% · stale legs {basket_stale}",
        "ok" if basket_avg_worst >= 5.0 else "warn",
    )
with basket_cols[2]:
    _card(
        "Resolved baskets",
        f"{basket_win:.1f}%",
        f"{basket_wins} wins / {basket_resolved} resolved",
        "ok" if basket_win >= 70.0 or basket_resolved == 0 else "warn",
    )

gate_col, freshness_col = st.columns([1.2, 1])
with gate_col:
    st.markdown("### Phase gate")
    st.caption(current_config.get("gate", "No gate configured"))
    if gate_target:
        st.progress(gate_progress)
        st.caption(
            f"{phase_decision_grade_entries}/{gate_target} decision-grade paper entries since reset · "
            f"{phase_closed} closed · cutoff {DECISION_GRADE_CUTOFF}"
        )
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

overview_tab, positions_tab, shadow_tab, performance_tab, audit_tab, system_tab = st.tabs(
    ["Overview", "Positions", "Shadow Details", "Performance", "Audit", "Advanced"]
)

with overview_tab:
    st.subheader("Equity curve")
    st.markdown('<p class="section-note">Total equity includes mark-to-market; realized equity shows closed P&L only.</p>', unsafe_allow_html=True)

    trades_all_q = (
        "SELECT id, market_id, side, amount, cost, status, pnl, "
        "created_at, resolved_at FROM trades"
    )
    trades_df = pd.DataFrame()
    prices_df = pd.DataFrame()
    if DB_PATH.exists():
        _eq_conn = None
        try:
            _eq_conn = _connect_db(readonly=True)
            trades_df = pd.read_sql_query(trades_all_q, _eq_conn)
        except Exception as exc:
            st.error(f"Dashboard DB read failed: {exc}")
        finally:
            if _eq_conn is not None:
                _eq_conn.close()

    if trades_df.empty:
        st.info("Nessun trade nel DB. La curva appare dopo il primo trade.")
    else:
        trades_df["created_at"] = pd.to_datetime(trades_df["created_at"], utc=True, errors="coerce")
        trades_df["resolved_at"] = pd.to_datetime(trades_df["resolved_at"], utc=True, errors="coerce")
        trades_df["pnl"] = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0)
        trades_df["amount"] = pd.to_numeric(trades_df["amount"], errors="coerce").fillna(0.0)
        trades_df["cost"] = pd.to_numeric(trades_df["cost"], errors="coerce").fillna(0.0)
        trade_market_ids = [
            str(mid)
            for mid in trades_df["market_id"].dropna().unique().tolist()
            if str(mid).strip()
        ]
        price_start_ts = trades_df["created_at"].dropna().min()
        if trade_market_ids and pd.notna(price_start_ts) and DB_PATH.exists():
            placeholders = ",".join("?" for _ in trade_market_ids)
            prices_q = (
                "SELECT market_id, yes_price, timestamp FROM market_prices "
                f"WHERE yes_price IS NOT NULL AND market_id IN ({placeholders}) "
                "AND timestamp >= ? ORDER BY market_id, timestamp"
            )
            params = [*trade_market_ids, price_start_ts.strftime("%Y-%m-%d %H:%M:%S")]
            _price_conn = None
            try:
                _price_conn = _connect_db(readonly=True)
                prices_df = pd.read_sql_query(prices_q, _price_conn, params=params)
            except Exception as exc:
                st.warning(f"MTM price history unavailable; showing realized-first equity. {exc}")
                prices_df = pd.DataFrame()
            finally:
                if _price_conn is not None:
                    _price_conn.close()
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
               mp.yes_price as current_yes_price, mp.best_bid, mp.best_ask,
               CASE
                   WHEN mp.yes_price IS NULL THEN NULL
                   WHEN t.side = 'buy_yes' THEN ROUND((mp.yes_price * t.amount) - t.cost, 2)
                   ELSE ROUND(((1.0 - mp.yes_price) * t.amount) - t.cost, 2)
               END as unrealized_pnl,
               CASE
                   WHEN mp.yes_price IS NULL THEN NULL
                   WHEN t.side = 'buy_yes' THEN ROUND((COALESCE(mp.best_bid, mp.yes_price) * t.amount) - t.cost, 2)
                   ELSE ROUND(((1.0 - COALESCE(mp.best_ask, mp.yes_price)) * t.amount) - t.cost, 2)
               END as liquidation_pnl,
               CASE
                   WHEN mp.yes_price IS NULL OR t.cost <= 0 THEN NULL
                   WHEN t.side = 'buy_yes' THEN ROUND(((mp.yes_price * t.amount) - t.cost) / t.cost * 100, 1)
                   ELSE ROUND((((1.0 - mp.yes_price) * t.amount) - t.cost) / t.cost * 100, 1)
               END as pnl_pct
        FROM trades t
        LEFT JOIN (
            SELECT market_id, yes_price, best_bid, best_ask,
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


with shadow_tab:
    st.subheader("Shadow trade details")
    st.markdown(
        '<p class="section-note">Isolated shadow simulations only. Rows here do not affect bankroll, risk limits, calibration, or phase gates.</p>',
        unsafe_allow_html=True,
    )
    shadow_detail_rows = _query_db("""
        WITH latest_prices AS (
            SELECT market_id, yes_price, timestamp
            FROM (
                SELECT market_id, yes_price, timestamp,
                       ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC, id DESC) AS rn
                FROM market_prices
                WHERE yes_price IS NOT NULL
            )
            WHERE rn = 1
        )
        SELECT s.id, s.universe_name AS twin_name, s.policy_name, s.opportunity_basket,
               s.market_id, s.platform, s.side, s.station, s.market_date,
               s.status, s.entry_yes_price, lp.yes_price AS current_yes_price,
                s.model_probability, s.raw_model_prob, s.net_edge_at_entry,
                s.amount, s.cost, s.bucket_low, s.bucket_high,
                s.entry_volume AS volume, s.entry_spread AS spread, s.liquidity_min_volume,
                s.forecast_timestamp, s.market_price_timestamp,
               CASE
                   WHEN s.forecast_timestamp IS NULL OR s.market_price_timestamp IS NULL THEN 'unknown'
                   WHEN julianday(s.forecast_timestamp) > julianday(s.market_price_timestamp) THEN 'non_causal'
                   ELSE 'clean'
               END AS causal_status,
                CASE
                    WHEN julianday(s.created_at) >= julianday(?) THEN 'post_temporal_fix'
                    ELSE 'pre_temporal_fix'
                END AS fix_version,
                CASE
                    WHEN julianday(s.created_at) >= julianday(?)
                         AND s.forecast_timestamp IS NOT NULL
                         AND s.market_price_timestamp IS NOT NULL
                         AND julianday(s.forecast_timestamp) <= julianday(s.market_price_timestamp)
                         AND s.entry_spread IS NOT NULL
                         AND s.entry_volume IS NOT NULL
                        THEN 'decision_grade_real_spread'
                    WHEN julianday(s.created_at) >= julianday(?)
                         AND s.forecast_timestamp IS NOT NULL
                         AND s.market_price_timestamp IS NOT NULL
                         AND julianday(s.forecast_timestamp) <= julianday(s.market_price_timestamp)
                        THEN 'decision_grade_clean_no_real_spread'
                    WHEN julianday(s.created_at) >= julianday(?)
                         AND s.forecast_timestamp IS NOT NULL
                         AND s.market_price_timestamp IS NOT NULL
                         AND julianday(s.forecast_timestamp) <= julianday(s.market_price_timestamp)
                        THEN 'post_temporal_fix_clean_pre_reset'
                    ELSE 'legacy_or_unclean'
                END AS evidence_scope,
                s.created_at, s.resolved_at, s.exit_reason, s.outcome,
                COALESCE(CASE WHEN s.status IN ('won','lost','exited') THEN s.pnl ELSE 0 END, 0) AS realized_pnl,
               CASE
                   WHEN s.status='open' AND lp.yes_price IS NOT NULL AND s.side='buy_yes'
                       THEN lp.yes_price * s.amount - s.cost
                   WHEN s.status='open' AND lp.yes_price IS NOT NULL
                       THEN (1.0 - lp.yes_price) * s.amount - s.cost
                   ELSE 0
               END AS unrealized_pnl,
               lp.timestamp AS current_price_ts
        FROM shadow_policy_trades s
        LEFT JOIN latest_prices lp ON s.market_id = lp.market_id
        ORDER BY s.created_at DESC, s.id DESC
    """, params=(
        AUDIT_TEMPORAL_FIX_CUTOFF,
        DECISION_GRADE_CUTOFF,
        DECISION_GRADE_CUTOFF,
        AUDIT_TEMPORAL_FIX_CUTOFF,
    ))
    if shadow_detail_rows:
        shadow_details_df = pd.DataFrame([dict(r) for r in shadow_detail_rows])
        core_stations, extra_stations, broad_stations = _shadow_station_groups()
        adaptive_floor = 250.0
        adaptive_multiple = 100.0
        try:
            from weatheredge.config import Settings

            _settings = Settings()
            adaptive_floor = float(_settings.shadow_twin_adaptive_liquidity_min_volume)
            adaptive_multiple = float(_settings.shadow_twin_adaptive_liquidity_volume_multiple)
        except Exception:
            pass

        def _universe(station: object) -> str:
            code = str(station or "").upper()
            if code in core_stations:
                return "core_main_city"
            if code in extra_stations:
                return "extra_city"
            if code in broad_stations:
                return "broad_city"
            return "other"

        shadow_details_df["universe"] = shadow_details_df["station"].map(_universe)
        shadow_details_df["temp_bucket"] = shadow_details_df.apply(
            lambda r: (
                f"{float(r['bucket_low']):.0f}-{float(r['bucket_high']):.0f}"
                if pd.notna(r["bucket_low"]) and pd.notna(r["bucket_high"])
                else "n/a"
            ),
            axis=1,
        )
        shadow_details_df["price_move_yes"] = shadow_details_df["current_yes_price"] - shadow_details_df["entry_yes_price"]
        shadow_details_df["total_pnl"] = shadow_details_df["realized_pnl"] + shadow_details_df["unrealized_pnl"]
        shadow_details_df["open_count"] = shadow_details_df["status"].eq("open").astype(int)
        shadow_details_df["closed_count"] = shadow_details_df["status"].isin(["won", "lost", "exited"]).astype(int)
        shadow_details_df["win_count"] = shadow_details_df["status"].eq("won").astype(int)
        shadow_details_df["loss_count"] = shadow_details_df["status"].eq("lost").astype(int)
        shadow_details_df["created_at_utc"] = pd.to_datetime(shadow_details_df["created_at"], utc=True, errors="coerce")
        adaptive_mask = (
            shadow_details_df["opportunity_basket"].eq("floor_bound_bucket_no_adaptive_liquidity")
            | shadow_details_df["twin_name"].astype(str).str.contains("adaptive_liquidity", regex=False)
        )
        shadow_details_df["adaptive_min_volume"] = shadow_details_df["liquidity_min_volume"]
        missing_adaptive_threshold = adaptive_mask & shadow_details_df["adaptive_min_volume"].isna()
        shadow_details_df.loc[missing_adaptive_threshold, "adaptive_min_volume"] = shadow_details_df.loc[missing_adaptive_threshold, "cost"].map(
            lambda cost: max(adaptive_floor, float(cost or 0.0) * adaptive_multiple)
        )

        evidence_order = {
            "decision_grade_real_spread": 0,
            "decision_grade_clean_no_real_spread": 1,
            "post_temporal_fix_clean_pre_reset": 2,
            "legacy_or_unclean": 3,
        }
        temporal_cutoff_ts = pd.to_datetime(AUDIT_TEMPORAL_FIX_CUTOFF, utc=True, errors="coerce")
        decision_cutoff_ts = pd.to_datetime(DECISION_GRADE_CUTOFF, utc=True, errors="coerce")
        clean_mask = shadow_details_df["causal_status"].eq("clean")
        post_temporal_mask = clean_mask & shadow_details_df["created_at_utc"].ge(temporal_cutoff_ts)
        decision_grade_mask = clean_mask & shadow_details_df["created_at_utc"].ge(decision_cutoff_ts)
        real_spread_mask = decision_grade_mask & shadow_details_df["spread"].notna() & shadow_details_df["volume"].notna()

        def _evidence_row(label: str, frame: pd.DataFrame) -> dict:
            return {
                "evidence_scope": label,
                "policy_rows": len(frame),
                "open": int(frame["open_count"].sum()),
                "closed": int(frame["closed_count"].sum()),
                "wins": int(frame["win_count"].sum()),
                "losses": int(frame["loss_count"].sum()),
                "realized_pnl": frame["realized_pnl"].sum(),
                "unrealized_pnl": frame["unrealized_pnl"].sum(),
                "total_pnl": frame["total_pnl"].sum(),
                "avg_volume": frame["volume"].mean(),
                "avg_spread": frame["spread"].mean(),
            }

        evidence_summary = pd.DataFrame([
            _evidence_row("all_policy_rows", shadow_details_df),
            _evidence_row("post_temporal_fix_clean", shadow_details_df.loc[post_temporal_mask]),
            _evidence_row("decision_grade_clean", shadow_details_df.loc[decision_grade_mask]),
            _evidence_row("decision_grade_real_spread", shadow_details_df.loc[real_spread_mask]),
            _evidence_row("legacy_or_unclean", shadow_details_df.loc[~post_temporal_mask]),
        ])
        evidence_numeric_cols = [
            "realized_pnl", "unrealized_pnl", "total_pnl", "avg_volume", "avg_spread",
        ]
        evidence_summary[evidence_numeric_cols] = evidence_summary[evidence_numeric_cols].round(4)
        st.markdown("#### Evidence tiers")
        st.caption(
            f"Decision-grade cutoff {DECISION_GRADE_CUTOFF} · temporal fix cutoff {AUDIT_TEMPORAL_FIX_CUTOFF}. "
            "Real-spread rows require non-null spread and volume captured at entry."
        )
        st.dataframe(evidence_summary, use_container_width=True, hide_index=True)

        detail_filter_cols = st.columns(7)
        twin_choices = sorted(shadow_details_df["twin_name"].dropna().unique().tolist())
        policy_choices = sorted(shadow_details_df["policy_name"].dropna().unique().tolist())
        universe_choices = sorted(shadow_details_df["universe"].dropna().unique().tolist())
        status_choices = sorted(shadow_details_df["status"].dropna().unique().tolist())
        causal_choices = sorted(shadow_details_df["causal_status"].dropna().unique().tolist())
        fix_choices = sorted(shadow_details_df["fix_version"].dropna().unique().tolist())
        evidence_choices = sorted(
            shadow_details_df["evidence_scope"].dropna().unique().tolist(),
            key=lambda item: evidence_order.get(str(item), 99),
        )
        default_causal = _first_available(("clean",), causal_choices)
        default_fix = _first_available(("post_temporal_fix",), fix_choices)
        default_evidence = _first_available(
            (
                "decision_grade_real_spread",
                "decision_grade_clean_no_real_spread",
                "post_temporal_fix_clean_pre_reset",
            ),
            evidence_choices,
        )
        with detail_filter_cols[0]:
            selected_twins = st.multiselect("Twin", twin_choices, default=twin_choices)
        with detail_filter_cols[1]:
            selected_policies = st.multiselect("Policy", policy_choices, default=policy_choices)
        with detail_filter_cols[2]:
            selected_universes = st.multiselect("Universe", universe_choices, default=universe_choices)
        with detail_filter_cols[3]:
            selected_statuses = st.multiselect("Status", status_choices, default=status_choices)
        with detail_filter_cols[4]:
            selected_causal = st.multiselect("Causal", causal_choices, default=default_causal)
        with detail_filter_cols[5]:
            selected_fix = st.multiselect("Fix window", fix_choices, default=default_fix)
        with detail_filter_cols[6]:
            selected_evidence = st.multiselect("Evidence", evidence_choices, default=default_evidence)

        filtered_shadow_df = shadow_details_df[
            shadow_details_df["twin_name"].isin(selected_twins)
            & shadow_details_df["policy_name"].isin(selected_policies)
            & shadow_details_df["universe"].isin(selected_universes)
            & shadow_details_df["status"].isin(selected_statuses)
            & shadow_details_df["causal_status"].isin(selected_causal)
            & shadow_details_df["fix_version"].isin(selected_fix)
            & shadow_details_df["evidence_scope"].isin(selected_evidence)
        ].copy()

        if filtered_shadow_df.empty:
            st.info("No shadow rows match the selected filters.")
        else:
            shadow_metric_cols = st.columns(4)
            shadow_metric_cols[0].metric("Rows", f"{len(filtered_shadow_df)}")
            shadow_metric_cols[1].metric("Realized", _fmt_signed_money(filtered_shadow_df["realized_pnl"].sum()))
            shadow_metric_cols[2].metric("MTM", _fmt_signed_money(filtered_shadow_df["unrealized_pnl"].sum()))
            shadow_metric_cols[3].metric("Total", _fmt_signed_money(filtered_shadow_df["total_pnl"].sum()))

            basket_summary = (
                filtered_shadow_df
                .groupby(["twin_name", "policy_name", "universe", "evidence_scope", "causal_status", "fix_version", "status"], dropna=False)
                .agg(
                    trades=("id", "count"),
                    realized_pnl=("realized_pnl", "sum"),
                    unrealized_pnl=("unrealized_pnl", "sum"),
                    total_pnl=("total_pnl", "sum"),
                    avg_entry_yes=("entry_yes_price", "mean"),
                    avg_current_yes=("current_yes_price", "mean"),
                    avg_raw_prob=("raw_model_prob", "mean"),
                    avg_edge=("net_edge_at_entry", "mean"),
                    avg_volume=("volume", "mean"),
                    avg_spread=("spread", "mean"),
                )
                .reset_index()
            )
            numeric_summary_cols = [
                "realized_pnl", "unrealized_pnl", "total_pnl", "avg_entry_yes",
                "avg_current_yes", "avg_raw_prob", "avg_edge", "avg_volume", "avg_spread",
            ]
            basket_summary[numeric_summary_cols] = basket_summary[numeric_summary_cols].round(4)
            st.markdown("#### Basket summary")
            st.dataframe(basket_summary, use_container_width=True, hide_index=True)

            detail_cols = [
                "twin_name", "policy_name", "universe", "evidence_scope", "causal_status", "fix_version",
                "opportunity_basket", "market_id", "station",
                "market_date", "temp_bucket", "status", "entry_yes_price", "current_yes_price",
                "price_move_yes", "raw_model_prob", "model_probability", "net_edge_at_entry",
                "cost", "volume", "spread", "adaptive_min_volume", "realized_pnl", "unrealized_pnl", "total_pnl", "created_at",
                "forecast_timestamp", "market_price_timestamp", "current_price_ts", "resolved_at", "exit_reason", "outcome",
            ]
            display_shadow_df = filtered_shadow_df[detail_cols].copy()
            numeric_detail_cols = [
                "entry_yes_price", "current_yes_price", "price_move_yes", "raw_model_prob",
                "model_probability", "net_edge_at_entry", "cost", "volume", "spread", "adaptive_min_volume", "realized_pnl",
                "unrealized_pnl", "total_pnl",
            ]
            display_shadow_df[numeric_detail_cols] = display_shadow_df[numeric_detail_cols].round(4)
            st.markdown("#### Trade ledger")
            st.dataframe(display_shadow_df, use_container_width=True, hide_index=True)
    else:
        st.info("No shadow policy entries yet. They will appear after the next decision cycles.")

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

    st.subheader("Main strategy attribution")
    st.markdown(
        '<p class="section-note">Derived audit buckets. Realized and MTM are separated so open losers cannot hide inside closed P&L.</p>',
        unsafe_allow_html=True,
    )
    strategy_case = _strategy_class_case("t", "m")
    attribution_rows = _query_db(f"""
        WITH latest_prices AS (
            SELECT market_id, yes_price
            FROM (
                SELECT market_id, yes_price,
                       ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC, id DESC) AS rn
                FROM market_prices
                WHERE yes_price IS NOT NULL
            )
            WHERE rn = 1
        ), valued AS (
            SELECT {strategy_case} AS strategy_class,
                   t.id, t.status, t.side, t.cost, t.amount, t.price, t.pnl,
                   t.edge_at_entry, t.raw_model_prob,
                   CASE
                       WHEN t.status='open' AND lp.yes_price IS NOT NULL AND t.side='buy_yes'
                           THEN lp.yes_price * t.amount - t.cost
                       WHEN t.status='open' AND lp.yes_price IS NOT NULL
                           THEN (1.0 - lp.yes_price) * t.amount - t.cost
                       ELSE 0
                   END AS unrealized_pnl
            FROM trades t
            LEFT JOIN markets m ON m.id = t.market_id
            LEFT JOIN latest_prices lp ON lp.market_id = t.market_id
            WHERE t.mode='paper' AND t.status != 'cancelled'
        )
        SELECT strategy_class,
               COUNT(*) AS accepted_trades,
               SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) AS open_trades,
               SUM(CASE WHEN status!='open' THEN 1 ELSE 0 END) AS closed_trades,
               SUM(CASE WHEN status!='open' AND COALESCE(pnl, 0) > 0 THEN 1 ELSE 0 END) AS winning_closed,
               COALESCE(SUM(CASE WHEN status!='open' THEN cost ELSE 0 END), 0) AS closed_cost,
               COALESCE(SUM(CASE WHEN status='open' THEN cost ELSE 0 END), 0) AS open_cost,
               COALESCE(SUM(COALESCE(pnl, 0)), 0) AS realized_pnl,
               COALESCE(SUM(unrealized_pnl), 0) AS unrealized_pnl,
               COALESCE(SUM(COALESCE(pnl, 0) + unrealized_pnl), 0) AS total_pnl,
               ROUND(AVG(edge_at_entry), 4) AS avg_edge,
               ROUND(AVG(price), 4) AS avg_entry_yes,
               ROUND(AVG(raw_model_prob), 5) AS avg_raw_prob
        FROM valued
        GROUP BY strategy_class
        ORDER BY total_pnl DESC
    """)
    if attribution_rows:
        attribution_df = pd.DataFrame([dict(r) for r in attribution_rows])
        attribution_df["closed_win_rate"] = attribution_df.apply(
            lambda r: (r["winning_closed"] / r["closed_trades"] * 100.0) if r["closed_trades"] else 0.0,
            axis=1,
        )
        attribution_df["pnl_per_closed_cost"] = attribution_df.apply(
            lambda r: (r["realized_pnl"] / r["closed_cost"]) if r["closed_cost"] else 0.0,
            axis=1,
        )
        numeric_cols = [
            "closed_cost", "open_cost", "realized_pnl", "unrealized_pnl", "total_pnl",
            "closed_win_rate", "pnl_per_closed_cost", "avg_edge", "avg_entry_yes", "avg_raw_prob",
        ]
        attribution_df[numeric_cols] = attribution_df[numeric_cols].round(4)
        st.dataframe(attribution_df, use_container_width=True, hide_index=True)
    else:
        st.info("No accepted paper trades to attribute yet.")

    st.subheader("Paper variant comparison")
    paper_variant_df = pd.DataFrame(paper_variant_rows)
    paper_variant_df[
        ["realized_pnl", "unrealized_pnl", "total_pnl", "win_rate"]
    ] = paper_variant_df[
        ["realized_pnl", "unrealized_pnl", "total_pnl", "win_rate"]
    ].round(4)
    st.dataframe(paper_variant_df, use_container_width=True, hide_index=True)
    paper_variant_detail_rows = _query_db("""
        WITH latest_prices AS (
            SELECT market_id, yes_price, timestamp
            FROM (
                SELECT market_id, yes_price, timestamp,
                       ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC, id DESC) AS rn
                FROM market_prices
                WHERE yes_price IS NOT NULL
            )
            WHERE rn = 1
        )
        SELECT p.variant_name, p.market_id, p.station, p.market_date, p.side, p.status,
               p.entry_yes_price, lp.yes_price AS current_yes_price,
               p.amount, p.cost, p.entry_volume AS volume, p.entry_spread AS spread,
               p.liquidity_min_volume, p.model_probability, p.raw_model_prob, p.edge_at_entry,
               COALESCE(CASE WHEN p.status IN ('won','lost') THEN p.pnl ELSE 0 END, 0) AS realized_pnl,
               CASE
                   WHEN p.status='open' AND lp.yes_price IS NOT NULL AND p.side='buy_yes'
                       THEN lp.yes_price * p.amount - p.cost
                   WHEN p.status='open' AND lp.yes_price IS NOT NULL
                       THEN (1.0 - lp.yes_price) * p.amount - p.cost
                   ELSE 0
               END AS unrealized_pnl,
               p.created_at, p.resolved_at, lp.timestamp AS current_price_ts
        FROM paper_variant_trades p
        LEFT JOIN latest_prices lp ON p.market_id = lp.market_id
        ORDER BY p.created_at DESC, p.id DESC
        LIMIT 200
    """)
    if paper_variant_detail_rows:
        paper_variant_detail_df = pd.DataFrame([dict(r) for r in paper_variant_detail_rows])
        paper_variant_detail_df["total_pnl"] = (
            paper_variant_detail_df["realized_pnl"] + paper_variant_detail_df["unrealized_pnl"]
        )
        variant_detail_numeric = [
            "entry_yes_price", "current_yes_price", "amount", "cost", "volume", "spread",
            "liquidity_min_volume", "model_probability", "raw_model_prob", "edge_at_entry",
            "realized_pnl", "unrealized_pnl", "total_pnl",
        ]
        paper_variant_detail_df[variant_detail_numeric] = paper_variant_detail_df[variant_detail_numeric].round(4)
        st.markdown("#### Paper variant ledger")
        st.dataframe(paper_variant_detail_df, use_container_width=True, hide_index=True)

    st.subheader("Shadow twin comparison")
    shadow_rows = _query_db("""
        WITH latest_prices AS (
            SELECT market_id, yes_price
            FROM (
                SELECT market_id, yes_price,
                       ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC, id DESC) AS rn
                FROM market_prices
                WHERE yes_price IS NOT NULL
            )
            WHERE rn = 1
        )
        SELECT s.twin_name, s.status, COUNT(*) AS trades,
               COALESCE(SUM(CASE WHEN s.status IN ('won','lost') THEN s.pnl ELSE 0 END), 0) AS realized_pnl,
               COALESCE(SUM(
                   CASE
                       WHEN s.status='open' AND lp.yes_price IS NOT NULL AND s.side='buy_yes'
                           THEN lp.yes_price * s.amount - s.cost
                       WHEN s.status='open' AND lp.yes_price IS NOT NULL
                           THEN (1.0 - lp.yes_price) * s.amount - s.cost
                       ELSE 0
                   END
               ), 0) AS unrealized_pnl,
               SUM(CASE WHEN s.status='open' AND lp.yes_price IS NULL THEN 1 ELSE 0 END) AS stale_open,
               COALESCE(SUM(s.cost), 0) AS cost,
               ROUND(AVG(s.entry_yes_price), 4) AS avg_yes_price,
               ROUND(AVG(s.raw_model_prob), 5) AS avg_raw_prob,
               ROUND(AVG(s.net_edge_at_entry), 4) AS avg_edge
        FROM shadow_twin_trades s
        LEFT JOIN latest_prices lp ON s.market_id = lp.market_id
        GROUP BY s.twin_name, s.status
        ORDER BY s.twin_name, s.status
    """)
    if shadow_rows:
        shadow_df = pd.DataFrame([dict(r) for r in shadow_rows])
        shadow_df["total_pnl"] = shadow_df["realized_pnl"] + shadow_df["unrealized_pnl"]
        shadow_chart_df = shadow_df.groupby("twin_name", as_index=False).agg({"total_pnl": "sum", "trades": "sum"})
        shadow_chart = alt.Chart(shadow_chart_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("twin_name:N", title="Twin"),
            y=alt.Y("total_pnl:Q", title="Total P&L"),
            color=alt.condition("datum.total_pnl >= 0", alt.value("#0c7a43"), alt.value("#b63b2e")),
            tooltip=["twin_name:N", "trades:Q", alt.Tooltip("total_pnl:Q", format="$.2f")],
        ).properties(height=220)
        st.altair_chart(shadow_chart, use_container_width=True)
        st.dataframe(shadow_df, use_container_width=True, hide_index=True)
    else:
        st.info("No shadow twin entries yet. They will appear after the next decision cycles.")

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

with audit_tab:
    st.subheader("Evidence audit")
    st.markdown(
        f'<p class="section-note">Temporal fix cutoff: <code>{AUDIT_TEMPORAL_FIX_CUTOFF}</code>. '
        f'Decision-grade gate reset: <code>{DECISION_GRADE_CUTOFF}</code>. '
        'Shadow tables default to clean post-fix evidence elsewhere; this tab exposes the split.</p>',
        unsafe_allow_html=True,
    )

    main_causal_rows = _query_db(f"""
        SELECT
            CASE
                WHEN forecast_timestamp IS NULL OR market_price_timestamp IS NULL THEN 'unknown'
                WHEN julianday(forecast_timestamp) > julianday(market_price_timestamp) THEN 'non_causal'
                ELSE 'clean'
            END AS causal_status,
            CASE
                WHEN julianday(created_at) >= julianday(?) THEN 'post_temporal_fix'
                ELSE 'pre_temporal_fix'
            END AS fix_version,
            COUNT(*) AS trades,
            SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) AS open_trades,
            SUM(CASE WHEN status IN {_CLOSED} THEN 1 ELSE 0 END) AS closed_trades,
            COALESCE(SUM(CASE WHEN status IN {_CLOSED} THEN pnl ELSE 0 END), 0) AS realized_pnl,
            COALESCE(SUM(CASE WHEN status='open' THEN cost ELSE 0 END), 0) AS open_cost
        FROM trades
        WHERE mode='paper' AND status != 'cancelled'
        GROUP BY causal_status, fix_version
        ORDER BY fix_version, causal_status
    """, params=(AUDIT_TEMPORAL_FIX_CUTOFF,))
    post_fix_leakage = int(_scalar(
        "SELECT COUNT(*) FROM leakage_violations WHERE julianday(violated_at) >= julianday(?)",
        params=(AUDIT_TEMPORAL_FIX_CUTOFF,),
    ))
    post_fix_paper_entries = int(_scalar(
        "SELECT COUNT(*) FROM trades WHERE mode='paper' AND julianday(created_at) >= julianday(?)",
        params=(AUDIT_TEMPORAL_FIX_CUTOFF,),
    ))
    known_clean = 0
    known_non_causal = 0
    if main_causal_rows:
        for row in main_causal_rows:
            if row["causal_status"] == "clean":
                known_clean += int(row["trades"] or 0)
            elif row["causal_status"] == "non_causal":
                known_non_causal += int(row["trades"] or 0)
    audit_cols = st.columns(4)
    audit_cols[0].metric("Post-fix leakage", post_fix_leakage)
    audit_cols[1].metric("Post-fix paper entries", post_fix_paper_entries)
    audit_cols[2].metric("Accepted clean ts", known_clean)
    audit_cols[3].metric("Accepted non-causal ts", known_non_causal)
    if main_causal_rows:
        main_causal_df = pd.DataFrame([dict(r) for r in main_causal_rows])
        main_causal_df[["realized_pnl", "open_cost"]] = main_causal_df[["realized_pnl", "open_cost"]].round(4)
        st.markdown("#### Main accepted-trade causal split")
        st.dataframe(main_causal_df, use_container_width=True, hide_index=True)

    shadow_split_rows = _query_db("""
        WITH latest_prices AS (
            SELECT market_id, yes_price
            FROM (
                SELECT market_id, yes_price,
                       ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY timestamp DESC, id DESC) AS rn
                FROM market_prices
                WHERE yes_price IS NOT NULL
            )
            WHERE rn = 1
        ), rows AS (
            SELECT 'twin' AS source, twin_name AS bucket, market_id, side, status, amount, cost, pnl,
                   forecast_timestamp, market_price_timestamp, created_at
            FROM shadow_twin_trades
            UNION ALL
            SELECT 'policy' AS source, universe_name || ' · ' || policy_name AS bucket, market_id, side, status, amount, cost, pnl,
                   forecast_timestamp, market_price_timestamp, created_at
            FROM shadow_policy_trades
        ), valued AS (
            SELECT r.*,
                   CASE
                       WHEN r.status='open' AND lp.yes_price IS NOT NULL AND r.side='buy_yes'
                           THEN lp.yes_price * r.amount - r.cost
                       WHEN r.status='open' AND lp.yes_price IS NOT NULL
                           THEN (1.0 - lp.yes_price) * r.amount - r.cost
                       ELSE 0
                   END AS unrealized_pnl
            FROM rows r
            LEFT JOIN latest_prices lp ON lp.market_id = r.market_id
        )
        SELECT source, bucket,
               CASE
                   WHEN forecast_timestamp IS NULL OR market_price_timestamp IS NULL THEN 'unknown'
                   WHEN julianday(forecast_timestamp) > julianday(market_price_timestamp) THEN 'non_causal'
                   ELSE 'clean'
               END AS causal_status,
               CASE
                   WHEN julianday(created_at) >= julianday(?) THEN 'post_temporal_fix'
                   ELSE 'pre_temporal_fix'
               END AS fix_version,
               COUNT(*) AS rows,
               COALESCE(SUM(cost), 0) AS cost,
               COALESCE(SUM(CASE WHEN status IN ('won','lost','exited') THEN pnl ELSE 0 END), 0) AS realized_pnl,
               COALESCE(SUM(unrealized_pnl), 0) AS unrealized_pnl,
               COALESCE(SUM(CASE WHEN status IN ('won','lost','exited') THEN pnl ELSE 0 END), 0)
                   + COALESCE(SUM(unrealized_pnl), 0) AS total_pnl
        FROM valued
        GROUP BY source, bucket, causal_status, fix_version
        ORDER BY source, bucket, fix_version, causal_status
    """, params=(AUDIT_TEMPORAL_FIX_CUTOFF,))
    if shadow_split_rows:
        shadow_split_df = pd.DataFrame([dict(r) for r in shadow_split_rows])
        shadow_split_df[["cost", "realized_pnl", "unrealized_pnl", "total_pnl"]] = shadow_split_df[
            ["cost", "realized_pnl", "unrealized_pnl", "total_pnl"]
        ].round(4)
        st.markdown("#### Shadow contamination split")
        st.dataframe(shadow_split_df, use_container_width=True, hide_index=True)

    st.subheader("Main P&L robustness")
    robustness_rows = _query_db(f"""
        SELECT t.id, t.market_id, t.station, t.market_date, t.status, t.pnl, t.cost,
               {_strategy_class_case('t', 'm')} AS strategy_class
        FROM trades t
        LEFT JOIN markets m ON m.id = t.market_id
        WHERE t.mode='paper' AND t.status IN {_CLOSED} AND t.pnl IS NOT NULL
    """)
    if robustness_rows:
        robustness_df = pd.DataFrame([dict(r) for r in robustness_rows])
        robustness_df["pnl"] = pd.to_numeric(robustness_df["pnl"], errors="coerce").fillna(0.0)
        total_realized = float(robustness_df["pnl"].sum())
        scenarios = [{"scenario": "All closed", "excluded": "none", "realized_pnl": total_realized, "delta": 0.0}]

        top_trades = robustness_df.sort_values("pnl", ascending=False)
        if not top_trades.empty:
            ex_top1 = robustness_df.drop(index=top_trades.head(1).index)
            scenarios.append({
                "scenario": "Exclude top 1 trade",
                "excluded": str(top_trades.iloc[0]["market_id"]),
                "realized_pnl": float(ex_top1["pnl"].sum()),
                "delta": float(ex_top1["pnl"].sum() - total_realized),
            })
            ex_top5 = robustness_df.drop(index=top_trades.head(5).index)
            scenarios.append({
                "scenario": "Exclude top 5 trades",
                "excluded": ", ".join(top_trades.head(5)["market_id"].astype(str).tolist()),
                "realized_pnl": float(ex_top5["pnl"].sum()),
                "delta": float(ex_top5["pnl"].sum() - total_realized),
            })

        for label, col in [
            ("Exclude top strategy", "strategy_class"),
            ("Exclude top station", "station"),
            ("Exclude top event date", "market_date"),
        ]:
            grouped = robustness_df.groupby(col, dropna=False)["pnl"].sum().sort_values(ascending=False)
            if not grouped.empty:
                top_key = grouped.index[0]
                remaining = robustness_df.loc[robustness_df[col] != top_key, "pnl"].sum()
                scenarios.append({
                    "scenario": label,
                    "excluded": str(top_key),
                    "realized_pnl": float(remaining),
                    "delta": float(remaining - total_realized),
                })

        scenarios_df = pd.DataFrame(scenarios)
        scenarios_df[["realized_pnl", "delta"]] = scenarios_df[["realized_pnl", "delta"]].round(4)
        st.dataframe(scenarios_df, use_container_width=True, hide_index=True)
    else:
        st.info("No closed paper trades for robustness analysis yet.")

    st.subheader("Exit vs hold-to-resolution")
    st.markdown(
        '<p class="section-note">Optional heavier query. Uses observed actual temperatures when available and only scores bucket markets with known bounds.</p>',
        unsafe_allow_html=True,
    )
    run_hold_audit = st.toggle("Run hold-to-resolution audit", value=False, key="run_hold_to_resolution_audit")
    if run_hold_audit:
        hold_rows = _query_db(f"""
            WITH closed AS (
                SELECT t.id, t.market_id, t.side, t.status, t.amount, t.cost, t.pnl,
                       t.station, t.market_date, m.threshold_low, m.threshold_high,
                       {_strategy_class_case('t', 'm')} AS strategy_class
                FROM trades t
                JOIN markets m ON m.id = t.market_id
                WHERE t.mode='paper'
                  AND t.status IN {_CLOSED}
                  AND m.threshold_low IS NOT NULL
                  AND m.threshold_high IS NOT NULL
            ), actuals AS (
                SELECT f.station, f.target_date, AVG(f.actual_temp) AS actual_temp
                FROM forecasts f
                JOIN (SELECT DISTINCT station, market_date FROM closed) c
                  ON c.station = f.station AND c.market_date = f.target_date
                WHERE f.actual_temp IS NOT NULL
                GROUP BY f.station, f.target_date
            ), scored AS (
                SELECT c.*,
                       a.actual_temp,
                       CASE
                           WHEN a.actual_temp IS NULL THEN NULL
                           WHEN c.side='buy_yes'
                                AND a.actual_temp >= c.threshold_low
                                AND a.actual_temp < c.threshold_high
                               THEN c.amount - c.cost
                           WHEN c.side='buy_yes' THEN -c.cost
                           WHEN c.side='buy_no'
                                AND NOT (a.actual_temp >= c.threshold_low AND a.actual_temp < c.threshold_high)
                               THEN c.amount - c.cost
                           ELSE -c.cost
                       END AS hold_to_resolution_pnl
                FROM closed c
                LEFT JOIN actuals a ON a.station = c.station AND a.target_date = c.market_date
            )
            SELECT strategy_class, status,
                   COUNT(*) AS trades,
                   SUM(CASE WHEN hold_to_resolution_pnl IS NOT NULL THEN 1 ELSE 0 END) AS scored_trades,
                   COALESCE(SUM(pnl), 0) AS actual_pnl,
                   COALESCE(SUM(hold_to_resolution_pnl), 0) AS hold_to_resolution_pnl,
                   COALESCE(SUM(pnl - hold_to_resolution_pnl), 0) AS exit_alpha
            FROM scored
            GROUP BY strategy_class, status
            ORDER BY exit_alpha DESC
        """)
        if hold_rows:
            hold_df = pd.DataFrame([dict(r) for r in hold_rows])
            hold_df[["actual_pnl", "hold_to_resolution_pnl", "exit_alpha"]] = hold_df[
                ["actual_pnl", "hold_to_resolution_pnl", "exit_alpha"]
            ].round(4)
            scored_total = int(hold_df["scored_trades"].sum()) if "scored_trades" in hold_df else 0
            actual_total = float(hold_df["actual_pnl"].sum())
            hold_total = float(hold_df["hold_to_resolution_pnl"].sum())
            exit_alpha_total = float(hold_df["exit_alpha"].sum())
            hold_cols = st.columns(4)
            hold_cols[0].metric("Scored trades", scored_total)
            hold_cols[1].metric("Actual exits", _fmt_signed_money(actual_total))
            hold_cols[2].metric("Hold-to-resolution", _fmt_signed_money(hold_total))
            hold_cols[3].metric("Exit alpha", _fmt_signed_money(exit_alpha_total))
            st.dataframe(hold_df, use_container_width=True, hide_index=True)
        else:
            st.info("No bucket trades with observed actual temperatures available yet.")

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
