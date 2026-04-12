"""Terminal dashboard to run three WeatherEdge scenarios in parallel.

Scenarios:
- rischioso
- bilanciato
- safer

Each scenario is isolated with its own:
- DB file (`WEATHEREDGE_DB_PATH`)
- log file (`WEATHEREDGE_LOG_FILE`)
- PID/stdout files under `runs/scenarios/<scenario>/`
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import os
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from weatheredge.data.station_time import hours_until_market_close


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "runs" / "scenarios"
COLLECTOR_PID_FILE = PROJECT_ROOT / "runs" / "scenario_snapshot_collector.pid"
COLLECTOR_LOG_FILE = PROJECT_ROOT / "logs" / "scenario_snapshot_collector.log"
COLLECTOR_OUTPUT_FILE = PROJECT_ROOT / "logs" / "scenario_snapshots.csv"
COLLECTOR_INTERVAL_SECONDS = 300

SCENARIO_ORDER = ("rischioso", "bilanciato", "safer")
SCENARIO_ALIASES = {
    "risky": "rischioso",
    "balanced": "bilanciato",
    "safe": "safer",
}

# Scenario-level risk overrides (base config still comes from .env)
SCENARIO_ENV_OVERRIDES: dict[str, dict[str, str]] = {
    "rischioso": {
        "MIN_EDGE_PERCENT": "10.0",
        "KELLY_FRACTION": "0.18",
        "MAX_OPEN_POSITIONS": "14",
        "MAX_SINGLE_MARKET_EXPOSURE": "0.07",
        "MAX_SINGLE_PLATFORM_EXPOSURE": "0.55",
        "VPIN_DANGER_THRESHOLD": "0.70",
        "ENABLE_MEAN_REVERSION_AGENT": "false",
        "CONSERVATIVE_MIN_EDGE": "0.18",
        "AGGRESSIVE_MIN_EDGE": "0.10",
    },
    "bilanciato": {
        "MIN_EDGE_PERCENT": "12.0",
        "KELLY_FRACTION": "0.10",
        "MAX_OPEN_POSITIONS": "10",
        "MAX_SINGLE_MARKET_EXPOSURE": "0.05",
        "MAX_SINGLE_PLATFORM_EXPOSURE": "0.45",
        "VPIN_DANGER_THRESHOLD": "0.65",
        "ENABLE_MEAN_REVERSION_AGENT": "false",
        "CONSERVATIVE_MIN_EDGE": "0.22",
        "AGGRESSIVE_MIN_EDGE": "0.12",
    },
    "safer": {
        "MIN_EDGE_PERCENT": "16.0",
        "KELLY_FRACTION": "0.06",
        "MAX_OPEN_POSITIONS": "6",
        "MAX_SINGLE_MARKET_EXPOSURE": "0.03",
        "MAX_SINGLE_PLATFORM_EXPOSURE": "0.35",
        "VPIN_DANGER_THRESHOLD": "0.55",
        "ENABLE_MEAN_REVERSION_AGENT": "false",
        "CONSERVATIVE_MIN_EDGE": "0.25",
        "AGGRESSIVE_MIN_EDGE": "0.16",
    },
}


@dataclass(frozen=True)
class ScenarioPaths:
    root: Path
    pid_file: Path
    db_file: Path
    log_file: Path
    stdout_file: Path


@dataclass(frozen=True)
class ScenarioStats:
    trades: int | None
    open_positions: int | None
    wins: int | None
    losses: int | None
    win_rate: float | None
    pnl: float | None
    unrealized_pnl: float | None
    forecasts: int | None
    market_prices: int | None
    markets: int | None
    last_forecast_ts: str | None
    last_market_price_ts: str | None
    forecast_age_minutes: float | None
    price_age_minutes: float | None


@dataclass(frozen=True)
class OpenTradeETA:
    trade_id: int
    market_id: str
    side: str
    hours_left: float | None


@dataclass(frozen=True)
class ScenarioStatus:
    name: str
    running: bool
    pid: int | None
    uptime: str
    stats: ScenarioStats
    open_trade_eta: list[OpenTradeETA]
    last_log_line: str
    paths: ScenarioPaths


EMPTY_STATS = ScenarioStats(
    trades=None,
    open_positions=None,
    wins=None,
    losses=None,
    win_rate=None,
    pnl=None,
    unrealized_pnl=None,
    forecasts=None,
    market_prices=None,
    markets=None,
    last_forecast_ts=None,
    last_market_price_ts=None,
    forecast_age_minutes=None,
    price_age_minutes=None,
)


def resolve_scenario_name(name: str) -> str:
    value = name.strip().lower()
    value = SCENARIO_ALIASES.get(value, value)
    if value not in SCENARIO_ENV_OVERRIDES:
        valid = ", ".join(SCENARIO_ORDER)
        raise ValueError(f"Scenario non valido '{name}'. Valori validi: {valid}, all")
    return value


def scenario_paths(name: str) -> ScenarioPaths:
    scenario = resolve_scenario_name(name)
    root = RUN_ROOT / scenario
    return ScenarioPaths(
        root=root,
        pid_file=root / "weatheredge.pid",
        db_file=root / "weatheredge.db",
        log_file=root / "weatheredge.log",
        stdout_file=root / "stdout.log",
    )


def _targets(target: str) -> list[str]:
    value = target.strip().lower()
    if value == "all":
        return list(SCENARIO_ORDER)
    return [resolve_scenario_name(value)]


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError as exc:
        # EPERM => process exists but we cannot signal it in this context.
        return exc.errno == errno.EPERM


def _read_pid(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None
    raw = pid_file.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _tail_lines(path: Path, lines: int = 20) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in deque(handle, maxlen=max(lines, 1))]


def _last_log_line(path: Path) -> str:
    lines = _tail_lines(path, lines=5)
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return "-"


def _safe_scalar(cursor: sqlite3.Cursor, sql: str) -> float:
    cursor.execute(sql)
    row = cursor.fetchone()
    if row is None or row[0] is None:
        return 0.0
    return float(row[0])


def _safe_optional_scalar(cursor: sqlite3.Cursor, sql: str) -> float | None:
    cursor.execute(sql)
    row = cursor.fetchone()
    if row is None or row[0] is None:
        return None
    return float(row[0])


def _safe_text(cursor: sqlite3.Cursor, sql: str) -> str | None:
    cursor.execute(sql)
    row = cursor.fetchone()
    if row is None or row[0] is None:
        return None
    return str(row[0])


def _table_columns(cursor: sqlite3.Cursor, table_name: str) -> set[str]:
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        rows = cursor.fetchall()
        return {str(row[1]) for row in rows}
    except sqlite3.DatabaseError:
        return set()


def _query_unrealized_pnl(db_path: Path) -> float | None:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            trade_cols = _table_columns(cursor, "trades")
            price_cols = _table_columns(cursor, "market_prices")
            required_trade = {"status", "side", "amount", "cost", "market_id"}
            required_price = {"market_id", "yes_price", "timestamp", "id"}
            if not required_trade.issubset(trade_cols) or not required_price.issubset(price_cols):
                return None

            cursor.execute(
                """
                WITH latest_price AS (
                    SELECT market_id, yes_price
                    FROM (
                        SELECT market_id, yes_price, timestamp, id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY market_id
                                   ORDER BY timestamp DESC, id DESC
                               ) AS rn
                        FROM market_prices
                        WHERE market_id IS NOT NULL
                    )
                    WHERE rn = 1
                )
                SELECT t.side, t.amount, t.cost, lp.yes_price
                FROM trades t
                LEFT JOIN latest_price lp ON lp.market_id = t.market_id
                WHERE t.status = 'open'
                """
            )
            rows = cursor.fetchall()
    except sqlite3.DatabaseError:
        return None

    unrealized = 0.0
    for row in rows:
        side = str(row["side"] or "").lower()
        amount = float(row["amount"] or 0.0)
        cost = float(row["cost"] or 0.0)
        yes = row["yes_price"]
        if yes is None:
            continue
        yes_price = float(yes)
        value = (1.0 - yes_price) * amount if side == "buy_no" else yes_price * amount
        unrealized += value - cost
    return unrealized


def _query_stats(db_path: Path) -> ScenarioStats:
    if not db_path.exists():
        return EMPTY_STATS

    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            trades = int(_safe_scalar(cursor, "SELECT COUNT(*) FROM trades"))
            open_positions = int(
                _safe_scalar(cursor, "SELECT COUNT(*) FROM trades WHERE status='open'")
            )
            wins = int(_safe_scalar(cursor, "SELECT COUNT(*) FROM trades WHERE status='won'"))
            losses = int(_safe_scalar(cursor, "SELECT COUNT(*) FROM trades WHERE status='lost'"))
            pnl = _safe_scalar(cursor, "SELECT COALESCE(SUM(pnl), 0) FROM trades")
            forecasts = int(_safe_scalar(cursor, "SELECT COUNT(*) FROM forecasts"))
            market_prices = int(_safe_scalar(cursor, "SELECT COUNT(*) FROM market_prices"))
            markets = int(_safe_scalar(cursor, "SELECT COUNT(*) FROM markets"))
            last_forecast_ts = _safe_text(
                cursor, "SELECT MAX(forecast_timestamp) FROM forecasts"
            )
            last_market_price_ts = _safe_text(cursor, "SELECT MAX(timestamp) FROM market_prices")
            forecast_age_minutes = _safe_optional_scalar(
                cursor,
                "SELECT (julianday('now') - julianday(MAX(forecast_timestamp))) * 1440.0 "
                "FROM forecasts",
            )
            price_age_minutes = _safe_optional_scalar(
                cursor,
                "SELECT (julianday('now') - julianday(MAX(timestamp))) * 1440.0 "
                "FROM market_prices",
            )
            unrealized = _query_unrealized_pnl(db_path)
    except sqlite3.DatabaseError:
        return EMPTY_STATS

    resolved = wins + losses
    win_rate = (wins / resolved) * 100.0 if resolved > 0 else None
    return ScenarioStats(
        trades=trades,
        open_positions=open_positions,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        pnl=pnl,
        unrealized_pnl=unrealized,
        forecasts=forecasts,
        market_prices=market_prices,
        markets=markets,
        last_forecast_ts=last_forecast_ts,
        last_market_price_ts=last_market_price_ts,
        forecast_age_minutes=forecast_age_minutes,
        price_age_minutes=price_age_minutes,
    )


def _uptime_for_pid(pid: int | None) -> str:
    if pid is None:
        return "-"
    try:
        result = subprocess.run(
            ["ps", "-o", "etime=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        value = result.stdout.strip()
        return value or "-"
    except OSError:
        return "-"


def _query_open_trade_eta(db_path: Path) -> list[OpenTradeETA]:
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT t.id AS trade_id, t.market_id, t.side, m.date, m.station
                FROM trades t
                LEFT JOIN markets m ON m.id = t.market_id
                WHERE t.status = 'open'
                ORDER BY t.created_at, t.id
                """
            )
            rows = cursor.fetchall()
    except sqlite3.DatabaseError:
        return []

    results: list[OpenTradeETA] = []
    for row in rows:
        date = row["date"]
        station = row["station"]
        hours_left: float | None = None
        if date and station:
            try:
                hours_left = hours_until_market_close(str(date), str(station))
                if hours_left is not None:
                    hours_left = max(hours_left, 0.0)
            except Exception:
                hours_left = None
        results.append(
            OpenTradeETA(
                trade_id=int(row["trade_id"]),
                market_id=str(row["market_id"] or ""),
                side=str(row["side"] or ""),
                hours_left=hours_left,
            )
        )
    return sorted(results, key=lambda x: (x.hours_left is None, x.hours_left or 0.0, x.trade_id))


def _read_collector_pid() -> int | None:
    return _read_pid(COLLECTOR_PID_FILE)


def _collector_running() -> bool:
    pid = _read_collector_pid()
    if pid is None:
        return False
    if _is_pid_running(pid):
        return True
    _safe_unlink(COLLECTOR_PID_FILE)
    return False


def _collector_status_text() -> str:
    pid = _read_collector_pid()
    if pid is not None and _is_pid_running(pid):
        return f"RUN (PID {pid}, every {COLLECTOR_INTERVAL_SECONDS}s)"
    return "STOP"


def _ensure_snapshot_collector_running() -> tuple[bool, str]:
    if _collector_running():
        pid = _read_collector_pid()
        return False, f"snapshot collector gia attivo (PID {pid})"

    COLLECTOR_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    COLLECTOR_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "weatheredge.dashboard.scenario_snapshot_collector",
        "--interval-seconds",
        str(COLLECTOR_INTERVAL_SECONDS),
        "--output",
        str(COLLECTOR_OUTPUT_FILE),
        "--quiet",
    ]
    try:
        with COLLECTOR_LOG_FILE.open("ab") as out:
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        COLLECTOR_PID_FILE.write_text(str(process.pid), encoding="utf-8")
        time.sleep(0.4)
        if process.poll() is not None:
            _safe_unlink(COLLECTOR_PID_FILE)
            return False, "snapshot collector: avvio fallito"
        return True, f"snapshot collector avviato (PID {process.pid})"
    except Exception as exc:
        _safe_unlink(COLLECTOR_PID_FILE)
        return False, f"snapshot collector: errore in avvio ({exc})"


def _any_scenario_running() -> bool:
    for scenario in SCENARIO_ORDER:
        pid = _read_pid(scenario_paths(scenario).pid_file)
        if pid is not None and _is_pid_running(pid):
            return True
    return False


def _stop_snapshot_collector_if_idle(timeout_seconds: float = 5.0) -> str | None:
    if _any_scenario_running():
        return None
    pid = _read_collector_pid()
    if pid is None:
        return None
    if not _is_pid_running(pid):
        _safe_unlink(COLLECTOR_PID_FILE)
        return "snapshot collector fermo (pid stale pulito)"
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        _safe_unlink(COLLECTOR_PID_FILE)
        return "snapshot collector: processo non trovato"

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _is_pid_running(pid):
            _safe_unlink(COLLECTOR_PID_FILE)
            return "snapshot collector fermato"
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
        _safe_unlink(COLLECTOR_PID_FILE)
        return "snapshot collector fermato forzatamente"
    except OSError:
        _safe_unlink(COLLECTOR_PID_FILE)
        return "snapshot collector: stop non riuscito"


def collect_status(name: str) -> ScenarioStatus:
    scenario = resolve_scenario_name(name)
    paths = scenario_paths(scenario)
    pid = _read_pid(paths.pid_file)
    running = bool(pid is not None and _is_pid_running(pid))
    if pid is not None and not running and paths.pid_file.exists():
        # stale pid file from crashed/old process
        _safe_unlink(paths.pid_file)
        pid = None

    return ScenarioStatus(
        name=scenario,
        running=running,
        pid=pid,
        uptime=_uptime_for_pid(pid if running else None),
        stats=_query_stats(paths.db_file),
        open_trade_eta=_query_open_trade_eta(paths.db_file),
        last_log_line=_last_log_line(paths.log_file),
        paths=paths,
    )


def _build_env(scenario: str, paths: ScenarioPaths) -> dict[str, str]:
    env = os.environ.copy()
    env.update(SCENARIO_ENV_OVERRIDES[scenario])
    env["WEATHEREDGE_DB_PATH"] = str(paths.db_file)
    env["WEATHEREDGE_LOG_FILE"] = str(paths.log_file)
    return env


def start_scenario(name: str) -> tuple[bool, str]:
    scenario = resolve_scenario_name(name)
    status = collect_status(scenario)
    if status.running:
        return False, f"{scenario}: gia avviato (PID {status.pid})"

    paths = status.paths
    paths.root.mkdir(parents=True, exist_ok=True)
    env = _build_env(scenario, paths)

    try:
        with paths.stdout_file.open("ab") as stdout:
            process = subprocess.Popen(
                [sys.executable, "-m", "weatheredge.main"],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        paths.pid_file.write_text(str(process.pid), encoding="utf-8")
        time.sleep(0.8)
        if process.poll() is not None:
            _safe_unlink(paths.pid_file)
            return False, f"{scenario}: avvio fallito (vedi {paths.stdout_file})"
        return True, f"{scenario}: avviato (PID {process.pid})"
    except Exception as exc:
        _safe_unlink(paths.pid_file)
        return False, f"{scenario}: errore in avvio ({exc})"


def stop_scenario(name: str, timeout_seconds: float = 8.0) -> tuple[bool, str]:
    scenario = resolve_scenario_name(name)
    paths = scenario_paths(scenario)
    pid = _read_pid(paths.pid_file)
    if pid is None:
        return False, f"{scenario}: non in esecuzione"

    if not _is_pid_running(pid):
        _safe_unlink(paths.pid_file)
        return False, f"{scenario}: pid stale rimosso ({pid})"

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        _safe_unlink(paths.pid_file)
        return False, f"{scenario}: processo non trovato ({pid})"

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _is_pid_running(pid):
            _safe_unlink(paths.pid_file)
            return True, f"{scenario}: fermato (SIGTERM)"
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
        _safe_unlink(paths.pid_file)
        return True, f"{scenario}: fermato forzatamente (SIGKILL)"
    except OSError:
        _safe_unlink(paths.pid_file)
        return False, f"{scenario}: non riuscito a fermare PID {pid}"


def _fmt_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def _fmt_float(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_age_minutes(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{max(value, 0.0):.0f}m"


def _fmt_ts(value: str | None) -> str:
    if not value:
        return "-"
    return value.replace("T", " ")[:19]


def print_status_table() -> None:
    statuses = [collect_status(name) for name in SCENARIO_ORDER]
    print("WeatherEdge Multi-Scenario Dashboard")
    print(f"Root run dir: {RUN_ROOT}")
    print(f"Snapshot collector: {_collector_status_text()}")
    print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print()
    header = (
        f"{'Scenario':<11} {'Stato':<8} {'PID':<8} {'Uptime':<10} "
        f"{'Trades':>7} {'Open':>5} {'WR%':>6} {'PnL':>10} {'UnrPnL':>10} {'Fcst':>6} "
        f"{'Px':>6} {'Mkt':>5} {'F_Age':>6} {'P_Age':>6}"
    )
    print(header)
    print("-" * len(header))

    for item in statuses:
        state = "RUN" if item.running else "STOP"
        print(
            f"{item.name:<11} {state:<8} {_fmt_int(item.pid):<8} {item.uptime:<10} "
            f"{_fmt_int(item.stats.trades):>7} {_fmt_int(item.stats.open_positions):>5} "
            f"{_fmt_float(item.stats.win_rate, 1):>6} {_fmt_float(item.stats.pnl, 2):>10} "
            f"{_fmt_float(item.stats.unrealized_pnl, 2):>10} "
            f"{_fmt_int(item.stats.forecasts):>6} {_fmt_int(item.stats.market_prices):>6} "
            f"{_fmt_int(item.stats.markets):>5} {_fmt_age_minutes(item.stats.forecast_age_minutes):>6} "
            f"{_fmt_age_minutes(item.stats.price_age_minutes):>6}"
        )

    print()
    print("Feed freshness:")
    for item in statuses:
        print(
            f"- {item.name}: forecast_ts={_fmt_ts(item.stats.last_forecast_ts)} | "
            f"price_ts={_fmt_ts(item.stats.last_market_price_ts)}"
        )

    print()
    print("Ultimo log per scenario:")
    for item in statuses:
        print(f"- {item.name}: {item.last_log_line}")

    print()
    print("Ore a fine evento per trade aperto:")
    for item in statuses:
        if not item.open_trade_eta:
            print(f"{item.name}: nessun trade aperto")
            continue
        for trade in item.open_trade_eta:
            eta = "-" if trade.hours_left is None else f"{trade.hours_left:.1f}h"
            print(
                f"{item.name} | trade={trade.trade_id} | market={trade.market_id} | "
                f"side={trade.side} | eta={eta}"
            )


def _print_action_results(messages: list[str]) -> None:
    for message in messages:
        print(message)


def _run_start_or_stop(action: str, target: str) -> int:
    try:
        scenarios = _targets(target)
    except ValueError as exc:
        print(str(exc))
        return 2

    messages: list[str] = []
    success_count = 0
    for scenario in scenarios:
        ok, message = start_scenario(scenario) if action == "start" else stop_scenario(scenario)
        messages.append(message)
        if ok:
            success_count += 1

    if action == "start" and success_count > 0:
        _collector_ok, collector_msg = _ensure_snapshot_collector_running()
        messages.append(collector_msg)
    if action == "stop":
        collector_stop_msg = _stop_snapshot_collector_if_idle()
        if collector_stop_msg:
            messages.append(collector_stop_msg)

    _print_action_results(messages)
    return 0 if success_count == len(scenarios) else 1


def print_logs(scenario: str, lines: int = 20) -> int:
    try:
        resolved = resolve_scenario_name(scenario)
    except ValueError as exc:
        print(str(exc))
        return 2

    paths = scenario_paths(resolved)
    print(f"{resolved} log file: {paths.log_file}")
    log_lines = _tail_lines(paths.log_file, lines=max(lines, 1))
    if not log_lines:
        print("(nessun log)")
        return 0
    for line in log_lines:
        print(line)
    return 0


def _clear_terminal() -> None:
    print("\033[2J\033[H", end="")


def run_dashboard() -> int:
    while True:
        _clear_terminal()
        print_status_table()
        print()
        print("Comandi: start [all|scenario], stop [all|scenario], status, logs <scenario> [n], quit")
        try:
            raw = input("> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        if not raw:
            continue

        parts = shlex.split(raw)
        command = parts[0].lower()

        if command in {"q", "quit", "exit"}:
            return 0

        if command == "status":
            continue

        if command in {"start", "stop"}:
            target = parts[1] if len(parts) > 1 else "all"
            _run_start_or_stop(command, target)
            time.sleep(1.2)
            continue

        if command == "logs":
            if len(parts) < 2:
                print("Uso: logs <scenario> [linee]")
                time.sleep(1.2)
                continue
            if len(parts) > 2:
                try:
                    line_count = int(parts[2])
                except ValueError:
                    print("Numero righe non valido.")
                    time.sleep(1.2)
                    continue
            else:
                line_count = 20
            _clear_terminal()
            print_logs(parts[1], lines=line_count)
            input("\nInvio per tornare alla dashboard...")
            continue

        print(f"Comando non valido: {command}")
        time.sleep(1.2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WeatherEdge multi-scenario dashboard (rischioso/bilanciato/safer)."
    )
    subparsers = parser.add_subparsers(dest="command")

    p_start = subparsers.add_parser("start", help="Avvia uno scenario o tutti.")
    p_start.add_argument("target", nargs="?", default="all")

    p_stop = subparsers.add_parser("stop", help="Ferma uno scenario o tutti.")
    p_stop.add_argument("target", nargs="?", default="all")

    subparsers.add_parser("status", help="Mostra stato di tutti gli scenari.")

    p_logs = subparsers.add_parser("logs", help="Mostra ultimi log di uno scenario.")
    p_logs.add_argument("scenario")
    p_logs.add_argument("--lines", type=int, default=20)

    subparsers.add_parser("dashboard", help="Apri dashboard interattiva.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "dashboard"):
        return run_dashboard()
    if args.command == "status":
        print_status_table()
        return 0
    if args.command in {"start", "stop"}:
        return _run_start_or_stop(args.command, args.target)
    if args.command == "logs":
        return print_logs(args.scenario, lines=args.lines)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
