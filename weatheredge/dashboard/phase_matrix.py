"""Run and monitor parallel paper engines for phase comparison.

Profiles currently supported:
- phase_1
- phase_2

Each profile runs in an isolated process with:
- WEATHEREDGE_PHASE set per process
- Dedicated DB file (WEATHEREDGE_DB_PATH)
- Dedicated log file (WEATHEREDGE_LOG_FILE)
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import os
from pathlib import Path
import shlex
import signal
import sqlite3
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "runs" / "phases"
PHASE_PROFILE_ORDER = ("phase_1", "phase_2")
PHASE_PROFILE_ALIASES = {
    "phase1": "phase_1",
    "phase2": "phase_2",
}

_CLOSED_STATUSES = (
    "won",
    "lost",
    "edge_stop",
    "capital_protect",
    "time_decay",
    "ev_based",
    "abandoned",
)


@dataclass(frozen=True)
class ProfilePaths:
    root: Path
    pid_file: Path
    db_file: Path
    log_file: Path
    stdout_file: Path


@dataclass(frozen=True)
class ProfileStats:
    trades: int | None
    open_positions: int | None
    wins_final: int | None
    losses_final: int | None
    resolved_closed: int | None
    win_rate_closed: float | None
    realized_pnl: float | None
    unrealized_pnl: float | None
    brier_score: float | None
    forecasts: int | None
    market_prices: int | None
    calibration_rows: int | None
    calibration_records: int | None
    forecast_age_minutes: float | None
    price_age_minutes: float | None


@dataclass(frozen=True)
class ProfileStatus:
    name: str
    running: bool
    pid: int | None
    uptime: str
    stats: ProfileStats
    last_log_line: str
    paths: ProfilePaths


EMPTY_STATS = ProfileStats(
    trades=None,
    open_positions=None,
    wins_final=None,
    losses_final=None,
    resolved_closed=None,
    win_rate_closed=None,
    realized_pnl=None,
    unrealized_pnl=None,
    brier_score=None,
    forecasts=None,
    market_prices=None,
    calibration_rows=None,
    calibration_records=None,
    forecast_age_minutes=None,
    price_age_minutes=None,
)


def resolve_profile_name(name: str) -> str:
    value = name.strip().lower()
    value = PHASE_PROFILE_ALIASES.get(value, value)
    if value not in PHASE_PROFILE_ORDER:
        valid = ", ".join(PHASE_PROFILE_ORDER)
        raise ValueError(f"Profilo non valido '{name}'. Valori validi: {valid}, all")
    return value


def profile_paths(name: str) -> ProfilePaths:
    profile = resolve_profile_name(name)
    root = RUN_ROOT / profile
    return ProfilePaths(
        root=root,
        pid_file=root / "weatheredge.pid",
        db_file=root / "weatheredge.db",
        log_file=root / "weatheredge.log",
        stdout_file=root / "stdout.log",
    )


def _targets(target: str) -> list[str]:
    value = target.strip().lower()
    if value == "all":
        return list(PHASE_PROFILE_ORDER)
    return [resolve_profile_name(value)]


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError as exc:
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


def _query_unrealized_pnl(cursor: sqlite3.Cursor) -> float:
    cursor.execute(
        """
        WITH latest_price AS (
            SELECT market_id, yes_price, no_price
            FROM (
                SELECT market_id, yes_price, no_price, timestamp, id,
                       ROW_NUMBER() OVER (
                           PARTITION BY market_id
                           ORDER BY timestamp DESC, id DESC
                       ) AS rn
                FROM market_prices
                WHERE market_id IS NOT NULL
            )
            WHERE rn = 1
        )
        SELECT t.side, t.amount, t.cost, lp.yes_price, lp.no_price
        FROM trades t
        LEFT JOIN latest_price lp ON lp.market_id = t.market_id
        WHERE t.status = 'open'
        """
    )
    rows = cursor.fetchall()
    unrealized = 0.0
    for side, amount, cost, yes_price, no_price in rows:
        if yes_price is None:
            continue
        side_val = str(side or "").lower()
        amt = float(amount or 0.0)
        cst = float(cost or 0.0)
        yes = float(yes_price)
        no = float(no_price) if no_price is not None and float(no_price) > 0 else (1.0 - yes)
        mark_price = no if side_val == "buy_no" else yes
        unrealized += (mark_price * amt) - cst
    return unrealized


def _query_brier_score(cursor: sqlite3.Cursor) -> float | None:
    cursor.execute(
        """
        SELECT model_probability, status, side
        FROM trades
        WHERE status IN ('won','lost') AND model_probability IS NOT NULL
        """
    )
    rows = cursor.fetchall()
    if not rows:
        return None
    total = 0.0
    for prob, status, side in rows:
        p = float(prob)
        side_val = str(side or "").lower()
        status_val = str(status or "").lower()
        if (side_val == "buy_yes" and status_val == "won") or (
            side_val == "buy_no" and status_val == "lost"
        ):
            outcome_yes = 1.0
        else:
            outcome_yes = 0.0
        total += (p - outcome_yes) ** 2
    return total / len(rows)


def _query_stats(db_path: Path) -> ProfileStats:
    if not db_path.exists():
        return EMPTY_STATS

    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            trades = int(_safe_scalar(cursor, "SELECT COUNT(*) FROM trades"))
            open_positions = int(_safe_scalar(cursor, "SELECT COUNT(*) FROM trades WHERE status='open'"))
            wins_final = int(_safe_scalar(cursor, "SELECT COUNT(*) FROM trades WHERE status='won'"))
            losses_final = int(_safe_scalar(cursor, "SELECT COUNT(*) FROM trades WHERE status='lost'"))
            resolved_closed = int(
                _safe_scalar(
                    cursor,
                    "SELECT COUNT(*) FROM trades "
                    f"WHERE status IN {tuple(_CLOSED_STATUSES)}",
                )
            )
            wins_closed = int(
                _safe_scalar(
                    cursor,
                    "SELECT COUNT(*) FROM trades "
                    f"WHERE status IN {tuple(_CLOSED_STATUSES)} AND COALESCE(pnl, 0) > 0",
                )
            )
            realized_pnl = _safe_scalar(
                cursor,
                "SELECT COALESCE(SUM(pnl), 0) FROM trades "
                f"WHERE status IN {tuple(_CLOSED_STATUSES)}",
            )
            forecasts = int(_safe_scalar(cursor, "SELECT COUNT(*) FROM forecasts"))
            market_prices = int(_safe_scalar(cursor, "SELECT COUNT(*) FROM market_prices"))
            calibration_rows = int(_safe_scalar(cursor, "SELECT COUNT(*) FROM calibration"))
            calibration_records = int(_safe_scalar(cursor, "SELECT COUNT(*) FROM calibration_records"))
            forecast_age_minutes = _safe_optional_scalar(
                cursor,
                "SELECT (julianday('now') - julianday(MAX(forecast_timestamp))) * 1440.0 FROM forecasts",
            )
            price_age_minutes = _safe_optional_scalar(
                cursor,
                "SELECT (julianday('now') - julianday(MAX(timestamp))) * 1440.0 FROM market_prices",
            )
            unrealized_pnl = _query_unrealized_pnl(cursor)
            brier_score = _query_brier_score(cursor)
    except sqlite3.DatabaseError:
        return EMPTY_STATS

    win_rate_closed = (wins_closed / resolved_closed * 100.0) if resolved_closed > 0 else None

    return ProfileStats(
        trades=trades,
        open_positions=open_positions,
        wins_final=wins_final,
        losses_final=losses_final,
        resolved_closed=resolved_closed,
        win_rate_closed=win_rate_closed,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        brier_score=brier_score,
        forecasts=forecasts,
        market_prices=market_prices,
        calibration_rows=calibration_rows,
        calibration_records=calibration_records,
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


def collect_status(name: str) -> ProfileStatus:
    profile = resolve_profile_name(name)
    paths = profile_paths(profile)
    pid = _read_pid(paths.pid_file)
    running = bool(pid is not None and _is_pid_running(pid))
    if pid is not None and not running and paths.pid_file.exists():
        _safe_unlink(paths.pid_file)
        pid = None

    return ProfileStatus(
        name=profile,
        running=running,
        pid=pid,
        uptime=_uptime_for_pid(pid if running else None),
        stats=_query_stats(paths.db_file),
        last_log_line=_last_log_line(paths.log_file),
        paths=paths,
    )


def _build_env(profile: str, paths: ProfilePaths) -> dict[str, str]:
    env = os.environ.copy()
    env["WEATHEREDGE_DB_PATH"] = str(paths.db_file)
    env["WEATHEREDGE_LOG_FILE"] = str(paths.log_file)
    env["WEATHEREDGE_PHASE"] = profile
    return env


def start_profile(name: str) -> tuple[bool, str]:
    profile = resolve_profile_name(name)
    status = collect_status(profile)
    if status.running:
        return False, f"{profile}: gia avviato (PID {status.pid})"

    paths = status.paths
    paths.root.mkdir(parents=True, exist_ok=True)
    env = _build_env(profile, paths)

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
            return False, f"{profile}: avvio fallito (vedi {paths.stdout_file})"
        return True, f"{profile}: avviato (PID {process.pid})"
    except Exception as exc:
        _safe_unlink(paths.pid_file)
        return False, f"{profile}: errore in avvio ({exc})"


def stop_profile(name: str, timeout_seconds: float = 8.0) -> tuple[bool, str]:
    profile = resolve_profile_name(name)
    paths = profile_paths(profile)
    pid = _read_pid(paths.pid_file)
    if pid is None:
        return False, f"{profile}: non in esecuzione"
    if not _is_pid_running(pid):
        _safe_unlink(paths.pid_file)
        return False, f"{profile}: pid stale rimosso ({pid})"

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        _safe_unlink(paths.pid_file)
        return False, f"{profile}: processo non trovato ({pid})"

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _is_pid_running(pid):
            _safe_unlink(paths.pid_file)
            return True, f"{profile}: fermato (SIGTERM)"
        time.sleep(0.2)

    try:
        os.kill(pid, signal.SIGKILL)
        _safe_unlink(paths.pid_file)
        return True, f"{profile}: fermato forzatamente (SIGKILL)"
    except OSError:
        _safe_unlink(paths.pid_file)
        return False, f"{profile}: non riuscito a fermare PID {pid}"


def _fmt_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def _fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_age_minutes(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{max(value, 0.0):.0f}m"


def print_status_table() -> None:
    statuses = [collect_status(name) for name in PHASE_PROFILE_ORDER]
    print("WeatherEdge Phase Matrix (paper A/B)")
    print(f"Run dir: {RUN_ROOT}")
    print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print()
    header = (
        f"{'Profile':<9} {'State':<6} {'PID':<8} {'Uptime':<10} {'Trades':>7} "
        f"{'Open':>5} {'Closed':>7} {'WR%':>6} {'RealPnL':>9} {'UnrPnL':>9} "
        f"{'Brier':>7} {'CalRec':>7} {'FcAge':>6} {'PxAge':>6}"
    )
    print(header)
    print("-" * len(header))
    for item in statuses:
        stats = item.stats
        print(
            f"{item.name:<9} "
            f"{('RUN' if item.running else 'STOP'):<6} "
            f"{_fmt_int(item.pid):<8} {item.uptime:<10} "
            f"{_fmt_int(stats.trades):>7} {_fmt_int(stats.open_positions):>5} "
            f"{_fmt_int(stats.resolved_closed):>7} {_fmt_float(stats.win_rate_closed, 1):>6} "
            f"{_fmt_float(stats.realized_pnl):>9} {_fmt_float(stats.unrealized_pnl):>9} "
            f"{_fmt_float(stats.brier_score, 4):>7} {_fmt_int(stats.calibration_records):>7} "
            f"{_fmt_age_minutes(stats.forecast_age_minutes):>6} {_fmt_age_minutes(stats.price_age_minutes):>6}"
        )
    print()
    print("Ultimo log per profilo:")
    for item in statuses:
        print(f"- {item.name}: {item.last_log_line}")


def _run_start_or_stop(action: str, target: str) -> int:
    try:
        profiles = _targets(target)
    except ValueError as exc:
        print(str(exc))
        return 2

    messages: list[str] = []
    success = 0
    for profile in profiles:
        ok, msg = start_profile(profile) if action == "start" else stop_profile(profile)
        messages.append(msg)
        if ok:
            success += 1
    for msg in messages:
        print(msg)
    return 0 if success == len(profiles) else 1


def print_logs(profile: str, lines: int = 30) -> int:
    try:
        resolved = resolve_profile_name(profile)
    except ValueError as exc:
        print(str(exc))
        return 2

    paths = profile_paths(resolved)
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
        print("Comandi: start [all|profile], stop [all|profile], status, logs <profile> [n], quit")
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
                print("Uso: logs <profile> [linee]")
                time.sleep(1.2)
                continue
            line_count = 30
            if len(parts) > 2:
                try:
                    line_count = int(parts[2])
                except ValueError:
                    print("Numero righe non valido.")
                    time.sleep(1.2)
                    continue
            _clear_terminal()
            print_logs(parts[1], lines=line_count)
            input("\nInvio per tornare alla dashboard...")
            continue

        print(f"Comando non valido: {command}")
        time.sleep(1.2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WeatherEdge phase matrix dashboard (phase_1 vs phase_2)."
    )
    subparsers = parser.add_subparsers(dest="command")

    p_start = subparsers.add_parser("start", help="Avvia un profilo o tutti.")
    p_start.add_argument("target", nargs="?", default="all")

    p_stop = subparsers.add_parser("stop", help="Ferma un profilo o tutti.")
    p_stop.add_argument("target", nargs="?", default="all")

    subparsers.add_parser("status", help="Mostra stato di tutti i profili.")

    p_logs = subparsers.add_parser("logs", help="Mostra ultimi log di un profilo.")
    p_logs.add_argument("profile")
    p_logs.add_argument("--lines", type=int, default=30)

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
        return print_logs(args.profile, lines=args.lines)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
