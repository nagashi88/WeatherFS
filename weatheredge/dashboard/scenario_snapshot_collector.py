"""Periodic snapshot collector for multi-scenario paper trading.

Writes CSV snapshots for:
- status/freshness metrics from scenario dashboard
- mark-to-market unrealized/realized/equity per scenario
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from weatheredge.dashboard.scenario_dashboard import SCENARIO_ORDER, collect_status


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "logs" / "scenario_snapshots.csv"

CSV_FIELDS = [
    "collected_at_utc",
    "scenario",
    "running",
    "pid",
    "uptime",
    "trades",
    "open_positions",
    "wins",
    "losses",
    "win_rate_pct",
    "pnl_reported",
    "forecasts",
    "market_prices",
    "markets",
    "forecast_age_min",
    "price_age_min",
    "forecast_ts",
    "price_ts",
    "open_cost",
    "mark_value",
    "unrealized_pnl",
    "realized_pnl",
    "equity_now",
    "missing_prices",
    "last_log_line",
]


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _compute_mark_to_market(db_path: Path, initial_bankroll: float) -> dict[str, float | int]:
    if not db_path.exists():
        return {
            "open_cost": 0.0,
            "mark_value": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "equity_now": initial_bankroll,
            "missing_prices": 0,
        }

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            """
            SELECT COALESCE(SUM(pnl), 0.0) AS realized
            FROM trades
            WHERE status != 'open'
            """
        )
        realized = _to_float(cur.fetchone()["realized"])

        cur.execute(
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
        rows = cur.fetchall()

    unrealized = 0.0
    open_cost = 0.0
    mark_value = 0.0
    missing_prices = 0

    for row in rows:
        side = (row["side"] or "").lower()
        amount = _to_float(row["amount"])
        cost = _to_float(row["cost"])
        yes = row["yes_price"]
        open_cost += cost

        if yes is None:
            missing_prices += 1
            mark_value += cost
            continue

        yes_value = float(yes)
        if side == "buy_no":
            value = (1.0 - yes_value) * amount
        else:
            value = yes_value * amount

        mark_value += value
        unrealized += (value - cost)

    equity_now = initial_bankroll + realized + unrealized
    return {
        "open_cost": round(open_cost, 6),
        "mark_value": round(mark_value, 6),
        "unrealized_pnl": round(unrealized, 6),
        "realized_pnl": round(realized, 6),
        "equity_now": round(equity_now, 6),
        "missing_prices": missing_prices,
    }


def collect_snapshot_rows(initial_bankroll: float = 1000.0, include_total: bool = True) -> list[dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []

    total_unrealized = 0.0
    total_realized = 0.0
    total_equity = 0.0
    total_open = 0
    total_trades = 0
    total_missing_prices = 0

    for scenario in SCENARIO_ORDER:
        status = collect_status(scenario)
        stats = status.stats
        mtm = _compute_mark_to_market(status.paths.db_file, initial_bankroll)

        row = {
            "collected_at_utc": timestamp,
            "scenario": scenario,
            "running": int(status.running),
            "pid": status.pid if status.pid is not None else "",
            "uptime": status.uptime,
            "trades": stats.trades if stats.trades is not None else "",
            "open_positions": stats.open_positions if stats.open_positions is not None else "",
            "wins": stats.wins if stats.wins is not None else "",
            "losses": stats.losses if stats.losses is not None else "",
            "win_rate_pct": round(stats.win_rate, 6) if stats.win_rate is not None else "",
            "pnl_reported": round(stats.pnl, 6) if stats.pnl is not None else "",
            "forecasts": stats.forecasts if stats.forecasts is not None else "",
            "market_prices": stats.market_prices if stats.market_prices is not None else "",
            "markets": stats.markets if stats.markets is not None else "",
            "forecast_age_min": round(stats.forecast_age_minutes, 6)
            if stats.forecast_age_minutes is not None
            else "",
            "price_age_min": round(stats.price_age_minutes, 6)
            if stats.price_age_minutes is not None
            else "",
            "forecast_ts": stats.last_forecast_ts or "",
            "price_ts": stats.last_market_price_ts or "",
            "open_cost": mtm["open_cost"],
            "mark_value": mtm["mark_value"],
            "unrealized_pnl": mtm["unrealized_pnl"],
            "realized_pnl": mtm["realized_pnl"],
            "equity_now": mtm["equity_now"],
            "missing_prices": mtm["missing_prices"],
            "last_log_line": status.last_log_line,
        }
        rows.append(row)

        total_unrealized += float(mtm["unrealized_pnl"])
        total_realized += float(mtm["realized_pnl"])
        total_equity += float(mtm["equity_now"])
        total_open += int(stats.open_positions or 0)
        total_trades += int(stats.trades or 0)
        total_missing_prices += int(mtm["missing_prices"])

    if include_total:
        rows.append(
            {
                "collected_at_utc": timestamp,
                "scenario": "TOTAL",
                "running": "",
                "pid": "",
                "uptime": "",
                "trades": total_trades,
                "open_positions": total_open,
                "wins": "",
                "losses": "",
                "win_rate_pct": "",
                "pnl_reported": total_realized,
                "forecasts": "",
                "market_prices": "",
                "markets": "",
                "forecast_age_min": "",
                "price_age_min": "",
                "forecast_ts": "",
                "price_ts": "",
                "open_cost": "",
                "mark_value": "",
                "unrealized_pnl": round(total_unrealized, 6),
                "realized_pnl": round(total_realized, 6),
                "equity_now": round(total_equity, 6),
                "missing_prices": total_missing_prices,
                "last_log_line": "",
            }
        )

    return rows


def append_snapshot_rows(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_brief(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        scenario = row["scenario"]
        if scenario == "TOTAL":
            print(
                f"TOTAL unrealized={row['unrealized_pnl']} realized={row['realized_pnl']} "
                f"equity={row['equity_now']}"
            )
        else:
            print(
                f"{scenario}: open={row['open_positions']} unrealized={row['unrealized_pnl']} "
                f"equity={row['equity_now']}"
            )


def run_collector(
    output_path: Path,
    interval_seconds: int = 1800,
    iterations: int | None = None,
    initial_bankroll: float = 1000.0,
    include_total: bool = True,
    print_stdout: bool = True,
) -> int:
    count = 0
    while True:
        rows = collect_snapshot_rows(
            initial_bankroll=initial_bankroll,
            include_total=include_total,
        )
        append_snapshot_rows(output_path, rows)
        count += 1

        if print_stdout:
            print(f"\n[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] snapshot saved: {output_path}")
            _print_brief(rows)

        if iterations is not None and count >= iterations:
            return 0
        time.sleep(max(interval_seconds, 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect periodic scenario snapshots (status + mark-to-market) into CSV."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=1800,
        help="Seconds between snapshots in loop mode (default: 1800).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations to run (default: infinite).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one snapshot and exit.",
    )
    parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=1000.0,
        help="Initial bankroll per scenario used for equity_now calculation.",
    )
    parser.add_argument(
        "--no-total",
        action="store_true",
        help="Do not append TOTAL aggregate row.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print per-snapshot summary to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output).expanduser()

    iterations = args.iterations
    if args.once:
        iterations = 1

    try:
        return run_collector(
            output_path=output_path,
            interval_seconds=args.interval_seconds,
            iterations=iterations,
            initial_bankroll=args.initial_bankroll,
            include_total=not args.no_total,
            print_stdout=not args.quiet,
        )
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

