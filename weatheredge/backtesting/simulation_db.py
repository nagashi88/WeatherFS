"""Build a SQLite simulation DB from backtest trades and market prices.

The resulting DB is meant to be consumed by models/agents to run realistic
profitability simulations over historical data with point-in-time prices.
"""

from __future__ import annotations

import argparse
import csv
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence


@dataclass(frozen=True)
class SimulationDbReport:
    market_rows: int
    trade_rows: int
    daily_rows: int
    station_rows: int
    bootstrap_rows: int
    output_db: str


def _parse_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_bool(value: str | bool | int | None) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return 1 if value else 0
    if value is None:
        return 0
    raw = str(value).strip().lower()
    if raw in {"1", "true", "t", "yes", "y"}:
        return 1
    return 0


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    w = pos - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def _bootstrap_projection(
    pnls: list[float],
    horizon_trades: int,
    simulations: int,
    rng: random.Random,
) -> dict[str, float]:
    totals: list[float] = []
    for _ in range(simulations):
        total = 0.0
        for _ in range(horizon_trades):
            total += rng.choice(pnls)
        totals.append(total)

    totals.sort()
    return {
        "mean_pnl": mean(totals),
        "std_pnl": pstdev(totals) if len(totals) > 1 else 0.0,
        "p05": _quantile(totals, 0.05),
        "p25": _quantile(totals, 0.25),
        "p50": _quantile(totals, 0.50),
        "p75": _quantile(totals, 0.75),
        "p95": _quantile(totals, 0.95),
    }


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS market_prices_hist (
            station TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            threshold REAL NOT NULL,
            direction TEXT NOT NULL,
            yes_price REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sim_trades (
            station TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            threshold REAL NOT NULL,
            direction TEXT NOT NULL,
            model_prob REAL NOT NULL,
            market_proxy REAL NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            won INTEGER NOT NULL,
            pnl REAL NOT NULL,
            edge REAL NOT NULL,
            brier REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS daily_pnl (
            trade_date TEXT PRIMARY KEY,
            trades INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            win_rate REAL NOT NULL,
            net_pnl REAL NOT NULL,
            avg_edge REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS station_pnl (
            station TEXT PRIMARY KEY,
            trades INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            win_rate REAL NOT NULL,
            net_pnl REAL NOT NULL,
            avg_edge REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bootstrap_pnl (
            horizon_trades INTEGER NOT NULL,
            simulations INTEGER NOT NULL,
            mean_pnl REAL NOT NULL,
            std_pnl REAL NOT NULL,
            p05 REAL NOT NULL,
            p25 REAL NOT NULL,
            p50 REAL NOT NULL,
            p75 REAL NOT NULL,
            p95 REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_market_prices_key
            ON market_prices_hist(station, trade_date, threshold, direction);
        CREATE INDEX IF NOT EXISTS idx_sim_trades_date
            ON sim_trades(trade_date);
        CREATE INDEX IF NOT EXISTS idx_sim_trades_station
            ON sim_trades(station);
        """
    )


def _load_market_rows(csv_path: Path | None) -> list[tuple[str, str, float, str, float]]:
    if csv_path is None:
        return []

    rows: list[tuple[str, str, float, str, float]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station = str(row.get("station", "")).strip().upper()
            trade_date = str(row.get("date", "")).strip()
            direction = str(row.get("direction", "")).strip().lower()
            threshold = _parse_float(row.get("threshold"))
            yes_price = _parse_float(row.get("yes_price"))
            if (
                not station
                or not trade_date
                or direction not in {"above", "below"}
                or threshold is None
                or yes_price is None
            ):
                continue
            rows.append((station, trade_date, round(threshold, 2), direction, yes_price))
    return rows


def _load_trade_rows(
    csv_path: Path,
) -> tuple[
    list[tuple[str, str, float, str, float, float, str, float, int, float, float, float]],
    list[float],
]:
    rows: list[tuple[str, str, float, str, float, float, str, float, int, float, float, float]] = []
    pnls: list[float] = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station = str(row.get("station", "")).strip().upper()
            trade_date = str(row.get("date", "")).strip()
            direction = str(row.get("direction", "")).strip().lower()
            side = str(row.get("side", "")).strip().lower()

            threshold = _parse_float(row.get("threshold"))
            model_prob = _parse_float(row.get("model_prob"))
            market_proxy = _parse_float(row.get("market_proxy"))
            entry_price = _parse_float(row.get("entry_price"))
            pnl = _parse_float(row.get("pnl"))
            won = _parse_bool(row.get("won"))

            if (
                not station
                or not trade_date
                or direction not in {"above", "below"}
                or side not in {"buy_yes", "buy_no"}
                or threshold is None
                or model_prob is None
                or market_proxy is None
                or entry_price is None
                or pnl is None
            ):
                continue

            edge = abs(model_prob - market_proxy)
            side_win_prob = model_prob if side == "buy_yes" else (1.0 - model_prob)
            brier = (side_win_prob - float(won)) ** 2
            rows.append(
                (
                    station,
                    trade_date,
                    round(threshold, 2),
                    direction,
                    model_prob,
                    market_proxy,
                    side,
                    entry_price,
                    won,
                    pnl,
                    edge,
                    brier,
                )
            )
            pnls.append(pnl)
    return rows, pnls


def build_simulation_db(
    output_db: str | Path,
    trades_csv: str | Path,
    market_prices_csv: str | Path | None = None,
    bootstrap_horizons: list[int] | None = None,
    bootstrap_sims: int = 3000,
    seed: int = 42,
) -> SimulationDbReport:
    """Build a SQLite DB for downstream model-driven performance simulation."""
    out = Path(output_db)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    trades_path = Path(trades_csv)
    market_path = Path(market_prices_csv) if market_prices_csv is not None else None
    if bootstrap_horizons is None:
        bootstrap_horizons = [20, 50, 100, 200]

    market_rows = _load_market_rows(market_path)
    trade_rows, pnls = _load_trade_rows(trades_path)

    with sqlite3.connect(out) as conn:
        _create_schema(conn)

        conn.executemany(
            "INSERT INTO market_prices_hist(station, trade_date, threshold, direction, yes_price) VALUES (?, ?, ?, ?, ?)",
            market_rows,
        )
        conn.executemany(
            """
            INSERT INTO sim_trades(
                station, trade_date, threshold, direction, model_prob, market_proxy,
                side, entry_price, won, pnl, edge, brier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            trade_rows,
        )

        conn.execute(
            """
            INSERT INTO daily_pnl(trade_date, trades, wins, win_rate, net_pnl, avg_edge)
            SELECT
                trade_date,
                COUNT(*) AS trades,
                SUM(won) AS wins,
                AVG(CAST(won AS REAL)) AS win_rate,
                SUM(pnl) AS net_pnl,
                AVG(edge) AS avg_edge
            FROM sim_trades
            GROUP BY trade_date
            ORDER BY trade_date
            """
        )
        conn.execute(
            """
            INSERT INTO station_pnl(station, trades, wins, win_rate, net_pnl, avg_edge)
            SELECT
                station,
                COUNT(*) AS trades,
                SUM(won) AS wins,
                AVG(CAST(won AS REAL)) AS win_rate,
                SUM(pnl) AS net_pnl,
                AVG(edge) AS avg_edge
            FROM sim_trades
            GROUP BY station
            ORDER BY station
            """
        )

        bootstrap_rows = 0
        if pnls and bootstrap_sims > 0:
            rng = random.Random(seed)
            for horizon in bootstrap_horizons:
                if horizon <= 0:
                    continue
                summary = _bootstrap_projection(
                    pnls=pnls,
                    horizon_trades=horizon,
                    simulations=bootstrap_sims,
                    rng=rng,
                )
                conn.execute(
                    """
                    INSERT INTO bootstrap_pnl(
                        horizon_trades, simulations, mean_pnl, std_pnl,
                        p05, p25, p50, p75, p95
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        horizon,
                        bootstrap_sims,
                        summary["mean_pnl"],
                        summary["std_pnl"],
                        summary["p05"],
                        summary["p25"],
                        summary["p50"],
                        summary["p75"],
                        summary["p95"],
                    ),
                )
                bootstrap_rows += 1

        created_at = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("created_at_utc", created_at),
                ("source_trades_csv", str(trades_path)),
                ("source_market_prices_csv", str(market_path) if market_path else ""),
                ("bootstrap_sims", str(bootstrap_sims)),
                ("bootstrap_seed", str(seed)),
            ],
        )
        conn.commit()

        daily_rows = int(conn.execute("SELECT COUNT(*) FROM daily_pnl").fetchone()[0])
        station_rows = int(conn.execute("SELECT COUNT(*) FROM station_pnl").fetchone()[0])

    return SimulationDbReport(
        market_rows=len(market_rows),
        trade_rows=len(trade_rows),
        daily_rows=daily_rows,
        station_rows=station_rows,
        bootstrap_rows=bootstrap_rows,
        output_db=str(out),
    )


def _parse_int_grid(raw: str) -> list[int]:
    result: list[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        result.append(int(p))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build SQLite simulation DB from backtest trades and prices CSV."
    )
    parser.add_argument("--trades-csv", required=True, help="Backtester trades CSV path.")
    parser.add_argument(
        "--market-prices-csv",
        default=None,
        help="Optional historical market prices CSV path.",
    )
    parser.add_argument(
        "--output-db",
        required=True,
        help="Output SQLite file path.",
    )
    parser.add_argument(
        "--bootstrap-horizons",
        default="20,50,100,200",
        help="Comma-separated horizons in number of trades.",
    )
    parser.add_argument(
        "--bootstrap-sims",
        type=int,
        default=3000,
        help="Number of bootstrap Monte Carlo simulations per horizon.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    report = build_simulation_db(
        output_db=args.output_db,
        trades_csv=args.trades_csv,
        market_prices_csv=args.market_prices_csv,
        bootstrap_horizons=_parse_int_grid(args.bootstrap_horizons),
        bootstrap_sims=args.bootstrap_sims,
        seed=args.seed,
    )
    print(
        "market_rows={market_rows} trade_rows={trade_rows} daily_rows={daily_rows} "
        "station_rows={station_rows} bootstrap_rows={bootstrap_rows} output={output}".format(
            market_rows=report.market_rows,
            trade_rows=report.trade_rows,
            daily_rows=report.daily_rows,
            station_rows=report.station_rows,
            bootstrap_rows=report.bootstrap_rows,
            output=report.output_db,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
