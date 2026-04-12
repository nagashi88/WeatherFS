"""Profit-first model/parameter selector for WeatherEdge backtesting.

Runs a small grid search over trading parameters, evaluates each candidate on:
- aggregate backtest metrics (net PnL, win rate, Sharpe, drawdown)
- walk-forward robustness (median Sharpe/WR/DD and aggregate net PnL)

Then applies hard risk gates and ranks surviving candidates by a profit score.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import aiohttp

from weatheredge.backtesting.backtester import (
    BacktestMetrics,
    HistoricalDataFetcher,
    SimulatedTrade,
    TradingBacktester,
    STATION_COORDS,
)
from weatheredge.backtesting.market_prices import HistoricalMarketPriceBook
from weatheredge.backtesting.walkforward import (
    WalkForwardSummary,
    summarize_walkforward,
    walkforward_evaluate,
)
from weatheredge.backtesting.simulation_db import build_simulation_db

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateConfig:
    """Parameter set for one backtest candidate."""

    name: str
    min_edge: float
    max_edge: float = 0.50
    min_price: float = 0.10
    bet_size: float = 10.0
    fee_rate: float = 0.02
    max_trades_per_station_day: int = 3
    require_real_market_price: bool = True


@dataclass(frozen=True)
class ProfitConstraints:
    """Risk/robustness gates candidates must satisfy."""

    min_total_trades: int = 50
    min_walkforward_windows: int = 1
    min_median_win_rate: float = 0.55
    min_median_sharpe: float = 0.50
    min_aggregate_net_pnl: float = 0.0
    # Backtester/walk-forward max_drawdown is reported in dollars (not fraction).
    max_median_max_drawdown: float = 25.0


@dataclass
class CandidateResult:
    """Full evaluation output for one candidate."""

    config: CandidateConfig
    metrics: BacktestMetrics
    walkforward: WalkForwardSummary
    score: float
    passed: bool
    reasons: list[str]
    trades: list[SimulatedTrade]


def profit_score(metrics: BacktestMetrics, wf: WalkForwardSummary) -> float:
    """Composite objective emphasizing robust profitability.

    Uses walk-forward quantities as primary signal, backtest aggregate as secondary.
    """
    return (
        wf.aggregate_net_pnl
        + 0.25 * metrics.net_pnl
        + 80.0 * (wf.median_win_rate - 0.50)
        + 20.0 * wf.median_sharpe
        - 1.5 * wf.median_max_drawdown
    )


def validate_constraints(
    metrics: BacktestMetrics,
    wf: WalkForwardSummary,
    constraints: ProfitConstraints,
) -> tuple[bool, list[str]]:
    """Return pass/fail and violation reasons."""
    reasons: list[str] = []

    if metrics.total_trades < constraints.min_total_trades:
        reasons.append(
            f"total_trades={metrics.total_trades} < {constraints.min_total_trades}"
        )
    if wf.windows < constraints.min_walkforward_windows:
        reasons.append(
            f"wf_windows={wf.windows} < {constraints.min_walkforward_windows}"
        )
    if wf.median_win_rate < constraints.min_median_win_rate:
        reasons.append(
            f"median_wr={wf.median_win_rate:.2%} < {constraints.min_median_win_rate:.2%}"
        )
    if wf.median_sharpe < constraints.min_median_sharpe:
        reasons.append(
            f"median_sharpe={wf.median_sharpe:.2f} < {constraints.min_median_sharpe:.2f}"
        )
    if wf.aggregate_net_pnl < constraints.min_aggregate_net_pnl:
        reasons.append(
            f"wf_net_pnl={wf.aggregate_net_pnl:.2f} < {constraints.min_aggregate_net_pnl:.2f}"
        )
    if wf.median_max_drawdown > constraints.max_median_max_drawdown:
        reasons.append(
            f"median_max_dd={wf.median_max_drawdown:.2f} > {constraints.max_median_max_drawdown:.2f}"
        )

    return (len(reasons) == 0, reasons)


async def _fetch_records_by_station(
    stations: list[str],
    months_back: int,
) -> dict[str, list]:
    """Fetch forecast/observation records once and reuse across candidates."""
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=months_back * 30)

    fetcher = HistoricalDataFetcher()
    warm_backtester = TradingBacktester()
    records_by_station: dict[str, list] = {}

    async with aiohttp.ClientSession() as session:
        for station in stations:
            forecasts = await fetcher.fetch_archive_forecasts(station, start, end, session)
            observations = await fetcher.fetch_iem_observations(station, start, end, session)
            records = warm_backtester.build_daily_records(forecasts, observations, station)
            if len(records) >= 60:
                records_by_station[station] = records
            else:
                log.warning(
                    "Skipping %s: only %d historical rows (need >=60)",
                    station,
                    len(records),
                )

    return records_by_station


def evaluate_candidate(
    candidate: CandidateConfig,
    records_by_station: dict[str, list],
    constraints: ProfitConstraints,
    market_price_book: HistoricalMarketPriceBook | None,
    train_days: int,
    test_days: int,
    step_days: int,
    min_test_trades: int,
) -> CandidateResult:
    """Backtest + walk-forward + gates for one candidate."""
    bt = TradingBacktester(
        min_edge=candidate.min_edge,
        max_edge=candidate.max_edge,
        min_price=candidate.min_price,
        bet_size=candidate.bet_size,
        fee_rate=candidate.fee_rate,
        max_trades_per_station_day=candidate.max_trades_per_station_day,
        require_real_market_price=(
            candidate.require_real_market_price and market_price_book is not None
        ),
    )

    trades: list[SimulatedTrade] = []
    for records in records_by_station.values():
        trades.extend(
            bt.simulate_trading(
                records=records,
                thresholds=None,
                market_price_book=market_price_book,
            )
        )

    metrics = bt.calculate_metrics(trades)
    wf_windows = walkforward_evaluate(
        trades=trades,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        min_test_trades=min_test_trades,
    )
    wf = summarize_walkforward(wf_windows)
    passed, reasons = validate_constraints(metrics, wf, constraints)

    score = profit_score(metrics, wf) if passed else float("-inf")
    return CandidateResult(
        config=candidate,
        metrics=metrics,
        walkforward=wf,
        score=score,
        passed=passed,
        reasons=reasons,
        trades=trades,
    )


def _parse_float_grid(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        piece = part.strip()
        if not piece:
            continue
        values.append(float(piece))
    if not values:
        raise ValueError("empty float grid")
    return values


def _write_leaderboard_csv(results: list[CandidateResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name",
                "passed",
                "score",
                "min_edge",
                "total_trades",
                "net_pnl",
                "win_rate",
                "sharpe",
                "max_drawdown",
                "wf_windows",
                "wf_median_wr",
                "wf_median_sharpe",
                "wf_median_max_dd",
                "wf_aggregate_net_pnl",
                "reasons",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.config.name,
                    r.passed,
                    f"{r.score:.6f}" if r.score != float("-inf") else "-inf",
                    f"{r.config.min_edge:.4f}",
                    r.metrics.total_trades,
                    f"{r.metrics.net_pnl:.6f}",
                    f"{r.metrics.win_rate:.6f}",
                    f"{r.metrics.sharpe_ratio:.6f}",
                    f"{r.metrics.max_drawdown:.6f}",
                    r.walkforward.windows,
                    f"{r.walkforward.median_win_rate:.6f}",
                    f"{r.walkforward.median_sharpe:.6f}",
                    f"{r.walkforward.median_max_drawdown:.6f}",
                    f"{r.walkforward.aggregate_net_pnl:.6f}",
                    "; ".join(r.reasons),
                ]
            )


def _write_best_trades_csv(best: CandidateResult, output_dir: Path) -> Path:
    path = output_dir / f"best_{best.config.name}_trades.csv"
    bt = TradingBacktester(
        min_edge=best.config.min_edge,
        max_edge=best.config.max_edge,
        min_price=best.config.min_price,
        bet_size=best.config.bet_size,
        fee_rate=best.config.fee_rate,
        max_trades_per_station_day=best.config.max_trades_per_station_day,
    )
    bt.save_trades_csv(best.trades, path)
    return path


async def run_selector(
    stations: list[str],
    months_back: int,
    min_edge_grid: list[float],
    constraints: ProfitConstraints,
    market_prices_csv: str | None,
    train_days: int,
    test_days: int,
    step_days: int,
    min_test_trades: int,
    output_dir: Path,
    simulation_db: str | None,
) -> int:
    records_by_station = await _fetch_records_by_station(stations, months_back)
    if not records_by_station:
        print("No stations with enough data (>=60 rows).")
        return 1

    market_price_book: HistoricalMarketPriceBook | None = None
    if market_prices_csv:
        market_price_book = HistoricalMarketPriceBook.from_csv(market_prices_csv)

    candidates = [
        CandidateConfig(name=f"edge_{edge:.3f}".replace(".", "p"), min_edge=edge)
        for edge in min_edge_grid
    ]

    results: list[CandidateResult] = []
    for candidate in candidates:
        result = evaluate_candidate(
            candidate=candidate,
            records_by_station=records_by_station,
            constraints=constraints,
            market_price_book=market_price_book,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            min_test_trades=min_test_trades,
        )
        results.append(result)

    results.sort(key=lambda r: r.score, reverse=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "profit_selector_leaderboard.csv"
    _write_leaderboard_csv(results, leaderboard_path)

    serializable = [
        {
            "config": asdict(r.config),
            "passed": r.passed,
            "score": None if r.score == float("-inf") else r.score,
            "reasons": r.reasons,
            "metrics": {
                "total_trades": r.metrics.total_trades,
                "net_pnl": r.metrics.net_pnl,
                "win_rate": r.metrics.win_rate,
                "sharpe_ratio": r.metrics.sharpe_ratio,
                "max_drawdown": r.metrics.max_drawdown,
            },
            "walkforward": asdict(r.walkforward),
        }
        for r in results
    ]
    (output_dir / "profit_selector_results.json").write_text(
        json.dumps(serializable, indent=2)
    )

    passed = [r for r in results if r.passed]
    if not passed:
        print("No candidate passed constraints.")
        print(f"Leaderboard: {leaderboard_path}")
        return 2

    best = passed[0]
    trades_path = _write_best_trades_csv(best, output_dir)
    print(
        "Best candidate: "
        f"{best.config.name} min_edge={best.config.min_edge:.3f} "
        f"net_pnl={best.metrics.net_pnl:.2f} "
        f"wf_net_pnl={best.walkforward.aggregate_net_pnl:.2f} "
        f"wf_median_sharpe={best.walkforward.median_sharpe:.2f} "
        f"wf_median_dd={best.walkforward.median_max_drawdown:.2f}"
    )
    print(f"Leaderboard: {leaderboard_path}")
    print(f"Best trades CSV: {trades_path}")

    if simulation_db:
        sim_report = build_simulation_db(
            output_db=simulation_db,
            trades_csv=trades_path,
            market_prices_csv=market_prices_csv,
        )
        print(
            "Simulation DB: "
            f"{sim_report.output_db} "
            f"(trades={sim_report.trade_rows}, prices={sim_report.market_rows}, "
            f"bootstrap={sim_report.bootstrap_rows})"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profit-first candidate selector (backtest + walk-forward)."
    )
    parser.add_argument(
        "--stations",
        default=",".join(sorted(STATION_COORDS.keys())),
        help="Comma-separated stations (default: all known stations).",
    )
    parser.add_argument("--months-back", type=int, default=6)
    parser.add_argument(
        "--min-edge-grid",
        default="0.08,0.10,0.12,0.14,0.16",
        help="Comma-separated min_edge grid values.",
    )
    parser.add_argument("--market-prices-csv", default=None)
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--min-test-trades", type=int, default=25)
    parser.add_argument("--min-total-trades", type=int, default=50)
    parser.add_argument("--min-wf-windows", type=int, default=1)
    parser.add_argument("--min-median-wr", type=float, default=0.55)
    parser.add_argument("--min-median-sharpe", type=float, default=0.50)
    parser.add_argument("--min-wf-net-pnl", type=float, default=0.0)
    parser.add_argument(
        "--max-median-dd",
        type=float,
        default=25.0,
        help="Max allowed median walk-forward drawdown in dollars.",
    )
    parser.add_argument("--output-dir", default="logs/profit_selector")
    parser.add_argument(
        "--simulation-db",
        default=None,
        help="Optional output SQLite path to build simulation DB for the best candidate.",
    )
    args = parser.parse_args(argv)

    stations = [s.strip() for s in args.stations.split(",") if s.strip()]
    min_edge_grid = _parse_float_grid(args.min_edge_grid)
    constraints = ProfitConstraints(
        min_total_trades=args.min_total_trades,
        min_walkforward_windows=args.min_wf_windows,
        min_median_win_rate=args.min_median_wr,
        min_median_sharpe=args.min_median_sharpe,
        min_aggregate_net_pnl=args.min_wf_net_pnl,
        max_median_max_drawdown=args.max_median_dd,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    return asyncio.run(
        run_selector(
            stations=stations,
            months_back=args.months_back,
            min_edge_grid=min_edge_grid,
            constraints=constraints,
            market_prices_csv=args.market_prices_csv,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            min_test_trades=args.min_test_trades,
            output_dir=Path(args.output_dir),
            simulation_db=args.simulation_db,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
