"""Walk-forward validation utilities for backtest trades."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import median
from typing import Sequence

from weatheredge.backtesting.backtester import SimulatedTrade, TradingBacktester


@dataclass
class WalkForwardWindow:
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    total_trades: int
    win_rate: float
    net_pnl: float
    roi_pct: float
    sharpe_ratio: float
    max_drawdown: float


@dataclass
class WalkForwardSummary:
    windows: int
    total_test_trades: int
    median_win_rate: float
    median_roi_pct: float
    median_sharpe: float
    median_max_drawdown: float
    aggregate_net_pnl: float


def load_trades_csv(path: str | Path) -> list[SimulatedTrade]:
    """Load simulated trades from backtester CSV output."""
    trades: list[SimulatedTrade] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(
                SimulatedTrade(
                    station=str(row["station"]),
                    trade_date=date.fromisoformat(str(row["date"])),
                    threshold=float(row["threshold"]),
                    direction=str(row["direction"]),
                    model_prob=float(row["model_prob"]),
                    market_proxy=float(row["market_proxy"]),
                    side=str(row["side"]),
                    entry_price=float(row["entry_price"]),
                    won=str(row["won"]).strip().lower() in {"true", "1", "yes"},
                    pnl=float(row["pnl"]),
                )
            )
    return trades


def walkforward_evaluate(
    trades: list[SimulatedTrade],
    train_days: int = 60,
    test_days: int = 30,
    step_days: int = 30,
    min_test_trades: int = 25,
) -> list[WalkForwardWindow]:
    """Compute rolling walk-forward windows over historical trades."""
    if train_days <= 0 or test_days <= 0 or step_days <= 0:
        raise ValueError("train_days, test_days and step_days must be > 0")
    if not trades:
        return []

    ordered = sorted(trades, key=lambda t: t.trade_date)
    first_day = ordered[0].trade_date
    last_day = ordered[-1].trade_date

    bt = TradingBacktester()
    windows: list[WalkForwardWindow] = []
    cursor = first_day
    while True:
        train_start = cursor
        train_end = train_start + timedelta(days=train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days - 1)

        if test_start > last_day:
            break

        test_slice = [t for t in ordered if test_start <= t.trade_date <= test_end]
        if len(test_slice) >= min_test_trades:
            metrics = bt.calculate_metrics(test_slice)
            windows.append(
                WalkForwardWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    total_trades=metrics.total_trades,
                    win_rate=metrics.win_rate,
                    net_pnl=metrics.net_pnl,
                    roi_pct=metrics.roi_pct,
                    sharpe_ratio=metrics.sharpe_ratio,
                    max_drawdown=metrics.max_drawdown,
                )
            )

        cursor = cursor + timedelta(days=step_days)
        if cursor > last_day:
            break

    return windows


def summarize_walkforward(windows: list[WalkForwardWindow]) -> WalkForwardSummary:
    if not windows:
        return WalkForwardSummary(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return WalkForwardSummary(
        windows=len(windows),
        total_test_trades=sum(w.total_trades for w in windows),
        median_win_rate=median(w.win_rate for w in windows),
        median_roi_pct=median(w.roi_pct for w in windows),
        median_sharpe=median(w.sharpe_ratio for w in windows),
        median_max_drawdown=median(w.max_drawdown for w in windows),
        aggregate_net_pnl=sum(w.net_pnl for w in windows),
    )


def save_walkforward_csv(windows: list[WalkForwardWindow], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "total_trades",
                "win_rate",
                "net_pnl",
                "roi_pct",
                "sharpe_ratio",
                "max_drawdown",
            ]
        )
        for w in windows:
            writer.writerow(
                [
                    w.train_start.isoformat(),
                    w.train_end.isoformat(),
                    w.test_start.isoformat(),
                    w.test_end.isoformat(),
                    w.total_trades,
                    f"{w.win_rate:.6f}",
                    f"{w.net_pnl:.6f}",
                    f"{w.roi_pct:.6f}",
                    f"{w.sharpe_ratio:.6f}",
                    f"{w.max_drawdown:.6f}",
                ]
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run walk-forward validation on backtest trades CSV.")
    parser.add_argument("--input", required=True, help="Input trades CSV path.")
    parser.add_argument("--output", required=True, help="Output walk-forward CSV path.")
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--min-test-trades", type=int, default=25)
    args = parser.parse_args(argv)

    trades = load_trades_csv(args.input)
    windows = walkforward_evaluate(
        trades=trades,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        min_test_trades=args.min_test_trades,
    )
    save_walkforward_csv(windows, args.output)
    summary = summarize_walkforward(windows)

    print(
        f"walk-forward windows={summary.windows} "
        f"trades={summary.total_test_trades} "
        f"median_wr={summary.median_win_rate:.2%} "
        f"median_roi={summary.median_roi_pct:.2f}% "
        f"median_sharpe={summary.median_sharpe:.2f} "
        f"net_pnl={summary.aggregate_net_pnl:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
