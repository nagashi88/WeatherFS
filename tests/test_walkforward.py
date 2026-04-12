"""Tests for walk-forward validation utilities."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from weatheredge.backtesting.backtester import SimulatedTrade
from weatheredge.backtesting.walkforward import (
    load_trades_csv,
    save_walkforward_csv,
    summarize_walkforward,
    walkforward_evaluate,
)


def _make_trades(days: int = 120, trades_per_day: int = 2) -> list[SimulatedTrade]:
    out: list[SimulatedTrade] = []
    start = date(2026, 1, 1)
    for i in range(days):
        d = start + timedelta(days=i)
        for j in range(trades_per_day):
            won = ((i + j) % 2 == 0)
            out.append(
                SimulatedTrade(
                    station="KNYC",
                    trade_date=d,
                    threshold=70.0 + j,
                    direction="above",
                    model_prob=0.60 if won else 0.40,
                    market_proxy=0.50,
                    side="buy_yes",
                    entry_price=0.50,
                    won=won,
                    pnl=1.0 if won else -1.0,
                )
            )
    return out


def test_walkforward_generates_windows():
    trades = _make_trades(days=150, trades_per_day=2)
    windows = walkforward_evaluate(
        trades,
        train_days=60,
        test_days=30,
        step_days=30,
        min_test_trades=10,
    )
    assert len(windows) >= 2
    assert all(w.total_trades > 0 for w in windows)


def test_walkforward_invalid_params_raise():
    with pytest.raises(ValueError):
        walkforward_evaluate(_make_trades(), train_days=0)
    with pytest.raises(ValueError):
        walkforward_evaluate(_make_trades(), test_days=0)
    with pytest.raises(ValueError):
        walkforward_evaluate(_make_trades(), step_days=0)


def test_walkforward_summary():
    windows = walkforward_evaluate(
        _make_trades(days=120, trades_per_day=1),
        train_days=30,
        test_days=30,
        step_days=30,
        min_test_trades=5,
    )
    summary = summarize_walkforward(windows)
    assert summary.windows == len(windows)
    assert summary.total_test_trades == sum(w.total_trades for w in windows)


def test_save_and_load_roundtrip(tmp_path):
    # Create minimal input trades CSV manually
    input_csv = tmp_path / "trades.csv"
    input_csv.write_text(
        "station,date,threshold,direction,model_prob,market_proxy,side,entry_price,won,pnl\n"
        "KNYC,2026-01-01,70,above,0.60,0.50,buy_yes,0.50,True,1.0\n"
        "KNYC,2026-01-02,70,above,0.40,0.50,buy_yes,0.50,False,-1.0\n"
    )
    trades = load_trades_csv(input_csv)
    assert len(trades) == 2

    windows = walkforward_evaluate(
        trades,
        train_days=1,
        test_days=1,
        step_days=1,
        min_test_trades=1,
    )
    out_csv = tmp_path / "walkforward.csv"
    save_walkforward_csv(windows, out_csv)
    assert out_csv.exists()
    lines = out_csv.read_text().strip().splitlines()
    assert len(lines) >= 2
