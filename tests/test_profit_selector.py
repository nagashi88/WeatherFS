from __future__ import annotations

from weatheredge.backtesting.backtester import BacktestMetrics
from weatheredge.backtesting.walkforward import WalkForwardSummary
from weatheredge.backtesting.profit_selector import (
    ProfitConstraints,
    profit_score,
    validate_constraints,
)


def test_profit_score_prefers_higher_pnl_and_lower_drawdown():
    m_a = BacktestMetrics(net_pnl=200.0)
    wf_a = WalkForwardSummary(
        windows=4,
        total_test_trades=200,
        median_win_rate=0.56,
        median_roi_pct=3.0,
        median_sharpe=0.60,
        median_max_drawdown=0.12,
        aggregate_net_pnl=180.0,
    )

    m_b = BacktestMetrics(net_pnl=120.0)
    wf_b = WalkForwardSummary(
        windows=4,
        total_test_trades=200,
        median_win_rate=0.53,
        median_roi_pct=2.0,
        median_sharpe=0.30,
        median_max_drawdown=0.25,
        aggregate_net_pnl=100.0,
    )

    assert profit_score(m_a, wf_a) > profit_score(m_b, wf_b)


def test_validate_constraints_reports_multiple_failures():
    metrics = BacktestMetrics(total_trades=40, net_pnl=-10.0)
    wf = WalkForwardSummary(
        windows=1,
        total_test_trades=20,
        median_win_rate=0.49,
        median_roi_pct=-1.0,
        median_sharpe=-0.20,
        median_max_drawdown=0.50,
        aggregate_net_pnl=-5.0,
    )
    constraints = ProfitConstraints(
        min_total_trades=100,
        min_walkforward_windows=3,
        min_median_win_rate=0.52,
        min_median_sharpe=0.10,
        min_aggregate_net_pnl=0.0,
        max_median_max_drawdown=0.30,
    )

    passed, reasons = validate_constraints(metrics, wf, constraints)

    assert passed is False
    assert len(reasons) >= 5
    assert any("total_trades" in r for r in reasons)
    assert any("wf_windows" in r for r in reasons)
    assert any("median_wr" in r for r in reasons)
    assert any("median_sharpe" in r for r in reasons)
    assert any("wf_net_pnl" in r for r in reasons)
    assert any("median_max_dd" in r for r in reasons)
