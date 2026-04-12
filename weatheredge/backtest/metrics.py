from __future__ import annotations
import math
import numpy as np

def sharpe_ratio(pnl_series: list[float], risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio from a daily PnL series."""
    if len(pnl_series) < 2:
        return 0.0
    excess = [p - risk_free_rate for p in pnl_series]
    mean = np.mean(excess)
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(mean / std * math.sqrt(252))  # Annualize

def sortino_ratio(pnl_series: list[float], risk_free_rate: float = 0.0) -> float:
    """Sortino ratio (downside deviation only)."""
    if len(pnl_series) < 2:
        return 0.0
    excess = [p - risk_free_rate for p in pnl_series]
    mean = np.mean(excess)
    downside = [min(0, e) for e in excess]
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return float(mean / downside_std * math.sqrt(252))

def max_drawdown(pnl_series: list[float]) -> float:
    """Maximum drawdown as a fraction of peak cumulative PnL."""
    if not pnl_series:
        return 0.0
    cumulative = np.cumsum(pnl_series)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    peak = np.max(running_max)
    if peak <= 0:
        return 0.0
    return float(np.max(drawdowns) / peak)
