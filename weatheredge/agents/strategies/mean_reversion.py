import numpy as np
from weatheredge.agents.base import Signal
from weatheredge.config import Settings

def mean_reversion_strategy(
    market_id: str,
    market_prices: list[float],
    settings: Settings,
) -> Signal:
    """
    Mean Reversion "Buy the Overreaction" strategy.
    Price-based, no weather opinion. Looks for >3sigma deviations
    from 7-day rolling mean.
    """
    lookback = settings.mean_reversion_lookback_hours
    min_sigma = settings.mean_reversion_min_sigma

    if len(market_prices) < max(lookback, 20):  # Need enough data
        return Signal(market_id=market_id, side="no_trade", confidence=0.0, agent_name="MeanReversion", reasoning="Insufficient price history")

    recent = market_prices[-lookback:]
    mean = float(np.mean(recent))
    std = float(np.std(recent, ddof=1))  # BugFix M2: sample std, not population
    current = market_prices[-1]

    if std == 0:
        return Signal(market_id=market_id, side="no_trade", confidence=0.0, agent_name="MeanReversion", reasoning="Zero volatility")

    deviation = (current - mean) / std

    if abs(deviation) < min_sigma:
        return Signal(market_id=market_id, side="no_trade", confidence=0.0, agent_name="MeanReversion", reasoning=f"Deviation {deviation:.2f}sigma < {min_sigma}")

    # Price crashed below mean -> buy YES (expect reversion up)
    # Price spiked above mean -> buy NO (expect reversion down)
    side = "buy_yes" if deviation < 0 else "buy_no"
    confidence = min(abs(deviation) / 5.0, 1.0)

    return Signal(
        market_id=market_id,
        side=side,
        confidence=confidence,
        agent_name="MeanReversion",
        reasoning=f"Deviation {deviation:.2f}sigma, expecting reversion"
    )
