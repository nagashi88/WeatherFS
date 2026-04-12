"""Sell NO strategy — buy NO contracts on overpriced YES brackets.

Trades when our model gives <30% probability but the market prices YES at >40c.
This means the market is overpricing the event, so we sell (buy NO).

Historically the highest ROI strategy (84%+ in backtest) with 94% win rate,
though fewer trades than buy-YES strategies.
"""

from weatheredge.agents.base import Signal, MarketOpportunity
from weatheredge.config import Settings


def sell_no_strategy(opportunity: MarketOpportunity, settings: Settings) -> Signal:
    """Sell NO when market overprices YES.

    Criteria:
    - Model probability < 30% (we're confident the event WON'T happen)
    - Market price > 40c (market disagrees — thinks it will)
    - Net edge must exceed aggressive minimum
    - Lead time <= 2 days (short-term forecasts only)
    """
    market_id = opportunity.market_id

    # Only short-term forecasts
    if opportunity.lead_time_days > 2:
        return Signal(
            market_id=market_id, side="no_trade", confidence=0.0,
            agent_name="SellNO", reasoning="Lead time > 2 days",
        )

    # Core filter: model says unlikely (<30%) but market says likely (>40c)
    if opportunity.model_probability >= 0.30:
        return Signal(
            market_id=market_id, side="no_trade", confidence=0.0,
            agent_name="SellNO", reasoning=f"Model prob {opportunity.model_probability:.2f} >= 0.30",
        )

    if opportunity.market_price <= 0.40:
        return Signal(
            market_id=market_id, side="no_trade", confidence=0.0,
            agent_name="SellNO", reasoning=f"Market price {opportunity.market_price:.2f} <= 0.40",
        )

    # Check minimum edge (use NO-side edge since we're buying NO)
    no_edge = opportunity.net_edge_no
    if no_edge < settings.aggressive_min_edge:
        return Signal(
            market_id=market_id, side="no_trade", confidence=0.0,
            agent_name="SellNO", reasoning=f"NO-side edge {no_edge:.2f} too small",
        )

    # Confidence: higher when model is more certain event won't happen
    # model_prob=0.05 → high confidence, model_prob=0.25 → lower confidence
    confidence = min((0.30 - opportunity.model_probability) / 0.25, 1.0)

    return Signal(
        market_id=market_id,
        side="buy_no",
        confidence=confidence,
        agent_name="SellNO",
        reasoning=f"Overpriced YES: model={opportunity.model_probability:.2f}, market={opportunity.market_price:.2f}",
    )
