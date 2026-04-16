from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Signal:
    """Output from a strategy function or agent."""
    market_id: str
    side: str  # 'buy_yes', 'buy_no', 'no_trade'
    confidence: float  # 0.0 to 1.0
    agent_name: str = ""
    reasoning: str = ""

@dataclass
class TradeOrder:
    """Final order to be executed by the paper trader or live executor."""
    market_id: str
    side: str  # 'buy_yes' or 'buy_no'
    price: float
    amount: float
    agent_name: str = ""
    model_probability: float = 0.0
    edge_at_entry: float = 0.0
    city: str = ""
    platform: str = ""
    station: str = ""
    market_date: str = ""
    gaussian_prob: float | None = None
    ensemble_prob: float | None = None
    blended_prob: float | None = None
    raw_model_prob: float | None = None  # Pre-clamp blended; None = not available (legacy/fallback)
    order_type: str = "market"

@dataclass
class MarketOpportunity:
    """A market opportunity with all data needed for strategy evaluation."""
    market_id: str
    platform: str  # 'polymarket' or 'kalshi'
    city: str
    station: str
    threshold: float  # Temperature threshold for the market
    direction: str  # 'above' or 'below'
    market_price: float  # Current YES price (0.01-0.99)
    model_probability: float  # Our estimated probability
    net_edge: float  # model_prob - market_price - fees (YES side)
    net_edge_no: float = 0.0  # (1-model_prob) - (1-market_price) - fees (NO side)
    market_date: str = ""  # YYYY-MM-DD resolution date for date-exposure cap
    ensemble_temps: list[float] | None = None  # 31 ensemble member values
    forecast_temp: float = 0.0
    rmse: float = 2.5
    lead_time_days: int = 1
    bucket_low: float | None = None  # For bucket markets
    bucket_high: float | None = None
    gaussian_prob: float | None = None  # For model agreement check
    ensemble_prob: float | None = None
    blended_prob: float | None = None
    timesfm_prob: float | None = None  # TimesFM quantile-based probability
    timesfm_error_adj: float | None = None  # TimesFM NWP error correction (degrees)
    volume: float | None = None  # Market volume for slippage modeling (US-205)
    raw_model_prob: float | None = None  # Pre-clamp blended (audit only, not used for decisions)
