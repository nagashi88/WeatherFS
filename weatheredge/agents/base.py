from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime

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
    cash_capped: bool = False  # PortfolioManager sets True when amount was reduced to fit cash
    forecast_timestamp: datetime | None = None
    market_price_timestamp: datetime | None = None
    signal_ids: list[int] = field(default_factory=list)

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
    market_type: str = "daily_high"
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
    # Structural invalidation: daily_high already exceeded bucket_high via observed METAR.
    # YES resolution physically impossible — exit_manager uses this to force close buy_yes.
    bucket_invalidated_for_yes: bool = False
    bucket_metar_high: float | None = None  # Observed daily_high °F when invalidation detected
    metar_daily_high: float | None = None
    metar_daily_low: float | None = None
    # Per-market price freshness: timestamp of the price row used to build this
    # opportunity. Used by the per-market freshness gate so an illiquid market
    # with a stale row is skipped even when the global newest-row age is fresh.
    market_price_timestamp: datetime | None = None
    # Forecast row timestamp used for this decision. Execution must prove this
    # timestamp is not newer than the market price snapshot being traded.
    forecast_timestamp: datetime | None = None
