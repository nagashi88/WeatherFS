"""Pydantic settings for WeatherEdge. Reads from .env file."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# PRD v3.1 — Feature flags & operational hypotheses
# ---------------------------------------------------------------------------

MODULES: dict[str, bool] = {
    # Agents
    "agent_mean_reversion": False,   # STRUCTURAL KILL — price spikes = new model run, not noise
    "agent_voting": False,           # DISABLED: use blended forecast + fixed thresholds
    "agent_qualitative": False,      # DISABLED: underspecified, validate later

    # Models
    "timesfm": False,                # DISABLED: weight=0.0, let learning loop promote it
    "student_t_fallback": True,      # KEEP: protects against zero-probability extremes

    # Risk modules
    "vpin": False,                   # STRUCTURAL KILL — no HFT order flow on weather markets
    "exit_pnl_stoploss": False,      # STRUCTURAL KILL — P&L drawdown ≠ risk on prediction markets
    "exit_edge_flip": True,          # KEEP: only rational exit trigger
    "exit_time_decay": True,         # KEEP: force-close near resolution
    "exit_ev_based": False,          # Phase 1: EV-based exit replaces P&L stop-loss

    # Learning
    "dynamic_model_weights": False,  # DISABLED: use fixed weights until 100+ trades
    "dynamic_agent_weights": False,  # DISABLED: agents are off anyway
    "weekly_error_analysis": True,   # KEEP: observational, doesn't affect decisions

    # Execution
    "cross_platform_arb": False,     # Phase 4 feature
    "limit_orders": False,           # Option A €100: use market orders for immediate fill (limit orders sat unfilled on illiquid books)

    # Phase 2 — Burst mode & timing
    "burst_mode": False,
    "market_drift_tracking": False,

    # Phase 1 — Calibration
    "bias_correction": False,
    "rmse_floor_dynamic": False,
    "reliability_tracking": False,

    # Phase 3 — Justified complexity
    "shrinkage_rmse": False,
    "seasonal_agent_weights": False,
    "meta_calibrator": False,

    # Always ON
    "orderbook_logging": True,       # Collect data for Phase 4 execution simulator
    "temporal_guard": True,          # Anti-leakage — NEVER disable
    "volume_filter": True,           # Skip illiquid markets
}

HYPOTHESES: dict[str, float] = {
    "brier_fail_gate": 0.25,
    "ablation_min_improvement": 0.01,
    "convergence_gate_ratio": 3.0,
    "burst_edge_discount": 0.05,
    "kelly_fraction": 0.15,
    "warmup_kelly_fraction": 0.10,
    "min_market_volume": 2000,
    "max_spread_pct": 0.08,
}

# Volume/liquidity filter thresholds (Phase 0)
MIN_MARKET_VOLUME: float = 2000
MAX_SPREAD_PCT: float = 0.08

# Burst mode (Phase 2)
BURST_EDGE_DISCOUNT: float = 0.05
BURST_WINDOW_MINUTES: int = 30

# Convergence gate (Phase 3)
CONVERGENCE_GATE_RATIO: float = 3.0

# TimesFM weight bounds (Phase 3)
TIMESFM_INITIAL_WEIGHT: float = 0.0
TIMESFM_MAX_WEIGHT: float = 0.30

# Seasonal agent weights (Phase 3)
SEASONAL_AGENT_WEIGHTS: dict[str, dict[str, float]] = {
    "summer": {"conservative": 0.35, "aggressive": 0.30, "sellno": 0.35},
    "winter": {"conservative": 0.50, "aggressive": 0.15, "sellno": 0.35},
    "shoulder": {"conservative": 0.40, "aggressive": 0.25, "sellno": 0.35},
}


class PlatformFees(BaseSettings):
    """Fee estimates per platform (Addendum Gap 2)."""

    model_config = {"env_prefix": ""}

    polymarket_fee_rate: float = 0.00
    polymarket_avg_spread: float = 0.03
    kalshi_fee_on_winnings: float = 0.02
    kalshi_avg_spread: float = 0.02

    @property
    def polymarket_round_trip(self) -> float:
        return self.polymarket_fee_rate + self.polymarket_avg_spread / 2

    @property
    def kalshi_round_trip(self) -> float:
        return self.kalshi_fee_on_winnings + self.kalshi_avg_spread / 2


class RiskTierParams:
    """Tiered risk parameters (Addendum Gap 8)."""

    TIERS: dict[int, dict[str, float]] = {
        100: {"max_total_drawdown": 0.15, "max_daily_loss": 0.10},
        500: {"max_total_drawdown": 0.25, "max_daily_loss": 0.15},
        1000: {"max_total_drawdown": 0.40, "max_daily_loss": 0.20},
    }

    @classmethod
    def get(cls, tier: int) -> dict[str, float]:
        if tier >= 1000:
            return cls.TIERS[1000]
        if tier >= 500:
            return cls.TIERS[500]
        return cls.TIERS[100]


class Settings(BaseSettings):
    """Main application settings. All values read from .env."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # Platform credentials
    polymarket_private_key: str = ""
    polymarket_proxy_address: str = ""
    kalshi_api_key: str = ""
    kalshi_private_key: str = ""

    # LLM (OpenRouter)
    openrouter_api_key: str = ""
    llm_model_decision: str = "anthropic/claude-sonnet-4"
    llm_model_cheap: str = "anthropic/claude-haiku-4.5"
    ollama_model: str = "qwen3:8b"

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Trading parameters
    execution_mode: str = "paper"
    initial_bankroll: float = 100.0
    min_edge_percent: float = 10.0
    kelly_fraction: float = 0.15  # €100 bankroll Option A: semi-aggressive (was 0.08)
    max_open_positions: int = 20
    max_single_market_exposure: float = 0.08
    max_cost_per_trade_fraction: float = 0.04  # €100 bankroll Option A: 4% cap → ~€4 max/trade (was 0.015)
    min_trade_cost: float = 1.0  # €100 bankroll Option 2: floor €1/trade to meet Polymarket minimum
    max_single_platform_exposure: float = 0.60
    max_single_date_exposure: float = 0.25  # Max 25% bankroll on a single resolution date
    max_single_region_exposure: float = 0.25  # Max 25% bankroll on a single geographic region (Gemini P2: tightened from 0.30 to limit weather-front correlation when 2 cities in the same region trigger together)

    # Tiered risk (Addendum Gap 8)
    current_risk_tier: int = 100

    # Volatility targeting
    target_volatility: float = 0.10
    max_leverage: float = 2.0
    min_allocation: float = 0.20
    # no_trade_band and vol_decay_factor removed — were never used in any module

    # VPIN / toxic flow
    vpin_window: int = 50
    vpin_danger_threshold: float = 0.70
    vpin_action: str = "reduce_50"

    # Limit orders
    use_limit_orders: bool = False
    limit_order_offset: float = 0.02

    # TimesFM integration
    enable_timesfm: bool = True
    timesfm_model_id: str = "google/timesfm-1.0-200m-pytorch"
    timesfm_context_length: int = 512  # Historical data points to feed (hourly = ~21 days)
    timesfm_max_horizon: int = 48  # Max forecast steps ahead (hours)
    timesfm_weight: float = 0.25  # Base weight in blended probability
    timesfm_error_correction: bool = True  # Use TimesFM for NWP error prediction (highest alpha)
    timesfm_quantile_probability: bool = True  # Use quantile outputs for direct P(T>threshold)

    # Strategy toggles
    enable_conservative_agent: bool = True
    enable_aggressive_agent: bool = True
    enable_mean_reversion_agent: bool = False  # Disabled: mean reversion is negative EV in weather markets
    enable_qualitative_agent: bool = True
    enable_timesfm_agent: bool = True

    # Mean reversion
    mean_reversion_lookback_hours: int = 168
    mean_reversion_min_sigma: float = 3.0

    # Cities
    target_cities: str = "NYC,Chicago,Miami,Denver,LAX"

    # Scheduling
    forecast_poll_interval_minutes: int = 15
    actuals_poll_interval_minutes: int = 180
    actuals_lookback_days: int = 30
    market_scan_interval_minutes: int = 5
    decision_loop_interval_minutes: int = 5
    resolution_check_time: str = "08:00"
    resolution_check_interval_minutes: int = 30

    # Thresholds
    conservative_min_edge: float = 0.20
    conservative_same_day_edge_bonus: float = 0.05  # CCG-bug-5: extra edge required when lead_time_days == 0 (predatory spreads, thin books)
    aggressive_min_edge: float = 0.08

    # Warm-up: reduce risk until model is calibrated
    warmup_kelly_fraction: float = 0.10  # €100 bankroll Option A: semi-aggressive (was 0.05)
    min_trades_for_full_kelly: int = 50  # Switch to full Kelly after N resolved trades
    warmup_hours: int = 72  # Shadow mode duration (log but don't trade)
    enable_warmup_shadow_mode: bool = False  # Set True in .env to enable 72h shadow warm-up

    # Learning
    weight_update_interval_trades: int = 50
    agent_weight_update_interval: int = 100
    rmse_recalc_interval_days: int = 30
    error_analysis_interval_days: int = 7

    # Go live thresholds
    min_paper_trades_before_live: int = 200
    min_win_rate_before_live: float = 0.55
    min_sharpe_before_live: float = 1.0
    max_drawdown_before_live: float = 0.15

    # Data freshness (Addendum Gap 12)
    weather_data_ttl_minutes: int = 120
    api_retry_count: int = 3
    api_retry_backoff_base: int = 1

    # Fee estimates (Addendum Gap 2)
    polymarket_fee_rate: float = 0.00
    polymarket_avg_spread: float = 0.03
    kalshi_fee_on_winnings: float = 0.02
    kalshi_avg_spread: float = 0.02

    @property
    def cities_list(self) -> list[str]:
        return [c.strip() for c in self.target_cities.split(",")]

    @property
    def risk_params(self) -> dict[str, float]:
        return RiskTierParams.get(self.current_risk_tier)

    @property
    def fees(self) -> PlatformFees:
        return PlatformFees(
            polymarket_fee_rate=self.polymarket_fee_rate,
            polymarket_avg_spread=self.polymarket_avg_spread,
            kalshi_fee_on_winnings=self.kalshi_fee_on_winnings,
            kalshi_avg_spread=self.kalshi_avg_spread,
        )
