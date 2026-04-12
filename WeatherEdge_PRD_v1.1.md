# PRD: WeatherEdge — Autonomous Weather Prediction Market Trading Bot

## Document Version
- **Version**: 1.1.0
- **Date**: 2026-03-29
- **Author**: Taulant (Dualis / solo founder)
- **Target**: Claude Code implementation
- **Status**: Ready for development

### Changelog v1.1.0
- Added VPIN / Toxic Flow Detection to Risk Manager (source: @hanakoxbt thread on math bot strategies)
- Added Wallet Intelligence Module (RAG pipeline) as Phase 4.5 (source: @xmayeth thread on RAG replacing $2K/month analyst)
- Added Volatility-Targeted Position Sizing as secondary risk layer (source: @noisyb0y1 thread)
- Upgraded agent architecture reference from ad-hoc to LangGraph-based, citing QuantAgent paper (arXiv:2509.09995)
- Added polymarket-cli (Rust, official) as research/intelligence tool alongside pmxt (Python, trading)
- Added mean reversion agent for market price overreaction detection
- Added adversarial fill monitoring to execution engine
- Hardened kill switch thresholds based on 0x8dxd case study ($313→$2.38M, -40% total drawdown kill switch)
- Added BTC 5/15m expansion path as future Phase 6
- Added TurboQuant monitoring note for Ollama/MLX model upgrades

---

## 1. EXECUTIVE SUMMARY

WeatherEdge is a standalone autonomous trading bot that exploits pricing inefficiencies in weather prediction markets on Polymarket and Kalshi. The bot compares professional meteorological forecast data (NOAA, ECMWF, GFS ensembles) against market-implied probabilities, identifies mispriced contracts, and executes trades when the edge exceeds configurable thresholds.

The system uses a multi-agent LLM decision layer built on LangGraph (referencing the QuantAgent architecture from Stony Brook University, arXiv:2509.09995), where specialized agents (conservative, aggressive, black-swan, mean-reversion, qualitative) generate independent signals that a portfolio manager aggregates. Risk management includes Kelly Criterion sizing, volatility-targeted position scaling, VPIN toxic flow detection, and a hard kill switch. The self-learning loop tracks every prediction vs. outcome to continuously calibrate the model.

**Revenue target**: $500-2,000/month passive income with $1,000-5,000 deployed capital.

**Key insight from research**: Successful weather traders (neobrother: $20K+, Hans323: $1.1M single trade, meropi: $30K automated) are NOT predicting weather better than meteorologists. They are systematically trading the gap between professional forecasts and what the crowd on Polymarket/Kalshi prices in, exploiting forecast latency (6-hour model update cycles) and market inefficiency.

**Validated edge metrics** (from @xmayeth RAG analysis of 23,000 Polymarket wallets): NOAA weather lag is consistently 9-15 minutes. Top weather-specialized wallets achieve 89% win rate with Sharpe >3.0. The edge has been stable over 90+ days as of March 2026.

**Risk context** (from 0x8dxd case study): The BTC latency arbitrage window compressed from 12s (2024) to 2.7s (Q1 2026). Weather markets face similar compression risk as more bots enter. This is a time-limited opportunity — extract value while the edge exists, build the self-learning loop to adapt when it narrows.

---

## 2. SYSTEM ARCHITECTURE

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WeatherEdge Bot                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────┐│
│  │  Data Layer   │  │ Probability  │  │   Multi-Agent Decision    ││
│  │              │  │   Engine     │  │        Layer (LLM)         ││
│  │ - Open-Meteo │  │              │  │                            ││
│  │ - NOAA NWS   │  │ - Gaussian   │  │ ┌────────┐ ┌────────────┐ ││
│  │ - Historical │  │   CDF        │  │ │Conserv.│ │ Aggressive │ ││
│  │   Station    │  │ - Ensemble   │  │ │ Agent  │ │   Agent    │ ││
│  │   Data       │  │   Counting   │  │ └───┬────┘ └─────┬──────┘ ││
│  └──────┬───────┘  │ - Bias       │  │     │            │        ││
│         │          │   Correction  │  │ ┌───┴────┐ ┌─────┴──────┐ ││
│         │          └──────┬───────┘  │ │BlackSw.│ │   Risk     │ ││
│         │                 │          │ │ Agent  │ │  Manager   │ ││
│         ▼                 ▼          │ └───┬────┘ └─────┬──────┘ ││
│  ┌──────────────────────────────┐    │     │            │        ││
│  │      Market Scanner          │    │ ┌───┴────────────┴──────┐ ││
│  │                              │    │ │  Portfolio Manager     │ ││
│  │  - Polymarket API (via pmxt) │    │ │  (Final Decision)      │ ││
│  │  - Kalshi API (via pmxt)     │    │ └────────────┬───────────┘ ││
│  │  - Cross-platform arb detect │    └──────────────┼────────────┘│
│  └──────────────┬───────────────┘                   │             │
│                 │                                    │             │
│                 ▼                                    ▼             │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   Execution Engine                           │  │
│  │  - Order placement (via pmxt)                                │  │
│  │  - Position tracking                                         │  │
│  │  - P&L calculation                                           │  │
│  │  - Paper trading mode                                        │  │
│  └──────────────────────────────┬───────────────────────────────┘  │
│                                 │                                  │
│                                 ▼                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   Self-Learning Loop                         │  │
│  │  - Resolution tracking (did we win?)                         │  │
│  │  - Brier score calibration                                   │  │
│  │  - Ensemble weight updates                                   │  │
│  │  - RMSE recalculation per station                            │  │
│  │  - LLM error pattern analysis                                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                   Notification Layer                         │  │
│  │  - Telegram bot (primary interface)                          │  │
│  │  - Trade alerts, daily P&L summaries, error alerts           │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Tech Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Runtime | Python 3.11+ | Ecosystem (scipy, pandas), pmxt SDK |
| Agent Orchestration | LangGraph | Multi-agent workflow graphs, state management (reference: QuantAgent) |
| Market Connector | `pmxt` (pip install pmxt) | Unified API for Polymarket + Kalshi trading |
| Market Intelligence | `polymarket-cli` (Rust, official) | Wallet tracking, orderbook analysis, leaderboard research |
| Weather Data | Open-Meteo API (free, no key) | GFS ensemble 31 members, ECMWF, HRRR |
| Ground Truth | NOAA NWS API (free) | Same source that resolves Kalshi markets |
| Database | SQLite | Local, zero-config, sufficient for this scale |
| LLM (decision) | Claude Sonnet via API | Quality reasoning for decision layer + qualitative agent |
| LLM (high-volume) | Ollama local (qwen3:8b) | Cost-effective for routine analysis, wallet profiling |
| LLM (future) | Monitor TurboQuant for MLX integration | 6x memory compression → run 35B models on Mac Mini M4 |
| Notifications | Telegram Bot API | Already in use for existing agent systems |
| Scheduling | APScheduler | In-process cron for Python |
| Deployment | Mac Mini M4 (local) | Already available, 24/7 operation |

### 2.3 Directory Structure

```
weatheredge/
├── main.py                     # Entry point, scheduler setup
├── config.py                   # Pydantic settings (reads .env)
├── .env                        # API keys, thresholds, config
│
├── data/
│   ├── weather.py              # Open-Meteo + NOAA data fetchers
│   ├── historical.py           # Station historical data loader
│   └── market.py               # Polymarket + Kalshi scanner (via pmxt)
│
├── probability/
│   ├── gaussian.py             # Gaussian CDF probability model
│   ├── ensemble.py             # Ensemble member counting model
│   ├── bias_correction.py      # Station-specific bias correction
│   └── calibration.py          # Brier score + reliability tracking
│
├── agents/
│   ├── base.py                 # BaseAgent ABC
│   ├── conservative.py         # High-threshold, low-size agent
│   ├── aggressive.py           # Temperature laddering agent
│   ├── black_swan.py           # Long-shot mispricing agent
│   ├── mean_reversion.py       # Market price overreaction agent (NEW v1.1)
│   ├── risk_manager.py         # Kelly + volatility targeting + VPIN
│   ├── portfolio_manager.py    # Signal aggregation + final decision
│   └── qualitative.py          # LLM-powered qualitative factor agent
│
├── execution/
│   ├── executor.py             # Order placement via pmxt
│   ├── paper_trader.py         # Simulated execution engine
│   ├── position_tracker.py     # Open positions + P&L tracking
│   ├── cross_platform.py      # Cross-platform arbitrage detector
│   └── fill_monitor.py         # Adversarial fill / VPIN detection (NEW v1.1)
│
├── learning/
│   ├── resolution_tracker.py   # Track market resolutions
│   ├── performance.py          # Win rate, ROI, Sharpe calculation
│   ├── weight_updater.py       # Update ensemble/agent weights
│   └── error_analyzer.py       # LLM analysis of losing trades
│
├── intelligence/
│   ├── wallet_profiler.py      # RAG pipeline for wallet analysis (NEW v1.1)
│   ├── edge_decay_monitor.py   # Track edge compression over time (NEW v1.1)
│   └── competitor_tracker.py   # Monitor top weather wallets (NEW v1.1)
│
├── notifications/
│   ├── telegram.py             # Telegram bot interface
│   └── formatters.py           # Message formatting
│
├── backtest/
│   ├── engine.py               # Historical backtesting engine
│   ├── data_loader.py          # Load historical weather + market data
│   └── metrics.py              # Sharpe, Sortino, max drawdown
│
├── db/
│   ├── models.py               # SQLAlchemy/SQLite models
│   ├── migrations.py           # Schema versioning
│   └── weatheredge.db          # SQLite database (gitignored)
│
├── tests/
│   ├── test_probability.py     # Unit tests for probability models
│   ├── test_agents.py          # Unit tests for agent logic
│   ├── test_execution.py       # Unit tests for execution
│   └── test_integration.py     # End-to-end integration tests
│
├── scripts/
│   ├── collect_historical.py   # One-time: download station history
│   ├── analyze_microstructure.py  # Research: market spread analysis
│   └── backtest_run.py         # Run backtests with params
│
├── docs/
│   ├── STRATEGY.md             # Detailed strategy documentation
│   ├── RESOLUTION_SOURCES.md   # Market resolution source mapping
│   └── DEPLOYMENT.md           # Deployment & operations guide
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 3. DATA LAYER

### 3.1 Weather Data Sources

#### 3.1.1 Open-Meteo (Primary Forecast Source)

**Endpoint**: `https://api.open-meteo.com/v1/`
**Cost**: Free for non-commercial use (CC BY 4.0)
**No API key required**

**APIs to use**:

1. **Forecast API** (`/v1/forecast`) — Combines best models per location, hourly data, 7-16 days
2. **GFS API** (`/v1/gfs`) — US-focused, includes HRRR (hourly updates), up to 16 days
3. **ECMWF API** (`/v1/ecmwf`) — Highest accuracy global model, 9km resolution, up to 15 days
4. **Ensemble API** (`/v1/ensemble`) — Access to 31 GFS ensemble members individually

**Variables to fetch**:
```python
HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "precipitation_probability",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "relative_humidity_2m",
]

ENSEMBLE_VARS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
]
```

**Polling schedule**:
- Ensemble API: Every 6 hours (aligned with model runs at 00, 06, 12, 18 UTC)
- Forecast API: Every hour (HRRR updates hourly)
- ECMWF API: Every 6 hours (aligned with runs)

**Critical implementation detail**: After each model run, there is a delay before data becomes available. GFS data typically arrives ~3.5 hours after initialization time. ECMWF open data has ~2 hour additional delay vs real-time dissemination. The bot should poll and detect when new data arrives, then immediately scan for market opportunities.

#### 3.1.2 NOAA NWS API (Ground Truth + US Forecast)

**Endpoint**: `https://api.weather.gov/`
**Cost**: Free, no API key (rate limited but generous)

**Key endpoints**:
- `/points/{lat},{lon}` → Get grid forecast endpoint for coordinates
- `/gridpoints/{office}/{gridX},{gridY}/forecast` → 12h period forecast, 7 days
- `/gridpoints/{office}/{gridX},{gridY}/forecast/hourly` → Hourly forecast, 7 days

**Why this matters**: NOAA forecasts are the same system that produces the NWS Daily Climate Report (CLI), which is the resolution source for Kalshi markets. Trading against the same source that resolves the market is a structural edge.

#### 3.1.3 Historical Station Data (Bias Correction)

**Source**: NOAA NCEI Local Climatological Data (LCDv2)
**Purpose**: Calculate station-specific RMSE for each forecast model

For each target station (Central Park, Midway Airport, Miami International, etc.):
1. Download last 2 years of daily high/low temperature actuals
2. Download corresponding forecast data from Open-Meteo Historical API
3. Calculate RMSE per model per lead time per station
4. Use these RMSE values as the σ parameter in Gaussian CDF model

```python
# Example: Station-specific RMSE lookup
STATION_RMSE = {
    "KNYC": {  # Central Park, NYC
        "gfs": {1: 2.3, 2: 3.1, 3: 4.2},      # lead_time_days: rmse_fahrenheit
        "ecmwf": {1: 2.0, 2: 2.8, 3: 3.8},
        "hrrr": {1: 1.8, 2: 2.5},               # HRRR only goes 2 days
    },
    "KMDW": {  # Midway Airport, Chicago
        "gfs": {1: 2.8, 2: 3.5, 3: 4.8},
        "ecmwf": {1: 2.4, 2: 3.2, 3: 4.3},
        "hrrr": {1: 2.1, 2: 3.0},
    },
    # ... etc for each station
}
```

### 3.2 Market Data Sources

#### 3.2.1 Market Scanner via pmxt

```python
import pmxt

# Polymarket
poly = pmxt.Polymarket(
    private_key=os.getenv('POLYMARKET_PRIVATE_KEY'),
    proxy_address=os.getenv('POLYMARKET_PROXY_ADDRESS'),
)

# Kalshi
kalshi = pmxt.Kalshi(
    api_key=os.getenv('KALSHI_API_KEY'),
    private_key=os.getenv('KALSHI_PRIVATE_KEY'),
)

# Scan for weather markets
poly_weather = poly.fetch_events(query='temperature')
kalshi_weather = kalshi.fetch_events(query='KXHIGH')  # Kalshi weather ticker prefix
```

#### 3.2.2 Resolution Source Mapping

**THIS IS THE MOST CRITICAL TABLE IN THE ENTIRE SYSTEM.**

Every market resolves against a specific data source. Getting this wrong means your model is predicting the wrong thing.

```python
RESOLUTION_SOURCES = {
    # KALSHI (all resolve against NWS Daily Climate Report)
    "kalshi": {
        "KXHIGHNY": {
            "station": "KNYC",       # Central Park
            "type": "daily_high",
            "source": "NWS CLI",
            "timezone": "EST",        # LST, not DST!
            "report_time": "~7:00 AM next day",
        },
        "KXHIGHCHI": {
            "station": "KMDW",       # Midway Airport
            "type": "daily_high",
            "source": "NWS CLI",
            "timezone": "CST",
            "report_time": "~7:00 AM next day",
        },
        "KXHIGHMIA": {
            "station": "KMIA",       # Miami International Airport
            "type": "daily_high",
            "source": "NWS CLI",
            "timezone": "EST",
            "report_time": "~7:00 AM next day",
        },
        "KXHIGHLAX": {
            "station": "KLAX",       # LAX
            "type": "daily_high",
            "source": "NWS CLI",
            "timezone": "PST",
            "report_time": "~7:00 AM next day",
        },
        "KXHIGHDEN": {
            "station": "KDEN",       # Denver International
            "type": "daily_high",
            "source": "NWS CLI",
            "timezone": "MST",
            "report_time": "~7:00 AM next day",
        },
    },

    # POLYMARKET (varies by market — ALWAYS check market rules)
    "polymarket": {
        "london_temp": {
            "station": "EGLC",       # London City Airport
            "type": "daily_high",
            "source": "Weather Underground",
            "timezone": "GMT/BST",
        },
        "nyc_precipitation": {
            "station": "Central Park",
            "type": "monthly_total_precipitation",
            "source": "NOAA",
            "precision": "0.01 inch",
        },
        # NOTE: Polymarket resolution sources vary by market.
        # The bot MUST parse each market's rules before trading.
    },
}
```

**CRITICAL TIMEZONE NOTE**: Kalshi NWS Climate Reports use Local Standard Time (LST), NOT Daylight Saving Time. During DST months, the "day" for climate reporting is shifted. The high temperature is recorded between 1:00 AM and 12:59 AM LST the following day. This catches traders who don't read the fine print.

---

## 4. PROBABILITY ENGINE

### 4.1 Model 1: Gaussian CDF

The simplest and most robust approach. Models forecast error as a Gaussian distribution centered on the forecast value, with standard deviation equal to the historical RMSE for that station/model/lead-time.

```python
from scipy.stats import norm

def gaussian_probability(
    forecast_temp: float,      # e.g., 85°F
    threshold: float,          # e.g., 80°F (market question threshold)
    rmse: float,               # e.g., 2.5°F (station-specific, lead-time-specific)
    direction: str = "above",  # "above" or "below"
) -> float:
    """
    Calculate probability that actual temp will be above/below threshold,
    given a forecast and known error distribution.
    """
    if direction == "above":
        return 1 - norm.cdf(threshold, loc=forecast_temp, scale=rmse)
    else:
        return norm.cdf(threshold, loc=forecast_temp, scale=rmse)
```

**Example**:
- NOAA says NYC high = 85°F tomorrow
- Market asks: "Will NYC high exceed 80°F?"
- Historical RMSE for 1-day NOAA forecast at KNYC = 2.5°F
- P(>80°F) = 1 - norm.cdf(80, loc=85, scale=2.5) = 97.7%
- If market prices YES at 35%, edge = 62.7% → TRADE

### 4.2 Model 2: Ensemble Member Counting

Uses the GFS 31-member ensemble directly. Each member represents a plausible outcome.

```python
def ensemble_probability(
    ensemble_temps: list[float],  # 31 temperature values from ensemble members
    threshold: float,
    direction: str = "above",
) -> float:
    """
    Count fraction of ensemble members above/below threshold.
    """
    if direction == "above":
        count = sum(1 for t in ensemble_temps if t > threshold)
    else:
        count = sum(1 for t in ensemble_temps if t <= threshold)
    return count / len(ensemble_temps)
```

**Advantage**: Does not assume Gaussian distribution. Better for edge cases (bimodal distributions when a front might or might not pass through).

### 4.3 Model 3: Blended Probability

Combine Gaussian CDF and Ensemble counting with configurable weights:

```python
def blended_probability(
    gaussian_prob: float,
    ensemble_prob: float,
    gaussian_weight: float = 0.5,  # Updated by self-learning loop
) -> float:
    ensemble_weight = 1 - gaussian_weight
    return gaussian_prob * gaussian_weight + ensemble_prob * ensemble_weight
```

Weights start at 50/50 and are updated by the self-learning loop based on which model has been more accurate recently (per station, per lead time).

### 4.4 Bias Correction

Station-specific systematic bias correction:

```python
def apply_bias_correction(
    forecast_temp: float,
    station: str,
    model: str,
    lead_time_hours: int,
    season: str,  # "winter", "spring", "summer", "fall"
) -> float:
    """
    Correct for systematic forecast bias.
    E.g., GFS consistently overpredicts Chicago summer highs by 1.2°F.
    """
    bias = BIAS_TABLE.get((station, model, lead_time_hours, season), 0.0)
    return forecast_temp - bias
```

The bias table is populated from historical data analysis (see Section 3.1.3) and updated monthly.

### 4.5 Edge Calculation

```python
def calculate_edge(
    model_probability: float,   # Our estimated probability
    market_price: float,        # Current market price (0.01 to 0.99)
) -> float:
    """
    Edge = our probability - market implied probability.
    Positive edge = market underprices the outcome.
    """
    return model_probability - market_price
```

### 4.6 Range/Bucket Markets

Many weather markets are bucket-style: "Temperature will be 80-84°F", "85-89°F", etc. For these:

```python
def bucket_probability(
    forecast_temp: float,
    bucket_low: float,
    bucket_high: float,
    rmse: float,
) -> float:
    """
    Probability that actual temp falls within [bucket_low, bucket_high].
    """
    p_below_high = norm.cdf(bucket_high, loc=forecast_temp, scale=rmse)
    p_below_low = norm.cdf(bucket_low, loc=forecast_temp, scale=rmse)
    return p_below_high - p_below_low
```

---

## 5. MULTI-AGENT DECISION LAYER

Inspired by `virattt/ai-hedge-fund` architecture. Each agent evaluates opportunities independently with a different philosophy, then the Portfolio Manager aggregates.

### 5.1 Agent: Conservative ("Ben Graham")

**Philosophy**: Only trade when the edge is overwhelming. High confidence, small but consistent profits.

```python
class ConservativeAgent(BaseAgent):
    name = "Conservative"
    min_edge_threshold = 0.20      # Only trade when edge > 20%
    min_model_agreement = 0.9      # Gaussian AND ensemble must agree within 10%
    max_position_pct = 0.03        # Max 3% of bankroll per trade
    preferred_lead_time = 1        # Only trade 1-day forecasts (highest accuracy)
    strategy = "single_bucket"     # Buy the single most probable bucket
```

### 5.2 Agent: Aggressive ("Cathie Wood")

**Philosophy**: Temperature laddering. Buy multiple adjacent buckets around the forecast. Higher volume, captures more of the probability distribution.

```python
class AggressiveAgent(BaseAgent):
    name = "Aggressive"
    min_edge_threshold = 0.08      # Trade at lower edge thresholds
    ladder_width = 3               # Buy 3 adjacent buckets
    max_position_pct = 0.05        # Max 5% of bankroll per trade (across ladder)
    preferred_lead_time = [1, 2]   # 1-2 day forecasts
    strategy = "temperature_ladder"
    
    def generate_orders(self, market, probabilities):
        """
        Buy YES on 3 adjacent buckets centered on the forecast.
        Size each position inversely proportional to distance from center.
        """
        # Implementation: find top-3 buckets by probability,
        # size center bucket at 50%, adjacent at 25% each
        pass
```

### 5.3 Agent: Black Swan ("Michael Burry")

**Philosophy**: Hunt for extremely mispriced long-shots. Buy at 1-5¢ where the model says 10-20% probability. High risk per trade but asymmetric payoff (20x-100x).

```python
class BlackSwanAgent(BaseAgent):
    name = "BlackSwan"
    max_entry_price = 0.05         # Only buy contracts priced at ≤5¢
    min_model_probability = 0.10   # Model must say ≥10% chance
    max_position_pct = 0.01        # Max 1% of bankroll per trade
    preferred_lead_time = [2, 3]   # Slightly longer range (more uncertainty = more mispricing)
    strategy = "long_shot"
    
    def generate_orders(self, market, probabilities):
        """
        Find buckets priced ≤5¢ where our model says ≥10%.
        These are the most mispriced contracts.
        """
        pass
```

### 5.4 Agent: Mean Reversion ("Buy the Overreaction") — NEW v1.1

**Philosophy**: Markets overreact. When a weather market price deviates significantly from its rolling mean (e.g., due to a headline, a temporary model glitch, or panic selling), buy the deviation expecting reversion. This is price-based, not forecast-based — no opinion on the weather, just statistics about the price.

**Source**: @hanakoxbt thread on math bot strategies (Strategy 3).

```python
class MeanReversionAgent(BaseAgent):
    name = "MeanReversion"
    lookback_hours = 168           # 7-day rolling window
    min_deviation_sigma = 3.0      # Only trade when price is >3σ from mean
    max_position_pct = 0.04        # Max 4% of bankroll per trade
    strategy = "mean_reversion"
    
    def generate_signals(self, market_prices: list[float]) -> Signal:
        """
        1. Calculate 7-day rolling mean and standard deviation of market price
        2. Compute current deviation in σ units
        3. If |deviation| > 3σ, signal to buy the opposite direction
        
        Example:
        - 7-day mean: $0.89, σ: $0.04
        - Current price crashes to $0.72 (headline panic)
        - Deviation: -4.25σ → BUY signal
        - Target: reversion toward $0.85-0.87
        """
        import numpy as np
        mean = np.mean(market_prices[-self.lookback_hours:])
        std = np.std(market_prices[-self.lookback_hours:])
        current = market_prices[-1]
        deviation = (current - mean) / std if std > 0 else 0
        
        if abs(deviation) >= self.min_deviation_sigma:
            side = "buy_yes" if deviation < 0 else "buy_no"
            return Signal(side=side, confidence=min(abs(deviation) / 5, 1.0))
        return Signal(side="no_trade", confidence=0)
```

### 5.5 Agent: Qualitative Factor (LLM-Powered)

**Philosophy**: Identify situations where quantitative models are likely to be wrong due to factors they don't capture.

```python
class QualitativeAgent(BaseAgent):
    name = "Qualitative"
    uses_llm = True
    
    def analyze(self, market, weather_context):
        """
        LLM evaluates qualitative factors:
        - Urban heat island effects not captured by models
        - Wildfire smoke affecting temperatures (models lag on this)
        - Sea breeze patterns at coastal stations
        - Inversions and microclimate effects
        - Recent model performance (has the model been running hot/cold?)
        - Resolution source quirks (DST/LST timing, station placement)
        
        Returns: adjustment factor (-0.1 to +0.1) to model probability
        and a confidence score.
        """
        prompt = f"""
        You are a weather market analyst. Evaluate whether quantitative 
        forecast models are likely to be wrong for this specific situation.
        
        City: {market.city}
        Station: {market.station}
        Date: {market.date}
        Current forecast: {weather_context.forecast}
        Model ensemble spread: {weather_context.ensemble_spread}
        Recent model bias (last 7 days): {weather_context.recent_bias}
        Current conditions: {weather_context.current_conditions}
        Notable factors: {weather_context.notable_factors}
        
        Consider:
        1. Are there microclimate factors this station is known for?
        2. Is there unusual weather (smoke, inversion, unusual wind)?
        3. Has the model been systematically biased recently?
        4. Are there timing issues (DST/LST, observation schedule)?
        
        Return a JSON with:
        - adjustment: float between -0.1 and 0.1 (negative = model is too warm)
        - confidence: float between 0 and 1
        - reasoning: string explaining your assessment
        """
        # Call LLM and parse response
        pass
```

### 5.7 Risk Manager — UPGRADED v1.1

Combines Kelly Criterion, volatility-targeted sizing, VPIN toxic flow detection, and hard circuit breakers.

```python
class RiskManager:
    # === Circuit Breakers ===
    max_daily_loss_pct = 0.20          # Stop trading if daily loss > 20% (0x8dxd standard)
    max_total_drawdown_pct = 0.40      # HARD KILL SWITCH at -40% total drawdown
    max_open_positions = 20             # Max concurrent positions
    max_single_market_exposure = 0.08   # Max 8% bankroll in one market (0x8dxd standard)
    max_single_platform_exposure = 0.60 # Max 60% on one platform
    
    # === Kelly Criterion ===
    kelly_fraction = 0.15               # Use 15% Kelly (conservative)
    
    # === Volatility Targeting (NEW v1.1) ===
    target_volatility = 0.10            # 10% annualized target vol
    max_leverage = 2.0                  # Never exceed 2x
    min_allocation = 0.20               # Always keep 20% deployed minimum
    no_trade_band = 0.10                # Only rebalance if drift > 10%
    vol_decay_factor = 0.94             # EWMA decay for vol calculation
    
    # === VPIN Toxic Flow Detection (NEW v1.1) ===
    vpin_window = 50                    # Last 50 fills to calculate VPIN
    vpin_danger_threshold = 0.70        # If >70% fills hitting one side → toxic
    
    def calculate_position_size(
        self, 
        edge: float, 
        win_probability: float, 
        bankroll: float,
        contract_vol: float = None,      # NEW: realized vol of the contract
    ) -> float:
        """
        Two-layer sizing:
        1. Kelly Criterion → base size from edge
        2. Volatility targeting → adjust for current contract volatility
        Final size = min(kelly_size, vol_adjusted_size, hard_caps)
        """
        if edge <= 0:
            return 0.0
        
        # Layer 1: Kelly
        odds = (1.0 / (1.0 - win_probability)) - 1
        q = 1 - win_probability
        kelly = (win_probability * odds - q) / odds
        kelly = max(0, kelly)
        kelly_size = kelly * self.kelly_fraction * bankroll
        
        # Layer 2: Volatility targeting (if vol data available)
        if contract_vol and contract_vol > 0:
            raw_leverage = self.target_volatility / contract_vol
            leverage = min(raw_leverage, self.max_leverage)
            leverage = max(leverage, self.min_allocation)
            vol_size = leverage * bankroll * (1 / self.max_open_positions)
        else:
            vol_size = kelly_size  # Fallback to Kelly only
        
        position = min(kelly_size, vol_size)
        
        # Hard caps
        position = min(position, bankroll * self.max_single_market_exposure)
        
        return round(position, 2)
    
    def calculate_vpin(self, recent_fills: list[dict]) -> float:
        """
        VPIN = |V_buy - V_sell| / (V_buy + V_sell)
        
        High VPIN → someone knows something you don't → toxic flow.
        When VPIN > 0.70: widen spread, reduce size, or go dark.
        
        Source: @hanakoxbt thread (Strategy 4: Kill Switch)
        """
        v_buy = sum(f['size'] for f in recent_fills if f['side'] == 'buy')
        v_sell = sum(f['size'] for f in recent_fills if f['side'] == 'sell')
        total = v_buy + v_sell
        if total == 0:
            return 0.0
        return abs(v_buy - v_sell) / total
    
    def check_circuit_breakers(self, daily_pnl: float, total_pnl: float, bankroll: float) -> dict:
        """Return dict with halt status and reason."""
        if total_pnl < 0 and abs(total_pnl) > bankroll * self.max_total_drawdown_pct:
            return {"halt": True, "reason": "KILL SWITCH: Total drawdown exceeded -40%", "severity": "critical"}
        if daily_pnl < 0 and abs(daily_pnl) > bankroll * self.max_daily_loss_pct:
            return {"halt": True, "reason": "Daily loss limit exceeded -20%", "severity": "high"}
        return {"halt": False}
    
    def check_adverse_selection(self, recent_fills: list[dict]) -> dict:
        """
        Monitor if our orders are being systematically picked off.
        If our fills consistently move against us immediately after execution,
        we're being hunted by informed traders.
        
        Source: @hanakoxbt thread (adversarial fill detection)
        """
        vpin = self.calculate_vpin(recent_fills)
        if vpin > self.vpin_danger_threshold:
            return {
                "toxic": True, 
                "vpin": vpin,
                "action": "REDUCE_SIZE_50_PCT",
                "alert": f"⚠️ TOXIC FLOW: VPIN={vpin:.2f}. Reducing position sizes by 50%."
            }
        return {"toxic": False, "vpin": vpin}
```

### 5.8 Portfolio Manager (Signal Aggregation) — UPDATED v1.1

```python
class PortfolioManager:
    """
    Aggregates signals from all agents and makes final trading decisions.
    Uses LangGraph StateGraph for orchestration (reference: QuantAgent architecture).
    """
    agent_weights = {
        "Conservative": 0.25,   # Consistent edge, highest trust
        "Aggressive": 0.25,     # Ladder strategy, high volume
        "BlackSwan": 0.10,      # Low weight but high asymmetry
        "MeanReversion": 0.15,  # Price-based, no weather opinion needed (NEW v1.1)
        "Qualitative": 0.15,    # LLM adjustment factor
        # Note: Risk Manager is not weighted — it has veto power
    }
    
    def aggregate_signals(self, agent_signals: dict) -> list[Order]:
        """
        1. Collect all agent signals
        2. Calculate weighted consensus
        3. Apply risk manager checks
        4. Generate final order list
        """
        pass
    
    def resolve_conflicts(self, signals: dict) -> Decision:
        """
        When agents disagree (e.g., Conservative says NO TRADE,
        Aggressive says BUY), apply weighted voting.
        
        If Risk Manager vetoes, always respect the veto.
        """
        pass
```

---

## 6. EXECUTION ENGINE

### 6.1 Paper Trading Mode (Phase 1)

**CRITICAL: The bot MUST start in paper trading mode.** No real money until the paper trader proves profitability over ≥200 trades.

```python
class PaperTrader:
    def __init__(self, initial_bankroll: float = 1000.0):
        self.bankroll = initial_bankroll
        self.positions = []
        self.trade_history = []
        self.start_date = datetime.utcnow()
    
    def execute_order(self, order: Order) -> PaperTrade:
        """
        Simulate order execution:
        - Deduct cost from bankroll
        - Add to open positions
        - Log everything
        """
        pass
    
    def resolve_market(self, market_id: str, outcome: str):
        """
        When a market resolves, calculate P&L for all positions.
        """
        pass
    
    def get_performance_report(self) -> PerformanceReport:
        """
        Return: win_rate, roi, sharpe_ratio, max_drawdown,
        total_trades, total_pnl, avg_edge_at_entry
        """
        pass
```

### 6.2 Live Trading Mode (Phase 2)

Only activated after paper trading validation:
- Minimum 200 paper trades completed
- Win rate > 55%
- Positive ROI after fees
- Sharpe ratio > 1.0
- Max drawdown < 15%

```python
class LiveExecutor:
    def __init__(self, exchange: pmxt.Exchange, mode: str = "paper"):
        self.exchange = exchange
        self.mode = mode
    
    def place_order(self, order: Order) -> ExecutionResult:
        if self.mode == "paper":
            return self.paper_trader.execute_order(order)
        
        # Live execution via pmxt
        result = self.exchange.create_order(
            outcome=order.outcome,
            side=order.side,
            type="limit",
            price=order.price,
            amount=order.amount,
        )
        return ExecutionResult.from_pmxt(result)
```

### 6.3 Cross-Platform Arbitrage Detector

```python
class CrossPlatformArbitrage:
    """
    Detect when the same weather event is priced differently
    on Polymarket and Kalshi.
    """
    def scan(self, poly_markets, kalshi_markets) -> list[ArbOpportunity]:
        """
        Match markets by:
        1. Same city
        2. Same date
        3. Same/overlapping temperature threshold
        
        If YES_poly + YES_kalshi < 1.00 for complementary outcomes,
        or if the same outcome is priced differently by > threshold,
        flag as arbitrage opportunity.
        """
        pass
```

---

## 7. SELF-LEARNING LOOP

### 7.1 Resolution Tracking

```python
class ResolutionTracker:
    def check_resolutions(self):
        """
        Run every morning at 8:00 AM local time for each city.
        
        1. Fetch NWS Daily Climate Report for previous day
        2. For each open position in that city/date:
           a. Compare actual high temp vs market threshold
           b. Mark position as WON or LOST
           c. Update bankroll
           d. Log: predicted_prob, market_price_at_entry, actual_outcome, pnl
        """
        pass
```

### 7.2 Calibration Metrics

```python
class CalibrationTracker:
    def update_brier_score(self, predicted_prob: float, actual_outcome: int):
        """
        Brier Score = mean((predicted - actual)^2)
        Lower is better. Perfect = 0. Random = 0.25.
        
        Track rolling 100-trade Brier score.
        """
        pass
    
    def update_reliability_diagram(self, predictions: list):
        """
        Group predictions into bins (0-10%, 10-20%, etc.)
        For each bin, compare mean predicted probability vs actual frequency.
        Perfect calibration: predicted 30% → wins 30% of the time.
        """
        pass
```

### 7.3 Weight Updates

```python
class WeightUpdater:
    def update_model_weights(self, recent_trades: list[Trade]):
        """
        Every 50 trades:
        1. Calculate Brier score for Gaussian model vs Ensemble model
        2. Shift blended_probability weights toward the better model
        3. Apply Bayesian update: new_weight ∝ old_weight * likelihood
        """
        pass
    
    def update_agent_weights(self, recent_trades: list[Trade]):
        """
        Every 100 trades:
        1. Calculate ROI per agent
        2. Increase weight for agents with positive ROI
        3. Decrease weight for underperforming agents
        4. Never let any agent weight drop below 0.05
        """
        pass
    
    def update_station_rmse(self, station: str, model: str, lead_time: int):
        """
        Monthly:
        1. Fetch last 30 days of forecasts vs actuals
        2. Recalculate RMSE
        3. Update STATION_RMSE table
        """
        pass
```

### 7.4 Error Pattern Analysis (LLM)

```python
class ErrorAnalyzer:
    def analyze_losing_trades(self, losing_trades: list[Trade]):
        """
        Weekly: send batch of losing trades to LLM for pattern analysis.
        
        Prompt:
        "Here are our losing trades from this week. For each, I'll show
        the forecast, the market price, our estimated probability, and 
        the actual outcome. Identify patterns:
        - Are we consistently wrong about a specific city?
        - Are we wrong at certain lead times?
        - Are there weather patterns (fronts, inversions) we consistently miss?
        - Are there times of day when our model breaks down?
        
        Return specific, actionable adjustments."
        """
        pass
```

---

## 8. WALLET INTELLIGENCE MODULE — NEW v1.1

A RAG-based competitive intelligence system that profiles Polymarket wallets specializing in weather markets. Replaces manual analysis that would cost $2K+/month with an automated pipeline costing ~$12 to build and $0.02/query to run.

**Source**: @xmayeth thread — RAG Agent replacing $2K/month Polymarket analyst.

### 8.1 Pipeline Architecture

```
POLYMARKET DATA → CHUNK + STATS → SUMMARIZE → INDEX → QUERY
  (trades, wallets)   (per-wallet)    (Claude)    (JSON)   (natural language)
```

### 8.2 Implementation

```python
class WalletIntelligence:
    """
    Uses polymarket-cli (Rust) for data extraction,
    Claude Haiku for profiling (cheap), Claude Sonnet for queries (quality).
    """
    
    def collect_weather_wallets(self):
        """
        Step 1: Use polymarket-cli to find active weather traders.
        
        Commands:
        - polymarket data leaderboard --period month --order-by pnl
        - polymarket data holders <WEATHER_CONDITION_ID>
        - polymarket data trades <WALLET> --limit 500
        
        Filter for wallets with >50 trades in weather markets.
        """
        pass
    
    def profile_wallet(self, wallet_stats: dict) -> str:
        """
        Step 2: Claude generates a text profile per wallet.
        
        Prompt: "Write a 3-4 sentence analytical profile of this
        Polymarket wallet. Focus on: strategy type, market category 
        specialization, edge sustainability, and whether behavior 
        is automated or manual.
        
        Address: {wallet}
        Trades: {trades} | Win rate: {wr}%
        Sharpe: {sharpe} | Total P&L: ${pnl}
        Max drawdown: {dd}% | Avg hold: {hold}
        Categories: {cats}"
        
        Use Claude Haiku for speed/cost (~$1.50 for all profiles).
        """
        pass
    
    def build_index(self, profiles: dict):
        """
        Step 3: Save as searchable JSON file (~15MB).
        No vector DB needed — keyword matching + stats filtering.
        """
        pass
    
    def query(self, question: str) -> str:
        """
        Step 4: Natural language queries against the index.
        Uses Claude Sonnet for quality reasoning.
        
        Key daily queries:
        1. "Identify the 5 most profitable weather strategies with Sharpe > 2"
        2. "Flag wallets with edge decay — compare last 30 days vs prior 90 days"
        3. "Which external data sources correlate with profitable weather trades?"
        4. "60-second briefing: what changed overnight in weather markets?"
        """
        pass

class EdgeDecayMonitor:
    """
    Track whether the overall weather market edge is compressing.
    If top wallets' Sharpe ratios are declining, the edge is eroding.
    
    Alert if: average top-10 Sharpe drops below 1.5 (was 3.0+ historically).
    """
    
    def weekly_check(self):
        """
        Compare this week's top weather wallet performance vs 4 weeks ago.
        If systematic decline: alert via Telegram with recommendation
        to reduce position sizes or shift to less crowded cities.
        """
        pass
```

### 8.3 Scheduling

Run wallet profiling weekly (Sunday night). Run edge decay check weekly. Run daily morning briefing query at 06:00 UTC.

---

## 9. NOTIFICATION LAYER (TELEGRAM)

### 9.1 Commands

```
/status          — Current bankroll, open positions, daily P&L
/positions       — List all open positions with current mark-to-market
/history [n]     — Last n resolved trades
/performance     — Win rate, ROI, Sharpe, max drawdown
/calibration     — Current Brier score, reliability diagram
/agents          — Per-agent performance breakdown
/vpin            — Current VPIN score + toxic flow status (NEW v1.1)
/edge            — Edge decay report: current vs 30-day-ago metrics (NEW v1.1)
/wallets [n]     — Top n competitor wallet profiles (NEW v1.1)
/briefing        — Run morning intelligence briefing now (NEW v1.1)
/pause           — Pause trading (keep scanning)
/resume          — Resume trading
/mode [paper|live] — Switch execution mode
/threshold [n]   — Set minimum edge threshold
/bankroll [n]    — Update bankroll amount
/kill            — Emergency: halt all trading immediately (NEW v1.1)
```

### 9.2 Automated Alerts

```python
ALERT_TRIGGERS = {
    "trade_executed": "🟢 TRADE: {side} {amount} shares @ ${price} on {market}. Edge: {edge}%",
    "market_resolved": "📊 RESOLVED: {market} → {outcome}. P&L: ${pnl}",
    "daily_summary": "📈 Daily: {trades} trades, P&L: ${pnl}, Win rate: {wr}%",
    "circuit_breaker": "🔴 CIRCUIT BREAKER: Daily loss limit hit (-20%). Trading paused.",
    "kill_switch": "💀 KILL SWITCH: Total drawdown exceeded -40%. ALL TRADING HALTED. Manual restart required.",
    "toxic_flow": "⚠️ TOXIC FLOW: VPIN={vpin}. Position sizes reduced 50%. Review immediately.",
    "high_edge": "⚡ HIGH EDGE: {market} — Edge: {edge}%, not traded (review)",
    "cross_arb": "🔄 CROSS-PLATFORM ARB: {market} — Poly: {poly_price}, Kalshi: {kalshi_price}",
    "model_update": "🌤️ New model run detected: {model} @ {time}UTC",
    "edge_decay": "📉 EDGE DECAY: Top-10 weather wallet avg Sharpe dropped from {old} to {new}. Consider reducing size.",
    "adversarial_fill": "🎯 ADVERSARIAL: {pct}% of last 50 fills went against us. Possible informed counterparty.",
    "morning_briefing": "☀️ MORNING BRIEFING:\n{briefing_text}",
    "error": "❌ ERROR: {error_message}",
}
```

---

## 10. DATABASE SCHEMA

```sql
-- Markets being tracked
CREATE TABLE markets (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,          -- 'polymarket' or 'kalshi'
    city TEXT NOT NULL,
    station TEXT NOT NULL,
    date TEXT NOT NULL,               -- YYYY-MM-DD
    market_type TEXT NOT NULL,        -- 'daily_high', 'daily_low', 'precipitation'
    threshold_low REAL,
    threshold_high REAL,
    resolution_source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Price snapshots
CREATE TABLE market_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT REFERENCES markets(id),
    timestamp TIMESTAMP NOT NULL,
    yes_price REAL,
    no_price REAL,
    volume REAL,
    spread REAL
);

-- Weather forecasts
CREATE TABLE forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station TEXT NOT NULL,
    target_date TEXT NOT NULL,
    model TEXT NOT NULL,              -- 'gfs', 'ecmwf', 'hrrr', 'ensemble'
    forecast_timestamp TIMESTAMP NOT NULL,
    lead_time_hours INTEGER NOT NULL,
    temperature_forecast REAL,
    ensemble_members TEXT,            -- JSON array of 31 values
    precipitation_forecast REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading signals from agents
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT REFERENCES markets(id),
    agent_name TEXT NOT NULL,
    signal_type TEXT NOT NULL,        -- 'buy_yes', 'buy_no', 'no_trade'
    model_probability REAL,
    market_price REAL,
    edge REAL,
    confidence REAL,
    reasoning TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Executed trades (paper and live)
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT REFERENCES markets(id),
    platform TEXT NOT NULL,
    mode TEXT NOT NULL,               -- 'paper' or 'live'
    side TEXT NOT NULL,               -- 'buy_yes' or 'buy_no'
    price REAL NOT NULL,
    amount REAL NOT NULL,
    cost REAL NOT NULL,
    agent_name TEXT,
    model_probability REAL,
    edge_at_entry REAL,
    gaussian_prob REAL,
    ensemble_prob REAL,
    blended_prob REAL,
    status TEXT DEFAULT 'open',       -- 'open', 'won', 'lost', 'sold'
    resolution_value REAL,           -- Actual temperature/precipitation
    pnl REAL,
    resolved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance tracking
CREATE TABLE daily_performance (
    date TEXT PRIMARY KEY,
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    total_pnl REAL,
    bankroll_start REAL,
    bankroll_end REAL,
    brier_score REAL,
    gaussian_brier REAL,
    ensemble_brier REAL,
    sharpe_ratio REAL
);

-- Model calibration
CREATE TABLE calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    station TEXT NOT NULL,
    lead_time_hours INTEGER NOT NULL,
    rmse REAL,
    bias REAL,
    sample_size INTEGER,
    period_start TEXT,
    period_end TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agent performance
CREATE TABLE agent_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    period TEXT NOT NULL,             -- 'last_50', 'last_100', 'all_time'
    total_trades INTEGER,
    win_rate REAL,
    total_pnl REAL,
    avg_edge REAL,
    current_weight REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- VPIN tracking (NEW v1.1)
CREATE TABLE vpin_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT REFERENCES markets(id),
    timestamp TIMESTAMP NOT NULL,
    vpin REAL NOT NULL,
    v_buy REAL,
    v_sell REAL,
    window_size INTEGER,
    is_toxic BOOLEAN DEFAULT FALSE,
    action_taken TEXT                  -- 'none', 'reduce_50', 'pause'
);

-- Fill quality monitoring (NEW v1.1)
CREATE TABLE fill_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER REFERENCES trades(id),
    fill_price REAL,
    market_mid_at_fill REAL,
    slippage REAL,                    -- fill_price - mid at time of fill
    price_5s_after REAL,              -- market mid 5 seconds after fill
    adverse_move REAL,                -- how much price moved against us after fill
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Wallet intelligence profiles (NEW v1.1)
CREATE TABLE wallet_profiles (
    wallet_address TEXT PRIMARY KEY,
    profile_text TEXT NOT NULL,        -- Claude-generated profile
    total_trades INTEGER,
    win_rate REAL,
    sharpe_ratio REAL,
    total_pnl REAL,
    max_drawdown REAL,
    categories TEXT,                   -- JSON: ["weather", "crypto", ...]
    strategy_type TEXT,                -- "weather_arb", "latency", "manual", etc.
    is_automated BOOLEAN,
    last_active TIMESTAMP,
    profiled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Edge decay tracking (NEW v1.1)
CREATE TABLE edge_decay (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    check_date TEXT NOT NULL,
    top10_avg_sharpe REAL,
    top10_avg_winrate REAL,
    total_weather_volume_7d REAL,
    new_weather_wallets_7d INTEGER,
    our_sharpe REAL,
    our_winrate REAL,
    assessment TEXT,                   -- LLM assessment of edge health
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 11. CONFIGURATION (.env)

```bash
# ============ PLATFORM CREDENTIALS ============
POLYMARKET_PRIVATE_KEY=
POLYMARKET_PROXY_ADDRESS=
KALSHI_API_KEY=
KALSHI_PRIVATE_KEY=

# ============ LLM ============
ANTHROPIC_API_KEY=                    # For qualitative agent + error analysis
OLLAMA_MODEL=qwen3:8b                 # For high-volume routine analysis

# ============ TELEGRAM ============
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# ============ TRADING PARAMETERS ============
EXECUTION_MODE=paper                   # paper | live
INITIAL_BANKROLL=1000.0
MIN_EDGE_PERCENT=10.0                  # Minimum edge to trade (%)
KELLY_FRACTION=0.15                    # Kelly fraction for position sizing
MAX_DAILY_LOSS_PERCENT=20.0            # UPDATED v1.1: aligned with 0x8dxd standard
MAX_TOTAL_DRAWDOWN_PERCENT=40.0        # NEW v1.1: HARD KILL SWITCH
MAX_OPEN_POSITIONS=20
MAX_SINGLE_MARKET_EXPOSURE=0.08        # UPDATED v1.1: 8% max (was 15%)
MAX_SINGLE_PLATFORM_EXPOSURE=0.60

# ============ VOLATILITY TARGETING (NEW v1.1) ============
TARGET_VOLATILITY=0.10                 # 10% annualized target
MAX_LEVERAGE=2.0
MIN_ALLOCATION=0.20                    # Always 20% deployed minimum
NO_TRADE_BAND=0.10                     # Rebalance only if drift > 10%
VOL_DECAY_FACTOR=0.94                  # EWMA decay

# ============ VPIN / TOXIC FLOW (NEW v1.1) ============
VPIN_WINDOW=50                         # Fill window for VPIN calculation
VPIN_DANGER_THRESHOLD=0.70             # Toxic flow threshold
VPIN_ACTION=reduce_50                  # reduce_50 | pause | alert_only

# ============ STRATEGY ============
ENABLE_CONSERVATIVE_AGENT=true
ENABLE_AGGRESSIVE_AGENT=true
ENABLE_BLACKSWAN_AGENT=true
ENABLE_MEAN_REVERSION_AGENT=true       # NEW v1.1
ENABLE_QUALITATIVE_AGENT=true
ENABLE_CROSS_PLATFORM_ARB=true

# ============ MEAN REVERSION (NEW v1.1) ============
MEAN_REVERSION_LOOKBACK_HOURS=168      # 7-day window
MEAN_REVERSION_MIN_SIGMA=3.0           # Only trade at >3σ deviation

# ============ CITIES ============
# Comma-separated list of cities to trade
TARGET_CITIES=NYC,Chicago,Miami,Denver,LAX

# ============ SCHEDULING ============
FORECAST_POLL_INTERVAL_MINUTES=15     # How often to check for new forecasts
MARKET_SCAN_INTERVAL_MINUTES=5        # How often to scan market prices
RESOLUTION_CHECK_TIME=08:00            # When to check previous day resolutions

# ============ THRESHOLDS ============
CONSERVATIVE_MIN_EDGE=0.20
AGGRESSIVE_MIN_EDGE=0.08
BLACKSWAN_MAX_ENTRY_PRICE=0.05
BLACKSWAN_MIN_MODEL_PROB=0.10

# ============ LEARNING ============
WEIGHT_UPDATE_INTERVAL_TRADES=50      # Update model weights every N trades
AGENT_WEIGHT_UPDATE_INTERVAL=100      # Update agent weights every N trades
RMSE_RECALC_INTERVAL_DAYS=30          # Recalculate station RMSE monthly
ERROR_ANALYSIS_INTERVAL_DAYS=7        # LLM error analysis weekly

# ============ WALLET INTELLIGENCE (NEW v1.1) ============
WALLET_PROFILING_DAY=sunday            # When to re-profile wallets
WALLET_PROFILING_TIME=22:00            # UTC
EDGE_DECAY_CHECK_DAY=monday            # When to check edge compression
MORNING_BRIEFING_TIME=06:00            # UTC — daily intelligence briefing
WALLET_MIN_TRADES=50                   # Minimum trades to profile a wallet
PROFILING_LLM=claude-haiku-4-5-20251001  # Cheap model for mass profiling
QUERY_LLM=claude-sonnet-4-20250514    # Quality model for intelligence queries

# ============ GO LIVE THRESHOLDS ============
MIN_PAPER_TRADES_BEFORE_LIVE=200
MIN_WIN_RATE_BEFORE_LIVE=0.55
MIN_SHARPE_BEFORE_LIVE=1.0
MAX_DRAWDOWN_BEFORE_LIVE=0.15
```

---

## 12. IMPLEMENTATION PHASES

### Phase 1: Data Layer + Paper Trader (Week 1-2)

**Deliverables**:
1. Open-Meteo data fetcher (forecast + ensemble + ECMWF)
2. NOAA NWS API data fetcher
3. pmxt integration for market scanning (Polymarket + Kalshi)
4. SQLite database setup with all tables
5. Gaussian CDF probability model
6. Ensemble counting probability model
7. Basic edge calculation
8. Paper trader with position tracking
9. Telegram notifications (trade alerts + daily summary)
10. APScheduler for periodic polling

**Validation**: Run for 1 week, verify data fetching works, probabilities are reasonable, paper trades are logged.

### Phase 2: Multi-Agent Decision Layer (Week 3-4)

**Deliverables**:
1. LangGraph StateGraph setup for agent orchestration (reference: QuantAgent)
2. Conservative Agent implementation
3. Aggressive Agent (temperature laddering)
4. Black Swan Agent
5. Mean Reversion Agent (price-based, no weather opinion) — NEW v1.1
6. Risk Manager (Kelly + volatility targeting + VPIN + kill switch) — UPGRADED v1.1
7. Fill Monitor (adversarial detection) — NEW v1.1
8. Portfolio Manager (weighted signal aggregation)
9. Cross-platform arbitrage detector
10. Extended Telegram commands (/status, /positions, /performance, /vpin, /kill)

**Validation**: Run paper trader with full agent system for 2 weeks. Verify agent signals make sense. Check that risk manager correctly limits exposure and VPIN detection works.

### Phase 3: Self-Learning Loop (Week 5-6)

**Deliverables**:
1. Resolution tracker (fetch NWS CLI reports, resolve positions)
2. Brier score calibration
3. Model weight updater
4. Agent weight updater
5. Station RMSE recalculation
6. Historical bias correction table

**Validation**: Verify resolution tracking is accurate. Check that weights update in the expected direction.

### Phase 4: Qualitative Agent + Error Analysis (Week 7-8)

**Deliverables**:
1. Qualitative Factor Agent (LLM-powered)
2. Weekly error pattern analysis
3. Performance dashboard in Telegram
4. Backtest engine for historical validation

**Validation**: Run full system in paper mode for 4+ weeks. Accumulate 200+ paper trades. Evaluate go-live criteria.

### Phase 4.5: Wallet Intelligence Module (Week 8-9) — NEW v1.1

**Deliverables**:
1. Install polymarket-cli (Rust) on Mac Mini for data extraction
2. Wallet data collection script (scrape weather market holders + trades)
3. Wallet profiler using Claude Haiku (~$1.50 for 23K profiles)
4. JSON index builder (no vector DB needed)
5. Query engine using Claude Sonnet for intelligence queries
6. Edge decay monitor (weekly Sharpe comparison of top weather wallets)
7. Competitor tracker (alert when new high-volume weather wallet appears)
8. Daily morning briefing integration with Telegram (/briefing command)
9. 5 automated daily intelligence queries (strategy reverse-engineering, insider detection, edge decay, correlation mapping, morning briefing)

**Validation**: Run 5 daily prompts for 1 week. Verify intelligence output is actionable. Check edge decay monitor against manual analysis.

### Phase 5: Go Live (Week 10+)

**Prerequisites** (all must be true):
- [ ] 200+ paper trades completed
- [ ] Win rate > 55%
- [ ] Positive ROI after estimated fees
- [ ] Sharpe ratio > 1.0
- [ ] Max drawdown < 15%
- [ ] No critical bugs in last 2 weeks
- [ ] VPIN never triggered critical alert for 3+ consecutive days (NEW v1.1)
- [ ] Kill switch tested and confirmed working (NEW v1.1)
- [ ] Edge decay monitor shows stable or improving edge (NEW v1.1)

**Go-live process**:
1. Start with $100 real capital ($1-5 per trade, as per 0x8dxd methodology)
2. Run for 1 week at minimum position sizes
3. Compare live results to paper trader — must converge within 5% win rate
4. Verify VPIN and adversarial fill monitoring works on live fills
5. Scale to $500 after 50 successful live trades
6. Scale to $1,000-5,000 after 200 successful live trades
7. Only trade markets with >$50K liquidity (avoid thin markets where exits are impossible)

### Phase 6: BTC 5/15m Expansion (Future — Optional) — NEW v1.1

**NOT part of initial build. Evaluate after weather bot is profitable for 3+ months.**

Based on the Krajekis playbook (@Kropanchik thread) and 0x8dxd latency arbitrage analysis, there is an opportunity to expand into BTC 5-min and 15-min Up/Down markets on Polymarket. This is a fundamentally different strategy (latency arbitrage / TA-based vs forecast arbitrage) but shares infrastructure.

**Architecture reuse from WeatherEdge**:
- pmxt connector (already connected to Polymarket)
- Risk Manager (Kelly + vol targeting + VPIN + kill switch)
- Paper trader + resolution tracking
- Telegram notification layer
- SQLite database
- LangGraph agent orchestration

**New components needed**:
- Binance WebSocket feed for real-time BTC/ETH prices
- TA engine (RSI, MACD, VWAP, EMA21/50, ATR) — reference: QuantAgent
- Session logic (Asia/London/NY with different default modes)
- Intra-candle timing (entry at 10-5 minutes remaining in window)
- Visual chart analysis via LLM (QuantAgent pattern: LLM analyzes chart images)

**Key metrics from research**:
- Latency arbitrage window: ~2.7 seconds (Q1 2026, compressing)
- Krajekis-style TA: 55-65% win rate human, 85-98% bot
- 0x8dxd achieved 98% win rate over 26,738 trades
- WARNING: This edge is compressing faster than weather edge. Time-limited.

**Decision criteria to proceed**: Weather bot running profitably for 3 months + available development bandwidth + latency window still >1.5 seconds.

---

## 13. KEY RISKS AND MITIGATIONS

| Risk | Severity | Mitigation |
|------|----------|------------|
| Edge erosion (more bots entering) | HIGH | Edge decay monitor (Phase 4.5), wallet intelligence for early warning, diversify across cities + precipitation markets |
| Liquidity too thin | HIGH | Only trade markets >$50K liquidity, track orderbook depth via polymarket-cli, reject trades with >5% slippage |
| Toxic flow / adverse selection | HIGH | VPIN monitoring (>70% → reduce 50%), adversarial fill tracking, auto-pause when being hunted (NEW v1.1) |
| Total drawdown blowup | CRITICAL | Hard kill switch at -40% total drawdown, non-negotiable, requires manual restart (NEW v1.1) |
| Resolution source changes | MEDIUM | Monitor market rules for changes, alert if resolution source not found |
| API rate limits (Open-Meteo, NOAA) | LOW | Implement exponential backoff, cache aggressively, respect rate limits |
| Polymarket geo-restrictions (Italy) | HIGH | VPN + operational security, or focus on Kalshi via US-based entity |
| Model miscalibration | MEDIUM | Continuous Brier score monitoring, automatic weight adjustment, weekly LLM error analysis |
| Flash crash / market manipulation | MEDIUM | Circuit breakers, max 8% per position, time-based order expiry |
| DST/LST timezone confusion | HIGH | Hardcode timezone handling per station, extensive unit tests |
| Platform API changes | MEDIUM | pmxt handles this abstractly, but monitor for breaking changes |
| Position sizing bug in live mode | HIGH | Paper trade minimum 200 trades first, compare live vs paper divergence, 0x8dxd lesson: a 98% WR bot with a sizing bug = account blowup (NEW v1.1) |
| Correlated positions masking risk | MEDIUM | Track portfolio-level vol with covariance matrix, two weather contracts on same city = one position 2x size (NEW v1.1) |

---

## 14. REFERENCE REPOS AND RESOURCES

### Repos — Tier 1: Install as Dependencies

| Repo | Usage | URL |
|------|-------|-----|
| pmxt | Unified Python SDK for Polymarket + Kalshi trading | https://github.com/pmxt-dev/pmxt |
| langgraph | Agent orchestration framework (StateGraph) | https://github.com/langchain-ai/langgraph |

### Repos — Tier 2: Architecture Reference (Study + Adapt Patterns)

| Repo | Usage | URL |
|------|-------|-----|
| QuantAgent | **PRIMARY reference** for multi-agent LLM trading with LangGraph. 4 specialized agents (Indicator, Pattern, Trend, Risk) + visual chart analysis. Paper: arXiv:2509.09995 | https://github.com/Y-Research-SBU/QuantAgent |
| ai-hedge-fund | Multi-agent architecture with 18 investor-philosophy agents, LangGraph orchestration, backtester with Sharpe/Sortino/drawdown, React frontend | https://github.com/virattt/ai-hedge-fund |
| nofx | AI Debate Arena (bull vs bear vs analyst), CoT logging, Strategy Studio web UI, OpenClaw skills integration, competition mode between models | https://github.com/NoFxAiOS/nofx |

### Repos — Tier 3: Implementation Reference (Read Code, Don't Install)

| Repo | Usage | URL |
|------|-------|-----|
| polyBot-Weather | Gaussian CDF implementation reference, NOAA connector, Kelly sizing | https://github.com/hcharper/polyBot-Weather |
| polymarket-kalshi-weather-bot | GFS 31-member ensemble counting, dual-platform (Kalshi KXHIGH + Polymarket), Brier score calibration | https://github.com/suislanchez/polymarket-kalshi-weather-bot |
| prediction-market-analysis | Kalshi market microstructure analysis framework, spread/volume research | https://github.com/jon-becker/prediction-market-analysis |

### Repos — Tier 4: Operational Tools

| Repo | Usage | URL |
|------|-------|-----|
| polymarket-cli | **Official Polymarket CLI (Rust)**. Use for wallet intelligence, orderbook analysis, leaderboard research, competitor tracking. NOT for bot trading (use pmxt instead) | https://github.com/Polymarket/polymarket-cli |

### Key APIs

| API | URL | Auth | Update Frequency |
|-----|-----|------|-----------------|
| Open-Meteo Forecast | https://api.open-meteo.com/v1/forecast | None | Hourly (HRRR) |
| Open-Meteo Ensemble | https://api.open-meteo.com/v1/ensemble | None | Every 6h |
| Open-Meteo ECMWF | https://api.open-meteo.com/v1/ecmwf | None | Every 6h |
| Open-Meteo GFS/HRRR | https://api.open-meteo.com/v1/gfs | None | Hourly/6h |
| NOAA NWS | https://api.weather.gov/ | None | Varies |
| Polymarket CLOB | via pmxt SDK | Private key | Real-time |
| Kalshi | via pmxt SDK | API key + RSA | Real-time |

### Research Papers

| Paper | Relevance |
|-------|-----------|
| QuantAgent (arXiv:2509.09995, 2025) | **Primary architecture reference**. Multi-agent LLM framework for HFT with Indicator/Pattern/Trend/Risk agents. Outperforms neural + rule-based baselines on BTC and Nasdaq futures |
| GenCast (Nature, Dec 2024) | AI weather model outperforming ECMWF ENS on 96% of targets — potential future data source |
| IMDEA Polymarket Arbitrage Study (2025) | $40M extracted from Polymarket via market rebalancing + combinatorial arbitrage |
| BAMS Prediction Markets for Climate (Oct 2024) | Expert prediction market design for weather, LMSR-AMM mechanics |

### Key Thread References (Analyzed in v1.1)

| Thread | Author | Key Takeaway |
|--------|--------|-------------|
| BTC 5/15m Playbook | @Kropanchik | Session logic, volatility-first, TA stack (VWAP/RSI/MACD/EMA), intra-candle timing |
| Math Bot Strategies | @hanakoxbt | 4 strategies: market making, weather arb, mean reversion, kill switch (VPIN) |
| $313→$2.38M 0x8dxd | @adiix_official | Latency arb mechanics, 2.7s window, risk management params, Claude vs OpenClaw comparison |
| Volatility Targeting | @noisyb0y1 | Position size = target_vol / realized_vol, EWMA decay 0.94, leverage cap 2-3x |
| RAG Agent for Polymarket | @xmayeth | Wallet profiling pipeline, $12 build cost, 5 daily intelligence prompts, NOAA lag = 9-15 min confirmed |

---

## 15. TESTING REQUIREMENTS

### Unit Tests

```python
# test_probability.py
def test_gaussian_cdf_above_threshold():
    """Forecast 85°F, threshold 80°F, RMSE 2.5 → P > 0.95"""
    
def test_gaussian_cdf_at_threshold():
    """Forecast 80°F, threshold 80°F, RMSE 2.5 → P ≈ 0.50"""
    
def test_gaussian_cdf_below_threshold():
    """Forecast 75°F, threshold 80°F, RMSE 2.5 → P < 0.05"""
    
def test_ensemble_counting_all_above():
    """31/31 members above threshold → P = 1.0"""
    
def test_ensemble_counting_split():
    """16/31 members above → P ≈ 0.516"""
    
def test_bucket_probability():
    """Forecast 82°F, bucket [80,85], RMSE 2.5 → P ≈ 0.65"""
    
def test_edge_calculation():
    """Model prob 0.80, market price 0.40 → edge = 0.40"""
    
def test_kelly_criterion():
    """Edge 0.20, win prob 0.70 → position > 0 and < max"""
    
def test_bias_correction():
    """Known bias of +1.5°F → corrected forecast is 1.5 lower"""

# test_agents.py
def test_conservative_rejects_low_edge():
    """Edge < 20% → no trade signal"""
    
def test_aggressive_generates_ladder():
    """Valid opportunity → 3-bucket ladder order"""
    
def test_blackswan_rejects_expensive_contracts():
    """Contract priced > 5¢ → no trade even with edge"""
    
def test_risk_manager_circuit_breaker():
    """Daily loss > 20% → halt trading"""

def test_risk_manager_kill_switch():
    """Total drawdown > 40% → HARD HALT, requires manual restart"""

def test_risk_manager_vpin_detection():
    """VPIN > 0.70 → toxic flow alert, reduce size 50%"""

def test_risk_manager_vpin_safe():
    """VPIN = 0.30 → no toxic flow, normal operation"""

def test_volatility_targeting_high_vol():
    """High contract vol → smaller position (leverage < 1.0)"""

def test_volatility_targeting_low_vol():
    """Low contract vol → larger position, capped at max_leverage"""

def test_volatility_targeting_no_trade_band():
    """Drift < 10% → no rebalance (avoid transaction costs)"""

def test_mean_reversion_triggers_at_3sigma():
    """Price 3σ below 7-day mean → BUY signal"""

def test_mean_reversion_no_signal_within_band():
    """Price 1σ from mean → no trade"""

def test_portfolio_manager_aggregation():
    """3 agents agree BUY → final decision is BUY"""

def test_portfolio_manager_risk_veto():
    """All agents say BUY but risk manager vetoes → NO TRADE"""

# test_resolution.py
def test_timezone_lst_handling():
    """Verify LST (not DST) is used for Kalshi resolution"""
    
def test_resolution_source_matching():
    """Market KXHIGHNY resolves against Central Park CLI"""
```

### Integration Tests

```python
def test_full_cycle_paper_trade():
    """
    1. Fetch real weather data
    2. Scan real markets
    3. Generate signals
    4. Execute paper trade
    5. Verify trade is logged correctly
    """
    
def test_open_meteo_ensemble_fetch():
    """Fetch 31-member GFS ensemble for NYC, verify data shape"""
    
def test_pmxt_market_fetch():
    """Fetch active weather markets from Polymarket via pmxt"""
    
def test_noaa_nws_forecast_fetch():
    """Fetch NOAA forecast for Central Park coordinates"""

def test_langgraph_agent_workflow():
    """All 5 agents execute in LangGraph StateGraph, portfolio manager produces decision"""

def test_kill_switch_halts_all_activity():
    """Simulate -40% drawdown → verify all scheduled tasks pause, Telegram alert fires"""

def test_vpin_reduces_position_sizes():
    """Inject toxic fill pattern → verify next trade uses 50% smaller size"""
```

---

## 16. SUCCESS METRICS

### Go/No-Go Criteria (Paper Trading Phase)

| Metric | Minimum | Target |
|--------|---------|--------|
| Total paper trades | 200 | 500+ |
| Win rate | 55% | 65%+ |
| ROI (net of fees) | Positive | >20% annualized |
| Sharpe ratio | 1.0 | 2.0+ |
| Max drawdown | <15% | <10% |
| Brier score | <0.20 | <0.15 |
| Average edge at entry | >8% | >12% |

### Ongoing Performance Metrics (Live)

| Metric | Alert Threshold | Action |
|--------|----------------|--------|
| Rolling 50-trade win rate | <50% | Review model calibration |
| Daily P&L | <-$100 | Circuit breaker activates |
| Weekly P&L | Negative for 3 consecutive weeks | Full strategy review |
| Brier score drift | >0.25 | Recalibrate models |
| Agent weight collapse (any agent <0.05) | Triggered | Review agent strategy |

---

## END OF PRD

This document should be passed to Claude Code as the complete specification. Implementation should follow the phased approach in Section 12. All code should be Python 3.11+, well-typed with Pydantic models, agent orchestration via LangGraph, and thoroughly tested.

**v1.1 Priority changes vs v1.0:**
1. Phase 1 remains the same — get data layer + paper trader running first
2. Phase 2 now includes LangGraph orchestration, Mean Reversion agent, and hardened risk management (VPIN + vol targeting + kill switch) — these are NOT optional, they are core
3. Phase 4.5 (Wallet Intelligence) is the key competitive advantage — it tells us whether our edge is eroding before we lose money
4. Phase 6 (BTC expansion) is explicitly deferred — don't build it until weather is profitable for 3 months

**Non-negotiable risk parameters (from 0x8dxd and @hanakoxbt analysis):**
- Max 8% of portfolio in any single position
- Daily loss halt at -20%
- Total drawdown kill switch at -40% (requires manual restart)
- VPIN monitoring on every fill, auto-reduce at 70%+ toxic flow
- Paper trade minimum 200 trades with >55% win rate before any live capital
- Only trade markets with >$50K liquidity

**Key dependencies to install first:**
```bash
pip install pmxt langgraph scipy pandas numpy pydantic apscheduler aiohttp anthropic
# For wallet intelligence (install separately on Mac Mini):
# brew install polymarket-cli  (or cargo install polymarket-cli)
```
