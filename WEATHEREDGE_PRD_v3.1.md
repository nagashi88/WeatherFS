# WeatherEdge Refactoring PRD v3.1
# Consolidated from 4 independent reviews + post-review calibration pass
# v3.1 changes: hardened thresholds as hypotheses, matched ablation, raised meta-cal bar, structural kills

---

## Purpose

This document guides Claude Code through a phased refactoring of WeatherEdge.
The core principle: **simplify first, validate, then add complexity only when proven useful.**

All changes preserve the existing architecture. No modules are deleted — they are disabled
via config flags so they can be re-enabled during ablation testing.

---

## OPERATIONAL HYPOTHESES (Not Constants)

The following thresholds are starting points, NOT ground truth. Treat them as hypotheses
to be validated and adjusted as data accumulates. Do NOT tune them to optimize backtest
PnL — that is overfitting disguised as policy tuning.

```python
# config.py — These are hypotheses. Log performance around each threshold
# and adjust only with strong out-of-sample evidence.

HYPOTHESES = {
    "brier_fail_gate": 0.25,             # Phase 0 gate. If too strict/loose after 50+ trades, revisit.
    "ablation_min_improvement": 0.01,    # Brier improvement needed to enable a module. May need
                                          # to be higher (0.02) if sample is small or noisy.
    "convergence_gate_ratio": 3.0,       # Agent agreement ratio. Starting guess — validate
                                          # after 100+ agent-voting trades.
    "burst_edge_discount": 0.05,         # Edge relaxation during burst window. If burst trades
                                          # underperform, tighten to 0.03 or remove entirely.
    "kelly_fraction": 0.08,              # Warm-up starts at 0.05. These are conservative guesses,
    "warmup_kelly_fraction": 0.05,       # not optimized values.
    "min_market_volume": 2000,           # Liquidity floor. Adjust based on observed fill quality.
    "max_spread_pct": 0.08,              # Spread cap. Same — adjust with execution data.
}
# RULE: Any change to HYPOTHESES values must be logged with justification and the
# out-of-sample evidence that motivated it. No silent tuning.
```

---

## PHASE 0 — SIMPLIFY & BUILD BASELINE B
**Timeline: Week 1 | Priority: CRITICAL | Risk: None (additive only)**

### 0.1 Create Feature Flags in config.py

Add a `MODULES` dict that controls every optional component. This is the kill switch system.
No module code is deleted — only gated behind these flags.

```python
# config.py — add this block

MODULES = {
    # Agents
    "agent_mean_reversion": False,      # STRUCTURAL KILL — do not re-enable without extraordinary evidence.
                                         # Rationale: price spikes on weather markets = new model run, not noise.
                                         # Unanimously rejected by all 4 reviewers.
    "agent_voting": False,              # DISABLED: use only blended forecast + fixed thresholds
    "agent_qualitative": False,         # DISABLED: underspecified, validate later

    # Models
    "timesfm": False,                   # DISABLED: weight=0.0, let learning loop promote it later
    "student_t_fallback": True,         # KEEP: protects against zero-probability extremes

    # Risk modules
    "vpin": False,                      # STRUCTURAL KILL — do not re-enable. VPIN requires HFT-style
                                         # order flow that does not exist on weather prediction markets.
                                         # Unanimously rejected by all 4 reviewers.
    "exit_pnl_stoploss": False,         # STRUCTURAL KILL — do not re-enable. On prediction markets,
                                         # P&L drawdown without model change = better odds, not more risk.
                                         # Replace with EV-based exit in Phase 1.
    "exit_edge_flip": True,             # KEEP: only rational exit trigger
    "exit_time_decay": True,            # KEEP: but review threshold

    # Learning
    "dynamic_model_weights": False,     # DISABLED: use fixed weights until 100+ trades
    "dynamic_agent_weights": False,     # DISABLED: agents are off anyway
    "weekly_error_analysis": True,      # KEEP: observational, doesn't affect decisions

    # Execution
    "cross_platform_arb": False,        # DISABLED: Phase 4 feature
    "limit_orders": True,               # KEEP
    "burst_mode": False,                # DISABLED: Phase 2 feature
}
```

### 0.2 Gate Every Module

For each module controlled by a flag, wrap the entry point:

```python
# Pattern for gating — apply to every controlled module
from config import MODULES

# In portfolio_manager.py or wherever agents are invoked:
def get_active_agents(self):
    agents = [self.conservative_agent, self.aggressive_agent, self.sellno_agent]
    if MODULES["agent_mean_reversion"]:
        agents.append(self.mean_reversion_agent)
    if MODULES["agent_qualitative"]:
        agents.append(self.qualitative_agent)
    return agents

# In probability_engine.py:
def blended_probability(self, ...):
    if not MODULES["agent_voting"]:
        # Baseline B mode: return blended forecast probability directly
        # No agent voting, no weighted consensus — just the model output
        return self._calculate_blended_prob(...)

# In risk_manager.py:
def check_toxic_flow(self, ...):
    if not MODULES["vpin"]:
        return False  # Never trigger VPIN reduction
```

**IMPORTANT**: Do NOT delete any module code. Only add `if MODULES["key"]:` gates.
Every module must remain importable and functional for ablation testing later.

### 0.3 Add Volume/Liquidity Filter

Add a hard pre-filter before any edge calculation runs. This saves compute and avoids
illiquidity traps.

```python
# config.py
MIN_MARKET_VOLUME = 2000      # Skip markets with < $2K cumulative volume
MAX_SPREAD_PCT = 0.08         # Skip markets with > 8% bid-ask spread
MIN_EDGE_FOR_TRADE = 0.10     # 10% gross edge minimum (already exists, confirm)
```

```python
# In opportunity builder (step [c] of decision loop):
def should_evaluate_market(self, market) -> bool:
    """Hard pre-filter. If False, skip all computation for this market."""
    if market.total_volume < config.MIN_MARKET_VOLUME:
        logger.info(f"SKIP {market.id}: volume {market.total_volume} < {config.MIN_MARKET_VOLUME}")
        return False
    if market.spread > config.MAX_SPREAD_PCT:
        logger.info(f"SKIP {market.id}: spread {market.spread:.1%} > {config.MAX_SPREAD_PCT:.0%}")
        return False
    return True
```

### 0.4 Anti-Leakage Temporal Guard (Lookahead Bias Prevention)

**This is the single most dangerous bug in any paper trading system.** If your bot
"executes" an order at the 12:00 market price using a GFS model run that became available
at 12:05, every metric downstream is contaminated. Brier scores, calibration curves,
PnL — all useless. And you'll never notice because everything looks great.

Insert a hard assertion in the execution path that makes it IMPOSSIBLE to trade on
future information. This is not a soft check — it's a system halt.

```python
# temporal_guard.py — NEW module, imported in paper_trader.py and any live execution

from datetime import datetime, timedelta

class TemporalLeakageError(Exception):
    """Raised when forecast data is newer than market price data.
    This is a hard system halt — not a warning, not a skip."""
    pass

# Maximum allowed staleness for market prices relative to forecast
MAX_PRICE_STALENESS = timedelta(minutes=0)  # Price must be SAME TIME or NEWER than forecast

def assert_no_lookahead(
    forecast_timestamp: datetime,
    market_price_timestamp: datetime,
    trade_decision_timestamp: datetime,
    forecast_source: str = "",
    market_id: str = "",
):
    """Call this BEFORE every trade execution. No exceptions.

    Rules:
    1. forecast_timestamp MUST be <= market_price_timestamp
       (you can't use a forecast that arrived after the price you're trading at)
    2. forecast_timestamp MUST be <= trade_decision_timestamp
       (you can't use a forecast from the future)
    3. market_price_timestamp MUST be <= trade_decision_timestamp
       (you can't trade at a price from the future)
    """

    if forecast_timestamp > market_price_timestamp:
        raise TemporalLeakageError(
            f"LOOKAHEAD DETECTED: forecast ({forecast_source}) at {forecast_timestamp} "
            f"is NEWER than market price at {market_price_timestamp} "
            f"for market {market_id}. "
            f"Delta: {forecast_timestamp - market_price_timestamp}. "
            f"THIS TRADE WOULD USE FUTURE INFORMATION. Halting."
        )

    if forecast_timestamp > trade_decision_timestamp:
        raise TemporalLeakageError(
            f"LOOKAHEAD DETECTED: forecast ({forecast_source}) at {forecast_timestamp} "
            f"is NEWER than decision time {trade_decision_timestamp}. "
            f"THIS IS A TIME TRAVEL BUG. Halting."
        )

    if market_price_timestamp > trade_decision_timestamp:
        raise TemporalLeakageError(
            f"LOOKAHEAD DETECTED: market price at {market_price_timestamp} "
            f"is NEWER than decision time {trade_decision_timestamp}. "
            f"THIS IS A TIME TRAVEL BUG. Halting."
        )
```

```python
# Integration in paper_trader.py — add to execute_trade() or place_order():

from temporal_guard import assert_no_lookahead, TemporalLeakageError

def execute_trade(self, order: TradeOrder):
    try:
        assert_no_lookahead(
            forecast_timestamp=order.forecast_timestamp,
            market_price_timestamp=order.market_price_timestamp,
            trade_decision_timestamp=datetime.utcnow(),
            forecast_source=order.primary_model,
            market_id=order.market_id,
        )
    except TemporalLeakageError as e:
        logger.critical(str(e))
        # DO NOT execute. Log the violation and skip.
        db.insert("leakage_violations", {
            "timestamp": datetime.utcnow(),
            "order": order.to_dict(),
            "error": str(e),
        })
        return None  # Trade rejected

    # ... proceed with normal execution
```

```python
# ALSO: add timestamp tracking to every forecast and price record in the DB

# In fetch_forecast_data():
forecast_record = {
    "model": "gfs",
    "data": forecast_data,
    "run_init_time": "2025-07-15T12:00:00Z",     # When the model run started
    "available_time": datetime.utcnow().isoformat(), # When YOUR system received it
    # available_time is what matters for leakage — not run_init_time
}

# In fetch_market_data():
price_record = {
    "market_id": market.id,
    "price": market.price,
    "fetched_at": datetime.utcnow().isoformat(),  # When YOUR system saw this price

    # Orderbook depth — CAPTURE FROM DAY 1 (Phase 0)
    # These fields are not used until Phase 4 (Execution Simulator), but if you
    # don't collect them now, you won't have historical data to build the simulator.
    # 3 extra fields today = 3 months of training data for slippage modeling later.
    "best_bid": market.best_bid,          # Best bid price
    "best_ask": market.best_ask,          # Best ask price
    "bid_size": market.bid_size,          # Contracts at best bid
    "ask_size": market.ask_size,          # Contracts at best ask
    # If the platform API doesn't expose bid/ask depth, log None — don't skip the fields.
    # Partial data is better than no data.
}
```

**The `available_time` vs `run_init_time` distinction is critical.** The GFS 12Z run
initializes at 12:00 UTC but the data doesn't become available until ~17:30-18:30 UTC.
If you stamp the forecast with 12:00 instead of 17:45, you're giving yourself 5+ hours
of phantom edge. Always use the time your system actually received the data.

### 0.5 Reduce Kelly Fraction for Warm-up

```python
# config.py — adjust existing values
KELLY_FRACTION = 0.08              # Was 0.15 — reduced per quant review
WARMUP_KELLY_FRACTION = 0.05       # Was 0.10 — reduced for safety
WARMUP_TRADE_THRESHOLD = 50        # Keep: 50 resolved trades before full Kelly
```

### 0.6 Baseline B Decision Flow (Simplified)

With modules disabled, the decision loop simplifies to:

```
run_decision_loop()
  → [a] Check weather data freshness (keep as-is)
  → [b] Load forecasts + market prices (keep as-is)
  → [b2] TEMPORAL GUARD: assert forecast_ts <= price_ts for all data (NEW)
  → [c] Pre-filter markets (NEW: volume + spread gate)
  → [c2] Calculate blended probability (HRRR-weighted, no TimesFM, no agent voting)
  → [c3] Calculate net edge both sides
  → [d] SKIP agent voting — use probability + edge directly
  → [e] If net_edge >= threshold: size via Kelly (reduced fraction)
  → [e2] TEMPORAL GUARD: assert_no_lookahead() before execution (NEW)
  → [f] Execute via PaperTrader (keep as-is)
  → [g] Exit Manager: edge flip + time decay ONLY (no P&L stop-loss)
  → [h] Record fills (keep as-is)
  → [i] Learning: weekly error analysis only (no weight updates yet)
  → [j] SKIP cross-platform arb
  → [k] Cancel expired orders (keep as-is)
```

### 0.7 Validation Gate for Phase 0

**DO NOT proceed to Phase 1 until:**

1. Baseline B runs in paper mode for 14+ days without crashes
2. At least 20 paper trades placed
3. Brier score computed and logged per trade
4. If Brier > 0.25 → debug the probability engine before adding anything

```python
# Add to main.py or a validation script:
def check_phase0_gate():
    trades = db.get_resolved_trades()
    if len(trades) < 20:
        logger.warning(f"Phase 0 gate: only {len(trades)} trades, need 20+")
        return False

    # Check for temporal leakage violations
    leakage_violations = db.get_all("leakage_violations")
    if leakage_violations:
        logger.critical(f"Phase 0 BLOCKED: {len(leakage_violations)} temporal leakage violations detected!")
        logger.critical("Fix the data pipeline before proceeding. Every metric is potentially contaminated.")
        return False

    brier = mean([(t.model_prob - t.outcome) ** 2 for t in trades])
    logger.info(f"Phase 0 Brier score: {brier:.4f}")
    if brier > config.HYPOTHESES["brier_fail_gate"]:
        logger.error(f"Phase 0 FAILED: Brier {brier:.4f} > {config.HYPOTHESES['brier_fail_gate']} — fix model before proceeding")
        return False
    return True
```

---

## PHASE 1 — CALIBRATE THE MODEL
**Timeline: Weeks 2-3 | Priority: HIGH | Depends on: Phase 0 gate passed**

### 1.1 Add Bias Correction per Station/Model/Month

The RMSE tells you how uncertain your forecast is. The bias tells you which direction
you're consistently wrong. Both are needed.

```python
# bias_correction.py — add alongside existing STATION_RMSE

STATION_BIAS = {}  # Populated by monthly updater
# Structure: STATION_BIAS[station][model][month] = mean(forecast - actual)

def get_bias_correction(station: str, model: str, month: int) -> float:
    """Returns the systematic bias to subtract from the forecast.
    Positive bias = model runs hot = subtract from forecast.
    Negative bias = model runs cold = add to forecast."""
    try:
        return STATION_BIAS[station][model][month]
    except KeyError:
        # Fall back to station-level mean across models
        try:
            all_model_biases = [
                STATION_BIAS[station][m][month]
                for m in STATION_BIAS.get(station, {})
                if month in STATION_BIAS[station].get(m, {})
            ]
            return mean(all_model_biases) if all_model_biases else 0.0
        except:
            return 0.0  # No data = no correction

def update_station_bias(station, model, month, forecasts, actuals):
    """Called by monthly RMSE updater. Rolling 30-day window."""
    residuals = [f - a for f, a in zip(forecasts, actuals)]
    if len(residuals) >= 10:  # Minimum sample for reliable bias
        bias = mean(residuals)
        STATION_BIAS.setdefault(station, {}).setdefault(model, {})[month] = bias
        logger.info(f"Bias update: {station}/{model}/month{month} = {bias:+.2f}°F")
```

**Integration point** — in `blended_probability()`, before computing CDF:

```python
# Before: forecast_temp used directly
# After: subtract bias
bias = get_bias_correction(station, model, current_month)
corrected_forecast = forecast_temp - bias
# Then use corrected_forecast in CDF calculation
```

### 1.2 Lower RMSE Floor for HRRR at Short Lead Times

```python
# bias_correction.py — modify RMSE floor logic

def get_rmse_floor(model: str, lead_hours: float) -> float:
    """Dynamic RMSE floor based on model capability at given lead time."""
    if model == "hrrr" and lead_hours < 12:
        return 1.5  # HRRR at 3km resolution is much tighter than 3°F
    elif model == "hrrr" and lead_hours < 24:
        return 2.0  # Still better than GFS but degrading
    else:
        return 3.0  # Default floor for GFS/ECMWF and long lead times

# In the RMSE lookup, replace the hardcoded floor:
# OLD: rmse = max(raw_rmse, 3.0)
# NEW:
rmse = max(raw_rmse, get_rmse_floor(model, lead_hours))
```

### 1.3 Replace P&L Stop-Loss with EV-Based Exit

The key insight: on prediction markets, a P&L drawdown without a model change means
your EV has IMPROVED (you're getting better odds on the same bet).

```python
# exit_manager.py — replace exit_pnl_stoploss logic

def evaluate_exit_ev(self, position, current_market_price, model_prob):
    """Exit only if holding has negative expected value vs closing."""

    # Cost of exit = spread cost to close position
    exit_cost = self._estimate_exit_cost(position, current_market_price)

    # EV of holding to resolution
    if position.side == "buy_yes":
        ev_hold = model_prob * (1.0 - position.entry_price) - (1 - model_prob) * position.entry_price
    else:  # buy_no
        ev_hold = (1 - model_prob) * position.entry_price - model_prob * (1.0 - position.entry_price)

    # EV of closing now
    if position.side == "buy_yes":
        ev_close = (current_market_price - position.entry_price) * position.contracts - exit_cost
    else:
        ev_close = (position.entry_price - current_market_price) * position.contracts - exit_cost

    # Hold if EV of holding > EV of closing
    should_exit = ev_close > ev_hold

    if should_exit:
        logger.info(
            f"EV EXIT: {position.market_id} | "
            f"EV_hold={ev_hold:.4f} EV_close={ev_close:.4f} | "
            f"model_prob={model_prob:.2f} market={current_market_price:.2f}"
        )

    return should_exit

# KEEP exit_edge_flip as-is (model reverses direction = always exit)
# KEEP exit_time_decay as-is (force close near resolution)
# DISABLE exit_pnl_stoploss via MODULES flag (already done in Phase 0)
```

### 1.4 Add Calibration Tracking & Reliability Diagram Data

```python
# calibration_tracker.py — enhance existing module

def compute_reliability_data(self, trades: list) -> dict:
    """Group trades into probability deciles and compute actual hit rates.
    This is the primary validation tool — if this isn't calibrated,
    nothing else matters."""

    buckets = {}  # {(0.5, 0.6): [outcomes], (0.6, 0.7): [outcomes], ...}

    for trade in trades:
        p = trade.model_prob
        # Bin into 10% buckets
        bucket_low = round(floor(p * 10) / 10, 1)
        bucket_high = round(bucket_low + 0.1, 1)
        key = (bucket_low, bucket_high)
        buckets.setdefault(key, []).append(trade.outcome)

    reliability = {}
    for (low, high), outcomes in sorted(buckets.items()):
        n = len(outcomes)
        if n < 5:
            continue  # Not enough data for this bucket
        actual_rate = mean(outcomes)
        predicted_mid = (low + high) / 2
        reliability[(low, high)] = {
            "n": n,
            "predicted": predicted_mid,
            "actual": actual_rate,
            "gap": actual_rate - predicted_mid,  # Positive = underconfident, negative = overconfident
        }

    return reliability

def compute_pnl_by_lead_time(self, trades: list) -> dict:
    """Separate PnL by lead time bucket to validate HRRR weighting thesis."""
    buckets = {"<12h": [], "12-24h": [], ">24h": []}
    for trade in trades:
        if trade.lead_hours < 12:
            buckets["<12h"].append(trade.pnl)
        elif trade.lead_hours < 24:
            buckets["12-24h"].append(trade.pnl)
        else:
            buckets[">24h"].append(trade.pnl)

    return {
        k: {"n": len(v), "total_pnl": sum(v), "avg_pnl": mean(v) if v else 0}
        for k, v in buckets.items()
    }
```

### 1.5 Phase 1 Validation Gate

```python
def check_phase1_gate():
    trades = db.get_resolved_trades()
    if len(trades) < 40:
        return False

    # Check calibration
    reliability = calibration_tracker.compute_reliability_data(trades)
    max_gap = max(abs(b["gap"]) for b in reliability.values()) if reliability else 1.0

    # Check PnL by lead time
    pnl_by_lt = calibration_tracker.compute_pnl_by_lead_time(trades)

    logger.info(f"Phase 1 gate:")
    logger.info(f"  Trades: {len(trades)}")
    logger.info(f"  Max calibration gap: {max_gap:.2%}")
    logger.info(f"  PnL <12h: ${pnl_by_lt['<12h']['total_pnl']:.2f}")
    logger.info(f"  PnL 12-24h: ${pnl_by_lt['12-24h']['total_pnl']:.2f}")
    logger.info(f"  PnL >24h: ${pnl_by_lt['>24h']['total_pnl']:.2f}")

    if max_gap > 0.15:
        logger.error("Phase 1 FAILED: calibration gap > 15% — model is overconfident or underconfident")
        return False

    return True
```

---

## PHASE 2 — BURST MODE & TIMING
**Timeline: Weeks 3-4 | Priority: HIGH | Depends on: Phase 1 gate passed**

### 2.1 Model Run Monitor

```python
# model_run_monitor.py — NEW module

class ModelRunMonitor:
    """Monitors NWP model data timestamps and triggers burst scans
    when fresh data is detected. The core thesis: edge is highest
    in the ~30min window after a new model run before the market adjusts."""

    SYNOPTIC_TIMES = ["00Z", "06Z", "12Z", "18Z"]

    def __init__(self):
        self.last_gfs_timestamp = None
        self.last_hrrr_timestamp = None
        self.last_ecmwf_timestamp = None

    def check_for_new_run(self, current_forecasts: dict) -> Optional[str]:
        """Returns model name if a new run is detected, else None."""
        new_runs = []

        gfs_ts = current_forecasts.get("gfs", {}).get("run_timestamp")
        if gfs_ts and gfs_ts != self.last_gfs_timestamp:
            self.last_gfs_timestamp = gfs_ts
            new_runs.append("gfs")

        hrrr_ts = current_forecasts.get("hrrr", {}).get("run_timestamp")
        if hrrr_ts and hrrr_ts != self.last_hrrr_timestamp:
            self.last_hrrr_timestamp = hrrr_ts
            new_runs.append("hrrr")

        ecmwf_ts = current_forecasts.get("ecmwf", {}).get("run_timestamp")
        if ecmwf_ts and ecmwf_ts != self.last_ecmwf_timestamp:
            self.last_ecmwf_timestamp = ecmwf_ts
            new_runs.append("ecmwf")

        return new_runs if new_runs else None
```

### 2.2 Burst Mode Integration in Decision Loop

```python
# In main.py decision loop:

model_monitor = ModelRunMonitor()

async def run_decision_loop():
    if not MODULES["burst_mode"]:
        # Standard 5-min cycle (Phase 0-1 behavior)
        await _standard_decision_cycle()
        return

    # Check for new model runs
    new_runs = model_monitor.check_for_new_run(latest_forecasts)

    if new_runs:
        logger.info(f"BURST MODE: New model run detected: {new_runs}")
        # Run aggressive scan with slightly relaxed edge thresholds
        await _burst_decision_cycle(
            edge_discount=0.05,  # Accept 5% less edge during burst window
            scan_all_markets=True,
            log_market_drift=True,  # Track if market moves toward our position
        )
    else:
        await _standard_decision_cycle()
```

### 2.3 Market Drift Tracking

```python
# market_drift_tracker.py — NEW module

class MarketDriftTracker:
    """After entering a trade, track whether the market price moves toward
    your model probability (positive signal) or away (negative signal).
    This is the ultimate validation: does the market eventually agree with you?"""

    def record_entry(self, trade_id, entry_price, model_prob, timestamp):
        db.insert("drift_tracking", {
            "trade_id": trade_id,
            "entry_price": entry_price,
            "model_prob": model_prob,
            "entry_time": timestamp,
            "price_snapshots": [],  # Will be appended every 15 min
        })

    def snapshot_price(self, trade_id, current_price, timestamp):
        """Called every 15 min for all open positions."""
        db.append_to("drift_tracking", trade_id, "price_snapshots", {
            "price": current_price,
            "time": timestamp,
        })

    def compute_drift_stats(self) -> dict:
        """After resolution, compute how often the market moved toward us.
        Positive drift_rate = market converges to our model = we're reading before the market.
        Negative drift_rate = market diverges = we're noise trading."""
        records = db.get_all("drift_tracking")
        drifts = []
        for r in records:
            if not r["price_snapshots"]:
                continue
            final_price = r["price_snapshots"][-1]["price"]
            entry_price = r["entry_price"]
            model_prob = r["model_prob"]
            # Did the market move toward our model probability?
            drift_toward_model = abs(final_price - model_prob) < abs(entry_price - model_prob)
            drifts.append(drift_toward_model)

        if not drifts:
            return {"n": 0, "drift_rate": None}
        return {
            "n": len(drifts),
            "drift_rate": mean(drifts),  # >0.5 = market converges to us = good
        }
```

### 2.4 Enable Burst Mode

```python
# Once Phase 2 code is implemented and tested:
# config.py
MODULES["burst_mode"] = True

# Also add to config.py:
BURST_EDGE_DISCOUNT = 0.05        # HYPOTHESIS — see OPERATIONAL HYPOTHESES section
BURST_WINDOW_MINUTES = 30         # Aggressive scanning for 30 min post-run
```

**CAVEAT on burst mode (from post-review):**
Burst mode is the part most sensitive to execution reality. Model run data arrives with
non-uniform delays (GFS 12Z may be available at 17:45 or 18:15 UTC depending on NOAA
processing). If the market adjusts faster than your data pipeline delivers, you're
chasing noise, not capturing edge.

Before relying on burst mode:
1. Log the actual delay between model run init time and when YOUR system sees the data
2. Log market price changes in the 60 min around each model run
3. If market moves >3% within 15 min of run availability → your window is tight, reduce
   BURST_WINDOW_MINUTES or increase BURST_EDGE_DISCOUNT to compensate
4. If market barely moves → your window is wide, burst mode is high-value

### 2.5 Phase 2 Validation Gate

```python
def check_phase2_gate():
    trades = db.get_resolved_trades()
    if len(trades) < 50:
        return False

    # Check overall Brier vs market-implied
    brier = mean([(t.model_prob - t.outcome) ** 2 for t in trades])
    market_brier = mean([(t.market_price - t.outcome) ** 2 for t in trades])

    # Check market drift
    drift = drift_tracker.compute_drift_stats()

    logger.info(f"Phase 2 gate:")
    logger.info(f"  Model Brier: {brier:.4f} vs Market Brier: {market_brier:.4f}")
    logger.info(f"  Drift rate: {drift['drift_rate']:.2%} (n={drift['n']})")

    if brier >= market_brier:
        logger.error("Phase 2 FAILED: model not better than market — do not go live")
        return False

    if drift["drift_rate"] and drift["drift_rate"] < 0.50:
        logger.warning("Phase 2 WARNING: market not converging to your positions")

    return True
```

---

## PHASE 3 — JUSTIFIED COMPLEXITY
**Timeline: Month 2-3 | Priority: MEDIUM | Depends on: Phase 2 gate + 50+ trades**

### CRITICAL RULE: Ablation-First Policy

**Every module re-enabled in Phase 3 must pass this test:**

1. Run system WITH the module for 30+ trades → compute Brier_with
2. Compare against Brier_without from the same period (or prior period)
3. If Brier_with < Brier_without by at least 0.01: module stays
4. If not: module stays OFF regardless of theoretical appeal

```python
# ablation_test.py

def run_ablation_test(module_name: str) -> dict:
    """Compare system performance with vs without a module.
    Call this BEFORE permanently enabling any Phase 3 module.

    CRITICAL: Naive period-based comparison is contaminated by regime dependence
    (season, city mix, volatility, market quality). We must match trades by context."""

    trades_with = db.get_trades_where(module_active=module_name)
    trades_without = db.get_trades_where(module_inactive=module_name)

    if len(trades_with) < 30 or len(trades_without) < 30:
        return {"status": "insufficient_data", "module": module_name}

    # MATCHED comparison: group by (season, lead_time_bucket, market_type)
    # and compare Brier within each stratum, then aggregate
    strata_results = []

    for season in ["summer", "winter", "shoulder"]:
        for lt_bucket in ["<12h", "12-24h", ">24h"]:
            tw = [t for t in trades_with
                  if _get_season(t.timestamp) == season and _get_lt_bucket(t.lead_hours) == lt_bucket]
            two = [t for t in trades_without
                   if _get_season(t.timestamp) == season and _get_lt_bucket(t.lead_hours) == lt_bucket]

            if len(tw) < 5 or len(two) < 5:
                continue  # Not enough data in this stratum

            brier_w = mean([(t.model_prob - t.outcome) ** 2 for t in tw])
            brier_wo = mean([(t.model_prob - t.outcome) ** 2 for t in two])
            n_matched = min(len(tw), len(two))

            strata_results.append({
                "season": season, "lead_time": lt_bucket,
                "brier_with": brier_w, "brier_without": brier_wo,
                "improvement": brier_wo - brier_w,
                "n": n_matched,
            })

    if not strata_results:
        return {"status": "insufficient_matched_data", "module": module_name}

    # Weighted average improvement (weighted by stratum size)
    total_n = sum(s["n"] for s in strata_results)
    weighted_improvement = sum(s["improvement"] * s["n"] / total_n for s in strata_results)

    # Require improvement in MAJORITY of strata, not just aggregate
    improving_strata = sum(1 for s in strata_results if s["improvement"] > 0)
    total_strata = len(strata_results)

    min_improvement = config.HYPOTHESES["ablation_min_improvement"]
    verdict = "ENABLE" if (weighted_improvement > min_improvement
                           and improving_strata > total_strata / 2) else "KEEP_OFF"

    result = {
        "module": module_name,
        "weighted_improvement": weighted_improvement,
        "improving_strata": f"{improving_strata}/{total_strata}",
        "strata_detail": strata_results,
        "verdict": verdict,
    }

    logger.info(f"Ablation test: {module_name} → {result['verdict']} "
                f"(Δ Brier = {weighted_improvement:+.4f}, "
                f"strata {improving_strata}/{total_strata})")
    return result


def _get_season(ts):
    m = ts.month
    if m in (6, 7, 8): return "summer"
    if m in (12, 1, 2): return "winter"
    return "shoulder"

def _get_lt_bucket(hours):
    if hours < 12: return "<12h"
    if hours < 24: return "12-24h"
    return ">24h"
```

### 3.1 Re-enable Agent Voting with Convergence Gate

```python
# ONLY after ablation test passes for "agent_voting"

# config.py
MODULES["agent_voting"] = True
CONVERGENCE_GATE_RATIO = 3.0  # Require 3:1 directional agreement

# portfolio_manager.py — add convergence check
def aggregate_signals(self, signals: list) -> Optional[TradeOrder]:
    # Count directional agreement
    buy_weight = sum(s.confidence * AGENT_WEIGHTS[s.agent]
                     for s in signals if s.side == "buy_yes")
    sell_weight = sum(s.confidence * AGENT_WEIGHTS[s.agent]
                      for s in signals if s.side == "buy_no")

    # Convergence gate: require strong directional agreement
    if max(buy_weight, sell_weight) == 0:
        return None
    ratio = max(buy_weight, sell_weight) / max(min(buy_weight, sell_weight), 0.01)
    if ratio < config.CONVERGENCE_GATE_RATIO:
        logger.info(f"Convergence gate blocked: ratio {ratio:.1f} < {config.CONVERGENCE_GATE_RATIO}")
        return None

    # Proceed with normal weighted voting...
```

### 3.2 Let TimesFM Earn Its Weight

```python
# ONLY after ablation test passes for "timesfm"

# TimesFM starts at weight 0.0 and the learning loop promotes it based on Brier score.
# Do NOT set a manual weight. The system decides.

# config.py
MODULES["timesfm"] = True
TIMESFM_INITIAL_WEIGHT = 0.0  # Learning loop will increase if earned
TIMESFM_MAX_WEIGHT = 0.30     # Cap to prevent over-reliance

# In the model weight updater (existing code), ensure TimesFM is included
# in the inverse-Brier weighting but starts from 0.0
```

### 3.3 Add Shrinkage to RMSE Estimates

```python
# bias_correction.py — modify RMSE updater

SHRINKAGE_FACTOR = 0.3  # 30% weight to global mean, 70% to station-specific

def get_shrunk_rmse(station, model, lead_days) -> float:
    """Hierarchical shrinkage: station-specific RMSE pulled toward global mean.
    Prevents overfitting when station has few observations."""

    station_rmse = STATION_RMSE.get(station, {}).get(model, {}).get(lead_days)
    global_rmse = _get_global_mean_rmse(model, lead_days)

    if station_rmse is None:
        return global_rmse  # No station data: use global

    # Count observations for this station/model/lead_days combo
    n_obs = _get_observation_count(station, model, lead_days)

    if n_obs < 20:
        # Few observations: lean heavily on global
        shrinkage = SHRINKAGE_FACTOR + (1 - SHRINKAGE_FACTOR) * (1 - n_obs / 20)
    else:
        shrinkage = SHRINKAGE_FACTOR

    shrunk = shrinkage * global_rmse + (1 - shrinkage) * station_rmse
    floor = get_rmse_floor(model, lead_days * 24)  # Convert days to hours
    return max(shrunk, floor)
```

### 3.4 Add Seasonal Agent Weights

```python
# config.py
SEASONAL_AGENT_WEIGHTS = {
    "summer": {  # Jun-Aug
        "conservative": 0.35,
        "aggressive": 0.30,
        "sellno": 0.35,
    },
    "winter": {  # Dec-Feb
        "conservative": 0.50,
        "aggressive": 0.15,
        "sellno": 0.35,
    },
    "shoulder": {  # Mar-May, Sep-Nov
        "conservative": 0.40,
        "aggressive": 0.25,
        "sellno": 0.35,
    },
}

def get_current_season():
    month = datetime.now().month
    if month in (6, 7, 8): return "summer"
    if month in (12, 1, 2): return "winter"
    return "shoulder"
```

### 3.5 Meta-Model Calibration (Replace Ad-Hoc Weights)

```python
# meta_calibrator.py — NEW module (Phase 3 only, after 200+ trades)

"""
Replace manual lead-time weights with a logistic regression meta-model.
Input features: individual model probabilities + context.
Output: calibrated event probability.

ONLY build this after you have 300+ resolved trades with reasonable diversity
across seasons, stations, and lead times. 200 is the absolute floor if trades
are well-distributed; 300+ is safer because weather trades are NOT independent
(same-day trades on nearby cities share atmospheric forcing).

Feature count must be kept LOW to avoid overfitting:
- 4 model probabilities + lead_hours + month = 6 features (preferred)
- Add spread/volume only if 500+ trades available
- NEVER add more features than trades/30 (rule of thumb)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np

class MetaCalibrator:
    def __init__(self):
        self.model = None
        self.isotonic = None

    def train(self, features: np.ndarray, outcomes: np.ndarray):
        """
        Preferred features (keep this list SHORT):
        [gfs_prob, ecmwf_prob, hrrr_prob, timesfm_prob, lead_hours, month]
        = 6 features → need 300+ trades minimum (50x features rule)

        Extended features (only if 500+ trades):
        + [hour_of_day, spread, volume] = 9 features

        outcomes: binary (0 or 1)
        """
        n_features = features.shape[1]
        min_trades = max(300, n_features * 50)  # 50x features rule

        if len(outcomes) < min_trades:
            logger.warning(f"MetaCalibrator: need {min_trades}+ trades "
                          f"for {n_features} features, have {len(outcomes)}. Skipping.")
            return

        # Stage 1: Logistic regression for probability
        self.model = LogisticRegression(C=1.0, max_iter=1000)
        self.model.fit(features, outcomes)

        # Stage 2: Isotonic regression for calibration
        raw_probs = self.model.predict_proba(features)[:, 1]
        self.isotonic = IsotonicRegression(out_of_bounds="clip")
        self.isotonic.fit(raw_probs, outcomes)

        logger.info("MetaCalibrator trained on {len(outcomes)} trades")

    def predict(self, features: np.ndarray) -> float:
        if self.model is None:
            return None  # Fall back to blended_probability
        raw = self.model.predict_proba(features.reshape(1, -1))[:, 1][0]
        calibrated = self.isotonic.predict([raw])[0]
        return float(calibrated)
```

---

## PHASE 4 — SCALE (Month 3+)
**Only after 200+ live trades with positive PnL**

### 4.1 Cross-Platform Arbitrage (Auto-Execute)
Enable `MODULES["cross_platform_arb"]` and convert logging to execution.

### 4.2 Execution Simulator
Build realistic fill model: partial fills, slippage as f(volume, spread), adverse selection post-fill.

**DATA DEPENDENCY**: This module requires historical orderbook depth data (best_bid, best_ask,
bid_size, ask_size) collected since Phase 0. If `fetch_market_data()` has been capturing these
fields since day 1, you'll have 3+ months of data to train a realistic slippage model.
If not, you'll need to collect for at least 30 days before building the simulator.

The execution simulator should answer: "given this orderbook state, if I place a market order
for N contracts, what price do I actually get?" This is the bridge from paper to live.

### 4.3 Correlation-Aware Kelly
Replace per-city caps with estimated correlation matrix between positions.
Requires 500+ trades for reliable correlation estimates.

### 4.4 Scale Capital
Only after Sharpe > 1.0 over 3+ months of live trading.

---

## ABLATION TEST SCHEDULE

Run these in order. Each test requires 30+ trades with module ON and 30+ trades with module OFF.

| Order | Module | Enable flag | Expected impact |
|-------|--------|-------------|-----------------|
| 1 | HRRR weighting | Already on in Baseline B | Validate vs GFS-only |
| 2 | Student-T tails | `student_t_fallback` | Compare Brier on extreme events |
| 3 | TimesFM error correction | `timesfm` | Bias correction improvement |
| 4 | Agent voting | `agent_voting` | Consensus vs single-model |
| 5 | Burst mode | `burst_mode` | Timing improvement |
| 6 | Meta-calibrator | N/A (replaces blending) | Calibration improvement |

---

## KEY METRICS TO LOG EVERY TRADE

```python
# Every trade record must include:
trade_record = {
    "trade_id": str,
    "timestamp": datetime,
    "market_id": str,
    "platform": str,               # "kalshi" or "polymarket"
    "station": str,
    "city": str,
    "threshold": float,
    "lead_hours": float,
    "side": str,                    # "buy_yes" or "buy_no"

    # Probabilities
    "model_prob": float,            # Blended probability at entry
    "gfs_prob": float,              # Individual model probs for ablation
    "ecmwf_prob": float,
    "hrrr_prob": float,
    "timesfm_prob": float,          # None if disabled
    "market_price": float,          # Market price at entry

    # Edge
    "gross_edge": float,
    "net_edge": float,
    "estimated_cost": float,

    # Execution
    "entry_price": float,
    "contracts": float,
    "kelly_fraction_used": float,

    # Orderbook snapshot at entry (Phase 0 — collect from day 1 for Phase 4 execution simulator)
    "best_bid": float,                  # Best bid price at entry time
    "best_ask": float,                  # Best ask price at entry time
    "bid_size": float,                  # Contracts available at best bid
    "ask_size": float,                  # Contracts available at best ask
    "mid_price": float,                 # (best_bid + best_ask) / 2

    # Context
    "volume_at_entry": float,
    "spread_at_entry": float,           # best_ask - best_bid (redundant with bid/ask but explicit)
    "active_modules": list,             # Which MODULES were True at trade time
    "is_burst_mode": bool,

    # Resolution (filled after outcome)
    "outcome": int,                 # 0 or 1
    "pnl": float,
    "brier_contribution": float,    # (model_prob - outcome)^2
}
```

---

## SUMMARY: What's ON vs OFF at each phase

| Module | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|---------|
| HRRR-weighted blending | ON | ON | ON | ON |
| Student-T fallback | ON | ON | ON | ON |
| GFS/ECMWF ensemble | ON | ON | ON | ON |
| Volume/spread filter | ON | ON | ON | ON |
| Orderbook depth logging | ON | ON | ON | ON |
| Temporal leakage guard | ON | ON | ON | ON |
| Edge flip exit | ON | ON | ON | ON |
| Time decay exit | ON | ON | ON | ON |
| METAR override | ON | ON | ON | ON |
| Bias correction | — | ON | ON | ON |
| RMSE floor 1.5°F | — | ON | ON | ON |
| EV-based exit | — | ON | ON | ON |
| Reliability tracking | — | ON | ON | ON |
| Burst mode | — | — | ON | ON |
| Market drift tracking | — | — | ON | ON |
| Agent voting + gate | — | — | — | If ablation passes |
| TimesFM (weight=0) | — | — | — | If ablation passes |
| Shrinkage RMSE | — | — | — | ON |
| Seasonal weights | — | — | — | ON |
| Meta-calibrator | — | — | — | If 200+ trades |
| MeanReversion | STRUCTURAL KILL | STRUCTURAL KILL | STRUCTURAL KILL | STRUCTURAL KILL |
| VPIN | STRUCTURAL KILL | STRUCTURAL KILL | STRUCTURAL KILL | STRUCTURAL KILL |
| P&L stop-loss | STRUCTURAL KILL | STRUCTURAL KILL | STRUCTURAL KILL | STRUCTURAL KILL |
| Cross-platform arb | OFF | OFF | OFF | Phase 4 |

---

## INSTRUCTIONS FOR CLAUDE CODE

1. Read this entire document before making any changes
2. Start with Phase 0 only — do not jump ahead
3. Never delete module code — only gate it behind `MODULES` flags
4. Run existing tests after each change to confirm nothing breaks
5. Commit after each numbered section (0.1, 0.2, etc.) with message format:
   `[WE-P0.1] Add MODULES feature flags to config.py`
6. After Phase 0 is complete, STOP and wait for human to confirm gate passage
7. Each phase depends on the previous phase's gate — do not skip gates
8. Log everything — when in doubt, log more, not less
9. If unsure about existing code structure, ask before modifying
10. The bias_correction.py and probability_engine.py are the most sensitive files —
    make minimal, targeted changes with clear comments
