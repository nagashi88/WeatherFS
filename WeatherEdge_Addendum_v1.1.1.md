# WeatherEdge PRD Addendum v1.1.1

## Document Info
- **Version**: 1.1.1
- **Date**: 2026-03-29
- **Companion to**: WeatherEdge PRD v1.1.0
- **Author**: Generated via deep-dive analysis (trace + 10-round Socratic interview + Planner/Architect/Critic consensus)
- **Purpose**: Resolves 20 specification gaps, corrects 2 formula bugs, and aligns internal contradictions identified in the PRD

> **READ THIS BEFORE IMPLEMENTING.** This addendum is a companion to `WeatherEdge_PRD_v1.1.md`. Every resolution explicitly references which PRD section it overrides. Where the addendum and PRD conflict, the addendum takes precedence.

---

## How to Use This Document

1. Read the original PRD first for domain context and overall architecture
2. Before implementing each phase, check the **Summary Table** below for gaps applicable to that phase
3. Each gap has: **Description** (what's wrong/missing), **Overrides** (PRD section reference), **Resolution** (concrete specification), and **Rationale** (why this decision was made)
4. Pseudocode is provided for formulas — it is specification, not implementation code

---

## Summary Table

| # | Gap | Category | Overrides PRD | Phase | Constraint |
|---|-----|----------|---------------|-------|------------|
| 1 | Kelly Criterion Formula Fix | Critical Corrections | Section 5.7, lines 700-705 | 1 | C1 |
| 2 | Fee-Aware Edge Calculation | Critical Corrections | Section 4.5, lines 475-484 | 1 | C4 |
| 3 | Probability Model Priority | Critical Corrections | Section 4.3, lines 434-448 | 1 | C5 |
| 4 | Revenue Projections | Critical Corrections | Section 1, line 30 | 1 | C9 |
| 5 | Simplified Agent Architecture | Architecture & Orchestration | Section 5, lines 507-800 | 2 | C2 |
| 6 | Agent Weight Initialization | Architecture & Orchestration | Section 5.8, lines 774-781 | 2 | C8 |
| 7 | Event-Driven Decision Pipeline | Architecture & Orchestration | Section 2.2, line 112 | 1 | C10 |
| 7a | Portfolio Manager Conflict Resolution | Architecture & Orchestration | Section 5.8, lines 790-799 | 2 | C2/C8 |
| 7b | Unknown Market Handling | Architecture & Orchestration | Section 3.2.2, lines 316-373 | 2 | C12 |
| 8 | Tiered Risk Scaling | Risk Management | Section 5.7, lines 664-665 | 2 | C3 |
| 9 | Position Lifecycle | Risk Management | NEW (not in PRD) | 2 | C6 |
| 10 | Kill Switch Recovery | Risk Management | Section 5.7, line 665 | 2 | C3/C6 |
| 11 | Same-City Correlation Grouping | Risk Management | Risk table, line 1546 | 2 | C15 |
| 12 | API Failure — Degraded Mode | Operational Resilience | NEW (not in PRD) | 1 | C13 |
| 13 | Logging Strategy | Operational Resilience | NEW (not in PRD) | 1 | C11 |
| 14 | Process Management & Graceful Shutdown | Operational Resilience | NEW (not in PRD) | 1 | C11 |
| 15 | Open-Meteo Licensing | Operational Resilience | Section 3.1.1, lines 206-208 | 1 | C7 |
| 16 | Secrets Management | Operational Resilience | NEW (not in PRD) | 1 | C11 |
| 17 | Testing Strategy | Testing | Section 15, lines 1614-1727 | 1 | C14 |
| 18 | Historical Market Price Data | Data & Backtesting | Section backtest/, lines 166-169 | 1 | — |
| — | Black Swan Agent | Deferred | Section 5.3, lines 549-567 | 3+ | C2 |

---

## Category 1: Critical Corrections

### Gap 1: Kelly Criterion Formula Fix

**Overrides PRD Section 5.7, lines 700-705**

**Description:** The PRD's Kelly formula derives `odds` from `win_probability` (the model's estimated true probability) instead of from `market_price` (the price paid for the contract). The formula simplifies to `f = (2p-1)/p` regardless of contract price, completely ignoring the market price. The `edge` parameter is passed into `calculate_position_size()` at line 684 but is never used.

**Resolution — Corrected Formula:**

```python
def calculate_kelly_size(
    win_probability: float,   # Model's estimated probability (e.g., 0.70)
    market_price: float,      # Current contract price (e.g., 0.35)
    kelly_fraction: float,    # Fractional Kelly (e.g., 0.15)
    bankroll: float,          # Current bankroll
) -> float:
    """
    Standard Kelly Criterion for prediction markets.
    b = decimal odds from MARKET PRICE (what you're paying)
    f* = (bp - q) / b
    """
    if win_probability <= market_price:
        return 0.0  # No edge, no bet

    b = (1.0 - market_price) / market_price  # Odds from MARKET price
    p = win_probability
    q = 1.0 - p

    kelly = (b * p - q) / b
    kelly = max(0.0, kelly)

    position = kelly * kelly_fraction * bankroll
    return position
```

**Worked Example 1 — Large edge (model_prob=0.70, market_price=0.35):**

| | PRD Formula (WRONG) | Correct Formula |
|---|---|---|
| b (odds) | (1/0.30) - 1 = 2.33 (from win_prob) | (1-0.35)/0.35 = 1.857 (from market_price) |
| Kelly f* | (0.70\*2.33 - 0.30)/2.33 = 0.571 | (1.857\*0.70 - 0.30)/1.857 = 0.539 |
| Position (15% Kelly, $1000) | $85.71 (capped at $80 by 8% max) | $80.77 (capped at $80 by 8% max) |
| **Verdict** | Similar due to cap, but formula logic is wrong | Correct — position scales with market price |

**Worked Example 2 — Small edge (model_prob=0.70, market_price=0.69):**

| | PRD Formula (WRONG) | Correct Formula |
|---|---|---|
| b (odds) | (1/0.30) - 1 = 2.33 (from win_prob) | (1-0.69)/0.69 = 0.449 (from market_price) |
| Kelly f* | (0.70\*2.33 - 0.30)/2.33 = 0.571 | (0.449\*0.70 - 0.30)/0.449 = 0.032 |
| Position (15% Kelly, $1000) | **$85.71** (capped at $80) | **$4.84** |
| **Verdict** | **~18x over-sized!** Bets $80 on a 1% edge | Correctly sizes a tiny bet for tiny edge |

**Rationale:** The PRD formula produces the same position size regardless of market price, which is financially nonsensical. A 1% edge trade should receive a tiny position, not an 8%-of-bankroll maximum. The hard caps (8% max, 15% fractional Kelly) contain the damage for large-edge trades but completely fail for small-edge trades where the correct position should be well below the cap.

---

### Gap 2: Fee-Aware Edge Calculation

**Overrides PRD Section 4.5, lines 475-484**

**Description:** The PRD calculates edge as `model_probability - market_price` but never accounts for trading costs. Fees and spreads eat into the edge, and at small bankroll sizes (~$1,000), trading costs of ~$135/month are material.

**Resolution:**

```python
# Platform fee estimates (as of March 2026 — verify before go-live)
PLATFORM_FEES = {
    "polymarket": {
        "explicit_fee": 0.00,           # No explicit trading fee
        "avg_spread": 0.03,             # ~3% typical bid-ask on weather markets
        "estimated_round_trip": 0.015,  # Half-spread as cost estimate
    },
    "kalshi": {
        "fee_on_winnings": 0.02,        # ~2% fee on winning trades
        "avg_spread": 0.02,             # ~2% typical spread
        "estimated_round_trip": 0.02,   # Combined estimate
    },
}

def calculate_net_edge(
    model_probability: float,
    market_price: float,
    platform: str,
) -> float:
    """
    Net edge = gross edge - estimated trading costs.
    Only trade when NET edge exceeds threshold.
    """
    fees = PLATFORM_FEES[platform]
    estimated_cost = fees["estimated_round_trip"]
    gross_edge = model_probability - market_price
    net_edge = gross_edge - estimated_cost
    return net_edge
```

**Updated min_edge_thresholds:**
- Conservative agent: `net_edge > 0.20` (was 0.20 gross — effectively now requires ~0.22-0.23 gross)
- Aggressive agent: `net_edge > 0.08` (was 0.08 gross — effectively now requires ~0.10 gross)

**Go-live criterion addition:** "Positive ROI after estimated fees" — track gross P&L and net P&L (after fee deduction) separately during paper trading. Both must be positive.

**Rationale:** At 75 trades/month with ~$60 avg position and ~2% round-trip cost, fees total ~$90-135/month. On a $1,000 bankroll targeting $100-500/month gross, fees consume 18-135% of returns. Fee-awareness is not optional at this scale.

---

### Gap 3: Probability Model Priority

**Overrides PRD Section 4.3, lines 434-448**

**Description:** The PRD blends Gaussian CDF and ensemble counting with a simple weighted average. But ensemble counting exists specifically for cases where the distribution is bimodal (e.g., a cold front that may or may not arrive). A weighted average of a correct model (ensemble) and a systematically wrong model (Gaussian assuming normality) corrupts the correct answer.

**Resolution — Ensemble-Primary with Gaussian Fallback:**

```python
def calculate_probability(
    forecast_temp: float,
    threshold: float,
    rmse: float,
    ensemble_temps: list[float] | None,
    direction: str = "above",
) -> tuple[float, str]:
    """
    Returns (probability, model_used).

    Priority:
    1. Ensemble counting if >=20 members available
    2. Gaussian CDF as fallback

    No weighted blending.
    """
    if ensemble_temps and len(ensemble_temps) >= 20:
        prob = ensemble_probability(ensemble_temps, threshold, direction)
        return prob, "ensemble"
    else:
        prob = gaussian_probability(forecast_temp, threshold, rmse, direction)
        return prob, "gaussian_fallback"
```

**Changes:**
- Remove `blended_probability()` function (PRD Section 4.3)
- Remove `gaussian_weight` config parameter
- Self-learning loop tracks Brier score per model and validates that ensemble > Gaussian in accuracy
- Bias correction (PRD Section 4.4) still applies to forecast inputs independently of model choice

**Rationale:** When the ensemble shows a bimodal distribution (15 members above threshold, 16 below — suggesting a front passage is uncertain), the Gaussian model centered on the mean will give ~50% probability regardless. The ensemble correctly captures the binary uncertainty. Blending dilutes this signal.

---

### Gap 4: Revenue Projections

**Overrides PRD Section 1, line 30**

**Description:** The PRD targets "$500-2,000/month passive income with $1,000-5,000 deployed capital" (10-40% monthly return). This is based on top-performer benchmarks (89% WR, Sharpe >3.0) that a new entrant should not expect to replicate immediately.

**Resolution — Conservative New-Entrant Projections:**

| Period | Capital | Monthly Target | Success Metric | Notes |
|--------|---------|---------------|----------------|-------|
| Months 1-3 | Paper → $100 live | Breakeven after fees | Sharpe > 1.0 | Validating the edge exists |
| Months 3-6 | $500 live | $50-200/month | Sharpe > 1.5, WR > 55% | Edge persistence check |
| Months 6+ | $1,000-5,000 live | $200-1,000/month | Sharpe > 2.0, WR > 60% | Scale only if edge stable |

**Contextual note:** The "89% win rate with Sharpe >3.0" figures from @xmayeth's analysis are **top-performer benchmarks from established wallets**, not targets for a new entrant. These are retained in the PRD as market context / ceiling estimates, not as operational targets.

**Rationale:** Setting aspirational targets as operational goals leads to premature capital deployment and disappointment. The tiered approach aligns capital at risk with demonstrated performance.

---

## Category 2: Architecture & Orchestration

### Gap 5: Simplified Agent Architecture

**Overrides PRD Section 5, lines 507-800**

**Description:** The PRD wraps all agents in a LangGraph StateGraph, but Conservative, Aggressive, and MeanReversion agents are deterministic threshold comparisons — no LLM calls, no complex state management. Wrapping if/else logic in a multi-agent framework adds latency, complexity, and failure modes.

**Resolution:**

**Strategy Functions** (plain Python, no LangGraph):
- `conservative_strategy(probabilities, market) -> Signal` — threshold checks, single-bucket selection
- `aggressive_strategy(probabilities, market) -> list[Signal]` — temperature laddering, multi-bucket
- `mean_reversion_strategy(price_history, market) -> Signal` — statistical deviation check

**LangGraph Components** (StateGraph orchestration):
- `QualitativeAgent` — LLM-powered analysis of microclimate factors, model bias, timing issues
- `PortfolioManager` — Signal aggregation with weighted voting (see Gap 7a)
- `ErrorAnalyzer` — Weekly LLM analysis of losing trade patterns

**Black Swan Agent:** Deferred to Phase 3+. Rationale: targets 1-5 cent contracts but the PRD mandates >$50K liquidity markets. Tail-price contracts have negligible liquidity. Will be added after market microstructure data is available to assess where liquidity actually exists at extreme prices.

**Updated directory structure impact:**
```
agents/
├── strategies/
│   ├── conservative.py      # Plain function
│   ├── aggressive.py        # Plain function
│   └── mean_reversion.py    # Plain function
├── qualitative.py           # LangGraph node (LLM)
├── portfolio_manager.py     # LangGraph node (aggregation)
├── risk_manager.py          # Plain class (deterministic)
└── base.py                  # Shared types: Signal, Order
```

**Rationale:** LangGraph is valuable for components that genuinely need LLM orchestration, state persistence, and error recovery. For deterministic threshold comparisons, it is overhead. This simplification reduces Phase 2 scope by ~30%.

---

### Gap 6: Agent Weight Initialization

**Overrides PRD Section 5.8, lines 774-781**

**Description:** PRD agent weights sum to 0.90 (missing 10% unexplained) with 5 agents. With Black Swan deferred, 4 agents remain. The weights need normalization.

**Resolution:**

| Agent | Initial Weight | Role |
|-------|---------------|------|
| Conservative | 0.30 | High-threshold, consistent edge |
| Aggressive | 0.30 | Temperature laddering, volume |
| MeanReversion | 0.20 | Price-based, no weather opinion |
| Qualitative | 0.20 | LLM adjustment factor |
| **Total** | **1.00** | |

- Weight updater runs every 100 trades, adjusts based on per-agent ROI
- Minimum weight floor: **0.05** per agent (prevents full suppression of any strategy)
- Maximum weight ceiling: **0.50** per agent (prevents single-strategy dominance)
- When Black Swan is added in Phase 3+, redistribute 0.10 proportionally from all agents

**Rationale:** Weights summing to 1.0 ensures the portfolio manager's weighted voting produces a clear signal. The self-learning loop is more important than initial weights — these are starting points, not permanent.

---

### Gap 7: Event-Driven Decision Pipeline

**Overrides PRD Section 2.2, line 112 and .env lines 1372-1373**

**Description:** Three independent schedulers (5-min market scan, 15-min forecast poll, 6-hour ensemble fetch) can trigger trading decisions concurrently, creating race conditions.

**Resolution:**

```
Scheduler Layer (data fetch ONLY)          Decision Layer (trading logic)
┌──────────────────────────────┐          ┌───────────────────────────┐
│ Market Scanner  (every 5m)   │──write──>│                           │
│ Forecast Poller (every 15m)  │──write──>│  Decision Loop (every 5m) │
│ Ensemble Fetch  (every 6h)   │──write──>│  - Read latest snapshot   │
│ HRRR Fetch      (every 1h)   │──write──>│  - Run all strategies     │
└──────────────────────────────┘          │  - Aggregate signals      │
         Data Store (SQLite)              │  - Risk checks            │
                                          │  - Execute orders         │
                                          └───────────────────────────┘
```

**Rules:**
1. Schedulers are **data fetchers only** — they update the database but NEVER trigger trading decisions
2. The **decision loop** runs on a fixed 5-minute interval, reads the latest data snapshot atomically, and generates all signals in a single pass
3. Only ONE decision loop instance runs at a time (APScheduler: `max_instances=1`, `coalesce=True`)
4. If a decision loop run takes longer than 5 minutes, the next run is skipped (not queued)

**APScheduler configuration:**
```python
scheduler = BackgroundScheduler()

# Data fetchers
scheduler.add_job(fetch_market_prices, 'interval', minutes=5, id='market_scan')
scheduler.add_job(fetch_forecasts, 'interval', minutes=15, id='forecast_poll')
scheduler.add_job(fetch_ensemble, 'interval', hours=6, id='ensemble_fetch')
scheduler.add_job(fetch_hrrr, 'interval', hours=1, id='hrrr_fetch')

# Decision loop (separate from data fetchers)
scheduler.add_job(
    run_decision_loop,
    'interval',
    minutes=5,
    id='decision_loop',
    max_instances=1,           # Never run two at once
    coalesce=True,             # Skip missed runs
    misfire_grace_time=120,    # Allow 2-minute grace for misfired jobs
)
```

**Rationale:** Separating data fetching from decision-making eliminates race conditions. The decision loop always reads a consistent data snapshot, regardless of which scheduler last updated the store.

---

### Gap 7a: Portfolio Manager Conflict Resolution Algorithm

**Overrides PRD Section 5.8, lines 790-799**

**Description:** The PRD's `aggregate_signals()` and `resolve_conflicts()` methods are both `pass` stubs with no algorithm specified.

**Resolution:**

```python
def aggregate_signals(self, agent_signals: dict[str, Signal]) -> list[Order]:
    """
    Weighted voting across all agents per market.

    1. For each market with at least one signal:
       a. Compute weighted_score = sum(agent_weight * signal_strength)
          - BUY signal: strength = confidence (0.0-1.0)
          - NO_TRADE signal: strength = 0.0
       b. If weighted_score > TRADE_THRESHOLD (default 0.50): generate order
       c. Apply risk manager checks (veto power)

    2. Risk Manager has ABSOLUTE VETO:
       - If any circuit breaker is triggered: NO TRADE regardless of consensus
       - If VPIN > danger threshold: reduce position size by 50%
       - If city exposure >= 16%: block new positions for that city
       - If total open positions >= max_open_positions: NO TRADE
    """
    orders = []
    for market_id, signals in group_by_market(agent_signals):
        weighted_score = sum(
            self.agent_weights[agent] * signal.confidence
            for agent, signal in signals.items()
            if signal.side != "no_trade"
        )

        if weighted_score > self.trade_threshold:
            # Determine side from majority of contributing agents
            side = majority_side(signals, self.agent_weights)

            # Risk manager check (veto power)
            risk_check = self.risk_manager.pre_trade_check(market_id, side)
            if risk_check["blocked"]:
                log.info(f"Risk veto: {market_id} - {risk_check['reason']}")
                continue

            # Size from corrected Kelly (Gap 1) with risk adjustments
            size = self.risk_manager.calculate_position_size(...)
            orders.append(Order(market_id, side, size))

    return orders
```

**Worked Example:**

| Agent | Weight | Signal | Confidence | Weighted |
|-------|--------|--------|------------|----------|
| Conservative (0.30) | 0.30 | NO_TRADE | 0.0 | 0.00 |
| Aggressive (0.30) | 0.30 | BUY_YES | 1.0 | 0.30 |
| MeanReversion (0.20) | 0.20 | BUY_YES | 1.0 | 0.20 |
| Qualitative (0.20) | 0.20 | BUY_YES | 0.8 | 0.16 |
| **Total** | | | | **0.66** |

Result: 0.66 > 0.50 threshold → **TRADE** (unless risk manager vetoes)

**Rationale:** Weighted voting naturally resolves conflicts — no special conflict-handling logic needed. The 0.50 threshold means >50% of weight must agree for a trade. The risk manager's veto is absolute and non-negotiable.

---

### Gap 7b: Unknown Market Handling

**Overrides PRD Section 3.2.2, lines 316-373**

**Description:** The `RESOLUTION_SOURCES` table is hardcoded for 5 Kalshi tickers and 2 Polymarket templates. PRD line 371 says "the bot MUST parse each market's rules before trading" for Polymarket, but no parser is specified.

**Resolution:**

```python
def process_market(self, market: Market) -> Signal | None:
    """
    Before generating any signals, verify market has a known resolution source.
    """
    resolution = RESOLUTION_SOURCES.get(market.platform, {}).get(market.ticker)

    if resolution is None:
        # Unknown market — skip and alert
        log.warning(f"Unknown market: {market.platform}/{market.ticker} - {market.title}")
        self.telegram.send_alert(
            f"Unknown market detected: {market.title}\n"
            f"Platform: {market.platform}\n"
            f"Ticker: {market.ticker}\n"
            f"Rules: {market.resolution_rules[:200]}\n"
            f"Action: Skipped. Add mapping to RESOLUTION_SOURCES if tradeable."
        )
        return None  # Do NOT trade

    # Known market — proceed with signal generation
    return self.generate_signal(market, resolution)
```

**Rules:**
1. NEVER trade a market without a verified resolution source mapping
2. Log all skipped markets for review (market_id, title, platform, resolution rules text)
3. New mappings are added **manually** to `RESOLUTION_SOURCES` after reviewing market rules
4. No LLM-assisted auto-mapping in Phase 1-2 — human verification required
5. Polymarket markets must have their resolution rules manually parsed before adding a mapping

**Rationale:** Trading against an unknown resolution source means your model might be predicting the wrong thing. The downside of skipping a market is a missed opportunity; the downside of trading a mis-mapped market is systematic losses. Skip + alert is the safe default.

---

## Category 3: Risk Management

### Gap 8: Tiered Risk Scaling

**Overrides PRD Section 5.7, lines 664-665 and .env lines 1337-1338**

**Description:** Paper trading validates at <15% max drawdown, but live allows -20% daily and -40% total kill switch. A system validated at 15% would immediately operate under 2.7x more permissive risk parameters.

**Resolution — Progressive Kill Switch:**

| Capital Phase | Trades | Total Drawdown Kill | Daily Loss Halt | Max Single Position |
|---------------|--------|--------------------|-----------------|--------------------|
| Paper trading | 0-200 | <15% (go-live gate) | N/A (not real money) | 8% (simulated) |
| $100 phase | 1-50 live | **-15%** | **-10%** | 8% |
| $500 phase | 51-200 live | **-25%** | **-15%** | 8% |
| $1,000+ phase | 200+ live | **-40%** | **-20%** | 8% |

**Tier transition:** Requires manual approval via Telegram `/scale` command:
```
/scale 500    → "Scaling to $500 phase. Kill switch: -25%. Daily halt: -15%. Confirm? [Y/N]"
/scale 1000   → "Scaling to $1K+ phase. Kill switch: -40%. Daily halt: -20%. Confirm? [Y/N]"
```

**Implementation:** Store current tier in the `daily_performance` table or a new `risk_tier` config table. The risk manager reads the current tier on each pre-trade check.

**Rationale:** Risk parameters should be earned, not assumed. The $100 phase uses the same conservative thresholds validated during paper trading. As the system proves itself with real capital, risk limits progressively loosen.

---

### Gap 9: Position Lifecycle — Hold-to-Resolution

**NEW — not in PRD (addresses undefined position exit strategy)**

**Description:** The PRD defines entry logic but never specifies how positions are managed after entry. The trades table includes a `'sold'` status, but no module generates sell signals.

**Resolution:**

**Default behavior:** All positions are held to market resolution (won/lost). Weather prediction markets resolve within 1-7 days. There are no routine early exits.

**No routine exits:**
- No stop-loss per position
- No take-profit targets
- No time-decay exit rules
- Positions are binary bets — they resolve or they don't

**Emergency exit only:** Triggered by kill switch (`/kill` command) or circuit breaker:
1. Cancel all unfilled limit orders immediately
2. For filled positions: mark as "abandoned-pending-resolution" (see Gap 10)
3. Halt all trading activity
4. Send Telegram alert: "EMERGENCY EXIT: All orders cancelled. {N} positions abandoned-pending-resolution."

**Unfilled order management:**
- Limit orders have a **TTL of 2 hours**
- If not filled within 2 hours: cancel automatically
- On next decision loop cycle: re-evaluate the market — place a new order only if edge still exists at current prices
- No cancel-replace logic — cancel and re-evaluate independently

**Updated trades table status values:**
```sql
-- Remove 'sold' status, add 'cancelled' and 'abandoned'
status TEXT DEFAULT 'open'  -- 'open', 'won', 'lost', 'cancelled', 'abandoned'
```

**Rationale:** Weather markets resolve on a known schedule (next-day NWS Climate Report). Early exit typically means selling at a loss due to spread costs. Hold-to-resolution is simpler, avoids spread costs, and matches the binary nature of the bets.

---

### Gap 10: Kill Switch Recovery Procedure

**Extends PRD Section 5.7, line 665**

**Description:** PRD says kill switch "requires manual restart" but never specifies what state is preserved, how orders are handled, or how to resume.

**Resolution — Kill Switch Sequence:**

**On kill switch activation (-40% total drawdown or /kill command):**

1. **Immediate:** Cancel all unfilled limit orders via pmxt
2. **Immediate:** Set `EXECUTION_MODE=halted` in state
3. **Immediate:** Send Telegram alert: "KILL SWITCH: All trading halted. Reason: {reason}. Manual restart required."
4. **Positions:** Mark all open positions as `abandoned-pending-resolution`
   - These positions CANNOT be sold in most prediction markets — they will resolve on-chain
   - Continue tracking resolutions for abandoned positions (self-learning loop needs clean data)
5. **Bankroll snapshot:** Record bankroll at halt time **EXCLUDING** outstanding abandoned positions' potential value (conservative accounting)
6. **Schedulers:** Pause data fetch schedulers (reduce API calls while halted)

**Manual restart checklist:**
```
/restart
1. Review what caused the kill switch
2. Check if abandoned positions have resolved since halt
3. Reconcile bankroll: apply P&L from any resolved abandoned positions
4. Verify API connections are healthy
5. Choose risk tier for restart (may want to drop a tier)
6. Resume trading: EXECUTION_MODE=paper (always restart in paper mode first)
7. After 10 successful paper trades: /mode live
```

**Learning loop continuity:** Resolution tracker continues running even during halt to resolve abandoned positions and update calibration metrics. The kill switch halts new trading, not data collection.

**Rationale:** Prediction market positions cannot be sold like stocks — they resolve on-chain. The bot must track resolutions even after halting to maintain accurate P&L and calibration data. Restarting in paper mode provides a safety buffer.

---

### Gap 11: Same-City Correlation Grouping

**Extends PRD risk table, line 1546**

**Description:** PRD acknowledges "two weather contracts on same city = one position 2x size" but has no implementation. Correlated positions mask true portfolio risk.

**Resolution:**

```python
def check_city_exposure(self, city: str, proposed_size: float) -> dict:
    """
    Pre-trade check: block new positions if city exposure exceeds 2x single-market cap.

    Max per city = 2 * max_single_market_exposure = 2 * 0.08 = 0.16 (16%)
    """
    current_city_exposure = sum(
        trade.cost for trade in self.open_positions
        if trade.city == city
    )
    total_exposure = current_city_exposure + proposed_size
    max_city_exposure = self.bankroll * 0.16  # 2x single-market cap

    if total_exposure > max_city_exposure:
        return {
            "blocked": True,
            "reason": f"City {city} exposure {total_exposure:.2f} exceeds 16% cap ({max_city_exposure:.2f})",
        }
    return {"blocked": False}
```

**Integration:** Called in `risk_manager.pre_trade_check()` alongside existing circuit breakers, VPIN, and position limits.

**Rationale:** Two concurrent NYC temperature contracts are effectively one bet with 2x exposure. The 16% cap (2x the 8% single-market limit) acknowledges that same-city bets are correlated but not identical (different thresholds, different dates).

---

## Category 4: Operational Resilience

### Gap 12: API Failure — Degraded Mode

**NEW — not in PRD**

**Description:** The PRD mentions "exponential backoff" in the risk table (line 1539) but has zero specification for what happens when data sources fail.

> **Note on C11/C13 tension:** The degraded mode is NOT infrastructure hardening — it is core trading logic. The bot must know what data it has before making decisions. This belongs in the decision loop, not in an ops layer. C11's "minimal" scope applies to infrastructure (no Docker, no Prometheus), not to data availability logic.

**Resolution — Retry Policy:**
- All API calls: **3 retry attempts** with exponential backoff (1s, 4s, 16s)
- On all retries exhausted: fall back per hierarchy below

**Source Hierarchy:**

| Failure | Fallback | Behavior | Telegram Alert |
|---------|----------|----------|----------------|
| Open-Meteo down | NOAA-only forecasts | Reduced model coverage (no ensemble, no ECMWF). Gaussian CDF only. Continue trading with wider uncertainty. | "DEGRADED: Open-Meteo unavailable. Running on NOAA-only." |
| NOAA down | Cached forecast data | Use cached forecasts with **2-hour TTL**. If cache age > 2h: skip decision cycle entirely. | "DEGRADED: NOAA unavailable. Using cached data (age: {X}min)." |
| Both weather APIs down | No trading | Skip decision cycle. Data fetchers continue retrying. | "HALTED: All weather data sources unavailable." |
| pmxt disconnected | Pause ALL trading | Cannot place or monitor orders. Resume on reconnect. | "HALTED: pmxt connection lost. No orders can be placed." |

**Decision loop integration:**
```python
def run_decision_loop(self):
    data = self.get_latest_data_snapshot()

    if data.pmxt_connected is False:
        log.warning("pmxt disconnected — skipping decision cycle")
        return

    if data.weather_age_minutes > 120:  # 2-hour TTL
        log.warning(f"Weather data stale ({data.weather_age_minutes}m) — skipping")
        return

    # Proceed with available data
    self.run_strategies(data)
```

**Rationale:** The bot should never trade on data it doesn't have or data that's too old. The hierarchy degrades gracefully — losing Open-Meteo is survivable (NOAA is the resolution source anyway), losing pmxt is not.

---

### Gap 13: Logging Strategy

**NEW — not in PRD**

**Resolution:**

```python
import logging
from logging.handlers import RotatingFileHandler

# Configuration
LOG_FILE = "logs/weatheredge.log"
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

# Module-level log levels
LOGGING_CONFIG = {
    "weatheredge.data":       logging.INFO,     # API calls, data freshness
    "weatheredge.agents":     logging.INFO,     # Strategy signals
    "weatheredge.execution":  logging.DEBUG,    # Order placement, fills
    "weatheredge.risk":       logging.WARNING,  # Circuit breakers, VPIN alerts
    "weatheredge.learning":   logging.INFO,     # Calibration updates
    "weatheredge.telegram":   logging.INFO,     # Notification delivery
}

handler = RotatingFileHandler(LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT)
handler.setFormatter(logging.Formatter(LOG_FORMAT))
```

**Directory:** Create `logs/` directory in project root. Add `logs/` to `.gitignore`.

**Rationale:** Rotating files prevent unbounded disk growth on Mac Mini. DEBUG level on execution module provides the audit trail needed for debugging order-related issues. WARNING level on risk module keeps the log focused on actionable events.

---

### Gap 14: Process Management & Graceful Shutdown

**NEW — not in PRD**

**Resolution — launchd plist:**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.weatheredge.bot</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/path/to/weatheredge/main.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/path/to/weatheredge/logs/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/path/to/weatheredge/logs/stderr.log</string>
</dict>
</plist>
```

**Graceful shutdown sequence** (on SIGTERM from launchd):
1. Set flag: `shutting_down = True`
2. Cancel all unfilled limit orders via pmxt
3. Persist current state to database (open positions, scheduler state)
4. Close SQLite connection cleanly
5. Exit process

**SQLite WAL mode** (set once at database initialization):
```python
connection.execute("PRAGMA journal_mode=WAL")
connection.execute("PRAGMA synchronous=NORMAL")
```

WAL mode allows concurrent reads from data fetchers while the decision loop writes trades.

**Rationale:** launchd is macOS-native, zero-dependency, and handles auto-restart on crash. WAL mode is the standard SQLite configuration for concurrent access patterns.

---

### Gap 15: Open-Meteo Licensing

**Overrides PRD Section 3.1.1, lines 206-208**

**Description:** Open-Meteo free tier is "non-commercial use" only (CC BY 4.0). The bot targets revenue, making it commercial.

**Resolution:**
- **Paper trading phase:** Use free tier. Paper trading generates no revenue — it is research/testing (non-commercial).
- **Before deploying live capital:** Upgrade to Open-Meteo paid API plan (~$29/month for 1M API calls).
- **Monthly operational cost budget:**

| Item | Cost |
|------|------|
| Open-Meteo API (paid) | ~$29/month |
| Anthropic API (Sonnet for qualitative agent) | ~$5-20/month (estimated) |
| Anthropic API (Haiku for wallet profiling) | ~$2/month |
| **Total** | **~$36-51/month** |

**Rationale:** $29/month is trivial relative to expected returns and is the legal compliance cost. Open-Meteo's paid tier also provides better rate limits and priority access to new model runs.

---

### Gap 16: Secrets Management

**NEW — not in PRD**

**Resolution:**

**Phase 1-2:** `.env` file with restricted permissions:
```bash
chmod 600 .env  # Owner read/write only
```

**Verify `.gitignore` includes:**
```
.env
weatheredge.db
logs/
```

**Secrets inventory:**
- `POLYMARKET_PRIVATE_KEY` — controls wallet funds (CRITICAL)
- `POLYMARKET_PROXY_ADDRESS` — wallet proxy
- `KALSHI_API_KEY` + `KALSHI_PRIVATE_KEY` — exchange access
- `ANTHROPIC_API_KEY` — LLM access (low risk if leaked — no funds)
- `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` — notification access

**Phase 5 upgrade path:** Migrate to macOS Keychain via `keyring` library:
```python
import keyring
keyring.set_password("weatheredge", "POLYMARKET_PRIVATE_KEY", "...")
key = keyring.get_password("weatheredge", "POLYMARKET_PRIVATE_KEY")
```

**Rationale:** `.env` with `chmod 600` is standard for single-machine deployments. The Keychain upgrade path is documented for when the system handles larger capital.

---

## Category 5: Testing

### Gap 17: Testing Strategy

**Overrides PRD Section 15, lines 1614-1727**

**Description:** PRD integration tests reference "real weather data" and "real markets" with no mocking strategy, no CI pipeline, and no test isolation.

**Resolution — Hybrid Approach:**

**Unit tests (run on every commit):**
- Hand-crafted fixtures with known values
- Mock external APIs at the function boundary
- Test deterministic logic only

```python
# Example fixtures
FIXTURE_FORECAST = {"station": "KNYC", "temp": 85.0, "model": "gfs", "lead_hours": 24}
FIXTURE_ENSEMBLE = [82.1, 83.4, 84.0, 85.2, ..., 87.8]  # 31 known values
FIXTURE_MARKET = {"price": 0.40, "threshold": 80, "direction": "above"}

def test_kelly_correct_formula():
    """Verify corrected Kelly from Gap 1."""
    size = calculate_kelly_size(win_probability=0.70, market_price=0.69, kelly_fraction=0.15, bankroll=1000)
    assert 4.0 < size < 5.0  # ~$4.70, NOT $80

def test_net_edge_subtracts_fees():
    """Verify fee-aware edge from Gap 2."""
    net = calculate_net_edge(model_probability=0.55, market_price=0.40, platform="polymarket")
    assert net < 0.15  # Gross edge 0.15, minus ~0.015 fee = ~0.135

def test_ensemble_primary_over_gaussian():
    """Verify ensemble-primary from Gap 3."""
    prob, model = calculate_probability(forecast_temp=85, threshold=80, rmse=2.5,
                                         ensemble_temps=[81]*10 + [86]*21, direction="above")
    assert model == "ensemble"
    assert abs(prob - 21/31) < 0.01

def test_city_exposure_cap():
    """Verify same-city correlation from Gap 11."""
    result = risk_mgr.check_city_exposure("NYC", proposed_size=100)
    # If NYC already has $140 exposure on $1000 bankroll (14%), adding $100 = 24% > 16% cap
    assert result["blocked"] is True
```

**Integration tests (tagged `@slow`, run separately):**
- Hit real APIs (Open-Meteo, NOAA — both free, no auth needed)
- Verify data shapes, endpoint availability, response parsing
- Accept network dependency and potential flakiness
- Run manually or on a weekly CI schedule

```python
@pytest.mark.slow
def test_open_meteo_ensemble_fetch():
    """Fetch 31-member GFS ensemble for NYC, verify data shape."""
    data = fetch_ensemble(lat=40.78, lon=-73.97)
    assert len(data["ensemble_members"]) >= 20  # At least 20 of 31

@pytest.mark.slow
def test_noaa_nws_forecast_fetch():
    """Fetch NOAA forecast for Central Park."""
    forecast = fetch_noaa_forecast(lat=40.78, lon=-73.97)
    assert "temperature" in forecast
```

**Running tests:**
```bash
pytest tests/ -m "not slow"          # Unit tests only (fast, offline)
pytest tests/ -m "slow"              # Integration tests (needs network)
pytest tests/                         # All tests
```

**Rationale:** Unit tests must be fast and deterministic for developer productivity. Integration tests verify real API contracts but are too slow and flaky for every commit. The `@slow` marker separates these concerns cleanly.

---

## Category 6: Data & Backtesting

### Gap 18: Historical Market Price Data

**Extends PRD backtest/ directory, lines 166-169**

**Description:** The backtest engine has a historical weather data source (NOAA NCEI) but no source for historical market prices. Without market prices, the backtest can validate probability accuracy (Brier score) but NOT trading P&L or strategy performance.

**Resolution:**

**Phase 1-4:** Acknowledge that backtesting is **forecast-accuracy-only**:
- Backtest validates: "Were our probability estimates calibrated?" (Brier score)
- Backtest CANNOT validate: "Would we have made money?" (requires historical orderbook data)
- This is acceptable because paper trading (Phase 1-2) serves as the live strategy validation

**Forward-collection (starts Phase 1):**
- The `market_prices` table (PRD lines 1154-1162) collects price snapshots every 5 minutes
- After 3+ months of collection, this data enables retroactive backtesting of strategy variants

**Phase 5+ options for historical data:**
- Polymarket: Explore subgraph data or `polymarket-cli` historical export capabilities
- Kalshi: Check if historical market data API is available
- Third-party: Evaluate if any data providers sell historical prediction market orderbook data

**Rationale:** The absence of historical market data is a known limitation, not a blocking issue. Paper trading provides real strategy validation. Forward-collecting market prices ensures backtesting capability grows over time.

---

## Phase Timeline Impact Assessment

| Phase | Original Estimate | Revised Estimate | Delta | Rationale |
|-------|-------------------|------------------|-------|-----------|
| Phase 1: Data Layer + Paper Trader | Week 1-2 | Week 1-2 | No change | Addendum adds fee model and degraded mode to Phase 1, but also removes blending complexity. Net neutral. |
| Phase 2: Multi-Agent Decision Layer | Week 3-4 | **Week 3** (1 week) | **-1 week** | Simplified architecture (Gap 5): 3 agents become plain functions, no LangGraph setup for them. Black Swan deferred. PM conflict resolution is straightforward weighted voting. |
| Phase 3: Self-Learning Loop | Week 5-6 | Week 4-5 | -1 week (shifted) | Starts earlier due to Phase 2 acceleration. Scope unchanged. |
| Phase 4: Qualitative Agent + Error Analysis | Week 7-8 | Week 6-7 | -1 week (shifted) | Starts earlier. Scope unchanged. |
| Phase 4.5: Wallet Intelligence | Week 8-9 | Week 7-8 | -1 week (shifted) | Starts earlier. Scope unchanged. |
| Phase 5: Go Live | Week 10+ | **Week 9+** | **-1 week** | Earlier due to cascading Phase 2 savings. |

**Net impact:** Simplified architecture saves approximately 1 week in Phase 2, which cascades forward. Total timeline compresses from ~10 weeks to ~9 weeks.

---

## Appendix: Constraint Traceability

| Constraint | Source | Gaps |
|------------|--------|------|
| C1: Kelly formula fix | Trace Lanes 1+2, Interview Round 1 | Gap 1 |
| C2: Simplified architecture | Trace Lane 1, Interview Round 4 | Gap 5, Gap 7a |
| C3: Tiered risk scaling | Trace Lane 3, Interview Round 3 | Gap 8, Gap 10 |
| C4: Fee-aware edge | Trace Lane 2, Interview Round 5 | Gap 2 |
| C5: Ensemble-primary model | Trace Lane 3, Interview Round 7 | Gap 3 |
| C6: Hold-to-resolution | Trace Lane 3, Interview Round 2 | Gap 9, Gap 10 |
| C7: Open-Meteo licensing | Trace Lane 1, Interview Round 3 | Gap 15 |
| C8: Agent weights | Trace Lane 2, Interview Round 5 | Gap 6, Gap 7a |
| C9: Revenue projections | Trace Lane 2, Interview Round 6 | Gap 4 |
| C10: Event-driven pipeline | Trace Lane 3, Interview Round 8 | Gap 7 |
| C11: Minimal ops | Trace Lane 3, Interview Round 8 | Gap 13, Gap 14, Gap 16 |
| C12: Unknown market handling | Trace Lane 3, Interview Round 9 | Gap 7b |
| C13: Degraded mode | Trace Lane 3, Interview Round 9 | Gap 12 |
| C14: Hybrid testing | Trace Lane 3, Interview Round 10 | Gap 17 |
| C15: Same-city correlation | Trace Lane 3, Interview Round 10 | Gap 11 |
