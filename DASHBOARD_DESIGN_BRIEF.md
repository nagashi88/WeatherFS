# WeatherEdge Dashboard — Design Brief

Input per **Claude Design**. Riprende il live rich dashboard e lo trasforma in web app.
CLI resta solo per test/debug; UX/UI operativa vive qui.

---

## 🏗 Stack (locked)

| Layer | Scelta | Perché |
|---|---|---|
| Backend | **FastAPI** (Python) | Vive accanto al bot, legge `weatheredge.db` read-only. Auto-docs via `/docs`. |
| Frontend | **Next.js 15 App Router + Tailwind + shadcn/ui** | Output naturale di Claude Design. |
| Live updates | **SSE** (`/api/stream`) | Stessa cadenza del rich (3s). No WebSocket complication. |
| Charts | **Recharts** (equity, cluster) + **Tremor** (KPI cards) | Dark-mode friendly, pochi KB. |
| State | React Query + context | Cache + refetch-on-SSE trigger. |
| Deploy | `uvicorn` + `next start` via `launch.py` | Entrambi dietro `localhost:3000` + `:8000`. |

**Fuori scope**: auth (single-user locale), responsive < 480px (si usa da mac + tel).

---

## 🔌 API contract

Tutti gli endpoint sono **read-only** (no POST) — algoritmo non si tocca.

```
GET /api/snapshot          → JSON completo dello stato (payload per homepage)
GET /api/stream            → SSE: emette "snapshot" ogni 3s o on-change
GET /api/trades?status=... → lista paginata trades, filtri status/station/date
GET /api/trade/{id}        → dettaglio singolo trade (drill-down)
GET /api/market/{id}       → dettaglio market (forecast + prezzi intraday)
GET /api/strategy-log      → markdown raw di STRATEGY_LOG.md
GET /api/equity-curve      → [(ts, realized, total), ...] per grafico
```

### Shape `GET /api/snapshot`

```ts
{
  systemStatus: {
    label: "NORMAL" | "CONCENTRATED" | "CRITICAL",
    hint: string,        // "All gauges green" / "Apr-19 exposure 16.9%" / ...
  },
  bot: { running: boolean, uptime: string },
  bankroll: {
    initial: number,
    realized: number,
    unrealized: number,
    delta24h: number,
    totalPct: number,    // realized / initial * 100
  },
  positions: Array<{
    tradeId, station, marketDate,
    thresholdLow, thresholdHigh,
    side: "buy_yes" | "buy_no",
    price, cost, currentYes,
    unrealizedPnl,
    distance, distanceRatio, // vs station RMSE
    rmse,
    edgeSource: "floor" | "model",
    createdAt,
  }>,
  trades: Array<{
    id, createdAt, marketId, station, side,
    price, edge, agentName,
    status, pnl,
    saved?: number,      // se status=edge_stop
  }>,
  modelPerf: {
    brier: number | null,
    winRateFinal: { won, lost },                    // only resolved
    decisive: { wins, losses, total, ci95Low, ci95High },
    edgeSource: { floorPct, modelPct, totalOpen },
    flipSave: { count, totalSaved },
    bestAgent: { name, winRate } | null,
    timesfm: { active: boolean, corrections: number },
  },
  riskCluster: Array<{
    date, count, cost, pctBankroll,
    tier: "safe" | "warn" | "critical",             // <10 / <20 / >=20
  }>,
  perAgent: Array<{ agent, wins, losses, resolved, pnl }>,
  nextResolution: { date, positions, hoursToResolution } | null,
  freshness: { forecastAgeS, marketAgeS, lastTradeAt },
  activeMarkets: Array<{ city, thresholdLow, thresholdHigh, yesPrice, noPrice, volume, modelTmp }>,
}
```

---

## 🗺 Screens

| Route | Scopo | Componenti chiave |
|---|---|---|
| `/` | Home — replica del rich + polish | StatusPill, BankrollRibbon, PositionsTable, ModelPerfCard, RiskClusterBars, RecentTradesFeed, EquityCurve |
| `/positions` | Tutte le posizioni (sort/filter) | PositionsTable con colonne estese, drill-down modal |
| `/trades` | History completa (filtri status/stazione/date) | TradesTable, EDGE-FLIP highlight row |
| `/trades/:id` | Drill-down singolo trade | ThresholdBandViz, MarkToMarketChart, AgentVotesTimeline |
| `/markets` | Mercati attivi | MarketsTable con mini-band viz |
| `/strategy` | STRATEGY_LOG.md renderizzato | HypothesisCard (status badge + evidence), DecisionTimeline |

---

## 🎨 Design system

**Palette (dark-first, trading aesthetic alla Ramp/Linear):**
- Base: `slate-950` bg, `slate-800` cards, `slate-700` borders
- Profit: `emerald-500` `#10b981`
- Loss: `rose-500` `#ef4444`
- Warning: `amber-500` `#f59e0b`
- Info: `sky-400` `#38bdf8`

**Semaforo pills:**
- NORMAL → green pill (`bg-emerald-500/15 text-emerald-400`)
- CONCENTRATED → amber pill with subtle pulse
- CRITICAL → red pill with pulse + icon (⚠)

**Numeri:**
- Monospace (`font-mono`) per tutti i price/P&L/%/age → evita "layout shift" nell'update live
- Sign prefix (`+$3.74`) sempre, colored verde/rosso
- Age: `5s` / `3m` / `1.2h` / `2d` con color-grade (<1min green, <10min green, <30min yellow, >30min red)

**Distance/RMSE badge** (quello che in rich è `0.25x ZONE` / `3.33x SAFE`):
- `< 1.0` → red pill "ZONE" (dentro margine di errore del modello)
- `< 1.25` → amber pill "MARGIN"
- `≥ 1.25` → green pill "SAFE"

**Risk Cluster gauge**: barre orizzontali colorate per % bankroll, label con `$X / N pos / Y%`.

**Typography**: Inter (sans) + JetBrains Mono (numeri).

---

## 🎯 Prompt pronti per Claude Design

### Prompt 1 — Home dashboard

> Build a dark-mode trading dashboard homepage for WeatherEdge, a weather prediction-market trading bot. Single-user, local app. Stack: Next.js 15 App Router + Tailwind + shadcn/ui + Recharts.
>
> **Top bar**: system status pill (NORMAL green / CONCENTRATED amber-pulse / CRITICAL red-pulse + icon), bot status (RUNNING + uptime), data freshness (forecast age | market age, color-graded).
>
> **Hero row** (KPI cards): Bankroll `$108.26`, 24h delta `+$3.74 (+3.7%)`, Realized, Unrealized, Positions count, Win Rate, Next Resolution countdown (`Apr-19 in 4.8h`).
>
> **Main grid** (3 columns):
> 1. **Open Positions** table with columns: Station (badge), Date, Band (`54-55°F`), Side, Px, Cost, Unr P&L, Distance/RMSE (colored badge: ZONE/MARGIN/SAFE).
> 2. **Model Performance** card: Brier score, Decisive Win Rate with binomial 95% CI rendered as a horizontal range bar, Edge Source as stacked horizontal bar (Arbitrage % vs Model %), Edge-Flip lifetime counter with $ saved, best agent.
> 3. **Per-Agent P&L** mini-table: W/L, resolved count, P&L.
>
> **Mid row**: **Risk Cluster** — horizontal bars per resolution date showing exposure % of bankroll, color-graded (>20% red, >10% amber, else green).
>
> **Bottom row**: **Recent Trades** feed (scrollable) with time, market, side, price, edge%, agent, status badge, P&L. Highlight `edge_stop` rows with a magenta EDGE-FLIP shield badge and `+$X saved` hint. Next to it: **Equity Curve** line chart (realized + total, last 14 days).
>
> Palette: slate-950 base, emerald-500/rose-500 for profit/loss, monospace numbers to avoid layout shift during live SSE updates. Inspiration: Ramp finance + Linear density.

### Prompt 2 — Strategy log page

> Build `/strategy` page that renders `STRATEGY_LOG.md` as a structured view (not raw markdown). Header: current phase + trade count progress bar toward next gate (40 trades). Three sections:
> 1. **Active hypotheses** — grid of cards. Each card: title (`H2: distance/RMSE è il discriminatore`), status badge (UNCERTAIN yellow / LIKELY blue / WEAK gray / VALIDATED green / KILLED red), evidence-for vs evidence-against split pane, progress bar "trades needed to validate" (e.g., 5/15).
> 2. **Decisions changelog** — vertical timeline with date, decision, rationale.
> 3. **Killed hypotheses** — collapsed accordion (dimmed).
>
> Dark-mode, typography like Notion. Same palette as home.

### Prompt 3 — Position/trade drill-down

> Drill-down page for a single trade (route: `/trades/:id`). Layout:
> - **Threshold band visualization**: horizontal thermometer scale spanning e.g. 40-60°F, the band (e.g. 54-55°F) highlighted, a marker for the latest forecast temp, a marker for the observed temp if resolved (different icon).
> - **Mark-to-market chart**: line chart of entry price + market price over time, shaded area between them = unrealized P&L. Annotate edge-flip exit if present.
> - **Distance analysis**: big numeric "3.33× RMSE" with text explanation ("forecast is 10°F outside the band; station RMSE is 3°F, so we're 3.33× beyond noise — SAFE zone").
> - **Agent votes timeline**: which agents voted, their edge estimates, who was the majority.
> - **Event log**: created → updates → exit.
>
> All consumed from `/api/trade/:id`. Dark-mode, same palette.

---

## 🧪 Dev workflow

1. **Lato backend**: genera `weatheredge/api/app.py` con FastAPI, riusa i `query_*` già esistenti in `live_dashboard.py` come funzioni puro-SQL.
2. **Lato frontend**: `npx create-next-app@latest weatheredge-web --ts --tailwind --app`. Passa Prompt 1/2/3 a Claude Design → genera i componenti → integrale.
3. **Live**: SSE con un hook `useSnapshot()` che sostituisce React Query initialData ogni evento.
4. **CLI resta**: `python -m weatheredge.dashboard.live_dashboard` vive per debug in tmux. La nuova web è per uso quotidiano.

---

## 📎 Riferimenti interni

- `weatheredge/dashboard/live_dashboard.py` — tutti i `query_*` sono già pronti da portare in FastAPI
- `weatheredge.db` — schema: `trades, markets, forecasts, signals, market_prices`
- `STRATEGY_LOG.md` — source per `/api/strategy-log`
- `weatheredge/probability/gaussian.py` — `PROB_FLOOR = 0.10` per determinare edge source
