# WeatherEdge — Strategy Log

Living document per tracciare ipotesi, piano e lezioni. **Non è un PRD**, è un diario operativo. Aggiornare quando un'ipotesi viene validata/falsificata o quando una decisione cambia.

Last updated: 2026-04-21 (Phase 1 week 2, post Apr-20 resolution + Apr-21 partial; EXIT 0 `bucket_invalidated` merged)

---

## 🎯 Stato attuale

- **Phase:** 1 (activated ~2026-04-11)
- **Trade chiusi:** 21 (11W / 2L / 7 edge_stop / 1 abandoned), win rate decisive **11/13 = 84.6%** [95% CI Clopper-Pearson ≈ 55–98%]
- **Aperti:** 8 (4 Apr-21 / 3 Apr-22 / 1 Apr-23)
- **Net realized PnL:** **+$15.39** (winners +$20.77, losers -$5.52, stops aggregati quasi flat +$0.17)
- **Apr-19 outcome** (concentration test): 5W / 1L / 1 edge_stop, net **+$5.69**
- **Apr-20 outcome** (post-resolution): **2W / 0L / 4 edge_stop**, net **+$5.21** — 3 dei 4 stop risolti **non** sarebbero stati loss (exit reattivo, non protettivo); id=24 KNYC buy_yes exit +$2.09 confermato YES
- **Apr-21 partial**: id=16 KMIA stop -$0.20, altri 4 aperti (17, 21, 28, 29)

---

## 🧪 Ipotesi attive (da validare con più dati)

| # | Ipotesi | Evidenza pro | Evidenza contro | Stato | Trade per validare |
|---|---------|--------------|-----------------|-------|---------------------|
| H1 | Strategia `buy_no` su bande narrow (1°F) ha edge reale | 11/13 decisive (84.6%) | CI [55-98%] — ancora largo ma bordo inferiore sale | **LIKELY** | 25+ indipendenti |
| H2 | `distance_to_band / RMSE` è il discriminatore chiave | Winners con dist ≥3°F, loser KMDW Apr-19 a 0°F (forecast dentro la banda) | n piccolo, solo 2 losers totali | **LIKELY** | ~10-15 trade con distance <3°F |
| H3 | Vero edge NON è il modello ma il mispricing strutturale del mercato sui narrow bands | Floor-bound NO su narrow bands continua a dominare: id=22 (p=0.05 post-shrink), id=23 (p=0.10) entrambi WON +$4.77 Apr-20. | — | **VALIDATED, RINFORZATA** (2026-04-21) | Mantenere monitoraggio, ricomputare corr con n=21 |
| H4 | KMDW ha station bias persistente (-4.4°F) | — | KMDW Apr-18 64-65F e 60-61F won + Apr-19 48-49F lost → nessun bias consistente visibile | **WEAK (verso falsificazione)** | 5+ giorni distinti prima di archiviare |
| H5 | PROB_FLOOR=0.10 è troppo alto | Raw prob 0.003-0.095 → floor appiattisce tutto | Floor è load-bearing per i NO. Ma la distribuzione p_model è ora **bimodale**: floor=0.10 (maggioranza) + override METAR=0.80-0.95 (YES su bucket già toccato). 2/2 YES override risolti (id=24, 25) exit positivi. Sono strategie diverse. | **LIKELY FALSIFIED as "too high"** + **nuova sotto-categoria YES-override** da trattare separatamente (2026-04-21) | Logging/metriche distinte per YES-override |
| H6 | Correlazione intra-day su same-regime rende `max_single_date=25%` troppo largo | — | **Apr-20 post-resolution: 3/4 stop non sarebbero stati loss** (id=13, 15, 18 → resolution win in altri bucket adiacenti es. id=22, 23). Il "cluster" era exit reattivo, non correlazione strutturale. | **WEAKENED again** (2026-04-21, post-resolution Apr-20) — il problema è `exit_edge_flip`, non concentration | Osservare Apr-21 cluster (4 pos) + confermare pattern |
| H7 | `exit_edge_flip` è troppo reattivo sui NO narrow bands | 7 stop su 21 (33%), aggregato +$0.17. Apr-20: id=13/15/18 stop -$1.65 mentre cugini floor-bound (id=22, 23) su bucket adiacenti stesso station-day hanno WON +$4.77. | Save reali: id=24 (+$2.09), id=25 (+$0.45), id=9 evitato loss peggiore. | **NEW, LIKELY** (2026-04-21); **parzialmente indirizzata** da EXIT 0 per i buy_yes post-invalidazione METAR (2026-04-21) | Event study Apr-20 stops (ora tutti risolti); osservare se dopo EXIT 0 il tasso di edge_stop sui `buy_no` resta sproporzionato |

---

## 📋 Piano (timeline)

### ✅ Apr 19 — FATTO
- **Concentration test**: 7 trade stesso giorno → 5W/1L/1 edge_flip, net +$5.69. Rho apparente basso.
- **Dashboard Distance/RMSE column**: implementata (col colored badge ZONE/MARGIN/SAFE)

### 🟡 Questa settimana (Apr 21-27) — **monitor-only mode** (nessun cambio da giorni, scelta consapevole)
1. **Event study Apr-20 stops ora possibile** (tutti e 3 risolti: id=13, 15, 18). Conclusione preliminare: stop reattivi, non protettivi — alimenta H7. **Formalizzare il write-up.** Priority 1.
2. ~~Validare id=24 KNYC buy_yes @0.85~~ → **Confermato**: exit +$2.09 prima della resolution (METAR override funziona come categoria separata).
3. **Fix signal logging gap** — trade chiusi senza agent signals loggati. Bug, non strategia.
4. **Aspettare 25+ trade chiusi** prima di qualsiasi cambio parametrico (attuali: 21 → mancano ~4)
5. **Monitorare H2** — serve 1-2 loser con distance alta (> 3°F) per testare se regge
6. **Monitorare Apr-21 cluster (4 pos)** + Apr-22 (3 pos): se stop rate resta elevato ma resolution PnL positivo → conferma H7 (exit reattivo), indebolisce H6 (concentration).
7. **Pre-flight DB audit** prima di qualsiasi scaling: side/prob consistency, leakage timestamps, duplicate exposure, mark-to-market history.
8. **Natural experiment Apr-21 KMIA** (id=28 YES 78-79 + id=29 NO 80-81): "straddle stretta" — entrambi vincono se daily_high ∈ [78, 80). Prima osservazione di doppia esposizione stesso station-day su bucket adiacenti.

### 🟢 Backtest-only (nessun cambio live)
- Filter proposto: `entry se distance > max(3°F, 1.25 × rolling_RMSE)` — avrebbe filtrato solo trade 12
- De-duplica station-day nelle stats di win rate
- PROB_FLOOR sweep (0.02, 0.05, 0.10)

### ❄️ Congelato fino a 30+ trade
- Attivazione Phase 2 (gate: 40+ trade + calibration gap <15%)
- Tuning floor/shrinkage in produzione
- Cambio pesi agent (Phase 3)

---

## 📜 Decisioni prese (changelog)

| Data | Decisione | Rationale |
|------|-----------|-----------|
| 2026-04-19 | Non implementare filter distance-to-band live | n=5 troppo piccolo, rischio kill di winners |
| 2026-04-19 | Non toccare PROB_FLOOR | Effetto su 6/6 casi → cambio strutturale, troppo rischioso ora |
| 2026-04-19 | Mantenere `max_single_date=25%` fino a dati Apr 19 | Serve vedere rho reale prima di stringere |
| 2026-04-20 | **Mantenere** `max_single_date=25%` anche dopo Apr-19 | Apr-19 (7 pos stesso giorno): 5W/1L/1 flip → rho basso, no cluster loss. Cap attuale ok. |
| 2026-04-20 | Dashboard redesign via Claude Design (web app) — algoritmo non si tocca | Blueprint in `DASHBOARD_DESIGN_BRIEF.md`; CLI resta per debug |
| 2026-04-20 | Dashboard rich: System Status semaforo + Distance/RMSE col + Risk Cluster gauge + Edge-Flip badge + Decisive CI + Source of Edge + Flip Save counter | Implementato senza toccare logica di trading |
| 2026-04-20 | CCG trade audit (Codex): H3 → **VALIDATED**, H5 → **LIKELY FALSIFIED as "too high"**, H6 → **RE-OPENED** | Corr(raw_p, PnL) ≈ 0.12 debole; 12 floor-bound +$3.00 avg +$0.25; 2 non-floor (id=3,11) +$5.30 avg +$1.77. Il floor è la strategia, non un bug. Apr-20 3/3 stops (p≈0.0088) ribalta conclusione Apr-19. |
| 2026-04-20 | **Monitor-only mode fino a resolution Apr-20 + 25 trade totali** | Nessun cambio parametrico senza event study su 3 stop Apr-20. Decidere `max_single_date` dopo che sappiamo se exit_edge_flip era reattivo o giustificato. |
| 2026-04-20 | id=24 KNYC buy_yes @0.85 confermato **legit** (non sign bug) | METAR bucket override in `main.py:1988-2030`: se daily_high osservato è già in [bucket_low, bucket_high] → prob forzata a ≥0.85. Monitor solo: deve risolvere YES. |
| 2026-04-21 | **Apr-20 chiuso: 2W / 0L / 4 edge_stop, net +$5.21** (non "disaster" come scritto ieri) | 3 dei 4 stop (id=13, 15, 18) risolti senza loss — cugini floor-bound su bucket adiacenti (id=22, 23) hanno WON +$4.77. H6 indebolita, H7 (`exit_edge_flip` reattivo) aperta come ipotesi prioritaria. |
| 2026-04-21 | **Riconoscere strategia bimodale**: floor-bound NO (H3) + YES-override METAR (nuova) | Distribuzione p_model bimodale (0.10 vs 0.80-0.95). id=24, 25 ambedue YES-override exit positivi. Richiede logging/metriche separate. Nessun cambio di codice, solo disambiguazione analitica. |
| 2026-04-21 | **Confermato monitor-only** (nessun intervento da giorni — scelta esplicita) | Mancano ~4 trade a raggiungere soglia 25 totali. Apr-21/22/23 apertura 8 pos già fatta dal sistema; aspettare resolution senza override manuale. |
| 2026-04-21 | **EXIT 0 `bucket_invalidated`** implementato (validato CCG: Codex + Gemini) | Bug strutturale: quando METAR osserva `daily_high > bucket_high`, YES è fisicamente impossibile ma nessuna regola esistente chiudeva la posizione (edge_flip non scatta se market≈0 e model≈0.05, EV marginalmente a favore di hold, time_decay aspetta l'ultima ora). Nuova regola pre-empts tutte: check flag `bucket_invalidated_for_yes` + side `buy_yes` → exit con status `bucket_invalidated`. Asimmetrica: `buy_no` non inclusa (edge_flip gestisce naturalmente). 7 test, 0 regressioni su 1868 test totali. Indirizza la root cause di H7 (exit reattivo su buy_yes post-invalidazione); H7 sui `buy_no` resta aperta. |

---

## 🪦 Ipotesi scartate

*(nessuna ancora — aggiungere quando qualcosa viene falsificato)*

---

## 📐 Formule di riferimento

- **Binomial 95% CI (Clopper-Pearson)** per win rate con k vittorie su n: `scipy.stats.beta.ppf([0.025, 0.975], k, n-k+1)` (approx)
- **Effective independent bets** sotto correlazione uniforme ρ: `n_eff = n / (1 + (n-1) × ρ)`
- **Edge per buy_no**: `edge = market_price_yes - P_model(yes)`
- **Distance-to-band normalized**: `min(|forecast - low|, |forecast - high|) / RMSE_station`

---

## 🔗 Riferimenti interni

- `WEATHEREDGE_PRD_v3.1.md` — phase gates e structural kills
- `weatheredge/phase_config.py` — moduli per phase
- `weatheredge/execution/exit_manager.py` — exit logic (edge_flip, time_decay, EV-based)
- `weatheredge/agents/risk_manager.py` — circuit breaker + sizing
- `weatheredge.db` — tabelle trades, markets, forecasts, signals
