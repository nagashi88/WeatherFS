# WeatherEdge — Strategy Log

Living document per tracciare ipotesi, piano e lezioni. **Non è un PRD**, è un diario operativo. Aggiornare quando un'ipotesi viene validata/falsificata o quando una decisione cambia.

Last updated: 2026-04-22 (✅ Piano di rientro Phase 1 completato — 5/5 step committati; runtime ora allineato a `current_phase.json`)

---

## 🎯 Stato attuale

- **Phase dichiarata:** 1 (file `current_phase.json` dal 2026-04-11)
- **Phase runtime reale:** **phase_1_actual** post-cutover 2026-04-22 (era phase_0_actual fino al 2026-04-21 — vedi CRITICAL FINDING + close-out sotto)
- **Trade chiusi:** 21 (11W / 2L / 7 edge_stop / 1 abandoned), win rate decisive **11/13 = 84.6%** [95% CI Clopper-Pearson ≈ 55–98%] — **tutti generati sotto phase_0_actual**
- **Aperti:** 8 (4 Apr-21 / 3 Apr-22 / 1 Apr-23) — anch'essi in phase_0_actual
- **Net realized PnL:** **+$15.39** (winners +$20.77, losers -$5.52, stops aggregati quasi flat +$0.17)
- **Apr-19 outcome** (concentration test): 5W / 1L / 1 edge_stop, net **+$5.69**
- **Apr-20 outcome** (post-resolution): **2W / 0L / 4 edge_stop**, net **+$5.21** — 3 dei 4 stop risolti **non** sarebbero stati loss (exit reattivo, non protettivo); id=24 KNYC buy_yes exit +$2.09 confermato YES
- **Apr-21 partial**: id=16 KMIA stop -$0.20, altri 4 aperti (17, 21, 28, 29)

---

## 🚨 CRITICAL FINDING 2026-04-21 — Phase 1 never actually activated

Audit CCG (Codex finding #1, verified live) ha rivelato che **Phase 1 non è mai stata applicata al runtime** dal 2026-04-11 in poi, nonostante `current_phase.json` dicesse `phase_1`.

**Root cause:** `main.py:_apply_runtime_phase_override` legge solo la env var `WEATHEREDGE_PHASE`. Se vuota (default), non c'è fallback a `current_phase.json` → `MODULES` resta sui default Phase 0 di `config.py:13`. Nessun `.sh`, `.env`, systemd, plist del repo setta quella env var → in produzione è sempre stata vuota.

**Evidenza live** (riprodotta su shell pulita):
```
current_phase.json says: phase_1
MODULES.exit_ev_based      = False   ← doveva essere True in phase_1
MODULES.bias_correction    = False   ← doveva essere True
MODULES.rmse_floor_dynamic = False   ← doveva essere True
MODULES.limit_orders       = False   ← doveva essere True
MODULES.exit_edge_flip     = True    ← già True anche in phase_0 defaults
MODULES.temporal_guard     = True    ← già True anche in phase_0 defaults
MODULES.volume_filter      = True    ← già True anche in phase_0 defaults
```

**Cosa significa per i 21 trade chiusi + 8 aperti:**
- EV-based exit **mai scattata** — `exit_edge_flip` ha fatto 100% del lavoro di chiusura; i 7 `edge_stop` sono tutti prodotto del solo edge_flip, senza competizione di EV.
- Nessun `bias_correction` sui forecast — H4 (KMDW station bias WEAK) va riletta: il bias poteva esserci ma non lo stavamo correggendo né misurando.
- `rmse_floor_dynamic=False` → RMSE statica; Distance/RMSE normalizzata su floor fisso.
- `limit_orders=False` → tutti market orders, zero limit. No slippage control.

**Re-tagging delle ipotesi:**
- **H3 "VALIDATED"** → `VALIDATED under phase_0_actual`. L'edge strutturale floor-bound esiste anche senza i moduli Phase 1; è stato validato in un regime più semplice. **Conclusione più forte, non più debole** (meno confound).
- **H7 "edge_flip troppo reattivo"** → rinforzata. Senza EV-based che competeva per tenere aperto, edge_flip ha chiuso 7/21 trade (33%): il tasso è 100% attribuibile al solo edge_flip.
- **H4 KMDW bias** → inconclusiva, non WEAK. Non stavamo correggendo nulla.
- **EXIT 0 `bucket_invalidated`** (merged Apr-21) → valido, non impattato dal finding.

**Cosa NON abbiamo mai visto funzionare:**
1. `_evaluate_exit_ev` — EV-based exit, mai scattata una volta
2. Limit orders e relativo fill logic
3. `rmse_floor_dynamic` — floor adattivo
4. `bias_correction` — correzione station-bias

**Piano di rientro (concordato 2026-04-21, completato 2026-04-22):**
1. ✅ **STEP 0** (`b65c1e8`): documentazione finding + re-tag regime runs
2. ✅ **STEP 1** (`b6e188f`): fix `paper_trader.py` `buy_no` price semantics canonica (prep per limit_orders)
3. ✅ **STEP 2** (`313a024`): fix `exit_manager.py` — EV prima di edge_flip, edge netto side-aware (prep per exit_ev_based)
4. ✅ **STEP 3** (`2caedd7`): fix `main.py` freshness gate per-market (safety net pre-scale)
5. ✅ **STEP 4** (`f3dc61e`): fix `main.py:_apply_runtime_phase_override` con fallback a `get_current_phase()` + boot assertion di coerenza MODULES vs phase dichiarata → **cutover ufficiale Phase 1 avvenuto 2026-04-22**
6. ✅ **STEP 5** (`0692b6a`): fix formula fee Kalshi `ceil(0.07·p·(1-p)·100)/100` per trade (round-trip ≈ 0.05 a p=0.5, non flat 0.02) — elimina trade sub-break-even post-fee
7. **Post-cutover**: counter strategia **reset a 0** sotto `phase_1_actual`. H3/H7 vanno rivalidate sotto il nuovo regime. Test suite verde (1881 passed, 39 skipped, 0 regressioni).

**Riferimenti audit:**
- `/tmp/ccg_codex_out.md` — output completo Codex
- `/tmp/ccg_gemini_out.md` — output completo Gemini (contributo debole, vedi sintesi)
- File chiave: `weatheredge/main.py:74,167-182`, `weatheredge/phase_config.py:170`, `weatheredge/config.py:13`

---

## 🧪 Ipotesi attive (da validare con più dati)

| # | Ipotesi | Evidenza pro | Evidenza contro | Stato | Trade per validare |
|---|---------|--------------|-----------------|-------|---------------------|
| H1 | Strategia `buy_no` su bande narrow (1°F) ha edge reale | 11/13 decisive (84.6%) | CI [55-98%] — ancora largo ma bordo inferiore sale | **LIKELY** | 25+ indipendenti |
| H2 | `distance_to_band / RMSE` è il discriminatore chiave | Winners con dist ≥3°F, loser KMDW Apr-19 a 0°F (forecast dentro la banda) | n piccolo, solo 2 losers totali | **LIKELY** | ~10-15 trade con distance <3°F |
| H3 | Vero edge NON è il modello ma il mispricing strutturale del mercato sui narrow bands | Floor-bound NO su narrow bands continua a dominare: id=22 (p=0.05 post-shrink), id=23 (p=0.10) entrambi WON +$4.77 Apr-20. | — | **VALIDATED under phase_0_actual** (2026-04-21 post-CCG). Re-validate after Phase 1 cutover. | Rivalidare post-cutover con nuovi trade sotto phase_1_actual |
| H4 | KMDW ha station bias persistente (-4.4°F) | — | KMDW Apr-18 64-65F e 60-61F won + Apr-19 48-49F lost → nessun bias consistente visibile | **WEAK (verso falsificazione)** | 5+ giorni distinti prima di archiviare |
| H5 | PROB_FLOOR=0.10 è troppo alto | Raw prob 0.003-0.095 → floor appiattisce tutto | Floor è load-bearing per i NO. Ma la distribuzione p_model è ora **bimodale**: floor=0.10 (maggioranza) + override METAR=0.80-0.95 (YES su bucket già toccato). 2/2 YES override risolti (id=24, 25) exit positivi. Sono strategie diverse. | **LIKELY FALSIFIED as "too high"** + **nuova sotto-categoria YES-override** da trattare separatamente (2026-04-21) | Logging/metriche distinte per YES-override |
| H6 | Correlazione intra-day su same-regime rende `max_single_date=25%` troppo largo | — | **Apr-20 post-resolution: 3/4 stop non sarebbero stati loss** (id=13, 15, 18 → resolution win in altri bucket adiacenti es. id=22, 23). Il "cluster" era exit reattivo, non correlazione strutturale. | **WEAKENED again** (2026-04-21, post-resolution Apr-20) — il problema è `exit_edge_flip`, non concentration | Osservare Apr-21 cluster (4 pos) + confermare pattern |
| H7 | `exit_edge_flip` è troppo reattivo sui NO narrow bands | 7 stop su 21 (33%), aggregato +$0.17. Apr-20: id=13/15/18 stop -$1.65 mentre cugini floor-bound (id=22, 23) su bucket adiacenti stesso station-day hanno WON +$4.77. Post-CCG: `exit_ev_based=False` in phase_0_actual → edge_flip ha fatto 100% lavoro di chiusura, no competizione. | Save reali: id=24 (+$2.09), id=25 (+$0.45), id=9 evitato loss peggiore. | **NEW, RINFORZATA** (2026-04-21 post-CCG); il tasso 33% è da attribuire al solo edge_flip, senza EV che potesse tenere aperto. Post-cutover Phase 1 osservare se EV-based riduce stop rate. | Event study Apr-20 stops (ora tutti risolti); osservare tasso edge_stop post-cutover con EV attivo |

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
| 2026-04-21 | 🚨 **CCG audit finding: Phase 1 never actually activated** (verified live) | `main.py:_apply_runtime_phase_override` legge solo env var `WEATHEREDGE_PHASE`, mai settata in prod. `current_phase.json=phase_1` dal 2026-04-11 è stato ignorato al boot. MODULES runtime = phase_0 defaults: `exit_ev_based=False`, `bias_correction=False`, `rmse_floor_dynamic=False`, `limit_orders=False`. Conseguenza: tutti i 21 trade chiusi + 8 aperti generati sotto **phase_0_actual**. H3 re-tagged `VALIDATED under phase_0_actual` (conclusione più forte, meno confound); H7 rinforzata (edge_flip ha fatto 100% lavoro chiusura senza EV-based a competere); H4 inconclusiva (nessuna bias_correction in gioco). |
| 2026-04-21 | Piano di rientro in 5 step | STEP 0 (fatto): update STRATEGY_LOG. STEP 1-3: fix preparatori zero-impact in Phase 0 (buy_no price semantics canonica, EV prima di edge_flip, freshness per-market). STEP 4 (post resolution 8 aperte ~Apr-23/24): fix attivazione phase + boot assertion → cutover Phase 1 + counter trade reset. STEP 5: conferma formula fee Kalshi (flat 0.02 vs 0.07·p·(1-p)). |
| 2026-04-22 | ✅ **STEP 1 chiuso** (`b6e188f`): `paper_trader.py` buy_no limit-fill ora persiste YES-price canonica (`1 - no_price`) su `entry_price`. Unblocka `limit_orders=True` Phase 1 senza rompere downstream (edge, EV, exit_manager). | Audit CCG: discrepanza tra entry_price (veniva persistito NO-price dopo fill limit) e `market_price_yes` avrebbe contaminato tutti i calcoli di edge/EV post-entry su ordini limit. Fix zero-impact in Phase 0 (limit_orders=False → codepath mai eseguito), necessario prima del cutover. |
| 2026-04-22 | ✅ **STEP 2 chiuso** (`313a024`): `exit_manager.py` riordina exit rules — `_evaluate_exit_ev` ora viene valutata PRIMA di `_evaluate_exit_edge_flip`. Edge netto side-aware (non più solo YES-side). | H7 (edge_flip troppo reattivo) identificava la root cause: in phase_0_actual, edge_flip ha fatto 100% del lavoro di chiusura perché EV-based era disattivo. Riordinando le rules, quando Phase 1 attiva `exit_ev_based=True`, EV può vincere la competizione e tenere aperta una posizione che edge_flip avrebbe chiuso prematuramente. Zero-impact fino al cutover. |
| 2026-04-22 | ✅ **STEP 3 chiuso** (`2caedd7`): `main.py` freshness gate ora per-market (oltre al gate globale esistente). `MarketOpportunity` porta `market_price_timestamp`; pre-pass in `_run_decision_cycle` skippa i mercati con age > `market_data_ttl_minutes`. Summary log include `stale_market_count`. | Il gate globale sul newest row non protegge i mercati illiquidi: con 20+ mercati per cluster, un row fresco su uno qualsiasi lascia passare tutti gli altri. Safety net pre-scaling — se il fetcher fallisce in modo parziale, ora i mercati con dati vecchi vengono esclusi singolarmente. Exit_manager gestisce già "opp is None → hold" sulle posizioni aperte. |
| 2026-04-22 | ✅ **STEP 4 chiuso** (`f3dc61e`) — **CUTOVER PHASE 1 UFFICIALE**: `_apply_runtime_phase_override` ora fa fallback su `get_current_phase()` quando `WEATHEREDGE_PHASE` env non è set. Boot assertion confronta `MODULES` vs `get_phase_modules(phase)` e logga `ERROR` su divergenza. Test `test_runtime_phase_boot_assertion_logs_on_mismatch` copre il caso April-2026 che avrebbe dovuto suonare l'allarme. | La safety net permanente contro questa classe di bug: ogni boot ora emette `boot_phase_check: MODULES match declared phase <X>` (info) o un ERROR ultra-rumoroso se qualcosa rompe il merge. **Da ora in poi i trade sono phase_1_actual** (EV-based exit, bias_correction, rmse_floor_dynamic, limit_orders tutti attivi). Counter H3/H7 **resettato a 0** sotto nuovo regime. |
| 2026-04-22 | ✅ **STEP 5 chiuso** (`0692b6a`): `probability/edge.py` corretta formula fee Kalshi. Prima: flat `0.02` round-trip. Ora: `kalshi_taker_fee(p) = ceil(0.07·p·(1-p)·100)/100` per trade, round-trip = `2·taker(p) + avg_spread/2`. A p=0.5, vera round-trip = $0.05 (non $0.02). | Sotto la vecchia formula, il bot stava accettando trade con edge netto positivo apparente ma reale sub-break-even (underpricing del fee di 2.5× nel worst case). Test `test_kalshi_taker_fee_formula_ceil_0_07_p_1mp` copre i punti chiave della curva. Test `test_net_edge_kalshi_exact_value` aggiornato con il nuovo costo atteso (0.18 → 0.15). **Impatto atteso**: soglia di entrata più alta sui mercati p∈[0.3, 0.7] — molti "win marginali" pre-fix non supereranno più il net-edge threshold. |
| 2026-04-22 | ✅ **Piano di rientro COMPLETO** (5/5 step committati) | Test suite verde: 1881 passed, 39 skipped, 0 regressioni. Branch `exp/phase1-vs-phase2-paper`. Prossimo boot del bot applicherà automaticamente phase_1 + logga boot assertion. Da monitorare Apr-22/23/24: tasso `edge_stop`, numero di exit_ev_based scattate, trade filtrati dal nuovo fee model. |

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
