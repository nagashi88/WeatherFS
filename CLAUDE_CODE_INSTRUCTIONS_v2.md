# WeatherEdge — Istruzioni per Claude Code v2
# Data: 30 Marzo 2026
# Contesto: Il bot è stato implementato (869 test verdi), il primo ciclo ha girato,
# 1580 forecast stored, 0 trade eseguiti perché il market parser non mappa i mercati Polymarket.
# Un'analisi strategica ha identificato 4 bug critici e problemi fondamentali nella strategia.
# Queste istruzioni risolvono TUTTO in ordine.

---

## REGOLA PRINCIPALE
Esegui le istruzioni in ordine sequenziale: STEP 1 → 2 → 3 → 4 → 5 → 6.
NON iniziare uno step successivo finché il precedente non è completato e testato.
Dopo ogni step, riporta cosa hai fatto e i risultati.

---

## STEP 1 — FIX 4 BUG CRITICI
Priorità: BLOCCANTE. Senza questi fix, il bot prende decisioni sbagliate.

### Bug 1: METAR Override sbagliato per mercati "below"
**Problema**: Se il mercato chiede "la massima di NYC resterà sotto 69°F?" e alle 7am il METAR legge 48°F, il bot dice probabilità=0.99. È SBAGLIATO. La massima pomeridiana potrebbe essere 72°F. Il bot confonde "la temperatura attuale è sotto la soglia" con "la massima giornaliera resterà sotto la soglia".

**Fix richiesto** — Modifica la logica METAR override con queste regole ESATTE:

```
SE market_type == "above" (es. "will high exceed 69°F?"):
    SE observed_temp >= threshold:
        probability = 0.99  # La massima HA GIÀ superato la soglia. Certezza. CORRETTO, mantieni.
    ALTRIMENTI:
        Usa i modelli forecast normalmente. NON fare override.

SE market_type == "below" (es. "will high stay below 69°F?"):
    SE ora_locale_stazione >= 17:00 (5pm) E temp_sta_scendendo E temp_attuale < (threshold - 5°F):
        probability = 0.95  # Tardo pomeriggio, temp in calo e lontana dalla soglia.
    ALTRIMENTI:
        Usa i modelli forecast normalmente. NON fare override.
        
SE market_date != oggi:
    MAI usare METAR override. Il METAR di oggi è irrilevante per un mercato di domani.
    Usa SOLO i modelli forecast.
```

Per determinare se la temperatura sta scendendo: confronta le ultime 2 letture METAR. Se la seconda è più bassa della prima, la temp sta scendendo.

### Bug 2: Nessun cap sull'edge massimo
**Problema**: Il bot calcola edge del 97%. Un edge del 97% significa che il nostro modello è rotto, non che il mercato è inefficiente.

**Fix**: Aggiungi nel RiskManager, PRIMA di qualsiasi decisione di trading:
```python
if edge > 0.50:  # 50%
    log.warning(f"Unrealistic edge {edge:.1%} for {market_id}. Likely model error. Skipping.")
    return None  # Non tradare
```

### Bug 3: Compra contratti sotto $0.10
**Problema**: Comprare YES a $0.03 significa scommettere contro il 97% del mercato usando gli stessi dati pubblici. Studio su 300K+ contratti Kalshi: contratti sotto $0.10 perdono il 60% del capitale investito in media.

**Fix**: Aggiungi filtro nel PortfolioManager:
```python
if market_price < 0.10:
    log.info(f"Skipping {market_id}: price ${market_price:.2f} below $0.10 minimum")
    return None
```

### Bug 4: Nessun check contraddizione METAR vs Forecast
**Problema**: Se il METAR dice probabilità=0.99 ma i forecast GFS/ECMWF/HRRR dicono probabilità=0.55, c'è una contraddizione. Qualcosa è sbagliato nel nostro calcolo. Non bisogna tradare.

**Fix**: Prima di ogni trade, confronta le due probabilità:
```python
metar_prob = ...  # probabilità derivata dal METAR
forecast_prob = ...  # probabilità dai modelli
if abs(metar_prob - forecast_prob) > 0.30:  # >30 punti percentuali di differenza
    log.warning(f"CONTRADICTION: METAR prob={metar_prob:.2f}, forecast prob={forecast_prob:.2f}. Skipping.")
    # Invia Telegram alert
    return None
```

**Test di verifica dopo Step 1**: Riesegui un ciclo di decisione. Verifica che:
- Nessun trade ha edge >50%
- Nessun trade su contratti sotto $0.10
- Mercati "below" per date odierne NON vengono overriddati dal METAR al mattino
- Mercati per date future NON usano il METAR

---

## STEP 2 — POLYMARKET MARKET PARSER
Priorità: BLOCCANTE. Senza questo, il bot non può tradare nessun mercato Polymarket.

**Problema**: I mercati Polymarket hanno titoli come "Highest temperature in NYC on April 4? - between 50-51°F" ma il bot ha solo mapping hardcoded per 5 ticker Kalshi. Tutti i mercati Polymarket risultano "Unknown".

**Cosa costruire**: Una funzione `parse_polymarket_title(title: str) -> ParsedMarket | None` che usa regex per estrarre:
- `city`: nome della città
- `station`: codice ICAO della stazione di risoluzione
- `date`: data del mercato (YYYY-MM-DD)
- `threshold_low`: soglia bassa (°F)
- `threshold_high`: soglia alta (°F) — può essere None per mercati "above/below"
- `market_type`: "daily_high", "daily_low", "range", "above", "below"

**Pattern da parsare** (esamina i titoli reali dalla scan precedente e adatta i regex):
```
"Highest temperature in NYC on April 4? - between 50-51°F"
→ city=NYC, station=KNYC, date=2026-04-04, threshold_low=50, threshold_high=51, type=range

"Will the high temperature in Chicago exceed 60°F on April 3?"
→ city=Chicago, station=KMDW, date=2026-04-03, threshold=60, type=above

"Highest temperature in Denver on April 4? - 55°F or above"
→ city=Denver, station=KDEN, date=2026-04-04, threshold=55, type=above
```

**Mapping città → stazione** (dizionario, case-insensitive):
```python
CITY_TO_STATION = {
    "nyc": "KNYC", "new york": "KNYC", "new york city": "KNYC",
    "chicago": "KMDW",
    "miami": "KMIA",
    "denver": "KDEN",
    "los angeles": "KLAX", "la": "KLAX", "lax": "KLAX",
    "london": "EGLC",
    "austin": "KAUS",
    "seattle": "KSEA",
    "atlanta": "KATL",
    "dallas": "KDFW",
    "phoenix": "KPHX",
    "houston": "KIAH",
    "san francisco": "KSFO", "sf": "KSFO",
    "boston": "KBOS",
}
```

**Se il regex non riesce a parsare un titolo**: log il titolo completo come WARNING + invia Telegram alert con il testo del titolo. NON usare LLM per parsare — troppo lento per ogni ciclo di scan.

**Se la città non è nel dizionario**: log come HIGH PRIORITY alert + Telegram. Nuove città = potenziale edge.

**Test di verifica**: Riesegui la scan dei mercati. Riporta:
- Quanti mercati parsati con successo
- Quanti mercati ancora "Unknown"
- I titoli esatti dei mercati non parsati (così posso aggiungere pattern)

---

## STEP 3 — KALSHI PUBLIC API
Priorità: MEDIA. Polymarket è sufficiente per il paper trading, ma Kalshi aggiunge mercati.

**Fatto verificato**: L'API pubblica di Kalshi NON richiede autenticazione per i dati di mercato.
Base URL: `https://api.elections.kalshi.com/trade-api/v2`
(il subdomain "elections" è fuorviante — serve TUTTI i mercati, incluso meteo)

**Cosa fare**:
1. Prova prima con pmxt: `kalshi = pmxt.Kalshi()` (senza credenziali) poi `kalshi.fetch_markets(query='KXHIGH')`
2. Se pmxt richiede credenziali anche per la lettura, crea un `KalshiPublicClient` con aiohttp:
```python
async def fetch_kalshi_weather_markets(self) -> list[dict]:
    series = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHLAX", "KXHIGHDEN"]
    markets = []
    for ticker in series:
        url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={ticker}&status=open"
        async with self.session.get(url) as resp:
            data = await resp.json()
            markets.extend(data.get("markets", []))
    return markets
```
3. Nessun header di autenticazione necessario.

**Test di verifica**: Riporta quanti mercati Kalshi weather sono stati trovati.

---

## STEP 4 — BACKTESTER (PRIORITÀ #1 DOPO I BUG FIX)
Questo è lo step più importante. TUTTO il resto è inutile se non sappiamo se la strategia funziona su dati storici.

**Cosa costruire**: Un modulo `weatheredge/backtesting/backtester.py` che:

### 4a. Scarica dati storici
- **Forecast storici**: Open-Meteo Archive API (https://archive-api.open-meteo.com/v1/archive) fornisce dati storici. Scarica GFS daily max temperature forecast per ogni stazione per gli ultimi 6-12 mesi.
- **Temperature effettive**: Iowa Environmental Mesonet (https://mesonet.agron.iastate.edu/request/download.phtml) per le osservazioni reali. Scarica daily max temperature per ogni stazione target.
- **Prezzi storici dei mercati**: Se disponibili da Kalshi API (mercati risolti) o dal repo github.com/evan-kolberg/prediction-market-backtesting

### 4b. Per ogni giorno storico, simula il ciclo di trading
```
Per ogni giorno D negli ultimi 6 mesi:
    Per ogni stazione S nelle 5 città principali:
        1. Leggi il forecast GFS per il giorno D (quello disponibile il giorno D-1 o la mattina di D)
        2. Calcola la probabilità del nostro modello per ogni bracket
        3. Confronta con il prezzo di mercato (se disponibile) o con la probabilità climatologica
        4. Registra: avremmo tradato? Su quale lato? Con quale sizing?
        5. Leggi la temperatura effettiva per il giorno D
        6. Registra: avremmo vinto o perso? Quanto P&L?
```

### 4c. Calcola metriche
- Win rate totale e per città
- ROI dopo fee stimate (2% per trade come proxy)
- Sharpe ratio
- Max drawdown
- Distribuzione degli edge (istogramma)
- Per-city breakdown: quali città sono profittevoli, quali no?

### 4d. Output
Stampa un report chiaro con tutte le metriche. Salva i risultati in un CSV per analisi.

**NOTA IMPORTANTE**: Se i dati storici dei prezzi di mercato non sono disponibili, usa come proxy la "probabilità climatologica" — la percentuale di giorni storici in cui la temperatura ha superato quella soglia in quel periodo dell'anno. Non è perfetto ma dà un'indicazione.

**Test di verifica**: Riporta i risultati completi del backtest. Se win rate <52% o ROI negativo dopo le fee → FERMATI e riporta. Non procedere allo step 5.

---

## STEP 5 — STATION-SPECIFIC BIAS CORRECTION
Esegui SOLO se il backtest (Step 4) mostra potenziale (WR >52% o aree di profitto per specifiche città).

**Concetto**: I modelli GFS/ECMWF hanno errori sistematici (bias) che variano per stazione, stagione e condizioni meteo. Se prevediamo l'errore del forecast, possiamo correggere il forecast e avere una stima migliore di quella del mercato.

### 5a. Calcola il bias storico
Usando i dati scaricati nello Step 4:
```python
# Per ogni stazione
forecast_error = gfs_predicted_high - actual_observed_high
# Aggregare per:
# - Mese/stagione
# - Regime di vento (METAR wind direction: N/NE/E/SE/S/SW/W/NW)
# - Copertura nuvolosa (clear/scattered/broken/overcast)
```

Salva come tabella di correzione:
```
station | month | wind_regime | cloud_cover | mean_error | std_error | sample_size
KNYC    | Mar   | SW          | clear       | +1.8°F     | 2.1°F     | 47
KNYC    | Mar   | NW          | overcast    | -0.4°F     | 1.8°F     | 31
KDEN    | Mar   | SW          | clear       | +3.2°F     | 3.5°F     | 42
```

### 5b. Applica la correzione
```python
corrected_forecast = gfs_raw - bias_correction(station, month, wind, clouds)
```
Usa il forecast corretto (non il raw) per calcolare la probabilità.

### 5c. Riesegui il backtest con la correzione
Confronta: WR e ROI con bias correction vs senza. Se la correzione migliora il risultato, tienila. Se no, scartala.

---

## STEP 6 — FORECAST ERROR PREDICTION MODEL (AVANZATO)
Esegui SOLO se lo Step 5 mostra che la bias correction funziona (migliora WR di almeno 2 punti).

**Concetto evoluto**: Invece di una semplice tabella di bias, allena un modello ML (XGBoost) che prevede QUANTO sbaglierà il forecast per una specifica stazione in un giorno specifico.

### Features del modello:
```
- GFS forecast high (°F)
- ECMWF forecast high (°F)
- HRRR forecast high (°F, se disponibile)
- Spread inter-modello: max(GFS,ECMWF,HRRR) - min(GFS,ECMWF,HRRR)
- METAR temperatura attuale
- METAR direzione vento (categorica: N/NE/E/SE/S/SW/W/NW)
- METAR velocità vento (kts)
- METAR copertura nuvolosa (categorica)
- Mese (1-12)
- Media mobile a 7 giorni dell'errore forecast per questa stazione
- Errore in tempo reale: GFS_predicted_temp_ora_corrente - METAR_temp_ora_corrente
  (SE il GFS sbaglia già di 3°F alle 10am, probabilmente sbaglierà anche per la massima)
```

### Target del modello:
```
forecast_error = GFS_predicted_daily_high - actual_observed_daily_high
```
(positivo = modello troppo caldo, negativo = troppo freddo)

### Validazione walk-forward:
```
Train su mesi 1-6, test su mese 7
Train su mesi 1-7, test su mese 8
... etc.
MAI testare su dati usati per il training.
```

### Integrazione:
```python
predicted_error = model.predict(features_today)
corrected_forecast = gfs_raw - predicted_error
# Usa corrected_forecast per il calcolo probabilità
```

### Metriche da riportare:
- R² del modello (se <0.15, il modello non funziona — l'errore è troppo random)
- MAE (mean absolute error) del modello di predizione errore
- Confronto backtest: WR/ROI con ML correction vs con bias correction vs senza correzione

---

## RIASSUNTO ORDINE DI ESECUZIONE

```
STEP 1: Fix 4 bug → test → riporta risultati
STEP 2: Market parser Polymarket → test → riporta quanti mercati mappati
STEP 3: Kalshi public API → test → riporta quanti mercati trovati
STEP 4: BACKTESTER → esegui → riporta WIN RATE, ROI, SHARPE, PER-CITY
         ↓
         SE WR < 52% → STOP. Riporta e aspetta istruzioni.
         SE WR ≥ 52% → continua
         ↓
STEP 5: Bias correction → riesegui backtest → confronta con/senza
         ↓
         SE la correzione migliora WR di ≥2 punti → continua
         SE no → STOP. La bias correction non aggiunge valore.
         ↓
STEP 6: ML error prediction → riesegui backtest → confronta con Step 5
         ↓
         Riporta risultati finali completi.
```

Ad ogni STOP, aspetta istruzioni prima di procedere. Non prendere decisioni autonome sulla strategia.
