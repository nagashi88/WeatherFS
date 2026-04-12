# Checkpoint — 2026-04-12 13:15:49 CEST

## Scope
Snapshot before implementing the next calibration-policy change
(include early-exit trades in calibration once market final outcome is known).

## Runtime Status
- Bot session: `tmux weatheredge` active
- Dashboard: `http://127.0.0.1:8502` -> `HTTP/1.1 200 OK`

## Data Status (SQLite)
- `forecasts.actual_temp` populated rows: **21330**
- `calibration` rows: **10**
- `calibration_records` rows: **0**
- Trades: **7 open**, **3 resolved**

## Implemented in current state
- Real observed-temperature ingestion via Open-Meteo archive into `Forecast.actual_temp`
- Station/model normalization for calibration pipeline
- RMSE persistence in `calibration` table
- Startup + scheduled backfill jobs for actuals
- Intraday resolution check scheduling
- Calibration record write path on final market resolution (won/lost)

## Known behavior (to be changed next)
- `calibration_records` currently updates only from final market resolution flow.
- Early exits (`edge_stop`, `capital_protect`, `time_decay`, `ev_based`) are not
  yet backfilled into calibration at eventual final market outcome.

