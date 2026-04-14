"""Backfill 120 giorni di dati forecast vs actual nella tabella forecasts.

Inserisce righe minimali (una per stazione × modello × giorno) con:
  - temperature_forecast  → Open-Meteo Historical Forecast API
  - actual_temp           → Open-Meteo Archive API

Alimenta:
  - RMSEUpdater (recalculate_rmse)
  - run_bias_correction_refresh → STATION_BIAS + BIAS_TABLE

Uso:
    python scripts/backfill_calibration.py
    python scripts/backfill_calibration.py --days 90
    python scripts/backfill_calibration.py --days 120 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
import urllib.request
import urllib.parse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

# ---------------------------------------------------------------------------
# Stazioni
# ---------------------------------------------------------------------------

STATIONS: dict[str, tuple[float, float]] = {
    "KNYC": (40.7128, -74.0060),
    "KMDW": (41.8781, -87.6298),
    "KMIA": (25.7617, -80.1918),
    "KDEN": (39.7392, -104.9903),
    "KLAX": (33.9425, -118.4081),
}

# Modelli da backfillare e il corrispondente nome Open-Meteo
MODELS: dict[str, str] = {
    "gfs": "gfs_seamless",
    "hrrr": "best_match",   # best_match include HRRR per USA
}

# ---------------------------------------------------------------------------
# Open-Meteo helpers
# ---------------------------------------------------------------------------

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def _get_json(url: str, params: dict, retries: int = 3) -> dict:
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(full_url, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            log.warning("HTTP attempt %d/%d failed (%s) — retry in %ds", attempt, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_actual_temps(
    lat: float,
    lon: float,
    start: date,
    end: date,
) -> dict[str, float]:
    """Ritorna {YYYY-MM-DD: temp_max_F} da Open-Meteo Archive."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
    }
    data = _get_json(ARCHIVE_URL, params)
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_max", [])
    return {d: float(t) for d, t in zip(dates, temps) if t is not None}


def fetch_forecast_temps(
    lat: float,
    lon: float,
    start: date,
    end: date,
    model: str,
) -> dict[str, float]:
    """Ritorna {YYYY-MM-DD: temp_max_F} da Open-Meteo Historical Forecast.

    Rappresenta la previsione del giorno precedente (lead ~24h).
    """
    # Le API historical forecast restituiscono i valori del modello in quella data.
    # Usiamo la previsione del giorno stesso come proxy del forecast 24h.
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": "UTC",
        "models": model,
    }
    data = _get_json(HIST_FORECAST_URL, params)
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_max", [])
    return {d: float(t) for d, t in zip(dates, temps) if t is not None}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _db_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "weatheredge.db"


def _existing_keys(conn: sqlite3.Connection) -> set[tuple[str, str, str]]:
    """Ritorna set di (station, model, target_date) già presenti come backfill."""
    cur = conn.execute(
        "SELECT station, model, target_date FROM forecasts "
        "WHERE lead_time_hours = 24 AND ensemble_members IS NULL"
    )
    return {(r[0], r[1], r[2]) for r in cur.fetchall()}


def insert_rows(
    conn: sqlite3.Connection,
    rows: list[tuple],
    dry_run: bool,
) -> int:
    """Inserisce righe (station, model, target_date, forecast_ts, temp_forecast, actual_temp)."""
    if not rows:
        return 0
    if dry_run:
        log.info("DRY-RUN: %d righe da inserire (non scritte)", len(rows))
        return len(rows)

    conn.executemany(
        """
        INSERT INTO forecasts
            (station, model, target_date, forecast_timestamp,
             lead_time_hours, temperature_forecast, actual_temp, created_at)
        VALUES (?, ?, ?, ?, 24, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Main backfill
# ---------------------------------------------------------------------------

def run_backfill(days: int, dry_run: bool) -> None:
    today = date.today()
    # Backfill fino a ieri (oggi non è ancora completo)
    end_date = today - timedelta(days=1)
    start_date = today - timedelta(days=days)

    log.info(
        "Backfill %d giorni: %s → %s | dry_run=%s",
        days, start_date, end_date, dry_run,
    )

    db = _db_path()
    if not db.exists():
        raise FileNotFoundError(f"DB non trovato: {db}")

    conn = sqlite3.connect(str(db))
    existing = _existing_keys(conn)
    log.info("Chiavi già presenti nel DB: %d", len(existing))

    now_ts = datetime.now(timezone.utc).isoformat()
    total_inserted = 0
    total_skipped = 0

    for station, (lat, lon) in STATIONS.items():
        log.info("── Stazione: %s (%.4f, %.4f)", station, lat, lon)

        # 1. Scarica actual temps (una sola chiamata per stazione)
        log.info("   Scarico actual temps da Open-Meteo Archive...")
        try:
            actuals = fetch_actual_temps(lat, lon, start_date, end_date)
            log.info("   Actual temps: %d giorni", len(actuals))
        except Exception as exc:
            log.error("   ERRORE actual temps %s: %s", station, exc)
            continue
        time.sleep(0.5)  # rate limit

        for model_name, om_model in MODELS.items():
            log.info("   Modello: %s (%s)", model_name, om_model)

            # 2. Scarica forecast temps storici
            try:
                forecasts = fetch_forecast_temps(lat, lon, start_date, end_date, om_model)
                log.info("   Forecast temps: %d giorni", len(forecasts))
            except Exception as exc:
                log.error("   ERRORE forecast temps %s/%s: %s", station, model_name, exc)
                continue
            time.sleep(0.5)

            # 3. Costruisci righe da inserire
            rows = []
            for day_str, actual in actuals.items():
                key = (station, model_name, day_str)
                if key in existing:
                    total_skipped += 1
                    continue
                forecast = forecasts.get(day_str)
                if forecast is None:
                    continue  # senza forecast non serve alla calibrazione
                forecast_ts = f"{day_str}T00:00:00+00:00"
                rows.append((station, model_name, day_str, forecast_ts, forecast, actual, now_ts))

            inserted = insert_rows(conn, rows, dry_run)
            total_inserted += inserted
            log.info(
                "   %s/%s: %d inseriti, %d già presenti",
                station, model_name, inserted, total_skipped,
            )

    conn.close()
    log.info(
        "Backfill completato: %d righe inserite, %d saltate (già presenti)",
        total_inserted, total_skipped,
    )
    if not dry_run and total_inserted > 0:
        log.info(
            "Prossimo passo: riavvia il bot oppure chiama run_bias_correction_refresh() "
            "e rmse_updater.update_all_rmse() per aggiornare le tabelle in-memory."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill calibration data (forecast vs actual temps)")
    parser.add_argument("--days", type=int, default=120, help="Giorni di storico (default: 120)")
    parser.add_argument("--dry-run", action="store_true", help="Simula senza scrivere nel DB")
    args = parser.parse_args()

    run_backfill(days=args.days, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
