"""Build backtester-compatible market price CSV from local SQLite snapshots."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from weatheredge.backtesting.market_prices import _extract_from_kalshi_ticker


@dataclass
class BuildDbMarketPricesReport:
    scanned_rows: int
    used_rows: int
    output_rows: int
    output_path: str


def _parse_date_arg(raw: str | None) -> date | None:
    if not raw:
        return None
    return date.fromisoformat(raw)


def _parse_timestamp_to_sort_key(value: object) -> float:
    if value is None:
        return 0.0

    raw = str(value).strip()
    if not raw:
        return 0.0

    try:
        ts = float(raw)
        if ts > 1e12:
            ts = ts / 1000.0
        return ts
    except ValueError:
        pass

    normalized = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return 0.0


def _parse_contract_date(value: object) -> date | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _parse_direction(value: object) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in {"above", "over", "higher", "gt", ">="}:
        return "above"
    if raw in {"below", "under", "lower", "lt", "<"}:
        return "below"
    return None


def _extract_key_fields(
    market_id: object,
    market_station: object,
    market_date: object,
    threshold_low: object,
    threshold_high: object,
    resolution_source: object,
) -> tuple[str | None, date | None, float | None, str | None]:
    # First try canonical Kalshi ticker parse.
    station, contract_date, threshold, direction = _extract_from_kalshi_ticker(
        None if market_id is None else str(market_id)
    )

    metadata: dict = {}
    if resolution_source is not None:
        try:
            metadata = json.loads(str(resolution_source))
        except json.JSONDecodeError:
            metadata = {}

    if station is None and market_station is not None and str(market_station).strip():
        station = str(market_station).strip().upper()

    if contract_date is None:
        contract_date = _parse_contract_date(market_date)

    if threshold is None:
        candidate = metadata.get("threshold")
        if candidate is None:
            candidate = threshold_high if threshold_high is not None else threshold_low
        try:
            threshold = round(float(candidate), 2)
        except (TypeError, ValueError):
            threshold = None

    if direction is None:
        direction = _parse_direction(metadata.get("direction"))
        # Bucket markets often omit direction; treat as YES on bucket condition.
        if direction is None and threshold_low is not None and threshold_high is not None:
            direction = "above"

    return station, contract_date, threshold, direction


def build_market_prices_csv_from_db(
    db_path: str | Path,
    output_csv: str | Path,
    start_date: date | None = None,
    end_date: date | None = None,
) -> BuildDbMarketPricesReport:
    db = Path(db_path)
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    latest: dict[tuple[str, date, float, str], tuple[float, float]] = {}
    scanned = 0
    used = 0

    with sqlite3.connect(db) as conn:
        has_markets = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='markets' LIMIT 1"
            ).fetchone()
            is not None
        )

        if has_markets:
            query = """
                SELECT
                    mp.market_id,
                    mp.timestamp,
                    mp.yes_price,
                    m.station,
                    m.date,
                    m.threshold_low,
                    m.threshold_high,
                    m.resolution_source
                FROM market_prices mp
                LEFT JOIN markets m ON m.id = mp.market_id
                WHERE mp.yes_price IS NOT NULL
                ORDER BY mp.timestamp ASC
            """
        else:
            query = """
                SELECT market_id, timestamp, yes_price
                FROM market_prices
                WHERE yes_price IS NOT NULL
                ORDER BY timestamp ASC
            """

        for row in conn.execute(query):
            scanned += 1
            if has_markets:
                (
                    market_id,
                    ts_raw,
                    yes_raw,
                    market_station,
                    market_date,
                    threshold_low,
                    threshold_high,
                    resolution_source,
                ) = row
            else:
                market_id, ts_raw, yes_raw = row
                market_station = None
                market_date = None
                threshold_low = None
                threshold_high = None
                resolution_source = None

            station, contract_date, threshold, direction = _extract_key_fields(
                market_id=market_id,
                market_station=market_station,
                market_date=market_date,
                threshold_low=threshold_low,
                threshold_high=threshold_high,
                resolution_source=resolution_source,
            )
            if station is None or contract_date is None or threshold is None or direction is None:
                continue
            if start_date and contract_date < start_date:
                continue
            if end_date and contract_date > end_date:
                continue

            try:
                yes_price = float(yes_raw)
            except (TypeError, ValueError):
                continue
            if yes_price < 0.0 or yes_price > 1.0:
                continue

            key = (station, contract_date, round(float(threshold), 2), direction)
            ts_key = _parse_timestamp_to_sort_key(ts_raw)
            prev = latest.get(key)
            if prev is None or ts_key >= prev[0]:
                latest[key] = (ts_key, yes_price)
                used += 1

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["station", "date", "threshold", "direction", "yes_price"])
        for station, contract_date, threshold, direction in sorted(latest.keys()):
            yes_price = latest[(station, contract_date, threshold, direction)][1]
            writer.writerow(
                [station, contract_date.isoformat(), f"{threshold:.2f}", direction, f"{yes_price:.6f}"]
            )

    return BuildDbMarketPricesReport(
        scanned_rows=scanned,
        used_rows=used,
        output_rows=len(latest),
        output_path=str(out),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build backtest price CSV from weatheredge.db market_prices table."
    )
    parser.add_argument("--db", required=True, help="Path to SQLite DB (weatheredge.db)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    args = parser.parse_args(argv)

    report = build_market_prices_csv_from_db(
        db_path=args.db,
        output_csv=args.output,
        start_date=_parse_date_arg(args.start_date),
        end_date=_parse_date_arg(args.end_date),
    )
    print(
        "built rows={rows} scanned={scanned} used={used} output={out}".format(
            rows=report.output_rows,
            scanned=report.scanned_rows,
            used=report.used_rows,
            out=report.output_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
