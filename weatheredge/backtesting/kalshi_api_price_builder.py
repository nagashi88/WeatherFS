"""Build backtester-compatible price CSV from Kalshi public API.

This builder targets weather temperature series (KXHIGH*), downloads historical
events/markets, and converts bucket prices into implied threshold probabilities:
P(T >= threshold) on a 5°F grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

SERIES_TO_STATION: dict[str, str] = {
    "KXHIGHNY": "KNYC",
    "KXHIGHCHI": "KMDW",
    "KXHIGHMIA": "KMIA",
    "KXHIGHDEN": "KDEN",
    "KXHIGHLAX": "KLAX",
}


@dataclass(frozen=True)
class _BucketQuote:
    low: float | None
    high: float | None
    prob: float


@dataclass
class BuildReport:
    events_scanned: int
    markets_scanned: int
    bucket_markets_used: int
    rows_written: int
    output_path: str


def _http_json(path: str, params: dict[str, object] | None = None) -> dict:
    query = ""
    if params:
        clean = {k: v for k, v in params.items() if v is not None}
        query = "?" + urllib.parse.urlencode(clean)
    url = f"{BASE_URL}/{path}{query}"
    req = urllib.request.Request(url, headers={"User-Agent": "WeatherEdge/1.0"})

    last_exc: Exception | None = None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            retry_after = exc.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_s = float(retry_after)
                except ValueError:
                    sleep_s = min(8.0, 0.5 * (2 ** attempt))
            else:
                sleep_s = min(8.0, 0.5 * (2 ** attempt))
            time.sleep(sleep_s)
        except urllib.error.URLError as exc:
            last_exc = exc
            time.sleep(min(8.0, 0.5 * (2 ** attempt)))

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("HTTP request failed without exception")


def _parse_event_date(event_ticker: str) -> date | None:
    # Expected: KXHIGHDEN-26APR08
    parts = str(event_ticker).split("-")
    if len(parts) < 2:
        return None
    try:
        return datetime.strptime(parts[1], "%y%b%d").date()
    except ValueError:
        return None


def _pick_market_prob(market: dict) -> float | None:
    # Prefer previous/last price, fallback to settlement value.
    candidates = (
        market.get("previous_price_dollars"),
        market.get("last_price_dollars"),
        market.get("settlement_value_dollars"),
    )
    for raw in candidates:
        if raw is None:
            continue
        try:
            p = float(raw)
        except (TypeError, ValueError):
            continue
        if 0.0 <= p <= 1.0:
            return p
    return None


def _bucket_midpoint(low: float | None, high: float | None) -> float | None:
    # Weather buckets are typically 1°F wide; use +/-1 fallback for open tails.
    if low is not None and high is not None:
        return (low + high) / 2.0
    if low is None and high is not None:
        return high - 1.0
    if low is not None and high is None:
        return low + 1.0
    return None


def _threshold_grid_from_midpoints(midpoints: list[float]) -> list[float]:
    if not midpoints:
        return []
    lo = min(midpoints)
    hi = max(midpoints)
    start = int((lo // 5) * 5)
    end = int(((hi + 4.999) // 5) * 5)
    return [float(t) for t in range(start, end + 1, 5)]


def _implied_rows_for_event(
    station: str,
    contract_date: date,
    quotes: list[_BucketQuote],
) -> list[tuple[str, str, str, str, str]]:
    # Normalize to probability mass across mutually-exclusive buckets.
    mass = sum(q.prob for q in quotes if q.prob > 0.0)
    if mass <= 0.0:
        return []

    masses: list[tuple[float, float]] = []  # (midpoint, normalized_prob)
    for q in quotes:
        if q.prob <= 0.0:
            continue
        mid = _bucket_midpoint(q.low, q.high)
        if mid is None:
            continue
        masses.append((mid, q.prob / mass))

    if not masses:
        return []

    thresholds = _threshold_grid_from_midpoints([m[0] for m in masses])
    rows: list[tuple[str, str, str, str, str]] = []
    for th in thresholds:
        p_above = sum(prob for mid, prob in masses if mid >= th)
        p_above = max(0.0, min(1.0, p_above))
        p_below = 1.0 - p_above
        rows.append((station, contract_date.isoformat(), f"{th:.2f}", "above", f"{p_above:.6f}"))
        rows.append((station, contract_date.isoformat(), f"{th:.2f}", "below", f"{p_below:.6f}"))
    return rows


def _fetch_events(series_ticker: str, max_events: int | None = None) -> list[dict]:
    events: list[dict] = []
    cursor: str | None = None
    while True:
        payload = _http_json(
            "events",
            {"series_ticker": series_ticker, "limit": 200, "cursor": cursor},
        )
        chunk = payload.get("events", [])
        if not chunk:
            break
        events.extend(chunk)
        if max_events is not None and len(events) >= max_events:
            return events[:max_events]
        cursor = payload.get("cursor")
        if not cursor:
            break
        time.sleep(0.05)
    return events


def _fetch_markets_for_event(event_ticker: str) -> list[dict]:
    markets: list[dict] = []
    cursor: str | None = None
    while True:
        payload = _http_json(
            "markets",
            {"event_ticker": event_ticker, "limit": 200, "cursor": cursor},
        )
        chunk = payload.get("markets", [])
        if not chunk:
            break
        markets.extend(chunk)
        cursor = payload.get("cursor")
        if not cursor:
            break
        time.sleep(0.05)
    return markets


def build_prices_csv_from_kalshi_api(
    output_csv: str | Path,
    start_date: date,
    end_date: date,
    series_tickers: list[str],
    max_events_per_series: int | None = None,
) -> BuildReport:
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows_set: set[tuple[str, str, str, str, str]] = set()
    events_scanned = 0
    markets_scanned = 0
    bucket_markets_used = 0

    for series in series_tickers:
        station = SERIES_TO_STATION.get(series)
        if station is None:
            continue
        events = _fetch_events(series, max_events=max_events_per_series)
        for event in events:
            event_ticker = event.get("event_ticker")
            if not event_ticker:
                continue
            event_date = _parse_event_date(str(event_ticker))
            if event_date is None:
                continue
            if event_date < start_date or event_date > end_date:
                continue

            events_scanned += 1
            markets = _fetch_markets_for_event(str(event_ticker))
            quotes: list[_BucketQuote] = []
            for m in markets:
                markets_scanned += 1
                if str(m.get("strike_type", "")).lower() != "between":
                    continue
                prob = _pick_market_prob(m)
                if prob is None:
                    continue
                low = m.get("floor_strike")
                high = m.get("cap_strike")
                low_f = float(low) if low is not None else None
                high_f = float(high) if high is not None else None
                quotes.append(_BucketQuote(low=low_f, high=high_f, prob=prob))
                bucket_markets_used += 1

            implied = _implied_rows_for_event(station, event_date, quotes)
            for row in implied:
                rows_set.add(row)
            time.sleep(0.05)

    rows = sorted(rows_set, key=lambda r: (r[0], r[1], float(r[2]), r[3]))
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["station", "date", "threshold", "direction", "yes_price"])
        writer.writerows(rows)

    return BuildReport(
        events_scanned=events_scanned,
        markets_scanned=markets_scanned,
        bucket_markets_used=bucket_markets_used,
        rows_written=len(rows),
        output_path=str(out),
    )


def _parse_date_arg(raw: str) -> date:
    return date.fromisoformat(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build backtester price CSV from Kalshi weather API."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path",
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--series",
        default="KXHIGHNY,KXHIGHCHI,KXHIGHMIA,KXHIGHDEN,KXHIGHLAX",
        help="Comma-separated Kalshi series tickers.",
    )
    parser.add_argument(
        "--max-events-per-series",
        type=int,
        default=None,
        help="Optional cap to reduce API load during initial runs.",
    )
    args = parser.parse_args(argv)

    report = build_prices_csv_from_kalshi_api(
        output_csv=args.output,
        start_date=_parse_date_arg(args.start_date),
        end_date=_parse_date_arg(args.end_date),
        series_tickers=[s.strip() for s in args.series.split(",") if s.strip()],
        max_events_per_series=args.max_events_per_series,
    )
    print(
        "events={events} markets={markets} bucket_markets={bucket} rows={rows} output={out}".format(
            events=report.events_scanned,
            markets=report.markets_scanned,
            bucket=report.bucket_markets_used,
            rows=report.rows_written,
            out=report.output_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
