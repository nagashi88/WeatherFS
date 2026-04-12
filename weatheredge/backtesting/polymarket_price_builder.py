"""Build backtester-compatible market price CSV from poly_data snapshots."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import IO, Sequence

import pandas as pd

from weatheredge.data.market_parser import parse_polymarket_title


@dataclass
class BuildPolymarketReport:
    markets_rows: int
    parsed_weather_markets: int
    trade_rows_scanned: int
    trade_rows_used: int
    output_rows: int
    output_path: str


# Common date/time metadata seen in market snapshot CSVs.
_MARKET_DATE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "end_date_iso",
    "end_date",
    "close_time",
    "closed_time",
    "start_date_iso",
    "start_date",
    "event_date",
    "event_start_time",
    "resolution_time",
    "resolved_at",
    "accepting_orders_until",
)


def _parse_ts_to_date(value: object) -> date | None:
    if value is None:
        return None
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None

    # poly_data timestamps are usually unix seconds; support ms as fallback.
    if ts > 1e12:
        ts = ts / 1000.0
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).date()
    except (OSError, OverflowError, ValueError):
        return None


def _parse_epoch_seconds(value: object) -> float | None:
    if value is None:
        return None
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    # Support ms timestamps too.
    if ts > 1e12:
        ts = ts / 1000.0
    return ts


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None

    ts = _parse_epoch_seconds(value)
    if ts is not None:
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "nat"}:
        return None

    normalized = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = pd.to_datetime(raw, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        dt = parsed.to_pydatetime()

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _infer_market_year(row: pd.Series, date_columns: Sequence[str]) -> int | None:
    for col in date_columns:
        dt = _parse_datetime(row.get(col))
        if dt is not None:
            return dt.year
    return None


def _side_to_yes_price(price: float, nonusdc_side: object, answer1: str | None, answer2: str | None) -> float | None:
    if price < 0.0 or price > 1.0:
        return None

    a1 = (answer1 or "").strip().lower()
    a2 = (answer2 or "").strip().lower()
    side = str(nonusdc_side).strip().lower()

    # If answer labels are explicit yes/no, map side->YES price accordingly.
    if a1 == "yes" and a2 == "no":
        if side in {"token1", "1", "answer1"}:
            return price
        if side in {"token2", "2", "answer2"}:
            return 1.0 - price
    if a1 == "no" and a2 == "yes":
        if side in {"token1", "1", "answer1"}:
            return 1.0 - price
        if side in {"token2", "2", "answer2"}:
            return price

    # Fallback: assume token1 is YES (common in binary markets where answer1=="Yes").
    if side in {"token1", "1", "answer1"}:
        return price
    if side in {"token2", "2", "answer2"}:
        return 1.0 - price
    return None


def build_market_prices_csv_from_poly_data(
    markets_csv: str | Path | IO[str],
    trades_csv: str | Path | IO[str] | IO[bytes],
    output_csv: str | Path,
    start_date: date | None = None,
    end_date: date | None = None,
    chunksize: int = 500_000,
) -> BuildPolymarketReport:
    markets_df = pd.read_csv(markets_csv)
    markets_df.columns = [str(c).strip() for c in markets_df.columns]
    lower_to_col = {str(c).strip().lower(): str(c).strip() for c in markets_df.columns}
    id_col = lower_to_col.get("id") or lower_to_col.get("market_id")
    q_col = lower_to_col.get("question")
    if q_col is None:
        raise ValueError("markets CSV missing `question` column")
    if id_col is None:
        raise ValueError("markets CSV missing market id column (`id` or `market_id`)")

    market_date_cols = [
        lower_to_col[name] for name in _MARKET_DATE_COLUMN_CANDIDATES if name in lower_to_col
    ]

    market_meta: dict[str, dict] = {}
    parsed_weather = 0
    for _, row in markets_df.iterrows():
        market_id = str(row.get(id_col, "")).strip()
        question = str(row.get(q_col, "")).strip()
        if not market_id or not question:
            continue
        inferred_year = _infer_market_year(row, market_date_cols)
        parsed = parse_polymarket_title(question, current_year=inferred_year)
        if parsed is None or parsed.threshold is None:
            continue
        try:
            contract_date = date.fromisoformat(parsed.date)
        except ValueError:
            continue
        market_meta[market_id] = {
            "station": parsed.station,
            "date": contract_date,
            "threshold": float(parsed.threshold),
            "direction": parsed.direction,
            "answer1": str(row.get("answer1", "")).strip(),
            "answer2": str(row.get("answer2", "")).strip(),
        }
        parsed_weather += 1

    if not market_meta:
        raise ValueError("No parseable weather markets found in markets CSV")

    latest: dict[tuple[str, date, float, str], tuple[float, float]] = {}
    scanned = 0
    used = 0
    row_counter = 0
    for chunk in pd.read_csv(trades_csv, chunksize=chunksize):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        scanned += len(chunk)
        if "market_id" not in chunk.columns:
            continue
        for _, row in chunk.iterrows():
            row_counter += 1
            market_id = str(row.get("market_id", "")).strip()
            meta = market_meta.get(market_id)
            if meta is None:
                continue

            # Contract key must be tied to market's target date, not trade timestamp date.
            contract_date = meta["date"]
            if start_date and contract_date < start_date:
                continue
            if end_date and contract_date > end_date:
                continue

            try:
                price = float(row.get("price"))
            except (TypeError, ValueError):
                continue

            yes_price = _side_to_yes_price(
                price=price,
                nonusdc_side=row.get("nonusdc_side"),
                answer1=meta.get("answer1"),
                answer2=meta.get("answer2"),
            )
            if yes_price is None:
                continue
            if yes_price < 0.0 or yes_price > 1.0:
                continue

            key = (meta["station"], contract_date, float(meta["threshold"]), meta["direction"])
            ts = _parse_epoch_seconds(row.get("timestamp"))
            if ts is None:
                dt = _parse_datetime(row.get("timestamp"))
                ts = dt.timestamp() if dt is not None else float(row_counter)
            prev = latest.get(key)
            if prev is None or ts >= prev[0]:
                latest[key] = (ts, yes_price)
                used += 1

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["station", "date", "threshold", "direction", "yes_price"])
        for station, d, threshold, direction in sorted(latest.keys()):
            yes_price = latest[(station, d, threshold, direction)][1]
            writer.writerow([station, d.isoformat(), f"{threshold:.2f}", direction, f"{yes_price:.6f}"])

    return BuildPolymarketReport(
        markets_rows=len(markets_df),
        parsed_weather_markets=parsed_weather,
        trade_rows_scanned=scanned,
        trade_rows_used=used,
        output_rows=len(latest),
        output_path=str(out_path),
    )


def _parse_date_arg(raw: str | None) -> date | None:
    if not raw:
        return None
    return date.fromisoformat(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build backtest price CSV from poly_data markets/trades.")
    parser.add_argument("--markets", required=True, help="Path to markets.csv")
    parser.add_argument("--trades", required=True, help="Path to processed/trades.csv")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--chunksize", type=int, default=500000)
    args = parser.parse_args(argv)

    report = build_market_prices_csv_from_poly_data(
        markets_csv=args.markets,
        trades_csv=args.trades,
        output_csv=args.output,
        start_date=_parse_date_arg(args.start_date),
        end_date=_parse_date_arg(args.end_date),
        chunksize=args.chunksize,
    )
    print(
        "built rows={rows} weather_markets={mkts} scanned={scanned} used={used} output={out}".format(
            rows=report.output_rows,
            mkts=report.parsed_weather_markets,
            scanned=report.trade_rows_scanned,
            used=report.trade_rows_used,
            out=report.output_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
