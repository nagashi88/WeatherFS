"""Build backtester-compatible market price CSVs from historical datasets.

Primary target: Kalshi weather market snapshots in Becker-style parquet data.
Output schema:
    station,date,threshold,direction,yes_price
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from weatheredge.backtesting.market_prices import HistoricalMarketPriceBook

log = logging.getLogger(__name__)


@dataclass
class BuildMarketPricesReport:
    input_files: int
    input_rows: int
    weather_rows: int
    output_rows: int
    output_path: str


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
        if f != f:  # NaN check
            return None
        return f
    except (TypeError, ValueError):
        return None


def _normalise_price(value: float | None) -> float | None:
    if value is None:
        return None
    p = float(value)
    if p > 1.0 and p <= 100.0:
        p = p / 100.0
    if p < 0.0 or p > 1.0:
        return None
    return p


def _choose_yes_price_from_row(row: dict) -> float | None:
    """Prefer spread midpoint, fallback to single-sided quotes then last price."""
    yes_bid = _normalise_price(_safe_float(row.get("yes_bid")))
    yes_ask = _normalise_price(_safe_float(row.get("yes_ask")))
    if yes_bid is not None and yes_ask is not None:
        return (yes_bid + yes_ask) / 2.0
    if yes_ask is not None:
        return yes_ask
    if yes_bid is not None:
        return yes_bid
    return _normalise_price(_safe_float(row.get("last_price")))


def kalshi_dataframe_to_backtest_rows(df: pd.DataFrame) -> list[dict]:
    """Convert a Kalshi snapshot dataframe into generic backtest price rows.

    Keeps only KXHIGH weather contracts and derives a single YES price per row.
    Returned rows are compatible with HistoricalMarketPriceBook.from_rows().
    """
    if df.empty:
        return []

    renamed = df.copy()
    renamed.columns = [str(c).strip().lower() for c in renamed.columns]
    if "ticker" not in renamed.columns:
        return []

    # Keep only daily-high weather contracts used by current trading logic.
    weather = renamed[renamed["ticker"].astype(str).str.upper().str.startswith("KXHIGH")].copy()
    if weather.empty:
        return []

    rows: list[dict] = []
    for _, rec in weather.iterrows():
        row = rec.to_dict()
        yes_price = _choose_yes_price_from_row(row)
        if yes_price is None:
            continue

        fetched_at = (
            row.get("_fetched_at")
            or row.get("fetched_at")
            or row.get("timestamp")
            or row.get("created_time")
        )

        rows.append(
            {
                "ticker": row.get("ticker"),
                "date": row.get("date"),  # optional; ticker parsing can supply date
                "threshold": row.get("threshold"),  # optional; ticker parsing can supply threshold
                "direction": row.get("direction"),  # optional; ticker parsing can supply direction
                "yes_price": yes_price,
                "_fetched_at": fetched_at,
            }
        )
    return rows


def _expand_parquet_inputs(paths: Sequence[str | Path]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() == ".parquet":
            files.append(p)
            continue
        if p.is_dir():
            files.extend(sorted(p.rglob("*.parquet")))
    return sorted(set(files))


def _rows_to_price_book(rows: Iterable[dict]) -> HistoricalMarketPriceBook:
    ordered = sorted(
        list(rows),
        key=lambda r: str(r.get("_fetched_at") or ""),
    )
    return HistoricalMarketPriceBook.from_rows(ordered)


def _book_to_rows(book: HistoricalMarketPriceBook) -> list[dict]:
    out: list[dict] = []
    for (station, d, threshold, direction), yes_price in book._prices.items():
        out.append(
            {
                "station": station,
                "date": d.isoformat(),
                "threshold": f"{threshold:.2f}",
                "direction": direction,
                "yes_price": f"{yes_price:.6f}",
            }
        )
    out.sort(key=lambda r: (r["station"], r["date"], float(r["threshold"]), r["direction"]))
    return out


def build_market_prices_csv_from_kalshi_parquet(
    parquet_inputs: Sequence[str | Path],
    output_csv: str | Path,
    start_date: date | None = None,
    end_date: date | None = None,
) -> BuildMarketPricesReport:
    """Build normalized market prices CSV from one or more Kalshi parquet sources."""
    files = _expand_parquet_inputs(parquet_inputs)
    if not files:
        raise FileNotFoundError("No parquet files found in provided inputs")

    all_rows: list[dict] = []
    input_rows = 0
    for file_path in files:
        frame = pd.read_parquet(file_path)
        input_rows += len(frame)
        all_rows.extend(kalshi_dataframe_to_backtest_rows(frame))

    book = _rows_to_price_book(all_rows)
    out_rows = _book_to_rows(book)

    if start_date is not None:
        out_rows = [r for r in out_rows if date.fromisoformat(r["date"]) >= start_date]
    if end_date is not None:
        out_rows = [r for r in out_rows if date.fromisoformat(r["date"]) <= end_date]

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["station", "date", "threshold", "direction", "yes_price"],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    report = BuildMarketPricesReport(
        input_files=len(files),
        input_rows=input_rows,
        weather_rows=len(all_rows),
        output_rows=len(out_rows),
        output_path=str(out_path),
    )
    log.info(
        "Built market prices CSV: files=%d rows=%d weather=%d output=%d path=%s",
        report.input_files,
        report.input_rows,
        report.weather_rows,
        report.output_rows,
        report.output_path,
    )
    return report


def _parse_date_arg(raw: str | None) -> date | None:
    if not raw:
        return None
    return date.fromisoformat(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build backtester market price CSV from Kalshi parquet snapshots."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Parquet file or directory (can be repeated).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument("--start-date", default=None, help="Filter start date YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Filter end date YYYY-MM-DD.")
    args = parser.parse_args(argv)

    report = build_market_prices_csv_from_kalshi_parquet(
        parquet_inputs=args.input,
        output_csv=args.output,
        start_date=_parse_date_arg(args.start_date),
        end_date=_parse_date_arg(args.end_date),
    )
    print(
        f"Built {report.output_rows} rows from {report.weather_rows} weather snapshots "
        f"({report.input_files} files) -> {report.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
