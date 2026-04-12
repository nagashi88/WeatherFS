"""Historical market price loader for backtests.

Supports loading direction-specific YES prices from CSV and querying by
(station, date, threshold, direction). When only one direction is present
for a key, the complementary direction is synthesized as (1 - yes_price).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

_CITY_TO_STATION: dict[str, str] = {
    "NYC": "KNYC",
    "NEW YORK": "KNYC",
    "NEW YORK CITY": "KNYC",
    "CHICAGO": "KMDW",
    "MIDWAY": "KMDW",
    "MIAMI": "KMIA",
    "DENVER": "KDEN",
    "LOS ANGELES": "KLAX",
    "LAX": "KLAX",
}

_KALSHI_TICKER_TO_STATION: dict[str, str] = {
    "NY": "KNYC",
    "DEN": "KDEN",
    "MIA": "KMIA",
    "LAX": "KLAX",
    "MDW": "KMDW",
}


def _norm_station(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    upper = raw.upper()
    if upper in _CITY_TO_STATION:
        return _CITY_TO_STATION[upper]
    return upper


def _norm_direction(value: str | None) -> str | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v:
        return None
    if v in {"above", "over", "higher", "gt", ">="}:
        return "above"
    if v in {"below", "under", "lower", "lt", "<"}:
        return "below"
    return None


def _infer_direction_from_text(value: str | None) -> str | None:
    if value is None:
        return None
    t = str(value).strip().lower()
    if not t:
        return None
    if any(token in t for token in ("above", "over", "exceed", "higher", "-t")):
        return "above"
    if any(token in t for token in ("below", "under", "lower", "-b")):
        return "below"
    return None


def _extract_from_kalshi_ticker(
    ticker: str | None,
) -> tuple[str | None, date | None, float | None, str | None]:
    """Extract (station, date, threshold, direction) from KXHIGH ticker.

    Examples:
    - KXHIGHDEN-26MAR30-T76
    - KXHIGHDEN-26MAR30-B76.5
    """
    if ticker is None:
        return None, None, None, None
    t = str(ticker).strip().upper()
    if not t.startswith("KXHIGH"):
        return None, None, None, None

    parts = t.split("-")
    prefix = parts[0]  # e.g. KXHIGHDEN
    city_code = prefix.replace("KXHIGH", "")
    station = _KALSHI_TICKER_TO_STATION.get(city_code)

    market_date: date | None = None
    threshold: float | None = None
    direction: str | None = None

    if len(parts) >= 2:
        try:
            market_date = datetime.strptime(parts[1], "%y%b%d").date()
        except ValueError:
            market_date = None

    if len(parts) >= 3:
        tail = parts[2]  # T76 or B76.5
        if tail.startswith("T"):
            direction = "above"
            threshold = _parse_threshold(tail[1:])
        elif tail.startswith("B"):
            direction = "below"
            threshold = _parse_threshold(tail[1:])

    return station, market_date, threshold, direction


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _parse_threshold(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _parse_yes_price(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None

    # Accept cents format (0..100) and convert to 0..1.
    if p > 1.0 and p <= 100.0:
        p = p / 100.0
    if p < 0.0 or p > 1.0:
        return None
    return p


@dataclass
class HistoricalMarketPriceBook:
    """In-memory lookup of historical market YES prices."""

    _prices: dict[tuple[str, date, float, str], float] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: Iterable[dict]) -> "HistoricalMarketPriceBook":
        book = cls()
        inserted = 0
        skipped = 0

        for row in rows:
            row_ci = {str(k).strip().lower(): v for k, v in row.items()}

            ticker = row_ci.get("ticker") or row_ci.get("market_id") or row_ci.get("event_ticker")
            ticker_station, ticker_date, ticker_threshold, ticker_direction = _extract_from_kalshi_ticker(
                str(ticker) if ticker is not None else None
            )

            station = _norm_station(row_ci.get("station") or row_ci.get("city")) or ticker_station
            d = _parse_date(
                row_ci.get("date") or row_ci.get("trade_date") or row_ci.get("timestamp")
            ) or ticker_date
            threshold = _parse_threshold(row_ci.get("threshold") or row_ci.get("strike")) or ticker_threshold
            direction = _norm_direction(
                row_ci.get("direction")
                or row_ci.get("market_direction")
                or row_ci.get("contract_direction")
                or _infer_direction_from_text(row_ci.get("market_id"))
                or _infer_direction_from_text(row_ci.get("title"))
            ) or ticker_direction
            yes_price = _parse_yes_price(
                row_ci.get("yes_price")
                or row_ci.get("yes")
                or row_ci.get("yes_ask")
                or row_ci.get("yes_bid")
                or row_ci.get("last_price")
                or row_ci.get("price")
                or row_ci.get("market_price")
                or row_ci.get("probability")
            )

            if station is None or d is None or threshold is None or yes_price is None:
                skipped += 1
                continue

            # Missing direction defaults to "above" so complement can fill "below".
            norm_direction = direction or "above"
            key = (station, d, threshold, norm_direction)
            book._prices[key] = yes_price
            inserted += 1

        # Fill complementary direction when missing.
        for station, d, threshold, direction in list(book._prices.keys()):
            complement = "below" if direction == "above" else "above"
            ckey = (station, d, threshold, complement)
            if ckey not in book._prices:
                book._prices[ckey] = max(0.0, min(1.0, 1.0 - book._prices[(station, d, threshold, direction)]))

        log.info(
            "HistoricalMarketPriceBook loaded: %d rows inserted, %d skipped, %d keyed prices",
            inserted,
            skipped,
            len(book._prices),
        )
        return book

    @classmethod
    def from_csv(cls, path: str | Path) -> "HistoricalMarketPriceBook":
        csv_path = Path(path)
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return cls.from_rows(rows)

    def get_yes_price(
        self,
        station: str,
        trade_date: date,
        threshold: float,
        direction: str,
    ) -> float | None:
        key = (_norm_station(station) or station.upper(), trade_date, round(float(threshold), 2), _norm_direction(direction) or direction)
        return self._prices.get(key)

    def __len__(self) -> int:
        return len(self._prices)
