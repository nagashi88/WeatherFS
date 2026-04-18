"""Parse Polymarket weather market titles into structured resolution data.

Uses regex only — no LLM calls. Falls back to Gap 7b (skip + warn) on parse failure.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from datetime import datetime

log = logging.getLogger(__name__)

CITY_TO_STATION: dict[str, str] = {
    # NYC variants
    "nyc": "KNYC", "new york city": "KNYC", "new york": "KNYC",
    # Chicago
    "chicago": "KMDW",
    # Miami
    "miami": "KMIA",
    # Denver
    "denver": "KDEN", "den": "KDEN",
    # LA variants
    "los angeles": "KLAX", "la": "KLAX", "lax": "KLAX",
    # London
    "london": "EGLC",
    # Austin
    "austin": "KAUS",
    # Seattle
    "seattle": "KSEA",
    # Buenos Aires
    "buenos aires": "SAEZ",
    # Ankara
    "ankara": "LTAC",
    # Seoul
    "seoul": "RKSS",
    # Atlanta
    "atlanta": "KATL",
    # Dallas
    "dallas": "KDFW",
    # Phoenix
    "phoenix": "KPHX",
    # Houston
    "houston": "KIAH",
    # San Francisco
    "san francisco": "KSFO", "sf": "KSFO",
    # Boston
    "boston": "KBOS",
    # --- International expansion ---
    # Tokyo
    "tokyo": "RJTT",
    # Moscow
    "moscow": "UUEE",
    # Milan
    "milan": "LIML",
    # Madrid
    "madrid": "LEMD",
    # Istanbul
    "istanbul": "LTFM",
    # Singapore
    "singapore": "WSSS",
    # Munich
    "munich": "EDDM",
    # Taipei
    "taipei": "RCTP",
    # Paris
    "paris": "LFPG",
    # Toronto
    "toronto": "CYYZ",
    # Shanghai
    "shanghai": "ZSPD",
    # Beijing
    "beijing": "ZBAA",
    # Wellington
    "wellington": "NZWN",
    # Warsaw
    "warsaw": "EPWA",
    # Tel Aviv (multi-word — long key matched first by sort)
    "tel aviv": "LLBG",
    # Chengdu
    "chengdu": "ZUUU",
    # Chongqing
    "chongqing": "ZUCK",
    # Wuhan
    "wuhan": "ZHHH",
    # Shenzhen
    "shenzhen": "ZGSZ",
    # Lucknow
    "lucknow": "VILK",
    # São Paulo (title may render as "Paulo" if ã breaks regex — add both)
    "são paulo": "SBGR", "sao paulo": "SBGR",
    # Hong Kong (multi-word — long key matched first by sort)
    "hong kong": "VHHH",
    # Mexico City (multi-word)
    "mexico city": "MMMX",
}

# Stations in US (HRRR coverage)
US_STATIONS: set[str] = {
    "KNYC", "KMDW", "KMIA", "KDEN", "KLAX", "KAUS", "KSEA",
    "KATL", "KDFW", "KPHX", "KIAH", "KSFO", "KBOS",
}

@dataclass
class ParsedMarket:
    city: str
    station: str
    date: str  # YYYY-MM-DD
    market_type: str  # "daily_high", "daily_low", etc.
    threshold: float | None = None  # Single threshold (above/below)
    threshold_low: float | None = None  # Bucket low
    threshold_high: float | None = None  # Bucket high
    direction: str = "above"  # "above" or "below"
    unit: str = "F"            # "F" = Fahrenheit, "C" = Celsius
    source: str = "parsed_from_title"

def parse_polymarket_title(title: str, current_year: int | None = None) -> ParsedMarket | None:
    """Parse a Polymarket weather market title into structured data.

    Handles these patterns:
    1. Bucket: "between 50-51°F" or "between 50-51 °F"
    2. Lower bound: "55°F or above" / "55°F or higher"
    3. Upper bound: "43°F or below" / "49°F or below"
    4. Exceed: "exceed 60°F"

    Returns None if parsing fails.
    """
    # BugFix M5: Derive current year dynamically instead of hardcoding 2026
    if current_year is None:
        current_year = datetime.now().year

    title_lower = title.lower()

    # --- Extract city ---
    city = None
    station = None
    # BugFix M13: Use word boundary matching to prevent substring false matches (e.g., "la" in "dallas")
    for city_name, station_code in sorted(CITY_TO_STATION.items(), key=lambda x: -len(x[0])):
        if re.search(r'\b' + re.escape(city_name) + r'\b', title_lower):
            city = city_name
            station = station_code
            break

    if not city or not station:
        # Try to extract city name from "in {City}" pattern for alerting
        city_match = re.search(r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', title)
        unknown_city = city_match.group(1) if city_match else "UNKNOWN"
        log.warning(
            "UNKNOWN_CITY_HIGH_PRIORITY: could not map city from title: '%s' (extracted: %s)",
            title[:80], unknown_city,
        )
        return None

    # --- Extract date ---
    # Pattern: "on April 4" or "on March 30"
    date_match = re.search(r'on\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})', title_lower)
    if not date_match:
        return None

    month_name = date_match.group(1)
    day = int(date_match.group(2))
    month_map = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
                 "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12}
    month = month_map[month_name]
    date_str = f"{current_year}-{month:02d}-{day:02d}"

    # --- Determine market type ---
    # Substring "low" matches "below" (e.g. "55°F or below"), so require
    # "low(est) temperature" as a word-adjacent phrase: classify as daily_low
    # only when the contract subject is the daily minimum.
    market_type = "daily_high"  # Default
    if re.search(r'\blow(?:est)?\s+temperature\b', title_lower):
        market_type = "daily_low"

    # --- Extract threshold/bucket ---

    # Pattern 1: "between X-Y°F" (bucket market)
    bucket_match = re.search(r'between\s+(\d+)\s*-\s*(\d+)\s*°?\s*f', title_lower)
    if bucket_match:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold_low=float(bucket_match.group(1)),
            threshold_high=float(bucket_match.group(2)),
            direction="above",
        )

    # Pattern 2: "X°F or above" / "X°F or higher" (lower bound)
    above_match = re.search(r'(\d+)\s*°?\s*f\s+or\s+(?:above|higher)', title_lower)
    if above_match:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(above_match.group(1)),
            direction="above",
        )

    # Pattern 3: "X°F or below" / "X°F or lower" (upper bound)
    below_match = re.search(r'(\d+)\s*°?\s*f\s+or\s+(?:below|lower)', title_lower)
    if below_match:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(below_match.group(1)),
            direction="below",
        )

    # Pattern 4: "exceed X°F"
    exceed_match = re.search(r'exceed\s+(\d+)\s*°?\s*f', title_lower)
    if exceed_match:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(exceed_match.group(1)),
            direction="above",
        )

    # Pattern 5: "be X°F" (exact or simple threshold) - less common
    be_match = re.search(r'be\s+(\d+)\s*°?\s*f', title_lower)
    if be_match:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(be_match.group(1)),
            direction="above",
        )

    # --- °C patterns (international markets) ---

    # Pattern 1C: "between X-Y°C" (bucket)
    bucket_c = re.search(r'between\s+(\d+)\s*-\s*(\d+)\s*°?\s*c\b', title_lower)
    if bucket_c:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold_low=float(bucket_c.group(1)),
            threshold_high=float(bucket_c.group(2)),
            direction="above", unit="C",
        )

    # Pattern 2C: "X°C or above/higher"
    above_c = re.search(r'(\d+)\s*°?\s*c\s+or\s+(?:above|higher)', title_lower)
    if above_c:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(above_c.group(1)),
            direction="above", unit="C",
        )

    # Pattern 3C: "X°C or below/lower"
    below_c = re.search(r'(\d+)\s*°?\s*c\s+or\s+(?:below|lower)', title_lower)
    if below_c:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(below_c.group(1)),
            direction="below", unit="C",
        )

    # Pattern 4C: "exceed X°C"
    exceed_c = re.search(r'exceed\s+(\d+)\s*°?\s*c\b', title_lower)
    if exceed_c:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(exceed_c.group(1)),
            direction="above", unit="C",
        )

    # Pattern 5C: "be X°C" (exact/simple)
    be_c = re.search(r'be\s+(\d+)\s*°?\s*c\b', title_lower)
    if be_c:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(be_c.group(1)),
            direction="above", unit="C",
        )

    return None
