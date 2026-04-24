"""Parse Polymarket weather market titles into structured resolution data.

Uses regex only — no LLM calls. Falls back to Gap 7b (skip + warn) on parse failure.

Concurrency / test-isolation notes
----------------------------------
The parser keeps two pieces of *module-level* state — `_PARSE_HISTORY`
(rolling failure-rate window) and `_last_parse_alert_at` (alert throttle).
Both are guarded by `_PARSE_LOCK` for thread safety inside a single
process, but they are NOT process-safe:

* Tests that exercise parse counters under `pytest-xdist` (`pytest -n auto`)
  will see counters from *one* worker only — the other workers each have
  their own copy. Tests that assert exact counts on `_PARSE_HISTORY` should
  either pin to a single worker (e.g. `--dist=loadfile` or `pytest -p no:xdist`)
  or call `reset_parse_history_for_tests()` in a fixture to start clean.
* The history is in-memory only — restart of the bot loses the window.
  This is intentional: a 1h window is short enough that warm-up is fast.

Timestamps stored in `_PARSE_HISTORY` use `time.monotonic()` (not
`datetime.now`) so a backward NTP step (clock skew) does not silently
prune live entries when the wall-clock jumps backward by minutes.
"""

from __future__ import annotations

import re
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

log = logging.getLogger(__name__)

# --- Parse-failure observability (P2) ---
# Sliding window of (monotonic_seconds, success_bool) tuples covering the
# last hour. Each call to parse_polymarket_title records its outcome here so
# the daily-summary job and live alerting can compute the failure rate
# without scanning logs. Bounded by both maxlen and time-window pruning.
#
# `time.monotonic()` is used instead of `datetime.now()` so a backward NTP
# clock jump cannot silently prune live entries (or hide a real failure
# spike behind a "future" cutoff). The monotonic counter is per-process and
# never resets while the process is alive.
_PARSE_WINDOW_SECONDS = 3600
_PARSE_HISTORY: deque[tuple[float, bool]] = deque(maxlen=10_000)
_PARSE_LOCK = Lock()
# Above this fraction of failures over the rolling window the parser logs a
# WARNING. Tuned for "title-drift event" detection — Polymarket usually
# fails < 5% (untradable cities + month false positives), so 30% is a clear
# anomaly without being so tight that a single bad scan triggers it.
_PARSE_FAILURE_ALERT_THRESHOLD = 0.30
# Don't alert until we've seen at least this many attempts in the window.
# Avoids spurious WARNINGs at startup. Lowered from 20 → 5 because real
# scrape cycles produce 5–15 titles, and the previous threshold meant a
# title-drift event in a small batch went undetected for ~3 cycles.
_PARSE_FAILURE_MIN_SAMPLES = 5
# Throttle the WARNING so we don't log it on every single parse once the
# threshold is crossed (we'd flood the log).
_PARSE_ALERT_COOLDOWN_SECONDS = 300
_last_parse_alert_at: float | None = None  # monotonic seconds; see above


def _record_parse_outcome(success: bool) -> None:
    """Append outcome to the rolling parse-failure window. Thread-safe."""
    now = time.monotonic()
    with _PARSE_LOCK:
        _PARSE_HISTORY.append((now, success))
        # Prune entries older than the window — the deque maxlen handles
        # absolute upper bound, but time-pruning keeps the rate accurate.
        cutoff = now - _PARSE_WINDOW_SECONDS
        while _PARSE_HISTORY and _PARSE_HISTORY[0][0] < cutoff:
            _PARSE_HISTORY.popleft()


def get_parse_failure_stats(window_minutes: int = 60) -> dict[str, float | int]:
    """Return (attempts, failures, failure_rate) over the last `window_minutes`.

    Used by the daily-summary job and ad-hoc diagnostics. Always safe to
    call — empty window returns zeros.
    """
    window_seconds = max(1, int(window_minutes) * 60)
    cutoff = time.monotonic() - window_seconds
    with _PARSE_LOCK:
        snapshot = [(ts, ok) for ts, ok in _PARSE_HISTORY if ts >= cutoff]
    attempts = len(snapshot)
    failures = sum(1 for _, ok in snapshot if not ok)
    rate = (failures / attempts) if attempts else 0.0
    return {"attempts": attempts, "failures": failures, "failure_rate": rate}


def _maybe_alert_parse_failure_rate() -> None:
    """Emit a single throttled WARNING when the rolling failure rate is high.

    The cooldown check AND the timestamp write happen atomically under
    `_PARSE_LOCK`, so concurrent parsers cannot both pass the cooldown gate
    and log duplicate warnings. Logging itself is moved outside the lock so
    a slow logging handler never serializes parser throughput.
    """
    global _last_parse_alert_at
    window_seconds = _PARSE_WINDOW_SECONDS
    cutoff_mono = time.monotonic() - window_seconds

    should_log = False
    snapshot_attempts = 0
    snapshot_failures = 0
    snapshot_rate = 0.0
    with _PARSE_LOCK:
        snapshot = [(ts, ok) for ts, ok in _PARSE_HISTORY if ts >= cutoff_mono]
        snapshot_attempts = len(snapshot)
        snapshot_failures = sum(1 for _, ok in snapshot if not ok)
        snapshot_rate = (snapshot_failures / snapshot_attempts) if snapshot_attempts else 0.0
        if (
            snapshot_attempts >= _PARSE_FAILURE_MIN_SAMPLES
            and snapshot_rate >= _PARSE_FAILURE_ALERT_THRESHOLD
        ):
            now_mono = time.monotonic()
            if (
                _last_parse_alert_at is None
                or (now_mono - _last_parse_alert_at) >= _PARSE_ALERT_COOLDOWN_SECONDS
            ):
                _last_parse_alert_at = now_mono
                should_log = True

    if should_log:
        log.warning(
            "PARSE_FAILURE_RATE_HIGH: %.1f%% (%d/%d) over last %dm — title drift?",
            snapshot_rate * 100,
            snapshot_failures,
            snapshot_attempts,
            window_seconds // 60,
        )


def reset_parse_history_for_tests() -> None:
    """Test helper — clears the rolling window and the alert cooldown."""
    global _last_parse_alert_at
    with _PARSE_LOCK:
        _PARSE_HISTORY.clear()
        _last_parse_alert_at = None

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

# Cities that appear in Polymarket weather universe but for which we have no
# edge infrastructure (no STATION_RMSE, no METAR timezone, no HRRR coverage).
# Suppress UNKNOWN_CITY_HIGH_PRIORITY warning for these — they're known-skip,
# not real "unmapped" surprises. Promote to CITY_TO_STATION + add an RMSE row
# in bias_correction.STATION_RMSE before removing from this list.
KNOWN_UNTRADABLE_CITIES: set[str] = {
    "cape town", "lagos", "helsinki", "amsterdam", "jeddah",
    "karachi", "jakarta", "manila", "kuala lumpur", "guangzhou",
    "busan", "panama city",
}

# Month names that the "in {City}" extraction regex falsely matches from
# fragments like "...on April 4". Filter to avoid log spam.
_MONTH_FALSE_POSITIVES: set[str] = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
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

    Returns None if parsing fails. Also records the outcome in the rolling
    failure-rate window so the daily summary can flag title drift.
    """
    parsed = _parse_polymarket_title_inner(title, current_year=current_year)
    _record_parse_outcome(success=parsed is not None)
    if parsed is None:
        _maybe_alert_parse_failure_rate()
    return parsed


def parse_kalshi_temperature_title(title: str, current_year: int | None = None) -> ParsedMarket | None:
    """Parse Kalshi KXHIGH-style weather titles.

    Kalshi bucket tickers such as ``B72.5`` are displayed as integer ranges
    ("72-73°"). Model them as half-degree settlement bands, e.g. [71.5, 73.5),
    instead of as a one-sided "below 72.5" threshold.
    """
    if current_year is None:
        current_year = datetime.now().year

    title_lower = title.lower()

    city = None
    station = None
    for city_name, station_code in sorted(CITY_TO_STATION.items(), key=lambda x: -len(x[0])):
        if re.search(r'\b' + re.escape(city_name) + r'\b', title_lower):
            city = city_name
            station = station_code
            break
    if not city or not station:
        return None

    month_map = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    month_pat = "|".join(sorted(month_map, key=len, reverse=True))
    date_match = re.search(
        rf'\b({month_pat})\.?\s+(\d{{1,2}})(?:,\s*(\d{{4}}))?',
        title_lower,
    )
    if not date_match:
        return None
    month = month_map[date_match.group(1)]
    day = int(date_match.group(2))
    year = int(date_match.group(3)) if date_match.group(3) else current_year
    date_str = f"{year:04d}-{month:02d}-{day:02d}"

    market_type = "daily_low" if re.search(r'\blow\s+temp|\blow(?:est)?\s+temperature\b', title_lower) else "daily_high"

    bucket_match = re.search(
        r'be\s+(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*°',
        title_lower,
    )
    if bucket_match:
        low_display = float(bucket_match.group(1))
        high_display = float(bucket_match.group(2))
        return ParsedMarket(
            city=city,
            station=station,
            date=date_str,
            market_type=market_type,
            threshold_low=low_display - 0.5,
            threshold_high=high_display + 0.5,
            direction="above",
            source="parsed_from_kalshi_title",
        )

    above_match = re.search(r'(?:>|above|higher than)\s*(-?\d+(?:\.\d+)?)\s*°?', title_lower)
    if above_match:
        return ParsedMarket(
            city=city,
            station=station,
            date=date_str,
            market_type=market_type,
            threshold=float(above_match.group(1)),
            direction="above",
            source="parsed_from_kalshi_title",
        )

    below_match = re.search(r'(-?\d+(?:\.\d+)?)\s*°?\s*(?:f)?\s+or\s+(?:below|lower)', title_lower)
    if below_match:
        return ParsedMarket(
            city=city,
            station=station,
            date=date_str,
            market_type=market_type,
            threshold=float(below_match.group(1)),
            direction="below",
            source="parsed_from_kalshi_title",
        )

    return None


def _parse_polymarket_title_inner(title: str, current_year: int | None = None) -> ParsedMarket | None:
    """Pure parsing logic — no observability side-effects. See public wrapper."""
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
        unknown_lower = unknown_city.lower()
        if unknown_lower in KNOWN_UNTRADABLE_CITIES or unknown_lower in _MONTH_FALSE_POSITIVES:
            log.debug(
                "Skipping non-tradable city '%s' (no edge infrastructure)",
                unknown_city,
            )
        else:
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
    # Note: thresholds use (-?\d+) so winter markets like "between -5-0°F"
    # parse correctly. Greedy match keeps "5-10" → 5,10 and "-5-10" → -5,10.

    # Pattern 1: "between X-Y°F" (bucket market)
    bucket_match = re.search(r'between\s+(-?\d+)\s*-\s*(-?\d+)\s*°?\s*f', title_lower)
    if bucket_match:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold_low=float(bucket_match.group(1)),
            threshold_high=float(bucket_match.group(2)),
            direction="above",
        )

    # Pattern 2: "X°F or above" / "X°F or higher" (lower bound)
    above_match = re.search(r'(-?\d+)\s*°?\s*f\s+or\s+(?:above|higher)', title_lower)
    if above_match:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(above_match.group(1)),
            direction="above",
        )

    # Pattern 3: "X°F or below" / "X°F or lower" (upper bound)
    below_match = re.search(r'(-?\d+)\s*°?\s*f\s+or\s+(?:below|lower)', title_lower)
    if below_match:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(below_match.group(1)),
            direction="below",
        )

    # Pattern 4: "exceed X°F"
    exceed_match = re.search(r'exceed\s+(-?\d+)\s*°?\s*f', title_lower)
    if exceed_match:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(exceed_match.group(1)),
            direction="above",
        )

    # Pattern 5: "be X°F" (exact or simple threshold) - less common
    be_match = re.search(r'be\s+(-?\d+)\s*°?\s*f', title_lower)
    if be_match:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(be_match.group(1)),
            direction="above",
        )

    # --- °C patterns (international markets) ---

    # Pattern 1C: "between X-Y°C" (bucket)
    bucket_c = re.search(r'between\s+(-?\d+)\s*-\s*(-?\d+)\s*°?\s*c\b', title_lower)
    if bucket_c:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold_low=float(bucket_c.group(1)),
            threshold_high=float(bucket_c.group(2)),
            direction="above", unit="C",
        )

    # Pattern 2C: "X°C or above/higher"
    above_c = re.search(r'(-?\d+)\s*°?\s*c\s+or\s+(?:above|higher)', title_lower)
    if above_c:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(above_c.group(1)),
            direction="above", unit="C",
        )

    # Pattern 3C: "X°C or below/lower"
    below_c = re.search(r'(-?\d+)\s*°?\s*c\s+or\s+(?:below|lower)', title_lower)
    if below_c:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(below_c.group(1)),
            direction="below", unit="C",
        )

    # Pattern 4C: "exceed X°C"
    exceed_c = re.search(r'exceed\s+(-?\d+)\s*°?\s*c\b', title_lower)
    if exceed_c:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(exceed_c.group(1)),
            direction="above", unit="C",
        )

    # Pattern 5C: "be X°C" (exact/simple)
    be_c = re.search(r'be\s+(-?\d+)\s*°?\s*c\b', title_lower)
    if be_c:
        return ParsedMarket(
            city=city, station=station, date=date_str, market_type=market_type,
            threshold=float(be_c.group(1)),
            direction="above", unit="C",
        )

    return None
