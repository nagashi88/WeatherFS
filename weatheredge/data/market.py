"""Market scanner for WeatherEdge via pmxt.

Scans Polymarket and Kalshi for weather markets, maps them to known resolution
sources, and stores price snapshots to the database.

Resolution source mapping from PRD Section 3.2.2.
Unknown market handling per Addendum Gap 7b.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pmxt

from weatheredge.backtesting.market_prices import _extract_from_kalshi_ticker
from weatheredge.config import Settings
from weatheredge.data.market_parser import (
    ParsedMarket,
    CITY_TO_STATION,
    parse_kalshi_temperature_title,
    parse_polymarket_title,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


def _kalshi_bucket_bounds_from_ticker(market_id: str) -> tuple[float, float] | None:
    """Return half-degree bucket bounds for KXHIGH ``Bxx.5`` bucket contracts."""
    parts = str(market_id).upper().split("-")
    if len(parts) < 3:
        return None
    tail = parts[-1]
    if not tail.startswith("B"):
        return None
    try:
        center = float(tail[1:])
    except ValueError:
        return None
    return center - 1.0, center + 1.0


def _title_is_one_sided_below(title: object) -> bool:
    raw = str(title or "").lower()
    return bool(re.search(r'<\s*-?\d|\bor\s+(?:below|lower)\b|\bbelow\b|\blower\b|\bless\s+than\b|\bunder\b', raw))

# ---------------------------------------------------------------------------
# Resolution source mapping — PRD Section 3.2.2
# THIS IS THE MOST CRITICAL TABLE IN THE ENTIRE SYSTEM.
# Every market resolves against a specific data source.
# ---------------------------------------------------------------------------

RESOLUTION_SOURCES: dict[str, dict[str, dict]] = {
    # KALSHI (all resolve against NWS Daily Climate Report)
    "kalshi": {
        "KXHIGHNY": {
            "station": "KNYC",  # Central Park
            "type": "daily_high",
            "source": "NWS CLI",
            "timezone": "EST",  # LST, not DST!
            "report_time": "~7:00 AM next day",
        },
        "KXHIGHCHI": {
            "station": "KMDW",  # Midway Airport
            "type": "daily_high",
            "source": "NWS CLI",
            "timezone": "CST",
            "report_time": "~7:00 AM next day",
        },
        "KXHIGHMIA": {
            "station": "KMIA",  # Miami International Airport
            "type": "daily_high",
            "source": "NWS CLI",
            "timezone": "EST",
            "report_time": "~7:00 AM next day",
        },
        "KXHIGHLAX": {
            "station": "KLAX",  # LAX
            "type": "daily_high",
            "source": "NWS CLI",
            "timezone": "PST",
            "report_time": "~7:00 AM next day",
        },
        "KXHIGHDEN": {
            "station": "KDEN",  # Denver International
            "type": "daily_high",
            "source": "NWS CLI",
            "timezone": "MST",
            "report_time": "~7:00 AM next day",
        },
    },
    # POLYMARKET (varies by market — ALWAYS check market rules)
    "polymarket": {
        "london_temp": {
            "station": "EGLC",  # London City Airport
            "type": "daily_high",
            "source": "Weather Underground",
            "timezone": "GMT/BST",
        },
        "nyc_precipitation": {
            "station": "Central Park",
            "type": "monthly_total_precipitation",
            "source": "NOAA",
            "precision": "0.01 inch",
        },
        # NOTE: Polymarket resolution sources vary by market.
        # The bot MUST parse each market's rules before trading.
    },
}

_STATION_TO_CITY: dict[str, str] = {
    "KNYC": "nyc",
    "KMDW": "chicago",
    "KMIA": "miami",
    "KDEN": "denver",
    "KLAX": "los angeles",
    "EGLC": "london",
    "KAUS": "austin",
    "KSEA": "seattle",
    "KATL": "atlanta",
    "KDFW": "dallas",
    "KPHX": "phoenix",
    "KIAH": "houston",
    "KSFO": "san francisco",
    "KBOS": "boston",
    "RJTT": "tokyo",
    "UUEE": "moscow",
    "LIML": "milan",
    "LIRF": "rome",
    "LEMD": "madrid",
    "LTFM": "istanbul",
    "WSSS": "singapore",
    "EDDM": "munich",
    "RCTP": "taipei",
    "LFPG": "paris",
    "CYYZ": "toronto",
    "ZSPD": "shanghai",
    "ZBAA": "beijing",
    "NZWN": "wellington",
    "EPWA": "warsaw",
    "LLBG": "tel aviv",
    "ZUUU": "chengdu",
    "ZUCK": "chongqing",
    "ZHHH": "wuhan",
    "ZGSZ": "shenzhen",
    "VILK": "lucknow",
    "SBGR": "sao paulo",
    "VHHH": "hong kong",
    "MMMX": "mexico city",
    "FACT": "cape town",
    "DNMM": "lagos",
    "EFHK": "helsinki",
    "EHAM": "amsterdam",
    "OEJN": "jeddah",
    "OPKC": "karachi",
    "WIII": "jakarta",
    "RPLL": "manila",
    "WMKK": "kuala lumpur",
    "ZGGG": "guangzhou",
    "RKPK": "busan",
    "MPTO": "panama city",
    "ZSQD": "qingdao",
}


def _outcome_price(market: object, side: str) -> object | None:
    outcome = getattr(market, side, None)
    return getattr(outcome, "price", None) if outcome is not None else None


def _outcome_id(market: object, side: str) -> str | None:
    outcome = getattr(market, side, None)
    value = getattr(outcome, "outcome_id", None) if outcome is not None else None
    return str(value) if value else None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MarketScanner:
    """Scans Polymarket and Kalshi for tradeable weather markets.

    Clients are created on-demand in scan methods, not in __init__, because
    credentials may not be available at construction time.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def _target_stations(cities: list[str]) -> set[str]:
        """Map configured target city labels to station codes."""
        stations: set[str] = set()
        for city in cities:
            raw = city.strip()
            if not raw:
                continue
            upper = raw.upper()
            station = CITY_TO_STATION.get(raw.lower())
            if station:
                stations.add(station)
                continue
            # Allow direct station-code configuration.
            if len(upper) == 4 and upper[0].isalpha():
                stations.add(upper)
        return stations

    # -----------------------------------------------------------------------
    # Resolution source lookup
    # -----------------------------------------------------------------------

    def get_resolution_source(self, platform: str, ticker: str) -> dict | None:
        """Look up resolution source for a market.

        Args:
            platform: 'kalshi' or 'polymarket'
            ticker: Market ticker / slug

        Returns:
            Resolution source dict, or None if the market is unknown.
        """
        return RESOLUTION_SOURCES.get(platform, {}).get(ticker)

    # -----------------------------------------------------------------------
    # Market processing — Addendum Gap 7b
    # -----------------------------------------------------------------------

    def process_market(self, market_data: dict, platform: str) -> dict | None:
        """Validate market has a known resolution source before proceeding.

        Per Addendum Gap 7b: unknown markets are skipped with a warning log.
        The bot must NOT trade unknown markets.

        Args:
            market_data: Raw market dict with at least 'ticker' and 'title' keys.
            platform: 'kalshi' or 'polymarket'

        Returns:
            Enriched market dict with 'resolution_source' key added, or None
            if the market has no known resolution source.
        """
        ticker = market_data.get("ticker", "")
        resolution = self.get_resolution_source(platform, ticker)

        # Kalshi KXHIGH tickers identify the city/date family, but bucket
        # contracts need the title to distinguish ranges ("72-73°") from
        # one-sided thresholds. Attach parsed metadata whenever possible.
        if platform == "kalshi":
            parsed = parse_kalshi_temperature_title(market_data.get("title", ""))
            if parsed:
                market_data["parsed"] = parsed

        # If hardcoded lookup fails for Polymarket, try title parsing
        if resolution is None and platform == "polymarket":
            title = market_data.get("title", "")
            parsed = parse_polymarket_title(title)
            if parsed:
                resolution = {
                    "station": parsed.station,
                    "type": parsed.market_type,
                    "source": "parsed_from_title",
                    "timezone": "EST",  # Will be refined per station
                }
                market_data["parsed"] = parsed  # Attach parsed data
                log.info(
                    "Parsed Polymarket market: %s → %s/%s %s",
                    title[:60],
                    parsed.station,
                    parsed.date,
                    f"{parsed.threshold_low}-{parsed.threshold_high}"
                    if parsed.threshold_low
                    else f"{parsed.threshold}",
                )

        if resolution is None:
            # Unknown market — skip and alert (Addendum Gap 7b)
            title = market_data.get("title", "")
            log.warning(
                "Unknown market: %s/%s - %s",
                platform,
                ticker,
                title,
            )
            # Fire Telegram alert for unknown city (HIGH PRIORITY)
            self._alert_unknown_market(platform, ticker, title)
            return None

        return {
            **market_data,
            "platform": platform,
            "resolution_source": resolution,
        }

    # -----------------------------------------------------------------------
    # Unknown market alerting (Gap 7b + Telegram)
    # -----------------------------------------------------------------------

    def _alert_unknown_market(self, platform: str, ticker: str, title: str) -> None:
        """Send a HIGH PRIORITY Telegram alert for an unmatched market."""
        if not self.settings.telegram_bot_token or not self.settings.telegram_chat_id:
            # Keep logs readable when alerts are not configured.
            return
        try:
            from weatheredge.notifications.telegram import TelegramBot
            bot = TelegramBot(self.settings)
            import asyncio
            msg = (
                f"🚨 UNKNOWN MARKET (HIGH PRIORITY)\n"
                f"Platform: {platform}\n"
                f"Ticker: {ticker}\n"
                f"Title: {title[:120]}\n\n"
                f"Action: Add city mapping to CITY_TO_STATION in market_parser.py"
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(bot.send_alert(msg))
            except RuntimeError:
                asyncio.run(bot.send_alert(msg))
        except Exception as exc:
            log.debug("Failed to send unknown-market Telegram alert: %s", exc)

    # -----------------------------------------------------------------------
    # Polymarket scan
    # -----------------------------------------------------------------------

    async def scan_polymarket(self) -> list[dict]:
        """Fetch weather events from Polymarket and return known markets.

        Uses pmxt.Polymarket. Connection errors are caught and logged so the
        scan loop continues in degraded mode (Addendum Gap 12).

        Returns:
            List of processed market dicts for markets with known resolution sources.
        """
        try:
            client = pmxt.Polymarket(
                private_key=self.settings.polymarket_private_key or None,
                proxy_address=self.settings.polymarket_proxy_address or None,
            )
            events = client.fetch_events(params={"query": "temperature"})
        except Exception as exc:
            log.warning("Polymarket scan failed: %s", exc)
            return []

        results: list[dict] = []
        orderbook_candidates: list[tuple[dict, object, float]] = []
        configured_cities = [c.strip() for c in self.settings.tracked_cities_list if c.strip()]
        target_stations = self._target_stations(configured_cities)
        orderbook_budget = int(getattr(self.settings, "polymarket_orderbook_max_markets_per_scan", 120))
        orderbook_min_volume = float(getattr(self.settings, "polymarket_orderbook_min_volume", 2000.0))
        adaptive_orderbook_min_volume = orderbook_min_volume
        if (
            bool(getattr(self.settings, "shadow_twin_enabled", False))
            and bool(getattr(self.settings, "shadow_twin_adaptive_liquidity_enabled", False))
        ):
            adaptive_orderbook_min_volume = min(
                orderbook_min_volume,
                float(getattr(self.settings, "shadow_twin_adaptive_liquidity_min_volume", orderbook_min_volume)),
            )
        orderbook_enabled = bool(getattr(self.settings, "polymarket_orderbook_quotes_enabled", True))
        if configured_cities and not target_stations:
            log.warning(
                "No target stations resolved from target_cities=%s — skipping Polymarket scan",
                configured_cities,
            )
            return []

        for event in events:
            for market in event.markets:
                market_data = {
                    "ticker": market.market_id,
                    "title": market.title,
                    "event_id": event.id,
                    "event_title": event.title,
                    "yes_price": _outcome_price(market, "yes"),
                    "no_price": _outcome_price(market, "no"),
                    "volume_24h": market.volume_24h,
                    "liquidity": market.liquidity,
                    "resolution_date": market.resolution_date,
                }
                processed = self.process_market(market_data, "polymarket")
                if processed is not None:
                    resolution_source = processed.get("resolution_source", {})
                    station = resolution_source.get("station") if isinstance(resolution_source, dict) else None
                    if target_stations and station not in target_stations:
                        continue
                    volume = _float_or_none(processed.get("volume_24h"))
                    if (
                        orderbook_enabled
                        and volume is not None
                        and volume >= adaptive_orderbook_min_volume
                    ):
                        orderbook_candidates.append((processed, market, volume))
                    results.append(processed)

        if orderbook_enabled and orderbook_budget > 0 and orderbook_candidates:
            main_volume_candidates = [
                item for item in orderbook_candidates if item[2] >= orderbook_min_volume
            ]
            adaptive_volume_candidates = [
                item for item in orderbook_candidates if item[2] < orderbook_min_volume
            ]
            main_volume_candidates.sort(key=lambda item: item[2], reverse=True)
            adaptive_volume_candidates.sort(key=lambda item: item[2], reverse=True)
            for processed, market, _volume in main_volume_candidates + adaptive_volume_candidates:
                if orderbook_budget <= 0:
                    break
                yes_outcome_id = _outcome_id(market, "yes")
                if not yes_outcome_id:
                    continue
                orderbook_budget -= 1
                self._attach_orderbook_quote(client, processed, yes_outcome_id)

        return results

    def _attach_orderbook_quote(
        self,
        client: object,
        market_data: dict,
        yes_outcome_id: str,
    ) -> None:
        """Attach real YES-side order-book quote data when Polymarket exposes it."""
        try:
            book = client.fetch_order_book(yes_outcome_id)
        except Exception as exc:
            log.debug("Polymarket orderbook fetch failed for %s: %s", yes_outcome_id, exc)
            return

        bid_levels = [
            (price, _float_or_none(getattr(level, "size", None)))
            for level in getattr(book, "bids", [])
            if (price := _float_or_none(getattr(level, "price", None))) is not None
        ]
        ask_levels = [
            (price, _float_or_none(getattr(level, "size", None)))
            for level in getattr(book, "asks", [])
            if (price := _float_or_none(getattr(level, "price", None))) is not None
        ]
        best_bid_level = max(bid_levels, key=lambda level: level[0]) if bid_levels else None
        best_ask_level = min(ask_levels, key=lambda level: level[0]) if ask_levels else None
        best_bid = best_bid_level[0] if best_bid_level else None
        best_ask = best_ask_level[0] if best_ask_level else None
        if best_bid_level is not None:
            market_data["best_bid"] = best_bid_level[0]
            market_data["bid_size"] = best_bid_level[1]
        if best_ask_level is not None:
            market_data["best_ask"] = best_ask_level[0]
            market_data["ask_size"] = best_ask_level[1]
        if best_bid is not None and best_ask is not None and best_ask >= best_bid:
            market_data["spread"] = best_ask - best_bid

    # -----------------------------------------------------------------------
    # Kalshi scan
    # -----------------------------------------------------------------------

    async def scan_kalshi(self) -> list[dict]:
        """Fetch KXHIGH weather events from Kalshi and return known markets.

        Tries pmxt first; if pmxt returns 0 results, falls back to the direct
        Kalshi public API (no auth required). Connection errors are caught and
        logged so the scan loop continues in degraded mode (Addendum Gap 12).

        Returns:
            List of processed market dicts for markets with known resolution sources.
        """
        results = await self._scan_kalshi_pmxt()

        # If pmxt returned nothing, fall back to direct API
        if not results:
            log.info("pmxt returned 0 Kalshi markets, trying direct public API")
            results = await self._scan_kalshi_direct()

        return results

    async def _scan_kalshi_pmxt(self) -> list[dict]:
        """Original pmxt-based Kalshi scan."""
        try:
            client = pmxt.Kalshi(
                api_key=self.settings.kalshi_api_key or None,
                private_key=self.settings.kalshi_private_key or None,
            )
            events = client.fetch_events(params={"query": "KXHIGH"})
        except Exception as exc:
            log.warning("Kalshi scan failed: %s", exc)
            return []

        results: list[dict] = []

        for event in events:
            for market in event.markets:
                # Extract KXHIGH ticker prefix from market_id
                ticker = market.market_id.split("-")[0] if "-" in market.market_id else market.market_id
                market_data = {
                    "ticker": ticker,
                    "market_id": market.market_id,
                    "title": market.title,
                    "event_id": event.id,
                    "event_title": event.title,
                    "yes_price": _outcome_price(market, "yes"),
                    "no_price": _outcome_price(market, "no"),
                    "volume_24h": market.volume_24h,
                    "liquidity": market.liquidity,
                    "resolution_date": market.resolution_date,
                }
                processed = self.process_market(market_data, "kalshi")
                if processed is not None:
                    results.append(processed)

        return results

    async def _scan_kalshi_direct(self) -> list[dict]:
        """Fallback: direct Kalshi public API (no auth required)."""
        from weatheredge.data.kalshi_client import fetch_kalshi_weather_markets

        try:
            raw_markets = await fetch_kalshi_weather_markets()
        except Exception as exc:
            log.warning("Kalshi direct API failed: %s", exc)
            return []

        results = []
        for market_data in raw_markets:
            processed = self.process_market(market_data, "kalshi")
            if processed is not None:
                results.append(processed)
        return results

    # -----------------------------------------------------------------------
    # Database write
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_iso_date(value: object) -> str | None:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        if len(raw) >= 10:
            candidate = raw[:10]
            try:
                datetime.strptime(candidate, "%Y-%m-%d")
                return candidate
            except ValueError:
                pass
        normalized = raw.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).date().isoformat()
        except ValueError:
            return None

    @staticmethod
    def _json_text(value: dict) -> str:
        return json.dumps(value, sort_keys=True, ensure_ascii=True)

    def store_market_metadata(
        self,
        db_session: Session,
        market_id: str,
        market_data: dict,
    ) -> None:
        """Upsert market metadata so price snapshots are backtest-usable later."""
        from weatheredge.db.models import Market

        parsed = market_data.get("parsed")
        resolution_source = market_data.get("resolution_source", {})
        ticker_or_market = str(market_data.get("market_id") or market_data.get("ticker") or market_id)

        station = None
        city = None
        market_date = None
        market_type = None
        threshold_low = None
        threshold_high = None
        direction = None
        threshold = None

        if isinstance(parsed, ParsedMarket):
            station = parsed.station
            city = parsed.city
            market_date = parsed.date
            market_type = parsed.market_type
            threshold_low = parsed.threshold_low
            threshold_high = parsed.threshold_high
            direction = parsed.direction
            threshold = parsed.threshold

        ticker_station, ticker_date, ticker_threshold, ticker_direction = _extract_from_kalshi_ticker(
            ticker_or_market
        )
        ticker_bucket_bounds = None
        title_is_below = _title_is_one_sided_below(market_data.get("title"))
        if not isinstance(parsed, ParsedMarket) and not title_is_below:
            ticker_bucket_bounds = _kalshi_bucket_bounds_from_ticker(ticker_or_market)
        if station is None:
            station = ticker_station
        if market_date is None and ticker_date is not None:
            market_date = ticker_date.isoformat()
        if ticker_bucket_bounds is not None and threshold is None and threshold_low is None and threshold_high is None:
            threshold_low, threshold_high = ticker_bucket_bounds
            threshold = (threshold_low + threshold_high) / 2.0
            direction = "above"
        elif ticker_threshold is not None and threshold is None and threshold_low is None and threshold_high is None:
            threshold = ticker_threshold
            if title_is_below:
                direction = "below"
        if direction is None:
            direction = ticker_direction

        if isinstance(resolution_source, dict):
            if station is None:
                station = resolution_source.get("station")
            if market_type is None:
                market_type = resolution_source.get("type")

        if market_date is None:
            market_date = (
                self._parse_iso_date(market_data.get("resolution_date"))
                or self._parse_iso_date(market_data.get("close_time"))
                or datetime.now(timezone.utc).date().isoformat()
            )

        if market_type is None:
            market_type = "daily_high"
        if station is None:
            station = "UNKNOWN"
        if city is None:
            city = _STATION_TO_CITY.get(station, station.lower())

        if threshold is not None and threshold_low is None and threshold_high is None:
            threshold_high = float(threshold)

        metadata_payload = {
            "direction": direction,
            "threshold": threshold,
            "ticker": market_data.get("ticker"),
            "title": market_data.get("title"),
            "source": resolution_source.get("source") if isinstance(resolution_source, dict) else None,
        }
        if isinstance(resolution_source, dict):
            for key in ("timezone", "report_time", "precision"):
                if key in resolution_source:
                    metadata_payload[key] = resolution_source.get(key)

        existing = db_session.get(Market, market_id)
        if existing is None:
            record = Market(
                id=market_id,
                platform=str(market_data.get("platform") or "unknown"),
                city=str(city),
                station=str(station),
                date=str(market_date),
                market_type=str(market_type),
                threshold_low=float(threshold_low) if threshold_low is not None else None,
                threshold_high=float(threshold_high) if threshold_high is not None else None,
                resolution_source=self._json_text(metadata_payload),
            )
            db_session.add(record)
            return

        existing.platform = str(market_data.get("platform") or existing.platform or "unknown")
        existing.city = str(city)
        existing.station = str(station)
        existing.date = str(market_date)
        existing.market_type = str(market_type)
        existing.threshold_low = float(threshold_low) if threshold_low is not None else None
        existing.threshold_high = float(threshold_high) if threshold_high is not None else None
        existing.resolution_source = self._json_text(metadata_payload)

    def store_market_price(
        self,
        db_session: Session,
        market_id: str,
        price_data: dict,
    ) -> None:
        """Write a price snapshot to the market_prices table.

        BugFix M8: Removed async — this method has no await calls,
        and wrapping in asyncio.run() per-market wastes ~10ms each.

        Args:
            db_session: Active SQLAlchemy session.
            market_id: Foreign key referencing markets.id.
            price_data: Dict with optional keys: yes_price, no_price, volume, spread.
        """
        from weatheredge.db.models import MarketPrice

        from weatheredge.config import MODULES

        _log_ob = MODULES.get("orderbook_logging", False)
        record = MarketPrice(
            market_id=market_id,
            timestamp=datetime.now(timezone.utc),
            yes_price=price_data.get("yes_price"),
            no_price=price_data.get("no_price"),
            volume=price_data.get("volume"),
            spread=price_data.get("spread"),
            best_bid=price_data.get("best_bid") if _log_ob else None,
            best_ask=price_data.get("best_ask") if _log_ob else None,
            bid_size=price_data.get("bid_size") if _log_ob else None,
            ask_size=price_data.get("ask_size") if _log_ob else None,
        )
        db_session.add(record)
        log.debug("Stored price snapshot for market %s", market_id)
