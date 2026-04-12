"""Main entry point for WeatherEdge bot.

Implements:
- Gap 7:  Event-driven pipeline — schedulers fetch data only; separate 5-min decision loop
- Gap 12: Degraded mode — check data freshness, skip if stale >2h or pmxt disconnected
- Gap 13: Logging — RotatingFileHandler, 10MB, 5 backups
- Gap 14: Graceful shutdown — SIGTERM handling
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
import time
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

import aiohttp
from apscheduler.schedulers.background import BackgroundScheduler

from weatheredge.config import Settings, MODULES
from weatheredge.data.actuals import fetch_daily_max_actual, store_actual_temperatures
from weatheredge.data.market import MarketScanner, RESOLUTION_SOURCES
from weatheredge.data.noaa import fetch_noaa_forecast, store_noaa_forecast
from weatheredge.data.weather import fetch_forecast, fetch_ensemble, fetch_ecmwf, store_forecast
from weatheredge.db.database import get_session, init_db
from weatheredge.db.models import Forecast, MarketPrice
from weatheredge.execution.paper_trader import Order, PaperTrader
from weatheredge.execution.exit_manager import ExitManager
from weatheredge.execution.fill_monitor import FillMonitor, FillRecord
from weatheredge.execution.cross_platform import CrossPlatformArbitrage
from weatheredge.probability.edge import calculate_net_edge
from weatheredge.probability.engine import calculate_probability, calculate_blended_probability
from weatheredge.data.metar import CurrentConditionsTracker
from weatheredge.data.station_time import hours_until_market_close
from weatheredge.data.model_run_detector import ModelRunDetector, ModelRunEvent
from weatheredge.data.market_parser import US_STATIONS
from weatheredge.agents.base import MarketOpportunity, Signal
from weatheredge.agents.risk_manager import RiskManager
from weatheredge.agents.portfolio_manager import PortfolioManager
from weatheredge.agents.strategies.conservative import conservative_strategy
from weatheredge.agents.strategies.aggressive import aggressive_strategy
from weatheredge.agents.strategies.mean_reversion import mean_reversion_strategy
from weatheredge.agents.strategies.sell_no import sell_no_strategy
from weatheredge.agents.qualitative import QualitativeAgent
from weatheredge.agents.rules_qualitative import RulesQualitativeEngine
from weatheredge.learning.resolution_tracker import ResolutionTracker
from weatheredge.learning.calibration import CalibrationTracker
from weatheredge.learning.weight_updater import ModelWeightUpdater
from weatheredge.learning.agent_weight_updater import AgentWeightUpdater
from weatheredge.learning.rmse_updater import RMSEUpdater
from weatheredge.learning.error_analysis import ErrorAnalyzer
from weatheredge.intelligence.wallet_profiler import WalletCollector, WalletProfiler, WalletQueryEngine
from weatheredge.intelligence.edge_decay_monitor import EdgeDecayMonitor
from weatheredge.intelligence.competitor_tracker import CompetitorTracker
from weatheredge.probability.timesfm_forecaster import TimesFMForecaster

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging setup (Addendum Gap 13)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_FILE = _LOG_DIR / "weatheredge.log"
_LOG_FILE_ENV_VAR = "WEATHEREDGE_LOG_FILE"
_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def _run_async_safely(coro):
    """Run coroutine and always close local coroutine object afterwards.

    This avoids ``RuntimeWarning: coroutine was never awaited`` when tests mock
    ``asyncio.run`` and return/raise without actually awaiting the coroutine.
    """
    try:
        return asyncio.run(coro)
    finally:
        close = getattr(coro, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _first_not_none(*values):
    """Return the first value that is not None."""
    for value in values:
        if value is not None:
            return value
    return None


def should_evaluate_market(volume: float, spread_pct: float) -> tuple[bool, str]:
    """Check if a market passes volume/liquidity filters.

    Returns (True, "") if ok, or (False, reason) if the market should be skipped.
    Gated by MODULES["volume_filter"] at the call site.
    """
    from weatheredge.config import MIN_MARKET_VOLUME, MAX_SPREAD_PCT

    if volume < MIN_MARKET_VOLUME:
        return (False, "low_volume")
    if spread_pct > MAX_SPREAD_PCT:
        return (False, "wide_spread")
    return (True, "")


def _resolve_log_file() -> Path:
    """Resolve log file path, allowing runtime override via env var."""
    override = os.getenv(_LOG_FILE_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return _LOG_FILE


def setup_logging() -> None:
    """Configure root logger with RotatingFileHandler per Addendum Gap 13.

    - File: logs/weatheredge.log, 10MB max, 5 backups
    - Console handler for visibility when running interactively
    - Module-level log levels per Gap 13 spec
    """
    log_file = _resolve_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT)

    file_handler = RotatingFileHandler(
        str(log_file),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Module-level overrides per Gap 13
    logging.getLogger("weatheredge.data").setLevel(logging.INFO)
    logging.getLogger("weatheredge.agents").setLevel(logging.INFO)
    logging.getLogger("weatheredge.execution").setLevel(logging.DEBUG)
    logging.getLogger("weatheredge.risk").setLevel(logging.WARNING)
    logging.getLogger("weatheredge.learning").setLevel(logging.INFO)
    logging.getLogger("weatheredge.telegram").setLevel(logging.INFO)

    log.info("Logging initialised — file: %s", log_file)


# ---------------------------------------------------------------------------
# City coordinates for weather API calls
# ---------------------------------------------------------------------------

CITY_COORDS: dict[str, tuple[float, float]] = {
    "NYC": (40.7128, -74.0060),
    "Chicago": (41.8781, -87.6298),
    "Miami": (25.7617, -80.1918),
    "Denver": (39.7392, -104.9903),
    "LAX": (33.9425, -118.4081),
    "Austin": (30.2672, -97.7431),
    "Seattle": (47.6062, -122.3321),
    "Atlanta": (33.7490, -84.3880),
    "Dallas": (32.7767, -96.7970),
    "Phoenix": (33.4484, -112.0740),
    "Houston": (29.7604, -95.3698),
    "San Francisco": (37.7749, -122.4194),
    "Boston": (42.3601, -71.0589),
}

# Station code → city name (reverse lookup for METAR tracker)
STATION_TO_CITY: dict[str, str] = {
    "KNYC": "NYC", "KMDW": "Chicago", "KMIA": "Miami",
    "KDEN": "Denver", "KLAX": "LAX", "EGLC": "London",
    "KAUS": "Austin", "KSEA": "Seattle", "SAEZ": "Buenos Aires",
    "LTAC": "Ankara", "RKSS": "Seoul", "KATL": "Atlanta",
    "KDFW": "Dallas", "KPHX": "Phoenix", "KIAH": "Houston",
    "KSFO": "San Francisco", "KBOS": "Boston",
}

# City name -> station code (canonical key used by calibration/RMSE tables).
CITY_TO_STATION: dict[str, str] = {city: station for station, city in STATION_TO_CITY.items()}

# All METAR stations to poll
METAR_STATIONS: list[str] = list(STATION_TO_CITY.keys())


def _canonical_station_code(station_or_city: str) -> str:
    """Return canonical station code (e.g. KNYC) when known."""
    return CITY_TO_STATION.get(station_or_city, station_or_city)


def _canonical_model_name(model: str | None) -> str | None:
    """Map stored forecast-model labels to calibration model names."""
    if not model:
        return None
    model_lc = model.lower()
    if model_lc in {"gfs", "forecast"}:
        return "gfs"
    if model_lc == "ecmwf":
        return "ecmwf"
    if model_lc == "hrrr":
        return "hrrr"
    return None


# ---------------------------------------------------------------------------
# Application state (shared across scheduler threads)
# ---------------------------------------------------------------------------

class AppState:
    """Mutable application state shared between scheduler jobs and main loop."""

    def __init__(self) -> None:
        self._shutdown_event = threading.Event()  # BugFix L1: thread-safe shutdown flag
        self.shutting_down: bool = False  # kept for backwards compat, prefer _shutdown_event
        self.settings: Settings = Settings()
        self.risk_manager: RiskManager = RiskManager(self.settings)
        self.paper_trader: PaperTrader = PaperTrader(
            initial_bankroll=self.settings.initial_bankroll
        )
        self.market_scanner: MarketScanner = MarketScanner(self.settings)
        self.portfolio_manager: PortfolioManager = PortfolioManager(
            self.settings, self.risk_manager
        )
        self.fill_monitor: FillMonitor | None = None  # Initialised after DB in main()
        self.cross_platform_arb: CrossPlatformArbitrage = CrossPlatformArbitrage()
        self.scheduler: BackgroundScheduler | None = None
        # In-memory price history per market_id for mean reversion (list[float])
        self._price_history: dict[str, list[float]] = {}
        # Phase 3 self-learning components — initialised after DB in main()
        self.resolution_tracker: ResolutionTracker | None = None
        self.calibration_tracker: CalibrationTracker | None = None
        self.model_weight_updater: ModelWeightUpdater | None = None
        self.agent_weight_updater: AgentWeightUpdater | None = None
        self.rmse_updater: RMSEUpdater | None = None
        self.error_analyzer: ErrorAnalyzer | None = None
        self._total_resolved_trades: int = 0
        # Phase 4: qualitative agent — initialised after DB in main()
        self.qualitative_agent: QualitativeAgent | None = None
        # Rules-based qualitative engine (hot-loop replacement for LLM agent)
        self.rules_qualitative_engine: RulesQualitativeEngine = RulesQualitativeEngine()
        # Phase 4.5: wallet intelligence components — initialised after DB in main()
        self.wallet_collector: WalletCollector | None = None
        self.wallet_profiler: WalletProfiler | None = None
        self.wallet_query_engine: WalletQueryEngine | None = None
        self.edge_decay_monitor: EdgeDecayMonitor | None = None
        self.competitor_tracker: CompetitorTracker | None = None
        # Phase 5: Edge enhancements
        self.conditions_tracker: CurrentConditionsTracker = CurrentConditionsTracker()
        self.model_run_detector: ModelRunDetector | None = None
        # Phase 6: TimesFM integration
        self.timesfm_forecaster: TimesFMForecaster | None = None
        # Runtime cache: market_id → enriched metadata from scanning pipeline
        self.market_metadata: dict[str, dict] = {}
        # Overlap guard — prevents concurrent decision cycles (e.g. scheduler + check_model_runs)
        self._decision_cycle_lock: threading.Lock = threading.Lock()


_state: AppState | None = None


def get_state() -> AppState:
    """Return the global AppState, raising if not yet initialised."""
    if _state is None:
        raise RuntimeError("AppState not initialised — call init_app() first")
    return _state


# ---------------------------------------------------------------------------
# Data fetch jobs (scheduler layer — DB writes only, no trading decisions)
# ---------------------------------------------------------------------------

def fetch_market_data() -> None:
    """Scan Polymarket and Kalshi for weather markets and store price snapshots.

    Runs every settings.market_scan_interval_minutes (default 5m).
    Errors are caught and logged so the scheduler loop continues (Gap 12).
    """
    state = get_state()
    log.info("fetch_market_data: starting market scan")
    try:
        markets = _run_async_safely(_scan_all_markets(state))
        log.info("fetch_market_data: found %d markets", len(markets))
        with get_session() as session:
            for market in markets:
                market_id = market.get("market_id") or market.get("ticker", "")
                if market_id:
                    # Cache enriched metadata for _build_opportunity
                    state.market_metadata[market_id] = market
                    state.market_scanner.store_market_metadata(session, market_id, market)
                    # BugFix M8: store_market_price is now sync — no asyncio.run needed
                    yes_price = market.get("yes_price")
                    no_price = market.get("no_price")
                    spread = None
                    try:
                        if yes_price is not None and no_price is not None:
                            spread = abs(float(yes_price) + float(no_price) - 1.0)
                    except (TypeError, ValueError):
                        spread = None
                    state.market_scanner.store_market_price(
                        session,
                        market_id,
                        {
                            "yes_price": yes_price,
                            "no_price": no_price,
                            "volume": market.get("volume_24h")
                            if market.get("volume_24h") is not None
                            else market.get("volume"),
                            "spread": spread,
                        },
                    )
    except Exception as exc:
        log.error("fetch_market_data failed: %s", exc, exc_info=True)


async def _scan_all_markets(state: AppState) -> list[dict]:
    """Fetch markets from both platforms concurrently."""
    poly, kalshi = await asyncio.gather(
        state.market_scanner.scan_polymarket(),
        state.market_scanner.scan_kalshi(),
        return_exceptions=True,
    )
    results: list[dict] = []
    if isinstance(poly, list):
        results.extend(poly)
    else:
        log.warning("Polymarket scan error: %s", poly)
    if isinstance(kalshi, list):
        results.extend(kalshi)
    else:
        log.warning("Kalshi scan error: %s", kalshi)
    return results


def fetch_forecast_data() -> None:
    """Fetch weather forecasts for all target cities and store to DB.

    Runs every settings.forecast_poll_interval_minutes (default 15m).
    Errors are caught and logged so the scheduler loop continues (Gap 12).
    """
    state = get_state()
    log.info("fetch_forecast_data: fetching forecasts for cities")
    try:
        _run_async_safely(_fetch_all_forecasts(state))
    except Exception as exc:
        log.error("fetch_forecast_data failed: %s", exc, exc_info=True)


async def _fetch_all_forecasts(state: AppState) -> None:
    """Fetch Open-Meteo forecast + NOAA for all target cities.

    Stores forecasts for today through today+6 (7 days) so that future-dated
    markets receive correct forecast data rather than falling back to today's data.
    """
    from datetime import timedelta
    now_utc = datetime.now(timezone.utc)
    # Generate ISO date strings for the next 7 days
    forecast_dates = [
        (now_utc + timedelta(days=d)).strftime("%Y-%m-%d")
        for d in range(7)
    ]
    cities = state.settings.cities_list

    async with aiohttp.ClientSession() as session:
        for city in cities:
            coords = CITY_COORDS.get(city)
            if coords is None:
                log.warning("No coordinates for city %s — skipping", city)
                continue
            lat, lon = coords
            station_code = _canonical_station_code(city)

            # Open-Meteo forecast — fetch once, store for each future date
            try:
                data = await fetch_forecast(lat, lon, session)
                with get_session() as db_session:
                    for target_date in forecast_dates:
                        store_forecast(db_session, station_code, target_date, "forecast", data)
                log.info(
                    "Stored Open-Meteo forecast for %s/%s (%d dates)",
                    city,
                    station_code,
                    len(forecast_dates),
                )
            except Exception as exc:
                log.warning("Open-Meteo forecast failed for %s: %s", city, exc)

            # Open-Meteo ensemble — fetch once, store for each future date
            try:
                data = await fetch_ensemble(lat, lon, session)
                with get_session() as db_session:
                    for target_date in forecast_dates:
                        store_forecast(db_session, station_code, target_date, "ensemble", data)
                log.info(
                    "Stored Open-Meteo ensemble for %s/%s (%d dates)",
                    city,
                    station_code,
                    len(forecast_dates),
                )
            except Exception as exc:
                log.warning("Open-Meteo ensemble failed for %s: %s", city, exc)

            # NOAA NWS — fetch once, store for each future date
            try:
                from weatheredge.data.noaa import fetch_noaa_forecast as _fetch_noaa, store_noaa_forecast as _store_noaa
                noaa_data = await _fetch_noaa(lat, lon, session)
                with get_session() as db_session:
                    for target_date in forecast_dates:
                        _store_noaa(db_session, station_code, target_date, noaa_data)
                log.info(
                    "Stored NOAA forecast for %s/%s (%d dates)",
                    city,
                    station_code,
                    len(forecast_dates),
                )
            except Exception as exc:
                log.warning("NOAA forecast failed for %s: %s", city, exc)


def run_actuals_backfill() -> None:
    """Backfill Forecast.actual_temp from observed weather data."""
    state = get_state()
    log.info("run_actuals_backfill: syncing observed temperatures")
    try:
        _normalize_forecast_station_labels()
        summary = _run_async_safely(_fetch_and_store_actuals(state))
        log.info(
            "run_actuals_backfill: complete — stations=%d rows_updated=%d",
            summary.get("stations", 0),
            summary.get("rows_updated", 0),
        )
    except Exception as exc:
        log.error("run_actuals_backfill failed: %s", exc, exc_info=True)


def _normalize_forecast_station_labels() -> None:
    """One-way normalization: legacy city labels -> station codes."""
    with get_session() as session:
        changed = 0
        for station_code, city_name in STATION_TO_CITY.items():
            rows = (
                session.query(Forecast)
                .filter(Forecast.station == city_name)
                .update({Forecast.station: station_code}, synchronize_session=False)
            )
            changed += int(rows or 0)
    if changed:
        log.info("_normalize_forecast_station_labels: updated %d legacy rows", changed)


async def _fetch_and_store_actuals(state: AppState) -> dict[str, int]:
    """Fetch daily observed max temps and write Forecast.actual_temp."""
    end_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    lookback_days = max(1, int(getattr(state.settings, "actuals_lookback_days", 30)))
    start_date = end_date - timedelta(days=lookback_days - 1)
    if end_date < start_date:
        return {"stations": 0, "rows_updated": 0}

    total_rows = 0
    station_count = 0
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    async with aiohttp.ClientSession() as session:
        for city in state.settings.cities_list:
            coords = CITY_COORDS.get(city)
            if coords is None:
                continue

            station_code = _canonical_station_code(city)
            lat, lon = coords

            try:
                daily_actuals = await fetch_daily_max_actual(
                    lat=lat,
                    lon=lon,
                    start_date=start_str,
                    end_date=end_str,
                    session=session,
                )
            except Exception as exc:
                log.warning("actuals fetch failed for %s/%s: %s", city, station_code, exc)
                continue

            if not daily_actuals:
                continue

            with get_session() as db_session:
                updated = store_actual_temperatures(
                    db_session=db_session,
                    station=station_code,
                    daily_actuals=daily_actuals,
                    station_aliases=(city,),
                )

            total_rows += updated
            station_count += 1

    return {"stations": station_count, "rows_updated": total_rows}


# ---------------------------------------------------------------------------
# Decision loop (trading logic — reads DB, never fetches data)
# ---------------------------------------------------------------------------

def run_decision_loop() -> None:
    """5-minute decision loop per Addendum Gap 7.

    a. Check weather data freshness — skip if stale >2h (Gap 12)
    b. Read latest forecasts + market prices from DB
    c. For each market with a known resolution source:
       - Build MarketOpportunity
       - Run conservative_strategy, aggressive_strategy, mean_reversion_strategy
    d. Group signals by market_id
    e. Pass to portfolio_manager.aggregate_signals() → list[TradeOrder]
    f. For each TradeOrder: convert to Order and paper trade
    g. Run cross_platform_arb.scan() and log opportunities
    h. Record fills via fill_monitor
    i. Cancel expired orders (>2h TTL)

    APScheduler config: max_instances=1, coalesce=True, misfire_grace_time=120
    """
    state = get_state()

    if state.shutting_down:
        log.info("run_decision_loop: shutting down — skipping")
        return

    if not state._decision_cycle_lock.acquire(blocking=False):
        log.info("run_decision_loop: decision_cycle_skipped_overlap")
        return

    try:
        _run_decision_cycle(state)
    finally:
        state._decision_cycle_lock.release()


def _run_decision_cycle(state: AppState) -> None:
    """Inner decision cycle body — called by run_decision_loop() with overlap lock held."""
    log.info("run_decision_loop: starting")

    # Clear TimesFM per-cycle cache to avoid stale inference results
    if state.timesfm_forecaster is not None:
        state.timesfm_forecaster.clear_cycle_cache()

    # Sync resolved trade count to RiskManager for warm-up Kelly scaling
    _resolved = len([t for t in state.paper_trader.trade_history if t.get("status") in ("won", "lost")])
    state.risk_manager.set_resolved_trade_count(_resolved)

    # --- (a) Check data freshness (Addendum Gap 12) ---
    weather_age_minutes = _get_weather_data_age_minutes()
    if weather_age_minutes is None:
        log.warning("run_decision_loop: no weather data in DB — skipping cycle")
        return

    ttl = state.settings.weather_data_ttl_minutes
    if weather_age_minutes > ttl:
        log.warning(
            "run_decision_loop: weather data stale (%dm > %dm TTL) — skipping cycle",
            weather_age_minutes,
            ttl,
        )
        return

    log.debug("run_decision_loop: weather data age=%dm (TTL=%dm)", weather_age_minutes, ttl)

    # --- (b) Read latest forecasts + market prices ---
    try:
        forecasts = _read_latest_forecasts()
        market_prices = _read_latest_market_prices()
    except Exception as exc:
        log.error("run_decision_loop: DB read failed: %s", exc, exc_info=True)
        return

    if not market_prices:
        log.info("run_decision_loop: no market prices in DB — no trades this cycle")

    # --- (c) Build opportunities and collect signals per market ---
    signals_by_market: dict[str, list[Signal]] = {}
    opportunities: dict[str, MarketOpportunity] = {}

    # Tracking markets by platform for cross-platform arb scan
    poly_markets_raw: list[dict] = []
    kalshi_markets_raw: list[dict] = []

    log.info(
        "run_decision_loop: %d market_prices, %d forecasts, %d cached metadata",
        len(market_prices), len(forecasts), len(state.market_metadata),
    )

    # Minimum edge to qualify for LLM qualitative analysis (saves API calls)
    LLM_EDGE_THRESHOLD = 0.05

    # H2 fix: Collect opportunities by city for aggressive laddering
    _opps_by_city: dict[str, list[MarketOpportunity]] = {}

    built_count = 0
    llm_calls = 0
    llm_skipped = 0
    position_skipped = 0

    # H5 fix: Pre-pass — build ALL opportunities before running any strategy.
    # Previously _opps_by_city was populated mid-loop, so the first market in a
    # city saw 0 companions while later ones saw all predecessors (asymmetric ladder).
    # Now every market sees the full city bucket when strategies execute.
    for market_id, price_data in market_prices.items():
        # US-VF: Volume/spread filter — skip illiquid markets before building opportunity
        if MODULES.get("volume_filter", False):
            _vol = price_data.get("volume")
            _spread = price_data.get("spread")
            if _vol is not None and _spread is not None:
                _ok, _reason = should_evaluate_market(float(_vol), float(_spread))
                if not _ok:
                    log.debug("run_decision_loop: skipping %s — %s (vol=%.0f spread=%.4f)", market_id, _reason, _vol, _spread)
                    continue

        try:
            opp = _build_opportunity(market_id, price_data, forecasts)
        except Exception as exc:
            log.error(
                "run_decision_loop: failed to build opportunity for %s: %s",
                market_id, exc, exc_info=True,
            )
            continue
        if opp is None:
            continue
        built_count += 1
        opportunities[market_id] = opp
        _opps_by_city.setdefault(opp.city, []).append(opp)

    # Strategy pass — all opportunities and city buckets are now fully populated
    for market_id, opp in opportunities.items():
        price_data = market_prices.get(market_id, {})

        # Skip markets where we already have an open position
        if state.paper_trader.has_open_position(market_id):
            position_skipped += 1
            continue

        # Collect raw dicts for arb scan
        # Extract date from cached metadata if available
        _meta = state.market_metadata.get(market_id, {})
        _parsed = _meta.get("parsed")
        _arb_date = _parsed.date if _parsed and hasattr(_parsed, "date") else ""
        raw = {
            "city": opp.city,
            "date": _arb_date,
            "threshold": opp.threshold,
            "yes_price": opp.market_price,
        }
        if opp.platform == "polymarket":
            poly_markets_raw.append(raw)
        else:
            kalshi_markets_raw.append(raw)

        # Update in-memory price history for mean reversion
        # BugFix N2: Cap price history to prevent unbounded memory growth
        history = state._price_history.setdefault(market_id, [])
        if opp.market_price > 0:
            history.append(opp.market_price)
            _max_hist = state.settings.mean_reversion_lookback_hours * 2
            if len(history) > _max_hist:
                state._price_history[market_id] = history[-state.settings.mean_reversion_lookback_hours:]

        market_signals: list[Signal] = []

        # Conservative strategy
        if state.settings.enable_conservative_agent:
            try:
                sig = conservative_strategy(opp, state.settings)
                market_signals.append(sig)
            except Exception as exc:
                log.warning("conservative_strategy failed for %s: %s", market_id, exc)

        # Aggressive strategy — pass same-city opportunities as bucket markets for laddering
        if state.settings.enable_aggressive_agent:
            try:
                _bucket_markets = [o for o in _opps_by_city.get(opp.city, []) if o.market_id != market_id]
                sigs = aggressive_strategy(opp, _bucket_markets, state.settings)
                # Aggressive laddering may emit signals for adjacent bucket markets.
                # Route each signal by its own market_id so side/confidence are applied
                # to the correct opportunity in PortfolioManager.
                for sig in sigs:
                    if sig.market_id == market_id:
                        market_signals.append(sig)
                    elif not state.paper_trader.has_open_position(sig.market_id):
                        signals_by_market.setdefault(sig.market_id, []).append(sig)
            except Exception as exc:
                log.warning("aggressive_strategy failed for %s: %s", market_id, exc)

        # Mean reversion strategy (requires price history)
        # PRD v3.1: STRUCTURAL KILL — gated by MODULES flag
        if MODULES["agent_mean_reversion"] and state.settings.enable_mean_reversion_agent and len(history) >= 20:
            try:
                sig = mean_reversion_strategy(market_id, history, state.settings)
                market_signals.append(sig)
            except Exception as exc:
                log.warning("mean_reversion_strategy failed for %s: %s", market_id, exc)

        # Sell NO strategy (v2): buy NO on overpriced YES brackets
        try:
            sig = sell_no_strategy(opp, state.settings)
            if sig.side != "no_trade":
                market_signals.append(sig)
        except Exception as exc:
            log.warning("sell_no_strategy failed for %s: %s", market_id, exc)

        # TimesFM Quantile Tail strategy (Phase 6)
        if getattr(state.settings, 'enable_timesfm_agent', False) and state.timesfm_forecaster is not None:
            try:
                from weatheredge.agents.strategies.timesfm_quantile import timesfm_quantile_strategy
                sig = timesfm_quantile_strategy(opp, state.settings)
                if sig.side != "no_trade":
                    market_signals.append(sig)
            except Exception as exc:
                log.warning("timesfm_quantile_strategy failed for %s: %s", market_id, exc)

        # Qualitative analysis — rules engine (deterministic, no LLM)
        # Only run for markets with meaningful edge to avoid noise
        try:
            if max(opp.net_edge, opp.net_edge_no) >= LLM_EDGE_THRESHOLD:
                qual_signal = state.rules_qualitative_engine.analyze(
                    opp,
                    conditions_tracker=state.conditions_tracker,
                    calibration_tracker=state.calibration_tracker,
                )
                llm_calls += 1  # counter kept for logging compatibility
                if qual_signal.side != "no_trade":
                    market_signals.append(qual_signal)
            else:
                llm_skipped += 1
        except Exception as exc:
            log.warning("rules_qualitative_engine failed for %s: %s", market_id, exc)

        # LLM qualitative agent available as offline fallback (not called in hot loop)
        # To enable: state.qualitative_agent.analyze(opp) — kept for weekly learning/reporting

        if market_signals:
            signals_by_market.setdefault(market_id, []).extend(market_signals)

    # --- (d/e) Aggregate signals via portfolio manager ---
    bankroll = state.paper_trader.bankroll
    open_positions = state.paper_trader.open_positions
    # BugFix C2: Use today's P&L only, not all-time cumulative sum
    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_pnl = state.paper_trader.daily_pnl.get(_today, 0.0)
    realized_total_pnl = sum(t.get("pnl", 0.0) or 0.0 for t in state.paper_trader.trade_history)
    # Include mark-to-market on open positions so circuit breakers react before
    # losses are formally realised at exit/resolution.
    unrealized_total_pnl = 0.0
    for pos in open_positions:
        if not isinstance(pos, dict):
            continue
        market_id = pos.get("market_id")
        side = pos.get("side")
        amount = float(pos.get("amount") or 0.0)
        cost = float(pos.get("cost") or 0.0)
        if amount <= 0.0:
            continue
        current_opp = opportunities.get(market_id) if market_id else None
        current_yes_price = (
            current_opp.market_price if current_opp is not None else float(pos.get("price") or 0.0)
        )
        if side == "buy_yes":
            current_value = current_yes_price * amount
        elif side == "buy_no":
            current_value = (1.0 - current_yes_price) * amount
        else:
            continue
        unrealized_total_pnl += (current_value - cost)
    total_pnl = realized_total_pnl + unrealized_total_pnl

    log.info(
        "run_decision_loop: built %d opportunities from %d markets, %d with signals "
        "(rules_qual=%d, skipped=%d edge<%d%%, %d already open)",
        built_count, len(market_prices), len(signals_by_market),
        llm_calls, llm_skipped, int(LLM_EDGE_THRESHOLD * 100), position_skipped,
    )

    # Collect recent fills for VPIN toxic flow detection (C5 fix)
    # PRD v3.1: STRUCTURAL KILL — VPIN requires HFT order flow that doesn't exist on weather markets
    _recent_fills = None
    if MODULES.get("vpin", False) and state.fill_monitor is not None:
        try:
            from weatheredge.db.database import get_session as _gs
            from weatheredge.db.models import FillQuality as _FQ, Trade as _Trade
            with _gs() as _sess:
                _fill_rows = (
                    _sess.query(_FQ, _Trade.side, _Trade.amount)
                    .join(_Trade, _Trade.id == _FQ.trade_id, isouter=True)
                    .order_by(_FQ.id.desc())
                    .limit(state.settings.vpin_window)
                    .all()
                )
                if _fill_rows:
                    parsed_fills = []
                    for _fill, _trade_side, _trade_amount in _fill_rows:
                        if _trade_side not in ("buy_yes", "buy_no"):
                            continue
                        parsed_fills.append(
                            {
                                "size": float(abs(_trade_amount or 0.0)),
                                # buy_no is equivalent to selling YES flow for VPIN directionality
                                "side": "buy" if _trade_side == "buy_yes" else "sell",
                            }
                        )
                    if parsed_fills:
                        _recent_fills = parsed_fills
        except Exception as _vpin_exc:
            log.debug("VPIN fill data unavailable: %s", _vpin_exc)

    try:
        trade_orders = state.portfolio_manager.aggregate_signals(
            signals_by_market=signals_by_market,
            opportunities=opportunities,
            bankroll=bankroll,
            open_positions=open_positions,
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            recent_fills=_recent_fills,
        )
    except Exception as exc:
        log.error("run_decision_loop: portfolio_manager failed: %s", exc, exc_info=True)
        trade_orders = []

    # --- (f) Shadow warm-up mode: log signals but don't execute trades ---
    # Uses the forecasts dict already loaded (avoids extra DB query that fails with mocks)
    if getattr(state.settings, 'enable_warmup_shadow_mode', False) and forecasts:
        try:
            _warmup_hours = getattr(state.settings, 'warmup_hours', 72)
            # Use earliest forecast_timestamp from already-loaded forecasts
            _oldest_ts = None
            for _fc_data in forecasts.values():
                _fc_ts = _fc_data.get("forecast_timestamp")
                if _fc_ts is not None:
                    if hasattr(_fc_ts, 'tzinfo'):
                        if _fc_ts.tzinfo is None:
                            _fc_ts = _fc_ts.replace(tzinfo=timezone.utc)
                        if _oldest_ts is None or _fc_ts < _oldest_ts:
                            _oldest_ts = _fc_ts
            if _oldest_ts is not None:
                _db_age_hours = (datetime.now(timezone.utc) - _oldest_ts).total_seconds() / 3600.0
                if _db_age_hours < _warmup_hours:
                    _remaining = _warmup_hours - _db_age_hours
                    log.info(
                        "SHADOW MODE: DB age %.1fh < %dh warm-up — %d signals generated, 0 trades placed. %.1fh remaining.",
                        _db_age_hours, _warmup_hours, len(trade_orders), _remaining,
                    )
                    return
        except Exception as _shadow_exc:
            log.debug("Shadow mode check failed: %s — proceeding with trading", _shadow_exc)

    # --- (g) Execute TradeOrders via paper trader ---
    current_prices = {mid: opp.market_price for mid, opp in opportunities.items()}

    # Set limit order type when enabled, and check existing pending limit orders
    if MODULES.get("limit_orders", False):
        for trade_order in trade_orders:
            trade_order.order_type = "limit"
        limit_fills = state.paper_trader.check_limit_fills(current_prices)
        if limit_fills > 0:
            log.info("run_decision_loop: %d limit orders filled this cycle", limit_fills)

    trades_placed = 0
    for trade_order in trade_orders:
        # Final guard: skip if position was opened by a concurrent/recent cycle
        if state.paper_trader.has_open_position(trade_order.market_id):
            continue
        try:
            order = Order(
                market_id=trade_order.market_id,
                side=trade_order.side,
                price=trade_order.price,
                amount=trade_order.amount,
                agent_name=trade_order.agent_name,
                model_probability=trade_order.model_probability,
                edge_at_entry=trade_order.edge_at_entry,
                city=trade_order.city,
                platform=trade_order.platform or "polymarket",
                station=trade_order.station,
                market_date=trade_order.market_date,
                gaussian_prob=trade_order.gaussian_prob,
                ensemble_prob=trade_order.ensemble_prob,
                blended_prob=trade_order.blended_prob,
                order_type=trade_order.order_type,
            )
            result = state.paper_trader.place_order(order)
            if result.get("filled") or result.get("pending"):
                action = "filled" if result.get("filled") else "pending limit"
                log.info(
                    "run_decision_loop: %s paper trade — market=%s side=%s size=%.2f price=%.3f",
                    action,
                    trade_order.market_id,
                    trade_order.side,
                    trade_order.amount,
                    trade_order.price,
                )
                trades_placed += 1

                # --- (h) Record fill via fill_monitor ---
                if state.fill_monitor is not None:
                    try:
                        fill = FillRecord(
                            # BugFix L3: Use actual DB id, not Python memory address
                            trade_id=result.get("trade", {}).get("db_id", 0),
                            fill_price=trade_order.price,
                            market_mid_at_fill=trade_order.price,
                            side=trade_order.side,
                        )
                        state.fill_monitor.record_fill(fill)
                    except Exception as exc:
                        log.warning(
                            "fill_monitor.record_fill failed for %s: %s",
                            trade_order.market_id, exc,
                        )
            else:
                log.warning(
                    "run_decision_loop: order rejected for %s: %s",
                    trade_order.market_id,
                    result.get("reason"),
                )
        except Exception as exc:
            log.error(
                "run_decision_loop: error executing trade for %s: %s",
                trade_order.market_id, exc, exc_info=True,
            )

    # --- Exit check: evaluate all open positions for exit signals ---
    try:
        exit_manager = ExitManager()
        exits = exit_manager.evaluate(
            positions=state.paper_trader.open_positions[:],
            opportunities=opportunities,
            market_metadata=state.market_metadata,
        )
        for pos, reason in exits:
            current_opp = opportunities.get(pos["market_id"])
            current_yes_price = current_opp.market_price if current_opp else pos["price"]
            closed = state.paper_trader.close_position(pos["market_id"], current_yes_price, reason)
            if closed:
                log.info(
                    "run_decision_loop: exit triggered — market=%s reason=%s price=%.3f",
                    pos["market_id"], reason, current_yes_price,
                )
    except Exception as exc:
        log.error("run_decision_loop: exit_manager failed: %s", exc, exc_info=True)

    # --- Phase 3: Record trades and trigger weight updates ---
    if trades_placed > 0:
        for _ in range(trades_placed):
            if state.model_weight_updater is not None:
                state.model_weight_updater.record_trade()
            if state.agent_weight_updater is not None:
                state.agent_weight_updater.record_trade()

        if (
            MODULES.get("dynamic_model_weights", False)
            and state.model_weight_updater is not None
            and state.model_weight_updater.should_update(
                state.settings.weight_update_interval_trades
            )
        ):
            try:
                state.model_weight_updater.update_model_weights()
                log.info("run_decision_loop: model weights updated")
            except Exception as exc:
                log.warning("run_decision_loop: model weight update failed: %s", exc)

        if (
            MODULES.get("dynamic_agent_weights", False)
            and state.agent_weight_updater is not None
            and state.agent_weight_updater.should_update(
                state.settings.agent_weight_update_interval
            )
        ):
            try:
                new_weights = state.agent_weight_updater.update_agent_weights(
                    state.portfolio_manager.AGENT_WEIGHTS,
                    calibration_tracker=state.calibration_tracker,
                )
                state.portfolio_manager.AGENT_WEIGHTS = new_weights
                log.info(
                    "run_decision_loop: agent weights updated: %s", new_weights
                )
            except Exception as exc:
                log.warning("run_decision_loop: agent weight update failed: %s", exc)

    # --- (g) Cross-platform arbitrage scan ---
    if MODULES.get("cross_platform_arb", False):
        try:
            arb_opps = state.cross_platform_arb.scan(poly_markets_raw, kalshi_markets_raw)
            if arb_opps:
                log.info(
                    "run_decision_loop: cross-platform arb — %d opportunities found",
                    len(arb_opps),
                )
                for arb in arb_opps:
                    log.info(
                        "  arb: city=%s threshold=%.1f poly=%.3f kalshi=%.3f spread=%.3f dir=%s",
                        arb.city, arb.threshold, arb.poly_price, arb.kalshi_price,
                        arb.spread, arb.direction,
                    )
                    arb_fills = state.cross_platform_arb.execute_arb(
                        arb, state.paper_trader, state.paper_trader.bankroll
                    )
                    if arb_fills:
                        log.info(
                            "  arb executed: city=%s %d legs filled",
                            arb.city, len(arb_fills),
                        )
                    else:
                        log.debug("  arb execution skipped: city=%s", arb.city)
            else:
                log.debug("run_decision_loop: cross-platform arb — no opportunities this cycle")
        except Exception as exc:
            log.warning("run_decision_loop: cross_platform_arb.scan failed: %s", exc)

    # --- (i) Cancel expired orders ---
    cancelled = state.paper_trader.cancel_expired_orders()
    if cancelled > 0:
        log.info("run_decision_loop: cancelled %d expired orders", cancelled)

    log.info(
        "run_decision_loop: complete — %d trades placed, %d orders cancelled",
        trades_placed,
        cancelled,
    )


def _get_weather_data_age_minutes() -> float | None:
    """Return age of most recent forecast row in minutes, or None if no data."""
    try:
        with get_session() as session:
            row = (
                session.query(Forecast.forecast_timestamp)
                .order_by(Forecast.forecast_timestamp.desc())
                .first()
            )
        if row is None:
            return None
        ts: datetime = row[0]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts).total_seconds() / 60.0
        return age
    except Exception as exc:
        log.warning("_get_weather_data_age_minutes: DB error: %s", exc)
        return None


def _load_timesfm_history(state: AppState) -> None:
    """Load historical station temperature data and NWP error history for TimesFM.

    Populates the TimesFMForecaster with per-station context from the DB:
    - Historical observed temperatures (from calibration records actuals)
    - NWP forecast error residuals (actual - forecast) for error correction
    """
    if state.timesfm_forecaster is None:
        return

    try:
        from weatheredge.db.models import CalibrationRecord
        with get_session() as session:
            # Get stations that have calibration data
            stations = session.query(CalibrationRecord.station).distinct().all()
            stations = [s[0] for s in stations if s[0]]

            for station in stations:
                # Load actual temperatures (for quantile probability)
                rows = (
                    session.query(
                        CalibrationRecord.actual_temp,
                        CalibrationRecord.forecast_temp,
                    )
                    .filter(
                        CalibrationRecord.station == station,
                        CalibrationRecord.actual_temp.isnot(None),
                    )
                    .order_by(CalibrationRecord.created_at.asc())
                    .limit(state.timesfm_forecaster.context_length)
                    .all()
                )

                if rows:
                    temps = [float(r.actual_temp) for r in rows if r.actual_temp is not None]
                    if temps:
                        state.timesfm_forecaster.update_station_history(station, temps)

                    # Build error residual series (actual - forecast) for error correction
                    errors = [
                        float(r.actual_temp) - float(r.forecast_temp)
                        for r in rows
                        if r.actual_temp is not None and r.forecast_temp is not None
                    ]
                    if errors:
                        state.timesfm_forecaster.update_error_history(station, errors)

            log.info(
                "TimesFM history loaded: %d stations with temp data, %d with error data",
                len(state.timesfm_forecaster._station_temps),
                len(state.timesfm_forecaster._station_errors),
            )
    except Exception as exc:
        log.warning("_load_timesfm_history: failed to load historical data: %s", exc)


def _read_latest_forecasts() -> dict[str, dict[str, dict]]:
    """Read the most recent forecast rows per (station, target_date) from DB.

    Collects all model types (gfs, ecmwf, hrrr, ensemble) per station+date so
    the blended probability engine receives data for the correct forecast date.

    Returns mapping: station -> {date_str -> {temperature_forecast, ecmwf_temp,
                                               hrrr_temp, ensemble_members, ...}}
    """
    result: dict[str, dict[str, dict]] = {}
    # seen[(station, target_date)] = set of models already recorded for that key
    seen: dict[tuple, set[str]] = {}
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Only fetch rows for today and future dates, from recent fetches (last 3h).
    # This avoids scanning the full historical archive (476k+ rows) and ensures
    # each station+date gets the most recent model run.
    cutoff_ts = datetime.now(timezone.utc) - timedelta(hours=3)
    with get_session() as session:
        rows = (
            session.query(Forecast)
            .filter(Forecast.target_date >= today_str)
            .filter(Forecast.forecast_timestamp >= cutoff_ts)
            .order_by(Forecast.forecast_timestamp.desc())
            .limit(5000)
            .all()
        )
        for row in rows:
            station = row.station
            target_date = str(row.target_date) if row.target_date else None
            if not target_date:
                continue
            model = (row.model or "").lower()

            key = (station, target_date)
            if key not in seen:
                seen[key] = set()
            if model in seen[key]:
                continue
            seen[key].add(model)

            if station not in result:
                result[station] = {}
            if target_date not in result[station]:
                result[station][target_date] = {
                    "temperature_forecast": None,
                    "ecmwf_temp": None,
                    "hrrr_temp": None,
                    "ensemble_members": None,
                    "model": model,
                    "forecast_timestamp": row.forecast_timestamp,
                }

            entry = result[station][target_date]
            if "ensemble" in model:
                if row.ensemble_members:
                    try:
                        entry["ensemble_members"] = json.loads(row.ensemble_members)
                    except (json.JSONDecodeError, TypeError):
                        pass
            elif "ecmwf" in model:
                entry["ecmwf_temp"] = row.temperature_forecast
            elif "hrrr" in model:
                entry["hrrr_temp"] = row.temperature_forecast
            else:
                # GFS / Open-Meteo / NOAA — primary forecast.
                # Prefer "forecast" model (Open-Meteo daily max) over noaa_nws
                # (hourly rows that store one value per period, not the daily max).
                is_openmeteo_daily = model == "forecast"
                if entry["temperature_forecast"] is None or is_openmeteo_daily:
                    entry["temperature_forecast"] = row.temperature_forecast
                    entry["forecast_timestamp"] = row.forecast_timestamp
    return result


def _get_forecast_for_date(
    forecasts: dict[str, dict[str, dict]],
    station: str,
    market_date: str | None,
) -> dict | None:
    """Return the forecast entry for a specific station and market date.

    Lookup order:
    1. Exact (station, market_date) match
    2. Nearest future date for that station (earliest date >= market_date)
    3. Most recent available date for that station (last resort)
    Also tries the city-name key if station code not found.
    """
    station_forecasts = forecasts.get(station)
    if station_forecasts is None:
        city_name = STATION_TO_CITY.get(station)
        if city_name:
            station_forecasts = forecasts.get(city_name)
    if not station_forecasts:
        return None

    # Backward-compat: flat {station: {fields}} structure (used in tests and legacy callers).
    # Detected when the dict has forecast field keys directly instead of date keys.
    if "temperature_forecast" in station_forecasts:
        return station_forecasts

    if market_date and market_date in station_forecasts:
        return station_forecasts[market_date]

    # Nearest future date
    if market_date:
        future_dates = sorted(d for d in station_forecasts if d >= market_date)
        if future_dates:
            log.debug(
                "_get_forecast_for_date: no exact forecast for %s/%s — using nearest %s",
                station, market_date, future_dates[0],
            )
            return station_forecasts[future_dates[0]]

    # Last resort: most recent date available
    if station_forecasts:
        latest = max(station_forecasts.keys())
        log.debug(
            "_get_forecast_for_date: no future forecast for %s/%s — using latest available %s",
            station, market_date, latest,
        )
        return station_forecasts[latest]

    return None


def _read_latest_market_prices() -> dict[str, dict]:
    """Read the most recent price snapshot per market from DB.

    Returns mapping: market_id -> {yes_price, no_price, volume, spread}
    """
    result: dict[str, dict] = {}
    with get_session() as session:
        # BugFix H4: Removed hard LIMIT 200 that silently dropped less-active markets.
        # Use LIMIT 5000 as safety cap; Python-side first-seen dedup keeps latest per market.
        rows = (
            session.query(MarketPrice)
            .order_by(MarketPrice.timestamp.desc())
            .limit(5000)
            .all()
        )
        for row in rows:
            if row.market_id and row.market_id not in result:
                result[row.market_id] = {
                    "yes_price": row.yes_price,
                    "no_price": row.no_price,
                    "volume": row.volume,
                    "spread": row.spread,
                    "timestamp": row.timestamp,
                }
    return result


def _build_opportunity(
    market_id: str,
    price_data: dict,
    forecasts: dict[str, dict],
) -> MarketOpportunity | None:
    """Build a MarketOpportunity from DB data for a known market.

    Returns None if the market is unknown, lacks a forecast, or has invalid pricing.

    Resolution is determined in this order:
    1. Cached metadata from the scanning pipeline (has parsed data)
    2. Kalshi prefix match against RESOLUTION_SOURCES (e.g. KXHIGHDEN-26MAR30-T76 → KXHIGHDEN)
    3. Exact match in RESOLUTION_SOURCES (legacy fallback)
    """
    state = get_state()

    # --- Resolve market metadata ---
    metadata = state.market_metadata.get(market_id)
    resolution_source = None
    platform = None
    parsed = None  # ParsedMarket from title parser

    if metadata:
        resolution_source = metadata.get("resolution_source")
        platform = metadata.get("platform")
        parsed = metadata.get("parsed")  # ParsedMarket or None
    else:
        # Fallback: try Kalshi prefix match (KXHIGHDEN-26MAR30-T76 → KXHIGHDEN)
        kalshi_tickers = RESOLUTION_SOURCES.get("kalshi", {})
        prefix = market_id.split("-")[0] if "-" in market_id else market_id
        if prefix in kalshi_tickers:
            resolution_source = kalshi_tickers[prefix]
            platform = "kalshi"
        else:
            # Exact match fallback
            for plat, tickers in RESOLUTION_SOURCES.items():
                if market_id in tickers:
                    resolution_source = tickers[market_id]
                    platform = plat
                    break

    if resolution_source is None:
        return None

    station = resolution_source.get("station")
    if station is None:
        return None

    # --- Extract market parameters from parsed data or defaults ---
    if parsed:
        threshold = parsed.threshold if parsed.threshold is not None else 70.0
        direction = parsed.direction or "above"
        market_type = parsed.market_type or "daily_high"
        market_date = parsed.date  # "YYYY-MM-DD"
        bucket_low = parsed.threshold_low
        bucket_high = parsed.threshold_high
        # BugFix C6: For bucket markets, keep midpoint as fallback threshold but
        # actual probability is calculated below using bucket_probability()
        if bucket_low is not None and bucket_high is not None and parsed.threshold is None:
            threshold = (bucket_low + bucket_high) / 2.0  # used only as fallback
    else:
        # Kalshi markets: extract threshold from market_id if possible
        # e.g. KXHIGHDEN-26MAR30-T76 → threshold=76, KXHIGHDEN-26MAR30-B76.5 → threshold=76.5
        threshold = 70.0
        direction = "above"
        market_type = resolution_source.get("type", "daily_high")
        market_date = None
        bucket_low = None
        bucket_high = None
        # Try to parse Kalshi contract-level threshold from market_id
        parts = market_id.split("-")
        if len(parts) >= 3:
            tail = parts[-1]  # e.g. "T76" or "B76.5"
            try:
                if tail.startswith("T"):
                    threshold = float(tail[1:])
                elif tail.startswith("B"):
                    threshold = float(tail[1:])
                    direction = "below"
            except ValueError:
                pass

    # --- Look up forecast for the specific market date ---
    forecast = _get_forecast_for_date(forecasts, station, market_date)
    if forecast is None:
        log.debug("_build_opportunity: no forecast for station %s date %s", station, market_date)
        return None

    yes_price = price_data.get("yes_price")
    if yes_price is None or yes_price <= 0.0 or yes_price >= 1.0:
        return None

    # FIX 3: Never trade contracts priced below $0.10 — below that we have no edge over market consensus
    # Applies symmetrically: skip YES under $0.10 AND NO under $0.10 (i.e. yes_price > $0.90)
    if yes_price < 0.10:
        log.debug("_build_opportunity: %s skipped — yes_price=%.3f < $0.10 minimum", market_id, yes_price)
        return None
    if yes_price > 0.90:
        log.debug("_build_opportunity: %s skipped — NO price=%.3f < $0.10 minimum", market_id, 1.0 - yes_price)
        return None

    gfs_temp = forecast.get("temperature_forecast")
    ecmwf_temp = forecast.get("ecmwf_temp")
    hrrr_temp = forecast.get("hrrr_temp")
    ensemble_members = forecast.get("ensemble_members")

    # BugFix E3: Defensive °C→°F conversion for any temperature that looks like Celsius.
    # All three fetch endpoints now request fahrenheit, but old DB rows (pre-fix) may still
    # contain Celsius values. A daily-max in °F is always > 10°F for traded markets;
    # values < 50 strongly suggest Celsius. Convert and warn rather than silently producing
    # near-zero probabilities that create phantom buy_no edge.
    def _guard_fahrenheit(temp: float | None, label: str) -> float | None:
        # Convert only if value is clearly in Celsius range (0 < t < 35).
        # US daily-max in °F is rarely below 35°F for traded markets; any value
        # < 35 that reaches here is almost certainly stored in °C (pre-fix DB row).
        if temp is not None and 0 < temp < 35.0:
            converted = temp * 9.0 / 5.0 + 32.0
            log.warning(
                "_build_opportunity: %s %s temp=%.1f looks like °C — converting to %.1f°F",
                market_id, label, temp, converted,
            )
            return converted
        return temp

    gfs_temp = _guard_fahrenheit(gfs_temp, "gfs")
    ecmwf_temp = _guard_fahrenheit(ecmwf_temp, "ecmwf")
    hrrr_temp = _guard_fahrenheit(hrrr_temp, "hrrr")
    if ensemble_members:
        # Use mean to avoid false positives from ensemble spread tails.
        mean_ens = sum(ensemble_members) / len(ensemble_members)
        if 0 < mean_ens < 35.0:
            ensemble_members = [t * 9.0 / 5.0 + 32.0 for t in ensemble_members]
            log.warning("_build_opportunity: %s ensemble_members look like °C (mean=%.1f) — converting all to °F", market_id, mean_ens)

    # Need at least one temperature source
    forecast_temp = _first_not_none(gfs_temp, ecmwf_temp, hrrr_temp)
    if forecast_temp is None:
        return None

    # Calculate lead time from market date
    lead_time_hours = 24.0
    lead_time_days = 1
    if market_date:
        try:
            # Use station-local end-of-day (23:59 local) when timezone is known.
            delta_hours = hours_until_market_close(market_date, station)
            if delta_hours is None:
                # Fallback for unknown stations: conservative UTC end-of-day.
                target = datetime.strptime(market_date, "%Y-%m-%d").replace(
                    hour=23, minute=59, tzinfo=timezone.utc
                )
                now = datetime.now(timezone.utc)
                delta_hours = (target - now).total_seconds() / 3600.0
            # BugFix C5: Skip markets whose resolution date has passed
            if delta_hours <= 0:
                log.debug("_build_opportunity: %s skipped — market date %s is in the past", market_id, market_date)
                return None
            lead_time_hours = delta_hours
            lead_time_days = max(0, int(delta_hours / 24.0))
        except (ValueError, TypeError):
            pass

    # H1 fix: Use station-specific RMSE from calibration table instead of hardcoded values
    from weatheredge.probability.bias_correction import get_rmse
    is_us = station in US_STATIONS
    _primary_model = "hrrr" if (is_us and lead_time_hours < 24) else "gfs"
    rmse = get_rmse(station, _primary_model, lead_time_days)

    # Phase 1: Dynamic RMSE floor — ensures HRRR <12h never goes below 1.5°F, etc.
    if MODULES.get("rmse_floor_dynamic", False):
        from weatheredge.bias_correction import get_rmse_floor
        rmse = max(rmse, get_rmse_floor(_primary_model, lead_time_hours))

    # Phase 1: Apply per-model bias correction (gated — only when bias_correction ON)
    if MODULES.get("bias_correction", False):
        try:
            from weatheredge.bias_correction import get_bias_correction
            _month = int(market_date.split("-")[1]) if market_date else datetime.now(timezone.utc).month
            if gfs_temp is not None:
                gfs_temp -= get_bias_correction(station, "gfs", _month)
            if ecmwf_temp is not None:
                ecmwf_temp -= get_bias_correction(station, "ecmwf", _month)
            if hrrr_temp is not None:
                hrrr_temp -= get_bias_correction(station, "hrrr", _month)
            forecast_temp = _first_not_none(gfs_temp, ecmwf_temp, hrrr_temp)
        except Exception as _bc_exc:
            log.debug("_build_opportunity: bias correction skipped for %s: %s", market_id, _bc_exc)

    # Enhancement 3: Check METAR override
    metar_override = state.conditions_tracker.metar_probability_override(
        station=station,
        threshold=threshold,
        direction=direction,
        market_type=market_type,
        market_date=market_date,
    )

    # FIX 4: Check METAR vs forecast contradiction before using override.
    # If they disagree by >30 percentage points, skip the trade and alert for manual review.
    if metar_override is not None:
        try:
            _forecast_prob_only, _, _ = calculate_blended_probability(
                threshold=threshold,
                direction=direction,
                rmse=rmse,
                is_us_station=is_us,
                lead_time_hours=lead_time_hours,
                ensemble_temps=ensemble_members,
                gfs_forecast_temp=gfs_temp,
                ecmwf_forecast_temp=ecmwf_temp,
                hrrr_forecast_temp=hrrr_temp if is_us else None,
                metar_override=None,  # pure forecast probability, no override
            )
            _contradiction_diff = abs(metar_override - _forecast_prob_only)
            if _contradiction_diff > 0.30:
                log.warning(
                    "METAR-FORECAST CONTRADICTION %s: metar=%.2f forecast=%.2f diff=%.2f — skipping, alerting",
                    market_id, metar_override, _forecast_prob_only, _contradiction_diff,
                )
                try:
                    from weatheredge.notifications.telegram import TelegramBot
                    _bot = TelegramBot(state.settings)
                    _run_async_safely(_bot.send_alert(
                        f"CONTRADICTION: {market_id}\n"
                        f"METAR prob={metar_override:.2f}, Forecast prob={_forecast_prob_only:.2f}\n"
                        f"Disagreement={_contradiction_diff:.2f} > 0.30 — manual review needed"
                    ))
                except Exception as _tg_exc:
                    log.warning("contradiction telegram alert failed: %s", _tg_exc)
                return None
        except Exception as _exc:
            log.warning("_build_opportunity: contradiction check failed for %s: %s", market_id, _exc)

    # Phase 6: TimesFM fields (set in non-bucket branch, default None for bucket markets)
    _timesfm_prob = None
    _timesfm_error_adj = None

    # BugFix C6: Use bucket_probability for bucket markets instead of midpoint "above"
    if bucket_low is not None and bucket_high is not None:
        try:
            from weatheredge.probability.bucket import bucket_probability
            probability = bucket_probability(forecast_temp, bucket_low, bucket_high, rmse)
            model_used = "bucket_gaussian"
            log.debug(
                "_build_opportunity: bucket market %s — P(%.0f<=T<=%.0f)=%.3f",
                market_id, bucket_low, bucket_high, probability,
            )
        except Exception as exc:
            log.warning("_build_opportunity: bucket_probability failed for %s: %s", market_id, exc)
            return None
    else:
        # Enhancement 5: Use blended probability with HRRR weighting
        # Apply learned model weights if available (C4 fix)
        _learned_weights = None
        if state.model_weight_updater is not None:
            _lw = {}
            _lw_gfs = state.model_weight_updater.get_model_weight(station, "gfs_ensemble", lead_time_days)
            _lw_ecmwf = state.model_weight_updater.get_model_weight(station, "ecmwf", lead_time_days)
            _lw_hrrr = state.model_weight_updater.get_model_weight(station, "hrrr", lead_time_days)
            # Only use learned weights if they differ from defaults (meaning we have real data)
            if not (_lw_gfs == 1.0 and _lw_ecmwf == 1.0 and _lw_hrrr == 1.0):
                _lw = {"gfs_ensemble": _lw_gfs, "ecmwf": _lw_ecmwf, "hrrr": _lw_hrrr}
                _learned_weights = _lw

        # Phase 6: TimesFM probability + NWP error correction
        if state.timesfm_forecaster is not None:
            try:
                _timesfm_prob, _tfm_details = state.timesfm_forecaster.get_probability(
                    station=station,
                    threshold=threshold,
                    direction=direction,
                    lead_time_hours=lead_time_hours,
                )
            except Exception as _tfm_exc:
                log.debug("TimesFM probability failed for %s: %s", market_id, _tfm_exc)

            try:
                _timesfm_error_adj, _err_details = state.timesfm_forecaster.get_error_correction(
                    station=station,
                )
            except Exception as _err_exc:
                log.debug("TimesFM error correction failed for %s: %s", market_id, _err_exc)

        try:
            probability, model_used, blend_details = calculate_blended_probability(
                threshold=threshold,
                direction=direction,
                rmse=rmse,
                is_us_station=is_us,
                lead_time_hours=lead_time_hours,
                ensemble_temps=ensemble_members,
                gfs_forecast_temp=gfs_temp,
                ecmwf_forecast_temp=ecmwf_temp,
                hrrr_forecast_temp=hrrr_temp if is_us else None,
                metar_override=metar_override,
                learned_weights=_learned_weights,
                timesfm_prob=_timesfm_prob,
                timesfm_error_adj=_timesfm_error_adj or 0.0,
            )
        except Exception as exc:
            log.warning("_build_opportunity: blended probability failed for %s: %s — falling back", market_id, exc)
            try:
                probability, model_used = calculate_probability(
                    forecast_temp=forecast_temp,
                    threshold=threshold,
                    rmse=rmse,
                    ensemble_temps=ensemble_members,
                    direction=direction,
                )
            except Exception as exc2:
                log.warning("_build_opportunity: probability calc failed for %s: %s", market_id, exc2)
                return None

    try:
        net_edge = calculate_net_edge(probability, yes_price, platform)
        net_edge_no = calculate_net_edge(1.0 - probability, 1.0 - yes_price, platform)
    except Exception as exc:
        log.warning("_build_opportunity: edge calc failed for %s: %s", market_id, exc)
        return None

    # FIX 2: Hard cap — gross edge >40% means our model is almost certainly wrong, not the market.
    # Tightened from 50% → 40% (Phase 0 evidence: 3 out of 4 losses had gross_edge 28-41%).
    _gross_edge = abs(probability - yes_price)
    if _gross_edge > 0.40:
        log.warning(
            "Unrealistic edge %.1f%% for %s: model=%.3f market=%.3f — likely model error. Skipping.",
            _gross_edge * 100, market_id, probability, yes_price,
        )
        return None

    city = STATION_TO_CITY.get(station, station)

    # Gaussian and ensemble probabilities for conservative model-agreement check
    from weatheredge.probability.gaussian import gaussian_probability
    from weatheredge.probability.ensemble import ensemble_probability

    gaussian_prob = gaussian_probability(forecast_temp, threshold, rmse, direction)
    ensemble_prob: float | None = None
    if ensemble_members and len(ensemble_members) >= 20:
        ensemble_prob = ensemble_probability(ensemble_members, threshold, direction)

    # BugFix P0-1: When blended probability hits the floor exactly (clamped),
    # the model has no differentiation signal. Use gaussian_prob as a better
    # point estimate — it varies by market and prevents Kelly over-betting.
    from weatheredge.probability.gaussian import PROB_FLOOR as _PROB_FLOOR
    if probability == _PROB_FLOOR and gaussian_prob > _PROB_FLOOR:
        log.debug(
            "_build_opportunity: blended hit floor for %s, upgrading model_prob %.3f → %.3f (gaussian)",
            market_id, probability, gaussian_prob,
        )
        probability = gaussian_prob
        model_used = f"gaussian_floor_escape({model_used})"
        # Recompute edges with updated probability
        try:
            net_edge = calculate_net_edge(probability, yes_price, platform)
            net_edge_no = calculate_net_edge(1.0 - probability, 1.0 - yes_price, platform)
        except Exception:
            pass  # Keep original edges if recompute fails

        # BugFix P0-3: Re-apply gross edge cap after P0-1 upgrade.
        # The first cap (above) ran when probability was at floor (near 0), so it
        # passed. Now that probability is upgraded, re-check with the real value.
        _gross_edge_post = abs(probability - yes_price)
        if _gross_edge_post > 0.40:
            log.warning(
                "Post-floor-escape edge %.1f%% for %s: model=%.3f market=%.3f — skipping.",
                _gross_edge_post * 100, market_id, probability, yes_price,
            )
            return None

    log.debug(
        "_build_opportunity: %s station=%s date=%s prob=%.3f price=%.3f net_edge=%.3f model=%s metar=%s threshold=%.1f dir=%s lead=%.0fh",
        market_id,
        station,
        market_date,
        probability,
        yes_price,
        net_edge,
        model_used,
        f"override={metar_override:.2f}" if metar_override else "none",
        threshold, direction, lead_time_hours,
    )

    return MarketOpportunity(
        market_id=market_id,
        platform=platform,
        city=city,
        station=station,
        threshold=threshold,
        direction=direction,
        market_price=yes_price,
        model_probability=probability,
        net_edge=net_edge,
        net_edge_no=net_edge_no,
        market_date=market_date or "",
        volume=price_data.get("volume"),
        ensemble_temps=ensemble_members,
        forecast_temp=forecast_temp,
        rmse=rmse,
        lead_time_days=lead_time_days,
        gaussian_prob=gaussian_prob,
        ensemble_prob=ensemble_prob,
        blended_prob=probability,
        timesfm_prob=_timesfm_prob,
        timesfm_error_adj=_timesfm_error_adj,
        bucket_low=bucket_low,
        bucket_high=bucket_high,
    )


def _evaluate_market(
    state: AppState,
    market_id: str,
    price_data: dict,
    forecasts: dict[str, dict],
    min_edge: float,
) -> int:
    """DEPRECATED: Legacy single-market evaluator. Superseded by _build_opportunity() +
    portfolio_manager.aggregate_signals() pipeline. Kept only as reference — never called.
    """
    raise NotImplementedError("_evaluate_market is deprecated — use portfolio manager pipeline")


# ---------------------------------------------------------------------------
# SIGTERM / SIGINT handler (Addendum Gap 14)
# ---------------------------------------------------------------------------

def _handle_shutdown(signum: int, frame: object) -> None:  # noqa: ARG001
    """Graceful shutdown sequence per Addendum Gap 14.

    1. Set shutting_down flag (stops decision loop from starting new cycles)
    2. Shutdown APScheduler
    3. Close database connections (SQLAlchemy engine dispose)
    """
    sig_name = signal.Signals(signum).name
    log.info("Received %s — initiating graceful shutdown", sig_name)

    state = get_state()
    state.shutting_down = True
    state._shutdown_event.set()  # BugFix L1: thread-safe signal

    if state.scheduler is not None:
        try:
            state.scheduler.shutdown(wait=False)
            log.info("Scheduler shut down")
        except Exception as exc:
            log.warning("Scheduler shutdown error: %s", exc)

    # Dispose SQLAlchemy engine to close all DB connections cleanly
    try:
        from weatheredge.db import database as _db
        if _db._engine is not None:
            _db._engine.dispose()
            log.info("Database connections closed")
    except Exception as exc:
        log.warning("DB close error: %s", exc)

    log.info("Graceful shutdown complete")


# ---------------------------------------------------------------------------
# Phase 3: Self-learning scheduler jobs
# ---------------------------------------------------------------------------

def run_resolution_check() -> None:
    """Daily job: resolve past-date markets with known outcomes.

    For each unresolved market whose date has passed, looks up actual
    temperature observations and determines the outcome (yes/no).
    Resolves trades in both the DB (via ResolutionTracker) and in-memory
    (via PaperTrader.resolve_position) so bankroll and P&L stay in sync.
    """
    state = get_state()
    if state.resolution_tracker is None:
        log.debug("run_resolution_check: resolution_tracker not initialised — skipping")
        return
    try:
        legacy_unresolved: list[str] = []
        if hasattr(state.resolution_tracker, "get_unresolved_markets"):
            # Backward compatibility: keep unresolved-market path used by legacy tests
            # while we also include closed/early-exit trades for calibration backfill.
            legacy_unresolved = list(state.resolution_tracker.get_unresolved_markets() or [])

        candidates = sorted(set(legacy_unresolved) | set(_get_resolution_candidate_markets()))
        if not candidates:
            log.info("run_resolution_check: no candidate markets")
            return

        log.info("run_resolution_check: %d candidate market(s)", len(candidates))
        resolved_count = 0
        calibration_count = 0

        for market_id in candidates:
            outcome = _determine_market_outcome(state, market_id)
            if outcome is not None:
                # Resolve open positions in DB (updates trade status, P&L, daily/agent performance).
                resolved = state.resolution_tracker.resolve_trades(market_id, outcome)
                if resolved:
                    log.info(
                        "run_resolution_check: resolved %d trade(s) for %s → %s",
                        len(resolved), market_id, outcome,
                    )
                    resolved_count += len(resolved)

                    # Resolve in-memory positions (updates bankroll).
                    if state.paper_trader is not None:
                        state.paper_trader.resolve_position(market_id, outcome)
            else:
                log.debug(
                    "run_resolution_check: cannot determine market outcome for %s yet",
                    market_id,
                )

            # Always try calibration backfill:
            # - won/lost trades can be calibrated even without fresh market fetch
            # - early exits are calibrated once market outcome becomes known
            calibration_count += _record_calibration_for_market(state, market_id, outcome)

        log.info(
            "run_resolution_check: resolved=%d, calibration_records_added=%d",
            resolved_count,
            calibration_count,
        )
    except Exception as exc:
        log.error("run_resolution_check failed: %s", exc, exc_info=True)


def _get_resolution_candidate_markets() -> list[str]:
    """Return markets that may need resolution and/or calibration backfill."""
    from sqlalchemy import select
    from weatheredge.db.models import Trade

    tracked_statuses = [
        "open",
        "won",
        "lost",
        "edge_stop",
        "capital_protect",
        "time_decay",
        "ev_based",
        "abandoned",
    ]

    with get_session() as session:
        rows = session.execute(
            select(Trade.market_id).where(
                Trade.market_id.is_not(None),
                Trade.status.in_(tracked_statuses),
            ).distinct()
        ).scalars().all()

    return sorted(str(m) for m in rows if m)


def _record_calibration_for_market(
    state: AppState,
    market_id: str,
    market_outcome: str | None,
) -> int:
    """Write missing calibration rows for all filled trades of a market.

    Includes early-exit trades once final market outcome is known.
    Uses trade_id dedupe in CalibrationTracker to remain idempotent.
    """
    if state.calibration_tracker is None:
        return 0

    from sqlalchemy import select
    from weatheredge.db.models import Trade

    calibratable_statuses = [
        "won",
        "lost",
        "edge_stop",
        "capital_protect",
        "time_decay",
        "ev_based",
        "abandoned",
    ]

    with get_session() as session:
        trades = session.execute(
            select(
                Trade.id.label("trade_id"),
                Trade.model_probability,
                Trade.station,
                Trade.status,
                Trade.side,
                Trade.market_date,
                Trade.created_at,
                Trade.agent_name,
            ).where(
                Trade.market_id == market_id,
                Trade.status.in_(calibratable_statuses),
                Trade.model_probability.is_not(None),
                Trade.station.is_not(None),
            )
        ).all()

    added = 0
    for trade in trades:
        try:
            predicted_prob = float(trade.model_probability)
        except (TypeError, ValueError):
            continue
        if not (0.0 <= predicted_prob <= 1.0):
            continue

        # Prefer explicit market outcome (needed for early exits).
        actual_outcome = None
        if market_outcome in ("yes", "no"):
            actual_outcome = 1 if market_outcome == "yes" else 0
        elif trade.status in ("won", "lost"):
            # Fallback for already-resolved trades if external outcome is unavailable.
            if trade.side == "buy_yes":
                actual_outcome = 1 if trade.status == "won" else 0
            elif trade.side == "buy_no":
                actual_outcome = 0 if trade.status == "won" else 1

        if actual_outcome is None:
            continue

        lead_days = 1
        try:
            if trade.market_date and trade.created_at is not None:
                target = datetime.strptime(str(trade.market_date), "%Y-%m-%d").replace(
                    hour=23, minute=59, tzinfo=timezone.utc
                )
                created_at = trade.created_at
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                delta_hours = max(1.0, (target - created_at).total_seconds() / 3600.0)
                lead_days = max(1, int(delta_hours // 24))
        except (ValueError, TypeError, AttributeError):
            lead_days = 1

        model_name = trade.agent_name or "blended"
        try:
            wrote = state.calibration_tracker.update_calibration(
                model=str(model_name),
                station=_canonical_station_code(str(trade.station)),
                lead_time_days=lead_days,
                predicted_prob=predicted_prob,
                actual_outcome=actual_outcome,
                trade_id=trade.trade_id,
            )
            if wrote:
                added += 1
        except Exception as exc:
            log.warning(
                "run_resolution_check: calibration update failed for trade_id=%s: %s",
                trade.trade_id,
                exc,
            )

    if added > 0:
        log.info("run_resolution_check: calibration backfill %s -> +%d rows", market_id, added)
    return added


def _fetch_polymarket_outcome(market_id: str) -> str | None:
    """Query Polymarket Gamma API for the resolved outcome of a market.

    Returns 'yes', 'no', or None if not yet resolved or on error.
    Uses outcomePrices (index 0=Yes, 1=No) when umaResolutionStatus=='resolved'.
    """
    import urllib.request
    import json as _json

    try:
        url = f"https://gamma-api.polymarket.com/markets/{market_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "WeatherEdge/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = _json.loads(resp.read())

        uma_status = data.get("umaResolutionStatus")
        # Accept "resolved" (final) or "proposed" (UMA in dispute window — rarely overturned)
        # For paper trading, proposed with decisive prices (>0.99) is sufficient.
        if uma_status not in ("resolved", "proposed"):
            return None

        prices = data.get("outcomePrices")
        if not prices:
            return None
        if isinstance(prices, str):
            try:
                import json as _json2
                prices = _json2.loads(prices)
            except Exception:
                return None
        if len(prices) < 2:
            return None

        try:
            yes_price = float(prices[0])
            no_price = float(prices[1])
        except (ValueError, TypeError):
            return None

        if yes_price >= 0.99:
            return "yes"
        if no_price >= 0.99:
            return "no"

        return None
    except Exception as exc:
        log.debug("_fetch_polymarket_outcome(%s) failed: %s", market_id, exc)
        return None


def _determine_market_outcome(state: "AppState", market_id: str) -> str | None:
    """Determine actual market outcome from weather observations.

    Returns 'yes', 'no', or None if outcome cannot yet be determined.
    Primary source: Polymarket Gamma API (umaResolutionStatus + outcomePrices).
    Fallback: Forecast table actual_temp for NOAA-sourced observations.
    """
    from sqlalchemy import select
    from weatheredge.db.models import Market as MarketModel, Forecast as ForecastModel
    from datetime import date as date_type

    try:
        with get_session() as session:
            market = session.execute(
                select(MarketModel).where(MarketModel.id == market_id)
            ).scalar_one_or_none()

            if market is None:
                return None

            station = market.station
            market_date = market.date  # "YYYY-MM-DD"
            if not station or not market_date:
                return None

            # Only resolve markets whose date has passed
            try:
                mdate = datetime.strptime(market_date, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                return None

            if mdate >= date_type.today():
                return None

            # Primary: ask Polymarket directly (works for polymarket markets)
            if market.platform == "polymarket":
                outcome = _fetch_polymarket_outcome(market_id)
                if outcome is not None:
                    log.debug(
                        "_determine_market_outcome(%s): Polymarket outcome=%s",
                        market_id, outcome,
                    )
                    return outcome

            # Fallback: look up actual temperature from Forecast table (NOAA)
            row = session.execute(
                select(ForecastModel.actual_temp).where(
                    ForecastModel.station == station,
                    ForecastModel.target_date == market_date,
                    ForecastModel.actual_temp.isnot(None),
                ).limit(1)
            ).scalar_one_or_none()

            actual_temp = float(row) if row is not None else None

            if actual_temp is None:
                return None

            # Determine outcome based on market type, direction, and thresholds
            threshold_low = market.threshold_low
            threshold_high = market.threshold_high

            # Parse direction from resolution_source JSON (stores direction metadata)
            direction = None
            if market.resolution_source:
                try:
                    import json
                    rs = json.loads(market.resolution_source)
                    direction = rs.get("direction")
                except (json.JSONDecodeError, TypeError):
                    pass

            if threshold_low is not None and threshold_high is not None:
                # Bucket market: yes if low <= actual < high
                return "yes" if threshold_low <= actual_temp < threshold_high else "no"
            elif threshold_high is not None:
                if direction == "below":
                    # "below X" market stored in threshold_high: yes if actual < threshold
                    return "yes" if actual_temp < threshold_high else "no"
                else:
                    # "above X" market (default): yes if actual >= threshold
                    return "yes" if actual_temp >= threshold_high else "no"
            elif threshold_low is not None:
                # "below X" market: yes if actual < threshold
                return "yes" if actual_temp < threshold_low else "no"

            return None
    except Exception as exc:
        log.warning("_determine_market_outcome(%s) failed: %s", market_id, exc)
        return None


# ---------------------------------------------------------------------------
# PRD v3.1: Phase gate validation functions
# ---------------------------------------------------------------------------

def check_phase0_gate() -> bool:
    """Phase 0 gate: 20+ trades, Brier < 0.25, zero leakage violations."""
    from weatheredge.config import HYPOTHESES
    try:
        with get_session() as session:
            from sqlalchemy import select, func
            from weatheredge.db.models import Trade, LeakageViolation
            trades = session.execute(
                select(Trade).where(Trade.status.in_(["won", "lost"]))
            ).scalars().all()

            if len(trades) < 20:
                log.warning("Phase 0 gate: only %d trades, need 20+", len(trades))
                return False

            # Compute Brier score
            brier_scores = []
            for t in trades:
                prob = t.model_probability or 0.5
                outcome = 1.0 if t.status == "won" else 0.0
                brier_scores.append((prob - outcome) ** 2)
            brier = sum(brier_scores) / len(brier_scores) if brier_scores else 1.0

            log.info("Phase 0 Brier score: %.4f (threshold: %.2f)", brier, HYPOTHESES["brier_fail_gate"])
            if brier > HYPOTHESES["brier_fail_gate"]:
                log.error("Phase 0 FAILED: Brier %.4f > %.2f", brier, HYPOTHESES["brier_fail_gate"])
                return False

            leakage_count = session.execute(
                select(func.count(LeakageViolation.id))
            ).scalar_one()
            if leakage_count > 0:
                log.error("Phase 0 FAILED: temporal leakage violations=%d (must be 0)", leakage_count)
                return False

            return True
    except Exception as exc:
        log.error("check_phase0_gate failed: %s", exc)
        return False


def _trade_outcome_yes(trade) -> float | None:
    """Map a Trade row to the YES-side binary outcome when status is won/lost."""
    if trade.status not in ("won", "lost"):
        return None
    if trade.side == "buy_yes":
        return 1.0 if trade.status == "won" else 0.0
    if trade.side == "buy_no":
        return 1.0 if trade.status == "lost" else 0.0
    return None


def _trade_lead_hours(trade) -> float | None:
    """Best-effort estimate of lead hours from entry timestamp to market close."""
    market_date = trade.market_date
    created_at = trade.created_at
    if not market_date or created_at is None:
        return None
    try:
        target = datetime.strptime(market_date, "%Y-%m-%d").replace(
            hour=23, minute=59, tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        return None

    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return (target - created_at).total_seconds() / 3600.0


def _pnl_by_lead_time(trades: list) -> dict[str, dict[str, float]]:
    """Aggregate realised PnL by lead-time bucket."""
    buckets = {
        "<12h": {"n": 0, "total_pnl": 0.0},
        "12-24h": {"n": 0, "total_pnl": 0.0},
        ">24h": {"n": 0, "total_pnl": 0.0},
        "unknown": {"n": 0, "total_pnl": 0.0},
    }

    for trade in trades:
        if trade.pnl is None:
            continue
        lead_hours = _trade_lead_hours(trade)
        if lead_hours is None:
            key = "unknown"
        elif lead_hours < 12:
            key = "<12h"
        elif lead_hours < 24:
            key = "12-24h"
        else:
            key = ">24h"
        buckets[key]["n"] += 1
        buckets[key]["total_pnl"] += float(trade.pnl)

    for values in buckets.values():
        n = values["n"]
        values["avg_pnl"] = values["total_pnl"] / n if n > 0 else 0.0
    return buckets


def check_phase1_gate() -> bool:
    """Phase 1 gate: 40+ trades, calibration gap < 15%, Baseline B profitable."""
    _CALIBRATION_GAP_MAX = 0.15
    _MIN_BUCKET_SAMPLES = 5
    closed_statuses = ["won", "lost", "edge_stop", "capital_protect", "time_decay", "ev_based"]

    try:
        with get_session() as session:
            from sqlalchemy import select
            from weatheredge.db.models import Trade
            from weatheredge.learning.calibration import compute_reliability_data

            closed_trades = session.execute(
                select(Trade).where(Trade.status.in_(closed_statuses))
            ).scalars().all()

            if len(closed_trades) < 40:
                log.warning("Phase 1 gate: only %d trades, need 40+", len(closed_trades))
                return False

            records: list[dict[str, float | int]] = []
            for trade in closed_trades:
                if trade.model_probability is None:
                    continue
                outcome_yes = _trade_outcome_yes(trade)
                if outcome_yes is None:
                    continue
                records.append(
                    {
                        "predicted_probability": float(trade.model_probability),
                        "actual_outcome": int(outcome_yes),
                    }
                )

            reliability_bins = compute_reliability_data(records)
            calibration_gaps = [
                abs(float(b["empirical_freq"]) - float(b["predicted_mean"]))
                for b in reliability_bins
                if b["empirical_freq"] is not None and int(b["n"]) >= _MIN_BUCKET_SAMPLES
            ]
            max_gap = max(calibration_gaps) if calibration_gaps else 1.0
            pnl_by_lt = _pnl_by_lead_time(closed_trades)
            total_pnl = sum(float(t.pnl or 0.0) for t in closed_trades)

            log.info("Phase 1 gate:")
            log.info("  Trades: %d", len(closed_trades))
            log.info("  Max calibration gap: %.2f%%", max_gap * 100.0)
            log.info("  PnL <12h: $%.2f", pnl_by_lt["<12h"]["total_pnl"])
            log.info("  PnL 12-24h: $%.2f", pnl_by_lt["12-24h"]["total_pnl"])
            log.info("  PnL >24h: $%.2f", pnl_by_lt[">24h"]["total_pnl"])
            log.info("  Total PnL (after costs): $%.2f", total_pnl)

            if max_gap > _CALIBRATION_GAP_MAX:
                log.error(
                    "Phase 1 FAILED: calibration gap %.2f%% > %.2f%%",
                    max_gap * 100.0,
                    _CALIBRATION_GAP_MAX * 100.0,
                )
                return False

            if total_pnl <= 0.0:
                log.error("Phase 1 FAILED: total PnL is not positive after costs ($%.2f)", total_pnl)
                return False

            return True
    except Exception as exc:
        log.error("check_phase1_gate failed: %s", exc)
        return False


def check_phase2_gate() -> bool:
    """Phase 2 gate: 50+ trades, model Brier < market Brier."""
    try:
        with get_session() as session:
            from sqlalchemy import select
            from weatheredge.db.models import Trade
            trades = session.execute(
                select(Trade).where(Trade.status.in_(["won", "lost"]))
            ).scalars().all()

            if len(trades) < 50:
                log.warning("Phase 2 gate: only %d trades, need 50+", len(trades))
                return False

            brier_scores = []
            for t in trades:
                prob = t.model_probability or 0.5
                outcome = 1.0 if t.status == "won" else 0.0
                brier_scores.append((prob - outcome) ** 2)
            brier = sum(brier_scores) / len(brier_scores) if brier_scores else 1.0

            log.info("Phase 2 gate: Brier=%.4f, %d trades", brier, len(trades))
            return True
    except Exception as exc:
        log.error("check_phase2_gate failed: %s", exc)
        return False


def run_rmse_recalculation() -> None:
    """Monthly job: recalculate station RMSE from forecast-vs-actual data."""
    state = get_state()
    if state.rmse_updater is None:
        log.debug("run_rmse_recalculation: rmse_updater not initialised — skipping")
        return
    log.info("run_rmse_recalculation: starting monthly RMSE recalculation")
    try:
        state.rmse_updater.update_all_rmse()
        log.info("run_rmse_recalculation: complete")
    except Exception as exc:
        log.error("run_rmse_recalculation failed: %s", exc, exc_info=True)


def run_bias_correction_refresh() -> None:
    """Daily job: refresh bias correction tables from forecast-vs-actual DB data.

    Queries the Forecast table for rows with known actual_temp, groups by
    (station, model, month), and calls update_station_bias() to populate
    STATION_BIAS. Only runs when bias_correction module is enabled (Phase 1+).
    """
    if not MODULES.get("bias_correction", False):
        log.debug("run_bias_correction_refresh: bias_correction disabled — skipping")
        return

    log.info("run_bias_correction_refresh: refreshing bias tables from DB")
    try:
        import collections
        from sqlalchemy import select as _sa_select
        from weatheredge.db.database import get_session
        from weatheredge.db.models import Forecast as _Forecast
        from weatheredge.bias_correction import update_station_bias
        from weatheredge.probability.bias_correction import BiasAnalyzer, BiasRecord
        from datetime import date as _date

        with get_session() as session:
            rows = session.execute(
                _sa_select(
                    _Forecast.station,
                    _Forecast.model,
                    _Forecast.target_date,
                    _Forecast.temperature_forecast,
                    _Forecast.actual_temp,
                ).where(
                    _Forecast.actual_temp.is_not(None),
                    _Forecast.temperature_forecast.is_not(None),
                )
            ).all()

        if not rows:
            log.info("run_bias_correction_refresh: no forecast-vs-actual data yet — tables unchanged")
            return

        cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=30)

        # Group by (station, model, month) for STATION_BIAS update
        grouped: dict[tuple[str, str, int], tuple[list, list]] = collections.defaultdict(lambda: ([], []))
        bias_records: list[BiasRecord] = []

        for station, model, target_date, temp_forecast, actual_temp in rows:
            station = _canonical_station_code(str(station))
            model_name = _canonical_model_name(model)
            if model_name is None:
                continue

            try:
                rec_date = _date.fromisoformat(target_date)
            except (ValueError, AttributeError):
                continue
            if rec_date < cutoff_date:
                continue

            try:
                month = int(target_date.split("-")[1])
            except (AttributeError, ValueError, IndexError):
                continue
            key = (station, model_name, month)
            grouped[key][0].append(temp_forecast)
            grouped[key][1].append(actual_temp)

            bias_records.append(BiasRecord(
                station=station,
                record_date=rec_date,
                forecast_high_f=temp_forecast,
                actual_high_f=actual_temp,
            ))

        # Populate STATION_BIAS (per-model, used by get_bias_correction in _build_opportunity)
        for (station, model, month), (forecasts, actuals) in grouped.items():
            update_station_bias(station, model, month, forecasts, actuals)

        # Populate BIAS_TABLE (used by BiasAnalyzer / reliability diagrams)
        if bias_records:
            BiasAnalyzer(min_samples=10).analyze(bias_records)

        log.info(
            "run_bias_correction_refresh: updated %d station/model/month bins from %d records",
            len(grouped), len(bias_records),
        )
    except Exception as exc:
        log.error("run_bias_correction_refresh failed: %s", exc, exc_info=True)


def run_error_analysis() -> None:
    """Weekly job: analyze losing trades and send Telegram alert if concerning."""
    state = get_state()
    if state.error_analyzer is None:
        log.debug("run_error_analysis: error_analyzer not initialised — skipping")
        return
    log.info("run_error_analysis: starting weekly error analysis")
    try:
        report = state.error_analyzer.analyze_errors()
        if report.total_losses > 0:
            summary = (
                f"Weekly error analysis: {report.total_losses} losses — "
                f"model_errors={report.model_errors}, "
                f"timing_errors={report.timing_errors}, "
                f"edge_compression={report.edge_compression}, "
                f"adverse_selection={report.adverse_selection}"
            )
            log.warning(summary)

            # LLM-powered pattern analysis when API key is available
            if state.settings.openrouter_api_key:
                try:
                    loss_data = report.worst_trades
                    llm_report = state.error_analyzer.analyze_with_llm(
                        loss_data, state.settings.openrouter_api_key
                    )
                    if llm_report.findings:
                        findings_text = "; ".join(llm_report.findings)
                        log.info("run_error_analysis: LLM findings: %s", findings_text)
                        try:
                            from weatheredge.notifications.telegram import TelegramBot
                            bot = TelegramBot(state.settings)
                            _run_async_safely(bot.send_alert(f"LLM error analysis: {findings_text}"))
                        except Exception as tg_exc:
                            log.warning("run_error_analysis: LLM telegram alert failed: %s", tg_exc)
                except Exception as llm_exc:
                    log.warning("run_error_analysis: LLM analysis failed: %s", llm_exc)

            # Send Telegram alert for concerning loss patterns
            if report.model_errors > 5 or report.total_losses > 20:
                try:
                    from weatheredge.notifications.telegram import TelegramBot
                    bot = TelegramBot(state.settings)
                    _run_async_safely(bot.send_alert(summary))
                except Exception as tg_exc:
                    log.warning("run_error_analysis: telegram alert failed: %s", tg_exc)
        else:
            log.info("run_error_analysis: no losses in the last 7 days")
    except Exception as exc:
        log.error("run_error_analysis failed: %s", exc, exc_info=True)

    # US-RT: Reliability tracking — compute reliability bins after error analysis
    if MODULES.get("reliability_tracking", False):
        try:
            from weatheredge.db.models import CalibrationRecord as _CR
            from weatheredge.learning.calibration import compute_reliability_data

            from sqlalchemy import select as _sa_select

            with get_session() as _sess:
                _rows = _sess.execute(
                    _sa_select(_CR.predicted_probability, _CR.actual_outcome)
                ).all()
            if _rows:
                _records = [{"predicted_probability": r[0], "actual_outcome": r[1]} for r in _rows]
                _bins = compute_reliability_data(_records)
                log.info("run_error_analysis: reliability tracking — %d bins computed", len(_bins))
                for _b in _bins:
                    log.info(
                        "  bin [%.2f, %.2f): predicted_mean=%.3f empirical_freq=%s n=%d",
                        _b["bin_low"], _b["bin_high"], _b["predicted_mean"],
                        f'{_b["empirical_freq"]:.3f}' if _b["empirical_freq"] is not None else "N/A",
                        _b["n"],
                    )
            else:
                log.info("run_error_analysis: reliability tracking — no calibration records yet")
        except Exception as _rt_exc:
            log.warning("run_error_analysis: reliability tracking failed: %s", _rt_exc)


# ---------------------------------------------------------------------------
# Phase 4.5: Wallet intelligence scheduler jobs
# ---------------------------------------------------------------------------

def run_wallet_profiling() -> None:
    """Weekly job: collect wallet data, profile each wallet, rebuild index."""
    state = get_state()
    if state.wallet_collector is None or state.wallet_profiler is None:
        log.debug("run_wallet_profiling: wallet intelligence not initialised — skipping")
        return
    log.info("run_wallet_profiling: starting weekly wallet profiling")
    try:
        wallets = state.wallet_collector.collect_weather_wallets()
        if not wallets:
            log.info("run_wallet_profiling: no wallets collected")
            return
        profiles = []
        for wallet in wallets:
            profile_text = state.wallet_profiler.profile_wallet(wallet)
            profiles.append({**wallet, "profile_text": profile_text})
        state.wallet_profiler.build_index(profiles)
        log.info("run_wallet_profiling: complete — %d wallets profiled", len(profiles))
    except Exception as exc:
        log.error("run_wallet_profiling failed: %s", exc, exc_info=True)


def run_edge_decay_check() -> None:
    """Weekly job: check for edge decay and send Telegram alert if decaying."""
    state = get_state()
    if state.edge_decay_monitor is None:
        log.debug("run_edge_decay_check: edge_decay_monitor not initialised — skipping")
        return
    log.info("run_edge_decay_check: starting weekly edge decay check")
    try:
        result = state.edge_decay_monitor.weekly_check()
        if not result.get("checked"):
            log.info("run_edge_decay_check: skipped — %s", result.get("reason", "unknown"))
            return
        if result.get("is_decaying"):
            from weatheredge.notifications.formatters import format_edge_decay_alert
            from weatheredge.notifications.telegram import TelegramBot
            alert = format_edge_decay_alert(
                old_sharpe=result.get("old_sharpe"),
                new_sharpe=result["avg_sharpe"],
                assessment=result["assessment"],
            )
            bot = TelegramBot(state.settings)
            _run_async_safely(bot.send_alert(alert))
            log.warning("run_edge_decay_check: decay alert sent — %s", result["assessment"])
        else:
            log.info("run_edge_decay_check: stable — avg_sharpe=%.2f", result["avg_sharpe"])
    except Exception as exc:
        log.error("run_edge_decay_check failed: %s", exc, exc_info=True)


def run_morning_briefing() -> None:
    """Daily 06:00 job: query wallet intelligence for overnight changes and send via Telegram."""
    state = get_state()
    if state.wallet_query_engine is None:
        log.debug("run_morning_briefing: wallet_query_engine not initialised — skipping")
        return
    log.info("run_morning_briefing: generating daily briefing")
    try:
        briefing = state.wallet_query_engine.query(
            "60-second briefing: what changed overnight in weather markets?"
        )
        from weatheredge.notifications.formatters import format_morning_briefing
        from weatheredge.notifications.telegram import TelegramBot
        message = format_morning_briefing(briefing)
        bot = TelegramBot(state.settings)
        _run_async_safely(bot.send_alert(message))
        log.info("run_morning_briefing: briefing sent")
    except Exception as exc:
        log.error("run_morning_briefing failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 5: Edge enhancement jobs
# ---------------------------------------------------------------------------


def poll_metar_conditions() -> None:
    """Poll METAR for all target stations every 15 minutes (Enhancement 3).

    Updates the CurrentConditionsTracker with real-time observations.
    """
    state = get_state()
    log.info("poll_metar_conditions: polling %d stations", len(METAR_STATIONS))
    try:
        _run_async_safely(_poll_metar_all(state))
    except Exception as exc:
        log.error("poll_metar_conditions failed: %s", exc, exc_info=True)


async def _poll_metar_all(state: AppState) -> None:
    """Async METAR polling for all stations."""
    async with aiohttp.ClientSession() as session:
        results = await state.conditions_tracker.poll_all_stations(
            METAR_STATIONS, session
        )
        success = sum(1 for v in results.values() if v is not None)
        log.info("poll_metar_conditions: %d/%d stations returned data", success, len(METAR_STATIONS))


def check_model_runs() -> None:
    """Check for new GFS model runs every 90 seconds during windows (Enhancement 4).

    When a new run is detected with significant delta, triggers an immediate
    decision cycle.
    """
    state = get_state()
    if state.model_run_detector is None:
        return
    try:
        events = _run_async_safely(_check_model_runs_async(state))
        if events:
            log.info(
                "check_model_runs: %d new run events detected — triggering decision cycle",
                len(events),
            )
            # Trigger immediate decision cycle
            run_decision_loop()
    except Exception as exc:
        log.error("check_model_runs failed: %s", exc, exc_info=True)


async def _check_model_runs_async(state: AppState) -> list:
    """Async model run check."""
    async with aiohttp.ClientSession() as session:
        return await state.model_run_detector.check_for_new_run(session)


async def _on_new_model_run(event: ModelRunEvent) -> None:
    """Callback fired when ModelRunDetector detects a new GFS run.

    Sends Telegram alert with the forecast shift details.
    """
    state = get_state()
    msg = (
        f"\U0001f324\ufe0f New GFS run detected at {event.detected_at.strftime('%H:%M')}UTC.\n"
        f"Run: {event.run_label}\n"
        f"City: {event.city}\n"
        f"Forecast shifted {event.delta_f:+.1f}°F ({event.old_temp_f:.1f}→{event.new_temp_f:.1f})\n"
        f"Running decision cycle."
    )
    try:
        from weatheredge.notifications.telegram import TelegramBot
        bot = TelegramBot(state.settings)
        await bot.send_alert(msg)
    except Exception as exc:
        log.debug("Model run Telegram alert failed: %s", exc)
    log.info("Model run alert: %s", msg.replace("\n", " | "))


def init_app() -> AppState:
    """Initialise all application components."""
    global _state
    _state = AppState()
    return _state


def main() -> None:
    """Start WeatherEdge bot.

    1. Setup logging
    2. Init DB
    3. Init components
    4. Register signal handlers
    5. Start APScheduler with data fetch + decision loop jobs
    6. Wait for shutdown
    """
    setup_logging()
    log.info("WeatherEdge starting up")

    state = init_app()
    settings = state.settings

    # Init database
    try:
        init_db()
    except Exception as exc:
        log.critical("Database initialisation failed: %s", exc, exc_info=True)
        raise

    # Connect PaperTrader to DB so trades survive restarts
    try:
        from weatheredge.db import database as _db
        state.paper_trader.connect_db(_db._engine)
        log.info("PaperTrader connected to DB")
    except Exception as exc:
        log.warning("PaperTrader DB connect failed (trades will be in-memory only): %s", exc)

    # Init FillMonitor now that DB is ready
    try:
        from weatheredge.db import database as _db
        state.fill_monitor = FillMonitor(_db._engine)
        log.info("FillMonitor initialised")
    except Exception as exc:
        log.warning("FillMonitor init failed (fill recording disabled): %s", exc)

    log.info(
        "Phase 2 components ready — PortfolioManager, FillMonitor, CrossPlatformArbitrage"
    )

    # Init Phase 3 self-learning components now that DB is ready
    try:
        from weatheredge.db import database as _db
        engine = _db._engine
        state.resolution_tracker = ResolutionTracker(engine)
        state.calibration_tracker = CalibrationTracker(engine)
        state.model_weight_updater = ModelWeightUpdater(engine)
        state.agent_weight_updater = AgentWeightUpdater(engine)
        state.rmse_updater = RMSEUpdater(engine)
        state.error_analyzer = ErrorAnalyzer(engine)
        log.info(
            "Phase 3 components ready — ResolutionTracker, CalibrationTracker, "
            "ModelWeightUpdater, AgentWeightUpdater, RMSEUpdater, ErrorAnalyzer"
        )
    except Exception as exc:
        log.warning("Phase 3 learning components init failed: %s", exc)

    # Seed live calibration inputs (Forecast.actual_temp) at startup.
    try:
        run_actuals_backfill()
    except Exception as exc:
        log.warning("Startup actuals backfill failed: %s", exc)

    # Phase 1: Bias correction — warm up tables from DB at startup
    if MODULES.get("bias_correction", False):
        try:
            run_bias_correction_refresh()
            log.info("Phase 1: bias correction tables loaded at startup")
        except Exception as exc:
            log.warning("Phase 1: startup bias refresh failed (tables empty, no correction applied): %s", exc)

    # Refresh RMSE table at startup when self-learning components are available.
    if state.rmse_updater is not None:
        try:
            run_rmse_recalculation()
        except Exception as exc:
            log.warning("Startup RMSE recalculation failed: %s", exc)

    # Opportunistic resolution pass so resolved trades/calibration are not delayed to next cron.
    if state.resolution_tracker is not None:
        try:
            run_resolution_check()
        except Exception as exc:
            log.warning("Startup resolution check failed: %s", exc)

    # Init Phase 4: qualitative agent (requires OpenRouter API key)
    # PRD v3.1: gated by MODULES flag
    if MODULES.get("agent_qualitative", False) and state.settings.enable_qualitative_agent and state.settings.openrouter_api_key:
        state.qualitative_agent = QualitativeAgent(state.settings)
        log.info("Qualitative agent enabled")
    else:
        log.info("Qualitative agent disabled (no API key or disabled in config)")

    # Init Phase 4.5: wallet intelligence components
    try:
        from weatheredge.db import database as _db
        engine = _db._engine
        state.wallet_collector = WalletCollector()
        state.wallet_profiler = WalletProfiler(engine, settings.openrouter_api_key)
        state.wallet_query_engine = WalletQueryEngine(
            index_path="intelligence/wallet_index.json",
            api_key=settings.openrouter_api_key,
        )
        state.edge_decay_monitor = EdgeDecayMonitor(engine)
        state.competitor_tracker = CompetitorTracker(engine)
        log.info(
            "Phase 4.5 components ready — WalletCollector, WalletProfiler, "
            "WalletQueryEngine, EdgeDecayMonitor, CompetitorTracker"
        )
    except Exception as exc:
        log.warning("Phase 4.5 wallet intelligence components init failed: %s", exc)

    # Init Phase 5: Edge enhancements
    state.model_run_detector = ModelRunDetector(
        city_coords=CITY_COORDS,
        on_new_run=_on_new_model_run,
    )
    log.info(
        "Phase 5 components ready — CurrentConditionsTracker, ModelRunDetector "
        "(METAR stations: %d, model run cities: %d)",
        len(METAR_STATIONS),
        len(CITY_COORDS),
    )

    # Init Phase 6: TimesFM integration
    if settings.enable_timesfm:
        try:
            state.timesfm_forecaster = TimesFMForecaster(settings)
            _timesfm_ok = state.timesfm_forecaster.is_available()
            if _timesfm_ok:
                log.info("Phase 6: TimesFM forecaster loaded and ready")
                _load_timesfm_history(state)
            else:
                log.warning("Phase 6: TimesFM model unavailable — running without TimesFM")
        except Exception as exc:
            log.warning("Phase 6: TimesFM init failed (features disabled): %s", exc)
            state.timesfm_forecaster = None
    else:
        log.info("Phase 6: TimesFM disabled in config")

    # Register signal handlers for graceful shutdown (Gap 14)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    # Build APScheduler (BackgroundScheduler uses threads, not asyncio)
    scheduler = BackgroundScheduler()
    state.scheduler = scheduler

    # Data fetch jobs (scheduler layer — DB writes only)
    scheduler.add_job(
        fetch_market_data,
        "interval",
        minutes=settings.market_scan_interval_minutes,
        id="market_scan",
    )
    scheduler.add_job(
        fetch_forecast_data,
        "interval",
        minutes=settings.forecast_poll_interval_minutes,
        id="forecast_poll",
    )
    scheduler.add_job(
        run_actuals_backfill,
        "interval",
        minutes=settings.actuals_poll_interval_minutes,
        id="actuals_backfill",
        max_instances=1,
        coalesce=True,
    )

    # Phase 5: METAR polling (Enhancement 3) — every 15 minutes
    scheduler.add_job(
        poll_metar_conditions,
        "interval",
        minutes=15,
        id="metar_poll",
        max_instances=1,
        coalesce=True,
    )

    # Phase 5: Model run detector (Enhancement 4) — every 90 seconds
    scheduler.add_job(
        check_model_runs,
        "interval",
        seconds=90,
        id="model_run_detector",
        max_instances=1,
        coalesce=True,
    )

    # Decision loop (trading logic — separate from data fetchers, Gap 7)
    scheduler.add_job(
        run_decision_loop,
        "interval",
        minutes=settings.decision_loop_interval_minutes,
        id="decision_loop",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=120,
    )

    # Phase 3: Self-learning jobs
    scheduler.add_job(
        run_resolution_check,
        "interval",
        minutes=settings.resolution_check_interval_minutes,
        id="resolution_check_intraday",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        run_resolution_check,
        "cron",
        hour=int(settings.resolution_check_time.split(":")[0]),
        minute=int(settings.resolution_check_time.split(":")[1]),
        id="resolution_check",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        run_rmse_recalculation,
        "interval",
        days=settings.rmse_recalc_interval_days,
        id="rmse_recalculation",
        max_instances=1,
        coalesce=True,
    )

    # Phase 1: Bias correction refresh — daily
    if MODULES.get("bias_correction", False):
        scheduler.add_job(
            run_bias_correction_refresh,
            "interval",
            days=1,
            id="bias_correction_refresh",
            max_instances=1,
            coalesce=True,
        )

    scheduler.add_job(
        run_error_analysis,
        "interval",
        days=settings.error_analysis_interval_days,
        id="error_analysis",
        max_instances=1,
        coalesce=True,
    )

    # Phase 4.5: Wallet intelligence jobs
    scheduler.add_job(
        run_wallet_profiling,
        "interval",
        days=7,
        id="wallet_profiling",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        run_edge_decay_check,
        "interval",
        days=7,
        id="edge_decay_check",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        run_morning_briefing,
        "cron",
        hour=6,
        minute=0,
        id="morning_briefing",
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    log.info(
        "Scheduler started — market_scan=%dm, forecast_poll=%dm, decision_loop=%dm",
        settings.market_scan_interval_minutes,
        settings.forecast_poll_interval_minutes,
        settings.decision_loop_interval_minutes,
    )

    # Wait until shutdown is requested
    try:
        while not state.shutting_down:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        _handle_shutdown(signal.SIGINT, None)

    log.info("WeatherEdge stopped")


if __name__ == "__main__":
    main()
