"""End-to-end pipeline tests for WeatherEdge.

Tests the full flow from market scan → opportunity building → strategy signals
→ portfolio aggregation → trade execution → DB persistence → restart recovery.
Uses realistic market IDs, forecast data, and strategy interactions.
"""

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from sqlalchemy import create_engine, select
from weatheredge.agents.base import MarketOpportunity, Signal, TradeOrder
from weatheredge.data.market_parser import ParsedMarket
from weatheredge.execution.paper_trader import Order, PaperTrader
from weatheredge.db.database import get_session, init_db
from weatheredge.db.models import Base, Trade


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_engine():
    """In-memory SQLite engine with all tables created."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def state():
    """Fresh AppState for each test."""
    from weatheredge.main import init_app, get_state
    init_app()
    return get_state()


def _make_forecast(station, gfs_temp=75.0, ecmwf_temp=None, hrrr_temp=None, ensemble=None):
    """Build a forecast dict matching _read_latest_forecasts output format."""
    if ensemble is None:
        ensemble = [gfs_temp + i * 0.3 - 4.5 for i in range(31)]
    return {
        "temperature_forecast": gfs_temp,
        "ecmwf_temp": ecmwf_temp,
        "hrrr_temp": hrrr_temp,
        "ensemble_members": ensemble,
    }


def _make_forecast_rows(station, gfs_temp=75.0, ecmwf_temp=None, hrrr_temp=None, ensemble=None):
    """Create mock DB forecast rows for multiple models."""
    ts = datetime.now(timezone.utc)
    rows = []

    gfs = MagicMock()
    gfs.station = station
    gfs.temperature_forecast = gfs_temp
    gfs.ensemble_members = None
    gfs.model = "gfs"
    gfs.forecast_timestamp = ts
    rows.append(gfs)

    if ecmwf_temp is not None:
        ecmwf = MagicMock()
        ecmwf.station = station
        ecmwf.temperature_forecast = ecmwf_temp
        ecmwf.ensemble_members = None
        ecmwf.model = "ecmwf"
        ecmwf.forecast_timestamp = ts
        rows.append(ecmwf)

    if ensemble is None:
        ensemble = [gfs_temp + i * 0.3 - 4.5 for i in range(31)]
    ens = MagicMock()
    ens.station = station
    ens.temperature_forecast = gfs_temp
    ens.ensemble_members = json.dumps(ensemble)
    ens.model = "ensemble"
    ens.forecast_timestamp = ts
    rows.append(ens)

    if hrrr_temp is not None:
        hrrr = MagicMock()
        hrrr.station = station
        hrrr.temperature_forecast = hrrr_temp
        hrrr.ensemble_members = None
        hrrr.model = "hrrr"
        hrrr.forecast_timestamp = ts
        rows.append(hrrr)

    return rows


def _setup_polymarket_metadata(state, market_id, station, threshold, direction="above",
                                date=None, bucket_low=None, bucket_high=None):
    """Register a Polymarket market in the metadata cache."""
    if date is None:
        date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    city_map = {"KNYC": "nyc", "KMDW": "chicago", "KMIA": "miami", "KDEN": "denver", "KLAX": "lax"}
    city = city_map.get(station, "unknown")
    parsed = ParsedMarket(
        city=city, station=station, date=date,
        market_type="daily_high", threshold=threshold, direction=direction,
        threshold_low=bucket_low, threshold_high=bucket_high,
    )
    state.market_metadata[market_id] = {
        "market_id": market_id,
        "platform": "polymarket",
        "resolution_source": {"station": station, "type": "daily_high", "source": "parsed_from_title"},
        "parsed": parsed,
    }


def _patch_forecasts_and_build(forecast_rows):
    """Context manager that patches get_session for _read_latest_forecasts."""
    mock_sess = MagicMock()
    mock_db = MagicMock()
    mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = forecast_rows
    mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
    mock_sess.return_value.__exit__ = MagicMock(return_value=False)
    return patch("weatheredge.main.get_session", mock_sess)


# ===========================================================================
# 1. METAR override: observed temp exceeds threshold → buy YES aggressively
# ===========================================================================

class TestMETAROverrideBuyYes:
    """When observed temp already exceeds the threshold, METAR fires probability=0.99."""

    def test_metar_override_high_edge_trade(self, state):
        market_id = "0xmetar_hot_day"
        _setup_polymarket_metadata(state, market_id, "KMDW", threshold=50.0, direction="above",
                                    date=datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        # METAR: Chicago observed 53°F, threshold 50°F above → override fires.
        # yes_price=0.62: >= $0.10 min (FIX 3) and gross edge 0.37 <= 0.40 cap (FIX 2).
        with patch.object(state.conditions_tracker, "metar_probability_override", return_value=0.99):
            forecasts = {"KMDW": _make_forecast("KMDW", gfs_temp=54.0, ecmwf_temp=55.0)}
            from weatheredge.main import _build_opportunity
            opp = _build_opportunity(market_id, {"yes_price": 0.62}, forecasts)

        assert opp is not None
        assert opp.model_probability == 0.99
        # Edge: prob 0.99 minus price 0.62 = gross 0.37, net ~0.355
        assert opp.net_edge > 0.30

        # Aggressive strategy should signal buy_yes (lower edge threshold than conservative)
        from weatheredge.agents.strategies.aggressive import aggressive_strategy
        sigs = aggressive_strategy(opp, [], state.settings)
        # Aggressive requires lead_time_days <= 2, net_edge >= 0.08 → should fire
        assert len(sigs) >= 1
        assert sigs[0].side == "buy_yes"


# ===========================================================================
# 2. METAR override: temp far below threshold late in day → skip or buy NO
# ===========================================================================

class TestMETARBelowThreshold:
    """Temp far below threshold with 'above' direction → low probability."""

    def test_low_temp_no_metar_override(self, state):
        market_id = "0xmetar_cold_day"
        _setup_polymarket_metadata(state, market_id, "KDEN", threshold=80.0, direction="above",
                                    date=datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        # No METAR override (temp hasn't exceeded threshold)
        with patch.object(state.conditions_tracker, "metar_probability_override", return_value=None):
            forecasts = {"KDEN": _make_forecast("KDEN", gfs_temp=55.0, ecmwf_temp=54.0)}
            from weatheredge.main import _build_opportunity
            opp = _build_opportunity(market_id, {"yes_price": 0.80}, forecasts)

        # 55°F forecast vs 80°F threshold → very low probability → gross edge >50%
        # FIX 2 (v2): unrealistic edge now returns None (skips trade entirely)
        assert opp is None


# ===========================================================================
# 3. High edge bucket market: forecast clearly in one bucket, market mispriced
# ===========================================================================

class TestHighEdgeBucketMarket:
    """Forecast clearly falls in a specific bucket, market price is low → big edge."""

    def test_bucket_market_high_edge(self, state):
        market_id = "0xbucket_mispriced"
        # Bucket: 74-75°F, forecast is 74.5°F → should be high probability
        _setup_polymarket_metadata(state, market_id, "KNYC", threshold=None,
                                    bucket_low=74.0, bucket_high=75.0,
                                    date=(datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d"))

        with patch.object(state.conditions_tracker, "metar_probability_override", return_value=None):
            forecasts = {"KNYC": _make_forecast("KNYC", gfs_temp=74.5, ecmwf_temp=74.8)}
            from weatheredge.main import _build_opportunity
            opp = _build_opportunity(market_id, {"yes_price": 0.10}, forecasts)

        assert opp is not None
        assert opp.bucket_low == 74.0
        assert opp.bucket_high == 75.0
        # Midpoint threshold: 74.5, forecast at 74.5 → reasonable probability
        assert opp.model_probability > 0
        assert opp.market_price == 0.10


# ===========================================================================
# 4. Low edge market: forecast close to threshold, no trade expected
# ===========================================================================

class TestLowEdgeNoTrade:
    """When forecast is near threshold and price is fair, no trade should occur."""

    def test_fair_priced_market_no_trade(self, state):
        market_id = "0xfair_price"
        _setup_polymarket_metadata(state, market_id, "KNYC", threshold=75.0, direction="above")

        with patch.object(state.conditions_tracker, "metar_probability_override", return_value=None):
            # Forecast right at threshold → prob ~0.50, price ~0.50 → no edge
            forecasts = {"KNYC": _make_forecast("KNYC", gfs_temp=75.0, ecmwf_temp=75.0)}
            from weatheredge.main import _build_opportunity
            opp = _build_opportunity(market_id, {"yes_price": 0.50}, forecasts)

        assert opp is not None
        # Edge should be very small (close to zero)
        assert abs(opp.net_edge) < 0.15

        # Conservative strategy should NOT trade with such low edge
        from weatheredge.agents.strategies.conservative import conservative_strategy
        sig = conservative_strategy(opp, state.settings)
        assert sig.side == "no_trade"


# ===========================================================================
# 5. Kalshi prefix matching: KXHIGHDEN-26MAR30-T76 → KXHIGHDEN
# ===========================================================================

class TestKalshiPrefixMatch:
    """Full pipeline: Kalshi contract ID with no cached metadata matches via prefix."""

    def test_kalshi_prefix_to_trade_execution(self, state, db_engine):
        state.paper_trader.connect_db(db_engine)

        # Dynamic future Kalshi ticker date so the market isn't rejected as past
        _kalshi_date = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%y%b%d").upper()
        market_id = f"KXHIGHDEN-{_kalshi_date}-T76"
        # Clear any cached metadata to force prefix matching
        state.market_metadata.pop(market_id, None)

        with patch.object(state.conditions_tracker, "metar_probability_override", return_value=None):
            forecasts = {"KDEN": _make_forecast("KDEN", gfs_temp=80.0, ecmwf_temp=81.0)}
            from weatheredge.main import _build_opportunity
            # yes_price=0.50: model≈0.90, gross_edge≈0.40 ≤ 0.50 cap (FIX 2 won't zero it)
            opp = _build_opportunity(market_id, {"yes_price": 0.50}, forecasts)

        assert opp is not None
        assert opp.station == "KDEN"
        assert opp.platform == "kalshi"
        assert opp.threshold == 76.0
        assert opp.city == "Denver"
        # 80°F forecast vs 76°F threshold → high probability above
        assert opp.model_probability > 0.60
        assert opp.net_edge > 0.20

        # Execute trade via paper trader
        order = Order(
            market_id=market_id, side="buy_yes", price=0.20, amount=50.0,
            agent_name="test", model_probability=opp.model_probability,
            edge_at_entry=opp.net_edge, platform="kalshi",
        )
        result = state.paper_trader.place_order(order)
        assert result["filled"]

        # Verify trade persisted to DB
        with get_session(db_engine) as session:
            db_trade = session.execute(select(Trade).where(Trade.market_id == market_id)).scalar_one()
            assert db_trade.platform == "kalshi"
            assert db_trade.mode == "paper"
            assert db_trade.status == "open"
            assert db_trade.price == 0.20
            assert db_trade.amount == 50.0


# ===========================================================================
# 6. Polymarket UUID flow: full pipeline from parsed title to trade
# ===========================================================================

class TestPolymarketFullPipeline:
    """Full pipeline: Polymarket UUID → opportunity → strategies → trade → DB."""

    def test_polymarket_uuid_to_persisted_trade(self, state, db_engine):
        state.paper_trader.connect_db(db_engine)

        market_id = "0xabcdef1234567890abcdef1234567890abcdef12"
        _setup_polymarket_metadata(state, market_id, "KMIA", threshold=88.0, direction="above",
                                    date=(datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d"))

        with patch.object(state.conditions_tracker, "metar_probability_override", return_value=None):
            forecasts = {"KMIA": _make_forecast("KMIA", gfs_temp=92.0, ecmwf_temp=93.0)}
            from weatheredge.main import _build_opportunity
            opp = _build_opportunity(market_id, {"yes_price": 0.55}, forecasts)

        assert opp is not None
        assert opp.platform == "polymarket"
        assert opp.city == "Miami"
        assert opp.model_probability > 0.70

        # Execute trade
        order = Order(
            market_id=market_id, side="buy_yes", price=0.55, amount=40.0,
            agent_name="e2e_test", model_probability=opp.model_probability,
            edge_at_entry=opp.net_edge, platform="polymarket",
        )
        result = state.paper_trader.place_order(order)
        assert result["filled"]

        # Verify persistence
        with get_session(db_engine) as session:
            trades = session.execute(select(Trade).where(Trade.mode == "paper")).scalars().all()
            assert len(trades) == 1
            assert trades[0].market_id == market_id
            assert trades[0].platform == "polymarket"

        # Resolve as win
        state.paper_trader.resolve_position(market_id, "yes")
        assert len(state.paper_trader.open_positions) == 0

        with get_session(db_engine) as session:
            trade = session.execute(select(Trade).where(Trade.market_id == market_id)).scalar_one()
            assert trade.status == "won"
            assert trade.pnl > 0
            assert trade.resolved_at is not None


# ===========================================================================
# 7. Cross-platform arbitrage: same event priced differently
# ===========================================================================

class TestCrossPlatformArbitrage:
    """Same weather event on both platforms with a price gap → arb opportunity."""

    def test_arb_detection(self, state):
        from weatheredge.execution.cross_platform import CrossPlatformArbitrage

        arb = CrossPlatformArbitrage()

        poly_markets = [
            {"city": "NYC", "date": "2026-04-01", "threshold": 70.0, "yes_price": 0.65},
        ]
        kalshi_markets = [
            {"city": "NYC", "date": "2026-04-01", "threshold": 70.0, "yes_price": 0.50},
        ]

        opportunities = arb.scan(poly_markets, kalshi_markets)

        # Should detect the 15-cent spread
        assert len(opportunities) >= 1
        best = opportunities[0]
        assert best.spread >= 0.10
        assert best.city == "NYC"


# ===========================================================================
# 8. HRRR short-range dominance: <12h lead time, HRRR remains top-weighted
# ===========================================================================

class TestHRRRShortRange:
    """At <12h lead time for US station, HRRR should dominate the blend."""

    def test_hrrr_dominates_short_lead(self):
        from weatheredge.probability.blended import blended_probability

        # HRRR says 80°F, GFS says 70°F, ECMWF says 72°F
        # Threshold 75°F above → HRRR strongly above, others weakly below
        prob, model_used, details = blended_probability(
            threshold=75.0, direction="above", rmse=5.0,
            is_us_station=True, lead_time_hours=6.0,
            ensemble_temps=[70.0 + i * 0.2 for i in range(31)],  # centered at 73
            gfs_forecast_temp=70.0,
            ecmwf_forecast_temp=72.0,
            hrrr_forecast_temp=80.0,
        )

        assert details["weight_bucket"] == "short"
        assert details["weights"]["hrrr"] == 0.45
        assert details["weights"]["nbm"] == 0.25
        # HRRR at 80°F strongly pushes probability above 0.5
        assert prob > 0.55
        assert "hrrr" in model_used

    def test_long_lead_hrrr_reduced(self):
        """At >24h lead, HRRR weight drops sharply and NBM/GFS/ECMWF dominate."""
        from weatheredge.probability.blended import blended_probability

        prob, model_used, details = blended_probability(
            threshold=75.0, direction="above", rmse=5.0,
            is_us_station=True, lead_time_hours=72.0,
            ensemble_temps=[70.0 + i * 0.2 for i in range(31)],
            gfs_forecast_temp=70.0,
            ecmwf_forecast_temp=72.0,
            hrrr_forecast_temp=80.0,
        )

        assert details["weight_bucket"] == "long"
        assert details["weights"]["hrrr"] == 0.05
        assert details["weights"]["nbm"] == 0.35
        # With reduced HRRR weight, probability should be lower than short-range
        assert prob < 0.55


# ===========================================================================
# 9. Model run shift: forecast change triggers opportunity reassessment
# ===========================================================================

class TestModelRunShift:
    """A significant forecast shift should change the opportunity's edge."""

    def test_forecast_shift_changes_edge(self, state):
        market_id = "0xmodel_shift"
        _setup_polymarket_metadata(state, market_id, "KNYC", threshold=75.0, direction="above")

        with patch.object(state.conditions_tracker, "metar_probability_override", return_value=None):
            # Before shift: GFS says 73°F → below threshold
            forecasts_before = {"KNYC": _make_forecast("KNYC", gfs_temp=73.0, ecmwf_temp=73.0)}
            from weatheredge.main import _build_opportunity
            opp_before = _build_opportunity(market_id, {"yes_price": 0.40}, forecasts_before)

            # After shift: GFS revised up to 80°F (market price 0.60 to keep edge <50%)
            forecasts_after = {"KNYC": _make_forecast("KNYC", gfs_temp=80.0, ecmwf_temp=80.0)}
            opp_after = _build_opportunity(market_id, {"yes_price": 0.60}, forecasts_after)

        assert opp_before is not None
        assert opp_after is not None
        # Edge should increase significantly after the warm shift
        assert opp_after.net_edge > opp_before.net_edge
        assert opp_after.model_probability > opp_before.model_probability


# ===========================================================================
# 10. Multiple agents disagreeing: not enough consensus → no trade
# ===========================================================================

class TestAgentDisagreement:
    """When agents disagree, weighted score stays below threshold → no trade."""

    def test_disagreement_blocks_trade(self, state):
        from weatheredge.agents.portfolio_manager import PortfolioManager

        opp = MarketOpportunity(
            market_id="0xdisagree",
            platform="polymarket",
            city="NYC",
            station="KNYC",
            threshold=70.0,
            direction="above",
            market_price=0.50,
            model_probability=0.60,
            net_edge=0.08,
            forecast_temp=72.0,
            lead_time_days=1,
            gaussian_prob=0.60,
            ensemble_prob=0.58,
        )

        # Conservative says buy_yes (low confidence), Aggressive says no_trade
        signals = {
            "0xdisagree": [
                Signal(market_id="0xdisagree", side="buy_yes", confidence=0.40, agent_name="Conservative"),
                Signal(market_id="0xdisagree", side="no_trade", confidence=0.0, agent_name="Aggressive"),
                Signal(market_id="0xdisagree", side="no_trade", confidence=0.0, agent_name="MeanReversion"),
                Signal(market_id="0xdisagree", side="no_trade", confidence=0.0, agent_name="Qualitative"),
            ]
        }

        pm = state.portfolio_manager
        trade_orders = pm.aggregate_signals(
            signals_by_market=signals,
            opportunities={"0xdisagree": opp},
            bankroll=1000.0,
            open_positions=[],
            daily_pnl=0.0,
            total_pnl=0.0,
        )

        # Weighted score: Conservative=0.30*0.40=0.12, rest=0 → 0.12 < 0.50 threshold
        assert len(trade_orders) == 0

    def test_strong_consensus_triggers_trade(self, state):
        """When all agents agree with high confidence, trade should go through."""
        opp = MarketOpportunity(
            market_id="0xagree",
            platform="polymarket",
            city="Miami",
            station="KMIA",
            threshold=85.0,
            direction="above",
            market_price=0.10,
            model_probability=0.95,
            net_edge=0.83,
            forecast_temp=92.0,
            lead_time_days=1,
            gaussian_prob=0.95,
            ensemble_prob=0.93,
            # Bucket market so the April-2026 YES guard-rail (threshold-markets
            # only) does not short-circuit this consensus-pipeline test.
            bucket_low=83.0,
            bucket_high=87.0,
        )

        signals = {
            "0xagree": [
                Signal(market_id="0xagree", side="buy_yes", confidence=0.95, agent_name="Conservative"),
                Signal(market_id="0xagree", side="buy_yes", confidence=0.90, agent_name="Aggressive"),
                Signal(market_id="0xagree", side="buy_yes", confidence=0.85, agent_name="Qualitative"),
            ]
        }

        pm = state.portfolio_manager
        trade_orders = pm.aggregate_signals(
            signals_by_market=signals,
            opportunities={"0xagree": opp},
            bankroll=1000.0,
            open_positions=[],
            daily_pnl=0.0,
            total_pnl=0.0,
        )

        # Weighted score: 0.30*0.95 + 0.30*0.90 + 0.20*0.85 = 0.285+0.27+0.17 = 0.725 > 0.50
        assert len(trade_orders) == 1
        assert trade_orders[0].side == "buy_yes"
        assert trade_orders[0].market_id == "0xagree"


# ===========================================================================
# 11. Position already open: should not re-buy the same market
# ===========================================================================

class TestDuplicatePositionPrevention:
    """The bot should detect existing open positions and avoid doubling up."""

    def test_has_open_position_detection(self, db_engine):
        pt = PaperTrader(initial_bankroll=1000.0)
        pt.connect_db(db_engine)

        order = Order(
            market_id="0xduplicate_test", side="buy_yes", price=0.50,
            amount=100.0, platform="polymarket",
        )
        result = pt.place_order(order)
        assert result["filled"]
        assert pt.has_open_position("0xduplicate_test")
        assert not pt.has_open_position("0xother_market")

    def test_open_position_survives_restart(self, db_engine):
        """Trade persisted to DB should be loaded by a fresh PaperTrader."""
        # First trader places a trade
        pt1 = PaperTrader(initial_bankroll=1000.0)
        pt1.connect_db(db_engine)
        pt1.place_order(Order(
            market_id="0xsurvive_restart", side="buy_yes", price=0.30,
            amount=50.0, platform="polymarket",
        ))
        assert pt1.has_open_position("0xsurvive_restart")
        assert pt1.bankroll == 1000.0 - (0.30 * 50.0)  # 985.0

        # Simulate restart: new PaperTrader connects to same DB
        pt2 = PaperTrader(initial_bankroll=1000.0)
        pt2.connect_db(db_engine)

        # Open position should be restored
        assert pt2.has_open_position("0xsurvive_restart")
        assert len(pt2.open_positions) == 1
        assert pt2.open_positions[0]["market_id"] == "0xsurvive_restart"
        # Bankroll should reflect the open position's cost
        assert pt2.bankroll == 1000.0 - (0.30 * 50.0)

    def test_resolved_trade_not_reopened(self, db_engine):
        """A resolved (won) trade should appear in history, not open positions."""
        pt1 = PaperTrader(initial_bankroll=1000.0)
        pt1.connect_db(db_engine)
        pt1.place_order(Order(
            market_id="0xresolved_win", side="buy_yes", price=0.20,
            amount=100.0, platform="polymarket",
        ))
        pt1.resolve_position("0xresolved_win", "yes")

        # Simulate restart
        pt2 = PaperTrader(initial_bankroll=1000.0)
        pt2.connect_db(db_engine)

        assert not pt2.has_open_position("0xresolved_win")
        assert len(pt2.open_positions) == 0
        assert len(pt2.trade_history) == 1
        assert pt2.trade_history[0]["status"] == "won"
        assert pt2.trade_history[0]["pnl"] > 0


# ===========================================================================
# 12. Race condition: decision loop before market scan → graceful handling
# ===========================================================================

class TestRaceConditionGraceful:
    """Decision loop runs before market scan populates metadata cache → 0 opportunities, no crash."""

    def test_empty_metadata_cache_no_crash(self, state):
        # Clear all metadata to simulate first-run race condition
        state.market_metadata.clear()

        from weatheredge.main import _build_opportunity

        # Polymarket UUID with no metadata and no RESOLUTION_SOURCES match → None
        opp = _build_opportunity(
            "0xunknown_uuid_before_scan",
            {"yes_price": 0.50},
            {"KNYC": _make_forecast("KNYC", gfs_temp=75.0)},
        )
        assert opp is None  # Gracefully returns None, no crash

    def test_empty_price_data_no_crash(self, state):
        """No market prices at all → decision loop should handle gracefully."""
        # With no market_prices, the loop should just skip
        from weatheredge.main import _build_opportunity
        opp = _build_opportunity("0xno_prices", {}, {"KNYC": _make_forecast("KNYC")})
        assert opp is None


# ===========================================================================
# 13. DB persistence round-trip: place → resolve → restart → verify
# ===========================================================================

class TestDBPersistenceRoundTrip:
    """Full lifecycle: place trade, resolve, restart, check all state is correct."""

    def test_full_lifecycle(self, db_engine):
        pt = PaperTrader(initial_bankroll=500.0)
        pt.connect_db(db_engine)

        # Place 3 trades
        pt.place_order(Order(market_id="M1", side="buy_yes", price=0.20, amount=50.0, platform="polymarket"))
        pt.place_order(Order(market_id="M2", side="buy_yes", price=0.60, amount=30.0, platform="kalshi"))
        pt.place_order(Order(market_id="M3", side="buy_no", price=0.40, amount=40.0, platform="polymarket"))

        assert len(pt.open_positions) == 3
        # Entry fees booked into cost on Kalshi only:
        #   M1 poly @ 0.20 × 50 → cost = 10.00 (no fee)
        #   M2 kalshi @ 0.60 × 30 → stake 18 + kalshi_fee(0.6)*30 = 0.60 → cost 18.60
        #   M3 poly buy_no @ 0.40 × 40 → stake (1-0.40)*40 = 24.00 (no fee)
        cost_total = 10.0 + 18.60 + 24.0   # 52.60
        assert abs(pt.bankroll - (500.0 - cost_total)) < 0.01

        # Resolve M1 as win, M2 as loss
        pt.resolve_position("M1", "yes")  # Win: pnl = amount - cost = 50 - 10 = 40
        pt.resolve_position("M2", "no")   # Loss: pnl = -cost = -18.60

        assert len(pt.open_positions) == 1  # Only M3 remains
        assert len(pt.trade_history) == 2

        # Verify DB state
        with get_session(db_engine) as session:
            all_trades = session.execute(select(Trade)).scalars().all()
            assert len(all_trades) == 3

            won = [t for t in all_trades if t.status == "won"]
            lost = [t for t in all_trades if t.status == "lost"]
            still_open = [t for t in all_trades if t.status == "open"]

            assert len(won) == 1
            assert won[0].market_id == "M1"
            # Polymarket: pnl = amount - cost = 50 - 10 = 40.0
            assert abs(won[0].pnl - 40.0) < 0.01

            assert len(lost) == 1
            assert lost[0].market_id == "M2"
            # Kalshi loss burns full cost including entry fee: -18.60
            assert abs(lost[0].pnl - (-18.60)) < 0.01

            assert len(still_open) == 1
            assert still_open[0].market_id == "M3"

        # Simulate restart
        pt2 = PaperTrader(initial_bankroll=500.0)
        pt2.connect_db(db_engine)

        assert len(pt2.open_positions) == 1
        assert pt2.open_positions[0]["market_id"] == "M3"
        assert len(pt2.trade_history) == 2
        # Bankroll reconstructed as initial + resolved_pnl - open_costs
        # resolved_pnl = 40.0 (M1 win) + (-18.60) (M2 loss with entry fee) = 21.40
        # open_cost = 24.0 (M3 buy_no: (1-0.40)*40)
        # bankroll = 500 + 21.40 - 24.0 = 497.40
        assert abs(pt2.bankroll - 497.40) < 0.01


# ===========================================================================
# 14. Performance report with DB-backed trades
# ===========================================================================

class TestPerformanceReport:
    """Performance metrics should work correctly with DB-persisted trades."""

    def test_performance_after_trades(self, db_engine):
        pt = PaperTrader(initial_bankroll=1000.0)
        pt.connect_db(db_engine)

        # Place and resolve several trades
        pt.place_order(Order(market_id="W1", side="buy_yes", price=0.30, amount=100.0, platform="polymarket",
                             model_probability=0.80, edge_at_entry=0.48))
        pt.place_order(Order(market_id="W2", side="buy_yes", price=0.50, amount=50.0, platform="polymarket",
                             model_probability=0.60, edge_at_entry=0.08))
        pt.place_order(Order(market_id="L1", side="buy_yes", price=0.70, amount=30.0, platform="kalshi",
                             model_probability=0.80, edge_at_entry=0.08))

        pt.resolve_position("W1", "yes")  # Poly win: pnl = 100 - 30 = 70
        pt.resolve_position("W2", "yes")  # Poly win: pnl = 50 - 25 = 25
        # L1 Kalshi buy_yes @ 0.70, amount=30:
        #   stake 21.00 + kalshi_fee(0.70)*30 = 0.60 → cost 21.60
        #   Loss burns full cost (incl. entry fee): pnl = -21.60
        pt.resolve_position("L1", "no")

        report = pt.get_performance_report()
        assert report.total_trades == 3
        assert report.winning_trades == 2
        assert report.losing_trades == 1
        assert abs(report.win_rate - 2 / 3) < 0.01
        # W1 70 + W2 25 + L1 -21.60 → total 73.40
        assert abs(report.total_pnl - 73.40) < 0.01
        assert report.sharpe_ratio != 0.0
