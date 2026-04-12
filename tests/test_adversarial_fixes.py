"""Regression tests for adversarial review fixes.

Fix 1 (critical): Resolution pipeline wired into scheduled job
Fix 2 (high): station/market_date propagated through order pipeline
Fix 3 (high): Time-decay exits work for Kalshi markets (position fallback)
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from weatheredge.execution.paper_trader import Order, PaperTrader
from weatheredge.execution.exit_manager import ExitManager
from weatheredge.agents.base import TradeOrder, MarketOpportunity
from weatheredge.db.database import init_db, make_engine, get_session
from weatheredge.db.models import Trade, Market, Forecast
from weatheredge.learning.resolution_tracker import ResolutionTracker


# ---------------------------------------------------------------------------
# Fix 2: station/market_date propagation
# ---------------------------------------------------------------------------


class TestMetadataPropagation:
    """Verify station and market_date flow end-to-end through the order pipeline."""

    def test_order_dataclass_has_station_and_market_date(self):
        order = Order(
            market_id="m1", side="buy_yes", price=0.5, amount=10.0,
            station="KNYC", market_date="2026-04-01",
        )
        assert order.station == "KNYC"
        assert order.market_date == "2026-04-01"

    def test_trade_order_dataclass_has_station(self):
        to = TradeOrder(
            market_id="m1", side="buy_yes", price=0.5, amount=10.0,
            station="KMDW", market_date="2026-04-01",
        )
        assert to.station == "KMDW"
        assert to.market_date == "2026-04-01"

    def test_place_order_persists_station_and_market_date_in_memory(self):
        """Filled trade dict in open_positions must contain station and market_date."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(
            market_id="m1", side="buy_yes", price=0.40, amount=10.0,
            station="KNYC", market_date="2026-04-05",
        )
        result = trader.place_order(order)
        assert result["filled"] is True
        pos = trader.open_positions[0]
        assert pos["station"] == "KNYC"
        assert pos["market_date"] == "2026-04-05"

    def test_place_order_limit_persists_station_and_market_date(self):
        """Pending limit order dict must also contain station and market_date."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(
            market_id="m1", side="buy_yes", price=0.40, amount=10.0,
            station="KDEN", market_date="2026-04-06", order_type="limit",
        )
        result = trader.place_order(order)
        assert result.get("pending") is True
        pending = trader.open_orders[0]
        assert pending["station"] == "KDEN"
        assert pending["market_date"] == "2026-04-06"

    def test_station_market_date_persisted_to_db(self):
        """station and market_date are written to the Trade DB row."""
        engine = make_engine("sqlite:///:memory:")
        init_db(engine=engine)
        trader = PaperTrader(initial_bankroll=1000.0)
        trader.connect_db(engine)

        order = Order(
            market_id="m1", side="buy_yes", price=0.40, amount=10.0,
            station="KMIA", market_date="2026-04-07",
        )
        result = trader.place_order(order)
        assert result["filled"] is True
        db_id = result["trade"]["db_id"]

        with get_session(engine) as session:
            trade = session.get(Trade, db_id)
            assert trade.station == "KMIA"
            assert trade.market_date == "2026-04-07"

        engine.dispose()

    def test_station_market_date_restored_from_db(self):
        """On restart, open positions loaded from DB include station and market_date."""
        engine = make_engine("sqlite:///:memory:")
        init_db(engine=engine)

        # Insert a trade directly
        with get_session(engine) as session:
            session.add(Trade(
                market_id="m1", platform="kalshi", mode="paper",
                side="buy_yes", price=0.40, amount=10.0, cost=4.0,
                station="KAUS", market_date="2026-04-08", status="open",
            ))

        trader = PaperTrader(initial_bankroll=1000.0)
        trader.connect_db(engine)

        assert len(trader.open_positions) == 1
        pos = trader.open_positions[0]
        assert pos["station"] == "KAUS"
        assert pos["market_date"] == "2026-04-08"

        engine.dispose()

    def test_limit_fill_preserves_station_and_market_date(self):
        """When a limit order fills via check_limit_fills, the resulting position
        must contain station and market_date for risk cap enforcement."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(
            market_id="m1", side="buy_yes", price=0.50, amount=10.0,
            station="KNYC", market_date="2026-04-10", order_type="limit",
        )
        trader.place_order(order)
        assert len(trader.open_orders) == 1

        # Simulate price drop that triggers fill
        filled = trader.check_limit_fills({"m1": 0.30})
        assert filled == 1
        assert len(trader.open_positions) == 1

        pos = trader.open_positions[0]
        assert pos["station"] == "KNYC"
        assert pos["market_date"] == "2026-04-10"

    def test_risk_manager_sees_station_on_second_cycle(self):
        """After order placement, open_positions have station so risk caps work."""
        trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(
            market_id="m1", side="buy_yes", price=0.40, amount=10.0,
            station="KNYC", market_date="2026-04-05",
        )
        trader.place_order(order)

        # Simulate what RiskManager.check_region_exposure reads
        pos = trader.open_positions[0]
        station = pos.get("station", "") if isinstance(pos, dict) else getattr(pos, "station", "")
        assert station == "KNYC", "station must be visible to risk checks on subsequent cycles"

        market_date = pos.get("market_date", "") if isinstance(pos, dict) else getattr(pos, "market_date", "")
        assert market_date == "2026-04-05", "market_date must be visible to risk checks"


# ---------------------------------------------------------------------------
# Fix 3: Kalshi time-decay exits via position fallback
# ---------------------------------------------------------------------------


class TestKalshiTimDecayFallback:
    """_lead_time_hours falls back to position metadata when parsed is absent."""

    def test_position_fallback_when_no_parsed_metadata(self):
        """Kalshi market with no parsed metadata uses position's market_date/station."""
        pos = {"market_date": "2026-12-25", "station": "KNYC"}
        meta = {}  # no parsed metadata (typical for Kalshi)

        with patch(
            "weatheredge.execution.exit_manager.hours_until_market_close",
            return_value=10.0,
        ) as mock_huc:
            result = ExitManager._lead_time_hours("KALSHI-1", {"KALSHI-1": meta}, pos)

        mock_huc.assert_called_once_with("2026-12-25", "KNYC")
        assert result == pytest.approx(10.0)

    def test_position_fallback_station_only(self):
        """Position has station but market_date comes from parsed."""
        parsed = MagicMock()
        parsed.date = "2026-06-15"
        del parsed.station
        meta = {"parsed": parsed, "resolution_source": {}}
        pos = {"station": "KDEN", "market_date": ""}

        with patch(
            "weatheredge.execution.exit_manager.hours_until_market_close",
            return_value=7.0,
        ) as mock_huc:
            result = ExitManager._lead_time_hours("MKT-1", {"MKT-1": meta}, pos)

        # market_date from parsed, station from position fallback
        mock_huc.assert_called_once_with("2026-06-15", "KDEN")
        assert result == pytest.approx(7.0)

    def test_no_position_no_parsed_returns_none(self):
        """Neither parsed nor position metadata → None."""
        result = ExitManager._lead_time_hours("MKT-1", {}, None)
        assert result is None

    def test_position_empty_market_date_returns_none(self):
        """Position with empty market_date and no parsed → None."""
        pos = {"market_date": "", "station": "KNYC"}
        result = ExitManager._lead_time_hours("MKT-1", {}, pos)
        assert result is None

    def test_position_utc_fallback_when_no_station(self):
        """Position has market_date but no station → UTC end-of-day fallback."""
        pos = {"market_date": "2026-12-25", "station": ""}
        result = ExitManager._lead_time_hours("MKT-1", {}, pos)
        assert result is not None
        assert result >= 0.0

    def test_parsed_takes_precedence_over_position(self):
        """When parsed metadata exists, it takes precedence over position fields."""
        parsed = MagicMock()
        parsed.date = "2026-06-15"
        parsed.station = "KMDW"
        meta = {"parsed": parsed, "resolution_source": {}}
        pos = {"market_date": "2026-07-01", "station": "KDEN"}  # different values

        with patch(
            "weatheredge.execution.exit_manager.hours_until_market_close",
            return_value=4.0,
        ) as mock_huc:
            result = ExitManager._lead_time_hours("MKT-1", {"MKT-1": meta}, pos)

        # Should use parsed values, not position values
        mock_huc.assert_called_once_with("2026-06-15", "KMDW")
        assert result == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Fix 1: Resolution pipeline
# ---------------------------------------------------------------------------


@pytest.fixture()
def resolution_engine():
    """In-memory SQLite engine with tables and seed data for resolution tests."""
    engine = make_engine("sqlite:///:memory:")
    init_db(engine=engine)
    yield engine
    engine.dispose()


class TestResolutionPipeline:
    """run_resolution_check resolves past-date markets with known outcomes."""

    def test_determine_market_outcome_above_threshold_yes(self, resolution_engine):
        """Actual temp >= threshold → 'yes' for daily_high above market."""
        with get_session(resolution_engine) as session:
            session.add(Market(
                id="MKT-ABOVE", platform="polymarket", city="New York",
                station="KNYC", date="2026-03-01", market_type="daily_high",
                threshold_high=70.0,
            ))
            session.add(Forecast(
                station="KNYC", target_date="2026-03-01", model="gfs",
                forecast_timestamp=datetime(2026, 2, 28, tzinfo=timezone.utc),
                lead_time_hours=24, temperature_forecast=72.0, actual_temp=75.0,
            ))

        from weatheredge.main import _determine_market_outcome
        state = MagicMock()
        state.market_metadata = {}

        with patch("weatheredge.main.get_session", lambda: get_session(resolution_engine)):
            outcome = _determine_market_outcome(state, "MKT-ABOVE")

        assert outcome == "yes"

    def test_determine_market_outcome_above_threshold_no(self, resolution_engine):
        """Actual temp < threshold → 'no' for daily_high above market."""
        with get_session(resolution_engine) as session:
            session.add(Market(
                id="MKT-BELOW", platform="polymarket", city="New York",
                station="KNYC", date="2026-03-01", market_type="daily_high",
                threshold_high=70.0,
            ))
            session.add(Forecast(
                station="KNYC", target_date="2026-03-01", model="gfs",
                forecast_timestamp=datetime(2026, 2, 28, tzinfo=timezone.utc),
                lead_time_hours=24, temperature_forecast=72.0, actual_temp=65.0,
            ))

        from weatheredge.main import _determine_market_outcome
        state = MagicMock()

        with patch("weatheredge.main.get_session", lambda: get_session(resolution_engine)):
            outcome = _determine_market_outcome(state, "MKT-BELOW")

        assert outcome == "no"

    def test_determine_market_outcome_below_direction_yes(self, resolution_engine):
        """'below X' market stored in threshold_high with direction=below: actual < threshold → yes."""
        import json
        with get_session(resolution_engine) as session:
            session.add(Market(
                id="MKT-BELOW-DIR", platform="kalshi", city="Denver",
                station="KDEN", date="2026-03-01", market_type="daily_high",
                threshold_high=70.0,
                resolution_source=json.dumps({"direction": "below"}),
            ))
            session.add(Forecast(
                station="KDEN", target_date="2026-03-01", model="gfs",
                forecast_timestamp=datetime(2026, 2, 28, tzinfo=timezone.utc),
                lead_time_hours=24, temperature_forecast=68.0, actual_temp=65.0,
            ))

        from weatheredge.main import _determine_market_outcome
        state = MagicMock()

        with patch("weatheredge.main.get_session", lambda: get_session(resolution_engine)):
            outcome = _determine_market_outcome(state, "MKT-BELOW-DIR")

        assert outcome == "yes"  # 65 < 70 → yes for "below" market

    def test_determine_market_outcome_below_direction_no(self, resolution_engine):
        """'below X' market: actual >= threshold → no."""
        import json
        with get_session(resolution_engine) as session:
            session.add(Market(
                id="MKT-BELOW-NO", platform="kalshi", city="Denver",
                station="KDEN", date="2026-03-01", market_type="daily_high",
                threshold_high=70.0,
                resolution_source=json.dumps({"direction": "below"}),
            ))
            session.add(Forecast(
                station="KDEN", target_date="2026-03-01", model="gfs",
                forecast_timestamp=datetime(2026, 2, 28, tzinfo=timezone.utc),
                lead_time_hours=24, temperature_forecast=72.0, actual_temp=75.0,
            ))

        from weatheredge.main import _determine_market_outcome
        state = MagicMock()

        with patch("weatheredge.main.get_session", lambda: get_session(resolution_engine)):
            outcome = _determine_market_outcome(state, "MKT-BELOW-NO")

        assert outcome == "no"  # 75 >= 70 → no for "below" market

    def test_determine_market_outcome_future_date_returns_none(self, resolution_engine):
        """Market with future date cannot be resolved yet."""
        with get_session(resolution_engine) as session:
            session.add(Market(
                id="MKT-FUTURE", platform="kalshi", city="Denver",
                station="KDEN", date="2099-12-31", market_type="daily_high",
                threshold_high=70.0,
            ))

        from weatheredge.main import _determine_market_outcome
        state = MagicMock()

        with patch("weatheredge.main.get_session", lambda: get_session(resolution_engine)):
            outcome = _determine_market_outcome(state, "MKT-FUTURE")

        assert outcome is None

    def test_determine_market_outcome_no_actual_temp_returns_none(self, resolution_engine):
        """Market with no actual observation cannot be resolved."""
        with get_session(resolution_engine) as session:
            session.add(Market(
                id="MKT-NOOBS", platform="polymarket", city="Miami",
                station="KMIA", date="2026-03-01", market_type="daily_high",
                threshold_high=80.0,
            ))
            # No Forecast with actual_temp

        from weatheredge.main import _determine_market_outcome
        state = MagicMock()

        with patch("weatheredge.main.get_session", lambda: get_session(resolution_engine)):
            outcome = _determine_market_outcome(state, "MKT-NOOBS")

        assert outcome is None

    def test_determine_market_outcome_bucket_market(self, resolution_engine):
        """Bucket market: yes if threshold_low <= actual < threshold_high."""
        with get_session(resolution_engine) as session:
            session.add(Market(
                id="MKT-BUCKET", platform="kalshi", city="Denver",
                station="KDEN", date="2026-03-01", market_type="daily_high",
                threshold_low=70.0, threshold_high=75.0,
            ))
            session.add(Forecast(
                station="KDEN", target_date="2026-03-01", model="gfs",
                forecast_timestamp=datetime(2026, 2, 28, tzinfo=timezone.utc),
                lead_time_hours=24, temperature_forecast=72.0, actual_temp=72.0,
            ))

        from weatheredge.main import _determine_market_outcome
        state = MagicMock()

        with patch("weatheredge.main.get_session", lambda: get_session(resolution_engine)):
            outcome = _determine_market_outcome(state, "MKT-BUCKET")

        assert outcome == "yes"

    def test_run_resolution_check_resolves_trades(self, resolution_engine):
        """Full integration: run_resolution_check resolves open trades for past markets."""
        # Seed market, forecast, and open trade
        with get_session(resolution_engine) as session:
            session.add(Market(
                id="MKT-RESOLVE", platform="polymarket", city="New York",
                station="KNYC", date="2026-03-01", market_type="daily_high",
                threshold_high=70.0,
            ))
            session.add(Forecast(
                station="KNYC", target_date="2026-03-01", model="gfs",
                forecast_timestamp=datetime(2026, 2, 28, tzinfo=timezone.utc),
                lead_time_hours=24, temperature_forecast=72.0, actual_temp=75.0,
            ))
            session.add(Trade(
                market_id="MKT-RESOLVE", platform="polymarket", mode="paper",
                side="buy_yes", price=0.40, amount=10.0, cost=4.0,
                station="KNYC", market_date="2026-03-01", status="open",
            ))

        tracker = ResolutionTracker(engine=resolution_engine)
        paper_trader = PaperTrader(initial_bankroll=1000.0)
        paper_trader.connect_db(resolution_engine)

        assert len(paper_trader.open_positions) == 1, "should have 1 open position"

        from weatheredge.main import run_resolution_check
        state = MagicMock()
        state.resolution_tracker = tracker
        state.paper_trader = paper_trader
        state.market_metadata = {}

        with patch("weatheredge.main.get_state", return_value=state), \
             patch("weatheredge.main.get_session", lambda: get_session(resolution_engine)):
            run_resolution_check()

        # Trade should now be resolved
        with get_session(resolution_engine) as session:
            trade = session.query(Trade).filter_by(market_id="MKT-RESOLVE").first()
            assert trade.status == "won"
            assert trade.pnl is not None
            assert trade.pnl > 0

        # Paper trader should have no open positions
        assert len(paper_trader.open_positions) == 0
