"""Integration tests for the market scanning → opportunity building → edge pipeline.

Uses realistic market IDs (Polymarket UUIDs, Kalshi contract IDs) to verify
the full data flow works end-to-end without mocking the lookup logic.
"""

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

# Dynamic future date so tests don't break when hardcoded dates drift into the past
_FUTURE_DATE = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
# Same date formatted as Kalshi event ticker segment: "YYMMMDD" (e.g. "26APR24").
_FUTURE_DATE_KALSHI = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%y%b%d").upper()

import pytest

from weatheredge.agents.base import MarketOpportunity
from weatheredge.data.market_parser import ParsedMarket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init():
    from weatheredge.main import init_app, get_state
    init_app()
    return get_state()


def _make_forecast_rows(station, gfs_temp=75.0, ecmwf_temp=76.0, ensemble=None, target_date=None):
    """Create mock DB forecast rows for multiple models."""
    ts = datetime.now(timezone.utc)
    td = target_date or _FUTURE_DATE
    rows = []

    gfs = MagicMock()
    gfs.station = station
    gfs.target_date = td
    gfs.temperature_forecast = gfs_temp
    gfs.ensemble_members = None
    gfs.model = "gfs"
    gfs.forecast_timestamp = ts
    rows.append(gfs)

    ecmwf = MagicMock()
    ecmwf.station = station
    ecmwf.target_date = td
    ecmwf.temperature_forecast = ecmwf_temp
    ecmwf.ensemble_members = None
    ecmwf.model = "ecmwf"
    ecmwf.forecast_timestamp = ts
    rows.append(ecmwf)

    if ensemble is None:
        ensemble = [gfs_temp + i * 0.3 - 4.5 for i in range(31)]
    ens = MagicMock()
    ens.station = station
    ens.target_date = td
    ens.temperature_forecast = gfs_temp
    ens.ensemble_members = json.dumps(ensemble)
    ens.model = "ensemble"
    ens.forecast_timestamp = ts
    rows.append(ens)

    return rows


# ===========================================================================
# Test: Polymarket market with realistic UUID flows through pipeline
# ===========================================================================

class TestPolymarketIntegration:
    """Polymarket markets use UUID market_ids and parsed title data."""

    def test_polymarket_uuid_builds_opportunity(self):
        """A Polymarket market with a UUID ID should build an opportunity
        when metadata cache contains parsed data."""
        state = _init()

        market_id = "0x1234567890abcdef1234567890abcdef12345678"
        future_date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        parsed = ParsedMarket(
            city="nyc", station="KNYC", date=future_date,
            market_type="daily_high", threshold=72.0, direction="above",
        )
        state.market_metadata[market_id] = {
            "market_id": market_id,
            "platform": "polymarket",
            "resolution_source": {
                "station": "KNYC",
                "type": "daily_high",
                "source": "parsed_from_title",
            },
            "parsed": parsed,
        }

        # yes_price=0.55: model≈0.97, gross_edge≈0.42 ≤ 0.50 cap (FIX 2 won't zero it)
        price_data = {"yes_price": 0.55}
        forecast_rows = _make_forecast_rows("KNYC", gfs_temp=78.0, ecmwf_temp=79.0, target_date=future_date)

        with patch("weatheredge.main.get_session") as mock_sess:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = forecast_rows
            mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_sess.return_value.__exit__ = MagicMock(return_value=False)

            from weatheredge.main import _read_latest_forecasts, _build_opportunity
            forecasts = _read_latest_forecasts()
            opp = _build_opportunity(market_id, price_data, forecasts)

        assert opp is not None
        assert isinstance(opp, MarketOpportunity)
        assert opp.station == "KNYC"
        assert opp.threshold == 72.0  # From parsed data, NOT hardcoded 70.0
        assert opp.direction == "above"
        assert opp.platform == "polymarket"
        assert opp.city == "NYC"  # Resolved via STATION_TO_CITY, not station code
        assert opp.market_price == 0.55
        assert 0.0 <= opp.model_probability <= 1.0
        # Edge should be significant: 78°F forecast vs 72°F threshold
        assert opp.net_edge != 0.0

    def test_polymarket_below_direction(self):
        """Below-direction markets should use correct direction."""
        state = _init()

        market_id = "0xbeef_below_market"
        parsed = ParsedMarket(
            city="miami", station="KMIA", date=_FUTURE_DATE,
            market_type="daily_high", threshold=85.0, direction="below",
        )
        state.market_metadata[market_id] = {
            "market_id": market_id,
            "platform": "polymarket",
            "resolution_source": {"station": "KMIA", "type": "daily_high"},
            "parsed": parsed,
        }

        price_data = {"yes_price": 0.60}
        forecast_rows = _make_forecast_rows("KMIA", gfs_temp=82.0, ecmwf_temp=81.0, target_date=_FUTURE_DATE)

        with patch("weatheredge.main.get_session") as mock_sess:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = forecast_rows
            mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_sess.return_value.__exit__ = MagicMock(return_value=False)

            from weatheredge.main import _read_latest_forecasts, _build_opportunity
            forecasts = _read_latest_forecasts()
            opp = _build_opportunity(market_id, price_data, forecasts)

        assert opp is not None
        assert opp.direction == "below"
        assert opp.threshold == 85.0
        # 82°F forecast, threshold 85°F, direction below → high probability
        assert opp.model_probability > 0.5

    def test_polymarket_bucket_market(self):
        """Bucket markets should populate bucket_low/bucket_high."""
        state = _init()

        market_id = "0xbucket_market_id"
        parsed = ParsedMarket(
            city="denver", station="KDEN", date=_FUTURE_DATE,
            market_type="daily_high",
            threshold_low=74.0, threshold_high=75.0,
            direction="above",
        )
        state.market_metadata[market_id] = {
            "market_id": market_id,
            "platform": "polymarket",
            "resolution_source": {"station": "KDEN", "type": "daily_high"},
            "parsed": parsed,
        }

        price_data = {"yes_price": 0.15}
        forecast_rows = _make_forecast_rows("KDEN", gfs_temp=74.5)

        with patch("weatheredge.main.get_session") as mock_sess:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = forecast_rows
            mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_sess.return_value.__exit__ = MagicMock(return_value=False)

            from weatheredge.main import _read_latest_forecasts, _build_opportunity
            forecasts = _read_latest_forecasts()
            opp = _build_opportunity(market_id, price_data, forecasts)

        assert opp is not None
        assert opp.bucket_low == 74.0
        assert opp.bucket_high == 75.0
        # Midpoint threshold used: (74 + 75) / 2 = 74.5
        assert opp.threshold == 74.5


# ===========================================================================
# Test: Kalshi contract IDs flow through pipeline via prefix match
# ===========================================================================

class TestKalshiIntegration:
    """Kalshi markets use contract-level IDs like KXHIGHDEN-26MAR30-T76."""

    def test_kalshi_contract_id_prefix_match(self):
        """A Kalshi contract ID should match via prefix to RESOLUTION_SOURCES."""
        state = _init()
        # No metadata cache entry — relies on prefix matching
        state.market_metadata.pop(f"KXHIGHNY-{_FUTURE_DATE_KALSHI}-T65", None)

        market_id = f"KXHIGHNY-{_FUTURE_DATE_KALSHI}-T65"
        price_data = {"yes_price": 0.55}
        forecast_rows = _make_forecast_rows("KNYC", gfs_temp=68.0)

        with patch("weatheredge.main.get_session") as mock_sess:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = forecast_rows
            mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_sess.return_value.__exit__ = MagicMock(return_value=False)

            from weatheredge.main import _read_latest_forecasts, _build_opportunity
            forecasts = _read_latest_forecasts()
            opp = _build_opportunity(market_id, price_data, forecasts)

        assert opp is not None
        assert opp.station == "KNYC"
        assert opp.platform == "kalshi"
        # Threshold extracted from contract ID: T65 → 65.0
        assert opp.threshold == 65.0
        assert opp.city == "NYC"

    def test_kalshi_bracket_contract(self):
        """Kalshi bracket contracts (B76.5) become bucket ranges, not below-threshold markets."""
        state = _init()

        market_id = f"KXHIGHDEN-{_FUTURE_DATE_KALSHI}-B76.5"
        price_data = {"yes_price": 0.20}
        forecast_rows = _make_forecast_rows("KDEN", gfs_temp=77.0)

        with patch("weatheredge.main.get_session") as mock_sess:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = forecast_rows
            mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_sess.return_value.__exit__ = MagicMock(return_value=False)

            from weatheredge.main import _read_latest_forecasts, _build_opportunity
            forecasts = _read_latest_forecasts()
            opp = _build_opportunity(market_id, price_data, forecasts)

        assert opp is not None
        assert opp.threshold == 76.5
        assert opp.bucket_low == pytest.approx(75.5)
        assert opp.bucket_high == pytest.approx(77.5)
        assert opp.platform == "kalshi"
        assert opp.station == "KDEN"
        assert opp.city == "Denver"

    def test_kalshi_with_metadata_cache(self):
        """Kalshi market with cached metadata should use it directly."""
        state = _init()

        market_id = f"KXHIGHCHI-{_FUTURE_DATE_KALSHI}-T55"
        state.market_metadata[market_id] = {
            "market_id": market_id,
            "ticker": "KXHIGHCHI",
            "platform": "kalshi",
            "resolution_source": {
                "station": "KMDW",
                "type": "daily_high",
                "source": "NWS CLI",
            },
        }

        price_data = {"yes_price": 0.70}
        forecast_rows = _make_forecast_rows("KMDW", gfs_temp=58.0)

        with patch("weatheredge.main.get_session") as mock_sess:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = forecast_rows
            mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_sess.return_value.__exit__ = MagicMock(return_value=False)

            from weatheredge.main import _read_latest_forecasts, _build_opportunity
            forecasts = _read_latest_forecasts()
            opp = _build_opportunity(market_id, price_data, forecasts)

        assert opp is not None
        assert opp.station == "KMDW"
        assert opp.city == "Chicago"
        # Threshold from contract ID since no parsed data
        assert opp.threshold == 55.0


# ===========================================================================
# Test: Multi-model forecast lookup
# ===========================================================================

class TestMultiModelForecasts:
    """_read_latest_forecasts should collect all models per station."""

    def test_all_models_collected(self):
        _init()
        rows = _make_forecast_rows("KNYC", gfs_temp=75.0, ecmwf_temp=76.0)

        with patch("weatheredge.main.get_session") as mock_sess:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = rows
            mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_sess.return_value.__exit__ = MagicMock(return_value=False)

            from weatheredge.main import _read_latest_forecasts
            result = _read_latest_forecasts()

        # New nested structure: {station: {date: {fields}}}
        entry = result["KNYC"][_FUTURE_DATE]
        assert entry["temperature_forecast"] == 75.0
        assert entry["ecmwf_temp"] == 76.0
        assert entry["ensemble_members"] is not None
        assert len(entry["ensemble_members"]) == 31

    def test_duplicate_models_skipped(self):
        """Only the most recent row per model per station should be kept."""
        _init()
        rows = _make_forecast_rows("KNYC", gfs_temp=75.0)
        # Add a duplicate older GFS row
        old_gfs = MagicMock()
        old_gfs.station = "KNYC"
        old_gfs.target_date = _FUTURE_DATE
        old_gfs.temperature_forecast = 72.0
        old_gfs.ensemble_members = None
        old_gfs.model = "gfs"
        old_gfs.forecast_timestamp = datetime.now(timezone.utc) - timedelta(hours=1)
        rows.append(old_gfs)

        with patch("weatheredge.main.get_session") as mock_sess:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = rows
            mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_sess.return_value.__exit__ = MagicMock(return_value=False)

            from weatheredge.main import _read_latest_forecasts
            result = _read_latest_forecasts()

        # Should use the first (most recent) GFS row, not the older one
        assert result["KNYC"][_FUTURE_DATE]["temperature_forecast"] == 75.0


# ===========================================================================
# Test: Lead time calculation
# ===========================================================================

class TestLeadTimeCalculation:
    """Lead time should be computed from market date, not hardcoded."""

    def test_same_day_market_short_lead(self):
        """A same-day market should have short lead time."""
        state = _init()

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        market_id = "0xtoday_market"
        parsed = ParsedMarket(
            city="nyc", station="KNYC", date=today,
            market_type="daily_high", threshold=70.0, direction="above",
        )
        state.market_metadata[market_id] = {
            "market_id": market_id,
            "platform": "polymarket",
            "resolution_source": {"station": "KNYC", "type": "daily_high"},
            "parsed": parsed,
        }

        price_data = {"yes_price": 0.50}
        forecast_rows = _make_forecast_rows("KNYC", gfs_temp=72.0, target_date=today)

        with patch("weatheredge.main.get_session") as mock_sess:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = forecast_rows
            mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_sess.return_value.__exit__ = MagicMock(return_value=False)

            from weatheredge.main import _read_latest_forecasts, _build_opportunity
            forecasts = _read_latest_forecasts()
            opp = _build_opportunity(market_id, price_data, forecasts)

        assert opp is not None
        # Same-day market has lead_time_days == 0 (within the day, not a full day out)
        assert opp.lead_time_days >= 0

    def test_future_market_longer_lead(self):
        """A market 3 days out should have lead_time_days >= 3."""
        state = _init()

        future = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%d")
        market_id = "0xfuture_market"
        parsed = ParsedMarket(
            city="denver", station="KDEN", date=future,
            market_type="daily_high", threshold=60.0, direction="above",
        )
        state.market_metadata[market_id] = {
            "market_id": market_id,
            "platform": "polymarket",
            "resolution_source": {"station": "KDEN", "type": "daily_high"},
            "parsed": parsed,
        }

        price_data = {"yes_price": 0.60}
        forecast_rows = _make_forecast_rows("KDEN", gfs_temp=65.0, target_date=future)

        with patch("weatheredge.main.get_session") as mock_sess:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = forecast_rows
            mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_sess.return_value.__exit__ = MagicMock(return_value=False)

            from weatheredge.main import _read_latest_forecasts, _build_opportunity
            forecasts = _read_latest_forecasts()
            opp = _build_opportunity(market_id, price_data, forecasts)

        assert opp is not None
        # With midnight target, 3 calendar days out gives ~62h delta → lead_time_days=2
        assert opp.lead_time_days >= 2


# ===========================================================================
# Test: Arb scan gets city name and date
# ===========================================================================

class TestArbScanData:
    """Cross-platform arb scan should receive proper city names and dates."""

    def test_arb_raw_has_city_and_date(self):
        """Verify the arb scan dict uses city name (not station code) and date."""
        state = _init()

        poly_id = "0xpoly_arb"
        parsed = ParsedMarket(
            city="nyc", station="KNYC", date=_FUTURE_DATE,
            market_type="daily_high", threshold=70.0, direction="above",
        )
        state.market_metadata[poly_id] = {
            "market_id": poly_id,
            "platform": "polymarket",
            "resolution_source": {"station": "KNYC", "type": "daily_high"},
            "parsed": parsed,
        }

        kalshi_id = f"KXHIGHNY-{_FUTURE_DATE_KALSHI}-T70"
        state.market_metadata[kalshi_id] = {
            "market_id": kalshi_id,
            "ticker": "KXHIGHNY",
            "platform": "kalshi",
            "resolution_source": {
                "station": "KNYC",
                "type": "daily_high",
                "source": "NWS CLI",
            },
        }

        forecast_rows = _make_forecast_rows("KNYC", gfs_temp=75.0)
        price_data = {
            poly_id: {"yes_price": 0.60, "timestamp": datetime.now(timezone.utc)},
            kalshi_id: {"yes_price": 0.55, "timestamp": datetime.now(timezone.utc)},
        }

        with patch("weatheredge.main.get_session") as mock_sess:
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = forecast_rows
            mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_sess.return_value.__exit__ = MagicMock(return_value=False)

            from weatheredge.main import _read_latest_forecasts, _build_opportunity
            forecasts = _read_latest_forecasts()

            opp_poly = _build_opportunity(poly_id, price_data[poly_id], forecasts)
            opp_kalshi = _build_opportunity(kalshi_id, price_data[kalshi_id], forecasts)

        assert opp_poly is not None
        assert opp_kalshi is not None
        # Both should resolve to "NYC", not "KNYC"
        assert opp_poly.city == "NYC"
        assert opp_kalshi.city == "NYC"


# ===========================================================================
# Test: Unknown / invalid markets still return None
# ===========================================================================

class TestEdgeCases:
    """Edge cases that should still return None."""

    def test_unknown_market_no_metadata_no_prefix_match(self):
        state = _init()
        from weatheredge.main import _build_opportunity
        opp = _build_opportunity("totally_unknown_id", {"yes_price": 0.50}, {"KNYC": {"temperature_forecast": 75.0}})
        assert opp is None

    def test_no_forecast_returns_none(self):
        state = _init()
        state.market_metadata["0xno_forecast"] = {
            "platform": "polymarket",
            "resolution_source": {"station": "KNYC", "type": "daily_high"},
            "parsed": ParsedMarket(city="nyc", station="KNYC", date=_FUTURE_DATE,
                                   market_type="daily_high", threshold=70.0),
        }
        from weatheredge.main import _build_opportunity
        opp = _build_opportunity("0xno_forecast", {"yes_price": 0.50}, {})
        assert opp is None

    def test_bad_price_returns_none(self):
        state = _init()
        state.market_metadata["0xbad_price"] = {
            "platform": "polymarket",
            "resolution_source": {"station": "KNYC", "type": "daily_high"},
            "parsed": ParsedMarket(city="nyc", station="KNYC", date=_FUTURE_DATE,
                                   market_type="daily_high", threshold=70.0),
        }
        from weatheredge.main import _build_opportunity
        forecasts = {"KNYC": {"temperature_forecast": 75.0, "ecmwf_temp": None, "hrrr_temp": None, "ensemble_members": None}}
        opp = _build_opportunity("0xbad_price", {"yes_price": 0.0}, forecasts)
        assert opp is None

    def test_no_temp_returns_none(self):
        state = _init()
        state.market_metadata["0xno_temp"] = {
            "platform": "polymarket",
            "resolution_source": {"station": "KNYC", "type": "daily_high"},
            "parsed": ParsedMarket(city="nyc", station="KNYC", date=_FUTURE_DATE,
                                   market_type="daily_high", threshold=70.0),
        }
        from weatheredge.main import _build_opportunity
        forecasts = {"KNYC": {"temperature_forecast": None, "ecmwf_temp": None, "hrrr_temp": None, "ensemble_members": None}}
        opp = _build_opportunity("0xno_temp", {"yes_price": 0.50}, forecasts)
        assert opp is None

    def test_metar_override_works_with_real_threshold(self):
        """METAR override should use the actual market threshold, not hardcoded 70."""
        state = _init()

        market_id = "0xmetar_test"
        # Use a future date so the market isn't rejected as expired (BugFix C5)
        from datetime import timedelta
        _future_date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        parsed = ParsedMarket(
            city="phoenix", station="KPHX", date=_future_date,
            market_type="daily_high", threshold=90.0, direction="above",
        )
        state.market_metadata[market_id] = {
            "platform": "polymarket",
            "resolution_source": {"station": "KPHX", "type": "daily_high"},
            "parsed": parsed,
        }

        # Mock METAR showing 92°F observed — should trigger override for 90°F threshold
        with patch.object(
            state.conditions_tracker,
            "metar_probability_override",
            return_value=0.99,
        ) as mock_metar:
            # gfs_temp=97, ecmwf_temp=97: both models well above threshold=90,
            # blended forecast_prob≈0.97, no contradiction with METAR override=0.99 (diff<0.30)
            forecast_rows = _make_forecast_rows("KPHX", gfs_temp=97.0, ecmwf_temp=97.0, target_date=_future_date)

            with patch("weatheredge.main.get_session") as mock_sess:
                mock_db = MagicMock()
                mock_db.query.return_value.filter.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = forecast_rows
                mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
                mock_sess.return_value.__exit__ = MagicMock(return_value=False)

                from weatheredge.main import _read_latest_forecasts, _build_opportunity
                forecasts = _read_latest_forecasts()
                opp = _build_opportunity(market_id, {"yes_price": 0.80}, forecasts)

            # Verify METAR was called with correct threshold (90.0), not hardcoded 70.0
            mock_metar.assert_called_once_with(
                station="KPHX",
                threshold=90.0,
                direction="above",
                market_type="daily_high",
                market_date=_future_date,
            )
            assert opp is not None
            assert opp.model_probability == 0.99
