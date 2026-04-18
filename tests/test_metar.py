"""Tests for Enhancement 3: METAR CurrentConditionsTracker."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock
from zoneinfo import ZoneInfo

from weatheredge.data.metar import (
    CurrentConditionsTracker,
    DailyTracker,
    StationObservation,
    _station_local_date,
)


class TestStationObservation:
    def test_defaults(self):
        obs = StationObservation(station="KNYC")
        assert obs.station == "KNYC"
        assert obs.temperature_f is None
        assert obs.raw_metar == ""


class TestDailyTracker:
    def test_update_tracks_high_low(self):
        tracker = DailyTracker(station="KNYC", date="2026-03-30")
        obs1 = StationObservation(station="KNYC", temperature_f=65.0)
        obs2 = StationObservation(station="KNYC", temperature_f=72.0)
        obs3 = StationObservation(station="KNYC", temperature_f=68.0)

        tracker.update(obs1)
        assert tracker.daily_high_f == 65.0
        assert tracker.daily_low_f == 65.0
        assert tracker.observation_count == 1

        tracker.update(obs2)
        assert tracker.daily_high_f == 72.0
        assert tracker.daily_low_f == 65.0

        tracker.update(obs3)
        assert tracker.daily_high_f == 72.0
        assert tracker.daily_low_f == 65.0
        assert tracker.observation_count == 3

    def test_update_skips_none_temp(self):
        tracker = DailyTracker(station="KNYC", date="2026-03-30")
        obs = StationObservation(station="KNYC", temperature_f=None)
        tracker.update(obs)
        assert tracker.daily_high_f == -999.0
        assert tracker.observation_count == 0


class TestCurrentConditionsTracker:
    def test_metar_override_threshold_exceeded(self):
        tracker = CurrentConditionsTracker()
        # Simulate an observation where high already exceeds threshold
        daily = DailyTracker(station="KNYC", date="2026-03-30")
        obs = StationObservation(station="KNYC", temperature_f=76.0)
        daily.update(obs)
        tracker._trackers["KNYC"] = daily

        result = tracker.metar_probability_override("KNYC", threshold=75.0, direction="above")
        assert result == 0.99

    def test_metar_override_below_when_exceeded(self):
        tracker = CurrentConditionsTracker()
        daily = DailyTracker(station="KNYC", date="2026-03-30")
        obs = StationObservation(station="KNYC", temperature_f=76.0)
        daily.update(obs)
        tracker._trackers["KNYC"] = daily

        # "Will high be below 75?" — already exceeded, so probability ~ 0
        result = tracker.metar_probability_override(
            "KNYC", threshold=75.0, direction="below", market_type="daily_high"
        )
        assert result == 0.01

    def test_metar_override_no_data(self):
        tracker = CurrentConditionsTracker()
        result = tracker.metar_probability_override("KNYC", threshold=75.0)
        assert result is None

    def test_metar_override_not_exceeded_no_override(self):
        tracker = CurrentConditionsTracker()
        daily = DailyTracker(station="KNYC", date="2026-03-30")
        obs = StationObservation(station="KNYC", temperature_f=70.0)
        daily.update(obs)
        tracker._trackers["KNYC"] = daily

        # Early in the day (default UTC hour won't be >= 20), no override
        result = tracker.metar_probability_override("KNYC", threshold=75.0, direction="above")
        assert result is None

    def test_get_tracker_none(self):
        tracker = CurrentConditionsTracker()
        assert tracker.get_tracker("KNYC") is None

    def test_get_tracker_exists(self):
        tracker = CurrentConditionsTracker()
        daily = DailyTracker(station="KNYC", date="2026-03-30")
        tracker._trackers["KNYC"] = daily
        assert tracker.get_tracker("KNYC") is daily

    @pytest.mark.asyncio
    async def test_poll_station_success(self):
        tracker = CurrentConditionsTracker()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[{
            "temp": 20.0,
            "dewp": 10.0,
            "wspd": 5,
            "wdir": 180,
            "altim": 1013.0,
            "reportTime": "2026-03-30T12:00:00Z",
            "rawOb": "KNYC 301200Z ...",
        }])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        obs = await tracker.poll_station("KNYC", session=mock_session)
        assert obs is not None
        assert obs.station == "KNYC"
        assert obs.temperature_c == 20.0
        assert obs.temperature_f == pytest.approx(68.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_poll_station_api_error(self):
        tracker = CurrentConditionsTracker()
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        obs = await tracker.poll_station("KNYC", session=mock_session)
        assert obs is None


# ---------------------------------------------------------------------------
# Regression: station-local date semantics (P0 fix)
#
# Before the fix, `today` was computed as datetime.now(UTC).strftime(...) at
# both poll_station (daily reset) and metar_probability_override (future-date
# guard). For Denver (UTC-7) this wiped the running daily HIGH 7h before local
# end-of-day — exactly when the afternoon peak sets. Symmetric problem for
# Asia (UTC+9-12) where the tracker stayed on yesterday well into local today.
# ---------------------------------------------------------------------------

def _fake_datetime_at(utc_iso: str):
    """Build a datetime subclass whose .now(tz) returns the given UTC moment.

    Patching `weatheredge.data.metar.datetime` with this lets tests freeze
    wall-clock time without altering tz behaviour."""
    fixed_utc = datetime.fromisoformat(utc_iso).replace(tzinfo=timezone.utc)

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return fixed_utc
            return fixed_utc.astimezone(tz)

    return FakeDateTime


class TestStationLocalDate:
    """Direct tests for the new _station_local_date helper."""

    def test_denver_just_after_utc_midnight_is_still_yesterday_local(self):
        """UTC 2026-04-05 04:00 == Denver 2026-04-04 ~22:00 (DST → -6).
        Local Denver is still April 4, so today() must be 2026-04-04."""
        FakeDT = _fake_datetime_at("2026-04-05T04:00:00")
        with patch("weatheredge.data.metar.datetime", FakeDT):
            assert _station_local_date("KDEN") == "2026-04-04"

    def test_seoul_just_before_utc_midnight_is_already_tomorrow_local(self):
        """UTC 2026-04-04 16:00 == Seoul (RKSS, +9) 2026-04-05 01:00.
        Local Seoul is already April 5, so today() must be 2026-04-05."""
        FakeDT = _fake_datetime_at("2026-04-04T16:00:00")
        with patch("weatheredge.data.metar.datetime", FakeDT):
            assert _station_local_date("RKSS") == "2026-04-05"

    def test_unknown_station_falls_back_to_utc(self):
        FakeDT = _fake_datetime_at("2026-04-05T04:00:00")
        with patch("weatheredge.data.metar.datetime", FakeDT):
            # No tz mapping for unknown station → UTC string
            assert _station_local_date("ZZZZ") == "2026-04-05"


class TestMetarOverrideLocalDateRegression:
    """The override compared market_date against UTC date; for Denver this
    returned None during the most decisive 7h of the trading day."""

    def test_override_fires_for_denver_local_day_after_utc_rollover(self):
        """KDEN market_date='2026-04-04', UTC has rolled to 04-05 but Denver is
        still on 04-04. Pre-fix this returned None (no override). Post-fix the
        override fires because local date matches."""
        FakeDT = _fake_datetime_at("2026-04-05T04:00:00")
        tracker = CurrentConditionsTracker()
        # Seed an observation that has already breached the threshold
        daily = DailyTracker(station="KDEN", date="2026-04-04")
        daily.update(StationObservation(station="KDEN", temperature_f=78.0))
        tracker._trackers["KDEN"] = daily

        with patch("weatheredge.data.metar.datetime", FakeDT):
            result = tracker.metar_probability_override(
                "KDEN", threshold=75.0, direction="above",
                market_type="daily_high", market_date="2026-04-04",
            )
        assert result == 0.99  # was None before fix

    def test_override_still_blocks_genuinely_future_market(self):
        """Counter-test: a market for *tomorrow* local must still get None."""
        FakeDT = _fake_datetime_at("2026-04-05T04:00:00")
        tracker = CurrentConditionsTracker()
        daily = DailyTracker(station="KDEN", date="2026-04-04")
        daily.update(StationObservation(station="KDEN", temperature_f=78.0))
        tracker._trackers["KDEN"] = daily

        with patch("weatheredge.data.metar.datetime", FakeDT):
            result = tracker.metar_probability_override(
                "KDEN", threshold=75.0, direction="above",
                market_type="daily_high", market_date="2026-04-05",  # future local
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_poll_station_keeps_denver_tracker_on_local_date(self):
        """Pre-fix poll_station reset the KDEN tracker the moment UTC rolled,
        wiping the still-building daily HIGH. Post-fix the tracker stays on
        the local date so the running max survives."""
        FakeDT = _fake_datetime_at("2026-04-05T04:00:00")
        tracker = CurrentConditionsTracker()
        # Pre-existing tracker for Denver-local 2026-04-04 with a peak HIGH
        prev = DailyTracker(station="KDEN", date="2026-04-04")
        prev.update(StationObservation(station="KDEN", temperature_f=78.0))
        tracker._trackers["KDEN"] = prev

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[{
            "temp": 22.0, "dewp": 5.0, "wspd": 3, "wdir": 270,
            "altim": 1015.0, "reportTime": "2026-04-05T04:00:00Z",
            "rawOb": "KDEN 050400Z ...",
        }])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)

        with patch("weatheredge.data.metar.datetime", FakeDT):
            await tracker.poll_station("KDEN", session=mock_session)

        kept = tracker.get_tracker("KDEN")
        assert kept is not None
        assert kept.date == "2026-04-04"  # still local-yesterday, NOT UTC-today
        assert kept.daily_high_f == 78.0  # peak survived UTC midnight
