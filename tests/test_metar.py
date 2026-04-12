"""Tests for Enhancement 3: METAR CurrentConditionsTracker."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

from weatheredge.data.metar import (
    CurrentConditionsTracker,
    DailyTracker,
    StationObservation,
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
