"""Tests for Enhancement 4: GFS Model Run Detector."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from weatheredge.data.model_run_detector import (
    ModelRunDetector,
    ModelRunEvent,
    GFS_RUN_WINDOWS,
    DELTA_THRESHOLD_F,
)


class TestModelRunDetector:
    def _make_detector(self, on_new_run=None):
        coords = {
            "NYC": (40.7128, -74.0060),
            "Chicago": (41.8781, -87.6298),
        }
        return ModelRunDetector(city_coords=coords, on_new_run=on_new_run)

    def test_is_in_check_window_during_00z(self):
        det = self._make_detector()
        # 03:30 UTC should be in the 00Z window
        t = datetime(2026, 3, 30, 3, 30, tzinfo=timezone.utc)
        in_window, label = det.is_in_check_window(t)
        assert in_window is True
        assert label == "00Z"

    def test_is_in_check_window_during_12z(self):
        det = self._make_detector()
        t = datetime(2026, 3, 30, 15, 30, tzinfo=timezone.utc)
        in_window, label = det.is_in_check_window(t)
        assert in_window is True
        assert label == "12Z"

    def test_is_in_check_window_outside(self):
        det = self._make_detector()
        t = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc)
        in_window, label = det.is_in_check_window(t)
        assert in_window is False
        assert label == ""

    def test_arm_status(self):
        det = self._make_detector()
        status = det.arm_status()
        assert status["armed"] is True
        assert "in_check_window" in status
        assert "cached_cities" in status
        assert "NYC" in status["cached_cities"]
        assert "Chicago" in status["cached_cities"]

    def test_detection_log_starts_empty(self):
        det = self._make_detector()
        assert det.detection_log == []

    @pytest.mark.asyncio
    async def test_check_outside_window_returns_empty(self):
        det = self._make_detector()
        # Patch is_in_check_window to return False
        det.is_in_check_window = lambda now=None: (False, "")
        events = await det.check_for_new_run()
        assert events == []

    @pytest.mark.asyncio
    async def test_check_detects_delta(self):
        callback = AsyncMock()
        det = self._make_detector(on_new_run=callback)

        # Force into a check window
        det.is_in_check_window = lambda now=None: (True, "12Z")

        # Pre-populate cache with old temps
        det._cache["NYC"].temperature_f = 70.0
        det._cache["Chicago"].temperature_f = 65.0

        # Mock the fetch to return shifted temps
        async def mock_fetch(lat, lon, session):
            if lat == pytest.approx(40.7128, abs=0.01):
                return 72.0  # NYC shifted +2°F
            return 65.0  # Chicago unchanged

        det._fetch_forecast_temp = mock_fetch

        mock_session = MagicMock()
        events = await det.check_for_new_run(session=mock_session)

        assert len(events) == 1
        assert events[0].city == "NYC"
        assert events[0].delta_f == pytest.approx(2.0, abs=0.1)
        assert events[0].run_label == "12Z"
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_no_delta_no_events(self):
        det = self._make_detector()
        det.is_in_check_window = lambda now=None: (True, "12Z")

        det._cache["NYC"].temperature_f = 70.0
        det._cache["Chicago"].temperature_f = 65.0

        async def mock_fetch(lat, lon, session):
            if lat == pytest.approx(40.7128, abs=0.01):
                return 70.1  # NYC shifted only 0.1°F — below threshold
            return 65.0

        det._fetch_forecast_temp = mock_fetch

        events = await det.check_for_new_run(session=MagicMock())
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_skips_already_detected_run(self):
        det = self._make_detector()
        det.is_in_check_window = lambda now=None: (True, "12Z")

        # Mark NYC as already detected for 12Z
        det._cache["NYC"].last_model_run_detected = "12Z"
        det._cache["NYC"].temperature_f = 70.0
        det._cache["Chicago"].temperature_f = 65.0

        async def mock_fetch(lat, lon, session):
            return 75.0  # big delta but should be skipped for NYC

        det._fetch_forecast_temp = mock_fetch

        events = await det.check_for_new_run(session=MagicMock())
        # Only Chicago should be checked (NYC skipped)
        nyc_events = [e for e in events if e.city == "NYC"]
        assert len(nyc_events) == 0


# ---------------------------------------------------------------------------
# Edge-path tests — coverage gaps: lines 96, 121, 139, 169, 229
# ---------------------------------------------------------------------------


class TestModelRunDetectorEdgeCases:
    def _make_detector(self, on_new_run=None):
        coords = {"NYC": (40.7128, -74.0060), "Chicago": (41.8781, -87.6298)}
        return ModelRunDetector(city_coords=coords, on_new_run=on_new_run)

    def test_is_in_check_window_no_args_uses_current_time(self):
        """is_in_check_window() without now= uses datetime.now (line 96)."""
        det = self._make_detector()
        # Must not raise; result is time-dependent but always valid types
        in_window, label = det.is_in_check_window()
        assert isinstance(in_window, bool)
        assert isinstance(label, str)

    @pytest.mark.asyncio
    async def test_check_for_new_run_creates_and_closes_own_session(self):
        """No-session call creates aiohttp.ClientSession and closes it (lines 121, 169)."""
        det = self._make_detector()
        det.is_in_check_window = lambda now=None: (True, "12Z")
        det._cache["NYC"].temperature_f = 70.0
        det._cache["Chicago"].temperature_f = 65.0

        async def mock_fetch(lat, lon, session):
            return 70.2  # delta 0.2 — below DELTA_THRESHOLD_F=0.5, no event

        det._fetch_forecast_temp = mock_fetch

        mock_session = AsyncMock()
        mock_session.close = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await det.check_for_new_run()  # no session arg

        mock_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_new_temp_none_skips_city(self):
        """_fetch_forecast_temp returning None causes city to be skipped (line 139)."""
        det = self._make_detector()
        det.is_in_check_window = lambda now=None: (True, "12Z")
        det._cache["NYC"].temperature_f = 70.0
        det._cache["Chicago"].temperature_f = 65.0

        async def mock_fetch(lat, lon, session):
            return None  # triggers the `if new_temp_f is None: continue` path

        det._fetch_forecast_temp = mock_fetch

        events = await det.check_for_new_run(session=MagicMock())
        assert events == []
        # Cached values must be unchanged because None was returned
        assert det._cache["NYC"].temperature_f == 70.0
        assert det._cache["Chicago"].temperature_f == 65.0

    def test_arm_status_all_windows_elapsed_wraps_to_tomorrow(self):
        """arm_status falls through to 'tomorrow' message when hour ≥ 21 (line 229)."""
        det = self._make_detector()
        # 22:00 UTC — past the last GFS window start (21Z), no future start_h > 22
        late_night = datetime(2026, 3, 30, 22, 0, tzinfo=timezone.utc)

        with patch("weatheredge.data.model_run_detector.datetime") as mock_dt:
            mock_dt.now.return_value = late_night
            status = det.arm_status()

        assert "tomorrow" in status["next_window"]
        assert "03:00" in status["next_window"]
