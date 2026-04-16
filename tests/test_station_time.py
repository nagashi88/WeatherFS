"""Tests for station-local market close utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from weatheredge.data.station_time import (
    hours_until_market_close,
    market_close_time_utc,
    station_local_today_str,
)
from weatheredge.execution.exit_manager import ExitManager


def test_market_close_time_utc_klax_respects_dst():
    """KLAX close on Apr 1 should map to 06:59 UTC next day (PDT)."""
    close_utc = market_close_time_utc("2026-04-01", "KLAX")
    assert close_utc == datetime(2026, 4, 2, 6, 59, tzinfo=timezone.utc)


def test_hours_until_market_close_stays_positive_after_utc_midnight():
    """UTC midnight should not imply market closed when local day is still active."""
    now_utc = datetime(2026, 4, 2, 1, 0, tzinfo=timezone.utc)
    hours_left = hours_until_market_close("2026-04-01", "KLAX", now_utc=now_utc)

    assert hours_left is not None
    assert hours_left == pytest.approx(5.9833, rel=1e-3)


def test_station_local_today_str_uses_local_date_across_utc_midnight():
    """After UTC midnight, station-local date can still be the previous day."""
    # 02:00 UTC = 19:00 PDT previous day for KLAX.
    now_utc = datetime(2026, 4, 17, 2, 0, tzinfo=timezone.utc)
    assert station_local_today_str("KLAX", now_utc=now_utc) == "2026-04-16"

    # 01:00 UTC = 21:00 EDT previous day for KNYC.
    assert station_local_today_str("KNYC", now_utc=now_utc.replace(hour=1)) == "2026-04-16"


def test_station_local_today_str_falls_back_to_utc_for_unknown_station():
    """Unknown station code → fall back to UTC date without raising."""
    now_utc = datetime(2026, 4, 17, 1, 0, tzinfo=timezone.utc)
    assert station_local_today_str("KZZZ", now_utc=now_utc) == "2026-04-17"


def test_exit_manager_lead_time_uses_station_local_close_helper():
    """ExitManager should delegate lead-time math to station-local close utility."""
    parsed = MagicMock()
    parsed.date = "2026-04-01"
    parsed.station = "KLAX"
    market_metadata = {
        "mkt-1": {
            "parsed": parsed,
            "resolution_source": {"station": "KLAX"},
        }
    }

    with patch("weatheredge.execution.exit_manager.hours_until_market_close", return_value=4.25) as mock_hours:
        lead = ExitManager._lead_time_hours("mkt-1", market_metadata)

    assert lead == pytest.approx(4.25)
    mock_hours.assert_called_once_with("2026-04-01", "KLAX")
