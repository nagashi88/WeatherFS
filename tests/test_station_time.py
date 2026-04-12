"""Tests for station-local market close utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from weatheredge.data.station_time import hours_until_market_close, market_close_time_utc
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
