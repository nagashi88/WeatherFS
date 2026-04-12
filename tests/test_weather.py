"""Tests for weatheredge.data.weather module.

Covers:
- fetch_forecast parses hourly data correctly
- fetch_ensemble returns ensemble member values
- Retry logic: first call fails, second succeeds -> returns data
- All retries fail -> raises exception
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from weatheredge.data.weather import (
    BASE_URL,
    ENSEMBLE_VARS,
    HOURLY_VARS,
    fetch_ecmwf,
    fetch_ensemble,
    fetch_forecast,
    store_forecast,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_TIMES = ["2026-03-30T00:00", "2026-03-30T01:00", "2026-03-30T02:00"]

SAMPLE_FORECAST_RESPONSE = {
    "latitude": 40.71,
    "longitude": -74.01,
    "hourly": {
        "time": SAMPLE_TIMES,
        "temperature_2m": [5.2, 5.0, 4.8],
        "apparent_temperature": [2.1, 1.9, 1.7],
        "precipitation": [0.0, 0.1, 0.2],
        "precipitation_probability": [10, 20, 30],
        "rain": [0.0, 0.0, 0.1],
        "snowfall": [0.0, 0.0, 0.0],
        "cloud_cover": [45, 50, 55],
        "wind_speed_10m": [12.3, 13.1, 14.0],
        "wind_gusts_10m": [20.0, 21.5, 22.0],
        "relative_humidity_2m": [70, 72, 74],
    },
}

SAMPLE_ENSEMBLE_RESPONSE = {
    "latitude": 40.71,
    "longitude": -74.01,
    "hourly": {
        "time": SAMPLE_TIMES,
        "temperature_2m": [4.5, 4.3, 4.1],
        "temperature_2m_member01": [4.5, 4.3, 4.1],
        "temperature_2m_member02": [4.3, 4.1, 3.9],
        "temperature_2m_member03": [4.1, 3.9, 3.7],
        "precipitation": [0.0, 0.05, 0.1],
        "wind_speed_10m": [11.0, 11.5, 12.0],
    },
}

SAMPLE_ECMWF_RESPONSE = {
    "latitude": 40.71,
    "longitude": -74.01,
    "hourly": {
        "time": SAMPLE_TIMES,
        "temperature_2m": [5.5, 5.3, 5.1],
        "apparent_temperature": [2.5, 2.3, 2.1],
        "precipitation": [0.0, 0.0, 0.1],
        "precipitation_probability": [5, 10, 15],
        "rain": [0.0, 0.0, 0.0],
        "snowfall": [0.0, 0.0, 0.0],
        "cloud_cover": [30, 35, 40],
        "wind_speed_10m": [10.0, 10.5, 11.0],
        "wind_gusts_10m": [18.0, 18.5, 19.0],
        "relative_humidity_2m": [65, 67, 69],
    },
}


def _make_mock_session(response_data: dict) -> MagicMock:
    """Build an aiohttp.ClientSession mock that returns response_data as JSON."""
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(return_value=response_data)

    mock_context = AsyncMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(return_value=mock_context)
    return mock_session


# ---------------------------------------------------------------------------
# fetch_forecast tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_forecast_returns_hourly_data():
    """fetch_forecast must return the full API response including hourly dict."""
    session = _make_mock_session(SAMPLE_FORECAST_RESPONSE)

    result = await fetch_forecast(40.71, -74.01, session)

    assert "hourly" in result
    hourly = result["hourly"]
    assert "temperature_2m" in hourly
    assert hourly["temperature_2m"] == [5.2, 5.0, 4.8]


@pytest.mark.asyncio
async def test_fetch_forecast_all_hourly_vars_present():
    """fetch_forecast response must contain all HOURLY_VARS keys."""
    session = _make_mock_session(SAMPLE_FORECAST_RESPONSE)

    result = await fetch_forecast(40.71, -74.01, session)
    hourly = result["hourly"]

    for var in HOURLY_VARS:
        assert var in hourly, f"Expected '{var}' in hourly response"


@pytest.mark.asyncio
async def test_fetch_forecast_calls_correct_endpoint():
    """fetch_forecast must call BASE_URL/v1/forecast with correct params."""
    session = _make_mock_session(SAMPLE_FORECAST_RESPONSE)

    await fetch_forecast(40.71, -74.01, session)

    session.get.assert_called_once()
    call_args = session.get.call_args
    url = call_args[0][0]
    params = call_args[1]["params"]

    assert url == f"{BASE_URL}/v1/forecast"
    assert params["latitude"] == 40.71
    assert params["longitude"] == -74.01
    assert params["hourly"] == ",".join(HOURLY_VARS)


# ---------------------------------------------------------------------------
# fetch_ensemble tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_ensemble_returns_member_values():
    """fetch_ensemble must return hourly data with ensemble variable arrays."""
    session = _make_mock_session(SAMPLE_ENSEMBLE_RESPONSE)

    result = await fetch_ensemble(40.71, -74.01, session)

    assert "hourly" in result
    hourly = result["hourly"]
    assert "temperature_2m" in hourly
    assert isinstance(hourly["temperature_2m"], list)
    assert len(hourly["temperature_2m"]) == 3


@pytest.mark.asyncio
async def test_fetch_ensemble_all_ensemble_vars_present():
    """fetch_ensemble response must contain all ENSEMBLE_VARS keys."""
    session = _make_mock_session(SAMPLE_ENSEMBLE_RESPONSE)

    result = await fetch_ensemble(40.71, -74.01, session)
    hourly = result["hourly"]

    for var in ENSEMBLE_VARS:
        assert var in hourly, f"Expected '{var}' in ensemble hourly response"


@pytest.mark.asyncio
async def test_fetch_ensemble_calls_correct_endpoint():
    """fetch_ensemble must call BASE_URL/v1/ensemble."""
    session = _make_mock_session(SAMPLE_ENSEMBLE_RESPONSE)

    await fetch_ensemble(40.71, -74.01, session)

    call_args = session.get.call_args
    url = call_args[0][0]
    from weatheredge.data.weather import ENSEMBLE_BASE_URL
    assert url == f"{ENSEMBLE_BASE_URL}/v1/ensemble"


# ---------------------------------------------------------------------------
# fetch_ecmwf tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_ecmwf_returns_hourly_data():
    """fetch_ecmwf must return the full API response including hourly dict."""
    session = _make_mock_session(SAMPLE_ECMWF_RESPONSE)

    result = await fetch_ecmwf(40.71, -74.01, session)

    assert "hourly" in result
    assert result["hourly"]["temperature_2m"] == [5.5, 5.3, 5.1]


@pytest.mark.asyncio
async def test_fetch_ecmwf_calls_correct_endpoint():
    """fetch_ecmwf must call BASE_URL/v1/ecmwf."""
    session = _make_mock_session(SAMPLE_ECMWF_RESPONSE)

    await fetch_ecmwf(40.71, -74.01, session)

    call_args = session.get.call_args
    url = call_args[0][0]
    assert url == f"{BASE_URL}/v1/ecmwf"


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_succeeds_on_second_attempt():
    """When first call raises, second attempt must succeed and return data."""
    import aiohttp

    # First call: raises ClientError; second call: returns valid data
    success_response = AsyncMock()
    success_response.raise_for_status = MagicMock()
    success_response.json = AsyncMock(return_value=SAMPLE_FORECAST_RESPONSE)

    success_ctx = AsyncMock()
    success_ctx.__aenter__ = AsyncMock(return_value=success_response)
    success_ctx.__aexit__ = AsyncMock(return_value=False)

    fail_ctx = AsyncMock()
    fail_ctx.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("connection refused"))
    fail_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=[fail_ctx, success_ctx])

    with patch("weatheredge.data.weather.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await fetch_forecast(40.71, -74.01, mock_session)

    assert result == SAMPLE_FORECAST_RESPONSE
    # Should have slept once between attempt 1 and 2 (delay=1s)
    mock_sleep.assert_called_once_with(1)
    # get() called exactly twice
    assert mock_session.get.call_count == 2


@pytest.mark.asyncio
async def test_retry_all_fail_raises_exception():
    """When all 3 retry attempts fail, the exception must propagate."""
    import aiohttp

    fail_ctx = AsyncMock()
    fail_ctx.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("server down"))
    fail_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    # Return a new fail_ctx for each of the 3 calls
    mock_session.get = MagicMock(side_effect=[fail_ctx, fail_ctx, fail_ctx])

    with patch("weatheredge.data.weather.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(aiohttp.ClientError):
            await fetch_forecast(40.71, -74.01, mock_session)

    # Should have slept twice: after attempt 1 (1s) and attempt 2 (4s)
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(1)
    mock_sleep.assert_any_call(4)
    # get() called exactly 3 times
    assert mock_session.get.call_count == 3


@pytest.mark.asyncio
async def test_retry_http_error_retries():
    """HTTP 500 errors (raise_for_status) must also trigger retry logic."""
    # Use a generic Exception subclass to simulate raise_for_status failure —
    # ClientResponseError requires a valid RequestInfo object; using OSError
    # keeps the test clean while still exercising the retry path.
    error_response = AsyncMock()
    error_response.raise_for_status = MagicMock(
        side_effect=OSError("HTTP 500 Internal Server Error")
    )

    error_ctx = AsyncMock()
    error_ctx.__aenter__ = AsyncMock(return_value=error_response)
    error_ctx.__aexit__ = AsyncMock(return_value=False)

    success_response = AsyncMock()
    success_response.raise_for_status = MagicMock()
    success_response.json = AsyncMock(return_value=SAMPLE_FORECAST_RESPONSE)

    success_ctx = AsyncMock()
    success_ctx.__aenter__ = AsyncMock(return_value=success_response)
    success_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=[error_ctx, success_ctx])

    with patch("weatheredge.data.weather.asyncio.sleep", new_callable=AsyncMock):
        result = await fetch_forecast(40.71, -74.01, mock_session)

    assert result == SAMPLE_FORECAST_RESPONSE
    assert mock_session.get.call_count == 2


# ---------------------------------------------------------------------------
# store_forecast tests
# ---------------------------------------------------------------------------

def test_store_forecast_writes_to_db():
    """store_forecast must create a Forecast row and add it to the session."""
    from weatheredge.db.database import init_db, get_session, make_engine

    engine = make_engine("sqlite:///:memory:")
    init_db(engine=engine)

    with get_session(engine=engine) as session:
        store_forecast(
            db_session=session,
            station="KNYC",
            target_date="2026-03-30",
            model="forecast",
            forecast_data=SAMPLE_FORECAST_RESPONSE,
        )

    from weatheredge.db.models import Forecast as ForecastModel
    with get_session(engine=engine) as session:
        row = session.query(ForecastModel).filter_by(station="KNYC").first()
        assert row is not None
        assert row.station == "KNYC"
        assert row.target_date == "2026-03-30"
        assert row.model == "forecast"
        assert row.temperature_forecast == 5.2
        assert row.precipitation_forecast == 0.0


def test_store_forecast_ensemble_serialises_members():
    """store_forecast for ensemble model must serialise temperature_2m as JSON."""
    from weatheredge.db.database import init_db, get_session, make_engine
    from weatheredge.db.models import Forecast as ForecastModel

    engine = make_engine("sqlite:///:memory:")
    init_db(engine=engine)

    with get_session(engine=engine) as session:
        store_forecast(
            db_session=session,
            station="KORD",
            target_date="2026-03-30",
            model="ensemble",
            forecast_data=SAMPLE_ENSEMBLE_RESPONSE,
        )

    with get_session(engine=engine) as session:
        row = session.query(ForecastModel).filter_by(station="KORD").first()
        assert row is not None
        assert row.ensemble_members is not None
        members = json.loads(row.ensemble_members)
        assert members == [4.5, 4.3, 4.1]


def test_store_forecast_non_ensemble_no_members():
    """store_forecast for non-ensemble model must leave ensemble_members as None."""
    from weatheredge.db.database import init_db, get_session, make_engine
    from weatheredge.db.models import Forecast as ForecastModel

    engine = make_engine("sqlite:///:memory:")
    init_db(engine=engine)

    with get_session(engine=engine) as session:
        store_forecast(
            db_session=session,
            station="KLAX",
            target_date="2026-03-30",
            model="ecmwf",
            forecast_data=SAMPLE_ECMWF_RESPONSE,
        )

    with get_session(engine=engine) as session:
        row = session.query(ForecastModel).filter_by(station="KLAX").first()
        assert row is not None
        assert row.ensemble_members is None


def test_store_forecast_uses_target_date_max_temperature():
    """store_forecast should use target-date max temp, not first hourly point."""
    mock_session = MagicMock()
    forecast_data = {
        "hourly": {
            "time": [
                "2026-03-29T22:00",
                "2026-03-30T00:00",
                "2026-03-30T12:00",
            ],
            "temperature_2m": [42.0, 65.0, 71.5],
            "precipitation": [0.0, 0.1, 0.2],
        }
    }

    row = store_forecast(
        db_session=mock_session,
        station="KNYC",
        target_date="2026-03-30",
        model="forecast",
        forecast_data=forecast_data,
    )

    assert row.temperature_forecast == pytest.approx(71.5)
    mock_session.add.assert_called_once()


def test_store_forecast_falls_back_to_first_temperature_when_target_date_missing():
    """When target_date points are absent, store_forecast falls back to first value."""
    mock_session = MagicMock()
    forecast_data = {
        "hourly": {
            "time": ["2026-03-29T00:00", "2026-03-29T01:00"],
            "temperature_2m": [50.0, 55.0],
            "precipitation": [0.0, 0.1],
        }
    }

    row = store_forecast(
        db_session=mock_session,
        station="KNYC",
        target_date="2026-03-30",
        model="forecast",
        forecast_data=forecast_data,
    )

    assert row.temperature_forecast == pytest.approx(50.0)
