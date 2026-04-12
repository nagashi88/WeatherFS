"""Tests for weatheredge.data.noaa — NOAA NWS API fetcher.

Covers:
- Two-step points → forecast/hourly lookup
- Correct parsing of NOAA hourly JSON format
- Retry logic: succeeds on second attempt after one failure
- Retry exhaustion: raises after 3 failures
- store_noaa_forecast writes correct rows to the forecasts table
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
import pytest_asyncio

from weatheredge.data.noaa import (
    fetch_noaa_forecast,
    store_noaa_forecast,
    NOAA_BASE,
    USER_AGENT,
)
from weatheredge.db.database import init_db, get_session, make_engine
from weatheredge.db.models import Forecast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_engine():
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    yield eng
    eng.dispose()


# Minimal NOAA /points response
POINTS_RESPONSE = {
    "properties": {
        "forecastHourly": "https://api.weather.gov/gridpoints/OKX/33,37/forecast/hourly",
    }
}

# Minimal NOAA /forecast/hourly response
HOURLY_RESPONSE = {
    "properties": {
        "periods": [
            {
                "startTime": "2026-03-29T14:00:00-04:00",
                "temperature": 65,
                "temperatureUnit": "F",
                "windSpeed": "10 mph",
                "shortForecast": "Partly Cloudy",
            },
            {
                "startTime": "2026-03-29T15:00:00-04:00",
                "temperature": 67,
                "temperatureUnit": "F",
                "windSpeed": "12 mph",
                "shortForecast": "Mostly Sunny",
            },
        ]
    }
}


def _make_mock_response(json_data: dict, status: int = 200):
    """Return an async context-manager mock that yields a response with json()."""
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    resp.text = AsyncMock(return_value=str(json_data))
    # Make it usable as `async with session.get(...) as resp:`
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_mock_session(*responses):
    """Build an aiohttp.ClientSession mock whose .get() returns *responses* in order."""
    session = MagicMock()
    session.get = MagicMock(side_effect=list(responses))
    return session


# ---------------------------------------------------------------------------
# Two-step lookup: points → forecast/hourly
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_step_lookup_calls_correct_urls():
    """fetch_noaa_forecast must call /points first, then the returned forecastHourly URL."""
    points_cm = _make_mock_response(POINTS_RESPONSE)
    hourly_cm = _make_mock_response(HOURLY_RESPONSE)
    session = _make_mock_session(points_cm, hourly_cm)

    result = await fetch_noaa_forecast(40.7128, -74.0060, session)

    assert session.get.call_count == 2
    first_url = session.get.call_args_list[0][0][0]
    second_url = session.get.call_args_list[1][0][0]
    assert first_url == f"{NOAA_BASE}/points/40.7128,-74.006"
    assert second_url == POINTS_RESPONSE["properties"]["forecastHourly"]


@pytest.mark.asyncio
async def test_two_step_lookup_returns_forecast_hourly_url():
    """Result must include the forecastHourly URL from the /points response."""
    session = _make_mock_session(
        _make_mock_response(POINTS_RESPONSE),
        _make_mock_response(HOURLY_RESPONSE),
    )
    result = await fetch_noaa_forecast(40.7128, -74.0060, session)
    assert result["forecast_hourly_url"] == POINTS_RESPONSE["properties"]["forecastHourly"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parses_hourly_periods_count():
    """All periods in the response must be parsed and returned."""
    session = _make_mock_session(
        _make_mock_response(POINTS_RESPONSE),
        _make_mock_response(HOURLY_RESPONSE),
    )
    result = await fetch_noaa_forecast(40.7128, -74.0060, session)
    assert len(result["periods"]) == 2


@pytest.mark.asyncio
async def test_parses_temperature_as_float():
    """temperature_f must be a float parsed from the integer in the response."""
    session = _make_mock_session(
        _make_mock_response(POINTS_RESPONSE),
        _make_mock_response(HOURLY_RESPONSE),
    )
    result = await fetch_noaa_forecast(40.7128, -74.0060, session)
    assert result["periods"][0]["temperature_f"] == 65.0
    assert isinstance(result["periods"][0]["temperature_f"], float)


@pytest.mark.asyncio
async def test_parses_wind_speed_and_forecast():
    """wind_speed and short_forecast must be copied verbatim from NOAA response."""
    session = _make_mock_session(
        _make_mock_response(POINTS_RESPONSE),
        _make_mock_response(HOURLY_RESPONSE),
    )
    result = await fetch_noaa_forecast(40.7128, -74.0060, session)
    assert result["periods"][0]["wind_speed"] == "10 mph"
    assert result["periods"][0]["short_forecast"] == "Partly Cloudy"


@pytest.mark.asyncio
async def test_parses_start_time_preserved():
    """start_time must be preserved exactly as the ISO string from NOAA."""
    session = _make_mock_session(
        _make_mock_response(POINTS_RESPONSE),
        _make_mock_response(HOURLY_RESPONSE),
    )
    result = await fetch_noaa_forecast(40.7128, -74.0060, session)
    assert result["periods"][0]["start_time"] == "2026-03-29T14:00:00-04:00"


@pytest.mark.asyncio
async def test_lat_lon_preserved_in_result():
    """lat and lon must be echoed back in the result."""
    session = _make_mock_session(
        _make_mock_response(POINTS_RESPONSE),
        _make_mock_response(HOURLY_RESPONSE),
    )
    result = await fetch_noaa_forecast(40.7128, -74.0060, session)
    assert result["lat"] == 40.7128
    assert result["lon"] == -74.0060


# ---------------------------------------------------------------------------
# User-Agent header
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_user_agent_header_sent():
    """Every request must include the NOAA-required User-Agent header."""
    points_cm = _make_mock_response(POINTS_RESPONSE)
    hourly_cm = _make_mock_response(HOURLY_RESPONSE)
    session = _make_mock_session(points_cm, hourly_cm)

    await fetch_noaa_forecast(40.7128, -74.0060, session)

    for c in session.get.call_args_list:
        headers = c[1].get("headers", {})
        assert headers.get("User-Agent") == USER_AGENT


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_succeeds_on_second_attempt():
    """A single transient failure on /points must be retried and succeed."""
    # First call raises, second returns the points data, third returns hourly
    fail_cm = MagicMock()
    fail_cm.__aenter__ = AsyncMock(side_effect=aiohttp.ClientConnectionError("timeout"))
    fail_cm.__aexit__ = AsyncMock(return_value=False)

    session = _make_mock_session(
        fail_cm,
        _make_mock_response(POINTS_RESPONSE),
        _make_mock_response(HOURLY_RESPONSE),
    )

    with patch("weatheredge.data.noaa.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await fetch_noaa_forecast(40.7128, -74.0060, session)

    assert len(result["periods"]) == 2
    mock_sleep.assert_called_once_with(4)  # backoff after attempt 1 → wait 4s


@pytest.mark.asyncio
async def test_retry_exhausted_raises():
    """After 3 consecutive failures, fetch_noaa_forecast must raise."""
    import aiohttp as _aiohttp

    fail_cm = MagicMock()
    fail_cm.__aenter__ = AsyncMock(side_effect=_aiohttp.ClientConnectionError("down"))
    fail_cm.__aexit__ = AsyncMock(return_value=False)

    # Return the same failing mock for all 3 attempts on /points
    session = _make_mock_session(fail_cm, fail_cm, fail_cm)

    with patch("weatheredge.data.noaa.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(_aiohttp.ClientConnectionError):
            await fetch_noaa_forecast(40.7128, -74.0060, session)

    assert session.get.call_count == 3


@pytest.mark.asyncio
async def test_retry_backoff_delays_are_correct():
    """Retry delays must follow the 1s → 4s → 16s schedule (Gap 12).

    We verify by failing twice then succeeding; the two sleeps must use
    4s (after attempt 1) only — the third attempt succeeds so no third sleep.
    """
    import aiohttp as _aiohttp

    fail_cm1 = MagicMock()
    fail_cm1.__aenter__ = AsyncMock(side_effect=_aiohttp.ClientConnectionError("fail1"))
    fail_cm1.__aexit__ = AsyncMock(return_value=False)

    fail_cm2 = MagicMock()
    fail_cm2.__aenter__ = AsyncMock(side_effect=_aiohttp.ClientConnectionError("fail2"))
    fail_cm2.__aexit__ = AsyncMock(return_value=False)

    session = _make_mock_session(
        fail_cm1,
        fail_cm2,
        _make_mock_response(POINTS_RESPONSE),
        _make_mock_response(HOURLY_RESPONSE),
    )

    with patch("weatheredge.data.noaa.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await fetch_noaa_forecast(40.7128, -74.0060, session)

    assert len(result["periods"]) == 2
    # Two failures → slept after attempt 0 (4s) and attempt 1 (16s)
    assert mock_sleep.call_args_list == [call(4), call(16)]


@pytest.mark.asyncio
async def test_http_error_status_triggers_retry():
    """An HTTP 500 response must be treated as a retryable failure."""
    err_cm = _make_mock_response({}, status=500)

    session = _make_mock_session(
        err_cm,
        _make_mock_response(POINTS_RESPONSE),
        _make_mock_response(HOURLY_RESPONSE),
    )

    with patch("weatheredge.data.noaa.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await fetch_noaa_forecast(40.7128, -74.0060, session)

    assert len(result["periods"]) == 2
    mock_sleep.assert_called_once_with(4)


# ---------------------------------------------------------------------------
# store_noaa_forecast
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_store_noaa_forecast_writes_rows(db_engine):
    """store_noaa_forecast must write one Forecast row per hourly period."""
    forecast_data = {
        "lat": 40.7128,
        "lon": -74.0060,
        "forecast_hourly_url": POINTS_RESPONSE["properties"]["forecastHourly"],
        "periods": [
            {
                "start_time": "2026-03-29T14:00:00-04:00",
                "temperature_f": 65.0,
                "wind_speed": "10 mph",
                "short_forecast": "Partly Cloudy",
            },
            {
                "start_time": "2026-03-29T15:00:00-04:00",
                "temperature_f": 67.0,
                "wind_speed": "12 mph",
                "short_forecast": "Mostly Sunny",
            },
        ],
    }

    with get_session(engine=db_engine) as session:
        rows = store_noaa_forecast(session, "KNYC", "2026-03-29", forecast_data)

    assert len(rows) == 2

    with get_session(engine=db_engine) as session:
        saved = session.query(Forecast).filter_by(station="KNYC").all()
    assert len(saved) == 2


@pytest.mark.asyncio
async def test_store_noaa_forecast_model_name(db_engine):
    """All stored rows must have model='noaa_nws'."""
    forecast_data = {
        "lat": 40.7128,
        "lon": -74.0060,
        "forecast_hourly_url": "",
        "periods": [
            {
                "start_time": "2026-03-29T14:00:00-04:00",
                "temperature_f": 65.0,
                "wind_speed": "10 mph",
                "short_forecast": "Partly Cloudy",
            }
        ],
    }

    with get_session(engine=db_engine) as session:
        store_noaa_forecast(session, "KNYC", "2026-03-29", forecast_data)

    with get_session(engine=db_engine) as session:
        row = session.query(Forecast).filter_by(station="KNYC").first()
        assert row.model == "noaa_nws"


@pytest.mark.asyncio
async def test_store_noaa_forecast_temperature(db_engine):
    """temperature_forecast column must match the period's temperature_f."""
    forecast_data = {
        "lat": 40.7128,
        "lon": -74.0060,
        "forecast_hourly_url": "",
        "periods": [
            {
                "start_time": "2026-03-29T14:00:00-04:00",
                "temperature_f": 72.5,
                "wind_speed": "5 mph",
                "short_forecast": "Clear",
            }
        ],
    }

    with get_session(engine=db_engine) as session:
        store_noaa_forecast(session, "KORD", "2026-03-29", forecast_data)

    with get_session(engine=db_engine) as session:
        row = session.query(Forecast).filter_by(station="KORD").first()
        assert row.temperature_forecast == 72.5


@pytest.mark.asyncio
async def test_store_noaa_forecast_target_date(db_engine):
    """target_date must be stored as provided."""
    forecast_data = {
        "lat": 25.7959,
        "lon": -80.2870,
        "forecast_hourly_url": "",
        "periods": [
            {
                "start_time": "2026-03-30T10:00:00-05:00",
                "temperature_f": 80.0,
                "wind_speed": "8 mph",
                "short_forecast": "Sunny",
            }
        ],
    }

    with get_session(engine=db_engine) as session:
        store_noaa_forecast(session, "KMIA", "2026-03-30", forecast_data)

    with get_session(engine=db_engine) as session:
        row = session.query(Forecast).filter_by(station="KMIA").first()
        assert row.target_date == "2026-03-30"


# import needed for the aiohttp exception in retry tests
import aiohttp  # noqa: E402 (imported here to keep test-local usage clear)
