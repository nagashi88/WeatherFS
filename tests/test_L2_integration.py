"""L2 Integration tests — hit real external APIs.

Run with:
    pytest tests/test_L2_integration.py -v

These tests require network access and are marked @pytest.mark.L2 and
@pytest.mark.slow so they can be deselected with -m "not slow".
"""

import asyncio
from datetime import datetime, timezone

import aiohttp
import pytest


@pytest.mark.L2
@pytest.mark.slow
class TestL2Integration:

    @pytest.mark.live_api
    def test_open_meteo_ensemble_real(self):
        """Fetch GFS ensemble for NYC, verify >=20 temperature members."""
        from weatheredge.data.weather import fetch_ensemble

        async def run():
            async with aiohttp.ClientSession() as session:
                return await fetch_ensemble(40.78, -73.97, session)

        data = asyncio.run(run())
        assert data is not None

        # Open-Meteo ensemble endpoint returns keys like
        # "temperature_2m_member00", "temperature_2m_member01", ...
        hourly = data.get("hourly", {})
        temp_keys = [
            k for k in hourly.keys()
            if "temperature_2m" in k and "member" in k
        ]
        assert len(temp_keys) >= 20, (
            f"Expected >=20 ensemble member keys, got {len(temp_keys)}: "
            f"{temp_keys[:5]}..."
        )

    @pytest.mark.live_api
    def test_open_meteo_forecast_real(self):
        """Fetch forecast for NYC, returns temperature_2m data."""
        from weatheredge.data.weather import fetch_forecast

        async def run():
            async with aiohttp.ClientSession() as session:
                return await fetch_forecast(40.78, -73.97, session)

        data = asyncio.run(run())
        assert data is not None

        hourly = data.get("hourly", {})
        assert "temperature_2m" in hourly, (
            f"Expected 'temperature_2m' in hourly, got keys: {list(hourly.keys())}"
        )
        assert len(hourly["temperature_2m"]) > 0

    @pytest.mark.live_api
    def test_noaa_nws_real(self):
        """Fetch hourly forecast for NYC from NOAA NWS."""
        from weatheredge.data.noaa import fetch_noaa_forecast

        async def run():
            async with aiohttp.ClientSession() as session:
                return await fetch_noaa_forecast(40.78, -73.97, session)

        data = asyncio.run(run())
        assert data is not None

        # fetch_noaa_forecast returns a NoaaForecast TypedDict with 'periods'
        assert "periods" in data, (
            f"Expected 'periods' key, got: {list(data.keys())}"
        )
        assert len(data["periods"]) > 0

        # HourlyPeriod uses 'temperature_f' (parsed from NOAA's 'temperature')
        first = data["periods"][0]
        assert "temperature_f" in first, (
            f"Expected 'temperature_f' in period, got keys: {list(first.keys())}"
        )
        assert isinstance(first["temperature_f"], float)

    @pytest.mark.live_api
    def test_kalshi_public_api_real(self):
        """Fetch real Kalshi weather markets from public API."""
        import asyncio
        from weatheredge.data.kalshi_client import fetch_kalshi_weather_markets
        markets = asyncio.run(fetch_kalshi_weather_markets(series_tickers=["KXHIGHNY"]))
        assert len(markets) >= 0  # May be 0 if no open markets, but should not error
        # If markets exist, verify structure
        if markets:
            assert "ticker" in markets[0]
            assert "title" in markets[0]

    def test_sqlite_read_write_cycle(self, tmp_path):
        """Create DB, write forecast row, read it back, verify match."""
        from sqlalchemy import select

        from weatheredge.db.database import get_session, init_db, make_engine
        from weatheredge.db.models import Forecast

        db_url = f"sqlite:///{tmp_path}/test.db"
        engine = make_engine(db_url)
        init_db(engine)

        # forecast_timestamp is a DateTime column — must be a datetime object
        ts = datetime(2026, 3, 29, 12, 0, 0, tzinfo=timezone.utc)

        # Write
        with get_session(engine) as session:
            f = Forecast(
                station="KNYC",
                target_date="2026-03-30",
                model="gfs",
                forecast_timestamp=ts,
                lead_time_hours=24,
                temperature_forecast=85.0,
            )
            session.add(f)

        # Read back
        with get_session(engine) as session:
            result = session.execute(
                select(Forecast).where(Forecast.station == "KNYC")
            ).scalar_one()
            assert result.temperature_forecast == 85.0
            assert result.model == "gfs"
            assert result.lead_time_hours == 24

        engine.dispose()
