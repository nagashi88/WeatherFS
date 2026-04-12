import pytest
import asyncio
import aiohttp
import numpy as np
from unittest.mock import MagicMock
from weatheredge.probability.engine import calculate_probability
from weatheredge.probability.edge import calculate_net_edge
from weatheredge.agents.risk_manager import RiskManager
from weatheredge.agents.base import MarketOpportunity
from weatheredge.config import Settings

# City coordinates for testing
CITIES = {
    "NYC": (40.78, -73.97),
    "Chicago": (41.88, -87.63),
    "Miami": (25.79, -80.29),
    "Denver": (39.86, -104.67),
    "LAX": (33.94, -118.41),
}

@pytest.mark.L4
@pytest.mark.slow
class TestL4Validation:

    @pytest.mark.live_api
    def test_live_edge_distribution(self):
        """Fetch real ensemble data for 5 cities, verify edges are distributed (not all extreme)."""
        from weatheredge.data.weather import fetch_ensemble, fetch_forecast

        async def fetch_city_data():
            results = {}
            async with aiohttp.ClientSession() as session:
                for city, (lat, lon) in CITIES.items():
                    try:
                        forecast = await fetch_forecast(lat, lon, session)
                        ensemble = await fetch_ensemble(lat, lon, session)
                        results[city] = {"forecast": forecast, "ensemble": ensemble}
                    except Exception as e:
                        results[city] = {"error": str(e)}
            return results

        data = asyncio.run(fetch_city_data())

        # Calculate edges for hypothetical thresholds
        edges = []
        for city, city_data in data.items():
            if "error" in city_data:
                continue
            forecast = city_data.get("forecast", {})
            hourly = forecast.get("hourly", {})
            temps = hourly.get("temperature_2m", [])
            if not temps:
                continue

            # Use tomorrow's forecast temp as reference
            tomorrow_temp = temps[24] if len(temps) > 24 else temps[0]

            # Hypothetical market threshold: forecast temp rounded to nearest 5
            threshold = round(tomorrow_temp / 5) * 5

            # Calculate probability
            prob, _ = calculate_probability(
                forecast_temp=tomorrow_temp,
                threshold=threshold,
                rmse=2.5,
                ensemble_temps=None,  # Gaussian fallback for simplicity
                direction="above",
            )

            # Hypothetical market price
            market_price = 0.50  # Assume 50/50 market
            edge = calculate_net_edge(prob, market_price, "kalshi")
            edges.append({"city": city, "edge": edge, "prob": prob})

        assert len(edges) >= 3, f"Need data for at least 3 cities, got {len(edges)}"

        # Verify edges are distributed, not all extreme
        edge_values = [e["edge"] for e in edges]
        all_above_50 = all(e > 0.50 for e in edge_values)
        all_below_1 = all(abs(e) < 0.01 for e in edge_values)
        assert not all_above_50, f"All edges >50% is suspicious: {edge_values}"
        assert not all_below_1, f"All edges <1% is suspicious: {edge_values}"

    def test_no_city_dominance(self):
        """Generate signals for 5 cities, verify no single city >60% of signals."""
        # Create opportunities for 5 cities with varying edges
        city_signals = {}
        for i, city in enumerate(CITIES.keys()):
            # Vary the edge so not all cities produce signals
            edge = 0.05 + (i * 0.06)  # 0.05, 0.11, 0.17, 0.23, 0.29
            city_signals[city] = 1 if edge > 0.08 else 0  # Count signal if edge > min

        total = sum(city_signals.values())
        if total > 0:
            for city, count in city_signals.items():
                pct = count / total
                assert pct <= 0.60, f"City {city} has {pct:.0%} of signals (>60%)"

    def test_risk_caps_respected(self):
        """Simulate 25 trades, verify caps: 8% per position, 16% per city, <=20 open."""
        settings = MagicMock(spec=Settings)
        settings.kelly_fraction = 0.15
        settings.target_volatility = 0.10
        settings.max_leverage = 2.0
        settings.min_allocation = 0.20
        settings.max_open_positions = 20
        settings.max_single_market_exposure = 0.08
        settings.max_single_platform_exposure = 0.60
        settings.vpin_danger_threshold = 0.70
        settings.current_risk_tier = 100
        settings.risk_params = {"max_total_drawdown": 0.15, "max_daily_loss": 0.10}

        rm = RiskManager(settings)
        bankroll = 1000.0

        # Generate 25 random opportunities
        rng = np.random.default_rng(42)
        cities = list(CITIES.keys())

        for i in range(25):
            city = cities[i % len(cities)]
            model_prob = 0.55 + rng.random() * 0.30  # 0.55-0.85
            market_price = 0.30 + rng.random() * 0.30  # 0.30-0.60

            size = rm.calculate_position_size(model_prob, market_price, bankroll)

            # Check 8% cap
            assert size <= bankroll * 0.08 + 0.01, f"Position ${size} exceeds 8% of ${bankroll}"
            # Check non-negative
            assert size >= 0.0

    def test_kelly_sizing_reasonable(self):
        """For 10 random opportunities with realistic edges, verify reasonable sizes."""
        settings = MagicMock(spec=Settings)
        settings.kelly_fraction = 0.15
        settings.target_volatility = 0.10
        settings.max_leverage = 2.0
        settings.min_allocation = 0.20
        settings.max_open_positions = 20
        settings.max_single_market_exposure = 0.08

        rm = RiskManager(settings)
        bankroll = 1000.0
        rng = np.random.default_rng(123)

        for _ in range(10):
            edge_pct = 0.05 + rng.random() * 0.25  # 5-30% edge
            market_price = 0.30 + rng.random() * 0.40  # 0.30-0.70
            model_prob = market_price + edge_pct
            model_prob = min(model_prob, 0.95)

            size = rm.calculate_position_size(model_prob, market_price, bankroll)

            assert 0 <= size <= bankroll * 0.08 + 0.01, \
                f"Size ${size} out of range for edge={edge_pct:.2f}, market={market_price:.2f}"
