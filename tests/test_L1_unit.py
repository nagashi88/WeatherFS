import pytest
from unittest.mock import MagicMock
from weatheredge.probability.gaussian import gaussian_probability
from weatheredge.probability.ensemble import ensemble_probability
from weatheredge.probability.edge import calculate_net_edge
from weatheredge.probability.bias_correction import apply_bias_correction, BIAS_TABLE
from weatheredge.probability.bucket import bucket_probability
from weatheredge.agents.risk_manager import RiskManager
from weatheredge.config import Settings


@pytest.mark.L1
class TestL1Unit:
    def test_gaussian_cdf_known_value(self):
        """forecast=85, threshold=80, rmse=2.5 -> Student-T(df=5) P(above) ≈ 0.9490"""
        p = gaussian_probability(85.0, 80.0, 2.5, "above")
        # Student-T with df=5 gives ~0.949 (fatter tails than Gaussian's 0.9772)
        assert abs(p - 0.9490) < 0.005

    def test_gaussian_cdf_at_threshold(self):
        """forecast=80, threshold=80 -> P(above) = 0.50"""
        p = gaussian_probability(80.0, 80.0, 2.5, "above")
        assert abs(p - 0.50) < 0.001

    def test_ensemble_counting_all_above(self):
        """31/31 above threshold -> clamped to 0.95 (PROB_CEIL)"""
        temps = [85.0] * 31
        assert ensemble_probability(temps, 80.0, "above") == 0.95

    def test_ensemble_counting_half(self):
        """16/31 above -> ~0.516"""
        temps = [81.0] * 16 + [79.0] * 15
        p = ensemble_probability(temps, 80.0, "above")
        assert abs(p - 16/31) < 0.01

    def test_kelly_correct_small_edge(self):
        """model_prob=0.70, market=0.69 -> sub-fee edge → $0.

        Post-fee-fix: 1-point gross edge is consumed by fees
        (entry=0.02, exit=0.03). Net Kelly correctly returns 0.
        """
        settings = MagicMock(spec=Settings)
        settings.kelly_fraction = 0.15
        settings.target_volatility = 0.10
        settings.max_leverage = 2.0
        settings.min_allocation = 0.20
        settings.max_open_positions = 20
        settings.max_single_market_exposure = 0.08
        rm = RiskManager(settings)
        size = rm.calculate_position_size(0.70, 0.69, 1000.0)
        assert size == 0.0, f"Expected $0 (sub-fee edge), got ${size}"

    def test_kelly_large_edge_capped(self):
        """model_prob=0.70, market=0.25 → $80 stake cap.

        Post-fee-fix: b=0.70/0.27≈2.593, kelly≈0.584, raw stake=$87.64 → capped to $80.
        CCG-bughunt3 G3-new: contracts = stake / (price + kalshi_fee) = 80 / 0.27 ≈ 296.3.
        Dividing by price alone would silently breach the $80 cash cap.
        """
        settings = MagicMock(spec=Settings)
        settings.kelly_fraction = 0.15
        settings.target_volatility = 0.10
        settings.max_leverage = 2.0
        settings.min_allocation = 0.20
        settings.max_open_positions = 20
        settings.max_single_market_exposure = 0.08
        settings.max_cost_per_trade_fraction = 1.0  # disable cost cap to test amount cap
        rm = RiskManager(settings)
        # Warm-up period uses reduced Kelly (0.10); set resolved trades past threshold
        rm.set_resolved_trade_count(50)
        size = rm.calculate_position_size(0.70, 0.25, 1000.0)
        assert size == pytest.approx(296.3, abs=0.40), f"Expected ≈296.3 contracts ($80 cap / 0.27), got {size}"
        # Invariant: realized cash outlay must not exceed the dollar cap.
        assert size * (0.25 + 0.02) <= 80.01, f"Realized cost {size * 0.27:.2f} exceeds $80 cap"

    def test_kelly_no_edge(self):
        """model_prob=0.40, market=0.50 -> $0"""
        settings = MagicMock(spec=Settings)
        settings.kelly_fraction = 0.15
        settings.max_single_market_exposure = 0.08
        rm = RiskManager(settings)
        size = rm.calculate_position_size(0.40, 0.50, 1000.0)
        assert size == 0.0

    def test_vpin_calculation(self):
        """80% buy fills -> VPIN=0.60; 95% buy -> VPIN=0.90"""
        settings = MagicMock(spec=Settings)
        rm = RiskManager(settings)
        fills_80 = [{"side": "buy", "size": 1}] * 80 + [{"side": "sell", "size": 1}] * 20
        assert abs(rm.calculate_vpin(fills_80) - 0.60) < 0.01

        fills_95 = [{"side": "buy", "size": 1}] * 95 + [{"side": "sell", "size": 1}] * 5
        assert abs(rm.calculate_vpin(fills_95) - 0.90) < 0.01

    def test_net_edge_with_fees(self):
        """model=0.60, market=0.40, polymarket -> net_edge ~0.185"""
        edge = calculate_net_edge(0.60, 0.40, "polymarket")
        assert abs(edge - 0.185) < 0.001

    def test_bias_correction(self):
        """Known bias entry -> corrected forecast reduced by bias."""
        from weatheredge.probability.bias_correction import BIAS_COUNT_TABLE
        # Add a test bias
        key = ("KNYC", "Jul", "SW", "clear")
        BIAS_TABLE[key] = 1.5
        BIAS_COUNT_TABLE[key] = 50
        corrected = apply_bias_correction(85.0, "KNYC", "Jul", "SW", "clear")
        assert corrected == 83.5
        # Clean up
        del BIAS_TABLE[key]
        del BIAS_COUNT_TABLE[key]

    def test_bucket_probability(self):
        """forecast=82, bucket=[80,85], rmse=2.5 -> ~0.65"""
        p = bucket_probability(82.0, 80.0, 85.0, 2.5)
        assert 0.55 < p < 0.75, f"Expected ~0.65, got {p}"
