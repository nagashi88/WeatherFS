"""Tests for trading strategies."""
import numpy as np
import pytest
from unittest.mock import MagicMock
from weatheredge.agents.base import Signal, MarketOpportunity
from weatheredge.agents.strategies.conservative import conservative_strategy
from weatheredge.agents.strategies.aggressive import aggressive_strategy
from weatheredge.agents.strategies.mean_reversion import mean_reversion_strategy
from weatheredge.config import Settings


def make_settings(lookback=168, min_sigma=3.0):
    s = MagicMock(spec=Settings)
    s.mean_reversion_lookback_hours = lookback
    s.mean_reversion_min_sigma = min_sigma
    return s


def make_prices(n=200, mean=0.5, std=0.05, last=None):
    """Build a price series of length n with given mean/std, optional override for last value."""
    rng = np.random.default_rng(42)
    prices = list(mean + std * rng.standard_normal(n))
    if last is not None:
        prices[-1] = last
    return prices


# ---------------------------------------------------------------------------
# Helper: make a MarketOpportunity
# ---------------------------------------------------------------------------

def make_opportunity(**kwargs):
    defaults = dict(
        market_id="MKT-1", platform="kalshi", city="NYC", station="KNYC",
        threshold=80.0, direction="above", market_price=0.40,
        model_probability=0.70, net_edge=0.25, forecast_temp=85.0,
        rmse=2.5, lead_time_days=1, gaussian_prob=0.70, ensemble_prob=0.72,
    )
    defaults.update(kwargs)
    return MarketOpportunity(**defaults)


def make_conservative_settings(**kwargs):
    s = MagicMock(spec=Settings)
    s.conservative_min_edge = kwargs.get("conservative_min_edge", 0.20)
    return s


def make_aggressive_settings(**kwargs):
    s = MagicMock(spec=Settings)
    s.aggressive_min_edge = kwargs.get("aggressive_min_edge", 0.08)
    return s


# ---------------------------------------------------------------------------
# Conservative tests
# ---------------------------------------------------------------------------

def test_conservative_rejects_low_edge():
    opp = make_opportunity(net_edge=0.10)
    signal = conservative_strategy(opp, make_conservative_settings())
    assert signal.side == "no_trade"

def test_conservative_rejects_model_disagreement():
    opp = make_opportunity(net_edge=0.25, gaussian_prob=0.90, ensemble_prob=0.70)
    signal = conservative_strategy(opp, make_conservative_settings())
    assert signal.side == "no_trade"
    assert "agreement" in signal.reasoning.lower() or "0.8" in signal.reasoning

def test_conservative_rejects_long_lead_time():
    opp = make_opportunity(net_edge=0.25, lead_time_days=2)
    signal = conservative_strategy(opp, make_conservative_settings())
    assert signal.side == "no_trade"

def test_conservative_accepts_high_edge():
    opp = make_opportunity(net_edge=0.25, gaussian_prob=0.70, ensemble_prob=0.72)
    signal = conservative_strategy(opp, make_conservative_settings())
    assert signal.side in ("buy_yes", "buy_no")
    assert signal.confidence > 0.0
    assert signal.agent_name == "Conservative"

def test_conservative_confidence_scales_with_edge():
    opp_low = make_opportunity(net_edge=0.20, gaussian_prob=0.70, ensemble_prob=0.72)
    opp_high = make_opportunity(net_edge=0.40, gaussian_prob=0.70, ensemble_prob=0.72)
    sig_low = conservative_strategy(opp_low, make_conservative_settings())
    sig_high = conservative_strategy(opp_high, make_conservative_settings())
    assert sig_low.confidence == pytest.approx(0.5, abs=0.05)
    assert sig_high.confidence == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Aggressive tests
# ---------------------------------------------------------------------------

def test_aggressive_rejects_low_edge():
    opp = make_opportunity(net_edge=0.05)
    signals = aggressive_strategy(opp, [], make_aggressive_settings())
    assert signals == []

def test_aggressive_rejects_long_lead_time():
    opp = make_opportunity(net_edge=0.10, lead_time_days=3)
    signals = aggressive_strategy(opp, [], make_aggressive_settings())
    assert signals == []

def test_aggressive_single_market_no_buckets():
    opp = make_opportunity(net_edge=0.10)
    signals = aggressive_strategy(opp, [], make_aggressive_settings())
    assert len(signals) == 1
    assert signals[0].agent_name == "Aggressive"

def test_aggressive_generates_ladder():
    # Center is a bucket [70, 75]; neighbours share boundaries so they ladder.
    center = make_opportunity(
        market_id="CENTER", net_edge=0.10, bucket_low=70.0, bucket_high=75.0,
    )
    buckets = [
        make_opportunity(market_id="UP", net_edge=0.12, bucket_low=75.0, bucket_high=80.0),
        make_opportunity(market_id="DOWN", net_edge=0.11, bucket_low=65.0, bucket_high=70.0),
        # Non-adjacent: must be excluded.
        make_opportunity(market_id="FAR", net_edge=0.20, bucket_low=90.0, bucket_high=95.0),
    ]
    signals = aggressive_strategy(center, buckets, make_aggressive_settings())
    market_ids = [s.market_id for s in signals]
    assert "CENTER" in market_ids
    assert "FAR" not in market_ids
    assert 1 <= len(signals) <= 3

def test_aggressive_center_bucket_highest_confidence():
    center = make_opportunity(
        market_id="CENTER", net_edge=0.10, model_probability=0.60,
        bucket_low=70.0, bucket_high=75.0,
    )
    buckets = [
        make_opportunity(market_id="UP", net_edge=0.15, model_probability=0.80, bucket_low=75.0, bucket_high=80.0),
        make_opportunity(market_id="DOWN", net_edge=0.12, model_probability=0.70, bucket_low=65.0, bucket_high=70.0),
    ]
    signals = aggressive_strategy(center, buckets, make_aggressive_settings())
    assert len(signals) >= 2
    # Center is always first in the returned list; adjacent neighbours are at 50% confidence.
    assert signals[0].market_id == "CENTER"
    assert all(s.confidence <= signals[0].confidence or s.market_id == "CENTER" for s in signals)


# ---------------------------------------------------------------------------
# Mean Reversion tests
# ---------------------------------------------------------------------------

def test_mean_reversion_triggers_at_3sigma():
    """Price crashes 4sigma below rolling mean -> buy_yes signal."""
    settings = make_settings()
    prices = make_prices(n=200, mean=0.5, std=0.05)
    # Compute what a 4-sigma crash looks like relative to the last 168 prices
    recent = prices[-168:]
    mu = float(np.mean(recent))
    sigma = float(np.std(recent))
    prices[-1] = mu - 4 * sigma

    signal = mean_reversion_strategy("MKT-1", prices, settings)
    assert signal.side == "buy_yes"
    assert signal.confidence > 0.0
    assert signal.agent_name == "MeanReversion"


def test_mean_reversion_no_signal_within_band():
    """Price 1sigma from mean -> no_trade."""
    settings = make_settings()
    prices = make_prices(n=200, mean=0.5, std=0.05)
    recent = prices[-168:]
    mu = float(np.mean(recent))
    sigma = float(np.std(recent))
    prices[-1] = mu + 1.0 * sigma

    signal = mean_reversion_strategy("MKT-2", prices, settings)
    assert signal.side == "no_trade"
    assert signal.confidence == 0.0


def test_mean_reversion_correct_side_positive_deviation():
    """Price 3.5sigma above mean -> buy_no."""
    settings = make_settings()
    prices = make_prices(n=200, mean=0.5, std=0.05)
    recent = prices[-168:]
    mu = float(np.mean(recent))
    sigma = float(np.std(recent))
    prices[-1] = mu + 3.5 * sigma

    signal = mean_reversion_strategy("MKT-3", prices, settings)
    assert signal.side == "buy_no"


def test_mean_reversion_confidence_scales():
    """confidence = min(|deviation| / 5.0, 1.0): verify scaling and cap."""
    settings = make_settings()
    lookback = 168

    # Helper: given a price list, compute the deviation the strategy will see.
    def actual_deviation(prices):
        recent = prices[-lookback:]
        mu = float(np.mean(recent))
        std = float(np.std(recent, ddof=1))  # Match BugFix M2: sample std
        return (prices[-1] - mu) / std if std != 0 else 0.0

    # --- 3-sigma confidence case ---
    # Use a multiplier of 3.5x against prior-window stats so that after the
    # extreme value enters the 168-price window the deviation lands near 3.0.
    prices_3 = make_prices(n=200, mean=0.5, std=0.05)
    prior = prices_3[-(lookback + 1):-1]
    mu_p = float(np.mean(prior))
    std_p = float(np.std(prior))
    prices_3[-1] = mu_p - 3.5 * std_p  # produces ~3.3 actual deviation
    dev3 = actual_deviation(prices_3)
    expected_conf3 = min(abs(dev3) / 5.0, 1.0)
    signal_3 = mean_reversion_strategy("MKT-4a", prices_3, settings)
    assert signal_3.side == "buy_yes"
    assert abs(signal_3.confidence - expected_conf3) < 1e-9

    # --- 5-sigma cap case ---
    # Use 6x to ensure the deviation exceeds 5.0 after window recalculation.
    prices_5 = make_prices(n=200, mean=0.5, std=0.05)
    prior5 = prices_5[-(lookback + 1):-1]
    mu5 = float(np.mean(prior5))
    std5 = float(np.std(prior5))
    prices_5[-1] = mu5 - 6.0 * std5  # produces ~5.6 actual deviation -> capped
    signal_5 = mean_reversion_strategy("MKT-4b", prices_5, settings)
    assert signal_5.confidence == 1.0


def test_mean_reversion_insufficient_data():
    """Fewer than 20 prices -> no_trade due to insufficient history."""
    settings = make_settings()
    prices = [0.5] * 10  # only 10 prices, well below min(lookback=168, 20)

    signal = mean_reversion_strategy("MKT-5", prices, settings)
    assert signal.side == "no_trade"
    assert "Insufficient" in signal.reasoning
