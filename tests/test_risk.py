"""Tests for RiskManager — corrected Kelly formula and risk controls."""

from __future__ import annotations

import pytest

from weatheredge.agents.risk_manager import RiskManager
from weatheredge.config import Settings


def make_settings(**overrides) -> Settings:
    """Build a Settings instance with test-friendly defaults (no .env required)."""
    defaults = dict(
        polymarket_private_key="",
        polymarket_proxy_address="",
        kalshi_api_key="",
        kalshi_private_key="",
        anthropic_api_key="",
        telegram_bot_token="",
        telegram_chat_id="",
        kelly_fraction=0.15,
        max_open_positions=20,
        max_single_market_exposure=0.08,
        max_cost_per_trade_fraction=1.0,  # disable cost cap in unit tests
        target_volatility=0.10,
        max_leverage=2.0,
        min_allocation=0.20,
        vpin_danger_threshold=0.70,
        current_risk_tier=100,
        initial_bankroll=1000.0,
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# Kelly sizing tests
# ---------------------------------------------------------------------------


def test_kelly_small_edge_approx_4_84():
    """model_prob=0.70, market_price=0.69 → 1-point gross edge.

    Post-fee-fix: entry=0.02, exit=0.03, net_win = 0.31 - 0.05 = 0.26
    b = 0.26/0.71 ≈ 0.366; kelly = 0.7 - 0.3/0.366 < 0 → 0.
    Net Kelly correctly rejects sub-fee edges.
    """
    rm = RiskManager(make_settings())
    rm.set_resolved_trade_count(100)  # bypass warmup Kelly
    size = rm.calculate_position_size(
        win_probability=0.70,
        market_price=0.69,
        bankroll=1000.0,
    )
    assert size == 0.0, f"Expected $0 (sub-fee edge), got {size}"


def test_kelly_large_edge_capped_at_80():
    """model_prob=0.70, market_price=0.25 → large edge → capped at 8% = $80.

    Post-fee-fix: entry=0.02, exit=0.03, net_win=0.70, b=0.70/0.27≈2.593
    kelly ≈ 0.584 → raw size = 87.64 → capped to $80.
    """
    rm = RiskManager(make_settings())
    rm.set_resolved_trade_count(100)  # bypass warmup Kelly
    size = rm.calculate_position_size(
        win_probability=0.70,
        market_price=0.25,
        bankroll=1000.0,
    )
    assert size == pytest.approx(80.0, abs=0.10), f"Expected $80.00, got {size}"


def test_kelly_zero_when_no_edge():
    """win_prob <= market_price → return 0.0."""
    rm = RiskManager(make_settings())
    assert rm.calculate_position_size(0.50, 0.50, 1000.0) == 0.0
    assert rm.calculate_position_size(0.49, 0.50, 1000.0) == 0.0
    assert rm.calculate_position_size(0.35, 0.70, 1000.0) == 0.0


# ---------------------------------------------------------------------------
# Circuit breaker tests
# ---------------------------------------------------------------------------


def test_circuit_breaker_fires_at_15pct_tier100():
    """$100 tier: max_total_drawdown=15%. At -$151 on $1000 bankroll → halt."""
    rm = RiskManager(make_settings(current_risk_tier=100))
    result = rm.check_circuit_breakers(
        daily_pnl=0.0,
        total_pnl=-151.0,
        bankroll=1000.0,
    )
    assert result["halt"] is True
    assert "15%" in result["reason"]


def test_circuit_breaker_no_fire_at_14pct_tier100():
    """$100 tier: -$140 on $1000 (14%) should NOT halt."""
    rm = RiskManager(make_settings(current_risk_tier=100))
    result = rm.check_circuit_breakers(
        daily_pnl=0.0,
        total_pnl=-140.0,
        bankroll=1000.0,
    )
    assert result["halt"] is False


def test_circuit_breaker_fires_at_25pct_tier500():
    """$500 tier: max_total_drawdown=25%. At -$251 on $1000 bankroll → halt."""
    rm = RiskManager(make_settings(current_risk_tier=500))
    result = rm.check_circuit_breakers(
        daily_pnl=0.0,
        total_pnl=-251.0,
        bankroll=1000.0,
    )
    assert result["halt"] is True
    assert "25%" in result["reason"]


# ---------------------------------------------------------------------------
# VPIN tests
# ---------------------------------------------------------------------------


def test_vpin_toxic_above_threshold():
    """VPIN > 0.70 → toxic=True."""
    rm = RiskManager(make_settings(vpin_danger_threshold=0.70))
    fills = [
        {"side": "buy", "size": 90},
        {"side": "sell", "size": 10},
    ]  # VPIN = |90-10|/100 = 0.80
    result = rm.check_adverse_selection(fills)
    assert result["toxic"] is True
    assert result["vpin"] == pytest.approx(0.80)


def test_vpin_not_toxic_below_threshold():
    """VPIN = 0.30 → toxic=False."""
    rm = RiskManager(make_settings(vpin_danger_threshold=0.70))
    fills = [
        {"side": "buy", "size": 65},
        {"side": "sell", "size": 35},
    ]  # VPIN = |65-35|/100 = 0.30
    result = rm.check_adverse_selection(fills)
    assert result["toxic"] is False
    assert result["vpin"] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# City exposure cap tests
# ---------------------------------------------------------------------------


class _FakePosition:
    """Minimal stand-in for a Trade ORM object."""

    def __init__(self, city: str, cost: float):
        self.city = city
        self.cost = cost


def test_city_exposure_blocks_above_16pct():
    """Existing $100 + proposed $70 = $170 > $160 (16% of $1000) → blocked."""
    rm = RiskManager(make_settings())
    positions = [_FakePosition("NYC", 100.0)]
    result = rm.check_city_exposure("NYC", 70.0, positions, bankroll=1000.0)
    assert result["blocked"] is True
    assert "NYC" in result["reason"]


def test_city_exposure_allows_under_16pct():
    """Existing $100 + proposed $50 = $150 < $160 → allowed."""
    rm = RiskManager(make_settings())
    positions = [_FakePosition("NYC", 100.0)]
    result = rm.check_city_exposure("NYC", 50.0, positions, bankroll=1000.0)
    assert result["blocked"] is False


# ---------------------------------------------------------------------------
# pre_trade_check integration tests
# ---------------------------------------------------------------------------


def test_pre_trade_check_blocks_on_circuit_breaker():
    """Kill switch active → pre_trade_check returns blocked=True."""
    rm = RiskManager(make_settings(current_risk_tier=100))
    result = rm.pre_trade_check(
        city="NYC",
        proposed_size=50.0,
        open_positions=[],
        bankroll=1000.0,
        daily_pnl=0.0,
        total_pnl=-200.0,  # 20% drawdown → exceeds 15% tier-100 limit
    )
    assert result["blocked"] is True


def test_pre_trade_check_blocks_on_max_positions():
    """At max_open_positions → blocked."""
    rm = RiskManager(make_settings(max_open_positions=3))
    positions = [_FakePosition("NYC", 10.0) for _ in range(3)]
    result = rm.pre_trade_check(
        city="Chicago",
        proposed_size=20.0,
        open_positions=positions,
        bankroll=1000.0,
        daily_pnl=0.0,
        total_pnl=0.0,
    )
    assert result["blocked"] is True
    assert "Max open positions" in result["reason"]


def test_pre_trade_check_counts_unique_markets_for_position_limit():
    """Duplicate rows for the same market should not consume multiple slots."""
    rm = RiskManager(make_settings(max_open_positions=2))
    positions = [
        {"market_id": "1814995", "cost": 10.0},
        {"market_id": 1814995, "cost": 11.0},  # same market, different type
    ]
    result = rm.pre_trade_check(
        city="Chicago",
        proposed_size=20.0,
        open_positions=positions,
        bankroll=1000.0,
        daily_pnl=0.0,
        total_pnl=0.0,
    )
    assert result["blocked"] is False


def test_pre_trade_check_blocks_on_city_exposure():
    """City at cap → blocked."""
    rm = RiskManager(make_settings())
    positions = [_FakePosition("Miami", 155.0)]
    result = rm.pre_trade_check(
        city="Miami",
        proposed_size=10.0,
        open_positions=positions,
        bankroll=1000.0,
        daily_pnl=0.0,
        total_pnl=0.0,
    )
    assert result["blocked"] is True
    assert "Miami" in result["reason"]


def test_pre_trade_check_passes_clean():
    """No violations → blocked=False."""
    rm = RiskManager(make_settings())
    result = rm.pre_trade_check(
        city="Denver",
        proposed_size=50.0,
        open_positions=[],
        bankroll=1000.0,
        daily_pnl=0.0,
        total_pnl=0.0,
    )
    assert result["blocked"] is False


def test_pre_trade_check_reduce_size_on_toxic_vpin():
    """Toxic VPIN → blocked=False but reduce_size=True."""
    rm = RiskManager(make_settings(vpin_danger_threshold=0.70))
    toxic_fills = [{"side": "buy", "size": 90}, {"side": "sell", "size": 10}]
    result = rm.pre_trade_check(
        city="LAX",
        proposed_size=50.0,
        open_positions=[],
        bankroll=1000.0,
        daily_pnl=0.0,
        total_pnl=0.0,
        recent_fills=toxic_fills,
    )
    assert result["blocked"] is False
    assert result.get("reduce_size") is True
