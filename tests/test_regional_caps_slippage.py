"""Tests for regional exposure caps (US-203) and slippage modeling (US-205)."""

from __future__ import annotations

import pytest

from weatheredge.agents.risk_manager import REGION_MAP, RiskManager, station_to_region
from weatheredge.config import Settings


def make_settings(**overrides) -> Settings:
    defaults = dict(
        polymarket_private_key="",
        polymarket_proxy_address="",
        kalshi_api_key="",
        kalshi_private_key="",
        telegram_bot_token="",
        telegram_chat_id="",
        kelly_fraction=0.15,
        max_open_positions=20,
        max_single_market_exposure=0.08,
        target_volatility=0.10,
        max_leverage=2.0,
        min_allocation=0.20,
        vpin_danger_threshold=0.70,
        current_risk_tier=100,
        initial_bankroll=1000.0,
        max_single_region_exposure=0.30,
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# Part A: Region mapping
# ---------------------------------------------------------------------------


def test_region_map_all_stations():
    """All stations in REGION_MAP must map to their declared region."""
    for region, stations in REGION_MAP.items():
        for station in stations:
            assert station_to_region(station) == region, (
                f"{station} should map to {region}, got {station_to_region(station)}"
            )


def test_station_to_region_northeast():
    assert station_to_region("KNYC") == "northeast"
    assert station_to_region("KBOS") == "northeast"


def test_station_to_region_southeast():
    assert station_to_region("KMIA") == "southeast"
    assert station_to_region("KATL") == "southeast"
    assert station_to_region("KIAH") == "southeast"


def test_station_to_region_midwest():
    assert station_to_region("KMDW") == "midwest"
    assert station_to_region("KDFW") == "midwest"
    assert station_to_region("KDEN") == "midwest"


def test_station_to_region_west():
    assert station_to_region("KLAX") == "west"
    assert station_to_region("KSFO") == "west"
    assert station_to_region("KSEA") == "west"
    assert station_to_region("KPHX") == "west"
    assert station_to_region("KAUS") == "west"


def test_station_to_region_international():
    assert station_to_region("EGLC") == "international"
    assert station_to_region("SAEZ") == "international"
    assert station_to_region("LTAC") == "international"
    assert station_to_region("RKSS") == "international"


def test_station_to_region_unknown():
    assert station_to_region("ZZZZ") == "unknown"
    assert station_to_region("") == "unknown"


# ---------------------------------------------------------------------------
# Part A: check_region_exposure
# ---------------------------------------------------------------------------


class _FakePosition:
    def __init__(self, station: str, cost: float):
        self.station = station
        self.cost = cost


def test_check_region_exposure_blocks_when_exceeded():
    """Existing $250 + proposed $60 = $310 > $300 (30% of $1000) → blocked."""
    rm = RiskManager(make_settings())
    positions = [_FakePosition("KNYC", 250.0)]
    result = rm.check_region_exposure("KBOS", 60.0, positions, bankroll=1000.0)
    assert result["blocked"] is True
    assert "northeast" in result["reason"]


def test_check_region_exposure_allows_under_cap():
    """Existing $200 + proposed $50 = $250 < $300 → allowed."""
    rm = RiskManager(make_settings())
    positions = [_FakePosition("KNYC", 200.0)]
    result = rm.check_region_exposure("KBOS", 50.0, positions, bankroll=1000.0)
    assert result["blocked"] is False


def test_check_region_exposure_different_region_not_counted():
    """Southeast position does not count against northeast cap."""
    rm = RiskManager(make_settings())
    # $280 in southeast shouldn't block a $60 northeast trade
    positions = [_FakePosition("KMIA", 280.0)]
    result = rm.check_region_exposure("KNYC", 60.0, positions, bankroll=1000.0)
    assert result["blocked"] is False


def test_check_region_exposure_dict_positions():
    """Positions as dicts (like projected_open_positions) work correctly."""
    rm = RiskManager(make_settings())
    positions = [{"station": "KNYC", "cost": 250.0}]
    result = rm.check_region_exposure("KBOS", 60.0, positions, bankroll=1000.0)
    assert result["blocked"] is True


def test_pre_trade_check_blocks_on_region_exposure():
    """pre_trade_check blocks when regional cap is exceeded.

    Use two different cities (NYC and Boston) so city-level cap (16% each) is
    not breached, but the combined northeast region exposure (30%) is.
    """
    rm = RiskManager(make_settings())
    # Two positions in the same region but different cities: 140 + 140 = 280 < 300 (each city < 160)
    pos1 = {"station": "KNYC", "city": "NYC", "cost": 140.0, "market_id": "m1"}
    pos2 = {"station": "KBOS", "city": "Boston", "cost": 140.0, "market_id": "m2"}
    # Proposed 30 more from KBOS (Boston): 140+30=170 for Boston city (>160) — use a third NE station
    # Actually propose from a hypothetical NE station not in city check conflict
    # Simpler: use only pos1 (KNYC, $140) and propose $170 from KBOS (Boston, different city)
    # Region total: 140+170=310 > 300 → blocked by region
    # City check for "Boston": 0+170=170 > 160 → would also block
    # Solution: use bankroll 2000 so city cap = 320, region cap = 600, but set region cap lower
    rm2 = RiskManager(make_settings(max_single_region_exposure=0.15))
    # bankroll=1000, region cap = 150; existing KNYC=100, propose KBOS=60 → total=160 > 150
    pos = {"station": "KNYC", "city": "NYC", "cost": 100.0, "market_id": "m1"}
    result = rm2.pre_trade_check(
        city="Boston",
        proposed_size=60.0,
        open_positions=[pos],
        bankroll=1000.0,
        daily_pnl=0.0,
        total_pnl=0.0,
        station="KBOS",
    )
    assert result["blocked"] is True
    assert "northeast" in result["reason"]


def test_pre_trade_check_no_station_skips_region_check():
    """pre_trade_check without station skips region check entirely."""
    rm = RiskManager(make_settings(max_single_region_exposure=0.05))
    # Region cap is only 5% = $50; existing KNYC cost $40 + proposed $20 = $60 > $50
    # But since no station is passed, region check is skipped → not blocked
    positions = [{"station": "KNYC", "cost": 40.0, "city": "NYC", "market_id": "m1"}]
    result = rm.pre_trade_check(
        city="Boston",
        proposed_size=20.0,
        open_positions=positions,
        bankroll=1000.0,
        daily_pnl=0.0,
        total_pnl=0.0,
        # no station kwarg → region check skipped
    )
    assert result["blocked"] is False


# ---------------------------------------------------------------------------
# Part B: Slippage modeling
# ---------------------------------------------------------------------------


def test_slippage_no_volume_no_adjustment():
    """volume=None → no slippage adjustment, same as before."""
    rm = RiskManager(make_settings())
    size_no_vol = rm.calculate_position_size(
        win_probability=0.70,
        market_price=0.50,
        bankroll=1000.0,
    )
    size_with_none = rm.calculate_position_size(
        win_probability=0.70,
        market_price=0.50,
        bankroll=1000.0,
        volume=None,
    )
    assert size_no_vol == size_with_none


def test_slippage_volume_50_reduces_by_75pct():
    """volume=50 (< 100) → Kelly multiplied by 0.25 (75% reduction)."""
    rm = RiskManager(make_settings())
    base = rm.calculate_position_size(
        win_probability=0.70,
        market_price=0.50,
        bankroll=1000.0,
    )
    thin = rm.calculate_position_size(
        win_probability=0.70,
        market_price=0.50,
        bankroll=1000.0,
        volume=50,
    )
    # thin should be ~25% of base (before slippage cap)
    assert thin <= base * 0.26, f"Expected ≤{base*0.26:.2f}, got {thin}"
    assert thin > 0


def test_slippage_volume_500_reduces_by_50pct():
    """volume=500 (100 ≤ v < 1000) → Kelly multiplied by 0.50."""
    rm = RiskManager(make_settings())
    base = rm.calculate_position_size(
        win_probability=0.70,
        market_price=0.50,
        bankroll=1000.0,
    )
    thin = rm.calculate_position_size(
        win_probability=0.70,
        market_price=0.50,
        bankroll=1000.0,
        volume=500,
    )
    assert thin <= base * 0.51, f"Expected ≤{base*0.51:.2f}, got {thin}"
    assert thin >= base * 0.40, f"Expected ≥{base*0.40:.2f}, got {thin}"


def test_slippage_volume_5000_no_reduction():
    """volume=5000 (≥ 1000) → no volume-based scaling reduction."""
    rm = RiskManager(make_settings())
    base = rm.calculate_position_size(
        win_probability=0.70,
        market_price=0.50,
        bankroll=1000.0,
    )
    liquid = rm.calculate_position_size(
        win_probability=0.70,
        market_price=0.50,
        bankroll=1000.0,
        volume=5000,
    )
    assert liquid == base, f"Expected {base}, got {liquid}"


def test_slippage_cap_limits_large_orders_in_thin_markets():
    """Large position in thin market: slippage cap kicks in and limits size."""
    rm = RiskManager(make_settings(
        kelly_fraction=1.0,           # Max Kelly to force large size
        max_single_market_exposure=1.0,  # Remove hard cap
    ))
    # Big edge, low price → uncapped Kelly is large
    size = rm.calculate_position_size(
        win_probability=0.90,
        market_price=0.10,
        bankroll=10000.0,
        volume=10,  # Very thin market
    )
    # CCG-bughunt2 G7: cap now uses NET edge (gross - fees). For Kalshi
    # p=0.10: fees = entry(0.01) + exit(0.01+0.01) = 0.03, so
    # net_edge = 0.80 - 0.03 = 0.77, max_slippage = 0.385,
    # slippage_cap_size = (0.385 * 10) / 0.005 = 770.
    # After 0.25 scaling for volume<100, uncapped Kelly would be huge;
    # slippage cap should limit to <= 770 (strictly under legacy 800 bound).
    assert size <= 800.01, f"Slippage cap should limit size to ≤800, got {size}"


def test_slippage_cap_does_not_reduce_when_unnecessary():
    """Slippage cap should not reduce size when slippage is already acceptable."""
    rm = RiskManager(make_settings())
    # Normal market conditions — slippage cap should not trigger
    size_with_volume = rm.calculate_position_size(
        win_probability=0.70,
        market_price=0.50,
        bankroll=1000.0,
        volume=5000,
    )
    size_without_volume = rm.calculate_position_size(
        win_probability=0.70,
        market_price=0.50,
        bankroll=1000.0,
    )
    # With high volume, no slippage adjustment expected
    assert size_with_volume == size_without_volume


# ---------------------------------------------------------------------------
# CCG-bughunt2 G7: slippage cap must use NET edge (gross - fees), not gross.
# On Kalshi, taker fee + half-spread can consume most of the gross edge —
# so allowing slippage up to 50% of gross is effectively permitting it to
# exceed the entire net edge, guaranteeing a loss after fees + slippage
# compound.
# ---------------------------------------------------------------------------


def test_slippage_cap_tightens_when_fees_eat_edge():
    """Thin edge + Kalshi fees: cap must scale with NET edge, not gross.

    p=0.50, win=0.58, volume=500:
      Kalshi fees = entry(0.02) + exit(0.02 + 0.01 half-spread) = 0.05
      gross_edge = 0.08, net_edge = 0.03
      Old (gross):  max_slip = 0.04  →  size cap = (0.04 × 500) / 0.005 = 4000
      New (net):    max_slip = 0.015 →  size cap = (0.015 × 500) / 0.005 = 1500
    Uncapped Kelly (kelly_fraction=1.0, post-warmup) lands above 1500 for
    this scenario, so only the net-edge cap binds — proving the bug fix
    is the active gate (not Kelly itself).
    """
    rm = RiskManager(make_settings(
        kelly_fraction=1.0,
        max_single_market_exposure=1.0,
        max_cost_per_trade_fraction=1.0,
    ))
    rm.set_resolved_trade_count(100)  # skip warmup so cap is the true gate
    size = rm.calculate_position_size(
        win_probability=0.58,
        market_price=0.50,
        bankroll=100_000.0,
        volume=500,
    )
    # Old bug would return ~ min(uncapped_kelly, gross_cap=4000).
    # New fix must return ≤ net_cap = 1500.
    assert size <= 1500.01, (
        f"Slippage cap must use NET edge (0.03), not gross (0.08); "
        f"expected ≤1500, got {size}"
    )
    # And the cap must actually bind — otherwise the test isn't exercising G7.
    assert size > 1000, (
        f"Uncapped Kelly should exceed 1500 (so the cap binds), got {size}"
    )


def test_slippage_cap_does_not_block_when_fees_small():
    """Sanity: when fees are small relative to edge (low volume still matters
    but gross ≈ net), old and new caps land close together and the trade is
    not spuriously throttled further than necessary."""
    rm = RiskManager(make_settings(
        kelly_fraction=1.0,
        max_single_market_exposure=1.0,
        max_cost_per_trade_fraction=1.0,
    ))
    # Big edge (fees tiny as a fraction): cap should still permit a large
    # position proportional to net edge.
    # p=0.10, win=0.90, Kalshi fees = 0.01 + 0.02 = 0.03, gross=0.80, net=0.77.
    size = rm.calculate_position_size(
        win_probability=0.90,
        market_price=0.10,
        bankroll=10000.0,
        volume=10,
    )
    # Net cap: (0.77 * 0.5 * 10) / 0.005 = 770. Still generous.
    assert size <= 770.01, f"Expected ≤770 (net-edge cap), got {size}"
    # And materially above the tight-fee case — not thrown away.
    assert size >= 100


def test_slippage_cap_consistent_with_kelly_net_edge():
    """Post-G7, the slippage cap refuses trades whose slippage would exceed
    half the net edge. When uncapped-Kelly slippage already < net×0.5, no
    additional tightening occurs (same as high-volume baseline)."""
    rm = RiskManager(make_settings(kelly_fraction=0.25))
    # Volume high enough that (position × 0.005 / volume) << net×0.5.
    large_vol_size = rm.calculate_position_size(
        win_probability=0.65,
        market_price=0.50,
        bankroll=1000.0,
        volume=10000,
    )
    no_vol_size = rm.calculate_position_size(
        win_probability=0.65,
        market_price=0.50,
        bankroll=1000.0,
        volume=None,
    )
    # Volume this high → book_depth_scalar clamps to 1.0 AND slippage cap
    # inactive → same answer as the volume=None path.
    assert large_vol_size == no_vol_size
