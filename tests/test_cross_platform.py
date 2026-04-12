"""Tests for CrossPlatformArbitrage — cross-platform spread detection."""

from __future__ import annotations

import pytest

from weatheredge.execution.cross_platform import CrossPlatformArbitrage, ArbOpportunity


@pytest.fixture()
def arb():
    return CrossPlatformArbitrage()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poly(city, date, threshold, yes_price):
    return {"city": city, "date": date, "threshold": threshold, "yes_price": yes_price}


def _kalshi(city, date, threshold, yes_price):
    return {"city": city, "date": date, "threshold": threshold, "yes_price": yes_price}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_matching_markets_found(arb):
    """Same city+date+threshold with 10% gross spread → net spread after fees.

    BugFix M6: spread is now net of platform fees (gross - 1.5% poly - 2% kalshi).
    """
    poly = [_poly("NYC", "2026-04-01", 75.0, 0.40)]
    kalshi = [_kalshi("NYC", "2026-04-01", 75.0, 0.50)]

    opps = arb.scan(poly, kalshi)

    assert len(opps) == 1
    opp = opps[0]
    assert opp.city == "NYC"
    assert opp.date == "2026-04-01"
    assert opp.threshold == pytest.approx(75.0)
    assert opp.poly_price == pytest.approx(0.40)
    assert opp.kalshi_price == pytest.approx(0.50)
    # Net spread = 10% gross - 1.5% poly fee - 2% kalshi fee = 6.5%
    assert opp.spread == pytest.approx(0.065)


def test_no_match_different_cities(arb):
    """Markets with different cities must not be matched."""
    poly = [_poly("NYC", "2026-04-01", 75.0, 0.40)]
    kalshi = [_kalshi("LAX", "2026-04-01", 75.0, 0.50)]

    opps = arb.scan(poly, kalshi)

    assert opps == []


def test_spread_below_threshold(arb):
    """A 3% spread (below the 5% minimum) must not produce an opportunity."""
    poly = [_poly("NYC", "2026-04-01", 75.0, 0.47)]
    kalshi = [_kalshi("NYC", "2026-04-01", 75.0, 0.50)]

    opps = arb.scan(poly, kalshi)

    assert opps == []


def test_direction_correct(arb):
    """When poly is cheaper than kalshi, direction must be buy_poly_sell_kalshi."""
    poly = [_poly("Chicago", "2026-04-02", 80.0, 0.38)]
    kalshi = [_kalshi("Chicago", "2026-04-02", 80.0, 0.50)]

    opps = arb.scan(poly, kalshi)

    assert len(opps) == 1
    assert opps[0].direction == "buy_poly_sell_kalshi"


def test_direction_reversed_when_kalshi_cheaper(arb):
    """When kalshi is cheaper than poly, direction must be buy_kalshi_sell_poly."""
    poly = [_poly("Miami", "2026-04-03", 90.0, 0.60)]
    kalshi = [_kalshi("Miami", "2026-04-03", 90.0, 0.45)]

    opps = arb.scan(poly, kalshi)

    assert len(opps) == 1
    assert opps[0].direction == "buy_kalshi_sell_poly"
