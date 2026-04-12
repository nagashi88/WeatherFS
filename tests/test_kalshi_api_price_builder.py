from __future__ import annotations

from datetime import date

from weatheredge.backtesting.kalshi_api_price_builder import (
    _BucketQuote,
    _implied_rows_for_event,
    _parse_event_date,
    _pick_market_prob,
)


def test_parse_event_date():
    assert _parse_event_date("KXHIGHDEN-26APR08") == date(2026, 4, 8)
    assert _parse_event_date("BAD") is None


def test_pick_market_prob_priority():
    market = {
        "previous_price_dollars": "0.42",
        "last_price_dollars": "0.51",
        "settlement_value_dollars": "1.00",
    }
    assert _pick_market_prob(market) == 0.42

    market2 = {"previous_price_dollars": None, "last_price_dollars": "0.33"}
    assert _pick_market_prob(market2) == 0.33


def test_implied_rows_for_event_from_bucket_quotes():
    quotes = [
        _BucketQuote(low=None, high=40.0, prob=0.10),
        _BucketQuote(low=40.0, high=45.0, prob=0.20),
        _BucketQuote(low=45.0, high=50.0, prob=0.30),
        _BucketQuote(low=50.0, high=None, prob=0.40),
    ]
    rows = _implied_rows_for_event("KNYC", date(2026, 4, 8), quotes)

    # Grid spans full midpoint range (including tails): 35..55 every 5°F.
    # 5 thresholds x (above+below) => 10 rows.
    assert len(rows) == 10

    lookup = {(r[2], r[3]): float(r[4]) for r in rows}
    # Midpoint approximation: above >=45 -> 0.30 + 0.40 = 0.70
    assert abs(lookup[("45.00", "above")] - 0.70) < 1e-9
    assert abs(lookup[("45.00", "below")] - 0.30) < 1e-9
