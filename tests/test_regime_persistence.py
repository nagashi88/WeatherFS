"""Integration tests: regime tag survives place_order persistence.

Covers the live persistence path that 1.B's backfill complements:
TradeOrder.regime → Order.regime → trade_dict["regime"] → Trade.regime row.
"""
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from weatheredge.db.database import init_db
from weatheredge.db.models import Trade
from weatheredge.execution.paper_trader import Order, PaperTrader


@pytest.fixture
def trader_with_db():
    eng = create_engine("sqlite:///:memory:")
    init_db(engine=eng)
    pt = PaperTrader(initial_bankroll=100.0)
    pt.connect_db(eng)
    yield pt, eng
    eng.dispose()


def _market_order(**overrides) -> Order:
    base = dict(
        market_id="M1",
        side="buy_no",
        price=0.30,
        amount=2.0,
        platform="kalshi",
        forecast_timestamp=datetime.now(timezone.utc),
        market_price_timestamp=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return Order(**base)


def test_market_order_persists_regime(trader_with_db, monkeypatch):
    pt, eng = trader_with_db
    # Bypass temporal guard for isolation; covered elsewhere.
    from weatheredge.config import MODULES
    monkeypatch.setitem(MODULES, "temporal_guard", False)

    result = pt.place_order(_market_order(regime="floor_bound_buy_no"))
    assert result["filled"] is True

    with Session(eng) as session:
        rows = session.query(Trade).all()
        assert len(rows) == 1
        assert rows[0].regime == "floor_bound_buy_no"


def test_limit_order_persists_regime_on_pending_row(trader_with_db, monkeypatch):
    pt, eng = trader_with_db
    from weatheredge.config import MODULES
    monkeypatch.setitem(MODULES, "temporal_guard", False)

    result = pt.place_order(
        _market_order(order_type="limit", regime="threshold_yes")
    )
    assert result["pending"] is True

    with Session(eng) as session:
        rows = session.query(Trade).all()
        assert len(rows) == 1
        assert rows[0].status == "pending"
        assert rows[0].regime == "threshold_yes"


def test_market_order_with_no_regime_persists_null(trader_with_db, monkeypatch):
    pt, eng = trader_with_db
    from weatheredge.config import MODULES
    monkeypatch.setitem(MODULES, "temporal_guard", False)

    pt.place_order(_market_order())  # regime=None default

    with Session(eng) as session:
        assert session.query(Trade).one().regime is None


def test_limit_fill_carries_regime_to_open_row(trader_with_db, monkeypatch):
    pt, eng = trader_with_db
    from weatheredge.config import MODULES
    monkeypatch.setitem(MODULES, "temporal_guard", False)

    pt.place_order(
        _market_order(order_type="limit", regime="metar_override_yes",
                      side="buy_yes", price=0.85)
    )
    # Drive a fill: for buy_yes, fills when current_yes <= limit_price
    # (limit_price = price - offset). Use an aggressive enough price.
    fills = pt.check_limit_fills({"M1": 0.50})
    assert fills == 1

    with Session(eng) as session:
        rows = session.query(Trade).all()
        assert len(rows) == 1
        assert rows[0].status == "open"
        assert rows[0].regime == "metar_override_yes"
