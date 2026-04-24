from __future__ import annotations

from datetime import datetime, timedelta, timezone

import weatheredge.config as cfg
from weatheredge.db.database import get_session, init_db, make_engine
from weatheredge.db.models import Trade
from weatheredge.execution.paper_trader import Order, PaperTrader


def test_test_database_is_not_live_database():
    """Autouse fixture must redirect DB writes away from the paper-trading DB."""
    from weatheredge.db import database as db

    resolved = db._resolve_db_path()
    assert resolved.name == "weatheredge_test.db"
    assert resolved != db._DEFAULT_DB_PATH


def test_temporal_guard_blocks_orders_without_required_timestamps(monkeypatch):
    monkeypatch.setitem(cfg.MODULES, "temporal_guard", True)
    trader = PaperTrader(initial_bankroll=100.0)

    result = trader.place_order(Order(
        market_id="missing_ts",
        side="buy_yes",
        price=0.40,
        amount=5.0,
    ))

    assert result["filled"] is False
    assert result["reason"] == "missing_temporal_timestamps"
    assert trader.open_positions == []


def test_pending_limit_order_persists_reloads_and_fills_same_db_row(tmp_path, monkeypatch):
    monkeypatch.setitem(cfg.MODULES, "temporal_guard", True)
    engine = make_engine(f"sqlite:///{tmp_path / 'paper.db'}")
    init_db(engine)

    forecast_ts = datetime.now(timezone.utc) - timedelta(minutes=10)
    price_ts = datetime.now(timezone.utc) - timedelta(minutes=1)

    trader = PaperTrader(initial_bankroll=100.0)
    trader.connect_db(engine)
    result = trader.place_order(Order(
        market_id="limit_same_row",
        side="buy_yes",
        price=0.50,
        amount=10.0,
        order_type="limit",
        forecast_timestamp=forecast_ts,
        market_price_timestamp=price_ts,
        signal_ids=[101, 102],
    ))

    assert result["pending"] is True
    pending_id = result["db_id"]
    assert pending_id is not None

    with get_session(engine) as session:
        row = session.get(Trade, pending_id)
        assert row is not None
        assert row.status == "pending"
        assert row.cost == 0.0
        assert row.limit_price == 0.48
        assert row.forecast_timestamp == forecast_ts.replace(tzinfo=None)
        assert row.market_price_timestamp == price_ts.replace(tzinfo=None)

    restarted = PaperTrader(initial_bankroll=100.0)
    restarted.connect_db(engine)
    assert len(restarted.open_orders) == 1
    assert restarted.open_orders[0]["db_id"] == pending_id

    filled = restarted.check_limit_fills({"limit_same_row": 0.47})

    assert filled == 1
    assert len(restarted.open_orders) == 0
    assert len(restarted.open_positions) == 1
    assert restarted.open_positions[0]["db_id"] == pending_id

    with get_session(engine) as session:
        rows = session.query(Trade).filter(Trade.market_id == "limit_same_row").all()
        assert len(rows) == 1
        assert rows[0].id == pending_id
        assert rows[0].status == "open"
        assert rows[0].order_type == "limit"
        assert rows[0].cost > 0.0


def test_expired_pending_limit_order_persists_cancelled_status(tmp_path, monkeypatch):
    monkeypatch.setitem(cfg.MODULES, "temporal_guard", False)
    engine = make_engine(f"sqlite:///{tmp_path / 'paper.db'}")
    init_db(engine)

    trader = PaperTrader(initial_bankroll=100.0)
    trader.connect_db(engine)
    result = trader.place_order(Order(
        market_id="limit_cancel",
        side="buy_yes",
        price=0.50,
        amount=10.0,
        order_type="limit",
    ))
    pending_id = result["db_id"]
    trader.open_orders[0]["created_at"] = datetime.now(timezone.utc) - timedelta(hours=3)

    cancelled = trader.cancel_expired_orders()

    assert cancelled == 1
    assert trader.open_orders == []
    with get_session(engine) as session:
        row = session.get(Trade, pending_id)
        assert row.status == "cancelled"
        assert row.pnl == 0.0
