"""Tests for FillMonitor — fill quality recording and adverse selection detection."""

from __future__ import annotations

import pytest

from weatheredge.db.database import init_db, make_engine
from weatheredge.execution.fill_monitor import FillMonitor, FillRecord


@pytest.fixture()
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def monitor(engine):
    return FillMonitor(db_engine=engine)


# ---------------------------------------------------------------------------
# record_fill
# ---------------------------------------------------------------------------

def test_record_fill_stores_to_db(monitor, engine):
    """record_fill() must persist a FillQuality row to the database."""
    from weatheredge.db.database import get_session
    from weatheredge.db.models import FillQuality

    record = FillRecord(
        trade_id=1,
        fill_price=0.52,
        market_mid_at_fill=0.50,
        price_5s_after=0.48,
    )
    monitor.record_fill(record)

    with get_session(engine) as session:
        row = session.query(FillQuality).filter_by(trade_id=1).first()
        assert row is not None
        assert row.fill_price == pytest.approx(0.52)
        assert row.market_mid_at_fill == pytest.approx(0.50)
        assert row.price_5s_after == pytest.approx(0.48)


def test_slippage_calculated_correctly(monitor, engine):
    """Slippage = fill_price - market_mid_at_fill."""
    from weatheredge.db.database import get_session
    from weatheredge.db.models import FillQuality

    record = FillRecord(
        trade_id=2,
        fill_price=0.55,
        market_mid_at_fill=0.50,
        price_5s_after=None,
    )
    monitor.record_fill(record)

    with get_session(engine) as session:
        row = session.query(FillQuality).filter_by(trade_id=2).first()
        assert row.slippage == pytest.approx(0.05)
        assert row.adverse_move is None


# ---------------------------------------------------------------------------
# check_adverse_selection
# ---------------------------------------------------------------------------

def _add_fills(monitor, count: int, adverse: bool):
    """Helper: add `count` fills, each either adverse or not."""
    for i in range(count):
        if adverse:
            # mid goes down after fill → adverse_move = mid - price_5s_after > 0
            record = FillRecord(trade_id=100 + i, fill_price=0.50, market_mid_at_fill=0.50, price_5s_after=0.45)
        else:
            # mid goes up after fill → adverse_move <= 0 (not adverse)
            record = FillRecord(trade_id=200 + i, fill_price=0.50, market_mid_at_fill=0.50, price_5s_after=0.55)
        monitor.record_fill(record)


def test_adverse_selection_detected(monitor):
    """When >70% of recent fills are adverse, adverse=True must be returned."""
    # 15 adverse + 5 non-adverse = 75% adverse
    _add_fills(monitor, count=15, adverse=True)
    _add_fills(monitor, count=5, adverse=False)

    result = monitor.check_adverse_selection(last_n=20)

    assert result["adverse"] is True
    assert result["adverse_pct"] == pytest.approx(0.75)
    assert result["fills_checked"] == 20


def test_no_adverse_selection(monitor):
    """When only 30% of fills are adverse, adverse=False must be returned."""
    # 6 adverse + 14 non-adverse = 30% adverse
    _add_fills(monitor, count=6, adverse=True)
    _add_fills(monitor, count=14, adverse=False)

    result = monitor.check_adverse_selection(last_n=20)

    assert result["adverse"] is False
    assert result["adverse_pct"] == pytest.approx(0.30)


def test_insufficient_fills_returns_false(monitor):
    """Fewer than 5 fills must return adverse=False with an explanatory reason."""
    _add_fills(monitor, count=3, adverse=True)

    result = monitor.check_adverse_selection(last_n=20)

    assert result["adverse"] is False
    assert "Insufficient" in result["reason"]
