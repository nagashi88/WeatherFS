"""Tests for ResolutionTracker — trade resolution and daily performance tracking."""

from __future__ import annotations

import pytest

from weatheredge.db.database import init_db, make_engine, get_session
from weatheredge.db.models import Trade, DailyPerformance
from weatheredge.learning.resolution_tracker import ResolutionTracker


@pytest.fixture()
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def tracker(engine):
    return ResolutionTracker(engine=engine)


def _make_trade(engine, market_id: str, side: str, price: float = 0.40,
                amount: float = 10.0, cost: float = 4.0) -> int:
    """Insert an open trade and return its id."""
    with get_session(engine) as session:
        trade = Trade(
            market_id=market_id,
            platform="polymarket",
            mode="paper",
            side=side,
            price=price,
            amount=amount,
            cost=cost,
            status="open",
        )
        session.add(trade)
        session.flush()
        trade_id = trade.id
    return trade_id


# ---------------------------------------------------------------------------
# resolve_trades
# ---------------------------------------------------------------------------

def test_resolve_winning_trade(tracker, engine):
    """buy_yes trade with outcome=yes must be marked won with positive pnl."""
    trade_id = _make_trade(engine, market_id="mkt-1", side="buy_yes", amount=10.0, cost=4.0)

    resolved = tracker.resolve_trades("mkt-1", "yes")

    assert len(resolved) == 1
    assert resolved[0]["status"] == "won"
    # Polymarket: pnl = amount - cost (0% fee on winnings)
    assert resolved[0]["pnl"] > 0

    with get_session(engine) as session:
        trade = session.get(Trade, trade_id)
        assert trade.status == "won"
        assert trade.pnl == pytest.approx(6.0)
        assert trade.resolved_at is not None


def test_resolve_losing_trade(tracker, engine):
    """buy_yes trade with outcome=no must be marked lost with negative pnl."""
    trade_id = _make_trade(engine, market_id="mkt-2", side="buy_yes", amount=10.0, cost=4.0)

    resolved = tracker.resolve_trades("mkt-2", "no")

    assert len(resolved) == 1
    assert resolved[0]["status"] == "lost"
    assert resolved[0]["pnl"] < 0  # -cost = -4

    with get_session(engine) as session:
        trade = session.get(Trade, trade_id)
        assert trade.status == "lost"
        assert trade.pnl == pytest.approx(-4.0)
        assert trade.resolved_at is not None


def test_daily_performance_updated(tracker, engine):
    """After resolution, DailyPerformance row has correct pnl, wins, losses."""
    _make_trade(engine, market_id="mkt-3", side="buy_yes", amount=10.0, cost=4.0)
    _make_trade(engine, market_id="mkt-3", side="buy_no", amount=10.0, cost=3.0)

    # outcome=yes: buy_yes wins (pnl=6), buy_no loses (pnl=-3)
    tracker.resolve_trades("mkt-3", "yes")

    with get_session(engine) as session:
        dp = session.query(DailyPerformance).first()
        assert dp is not None
        assert dp.total_trades == 2
        assert dp.winning_trades == 1
        assert dp.losing_trades == 1
        # Polymarket: buy_yes win pnl = 10-4 = 6.0, buy_no loss = -3.0, total = 3.0
        assert dp.total_pnl == pytest.approx(3.0)


def test_daily_performance_accumulates(tracker, engine):
    """Resolving two batches on the same day accumulates into one row."""
    _make_trade(engine, market_id="mkt-4a", side="buy_yes", amount=10.0, cost=4.0)
    _make_trade(engine, market_id="mkt-4b", side="buy_yes", amount=10.0, cost=4.0)

    tracker.resolve_trades("mkt-4a", "yes")
    tracker.resolve_trades("mkt-4b", "yes")

    with get_session(engine) as session:
        rows = session.query(DailyPerformance).all()
        assert len(rows) == 1
        assert rows[0].total_trades == 2
        assert rows[0].winning_trades == 2
        # Polymarket: each win pnl = 10-4 = 6.0, total = 12.0
        assert rows[0].total_pnl == pytest.approx(12.0)


def test_handles_no_open_trades(tracker, engine):
    """Market with no open trades must return an empty list."""
    result = tracker.resolve_trades("nonexistent-market", "yes")
    assert result == []


# ---------------------------------------------------------------------------
# get_unresolved_markets
# ---------------------------------------------------------------------------

def test_get_unresolved_markets(tracker, engine):
    """Two markets with open trades must both appear in the result."""
    _make_trade(engine, market_id="mkt-a", side="buy_yes")
    _make_trade(engine, market_id="mkt-b", side="buy_no")

    markets = tracker.get_unresolved_markets()

    assert set(markets) == {"mkt-a", "mkt-b"}


def test_get_unresolved_markets_excludes_resolved(tracker, engine):
    """Resolved trades must not appear in get_unresolved_markets()."""
    _make_trade(engine, market_id="mkt-open", side="buy_yes")
    _make_trade(engine, market_id="mkt-closed", side="buy_yes")

    # Resolve one market
    tracker.resolve_trades("mkt-closed", "yes")

    markets = tracker.get_unresolved_markets()
    assert markets == ["mkt-open"]
