"""Tests for EdgeDecayMonitor and CompetitorTracker."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from weatheredge.db.database import init_db, get_session, make_engine
from weatheredge.db.models import EdgeDecay, WalletProfile
from weatheredge.intelligence.edge_decay_monitor import EdgeDecayMonitor
from weatheredge.intelligence.competitor_tracker import CompetitorTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    yield eng
    eng.dispose()


def _make_wallet(address: str, sharpe: float | None, win_rate: float | None,
                 total_pnl: float | None, total_trades: int,
                 profiled_at: datetime | None = None) -> WalletProfile:
    return WalletProfile(
        wallet_address=address,
        profile_text="test profile",
        sharpe_ratio=sharpe,
        win_rate=win_rate,
        total_pnl=total_pnl,
        total_trades=total_trades,
        profiled_at=profiled_at or datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# EdgeDecayMonitor tests
# ---------------------------------------------------------------------------

def test_edge_decay_detects_low_sharpe(engine):
    """Top wallets with avg Sharpe 1.2 should trigger is_decaying=True."""
    with get_session(engine=engine) as session:
        for i in range(5):
            session.add(_make_wallet(f"0xlow{i}", sharpe=1.2, win_rate=0.55,
                                     total_pnl=100.0, total_trades=100))

    monitor = EdgeDecayMonitor(engine)
    result = monitor.weekly_check()

    assert result["checked"] is True
    assert result["is_decaying"] is True
    assert abs(result["avg_sharpe"] - 1.2) < 1e-6


def test_edge_decay_stable(engine):
    """Top wallets with avg Sharpe 2.5 should be stable (is_decaying=False)."""
    with get_session(engine=engine) as session:
        for i in range(5):
            session.add(_make_wallet(f"0xhigh{i}", sharpe=2.5, win_rate=0.65,
                                     total_pnl=500.0, total_trades=150))

    monitor = EdgeDecayMonitor(engine)
    result = monitor.weekly_check()

    assert result["checked"] is True
    assert result["is_decaying"] is False
    assert abs(result["avg_sharpe"] - 2.5) < 1e-6


def test_edge_decay_insufficient_data(engine):
    """Fewer than 3 profiled wallets should return checked=False."""
    with get_session(engine=engine) as session:
        session.add(_make_wallet("0xonly1", sharpe=2.0, win_rate=0.6,
                                 total_pnl=200.0, total_trades=80))
        session.add(_make_wallet("0xonly2", sharpe=1.8, win_rate=0.58,
                                 total_pnl=180.0, total_trades=70))

    monitor = EdgeDecayMonitor(engine)
    result = monitor.weekly_check()

    assert result["checked"] is False
    assert "reason" in result


def test_edge_decay_stores_to_db(engine):
    """After weekly_check(), a new EdgeDecay row must exist in the DB."""
    with get_session(engine=engine) as session:
        for i in range(4):
            session.add(_make_wallet(f"0xstore{i}", sharpe=2.0, win_rate=0.6,
                                     total_pnl=300.0, total_trades=120))

    monitor = EdgeDecayMonitor(engine)
    result = monitor.weekly_check()

    assert result["checked"] is True

    from sqlalchemy import select as sa_select
    with get_session(engine=engine) as session:
        rows = session.execute(sa_select(EdgeDecay)).scalars().all()
        assert len(rows) == 1
        assert abs(rows[0].top10_avg_sharpe - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# CompetitorTracker tests
# ---------------------------------------------------------------------------

def test_competitor_tracks_new_wallets(engine):
    """Wallet profiled 2 days ago with 60 trades should be returned."""
    recent = datetime.now(timezone.utc) - timedelta(days=2)
    with get_session(engine=engine) as session:
        session.add(_make_wallet("0xnew1", sharpe=1.8, win_rate=0.6,
                                 total_pnl=200.0, total_trades=60,
                                 profiled_at=recent))

    tracker = CompetitorTracker(engine)
    results = tracker.track_new_wallets(min_trades=50)

    assert len(results) == 1
    assert results[0]["address"] == "0xnew1"
    assert results[0]["trades"] == 60


def test_competitor_ignores_old_wallets(engine):
    """Wallet profiled 10 days ago should not be returned."""
    old = datetime.now(timezone.utc) - timedelta(days=10)
    with get_session(engine=engine) as session:
        session.add(_make_wallet("0xold1", sharpe=1.8, win_rate=0.6,
                                 total_pnl=200.0, total_trades=60,
                                 profiled_at=old))

    tracker = CompetitorTracker(engine)
    results = tracker.track_new_wallets(min_trades=50)

    assert len(results) == 0


def test_competitor_ignores_low_volume(engine):
    """Wallet with only 20 trades should not meet the default threshold."""
    recent = datetime.now(timezone.utc) - timedelta(days=2)
    with get_session(engine=engine) as session:
        session.add(_make_wallet("0xlowvol", sharpe=1.5, win_rate=0.55,
                                 total_pnl=50.0, total_trades=20,
                                 profiled_at=recent))

    tracker = CompetitorTracker(engine)
    results = tracker.track_new_wallets(min_trades=50)

    assert len(results) == 0


def test_get_top_competitors(engine):
    """get_top_competitors should return wallets sorted by PnL descending."""
    with get_session(engine=engine) as session:
        pnls = [100.0, 500.0, 250.0, 750.0, 50.0]
        for i, pnl in enumerate(pnls):
            session.add(_make_wallet(f"0xtop{i}", sharpe=1.5, win_rate=0.6,
                                     total_pnl=pnl, total_trades=100))

    tracker = CompetitorTracker(engine)
    results = tracker.get_top_competitors(n=5)

    assert len(results) == 5
    pnl_values = [r["pnl"] for r in results]
    assert pnl_values == sorted(pnl_values, reverse=True)
    assert pnl_values[0] == 750.0
