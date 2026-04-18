"""Tests for _persist_signals — per-agent audit trail for the decision loop.

Covers:
- Multi-agent signals get persisted with correct fields
- no_trade signals are persisted (audit completeness)
- Edge field uses net_edge for buy_yes / net_edge_no for buy_no, None for no_trade
- Empty input is a no-op
- Missing opportunity gracefully degrades to None fields
- Persistence uses bulk_save_objects (single round-trip)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import select

from weatheredge.agents.base import MarketOpportunity, Signal
from weatheredge.db.database import get_session, init_db, make_engine
from weatheredge.db.models import Signal as SignalRow
from weatheredge.main import _persist_signals


@pytest.fixture()
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def patched_session(engine):
    """Patch get_session in main to use our in-memory engine."""
    from contextlib import contextmanager
    from sqlalchemy.orm import sessionmaker

    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    @contextmanager
    def fake_get_session(_engine=None):
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    with patch("weatheredge.main.get_session", fake_get_session):
        yield


def _opp(market_id: str, **overrides) -> MarketOpportunity:
    defaults = dict(
        market_id=market_id,
        platform="polymarket",
        city="New York",
        station="KNYC",
        threshold=70.0,
        direction="above",
        market_price=0.45,
        model_probability=0.62,
        net_edge=0.12,
        net_edge_no=-0.08,
    )
    defaults.update(overrides)
    return MarketOpportunity(**defaults)


def _read_all(engine) -> list[dict]:
    """Snapshot signal rows into plain dicts so callers can assert outside
    the session (avoids DetachedInstanceError on lazy-loaded attributes)."""
    with get_session(engine=engine) as session:
        rows = session.execute(select(SignalRow)).scalars().all()
        return [
            {
                "id": r.id,
                "market_id": r.market_id,
                "agent_name": r.agent_name,
                "signal_type": r.signal_type,
                "model_probability": r.model_probability,
                "market_price": r.market_price,
                "edge": r.edge,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_persists_multi_agent_signals(engine, patched_session):
    sigs = {
        "mkt1": [
            Signal(market_id="mkt1", side="buy_yes", confidence=0.8, agent_name="Aggressive", reasoning="strong edge"),
            Signal(market_id="mkt1", side="buy_yes", confidence=0.6, agent_name="Conservative", reasoning="moderate"),
            Signal(market_id="mkt1", side="no_trade", confidence=0.0, agent_name="MeanReversion", reasoning="no signal"),
        ],
    }
    opps = {"mkt1": _opp("mkt1")}

    _persist_signals(sigs, opps)

    rows = _read_all(engine)
    assert len(rows) == 3
    by_agent = {r["agent_name"]: r for r in rows}
    assert by_agent["Aggressive"]["signal_type"] == "buy_yes"
    assert by_agent["Aggressive"]["confidence"] == 0.8
    assert by_agent["Aggressive"]["edge"] == pytest.approx(0.12)  # net_edge for buy_yes
    assert by_agent["Aggressive"]["market_price"] == 0.45
    assert by_agent["Aggressive"]["model_probability"] == 0.62
    assert by_agent["MeanReversion"]["edge"] is None  # no_trade has no edge
    assert by_agent["MeanReversion"]["signal_type"] == "no_trade"


def test_buy_no_uses_net_edge_no(engine, patched_session):
    sigs = {
        "mkt1": [
            Signal(market_id="mkt1", side="buy_no", confidence=0.7, agent_name="SellNO"),
        ],
    }
    opps = {"mkt1": _opp("mkt1", net_edge=0.05, net_edge_no=0.18)}

    _persist_signals(sigs, opps)

    rows = _read_all(engine)
    assert len(rows) == 1
    assert rows[0]["edge"] == pytest.approx(0.18)  # net_edge_no, not net_edge


def test_empty_input_is_noop(engine, patched_session):
    _persist_signals({}, {})
    assert _read_all(engine) == []


def test_market_with_no_signals_skipped(engine, patched_session):
    _persist_signals({"mkt1": []}, {"mkt1": _opp("mkt1")})
    assert _read_all(engine) == []


def test_missing_opportunity_writes_none_audit_fields(engine, patched_session):
    """Signals can arrive for markets whose opportunity dropped from the dict
    (e.g., dedupe between phases). Persist with None audit fields rather than
    crash — the agent vote itself is still useful."""
    sigs = {
        "orphan": [
            Signal(market_id="orphan", side="buy_yes", confidence=0.5, agent_name="Aggressive"),
        ],
    }
    _persist_signals(sigs, {})  # no opportunity for "orphan"

    rows = _read_all(engine)
    assert len(rows) == 1
    assert rows[0]["agent_name"] == "Aggressive"
    assert rows[0]["signal_type"] == "buy_yes"
    assert rows[0]["market_price"] is None
    assert rows[0]["model_probability"] is None
    assert rows[0]["edge"] is None


def test_multiple_markets_same_call(engine, patched_session):
    sigs = {
        "mkt_a": [Signal(market_id="mkt_a", side="buy_yes", confidence=0.7, agent_name="Aggressive")],
        "mkt_b": [Signal(market_id="mkt_b", side="buy_no", confidence=0.6, agent_name="SellNO")],
    }
    opps = {
        "mkt_a": _opp("mkt_a"),
        "mkt_b": _opp("mkt_b", net_edge_no=0.20),
    }

    _persist_signals(sigs, opps)

    rows = _read_all(engine)
    assert len(rows) == 2
    by_market = {r["market_id"]: r for r in rows}
    assert by_market["mkt_a"]["signal_type"] == "buy_yes"
    assert by_market["mkt_b"]["signal_type"] == "buy_no"
    assert by_market["mkt_b"]["edge"] == pytest.approx(0.20)


def test_empty_agent_name_falls_back_to_unknown(engine, patched_session):
    """agent_name is NOT NULL in the DB schema; defensive fallback prevents
    integrity errors if a strategy forgets to set it."""
    sigs = {
        "mkt1": [
            Signal(market_id="mkt1", side="buy_yes", confidence=0.5, agent_name=""),
        ],
    }
    opps = {"mkt1": _opp("mkt1")}

    _persist_signals(sigs, opps)

    rows = _read_all(engine)
    assert len(rows) == 1
    assert rows[0]["agent_name"] == "unknown"
