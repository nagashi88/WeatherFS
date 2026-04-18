"""Tests for _persist_signals — per-agent audit trail for the decision loop.

Covers:
- Multi-agent signals get persisted with correct fields
- no_trade signals are persisted (audit completeness)
- Edge field uses net_edge for buy_yes / net_edge_no for buy_no, None for no_trade
- Empty input is a no-op
- Missing opportunity gracefully degrades to None fields
- Persistence uses add_all+flush (single round-trip + IDs)
- _link_signals_to_trade back-link populates signals.trade_id by primary key
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import select

from weatheredge.agents.base import MarketOpportunity, Signal
from weatheredge.db.database import get_session, init_db, make_engine
from weatheredge.db.models import Signal as SignalRow
from weatheredge.main import _link_signals_to_trade, _persist_signals


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


# ---------------------------------------------------------------------------
# _link_signals_to_trade — P1 fix #4 back-link
# ---------------------------------------------------------------------------


def _read_trade_ids(engine) -> dict[int, int | None]:
    with get_session(engine=engine) as session:
        rows = session.execute(select(SignalRow)).scalars().all()
        return {r.id: r.trade_id for r in rows}


def test_back_link_after_successful_trade(engine, patched_session):
    """The signals from this cycle get their trade_id populated when a trade
    fills. Lets a losing trade JOIN back to every per-agent vote later.

    Fix #8 changed the linking contract from a time-window filter
    (`created_at >= cycle_start`) to explicit primary-key lookup using IDs
    returned by `_persist_signals`. The new test threads IDs end-to-end."""
    sigs = {
        "mkt1": [
            Signal(market_id="mkt1", side="buy_yes", confidence=0.8, agent_name="Aggressive"),
            Signal(market_id="mkt1", side="buy_yes", confidence=0.6, agent_name="Conservative"),
        ],
    }
    ids_by_market = _persist_signals(sigs, {"mkt1": _opp("mkt1")})
    assert len(ids_by_market["mkt1"]) == 2

    updated = _link_signals_to_trade(42, ids_by_market["mkt1"])

    assert updated == 2
    trade_ids = _read_trade_ids(engine)
    assert all(tid == 42 for tid in trade_ids.values())


def test_link_does_not_touch_other_markets(engine, patched_session):
    """Signals belonging to a different market stay untouched — only the IDs
    explicitly passed are mutated."""
    sigs = {
        "mkt_kept": [Signal(market_id="mkt_kept", side="buy_yes", confidence=0.7, agent_name="Aggressive")],
        "mkt_other": [Signal(market_id="mkt_other", side="buy_yes", confidence=0.5, agent_name="Aggressive")],
    }
    ids_by_market = _persist_signals(
        sigs, {"mkt_kept": _opp("mkt_kept"), "mkt_other": _opp("mkt_other")}
    )

    updated = _link_signals_to_trade(99, ids_by_market["mkt_kept"])

    assert updated == 1
    with get_session(engine=engine) as session:
        rows = session.execute(select(SignalRow)).scalars().all()
        by_market = {r.market_id: r.trade_id for r in rows}
    assert by_market["mkt_kept"] == 99
    assert by_market["mkt_other"] is None


def test_link_does_not_touch_stale_pre_cycle_signals(engine, patched_session):
    """Signals written before the cycle must NOT be relinked.

    Under Fix #8 the protection is structural rather than time-based: only
    the IDs returned by *this* cycle's `_persist_signals` call are passed
    in, so a stale row from a previous cycle is invisible to the linker."""
    # Write a stale signal (simulates a previous cycle's leftover NULL row).
    stale_ts = datetime.now(timezone.utc) - timedelta(minutes=10)
    with get_session(engine=engine) as session:
        session.add(SignalRow(
            market_id="mkt1",
            agent_name="StaleAgent",
            signal_type="buy_yes",
            created_at=stale_ts,
        ))

    ids_by_market = _persist_signals(
        {"mkt1": [Signal(market_id="mkt1", side="buy_yes", confidence=0.8, agent_name="FreshAgent")]},
        {"mkt1": _opp("mkt1")},
    )

    updated = _link_signals_to_trade(7, ids_by_market["mkt1"])

    assert updated == 1
    with get_session(engine=engine) as session:
        rows = session.execute(select(SignalRow)).scalars().all()
        by_agent = {r.agent_name: r.trade_id for r in rows}
    assert by_agent["FreshAgent"] == 7
    assert by_agent["StaleAgent"] is None  # untouched


def test_link_skips_already_linked_signals(engine, patched_session):
    """If a signal already has a trade_id (rare, but defensive), do not overwrite.

    The `trade_id IS NULL` filter inside `_link_signals_to_trade` is the
    safety net — even if a stale ID gets passed in, an existing link wins."""
    # Insert two signals manually and capture IDs via flush.
    with get_session(engine=engine) as session:
        already = SignalRow(
            market_id="mkt1",
            agent_name="AlreadyLinked",
            signal_type="buy_yes",
            trade_id=111,
            created_at=datetime.now(timezone.utc),
        )
        unlinked = SignalRow(
            market_id="mkt1",
            agent_name="Unlinked",
            signal_type="buy_yes",
            created_at=datetime.now(timezone.utc),
        )
        session.add_all([already, unlinked])
        session.flush()
        already_id = int(already.id)
        unlinked_id = int(unlinked.id)

    # Pass BOTH IDs — the filter should preserve the already-linked row.
    updated = _link_signals_to_trade(222, [already_id, unlinked_id])

    assert updated == 1
    with get_session(engine=engine) as session:
        rows = session.execute(select(SignalRow)).scalars().all()
        by_agent = {r.agent_name: r.trade_id for r in rows}
    assert by_agent["AlreadyLinked"] == 111  # preserved
    assert by_agent["Unlinked"] == 222


def test_link_returns_zero_on_missing_inputs(engine, patched_session):
    """Defensive: trade_id=0 or empty signal_ids must short-circuit and never
    issue a wildcard UPDATE."""
    ids_by_market = _persist_signals(
        {"mkt1": [Signal(market_id="mkt1", side="buy_yes", confidence=0.7, agent_name="A")]},
        {"mkt1": _opp("mkt1")},
    )
    sig_ids = ids_by_market["mkt1"]

    assert _link_signals_to_trade(0, sig_ids) == 0
    assert _link_signals_to_trade(1, []) == 0

    # Confirm nothing was relinked.
    trade_ids = _read_trade_ids(engine)
    assert all(tid is None for tid in trade_ids.values())
