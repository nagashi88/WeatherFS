"""Tests for weatheredge.daily_summary — daily aggregation/observability job.

Covers:
- Empty day → zero counters across the board
- Resolved trades counted, open trades excluded from win-rate math
- Per-agent P&L splits correctly
- Cash-cap heuristic triggers on sub-unit amounts
- Agent disagreement: same-side multi-agent ≠ disagreement; split sides → 100%
- Parser failure stats include the rolling-window data
- format_daily_summary_text produces a non-empty multi-line string
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, time, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy.orm import sessionmaker

from weatheredge.data.market_parser import (
    parse_polymarket_title,
    reset_parse_history_for_tests,
)
from weatheredge.db.database import init_db, make_engine
from weatheredge.db.models import Signal as SignalRow
from weatheredge.db.models import Trade as TradeRow


@pytest.fixture()
def engine():
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def patched_session(engine):
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

    # daily_summary imports get_session at call time, so patch in both modules
    with patch("weatheredge.daily_summary.get_session", fake_get_session):
        yield


@pytest.fixture(autouse=True)
def _reset_parser_history():
    reset_parse_history_for_tests()
    yield
    reset_parse_history_for_tests()


def _today_utc():
    return datetime.now(timezone.utc).date()


def _today_at(hour: int, minute: int = 0) -> datetime:
    """Return a UTC datetime inside today's window for clean test data."""
    return datetime.combine(_today_utc(), time(hour, minute), tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# compute_daily_summary
# ---------------------------------------------------------------------------


def test_empty_day_returns_zeroed_summary(engine, patched_session):
    from weatheredge.daily_summary import compute_daily_summary

    s = compute_daily_summary()

    assert s["trades"] == 0
    assert s["resolved_trades"] == 0
    assert s["winning_trades"] == 0
    assert s["losing_trades"] == 0
    assert s["win_rate"] == 0.0
    assert s["realized_pnl"] == 0.0
    assert s["per_agent_pnl"] == {}
    assert s["per_regime_pnl"] == {}
    assert s["signal_count"] == 0
    assert s["agent_disagreement_rate"] == 0.0
    assert s["cash_cap_hits"] == 0


_RESOLVED_TEST_STATUSES = {
    "won", "lost", "edge_stop", "capital_protect", "time_decay",
    "ev_based", "abandoned",
}


def _add_trade(engine, **overrides):
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = factory()
    defaults = dict(
        market_id="mkt1",
        platform="polymarket",
        mode="paper",
        side="buy_yes",
        price=0.5,
        amount=10.0,
        cost=5.0,
        agent_name="Aggressive",
        status="won",
        pnl=2.5,
        created_at=_today_at(12),
    )
    defaults.update(overrides)
    # Auto-fill resolved_at when status implies a resolved trade and the
    # caller did not set it explicitly. Daily summary now buckets resolved
    # P&L by resolved_at (not created_at), so tests need realistic timestamps
    # in the same UTC day to assert wins/losses/realized_pnl.
    if (
        defaults["status"] in _RESOLVED_TEST_STATUSES
        and "resolved_at" not in overrides
    ):
        defaults["resolved_at"] = _today_at(13)
    row = TradeRow(**defaults)
    session.add(row)
    session.commit()
    session.close()


def _add_signal(engine, **overrides):
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = factory()
    defaults = dict(
        market_id="mkt1",
        agent_name="Aggressive",
        signal_type="buy_yes",
        created_at=_today_at(12),
    )
    defaults.update(overrides)
    row = SignalRow(**defaults)
    session.add(row)
    session.commit()
    session.close()


def test_resolved_trades_counted_open_excluded(engine, patched_session):
    from weatheredge.daily_summary import compute_daily_summary

    _add_trade(engine, status="won", pnl=3.0, agent_name="A")
    _add_trade(engine, status="lost", pnl=-1.0, agent_name="A")
    _add_trade(engine, status="open", pnl=None, agent_name="A")

    s = compute_daily_summary()

    assert s["trades"] == 3
    assert s["resolved_trades"] == 2
    assert s["winning_trades"] == 1
    assert s["losing_trades"] == 1
    assert s["win_rate"] == pytest.approx(0.5)
    assert s["realized_pnl"] == pytest.approx(2.0)


def test_per_agent_pnl_splits_correctly(engine, patched_session):
    from weatheredge.daily_summary import compute_daily_summary

    _add_trade(engine, status="won", pnl=2.0, agent_name="Aggressive")
    _add_trade(engine, status="lost", pnl=-1.0, agent_name="Aggressive")
    _add_trade(engine, status="won", pnl=4.0, agent_name="Conservative")

    s = compute_daily_summary()

    assert set(s["per_agent_pnl"].keys()) == {"Aggressive", "Conservative"}
    assert s["per_agent_pnl"]["Aggressive"]["pnl"] == pytest.approx(1.0)
    assert s["per_agent_pnl"]["Aggressive"]["wins"] == 1
    assert s["per_agent_pnl"]["Aggressive"]["losses"] == 1
    assert s["per_agent_pnl"]["Conservative"]["pnl"] == pytest.approx(4.0)
    assert s["per_agent_pnl"]["Conservative"]["wins"] == 1


def test_per_regime_pnl_splits_correctly(engine, patched_session):
    from weatheredge.daily_summary import compute_daily_summary

    _add_trade(engine, status="won", pnl=2.0, regime="floor_bound_buy_no")
    _add_trade(engine, status="lost", pnl=-1.0, regime="floor_bound_buy_no")
    _add_trade(engine, status="won", pnl=4.0, regime="metar_override_yes")

    s = compute_daily_summary()

    assert set(s["per_regime_pnl"].keys()) == {
        "floor_bound_buy_no", "metar_override_yes",
    }
    fb = s["per_regime_pnl"]["floor_bound_buy_no"]
    assert fb["pnl"] == pytest.approx(1.0)
    assert fb["wins"] == 1
    assert fb["losses"] == 1
    assert s["per_regime_pnl"]["metar_override_yes"]["pnl"] == pytest.approx(4.0)


def test_per_regime_pnl_counts_edge_stop_and_ev_based(engine, patched_session):
    """edge_stop and ev_based are surfaced as first-class regime counters
    so H7 (edge_flip reactiveness) and Phase 1 EV-exit success are visible."""
    from weatheredge.daily_summary import compute_daily_summary

    _add_trade(engine, status="edge_stop", pnl=-0.5, regime="floor_bound_buy_no")
    _add_trade(engine, status="edge_stop", pnl=0.3, regime="floor_bound_buy_no")
    _add_trade(engine, status="ev_based", pnl=1.2, regime="floor_bound_buy_no")

    s = compute_daily_summary()

    fb = s["per_regime_pnl"]["floor_bound_buy_no"]
    assert fb["edge_stop"] == 2
    assert fb["ev_based"] == 1
    assert fb["wins"] == 0
    assert fb["losses"] == 0
    assert fb["resolved"] == 3
    assert fb["pnl"] == pytest.approx(1.0)


def test_per_regime_pnl_bucketizes_missing_regime_as_unknown(engine, patched_session):
    """Trades pre-dating the live tagger or backfilled with NULL get
    aggregated under 'unknown' rather than dropping them silently."""
    from weatheredge.daily_summary import compute_daily_summary

    _add_trade(engine, status="won", pnl=1.0, regime=None)

    s = compute_daily_summary()
    assert "unknown" in s["per_regime_pnl"]
    assert s["per_regime_pnl"]["unknown"]["wins"] == 1


def test_early_exit_status_counts_in_resolved_pnl(engine, patched_session):
    """edge_stop / capital_protect / time_decay are real exits — they must
    contribute to realized P&L even though they are not 'won' or 'lost'."""
    from weatheredge.daily_summary import compute_daily_summary

    _add_trade(engine, status="edge_stop", pnl=1.5, agent_name="A")
    _add_trade(engine, status="capital_protect", pnl=-0.5, agent_name="A")

    s = compute_daily_summary()

    assert s["resolved_trades"] == 2
    assert s["realized_pnl"] == pytest.approx(1.0)
    assert s["winning_trades"] == 0  # only "won" counts as a win
    assert s["losing_trades"] == 0
    assert s["per_agent_pnl"]["A"]["pnl"] == pytest.approx(1.0)


def test_cash_cap_hits_counted_via_explicit_flag(engine, patched_session):
    """cash_cap_hits reads the `cash_capped` column set by PortfolioManager
    when the per-market cash cap reduced order size. Amount-based heuristics
    were dropped — they produced false positives for legitimately small
    orders (e.g. a 0.5-contract trade below cap).
    """
    from weatheredge.daily_summary import compute_daily_summary

    _add_trade(engine, amount=10.0, cash_capped=False)  # full size — no cap
    _add_trade(engine, amount=0.42, cash_capped=True)   # cash-cap reduction
    _add_trade(engine, amount=0.05, cash_capped=True)   # ditto

    s = compute_daily_summary()
    assert s["cash_cap_hits"] == 2


def test_cash_cap_hits_ignores_small_amount_when_flag_false(engine, patched_session):
    """Legitimately small orders (below the 1-contract mark but *not* cash-capped)
    must NOT be counted. The previous heuristic (amount<1.0) caught these by
    mistake."""
    from weatheredge.daily_summary import compute_daily_summary

    _add_trade(engine, amount=0.3, cash_capped=False)
    _add_trade(engine, amount=0.7, cash_capped=False)

    s = compute_daily_summary()
    assert s["cash_cap_hits"] == 0


def test_agent_disagreement_zero_when_same_side(engine, patched_session):
    from weatheredge.daily_summary import compute_daily_summary

    _add_signal(engine, market_id="m1", agent_name="A", signal_type="buy_yes")
    _add_signal(engine, market_id="m1", agent_name="B", signal_type="buy_yes")

    s = compute_daily_summary()
    assert s["agent_disagreement_rate"] == 0.0


def test_agent_disagreement_full_when_split(engine, patched_session):
    from weatheredge.daily_summary import compute_daily_summary

    _add_signal(engine, market_id="m1", agent_name="A", signal_type="buy_yes")
    _add_signal(engine, market_id="m1", agent_name="B", signal_type="buy_no")
    _add_signal(engine, market_id="m2", agent_name="A", signal_type="buy_yes")
    _add_signal(engine, market_id="m2", agent_name="B", signal_type="buy_yes")

    s = compute_daily_summary()
    # m1 splits, m2 agrees → 1/2 = 0.5
    assert s["agent_disagreement_rate"] == pytest.approx(0.5)


def test_no_trade_signals_excluded_from_disagreement(engine, patched_session):
    """A 'no_trade' vote alongside a 'buy_yes' is not real disagreement —
    one agent saw no edge, the other did. Only conflicting BUY sides count."""
    from weatheredge.daily_summary import compute_daily_summary

    _add_signal(engine, market_id="m1", agent_name="A", signal_type="buy_yes")
    _add_signal(engine, market_id="m1", agent_name="B", signal_type="no_trade")

    s = compute_daily_summary()
    assert s["agent_disagreement_rate"] == 0.0


def test_parser_failure_stats_propagated(engine, patched_session):
    from weatheredge.daily_summary import compute_daily_summary

    parse_polymarket_title("Will the highest temperature in NYC be 50°F or above on April 4?")
    parse_polymarket_title("garbage title")

    s = compute_daily_summary()
    assert s["parser_failures"]["attempts"] == 2
    assert s["parser_failures"]["failures"] == 1


def test_yesterday_window_excludes_today(engine, patched_session):
    """Filtering by created_at must respect the requested target_day."""
    from weatheredge.daily_summary import compute_daily_summary

    _add_trade(engine, status="won", pnl=10.0, created_at=_today_at(12))

    yesterday = _today_utc() - timedelta(days=1)
    s = compute_daily_summary(target_day=yesterday)

    assert s["trades"] == 0
    assert s["realized_pnl"] == 0.0


# ---------------------------------------------------------------------------
# format_daily_summary_text
# ---------------------------------------------------------------------------


def test_format_text_includes_key_fields():
    from weatheredge.daily_summary import format_daily_summary_text

    summary = {
        "day": "2026-04-18",
        "trades": 5,
        "resolved_trades": 3,
        "winning_trades": 2,
        "losing_trades": 1,
        "win_rate": 2 / 3,
        "realized_pnl": 12.5,
        "per_agent_pnl": {"Aggressive": {"trades": 3, "resolved": 2, "wins": 2, "losses": 0, "pnl": 8.0}},
        "parser_failures": {"attempts": 100, "failures": 4, "failure_rate": 0.04},
        "signal_count": 50,
        "agent_disagreement_rate": 0.20,
        "cash_cap_hits": 2,
    }
    text = format_daily_summary_text(summary)

    assert "2026-04-18" in text
    assert "$+12.50" in text
    assert "Aggressive" in text
    assert "66.7%" in text  # win rate
    assert "20.0%" in text  # disagreement
    assert "Cash-cap hits: 2" in text
    assert "Parser: 4/100" in text


def test_format_text_includes_per_regime_section_when_present():
    from weatheredge.daily_summary import format_daily_summary_text

    summary = {
        "day": "2026-04-26",
        "trades": 4, "resolved_trades": 3, "winning_trades": 1,
        "losing_trades": 1, "win_rate": 1 / 3, "realized_pnl": 0.5,
        "per_agent_pnl": {},
        "per_regime_pnl": {
            "floor_bound_buy_no": {
                "trades": 3, "resolved": 3, "wins": 1, "losses": 1,
                "edge_stop": 1, "ev_based": 0, "pnl": 0.5,
            },
        },
        "parser_failures": {"attempts": 0, "failures": 0, "failure_rate": 0.0},
        "signal_count": 0, "agent_disagreement_rate": 0.0, "cash_cap_hits": 0,
    }
    text = format_daily_summary_text(summary)

    assert "Per-regime P&L" in text
    assert "floor_bound_buy_no" in text
    assert "1 edge_stop" in text
    assert "0 ev_based" in text


def test_format_text_handles_zero_parser_attempts():
    from weatheredge.daily_summary import format_daily_summary_text

    summary = {
        "day": "2026-04-18",
        "trades": 0, "resolved_trades": 0, "winning_trades": 0,
        "losing_trades": 0, "win_rate": 0.0, "realized_pnl": 0.0,
        "per_agent_pnl": {}, "parser_failures": {"attempts": 0, "failures": 0, "failure_rate": 0.0},
        "signal_count": 0, "agent_disagreement_rate": 0.0, "cash_cap_hits": 0,
    }
    text = format_daily_summary_text(summary)
    assert "Parser: 0 attempts" in text
