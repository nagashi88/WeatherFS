"""Tests for weatheredge.dashboard.live_dashboard — P2 panels.

Covers:
- query_open_positions joins with latest market_prices for current_yes
- query_per_agent_pnl groups by agent_name with the right SUMs
- _row_unrealized_pnl computes per-position mark-to-market for buy_yes/buy_no
- build_open_positions_panel renders the new Unr. P&L column
- build_per_agent_pnl_panel renders rows
- build_layout still wires every panel and runs without raising
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from weatheredge.dashboard.live_dashboard import (
    _row_unrealized_pnl,
    build_layout,
    build_open_positions_panel,
    build_per_agent_pnl_panel,
    query_open_positions,
    query_per_agent_pnl,
    query_performance,
)
from weatheredge.db.database import init_db, make_engine


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    """SQLite connection on a fresh DB with all WeatherEdge tables created."""
    db_path = tmp_path / "test.db"
    eng = make_engine(f"sqlite:///{db_path}")
    init_db(engine=eng)
    eng.dispose()
    c = sqlite3.connect(str(db_path))
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def _now_iso(offset_minutes: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=offset_minutes)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _insert_trade(conn, **kwargs) -> int:
    defaults = dict(
        market_id="mkt1",
        platform="polymarket",
        mode="paper",
        side="buy_yes",
        price=0.50,
        amount=10.0,
        cost=5.0,
        agent_name="Aggressive",
        status="open",
        pnl=None,
        created_at=_now_iso(),
    )
    defaults.update(kwargs)
    cols = ",".join(defaults.keys())
    placeholders = ",".join(["?"] * len(defaults))
    cur = conn.execute(
        f"INSERT INTO trades ({cols}) VALUES ({placeholders})",
        tuple(defaults.values()),
    )
    conn.commit()
    return cur.lastrowid


def _insert_market_price(conn, market_id: str, yes_price: float, when: str | None = None):
    when = when or _now_iso()
    conn.execute(
        """
        INSERT INTO market_prices (market_id, yes_price, no_price, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (market_id, yes_price, 1.0 - yes_price, when),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# query_open_positions joins with latest market_prices
# ---------------------------------------------------------------------------


def test_open_positions_includes_current_yes(conn):
    _insert_trade(conn, market_id="m1", side="buy_yes", price=0.40, amount=5.0, cost=2.0)
    _insert_market_price(conn, "m1", yes_price=0.65)

    rows = query_open_positions(conn)
    assert len(rows) == 1
    assert rows[0]["market_id"] == "m1"
    assert pytest.approx(rows[0]["current_yes"]) == 0.65


def test_open_positions_returns_null_current_yes_when_no_market_price(conn):
    """When no market_prices row exists for the market, current_yes is NULL.

    Previously COALESCE substituted the entry price, which silently masked
    scraper outages as $0 unrealized P&L. Now NULL flows through so the
    renderer can mark the row STALE.
    """
    _insert_trade(conn, market_id="lonely", price=0.55)

    rows = query_open_positions(conn)
    assert len(rows) == 1
    assert rows[0]["current_yes"] is None


def test_open_positions_uses_most_recent_price(conn):
    _insert_trade(conn, market_id="m1", price=0.50)
    _insert_market_price(conn, "m1", yes_price=0.40, when=_now_iso(offset_minutes=-30))
    _insert_market_price(conn, "m1", yes_price=0.70)  # newer

    rows = query_open_positions(conn)
    assert pytest.approx(rows[0]["current_yes"]) == 0.70


def test_open_positions_excludes_resolved(conn):
    _insert_trade(conn, status="open", market_id="kept")
    _insert_trade(conn, status="won", market_id="gone")
    rows = query_open_positions(conn)
    assert {r["market_id"] for r in rows} == {"kept"}


# ---------------------------------------------------------------------------
# _row_unrealized_pnl
# ---------------------------------------------------------------------------


def _row(side: str, price: float, amount: float, cost: float, current_yes: float | None):
    return {"side": side, "price": price, "amount": amount, "cost": cost, "current_yes": current_yes}


def test_unrealized_pnl_buy_yes_profit():
    # bought 10 @ 0.40 → cost 4.00, market now 0.65 → value 6.50, unrealized +2.50
    row = _row("buy_yes", 0.40, 10.0, 4.0, 0.65)
    assert _row_unrealized_pnl(row) == pytest.approx(2.5)


def test_unrealized_pnl_buy_yes_loss():
    row = _row("buy_yes", 0.50, 10.0, 5.0, 0.30)
    assert _row_unrealized_pnl(row) == pytest.approx(-2.0)


def test_unrealized_pnl_buy_no_profit():
    # bought 10 NO @ 0.40 (price of NO = 1 - 0.60 yes), cost 4.00.
    # Now yes=0.30 → NO=0.70 → value 7.00, unrealized +3.00
    row = _row("buy_no", 0.40, 10.0, 4.0, 0.30)
    assert _row_unrealized_pnl(row) == pytest.approx(3.0)


def test_unrealized_pnl_returns_none_without_current_yes():
    row = _row("buy_yes", 0.5, 10, 5, None)
    assert _row_unrealized_pnl(row) is None


def test_unrealized_pnl_unknown_side_returns_none():
    row = _row("buy_what", 0.5, 10, 5, 0.6)
    assert _row_unrealized_pnl(row) is None


# ---------------------------------------------------------------------------
# query_per_agent_pnl
# ---------------------------------------------------------------------------


def test_per_agent_pnl_groups_resolved_trades(conn):
    _insert_trade(conn, agent_name="A", status="won", pnl=2.0)
    _insert_trade(conn, agent_name="A", status="lost", pnl=-1.0)
    _insert_trade(conn, agent_name="B", status="won", pnl=5.0)
    _insert_trade(conn, agent_name="A", status="open", pnl=None)  # excluded

    rows = query_per_agent_pnl(conn)
    by_agent = {r["agent"]: r for r in rows}
    assert by_agent["A"]["resolved"] == 2
    assert by_agent["A"]["wins"] == 1
    assert by_agent["A"]["losses"] == 1
    assert pytest.approx(by_agent["A"]["total_pnl"]) == 1.0
    assert by_agent["B"]["resolved"] == 1
    assert pytest.approx(by_agent["B"]["total_pnl"]) == 5.0


def test_per_agent_pnl_sorted_by_pnl_desc(conn):
    _insert_trade(conn, agent_name="Loser", status="lost", pnl=-3.0)
    _insert_trade(conn, agent_name="Winner", status="won", pnl=10.0)
    _insert_trade(conn, agent_name="Mid", status="won", pnl=2.0)

    rows = query_per_agent_pnl(conn)
    assert [r["agent"] for r in rows] == ["Winner", "Mid", "Loser"]


def test_per_agent_pnl_treats_null_agent_as_unknown(conn):
    _insert_trade(conn, agent_name=None, status="won", pnl=1.0)
    rows = query_per_agent_pnl(conn)
    assert rows[0]["agent"] == "unknown"


def test_per_agent_pnl_includes_early_exits(conn):
    _insert_trade(conn, agent_name="A", status="edge_stop", pnl=0.5)
    _insert_trade(conn, agent_name="A", status="capital_protect", pnl=-0.2)
    rows = query_per_agent_pnl(conn)
    assert rows[0]["resolved"] == 2
    assert pytest.approx(rows[0]["total_pnl"]) == 0.3


def test_per_agent_pnl_empty_when_no_resolved_trades(conn):
    _insert_trade(conn, status="open", pnl=None)
    rows = query_per_agent_pnl(conn)
    assert rows == []


# ---------------------------------------------------------------------------
# Panel rendering — these only need to not raise; rich Console renders fine
# ---------------------------------------------------------------------------


def test_open_positions_panel_renders_with_data(conn):
    _insert_trade(conn, market_id="abc", side="buy_yes", price=0.40, amount=10.0, cost=4.0)
    _insert_market_price(conn, "abc", yes_price=0.65)

    rows = query_open_positions(conn)
    panel = build_open_positions_panel(rows)
    # Render to string via Console (smoke test).
    from rich.console import Console
    out = Console(record=True, width=120).render(panel)
    rendered = "".join(seg.text for seg in out)
    # Panel no longer renders market_id; verify the header and the
    # unrealized P&L (+2.50 = (0.65 - 0.40) * 10) are present.
    assert "OPEN POSITIONS" in rendered
    assert "2.5" in rendered or "+2.5" in rendered


def test_open_positions_panel_renders_when_empty():
    panel = build_open_positions_panel([])
    from rich.console import Console
    out = Console(record=True, width=120).render(panel)
    rendered = "".join(seg.text for seg in out)
    assert "Waiting for data" in rendered


def test_per_agent_pnl_panel_renders_with_data(conn):
    _insert_trade(conn, agent_name="Agg", status="won", pnl=3.0)
    _insert_trade(conn, agent_name="Agg", status="lost", pnl=-1.0)
    rows = query_per_agent_pnl(conn)
    panel = build_per_agent_pnl_panel(rows)
    from rich.console import Console
    out = Console(record=True, width=120).render(panel)
    rendered = "".join(seg.text for seg in out)
    assert "Agg" in rendered


def test_per_agent_pnl_panel_renders_when_empty():
    panel = build_per_agent_pnl_panel([])
    from rich.console import Console
    out = Console(record=True, width=120).render(panel)
    rendered = "".join(seg.text for seg in out)
    assert "Waiting for data" in rendered


# ---------------------------------------------------------------------------
# build_layout accepts the new per_agent_pnl panel
# ---------------------------------------------------------------------------


def test_build_layout_accepts_per_agent_panel():
    """Smoke test — every panel slot must be filled, the new layout must render."""
    panel = build_per_agent_pnl_panel([])
    layout = build_layout(
        header=panel,
        open_positions=panel,
        model_perf=panel,
        recent_trades=panel,
        active_markets=panel,
        pnl_history=panel,
        per_agent_pnl=panel,
    )
    from rich.console import Console
    Console(record=True, width=180).print(layout)  # should not raise


# ---------------------------------------------------------------------------
# query_performance — phase-scoped win rate (Finding 3)
# ---------------------------------------------------------------------------


def test_query_performance_all_time_when_no_activation(conn):
    """Default (activated_at=None) behaves as before: aggregates everything."""
    _insert_trade(conn, status="won", pnl=0.40, created_at=_now_iso(-48 * 60))
    _insert_trade(conn, status="lost", pnl=-0.50, created_at=_now_iso(-36 * 60))
    _insert_trade(conn, status="won", pnl=0.30, created_at=_now_iso(-1 * 60))

    perf = query_performance(conn)
    assert perf["resolved"] == 3
    assert perf["wins_all"] == 2
    assert perf["win_rate"] == pytest.approx(200.0 / 3.0)
    assert perf["phase_since"] is None


def test_query_performance_filters_wins_by_phase(conn):
    """With activated_at, Phase 0 trades are excluded from wins/resolved/win_rate."""
    activation = _now_iso(-24 * 60)
    # Phase 0: 3 losses (lottery-ticket regime)
    for _ in range(3):
        _insert_trade(conn, status="lost", pnl=-0.50, created_at=_now_iso(-48 * 60))
    # Phase 1: 2 wins (calibrated regime)
    _insert_trade(conn, status="won", pnl=0.40, created_at=_now_iso(-12 * 60))
    _insert_trade(conn, status="won", pnl=0.60, created_at=_now_iso(-6 * 60))

    perf = query_performance(conn, activated_at=activation)
    assert perf["resolved"] == 2, "only Phase 1 trades count toward resolved"
    assert perf["wins_all"] == 2
    assert perf["won"] == 2
    assert perf["lost"] == 0
    assert perf["win_rate"] == pytest.approx(100.0)
    assert perf["phase_since"] == activation


def test_query_performance_total_pnl_stays_all_time(conn):
    """total_pnl MUST include Phase 0 so header bankroll matches real cash position."""
    activation = _now_iso(-24 * 60)
    _insert_trade(conn, status="lost", pnl=-10.0, created_at=_now_iso(-48 * 60))
    _insert_trade(conn, status="won", pnl=3.0, created_at=_now_iso(-6 * 60))

    perf = query_performance(conn, activated_at=activation)
    # All-time P&L = -10 + 3 = -7 (Phase 0 loss still depresses bankroll)
    assert perf["total_pnl"] == pytest.approx(-7.0)


def test_query_performance_handles_t_separated_activation(conn):
    """phase_config.json stores 'T'-separated ISO; created_at stores space-separated.

    SQL must normalize both sides so comparison is correct across formats.
    """
    # Activation with 'T' and timezone offset — matches current_phase.json format.
    activation = "2026-04-22T00:00:00+00:00"
    # Trade after activation (space-separated — SQLAlchemy default).
    _insert_trade(conn, status="won", pnl=0.50, created_at="2026-04-22 12:00:00")
    # Trade before activation — must be excluded despite close numeric proximity.
    _insert_trade(conn, status="lost", pnl=-0.50, created_at="2026-04-21 23:30:00")

    perf = query_performance(conn, activated_at=activation)
    assert perf["resolved"] == 1
    assert perf["wins_all"] == 1


def test_query_performance_preserves_sub_second_precision(conn):
    """Activation timestamp has microseconds; trades in the same second must be
    correctly partitioned pre/post. SQLite datetime() truncates fractional
    seconds and would wrongly include pre-activation rows — julianday() must
    preserve microsecond precision.
    """
    activation = "2026-04-21T23:13:56.475164+00:00"
    # 100ms into that second — BEFORE activation (0.475s), should be excluded.
    _insert_trade(conn, status="lost", pnl=-0.10, created_at="2026-04-21 23:13:56.100000")
    # 900ms into that second — AFTER activation, should be included.
    _insert_trade(conn, status="won", pnl=0.20, created_at="2026-04-21 23:13:56.900000")

    perf = query_performance(conn, activated_at=activation)
    assert perf["resolved"] == 1, "sub-second trades must not leak across activation boundary"
    assert perf["wins_all"] == 1
    assert perf["lost"] == 0
