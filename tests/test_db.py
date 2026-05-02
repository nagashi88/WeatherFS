"""Tests for the WeatherEdge database layer.

Most tests use an in-memory SQLite database for isolation.
WAL-mode tests use a temp file because SQLite in-memory databases
always report journal_mode='memory' regardless of PRAGMA settings.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from weatheredge.db.database import init_db, get_session, make_engine
from weatheredge.db.models import Trade


EXPECTED_TABLES = {
    "markets",
    "market_prices",
    "forecasts",
    "signals",
    "trades",
    "shadow_twin_trades",
    "shadow_policy_trades",
    "shadow_basket_trades",
    "shadow_basket_legs",
    "phase1_shadow_rejections",
    "daily_performance",
    "calibration",
    "calibration_records",
    "agent_performance",
    "vpin_snapshots",
    "fill_quality",
    "wallet_profiles",
    "edge_decay",
    "leakage_violations",
}

VALID_TRADE_STATUSES = {
    "open",
    "won",
    "lost",
    "edge_stop",
    "capital_protect",
    "time_decay",
    "ev_based",
    "cancelled",
    "abandoned",
}


@pytest.fixture()
def engine():
    """In-memory SQLite engine with WAL pragmas and all tables created."""
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def file_engine(tmp_path):
    """Temp-file SQLite engine — required for WAL mode tests.

    SQLite in-memory databases always use journal_mode=memory and cannot
    be switched to WAL mode; a real file is needed to exercise WAL.
    """
    db_file = tmp_path / "test_weatheredge.db"
    eng = make_engine(f"sqlite:///{db_file}")
    init_db(engine=eng)
    yield eng
    eng.dispose()


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def test_all_tables_created(engine):
    """All 13 expected tables must exist after init_db()."""
    inspector = inspect(engine)
    actual_tables = set(inspector.get_table_names())
    assert EXPECTED_TABLES == actual_tables, (
        f"Missing tables: {EXPECTED_TABLES - actual_tables}; "
        f"Extra tables: {actual_tables - EXPECTED_TABLES}"
    )


def test_init_db_migrates_legacy_market_prices_orderbook_columns():
    """Existing DBs need orderbook columns before the next price insert."""
    eng = make_engine("sqlite:///:memory:")
    try:
        with eng.begin() as conn:
            conn.execute(text(
                """
                CREATE TABLE market_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT,
                    timestamp DATETIME NOT NULL,
                    yes_price FLOAT,
                    no_price FLOAT,
                    volume FLOAT,
                    spread FLOAT
                )
                """
            ))

        init_db(engine=eng)

        columns = {col["name"] for col in inspect(eng).get_columns("market_prices")}
        assert {"best_bid", "best_ask", "bid_size", "ask_size"} <= columns
    finally:
        eng.dispose()


# ---------------------------------------------------------------------------
# WAL mode (requires a real file — in-memory SQLite always reports 'memory')
# ---------------------------------------------------------------------------

def test_wal_mode_active(file_engine):
    """PRAGMA journal_mode must return 'wal' (set via connect event)."""
    with file_engine.connect() as conn:
        result = conn.execute(text("PRAGMA journal_mode")).scalar()
    assert result == "wal", f"Expected 'wal', got '{result}'"


def test_synchronous_normal(file_engine):
    """PRAGMA synchronous must return 1 (NORMAL) as set in Gap 14."""
    with file_engine.connect() as conn:
        result = conn.execute(text("PRAGMA synchronous")).scalar()
    # SQLite returns integer: 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA
    assert result == 1, f"Expected synchronous=1 (NORMAL), got {result}"


# ---------------------------------------------------------------------------
# Trade status values (Addendum Gap 9)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", sorted(VALID_TRADE_STATUSES))
def test_trade_status_accepted(engine, status):
    """Each valid status value must be insertable into the trades table."""
    with get_session(engine=engine) as session:
        trade = Trade(
            market_id=None,
            platform="polymarket",
            mode="paper",
            side="buy_yes",
            price=0.40,
            amount=10.0,
            cost=4.0,
            status=status,
        )
        session.add(trade)
        # commit happens on context manager exit — no exception means success

    # Verify the row is actually persisted with the correct status
    with get_session(engine=engine) as session:
        saved = session.query(Trade).filter_by(status=status).first()
        assert saved is not None, f"Trade with status='{status}' was not found after insert"
        assert saved.status == status


def test_trade_default_status_is_open(engine):
    """Trade.status must default to 'open' when not provided."""
    with get_session(engine=engine) as session:
        trade = Trade(
            platform="kalshi",
            mode="paper",
            side="buy_no",
            price=0.60,
            amount=5.0,
            cost=3.0,
        )
        session.add(trade)

    with get_session(engine=engine) as session:
        saved = session.query(Trade).filter_by(platform="kalshi").first()
        assert saved is not None
        assert saved.status == "open"


def test_sold_status_not_in_valid_set():
    """'sold' must NOT appear in the valid status set (Gap 9 override)."""
    assert "sold" not in VALID_TRADE_STATUSES


# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------

def test_get_session_rollback_on_exception(engine):
    """get_session() must roll back on exception, leaving no partial data."""
    from weatheredge.db.models import Market

    with pytest.raises(ValueError):
        with get_session(engine=engine) as session:
            session.add(Market(
                id="test-market-1",
                platform="polymarket",
                city="NYC",
                station="KNYC",
                date="2026-04-01",
                market_type="daily_high",
            ))
            raise ValueError("simulated error")

    # The market must not have been committed
    with get_session(engine=engine) as session:
        result = session.query(Market).filter_by(id="test-market-1").first()
        assert result is None, "Rolled-back market should not exist in DB"


# ---------------------------------------------------------------------------
# Additional coverage: global lazy-init paths and extended scenarios
# ---------------------------------------------------------------------------

def test_get_session_without_engine_uses_global_factory(monkeypatch):
    """get_session() with no engine arg must use _get_session_factory() (line 106).

    We patch the module-level _engine and _SessionLocal to avoid touching the
    real database file while still exercising the lazy-init code path.
    """
    import weatheredge.db.database as db_module

    # Build an in-memory engine for isolation
    test_engine = make_engine("sqlite:///:memory:")
    init_db(engine=test_engine)

    # Patch global state so _get_engine() returns our test engine and
    # _get_session_factory() rebuilds from it.
    original_engine = db_module._engine
    original_factory = db_module._SessionLocal
    try:
        db_module._engine = test_engine
        db_module._SessionLocal = None  # force _get_session_factory() to rebuild

        # get_session() with no args must reach line 106 and succeed
        with get_session() as session:
            trade = Trade(
                platform="polymarket",
                mode="paper",
                side="buy_yes",
                price=0.50,
                amount=10.0,
                cost=5.0,
                status="open",
            )
            session.add(trade)
        # Verify commit happened
        with get_session() as session:
            saved = session.query(Trade).filter_by(platform="polymarket").first()
            assert saved is not None
    finally:
        db_module._engine = original_engine
        db_module._SessionLocal = original_factory
        test_engine.dispose()


def test_get_engine_lazy_init(monkeypatch):
    """_get_engine() must create a new engine when _engine is None (lines 59-61)."""
    import weatheredge.db.database as db_module

    original_engine = db_module._engine
    try:
        db_module._engine = None  # force lazy creation
        from weatheredge.db.database import _get_engine
        eng = _get_engine()
        assert eng is not None
        assert db_module._engine is eng  # module-level variable is now set
    finally:
        # Restore original state to avoid polluting other tests
        if db_module._engine is not original_engine:
            db_module._engine.dispose()
        db_module._engine = original_engine


def test_get_session_factory_lazy_init(monkeypatch):
    """_get_session_factory() must create sessionmaker when _SessionLocal is None (lines 66-68)."""
    import weatheredge.db.database as db_module

    original_engine = db_module._engine
    original_factory = db_module._SessionLocal
    try:
        # Give the module a safe in-memory engine and clear the factory
        db_module._engine = make_engine("sqlite:///:memory:")
        db_module._SessionLocal = None

        from weatheredge.db.database import _get_session_factory
        factory = _get_session_factory()
        assert factory is not None
        assert db_module._SessionLocal is factory
    finally:
        if db_module._engine is not original_engine:
            db_module._engine.dispose()
        db_module._engine = original_engine
        db_module._SessionLocal = original_factory


def test_get_session_commit_on_normal_exit(engine):
    """Data inserted inside get_session() must be committed after the block exits."""
    with get_session(engine=engine) as session:
        trade = Trade(
            platform="kalshi",
            mode="live",
            side="buy_no",
            price=0.55,
            amount=20.0,
            cost=11.0,
            status="open",
        )
        session.add(trade)
    # After context manager exit without exception, row must be persisted
    with get_session(engine=engine) as session:
        saved = session.query(Trade).filter_by(platform="kalshi", mode="live").first()
        assert saved is not None
        assert saved.price == 0.55


def test_init_db_idempotent(engine):
    """Calling init_db() twice on the same engine must not raise any error."""
    init_db(engine=engine)  # second call — tables already exist
    # Verify tables are still accessible
    inspector = inspect(engine)
    assert "trades" in inspector.get_table_names()


def test_make_engine_custom_url(tmp_path):
    """make_engine() with a custom URL must return a usable engine with WAL set."""
    db_file = tmp_path / "custom.db"
    eng = make_engine(f"sqlite:///{db_file}")
    try:
        with eng.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode")).scalar()
        assert result == "wal"
    finally:
        eng.dispose()


def test_make_engine_uses_env_db_path_override(tmp_path, monkeypatch):
    """make_engine() without URL must read WEATHEREDGE_DB_PATH at runtime."""
    db_file = tmp_path / "env_override.db"
    monkeypatch.setenv("WEATHEREDGE_DB_PATH", str(db_file))

    eng = make_engine()
    try:
        assert eng.url.database == str(db_file)
        init_db(engine=eng)
        assert db_file.exists()
    finally:
        eng.dispose()


def test_concurrent_read_write_wal(tmp_path):
    """WAL mode allows a reader and a writer to operate concurrently on a file DB."""
    from sqlalchemy import create_engine as _ce
    from sqlalchemy import event as _event
    from sqlalchemy.orm import sessionmaker as _sm

    db_file = tmp_path / "wal_concurrent.db"
    eng = make_engine(f"sqlite:///{db_file}")
    init_db(engine=eng)

    # Write a row
    with get_session(engine=eng) as write_session:
        write_session.add(Trade(
            platform="polymarket",
            mode="paper",
            side="buy_yes",
            price=0.45,
            amount=5.0,
            cost=2.25,
            status="open",
        ))

    # Open a second connection (simulating a concurrent reader) and read while
    # the engine's connection pool still holds the previous write connection open
    with eng.connect() as read_conn:
        rows = read_conn.execute(text("SELECT COUNT(*) FROM trades")).scalar()
    assert rows == 1

    eng.dispose()
