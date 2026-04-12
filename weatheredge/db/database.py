"""Database engine setup for WeatherEdge.

- SQLite with WAL mode (Addendum Gap 14)
- PRAGMA journal_mode=WAL and PRAGMA synchronous=NORMAL set on every connection
- init_db() creates all tables
- get_session() provides a context manager for database sessions

Run as __main__ to create the database:
    python -m weatheredge.db.database
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import event, text
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from weatheredge.db.models import Base

log = logging.getLogger(__name__)

# Database file lives in the project root (one level above the package directory)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB_PATH = _PROJECT_ROOT / "weatheredge.db"
_DB_PATH_ENV_VAR = "WEATHEREDGE_DB_PATH"


def _resolve_db_path() -> Path:
    """Resolve SQLite DB path, allowing runtime override via env var."""
    override = os.getenv(_DB_PATH_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return _DEFAULT_DB_PATH


def _resolve_db_url() -> str:
    """Build sqlite URL from the resolved DB path."""
    return f"sqlite:///{_resolve_db_path()}"


# Kept for backwards compatibility for code/tests that import these constants.
DB_PATH = _resolve_db_path()
DB_URL = _resolve_db_url()


def _set_wal_pragmas(dbapi_connection, connection_record) -> None:  # noqa: ANN001
    """Enable WAL mode and NORMAL synchronous on every new connection.

    Called via SQLAlchemy's connect event. Matches Addendum Gap 14 specification.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


def make_engine(url: str | None = None) -> Engine:
    """Create and return a SQLAlchemy engine with WAL pragmas registered."""
    target_url = url or _resolve_db_url()
    # BugFix M7: Allow SQLite access from multiple scheduler threads
    engine = create_engine(target_url, echo=False, connect_args={"check_same_thread": False})
    event.listen(engine, "connect", _set_wal_pragmas)
    return engine


# Module-level engine and session factory (lazy — not created at import time)
_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


def _get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=_get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal


def init_db(engine: Engine | None = None) -> None:
    """Create all tables defined in models.py.

    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS semantics
    via SQLAlchemy's Base.metadata.create_all().

    Args:
        engine: Optional engine to use. Defaults to the module-level engine.
    """
    target = engine or _get_engine()
    Base.metadata.create_all(bind=target)
    # Migrate existing trades table: add station/market_date if absent (SQLite safe)
    _migrate_trades_columns(target)
    # Migrate calibration_records for idempotent per-trade writes.
    _migrate_calibration_records_columns(target)
    log.info("Database tables created (or already exist) at %s", target.url)


def _migrate_trades_columns(engine: Engine) -> None:
    """Add station and market_date columns to trades table if they don't exist.

    SQLite supports ADD COLUMN for nullable columns without rewriting the table.
    Silently skips if columns already exist.
    """
    from sqlalchemy import text, inspect as sa_inspect
    inspector = sa_inspect(engine)
    if "trades" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("trades")}
    with engine.begin() as conn:
        if "station" not in existing:
            conn.execute(text("ALTER TABLE trades ADD COLUMN station TEXT"))
            log.info("Migrated trades table: added 'station' column")
        if "market_date" not in existing:
            conn.execute(text("ALTER TABLE trades ADD COLUMN market_date TEXT"))
            log.info("Migrated trades table: added 'market_date' column")


def _migrate_calibration_records_columns(engine: Engine) -> None:
    """Add trade_id to calibration_records and unique index for dedupe."""
    from sqlalchemy import inspect as sa_inspect

    inspector = sa_inspect(engine)
    if "calibration_records" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("calibration_records")}
    with engine.begin() as conn:
        if "trade_id" not in existing:
            conn.execute(text("ALTER TABLE calibration_records ADD COLUMN trade_id INTEGER"))
            log.info("Migrated calibration_records table: added 'trade_id' column")
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_calibration_records_trade_id "
                "ON calibration_records(trade_id) WHERE trade_id IS NOT NULL"
            )
        )


@contextmanager
def get_session(engine: Engine | None = None) -> Generator[Session, None, None]:
    """Context manager that yields a database Session and handles commit/rollback.

    Usage::

        with get_session() as session:
            session.add(trade)

    Args:
        engine: Optional engine to use. Defaults to the module-level engine.

    Yields:
        A SQLAlchemy Session.

    Raises:
        Any exception raised inside the block after rolling back.
    """
    if engine is not None:
        factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    else:
        factory = _get_session_factory()

    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log.info("Initialising WeatherEdge database at %s", _resolve_db_path())
    init_db()
    log.info("Done.")
