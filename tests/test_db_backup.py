"""Tests for daily SQLite hot-backup."""
from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from weatheredge.db.backup import _prune_old_backups, backup_database


@pytest.fixture
def source_db(tmp_path: Path) -> Path:
    db = tmp_path / "weatheredge.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
    conn.execute("INSERT INTO t (val) VALUES ('hello')")
    conn.commit()
    conn.close()
    return db


def test_backup_writes_file_and_returns_path(source_db, tmp_path):
    target = tmp_path / "backups"

    out = backup_database(source_db, target)

    assert out is not None
    assert out.exists()
    assert out.parent == target
    assert out.name.startswith("weatheredge_") and out.name.endswith(".db")


def test_backup_contents_match_source(source_db, tmp_path):
    target = tmp_path / "backups"
    out = backup_database(source_db, target)

    conn = sqlite3.connect(str(out))
    try:
        rows = conn.execute("SELECT val FROM t").fetchall()
    finally:
        conn.close()
    assert rows == [("hello",)]


def test_backup_returns_none_when_source_missing(tmp_path):
    target = tmp_path / "backups"
    missing = tmp_path / "does-not-exist.db"

    out = backup_database(missing, target)

    assert out is None
    assert not target.exists() or not any(target.iterdir())


def test_backup_creates_target_dir_if_absent(source_db, tmp_path):
    target = tmp_path / "deep" / "nested" / "backups"
    assert not target.exists()

    backup_database(source_db, target)

    assert target.is_dir()


def test_backup_prunes_old_files_beyond_retention(source_db, tmp_path):
    target = tmp_path / "backups"
    target.mkdir()

    # Create one fake "old" backup with mtime 60 days ago.
    old = target / "weatheredge_19990101_000000.db"
    old.write_bytes(b"old")
    old_mtime = (datetime.now(timezone.utc) - timedelta(days=60)).timestamp()
    os.utime(old, (old_mtime, old_mtime))

    backup_database(source_db, target, retention_days=30)

    assert not old.exists()
    # The fresh one is still there.
    fresh = list(target.glob("weatheredge_*.db"))
    assert len(fresh) == 1


def test_prune_keeps_recent_backups(tmp_path):
    target = tmp_path / "backups"
    target.mkdir()

    # Make two backups: one 5 days old, one 1 day old.
    for days_ago, name in [(5, "weatheredge_a.db"), (1, "weatheredge_b.db")]:
        f = target / name
        f.write_bytes(b"x")
        mtime = (datetime.now(timezone.utc) - timedelta(days=days_ago)).timestamp()
        os.utime(f, (mtime, mtime))

    pruned = _prune_old_backups(target, retention_days=30)

    assert pruned == 0
    assert {p.name for p in target.iterdir()} == {"weatheredge_a.db", "weatheredge_b.db"}


def test_prune_with_zero_retention_is_noop(tmp_path):
    target = tmp_path / "backups"
    target.mkdir()
    f = target / "weatheredge_x.db"
    f.write_bytes(b"x")

    assert _prune_old_backups(target, retention_days=0) == 0
    assert f.exists()


def test_consecutive_backups_get_unique_filenames(source_db, tmp_path):
    target = tmp_path / "backups"
    out1 = backup_database(source_db, target)
    time.sleep(1.1)  # filename stamp has 1s resolution
    out2 = backup_database(source_db, target)

    assert out1 != out2
    assert out1.exists() and out2.exists()
