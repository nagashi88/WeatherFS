"""Daily SQLite backup with retention.

Uses sqlite3 native backup API: atomic, hot-backup safe even with the live
WAL connection open. Filename includes a UTC timestamp so backups sort
chronologically and survive multiple runs in the same day.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def backup_database(
    source_db: Path | str,
    target_dir: Path | str,
    retention_days: int = 30,
) -> Path | None:
    """Hot-backup the SQLite DB into target_dir, then prune older backups.

    Returns the new backup file path, or None if source_db doesn't exist.
    """
    source_db = Path(source_db)
    target_dir = Path(target_dir)

    if not source_db.exists():
        log.warning("backup_database: source %s does not exist — skipped", source_db)
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = target_dir / f"weatheredge_{stamp}.db"

    src = sqlite3.connect(str(source_db))
    try:
        dst = sqlite3.connect(str(backup_path))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()

    log.info(
        "backup_database: wrote %s (%d bytes)",
        backup_path, backup_path.stat().st_size,
    )

    _prune_old_backups(target_dir, retention_days)
    return backup_path


def _prune_old_backups(target_dir: Path, retention_days: int) -> int:
    """Delete weatheredge_*.db backups older than retention_days. Returns count."""
    if retention_days <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    pruned = 0
    for backup in target_dir.glob("weatheredge_*.db"):
        try:
            mtime = datetime.fromtimestamp(backup.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            try:
                backup.unlink()
                pruned += 1
                log.info("backup_database: pruned old backup %s", backup)
            except OSError as exc:
                log.warning("backup_database: failed to prune %s: %s", backup, exc)
    return pruned
