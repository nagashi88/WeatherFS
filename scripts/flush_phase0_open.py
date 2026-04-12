#!/usr/bin/env python3
"""One-shot script: close all open Phase 0 trades at mark-to-market value.

Sets status='manual_close_phase0_flush' so they are excluded from Phase 1 metrics.
Makes a DB backup before any writes.

Usage:
    python scripts/flush_phase0_open.py
    python scripts/flush_phase0_open.py --dry-run
"""

import argparse
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "weatheredge.db"
FLUSH_STATUS = "manual_close_phase0_flush"


def main(dry_run: bool = False):
    if not DB_PATH.exists():
        print(f"ERROR: DB not found at {DB_PATH}")
        return

    # Backup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = DB_PATH.with_suffix(f".pre_phase1_flush_{ts}")
    if not dry_run:
        shutil.copy2(DB_PATH, backup_path)
        print(f"Backup: {backup_path.name}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Fetch open trades
    cur.execute("SELECT * FROM trades WHERE status='open'")
    open_trades = cur.fetchall()
    print(f"\nOpen trades found: {len(open_trades)}")

    if not open_trades:
        print("Nothing to flush.")
        conn.close()
        return

    now = datetime.now(timezone.utc).isoformat()
    flushed, skipped, errors = [], [], []

    for trade in open_trades:
        tid = trade["id"]
        market_id = trade["market_id"]
        side = trade["side"]
        amount = trade["amount"]
        cost = trade["cost"]

        # Get latest price snapshot
        cur.execute("""
            SELECT yes_price, no_price FROM market_prices
            WHERE market_id = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (market_id,))
        row = cur.fetchone()

        if row is None:
            skipped.append((tid, market_id, "no market_prices snapshot"))
            continue

        yes_price = row["yes_price"]
        no_price = row["no_price"]

        # Validate prices
        if side == "buy_yes":
            p = yes_price
            if p is None or not (0 <= p <= 1):
                skipped.append((tid, market_id, f"invalid yes_price={p}"))
                continue
            pnl = round(amount * p - cost, 4)
            res_val = p
        elif side == "buy_no":
            p = no_price if (no_price is not None and 0 <= no_price <= 1) else (1 - yes_price if yes_price is not None else None)
            if p is None or not (0 <= p <= 1):
                skipped.append((tid, market_id, f"invalid no_price={p}"))
                continue
            pnl = round(amount * p - cost, 4)
            res_val = p
        else:
            skipped.append((tid, market_id, f"unknown side={side}"))
            continue

        flushed.append((tid, market_id, side, cost, pnl))

        if not dry_run:
            cur.execute("""
                UPDATE trades
                SET status=?, pnl=?, resolution_value=?, resolved_at=?
                WHERE id=?
            """, (FLUSH_STATUS, pnl, res_val, now, tid))

    if not dry_run:
        conn.commit()

    conn.close()

    # Report
    total_cost = sum(r[3] for r in flushed)
    total_pnl = sum(r[4] for r in flushed)
    wins = sum(1 for r in flushed if r[4] > 0)

    prefix = "[DRY RUN] " if dry_run else ""
    print(f"\n{prefix}FLUSHED: {len(flushed)} trade")
    print(f"  Costo totale:    ${total_cost:.2f}")
    print(f"  P&L mark-to-mkt: ${total_pnl:.2f}")
    print(f"  Positivi/negativi: {wins}/{len(flushed)-wins}")

    if skipped:
        print(f"\n{prefix}SKIPPED (rimangono 'open'): {len(skipped)}")
        for tid, mid, reason in skipped:
            print(f"  trade_id={tid} market={mid}: {reason}")

    if not dry_run and not skipped:
        print(f"\nVerifica: tutti i trade open sono stati chiusi con status='{FLUSH_STATUS}'")
    elif not dry_run and skipped:
        print(f"\nATTENZIONE: {len(skipped)} trade rimangono 'open' senza snapshot prezzi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Solo stampa, nessuna modifica al DB")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
