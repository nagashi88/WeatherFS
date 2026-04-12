from __future__ import annotations

import csv
import sqlite3

from weatheredge.backtesting.simulation_db import build_simulation_db


def _write_prices_csv(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["station", "date", "threshold", "direction", "yes_price"])
        w.writerow(["KDEN", "2026-04-01", "70", "above", "0.45"])
        w.writerow(["KDEN", "2026-04-01", "70", "below", "0.55"])
        w.writerow(["KLAX", "2026-04-01", "80", "above", "0.30"])


def _write_trades_csv(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "station",
                "date",
                "threshold",
                "direction",
                "model_prob",
                "market_proxy",
                "side",
                "entry_price",
                "won",
                "pnl",
            ]
        )
        w.writerow(["KDEN", "2026-04-01", "70", "above", "0.62", "0.45", "buy_yes", "0.45", "True", "5.30"])
        w.writerow(["KDEN", "2026-04-01", "70", "above", "0.40", "0.55", "buy_no", "0.55", "False", "-5.50"])
        w.writerow(["KLAX", "2026-04-02", "80", "above", "0.70", "0.30", "buy_yes", "0.30", "True", "6.80"])
        w.writerow(["KLAX", "2026-04-02", "80", "below", "0.20", "0.70", "buy_no", "0.70", "True", "2.70"])


def test_build_simulation_db_creates_core_tables(tmp_path):
    prices_csv = tmp_path / "prices.csv"
    trades_csv = tmp_path / "trades.csv"
    out_db = tmp_path / "simulation.db"
    _write_prices_csv(prices_csv)
    _write_trades_csv(trades_csv)

    report = build_simulation_db(
        output_db=out_db,
        trades_csv=trades_csv,
        market_prices_csv=prices_csv,
        bootstrap_horizons=[2, 4],
        bootstrap_sims=200,
        seed=7,
    )

    assert report.market_rows == 3
    assert report.trade_rows == 4
    assert report.daily_rows == 2
    assert report.station_rows == 2
    assert report.bootstrap_rows == 2

    with sqlite3.connect(out_db) as conn:
        sim_rows = conn.execute("SELECT COUNT(*) FROM sim_trades").fetchone()[0]
        daily_total = conn.execute(
            "SELECT ROUND(SUM(net_pnl), 2) FROM daily_pnl"
        ).fetchone()[0]
        horizons = {
            row[0] for row in conn.execute("SELECT horizon_trades FROM bootstrap_pnl").fetchall()
        }
        assert sim_rows == 4
        assert float(daily_total) == 9.3
        assert horizons == {2, 4}

