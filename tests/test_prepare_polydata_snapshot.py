"""Tests for poly_data archive preparation pipeline."""

from __future__ import annotations

import io
import tarfile
from datetime import datetime, timezone

from weatheredge.backtesting.prepare_polydata_snapshot import prepare_poly_data_archive


def _build_small_archive(path):
    markets_csv = (
        "id,question,answer1,answer2\n"
        "m1,Will the temperature in NYC on March 30 be 76°F or above?,Yes,No\n"
    ).encode()
    ts = int(datetime(2026, 3, 30, 10, tzinfo=timezone.utc).timestamp())
    trades_csv = (
        "timestamp,market_id,nonusdc_side,price\n"
        f"{ts},m1,token1,0.58\n"
    ).encode()

    with tarfile.open(path, mode="w:xz") as tf:
        info_m = tarfile.TarInfo(name="poly_data/markets.csv")
        info_m.size = len(markets_csv)
        tf.addfile(info_m, io.BytesIO(markets_csv))

        info_t = tarfile.TarInfo(name="poly_data/processed/trades.csv")
        info_t.size = len(trades_csv)
        tf.addfile(info_t, io.BytesIO(trades_csv))


def test_prepare_poly_data_archive_end_to_end(tmp_path):
    archive = tmp_path / "archive.tar.xz"
    work = tmp_path / "work"
    out = tmp_path / "prices.csv"
    _build_small_archive(archive)

    report = prepare_poly_data_archive(
        archive_path=archive,
        work_dir=work,
        output_prices_csv=out,
    )

    assert report.build_report.output_rows == 1
    assert "(stream)" in report.extracted_trades_csv
    assert (work / "markets.csv").exists()
    assert not (work / "trades.csv").exists()
    assert out.exists()


def test_prepare_poly_data_archive_with_trades_extraction(tmp_path):
    archive = tmp_path / "archive.tar.xz"
    work = tmp_path / "work"
    out = tmp_path / "prices.csv"
    _build_small_archive(archive)

    report = prepare_poly_data_archive(
        archive_path=archive,
        work_dir=work,
        output_prices_csv=out,
        extract_trades_csv=True,
    )

    assert report.build_report.output_rows == 1
    assert report.extracted_trades_csv.endswith("trades.csv")
    assert (work / "trades.csv").exists()
    assert out.exists()
