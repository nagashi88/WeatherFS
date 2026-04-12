"""Prepare poly_data snapshot for backtesting prices.

Pipeline:
1) Read archive.tar.xz
2) Extract markets.csv (small) and stream trades.csv from archive by default
3) Build normalized price CSV via polymarket_price_builder
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence
import tarfile
import shutil

from weatheredge.backtesting.polymarket_price_builder import (
    BuildPolymarketReport,
    build_market_prices_csv_from_poly_data,
)


@dataclass
class PreparePolyDataReport:
    extracted_markets_csv: str
    extracted_trades_csv: str
    build_report: BuildPolymarketReport

def _extract_first_matching_member(
    archive_path: Path,
    suffixes: Sequence[str],
    output_path: Path,
) -> str | None:
    lowered = [s.lower() for s in suffixes]
    with tarfile.open(archive_path, mode="r:xz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            member_name = member.name.lower()
            if not any(member_name.endswith(s) for s in lowered):
                continue
            src = tf.extractfile(member)
            if src is None:
                raise RuntimeError(f"Failed to open member stream: {member.name}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            return member.name
    return None


def prepare_poly_data_archive(
    archive_path: str | Path,
    work_dir: str | Path,
    output_prices_csv: str | Path,
    start_date: date | None = None,
    end_date: date | None = None,
    extract_trades_csv: bool = False,
) -> PreparePolyDataReport:
    archive = Path(archive_path)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    markets_out = work / "markets.csv"
    markets_member_name = _extract_first_matching_member(archive, ("markets.csv",), markets_out)
    if markets_member_name is None:
        raise FileNotFoundError("markets.csv not found in archive")

    extracted_trades_csv: str
    if extract_trades_csv:
        trades_out = work / "trades.csv"
        trades_member_name = _extract_first_matching_member(
            archive, ("processed/trades.csv", "trades.csv"), trades_out
        )
        if trades_member_name is None:
            raise FileNotFoundError("processed/trades.csv (or trades.csv) not found in archive")
        extracted_trades_csv = str(trades_out)
        build_report = build_market_prices_csv_from_poly_data(
            markets_csv=markets_out,
            trades_csv=trades_out,
            output_csv=output_prices_csv,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        build_report = None
        trades_member_name = None
        with tarfile.open(archive, mode="r:xz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                member_name = member.name.lower()
                if not (
                    member_name.endswith("processed/trades.csv")
                    or member_name.endswith("trades.csv")
                ):
                    continue
                with ExitStack() as stack:
                    trades_bin = stack.enter_context(tf.extractfile(member))
                    if trades_bin is None:
                        raise RuntimeError("Failed to open trades member stream")
                    build_report = build_market_prices_csv_from_poly_data(
                        markets_csv=markets_out,
                        trades_csv=trades_bin,
                        output_csv=output_prices_csv,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    trades_member_name = member.name
                break

        if build_report is None or trades_member_name is None:
            raise FileNotFoundError("processed/trades.csv (or trades.csv) not found in archive")
        extracted_trades_csv = f"{archive}:{trades_member_name} (stream)"

    return PreparePolyDataReport(
        extracted_markets_csv=str(markets_out),
        extracted_trades_csv=extracted_trades_csv,
        build_report=build_report,
    )


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    return date.fromisoformat(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare poly_data archive and build backtest prices CSV.")
    parser.add_argument("--archive", required=True, help="Path to archive.tar.xz")
    parser.add_argument("--work-dir", required=True, help="Working extraction directory")
    parser.add_argument("--output", required=True, help="Output prices CSV path")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--extract-trades",
        action="store_true",
        help="Extract trades.csv to work-dir (off by default; stream from archive otherwise).",
    )
    args = parser.parse_args(argv)

    report = prepare_poly_data_archive(
        archive_path=args.archive,
        work_dir=args.work_dir,
        output_prices_csv=args.output,
        start_date=_parse_date(args.start_date),
        end_date=_parse_date(args.end_date),
        extract_trades_csv=args.extract_trades,
    )
    print(
        "prepared markets={m} trades={t} rows={rows} output={out}".format(
            m=report.extracted_markets_csv,
            t=report.extracted_trades_csv,
            rows=report.build_report.output_rows,
            out=report.build_report.output_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
