from __future__ import annotations

from datetime import date, timedelta

from weatheredge.backtesting.backtester import DailyRecord, TradingBacktester
from weatheredge.backtesting.market_prices import HistoricalMarketPriceBook


def _build_records(days: int = 70) -> list[DailyRecord]:
    start = date(2025, 1, 1)
    return [
        DailyRecord(
            station="KNYC",
            date=start + timedelta(days=i),
            forecast_high_f=80.0,
            actual_high_f=75.0 if i % 2 == 0 else 65.0,
        )
        for i in range(days)
    ]


def test_require_real_market_price_skips_missing_quotes():
    records = _build_records()
    empty_book = HistoricalMarketPriceBook.from_rows([])

    bt = TradingBacktester(min_edge=0.10, require_real_market_price=True)
    trades = bt.simulate_trading(
        records=records,
        thresholds=[70.0],
        market_price_book=empty_book,
    )

    assert trades == []


def test_without_require_real_market_price_uses_fallback_proxy():
    records = _build_records()
    empty_book = HistoricalMarketPriceBook.from_rows([])

    bt = TradingBacktester(min_edge=0.10, require_real_market_price=False)
    trades = bt.simulate_trading(
        records=records,
        thresholds=[70.0],
        market_price_book=empty_book,
    )

    assert len(trades) > 0
