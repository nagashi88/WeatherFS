"""Tests for US-TG — temporal guard integration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import weatheredge.config as _cfg
from weatheredge.temporal_guard import assert_no_lookahead, TemporalLeakageError
from weatheredge.execution.paper_trader import Order, PaperTrader


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TestAssertNoLookahead:
    """Direct unit tests for assert_no_lookahead."""

    def test_valid_timestamps_pass(self):
        t0 = _now() - timedelta(minutes=10)
        t1 = _now() - timedelta(minutes=5)
        t2 = _now()
        # Should not raise
        assert_no_lookahead(t0, t1, t2, "gfs", "mkt-1")

    def test_forecast_after_market_price_raises(self):
        market_ts = _now() - timedelta(minutes=10)
        forecast_ts = _now() - timedelta(minutes=5)  # forecast NEWER than market price
        decision_ts = _now()
        with pytest.raises(TemporalLeakageError, match="LOOKAHEAD DETECTED"):
            assert_no_lookahead(forecast_ts, market_ts, decision_ts, "gfs", "mkt-1")

    def test_market_price_after_decision_raises(self):
        forecast_ts = _now() - timedelta(minutes=10)
        decision_ts = _now() - timedelta(minutes=5)
        market_ts = _now()  # market price NEWER than decision
        with pytest.raises(TemporalLeakageError, match="LOOKAHEAD DETECTED"):
            assert_no_lookahead(forecast_ts, market_ts, decision_ts, "gfs", "mkt-1")


class TestPlaceOrderTemporalGuard:
    """Integration tests: temporal guard in PaperTrader.place_order()."""

    def test_valid_timestamps_trade_fills(self):
        pt = PaperTrader(initial_bankroll=1000.0)
        order = Order(
            market_id="mkt-1",
            side="buy_yes",
            price=0.50,
            amount=10.0,
            forecast_timestamp=_now() - timedelta(minutes=10),
            market_price_timestamp=_now() - timedelta(minutes=5),
        )
        result = pt.place_order(order)
        assert result["filled"] is True

    def test_lookahead_returns_filled_false(self):
        pt = PaperTrader(initial_bankroll=1000.0)
        market_ts = _now() - timedelta(minutes=10)
        forecast_ts = _now() - timedelta(minutes=5)  # forecast NEWER than market
        order = Order(
            market_id="mkt-1",
            side="buy_yes",
            price=0.50,
            amount=10.0,
            forecast_timestamp=forecast_ts,
            market_price_timestamp=market_ts,
        )
        result = pt.place_order(order)
        assert result["filled"] is False
        assert result["reason"] == "temporal_leakage"
        assert "error" in result

    def test_modules_flag_off_bypasses_guard(self, monkeypatch):
        monkeypatch.setitem(_cfg.MODULES, "temporal_guard", False)
        pt = PaperTrader(initial_bankroll=1000.0)
        market_ts = _now() - timedelta(minutes=10)
        forecast_ts = _now() - timedelta(minutes=5)  # would normally fail
        order = Order(
            market_id="mkt-1",
            side="buy_yes",
            price=0.50,
            amount=10.0,
            forecast_timestamp=forecast_ts,
            market_price_timestamp=market_ts,
        )
        result = pt.place_order(order)
        assert result["filled"] is True

    def test_no_timestamps_bypasses_guard(self):
        """When timestamps are None, guard is skipped even if flag is on."""
        pt = PaperTrader(initial_bankroll=1000.0)
        order = Order(
            market_id="mkt-1",
            side="buy_yes",
            price=0.50,
            amount=10.0,
        )
        result = pt.place_order(order)
        assert result["filled"] is True
