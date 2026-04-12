"""Tests for weatheredge.agents.base dataclasses."""

from __future__ import annotations

import pytest

from weatheredge.agents.base import MarketOpportunity, Signal, TradeOrder


class TestSignal:
    def test_creation_all_fields(self):
        sig = Signal(
            market_id="mkt-001",
            side="buy_yes",
            confidence=0.75,
            agent_name="conservative",
            reasoning="strong edge",
        )
        assert sig.market_id == "mkt-001"
        assert sig.side == "buy_yes"
        assert sig.confidence == 0.75
        assert sig.agent_name == "conservative"
        assert sig.reasoning == "strong edge"

    def test_defaults(self):
        sig = Signal(market_id="mkt-002", side="no_trade", confidence=0.0)
        assert sig.agent_name == ""
        assert sig.reasoning == ""


class TestTradeOrder:
    def test_creation(self):
        order = TradeOrder(
            market_id="mkt-003",
            side="buy_no",
            price=0.42,
            amount=50.0,
            agent_name="aggressive",
            model_probability=0.65,
            edge_at_entry=0.12,
            city="NYC",
            platform="polymarket",
        )
        assert order.market_id == "mkt-003"
        assert order.side == "buy_no"
        assert order.price == 0.42
        assert order.amount == 50.0
        assert order.agent_name == "aggressive"
        assert order.model_probability == 0.65
        assert order.edge_at_entry == 0.12
        assert order.city == "NYC"
        assert order.platform == "polymarket"

    def test_defaults(self):
        order = TradeOrder(market_id="mkt-004", side="buy_yes", price=0.55, amount=25.0)
        assert order.agent_name == ""
        assert order.model_probability == 0.0
        assert order.edge_at_entry == 0.0
        assert order.city == ""
        assert order.platform == ""


class TestMarketOpportunity:
    def test_creation_with_optional_fields(self):
        opp = MarketOpportunity(
            market_id="mkt-005",
            platform="kalshi",
            city="Chicago",
            station="KORD",
            threshold=32.0,
            direction="below",
            market_price=0.30,
            model_probability=0.45,
            net_edge=0.12,
            ensemble_temps=[30.1, 31.5, 29.8],
            forecast_temp=31.0,
            rmse=2.0,
            lead_time_days=2,
            bucket_low=28.0,
            bucket_high=34.0,
            gaussian_prob=0.44,
            ensemble_prob=0.46,
        )
        assert opp.market_id == "mkt-005"
        assert opp.platform == "kalshi"
        assert opp.city == "Chicago"
        assert opp.station == "KORD"
        assert opp.threshold == 32.0
        assert opp.direction == "below"
        assert opp.market_price == 0.30
        assert opp.model_probability == 0.45
        assert opp.net_edge == 0.12
        assert opp.ensemble_temps == [30.1, 31.5, 29.8]
        assert opp.forecast_temp == 31.0
        assert opp.rmse == 2.0
        assert opp.lead_time_days == 2
        assert opp.bucket_low == 28.0
        assert opp.bucket_high == 34.0
        assert opp.gaussian_prob == 0.44
        assert opp.ensemble_prob == 0.46

    def test_defaults(self):
        opp = MarketOpportunity(
            market_id="mkt-006",
            platform="polymarket",
            city="Miami",
            station="KMIA",
            threshold=90.0,
            direction="above",
            market_price=0.60,
            model_probability=0.72,
            net_edge=0.09,
        )
        assert opp.ensemble_temps is None
        assert opp.forecast_temp == 0.0
        assert opp.rmse == 2.5
        assert opp.lead_time_days == 1
        assert opp.bucket_low is None
        assert opp.bucket_high is None
        assert opp.gaussian_prob is None
        assert opp.ensemble_prob is None
