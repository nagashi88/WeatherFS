"""Tests for US-OL — orderbook logging columns on MarketPrice."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import weatheredge.config as _cfg
from weatheredge.db.models import Base, MarketPrice, Market


def _make_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


def _seed_market(session: Session, market_id: str = "test-mkt") -> None:
    session.add(Market(
        id=market_id, platform="polymarket", city="NYC",
        station="KNYC", date="2026-04-10", market_type="daily_high",
    ))
    session.commit()


class TestOrderbookColumns:
    def test_columns_populated_when_flag_on(self):
        engine = _make_engine()
        with Session(engine) as sess:
            _seed_market(sess)
            row = MarketPrice(
                market_id="test-mkt",
                timestamp=datetime.now(timezone.utc),
                yes_price=0.55,
                no_price=0.45,
                volume=5000,
                spread=0.02,
                best_bid=0.54,
                best_ask=0.56,
                bid_size=100.0,
                ask_size=150.0,
            )
            sess.add(row)
            sess.commit()
            fetched = sess.query(MarketPrice).first()
            assert fetched.best_bid == 0.54
            assert fetched.best_ask == 0.56
            assert fetched.bid_size == 100.0
            assert fetched.ask_size == 150.0

    def test_columns_nullable_when_flag_off(self):
        engine = _make_engine()
        with Session(engine) as sess:
            _seed_market(sess)
            row = MarketPrice(
                market_id="test-mkt",
                timestamp=datetime.now(timezone.utc),
                yes_price=0.55,
                no_price=0.45,
                volume=5000,
                spread=0.02,
                best_bid=None,
                best_ask=None,
                bid_size=None,
                ask_size=None,
            )
            sess.add(row)
            sess.commit()
            fetched = sess.query(MarketPrice).first()
            assert fetched.best_bid is None
            assert fetched.best_ask is None
            assert fetched.bid_size is None
            assert fetched.ask_size is None

    def test_capture_site_respects_flag(self, monkeypatch):
        """When MODULES['orderbook_logging'] is False, fields stored as None."""
        monkeypatch.setitem(_cfg.MODULES, "orderbook_logging", False)

        engine = _make_engine()
        with Session(engine) as sess:
            _seed_market(sess)

            from weatheredge.data.market import MarketScanner
            scanner = MarketScanner.__new__(MarketScanner)
            price_data = {
                "yes_price": 0.55, "no_price": 0.45,
                "volume": 5000, "spread": 0.02,
                "best_bid": 0.54, "best_ask": 0.56,
                "bid_size": 100.0, "ask_size": 150.0,
            }
            scanner.store_market_price(sess, "test-mkt", price_data)
            sess.commit()

            fetched = sess.query(MarketPrice).first()
            assert fetched.best_bid is None
            assert fetched.best_ask is None

    def test_capture_site_stores_when_flag_on(self, monkeypatch):
        """When MODULES['orderbook_logging'] is True, fields are populated."""
        monkeypatch.setitem(_cfg.MODULES, "orderbook_logging", True)

        engine = _make_engine()
        with Session(engine) as sess:
            _seed_market(sess)

            from weatheredge.data.market import MarketScanner
            scanner = MarketScanner.__new__(MarketScanner)
            price_data = {
                "yes_price": 0.55, "no_price": 0.45,
                "volume": 5000, "spread": 0.02,
                "best_bid": 0.54, "best_ask": 0.56,
                "bid_size": 100.0, "ask_size": 150.0,
            }
            scanner.store_market_price(sess, "test-mkt", price_data)
            sess.commit()

            fetched = sess.query(MarketPrice).first()
            assert fetched.best_bid == 0.54
            assert fetched.best_ask == 0.56
