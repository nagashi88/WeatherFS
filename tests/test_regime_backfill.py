"""Tests for regime backfill on the trades table."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from weatheredge.db.database import _backfill_trade_regimes
from weatheredge.db.models import Base, Market, Trade


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


def _add_market(session, market_id, threshold_low=None, threshold_high=None):
    session.add(Market(
        id=market_id, platform="kalshi", city="nyc", station="KNYC",
        date="2026-04-26", market_type="daily_high",
        threshold_low=threshold_low, threshold_high=threshold_high,
    ))


def _add_trade(session, **kwargs):
    defaults = dict(platform="kalshi", mode="paper", price=0.5, amount=1.0, cost=0.5)
    defaults.update(kwargs)
    session.add(Trade(**defaults))


def test_backfill_floor_bound_buy_no_on_bucket(engine):
    with Session(engine) as session:
        _add_market(session, "M1", threshold_low=76.0, threshold_high=77.0)
        _add_trade(session, market_id="M1", side="buy_no",
                   model_probability=0.90, raw_model_prob=0.05)
        session.commit()

    _backfill_trade_regimes(engine)

    with Session(engine) as session:
        t = session.query(Trade).one()
        assert t.regime == "floor_bound_buy_no"


def test_backfill_normal_bucket_for_buy_no_above_floor(engine):
    with Session(engine) as session:
        _add_market(session, "M1", threshold_low=76.0, threshold_high=77.0)
        _add_trade(session, market_id="M1", side="buy_no",
                   model_probability=0.75, raw_model_prob=0.25)
        session.commit()

    _backfill_trade_regimes(engine)

    with Session(engine) as session:
        assert session.query(Trade).one().regime == "normal_bucket"


def test_backfill_metar_override_yes_heuristic_on_bucket(engine):
    with Session(engine) as session:
        _add_market(session, "M1", threshold_low=76.0, threshold_high=77.0)
        _add_trade(session, market_id="M1", side="buy_yes",
                   model_probability=0.90, raw_model_prob=0.05)
        session.commit()

    _backfill_trade_regimes(engine)

    with Session(engine) as session:
        assert session.query(Trade).one().regime == "metar_override_yes"


def test_backfill_no_metar_heuristic_below_threshold(engine):
    with Session(engine) as session:
        _add_market(session, "M1", threshold_low=76.0, threshold_high=77.0)
        _add_trade(session, market_id="M1", side="buy_yes",
                   model_probability=0.80, raw_model_prob=0.50)
        session.commit()

    _backfill_trade_regimes(engine)

    with Session(engine) as session:
        # 0.80 < 0.85 heuristic floor → not classified as METAR override.
        assert session.query(Trade).one().regime == "normal_bucket"


def test_backfill_threshold_no_on_directional_market(engine):
    with Session(engine) as session:
        _add_market(session, "M1", threshold_low=None, threshold_high=None)
        _add_trade(session, market_id="M1", side="buy_no",
                   model_probability=0.20, raw_model_prob=0.20)
        session.commit()

    _backfill_trade_regimes(engine)

    with Session(engine) as session:
        assert session.query(Trade).one().regime == "threshold_no"


def test_backfill_orphan_market_id_treated_as_non_bucket(engine):
    with Session(engine) as session:
        _add_trade(session, market_id="DOES-NOT-EXIST", side="buy_no",
                   model_probability=0.30, raw_model_prob=0.30)
        session.commit()

    _backfill_trade_regimes(engine)

    with Session(engine) as session:
        # Missing markets row → LEFT JOIN yields NULL thresholds → not bucket.
        assert session.query(Trade).one().regime == "threshold_no"


def test_backfill_idempotent_skips_already_set_rows(engine):
    with Session(engine) as session:
        _add_market(session, "M1", threshold_low=76.0, threshold_high=77.0)
        _add_trade(session, market_id="M1", side="buy_no",
                   model_probability=0.90, raw_model_prob=0.05,
                   regime="hand_set_value")
        session.commit()

    _backfill_trade_regimes(engine)

    with Session(engine) as session:
        # Pre-existing regime preserved; backfill ignores rows where regime IS NOT NULL.
        assert session.query(Trade).one().regime == "hand_set_value"


def test_backfill_re_run_is_noop(engine):
    with Session(engine) as session:
        _add_market(session, "M1", threshold_low=76.0, threshold_high=77.0)
        _add_trade(session, market_id="M1", side="buy_no",
                   model_probability=0.90, raw_model_prob=0.05)
        session.commit()

    _backfill_trade_regimes(engine)
    _backfill_trade_regimes(engine)

    with Session(engine) as session:
        assert session.query(Trade).one().regime == "floor_bound_buy_no"


def test_backfill_handles_empty_trades_table(engine):
    # No trades → backfill should be a silent no-op.
    _backfill_trade_regimes(engine)
    with Session(engine) as session:
        assert session.query(Trade).count() == 0
