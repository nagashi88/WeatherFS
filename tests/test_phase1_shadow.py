from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weatheredge.agents.base import MarketOpportunity
from weatheredge.db.database import get_session, init_db
from weatheredge.db.models import Phase1ShadowRejection
from weatheredge.execution.phase1_shadow import (
    PHASE1_SHADOW_POLICY,
    record_phase1_shadow_only_rejection,
    resolve_phase1_shadow_rejections_for_market,
)


def _opp(**overrides) -> MarketOpportunity:
    base = dict(
        market_id="phase1-mkt",
        platform="polymarket",
        city="NYC",
        station="KNYC",
        threshold=65.0,
        direction="above",
        market_price=0.40,
        model_probability=0.12,
        net_edge=-0.25,
        net_edge_no=0.20,
        market_type="daily_high",
        market_date="2026-05-01",
        bucket_low=64.0,
        bucket_high=65.0,
        raw_model_prob=0.12,
        forecast_temp=61.0,
        volume=2500.0,
        spread=0.02,
        forecast_timestamp=datetime(2026, 5, 1, 10, tzinfo=timezone.utc),
        market_price_timestamp=datetime(2026, 5, 1, 10, 5, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return MarketOpportunity(**base)


@pytest.fixture()
def db():
    init_db()
    yield


def test_phase1_shadow_rejection_records_even_when_not_floor_twin_candidate(db):
    inserted = record_phase1_shadow_only_rejection(
        _opp(raw_model_prob=0.14),
        reason="phase1_same_day_narrow_bucket_buy_no",
        majority_side="buy_no",
        majority_score=1.30,
        side_margin=1.10,
        distinct_majority=2,
    )
    inserted_again = record_phase1_shadow_only_rejection(
        _opp(raw_model_prob=0.14, market_price=0.50),
        reason="phase1_same_day_narrow_bucket_buy_no",
        majority_side="buy_no",
        majority_score=1.10,
        side_margin=0.90,
        distinct_majority=2,
    )

    assert inserted is True
    assert inserted_again is False
    with get_session() as session:
        row = session.query(Phase1ShadowRejection).one()
        assert row.policy_name == PHASE1_SHADOW_POLICY
        assert row.market_id == "phase1-mkt"
        assert row.status == "open"
        assert row.raw_model_prob == pytest.approx(0.14)
        assert row.entry_yes_price == pytest.approx(0.40)
        assert row.cost == pytest.approx(1.0)
        assert row.observation_count == 2
        assert row.majority_score == pytest.approx(1.10)


def test_phase1_shadow_rejection_resolves_counterfactual_pnl(db):
    record_phase1_shadow_only_rejection(
        _opp(market_price=0.40),
        reason="phase1_same_day_narrow_bucket_buy_no",
        majority_side="buy_no",
        majority_score=1.30,
        side_margin=1.10,
        distinct_majority=2,
    )

    resolved = resolve_phase1_shadow_rejections_for_market("phase1-mkt", "no")

    assert resolved == 1
    with get_session() as session:
        row = session.query(Phase1ShadowRejection).one()
        assert row.status == "won"
        assert row.outcome == "no"
        assert row.pnl == pytest.approx((1.0 / 0.60) - 1.0, abs=1e-6)


def test_phase1_shadow_rejection_ignores_non_no_majority(db):
    inserted = record_phase1_shadow_only_rejection(
        _opp(),
        reason="phase1_same_day_narrow_bucket_buy_no",
        majority_side="buy_yes",
        majority_score=1.30,
        side_margin=1.10,
        distinct_majority=2,
    )

    assert inserted is False
    with get_session() as session:
        assert session.query(Phase1ShadowRejection).count() == 0
