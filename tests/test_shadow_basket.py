from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weatheredge.agents.base import MarketOpportunity
from weatheredge.config import Settings
from weatheredge.db.database import get_session, init_db
from weatheredge.db.models import ShadowBasketLeg, ShadowBasketTrade
from weatheredge.execution.shadow_basket import (
    SHADOW_BASKET_POLICY,
    build_shadow_basket_plan,
    record_shadow_baskets,
    resolve_shadow_baskets_for_market,
)


def _settings() -> Settings:
    settings = Settings()
    settings.shadow_twin_enabled = True
    settings.shadow_basket_enabled = True
    settings.shadow_twin_broad_enabled = False
    settings.shadow_twin_adaptive_liquidity_enabled = False
    settings.shadow_basket_min_volume = 300.0
    settings.shadow_basket_max_spread_pct = 0.05
    settings.shadow_basket_adverse_tick = 0.01
    settings.shadow_basket_min_worst_case_abs = 0.20
    settings.shadow_basket_min_worst_case_fraction = 0.05
    return settings


def _opp(market_id: str, low: float, high: float, yes: float, **overrides) -> MarketOpportunity:
    base = dict(
        market_id=market_id,
        platform="polymarket",
        city="Houston",
        station="KIAH",
        threshold=(low + high) / 2.0,
        direction="above",
        market_price=yes,
        model_probability=0.10,
        net_edge=-0.25,
        net_edge_no=0.20,
        market_type="daily_high",
        market_date="2026-04-30",
        bucket_low=low,
        bucket_high=high,
        bucket_low_int=int(low),
        bucket_high_int=int(high),
        raw_model_prob=0.04,
        volume=2000.0,
        spread=0.02,
        forecast_temp=78.0,
        forecast_timestamp=datetime(2026, 4, 30, 10, tzinfo=timezone.utc),
        market_price_timestamp=datetime(2026, 4, 30, 10, 5, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return MarketOpportunity(**base)


@pytest.fixture()
def db():
    init_db()
    yield


def _houston_opps() -> dict[str, MarketOpportunity]:
    return {
        "kia-82": _opp("kia-82", 82, 83, 0.255),
        "kia-84": _opp("kia-84", 84, 85, 0.2155),
        "kia-86": _opp("kia-86", 86, 87, 0.365),
        "kia-88": _opp("kia-88", 88, 89, 0.360),
    }


def test_build_shadow_basket_plan_flattens_selected_bucket_payoffs():
    settings = _settings()
    plans = build_shadow_basket_plan(_houston_opps(), settings)

    assert len(plans) == 1
    plan = plans[0]
    assert plan.policy_name == SHADOW_BASKET_POLICY
    assert len(plan.legs) == 4
    assert plan.cost == pytest.approx(4.0)
    assert plan.payoff_none > 1.50
    assert plan.worst_case_payoff >= plan.worst_case_threshold
    assert set(plan.scenario_payoffs) == set(_houston_opps())
    assert len({round(v, 6) for v in plan.scenario_payoffs.values()}) == 1


def test_record_shadow_basket_is_idempotent(db):
    settings = _settings()

    counts = record_shadow_baskets(_houston_opps(), settings)
    counts_again = record_shadow_baskets(_houston_opps(), settings)

    assert counts == {SHADOW_BASKET_POLICY: 1}
    assert counts_again == {SHADOW_BASKET_POLICY: 0}
    with get_session() as session:
        basket = session.query(ShadowBasketTrade).one()
        legs = session.query(ShadowBasketLeg).all()
        assert basket.leg_count == 4
        assert basket.status == "open"
        assert basket.worst_case_payoff >= basket.worst_case_threshold
        assert len(legs) == 4
        assert {leg.side for leg in legs} == {"buy_no"}
        assert all(leg.fill_no_price > leg.entry_no_price for leg in legs)


def test_shadow_basket_rejects_fragile_two_leg_or_non_causal_candidates(db):
    settings = _settings()
    two_legs = {
        "kia-82": _opp("kia-82", 82, 83, 0.255),
        "kia-84": _opp("kia-84", 84, 85, 0.2155),
    }
    assert record_shadow_baskets(two_legs, settings) == {SHADOW_BASKET_POLICY: 0}

    non_causal = _houston_opps()
    non_causal["kia-82"] = _opp(
        "kia-82",
        82,
        83,
        0.255,
        forecast_timestamp=datetime(2026, 4, 30, 10, 10, tzinfo=timezone.utc),
        market_price_timestamp=datetime(2026, 4, 30, 10, 5, tzinfo=timezone.utc),
    )
    assert record_shadow_baskets(non_causal, settings) == {SHADOW_BASKET_POLICY: 0}

    with get_session() as session:
        assert session.query(ShadowBasketTrade).count() == 0
        assert session.query(ShadowBasketLeg).count() == 0


def test_resolve_shadow_basket_closes_only_after_all_legs_resolve(db):
    settings = _settings()
    record_shadow_baskets(_houston_opps(), settings)

    assert resolve_shadow_baskets_for_market("kia-82", "no") == 1
    assert resolve_shadow_baskets_for_market("kia-84", "no") == 1
    assert resolve_shadow_baskets_for_market("kia-86", "yes") == 1

    with get_session() as session:
        basket = session.query(ShadowBasketTrade).one()
        assert basket.status == "open"

    assert resolve_shadow_baskets_for_market("kia-88", "no") == 2

    with get_session() as session:
        basket = session.query(ShadowBasketTrade).one()
        legs = session.query(ShadowBasketLeg).all()
        assert basket.status == "won"
        assert basket.outcome_market_id == "kia-86"
        assert basket.pnl == pytest.approx(sum(float(leg.pnl or 0.0) for leg in legs))
        assert basket.pnl >= basket.worst_case_payoff - 0.01
