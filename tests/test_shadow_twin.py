from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weatheredge.agents.base import MarketOpportunity
from weatheredge.config import Settings
from weatheredge.db.database import get_session, init_db
from weatheredge.db.models import ShadowPolicyTrade, ShadowTwinTrade
from weatheredge.execution.shadow_twin import (
    OPPORTUNITY_BASKET_FLOOR_NO_ADAPTIVE_LIQUIDITY,
    POLICY_HIGH_EDGE,
    POLICY_HOLD,
    POLICY_MAIN_EXIT,
    POLICY_PRICE_BAND,
    TWIN_BROAD,
    TWIN_BROAD_ADAPTIVE_LIQUIDITY,
    TWIN_CORE,
    TWIN_CORE_ADAPTIVE_LIQUIDITY,
    TWIN_EXPANDED,
    TWIN_EXPANDED_ADAPTIVE_LIQUIDITY,
    adaptive_liquidity_passes,
    is_floor_bound_buy_no_candidate,
    is_main_trading_station,
    record_adaptive_liquidity_shadow_twins,
    record_shadow_twins,
    resolve_shadow_trades_for_market,
)

ALL_POLICIES = {POLICY_HOLD, POLICY_MAIN_EXIT, POLICY_PRICE_BAND, POLICY_HIGH_EDGE}


def _opp(**overrides) -> MarketOpportunity:
    base = dict(
        market_id="m1",
        platform="polymarket",
        city="NYC",
        station="KNYC",
        threshold=70.0,
        direction="above",
        market_price=0.40,
        model_probability=0.10,
        net_edge=-0.30,
        net_edge_no=0.20,
        market_type="daily_high",
        market_date="2026-04-30",
        bucket_low=70.0,
        bucket_high=71.0,
        raw_model_prob=0.04,
        volume=300.0,
        spread=0.02,
        forecast_timestamp=datetime(2026, 4, 30, 10, tzinfo=timezone.utc),
        market_price_timestamp=datetime(2026, 4, 30, 10, 5, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return MarketOpportunity(**base)


@pytest.fixture()
def db():
    init_db()
    yield


def _settings() -> Settings:
    settings = Settings()
    settings.shadow_twin_enabled = True
    settings.shadow_twin_broad_enabled = True
    settings.shadow_twin_adaptive_liquidity_enabled = True
    return settings


def test_shadow_candidate_detects_floor_bound_bucket_no():
    settings = _settings()
    assert is_floor_bound_buy_no_candidate(_opp(), settings) is True
    assert is_floor_bound_buy_no_candidate(_opp(raw_model_prob=0.12), settings) is False
    assert is_floor_bound_buy_no_candidate(_opp(metar_override=0.85), settings) is False
    assert is_floor_bound_buy_no_candidate(_opp(net_edge_no=0.05), settings) is False


def test_main_station_filter_excludes_expanded_shadow_only_city():
    settings = _settings()
    assert is_main_trading_station("KNYC", settings) is True
    assert is_main_trading_station("KAUS", settings) is False


def test_record_shadow_twins_core_and_duplicate_guard(db):
    settings = _settings()
    counts = record_shadow_twins({"m1": _opp()}, settings)
    assert counts == {TWIN_CORE: 1, TWIN_EXPANDED: 1, TWIN_BROAD: 1}

    counts = record_shadow_twins({"m1": _opp()}, settings)
    assert counts == {TWIN_CORE: 0, TWIN_EXPANDED: 0, TWIN_BROAD: 0}

    with get_session() as session:
        rows = session.query(ShadowTwinTrade).all()
        assert len(rows) == 3
        assert {row.twin_name for row in rows} == {TWIN_CORE, TWIN_EXPANDED, TWIN_BROAD}
        assert all(row.side == "buy_no" for row in rows)
        assert all(row.status == "open" for row in rows)
        policy_rows = session.query(ShadowPolicyTrade).all()
        assert len(policy_rows) == 12
        assert {row.policy_name for row in policy_rows} == ALL_POLICIES
        assert {row.universe_name for row in policy_rows} == {TWIN_CORE, TWIN_EXPANDED, TWIN_BROAD}


def test_record_shadow_twins_expanded_extra_city(db):
    settings = _settings()
    counts = record_shadow_twins({"m2": _opp(market_id="m2", city="Austin", station="KAUS")}, settings)
    assert counts == {TWIN_CORE: 0, TWIN_EXPANDED: 1, TWIN_BROAD: 1}

    with get_session() as session:
        rows = session.query(ShadowTwinTrade).all()
        assert len(rows) == 2
        assert {row.twin_name for row in rows} == {TWIN_EXPANDED, TWIN_BROAD}
        assert {row.station for row in rows} == {"KAUS"}
        policy_rows = session.query(ShadowPolicyTrade).all()
        assert len(policy_rows) == 8
        assert {row.policy_name for row in policy_rows} == ALL_POLICIES
        assert {row.universe_name for row in policy_rows} == {TWIN_EXPANDED, TWIN_BROAD}


def test_record_shadow_twins_broad_only_city(db):
    settings = _settings()
    counts = record_shadow_twins({"m3": _opp(market_id="m3", city="London", station="EGLC")}, settings)
    assert counts == {TWIN_CORE: 0, TWIN_EXPANDED: 0, TWIN_BROAD: 1}

    with get_session() as session:
        row = session.query(ShadowTwinTrade).one()
        assert row.twin_name == TWIN_BROAD
        assert row.station == "EGLC"
        policy_rows = session.query(ShadowPolicyTrade).all()
        assert len(policy_rows) == 4
        assert {row.policy_name for row in policy_rows} == ALL_POLICIES
        assert {row.universe_name for row in policy_rows} == {TWIN_BROAD}


def test_record_shadow_twins_four_letter_broad_city_name(db):
    settings = _settings()
    counts = record_shadow_twins({"m4": _opp(market_id="m4", city="Rome", station="LIRF")}, settings)
    assert counts == {TWIN_CORE: 0, TWIN_EXPANDED: 0, TWIN_BROAD: 1}

    with get_session() as session:
        row = session.query(ShadowTwinTrade).one()
        assert row.twin_name == TWIN_BROAD
        assert row.station == "LIRF"


def test_default_broad_universe_includes_promoted_weather_cities():
    settings = _settings()
    broad = set(settings.shadow_twin_broad_cities_list)

    assert {
        "Rome",
        "Amsterdam",
        "Cape Town",
        "Lagos",
        "Helsinki",
        "Jeddah",
        "Karachi",
        "Jakarta",
        "Manila",
        "Kuala Lumpur",
        "Guangzhou",
        "Busan",
        "Panama City",
        "Qingdao",
    }.issubset(broad)
    assert len(settings.shadow_twin_broad_cities_list) == 41
    assert len(settings.tracked_cities_list) == 54


def test_broad_universe_can_be_disabled_for_core_expanded_only(db):
    settings = _settings()
    settings.shadow_twin_broad_enabled = False

    assert len(settings.tracked_cities_list) == 13

    core_counts = record_shadow_twins({"m1": _opp()}, settings)
    expanded_counts = record_shadow_twins(
        {"m2": _opp(market_id="m2", city="Austin", station="KAUS")},
        settings,
    )
    broad_counts = record_shadow_twins(
        {"m3": _opp(market_id="m3", city="London", station="EGLC")},
        settings,
    )

    assert core_counts == {TWIN_CORE: 1, TWIN_EXPANDED: 1, TWIN_BROAD: 0}
    assert expanded_counts == {TWIN_CORE: 0, TWIN_EXPANDED: 1, TWIN_BROAD: 0}
    assert broad_counts == {TWIN_CORE: 0, TWIN_EXPANDED: 0, TWIN_BROAD: 0}

    with get_session() as session:
        assert {row.twin_name for row in session.query(ShadowTwinTrade).all()} == {
            TWIN_CORE,
            TWIN_EXPANDED,
        }


def test_record_shadow_twins_rejects_non_causal_opportunity(db):
    settings = _settings()
    opp = _opp(
        forecast_timestamp=datetime(2026, 4, 30, 10, 10, tzinfo=timezone.utc),
        market_price_timestamp=datetime(2026, 4, 30, 10, 5, tzinfo=timezone.utc),
    )
    counts = record_shadow_twins({"m1": opp}, settings)
    assert counts == {TWIN_CORE: 0, TWIN_EXPANDED: 0, TWIN_BROAD: 0}

    with get_session() as session:
        assert session.query(ShadowTwinTrade).count() == 0
        assert session.query(ShadowPolicyTrade).count() == 0


def test_adaptive_liquidity_gate_can_pass_below_main_volume():
    settings = _settings()
    ok, reason, min_volume = adaptive_liquidity_passes(300.0, 0.02, 1.0, settings)
    assert ok is True
    assert reason == ""
    assert min_volume == pytest.approx(250.0)

    ok, reason, min_volume = adaptive_liquidity_passes(200.0, 0.02, 1.0, settings)
    assert ok is False
    assert reason == "adaptive_liquidity_low_volume"
    assert min_volume == pytest.approx(250.0)


def test_record_adaptive_liquidity_shadow_twins_separate_universe_and_basket(db):
    settings = _settings()
    counts = record_adaptive_liquidity_shadow_twins({"m1": _opp()}, settings)
    assert counts == {
        TWIN_CORE_ADAPTIVE_LIQUIDITY: 1,
        TWIN_EXPANDED_ADAPTIVE_LIQUIDITY: 1,
        TWIN_BROAD_ADAPTIVE_LIQUIDITY: 1,
    }

    with get_session() as session:
        rows = session.query(ShadowTwinTrade).all()
        assert len(rows) == 3
        assert {row.twin_name for row in rows} == {
            TWIN_CORE_ADAPTIVE_LIQUIDITY,
            TWIN_EXPANDED_ADAPTIVE_LIQUIDITY,
            TWIN_BROAD_ADAPTIVE_LIQUIDITY,
        }
        policy_rows = session.query(ShadowPolicyTrade).all()
        assert len(policy_rows) == 12
        assert {row.universe_name for row in policy_rows} == {
            TWIN_CORE_ADAPTIVE_LIQUIDITY,
            TWIN_EXPANDED_ADAPTIVE_LIQUIDITY,
            TWIN_BROAD_ADAPTIVE_LIQUIDITY,
        }
        assert {row.opportunity_basket for row in policy_rows} == {OPPORTUNITY_BASKET_FLOOR_NO_ADAPTIVE_LIQUIDITY}
        assert {row.entry_volume for row in policy_rows} == {300.0}
        assert {row.entry_spread for row in policy_rows} == {0.02}
        assert all(row.liquidity_min_volume == pytest.approx(250.0) for row in policy_rows)


def test_resolve_shadow_trades_hold_to_resolution(db):
    settings = _settings()
    record_shadow_twins({"m1": _opp()}, settings)

    resolved = resolve_shadow_trades_for_market("m1", "no")
    assert resolved == 15

    with get_session() as session:
        rows = session.query(ShadowTwinTrade).all()
        assert len(rows) == 3
        assert all(row.status == "won" for row in rows)
        assert all(row.pnl == pytest.approx(row.amount - row.cost) for row in rows)
        policy_rows = session.query(ShadowPolicyTrade).all()
        assert len(policy_rows) == 12
        assert all(row.status == "won" for row in policy_rows)
        assert all(row.pnl == pytest.approx(row.amount - row.cost) for row in policy_rows)
