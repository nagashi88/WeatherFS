"""Property-based tests via hypothesis for parser + portfolio aggregation.

These tests look for invariants the unit tests miss:

- The Polymarket title parser must NEVER raise on arbitrary string input —
  invalid titles return None, never crash. This is a critical property because
  the live loop iterates hundreds of titles and one rogue input must not abort
  the cycle.
- A round-trip property: take a known city/date/threshold, render it as a
  canonical title, parse it, and check the structured fields round-trip
  cleanly. This catches regex regressions that unit-test fixtures miss.
- Failure-rate counter monotonicity: every parse increments attempts by
  exactly 1, and failures by exactly 0 or 1.
- Aggregator invariant: aggregate_signals never returns more orders than
  markets passed in (one market = at most one order in our model).
- Aggregator invariant: a market with only no_trade votes never produces
  an order.
- Aggregator invariant: a market with side_scores tied between two sides
  never produces an order (CCG-bug-1 tied-side guard).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from weatheredge.agents.base import MarketOpportunity, Signal
from weatheredge.agents.portfolio_manager import PortfolioManager
from weatheredge.config import Settings
from weatheredge.data import market_parser as mp_module
from weatheredge.data.market_parser import (
    CITY_TO_STATION,
    ParsedMarket,
    get_parse_failure_stats,
    parse_polymarket_title,
    reset_parse_history_for_tests,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_parse_history():
    """Each property test runs many examples; isolate the rolling counter."""
    reset_parse_history_for_tests()
    yield
    reset_parse_history_for_tests()


def _make_settings() -> Settings:
    return Settings(
        polymarket_private_key="",
        polymarket_proxy_address="",
        kalshi_api_key="",
        kalshi_private_key="",
        anthropic_api_key="",
        telegram_bot_token="",
        telegram_chat_id="",
        kelly_fraction=0.15,
        max_open_positions=20,
        max_single_market_exposure=0.08,
        target_volatility=0.10,
        max_leverage=2.0,
        min_allocation=0.20,
        vpin_danger_threshold=0.70,
        current_risk_tier=100,
        initial_bankroll=1000.0,
    )


def _make_pm() -> PortfolioManager:
    rm = MagicMock()
    rm.pre_trade_check.return_value = {"blocked": False}
    rm.calculate_position_size.return_value = 10.0
    return PortfolioManager(_make_settings(), rm)


# ---------------------------------------------------------------------------
# Parser — robustness on arbitrary input
# ---------------------------------------------------------------------------


@given(st.text(max_size=200))
@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_parser_never_raises_on_arbitrary_text(title: str):
    """Invariant: parser returns either None or a ParsedMarket — never raises."""
    result = parse_polymarket_title(title)
    assert result is None or isinstance(result, ParsedMarket)


@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=300))
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_parser_unicode_safe(title: str):
    """Invariant: parser tolerates arbitrary unicode without raising.

    Polymarket sometimes ships titles with non-ASCII glyphs (degree sign,
    accented city names). The parser must absorb anything without crashing.
    """
    parse_polymarket_title(title)  # we only care about not raising


# ---------------------------------------------------------------------------
# Parser — round-trip property on canonical templates
# ---------------------------------------------------------------------------


# Cities we know the parser maps. Pull from CITY_TO_STATION so this stays in
# sync if new cities are added.
_TRADABLE_CITIES = sorted(
    {
        name
        for name in CITY_TO_STATION.keys()
        # Multi-word names get capitalised in titles; the parser handles both
        # cases via title.lower(), so the strategy can include them.
    }
)


@given(
    city=st.sampled_from(_TRADABLE_CITIES),
    threshold=st.integers(min_value=-30, max_value=120),
    month=st.sampled_from(
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"]
    ),
    day=st.integers(min_value=1, max_value=28),
    direction=st.sampled_from(["above", "below"]),
)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_parser_roundtrip_above_or_below_pattern(
    city: str, threshold: int, month: str, day: int, direction: str
):
    """Canonical "X°F or above/below" titles round-trip cleanly."""
    direction_word = "above" if direction == "above" else "below"
    title = (
        f"Will the highest temperature in {city.title()} be "
        f"{threshold}°F or {direction_word} on {month} {day}?"
    )
    result = parse_polymarket_title(title, current_year=2026)
    assert result is not None, f"failed to parse canonical title: {title!r}"
    assert result.city == city
    assert result.station == CITY_TO_STATION[city]
    assert result.threshold == float(threshold)
    assert result.direction == direction
    assert result.unit == "F"
    # Date string format check: YYYY-MM-DD.
    assert result.date == f"2026-{_month_num(month):02d}-{day:02d}"


@given(
    city=st.sampled_from(_TRADABLE_CITIES),
    low=st.integers(min_value=-20, max_value=110),
    span=st.integers(min_value=1, max_value=10),
    month=st.sampled_from(
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"]
    ),
    day=st.integers(min_value=1, max_value=28),
)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_parser_roundtrip_bucket_pattern(
    city: str, low: int, span: int, month: str, day: int
):
    """Canonical "between X-Y°F" bucket titles round-trip cleanly."""
    high = low + span
    title = (
        f"Will the highest temperature in {city.title()} be "
        f"between {low}-{high}°F on {month} {day}?"
    )
    result = parse_polymarket_title(title, current_year=2026)
    assert result is not None, f"failed to parse bucket title: {title!r}"
    assert result.threshold_low == float(low)
    assert result.threshold_high == float(high)
    assert result.threshold is None  # bucket markets do not set the single threshold
    assert result.city == city


@given(title=st.text(max_size=200))
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_parser_idempotent(title: str):
    """Calling the parser twice with the same input yields equal output."""
    a = parse_polymarket_title(title, current_year=2026)
    b = parse_polymarket_title(title, current_year=2026)
    if a is None:
        assert b is None
    else:
        assert b is not None
        assert (a.city, a.station, a.date, a.market_type, a.threshold,
                a.threshold_low, a.threshold_high, a.direction, a.unit) == (
                b.city, b.station, b.date, b.market_type, b.threshold,
                b.threshold_low, b.threshold_high, b.direction, b.unit)


# ---------------------------------------------------------------------------
# Failure-rate counter monotonicity
# ---------------------------------------------------------------------------


@given(titles=st.lists(st.text(max_size=80), min_size=1, max_size=50))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_parse_counter_monotonic(titles: list[str]):
    """attempts grows by exactly len(titles); failures grows by 0..len(titles).

    Per-call increments must be deterministic so the daily-summary aggregator
    can rely on the counter.
    """
    reset_parse_history_for_tests()
    before = get_parse_failure_stats(window_minutes=60)
    assert before["attempts"] == 0
    expected_failures = 0
    for t in titles:
        if parse_polymarket_title(t, current_year=2026) is None:
            expected_failures += 1
    after = get_parse_failure_stats(window_minutes=60)
    assert after["attempts"] == len(titles)
    assert after["failures"] == expected_failures
    assert 0 <= after["failure_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Portfolio aggregator invariants
# ---------------------------------------------------------------------------


def _market_id_strategy() -> st.SearchStrategy[str]:
    return st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
        min_size=3,
        max_size=12,
    )


_AGENT_NAMES = ["Conservative", "Aggressive", "MeanReversion", "Qualitative", "SellNO"]


def _signal_strategy(market_id: str) -> st.SearchStrategy[Signal]:
    return st.builds(
        Signal,
        market_id=st.just(market_id),
        side=st.sampled_from(["buy_yes", "buy_no", "no_trade"]),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        agent_name=st.sampled_from(_AGENT_NAMES),
    )


@st.composite
def _market_signal_payload(draw, mid: str) -> list[Signal]:
    """Generate a non-trivial mix of agent signals for one market.

    The aggregator's per-market invariant must hold under realistic
    heterogeneity: some markets vote unanimously, some have mixed sides,
    some are pure no_trade abstentions, some have ties (CCG-bug-1 path).
    Generating identical buy_yes votes — as the original test did — only
    exercises the happy path and lets a regression in the side-tie or
    abstention-only branch slip through.
    """
    n = draw(st.integers(min_value=1, max_value=4))
    agents = draw(
        st.lists(
            st.sampled_from(_AGENT_NAMES),
            min_size=n, max_size=n, unique=True,
        )
    )
    sides = draw(
        st.lists(
            st.sampled_from(["buy_yes", "buy_no", "no_trade"]),
            min_size=n, max_size=n,
        )
    )
    confidences = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n, max_size=n,
        )
    )
    return [
        Signal(market_id=mid, side=s, confidence=c, agent_name=a)
        for a, s, c in zip(agents, sides, confidences)
    ]


@given(
    market_ids=st.lists(_market_id_strategy(), min_size=0, max_size=4, unique=True),
    bankroll=st.floats(min_value=10.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
    data=st.data(),
)
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_aggregate_signals_returns_at_most_one_order_per_market(
    market_ids: list[str], bankroll: float, data
):
    """Invariant: |orders| <= |markets| (one market = at most one TradeOrder).

    Tested under heterogeneous signal mixes — unanimous, split, abstention-only,
    and tied — so a regression in any aggregator branch trips this test.
    """
    pm = _make_pm()
    signals_by_market: dict[str, list[Signal]] = {}
    opportunities: dict[str, MarketOpportunity] = {}
    for mid in market_ids:
        signals_by_market[mid] = data.draw(_market_signal_payload(mid))
        opportunities[mid] = MarketOpportunity(
            market_id=mid, platform="polymarket", city="NYC", station="KNYC",
            threshold=70.0, direction="above", market_price=0.45,
            model_probability=0.65, net_edge=0.18, net_edge_no=-0.18,
        )
    orders = pm.aggregate_signals(
        signals_by_market=signals_by_market,
        opportunities=opportunities,
        bankroll=bankroll,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )
    assert len(orders) <= len(market_ids)
    # Stronger: no two orders may share the same market_id.
    market_ids_in_orders = [o.market_id for o in orders]
    assert len(market_ids_in_orders) == len(set(market_ids_in_orders))


@given(
    n_signals=st.integers(min_value=1, max_value=5),
    bankroll=st.floats(min_value=100.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_aggregate_signals_no_trade_only_yields_no_orders(
    n_signals: int, bankroll: float
):
    """Invariant: a market with only no_trade votes produces zero orders."""
    pm = _make_pm()
    signals = [
        Signal(market_id="M1", side="no_trade", confidence=1.0,
               agent_name=_AGENT_NAMES[i % len(_AGENT_NAMES)])
        for i in range(n_signals)
    ]
    opp = MarketOpportunity(
        market_id="M1", platform="polymarket", city="NYC", station="KNYC",
        threshold=70.0, direction="above", market_price=0.50,
        model_probability=0.55, net_edge=0.05, net_edge_no=-0.05,
    )
    orders = pm.aggregate_signals(
        signals_by_market={"M1": signals},
        opportunities={"M1": opp},
        bankroll=bankroll,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )
    assert orders == []


@given(
    confidence=st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_tied_side_scores_skipped(confidence: float):
    """When side_scores are tied between buy_yes and buy_no, skip the market.

    CCG-bug-1: max() on a tied dict picks insertion order, which is non-
    deterministic across Python versions. The aggregator must skip rather
    than gamble on the tie. We exploit the fact that Aggressive and SellNO
    share the same weight (0.25) so equal-confidence votes from each on
    opposite sides produce a perfect tie.
    """
    assert (
        PortfolioManager.AGENT_WEIGHTS["Aggressive"]
        == PortfolioManager.AGENT_WEIGHTS["SellNO"]
    ), "test premise broken — Aggressive/SellNO weights drifted"
    pm = _make_pm()
    signals = [
        Signal(market_id="M1", side="buy_yes", confidence=confidence, agent_name="Aggressive"),
        Signal(market_id="M1", side="buy_no", confidence=confidence, agent_name="SellNO"),
    ]
    opp = MarketOpportunity(
        market_id="M1", platform="polymarket", city="NYC", station="KNYC",
        threshold=70.0, direction="above", market_price=0.50,
        model_probability=0.55, net_edge=0.05, net_edge_no=-0.05,
    )
    orders = pm.aggregate_signals(
        signals_by_market={"M1": signals},
        opportunities={"M1": opp},
        bankroll=1000.0,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )
    assert orders == [], f"tied side scores should be skipped — got {orders!r}"


@given(
    n_markets=st.integers(min_value=1, max_value=5),
    bankroll=st.floats(
        min_value=10.0, max_value=10_000.0,
        allow_nan=False, allow_infinity=False,
    ),
)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_markets_with_empty_signal_lists_yield_no_orders(
    n_markets: int, bankroll: float
):
    """Invariant: a market keyed in the dict but carrying an empty signal list
    must not produce an order (no votes ⇒ no decision).

    The original test ignored `n_signals` and only exercised the
    `signals_by_market == {}` shortcut. This version actually walks the
    aggregator over real (but empty) market entries — the path that runs in
    production when every agent times out or aborts before voting.
    """
    pm = _make_pm()
    market_ids = [f"M{i}" for i in range(n_markets)]
    signals_by_market: dict[str, list[Signal]] = {mid: [] for mid in market_ids}
    opportunities = {
        mid: MarketOpportunity(
            market_id=mid, platform="polymarket", city="NYC", station="KNYC",
            threshold=70.0, direction="above", market_price=0.45,
            model_probability=0.65, net_edge=0.18, net_edge_no=-0.18,
        )
        for mid in market_ids
    }
    orders = pm.aggregate_signals(
        signals_by_market=signals_by_market,
        opportunities=opportunities,
        bankroll=bankroll,
        open_positions=[],
        daily_pnl=0.0,
        total_pnl=0.0,
    )
    assert orders == []


# ---------------------------------------------------------------------------
# Daily summary — midnight boundary invariant
# ---------------------------------------------------------------------------


from contextlib import contextmanager
from datetime import date, datetime, time, timedelta, timezone
from unittest.mock import patch

from sqlalchemy.orm import sessionmaker

from weatheredge.db.database import init_db, make_engine
from weatheredge.db.models import Trade as TradeRow


@given(
    pnl=st.floats(
        min_value=-100.0, max_value=100.0,
        allow_nan=False, allow_infinity=False,
    ),
    minutes_after_midnight=st.integers(min_value=0, max_value=24 * 60 - 1),
)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_resolved_at_falls_on_correct_summary_day(
    pnl: float, minutes_after_midnight: int,
):
    """Invariant (Fix #2): a trade with `resolved_at = D + minutes` must
    contribute its P&L to day D's summary, never the previous or next day.

    The midnight boundary is the silent-corruption regression magnet — a
    trade resolved at 00:00:01 on day D should *not* land on day D-1, and a
    trade resolved at 23:59:59 should *not* land on day D+1. Generates random
    times across the 24h window and asserts the bucketing exactly.
    """
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    factory = sessionmaker(bind=eng, autocommit=False, autoflush=False)

    @contextmanager
    def fake_get_session(_engine=None):
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    target_day = date(2026, 4, 18)
    midnight = datetime.combine(target_day, time.min, tzinfo=timezone.utc)
    resolved_at = midnight + timedelta(minutes=minutes_after_midnight)
    # created_at can be earlier — the bug was specifically that the original
    # COALESCE-style query mixed `created_at` and `resolved_at` and lost
    # cross-day trades. Place the trade *yesterday* so the only thing
    # bucketing it into target_day is `resolved_at`.
    created_at = midnight - timedelta(hours=2)

    session = factory()
    session.add(TradeRow(
        market_id="m_boundary",
        platform="polymarket",
        mode="paper",
        side="buy_yes",
        price=0.5,
        amount=10.0,
        cost=5.0,
        agent_name="Aggressive",
        status="won",
        pnl=pnl,
        created_at=created_at,
        resolved_at=resolved_at,
    ))
    session.commit()
    session.close()

    with patch("weatheredge.daily_summary.get_session", fake_get_session):
        from weatheredge.daily_summary import compute_daily_summary
        s_today = compute_daily_summary(target_day=target_day)
        s_yesterday = compute_daily_summary(target_day=target_day - timedelta(days=1))
        s_tomorrow = compute_daily_summary(target_day=target_day + timedelta(days=1))

    eng.dispose()

    # Resolved P&L must land on target_day, not adjacent days.
    assert s_today["resolved_trades"] == 1
    assert s_today["realized_pnl"] == pytest.approx(pnl)
    assert s_yesterday["resolved_trades"] == 0
    assert s_yesterday["realized_pnl"] == 0.0
    assert s_tomorrow["resolved_trades"] == 0
    assert s_tomorrow["realized_pnl"] == 0.0
    # And the placement count goes to *yesterday* (created_at), not today.
    assert s_yesterday["trades"] == 1
    assert s_today["trades"] == 0


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _month_num(name: str) -> int:
    return [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ].index(name.lower()) + 1
