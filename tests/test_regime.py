"""Unit tests for trade regime classification."""
from weatheredge.regime import (
    classify_regime,
    REGIME_FLOOR_BOUND_NO,
    REGIME_METAR_OVERRIDE_NO,
    REGIME_METAR_OVERRIDE_YES,
    REGIME_NORMAL_BUCKET,
    REGIME_THRESHOLD_NO,
    REGIME_THRESHOLD_YES,
    REGIME_UNKNOWN,
)


def test_metar_override_takes_priority_over_floor_bound():
    assert (
        classify_regime(
            side="buy_no",
            is_bucket=True,
            raw_model_prob=0.005,
            metar_override=0.90,
        )
        == REGIME_METAR_OVERRIDE_NO
    )


def test_metar_override_yes_on_directional():
    assert (
        classify_regime(
            side="buy_yes",
            is_bucket=False,
            raw_model_prob=0.50,
            metar_override=0.90,
        )
        == REGIME_METAR_OVERRIDE_YES
    )


def test_floor_bound_buy_no_on_bucket():
    assert (
        classify_regime(
            side="buy_no",
            is_bucket=True,
            raw_model_prob=0.05,
            metar_override=None,
        )
        == REGIME_FLOOR_BOUND_NO
    )


def test_floor_bound_boundary_at_floor_value_excludes():
    # raw_model_prob == PROB_FLOOR is NOT below floor → not floor-bound.
    assert (
        classify_regime(
            side="buy_no",
            is_bucket=True,
            raw_model_prob=0.10,
            metar_override=None,
        )
        == REGIME_NORMAL_BUCKET
    )


def test_normal_bucket_when_raw_prob_above_floor():
    assert (
        classify_regime(
            side="buy_no",
            is_bucket=True,
            raw_model_prob=0.25,
            metar_override=None,
        )
        == REGIME_NORMAL_BUCKET
    )


def test_normal_bucket_for_buy_yes_even_with_low_raw_prob():
    assert (
        classify_regime(
            side="buy_yes",
            is_bucket=True,
            raw_model_prob=0.05,
            metar_override=None,
        )
        == REGIME_NORMAL_BUCKET
    )


def test_threshold_yes_on_directional_market():
    assert (
        classify_regime(
            side="buy_yes",
            is_bucket=False,
            raw_model_prob=0.65,
            metar_override=None,
        )
        == REGIME_THRESHOLD_YES
    )


def test_threshold_no_on_directional_market():
    assert (
        classify_regime(
            side="buy_no",
            is_bucket=False,
            raw_model_prob=0.20,
            metar_override=None,
        )
        == REGIME_THRESHOLD_NO
    )


def test_normal_bucket_when_raw_prob_missing():
    assert (
        classify_regime(
            side="buy_no",
            is_bucket=True,
            raw_model_prob=None,
            metar_override=None,
        )
        == REGIME_NORMAL_BUCKET
    )


def test_unknown_for_unsupported_side():
    assert (
        classify_regime(
            side="no_trade",
            is_bucket=False,
            raw_model_prob=0.5,
            metar_override=None,
        )
        == REGIME_UNKNOWN
    )


def test_unknown_for_unsupported_side_under_metar_override():
    assert (
        classify_regime(
            side="no_trade",
            is_bucket=False,
            raw_model_prob=0.5,
            metar_override=0.85,
        )
        == REGIME_UNKNOWN
    )
