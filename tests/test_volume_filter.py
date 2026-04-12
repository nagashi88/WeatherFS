"""Tests for US-VF — volume/spread filter."""

from __future__ import annotations

import pytest

import weatheredge.config as _cfg
from weatheredge.main import should_evaluate_market


class TestShouldEvaluateMarket:
    def test_low_volume_rejected(self):
        ok, reason = should_evaluate_market(1999, 0.05)
        assert ok is False
        assert reason == "low_volume"

    def test_volume_at_threshold_accepted(self):
        ok, reason = should_evaluate_market(2000, 0.05)
        assert ok is True
        assert reason == ""

    def test_wide_spread_rejected(self):
        ok, reason = should_evaluate_market(3000, 0.081)
        assert ok is False
        assert reason == "wide_spread"

    def test_spread_at_threshold_accepted(self):
        ok, reason = should_evaluate_market(3000, 0.08)
        assert ok is True
        assert reason == ""

    def test_both_ok(self):
        ok, reason = should_evaluate_market(5000, 0.04)
        assert ok is True
        assert reason == ""

    def test_volume_checked_before_spread(self):
        """When both fail, low_volume is returned first."""
        ok, reason = should_evaluate_market(100, 0.99)
        assert ok is False
        assert reason == "low_volume"
