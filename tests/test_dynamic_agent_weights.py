"""Tests for US-202: Dynamic Brier-score-based agent weight auto-tuning."""

import pytest
from unittest.mock import MagicMock, patch

from weatheredge.learning.agent_weight_updater import AgentWeightUpdater, MIN_WEIGHT, MAX_WEIGHT


class MockCalibrationTracker:
    """Mock CalibrationTracker that returns configurable Brier scores."""

    def __init__(self, brier_scores: dict[str, float] | None = None):
        self._brier_scores = brier_scores or {}

    def get_agent_brier_scores(self) -> dict[str, float]:
        return self._brier_scores


class TestInverseBrierWeighting:
    """Test that lower Brier score -> higher weight."""

    def _make_updater(self):
        engine = MagicMock()
        updater = AgentWeightUpdater(engine)
        return updater

    def test_lower_brier_gets_higher_weight(self):
        updater = self._make_updater()
        current = {"AgentA": 0.50, "AgentB": 0.50}
        tracker = MockCalibrationTracker({"AgentA": 0.10, "AgentB": 0.30})

        result = updater.update_agent_weights(current, calibration_tracker=tracker)
        assert result["AgentA"] > result["AgentB"]

    def test_three_agents_ranking(self):
        updater = self._make_updater()
        current = {"Good": 0.33, "Medium": 0.34, "Bad": 0.33}
        tracker = MockCalibrationTracker({"Good": 0.05, "Medium": 0.15, "Bad": 0.40})

        result = updater.update_agent_weights(current, calibration_tracker=tracker)
        assert result["Good"] > result["Medium"] > result["Bad"]

    def test_weights_sum_to_one(self):
        updater = self._make_updater()
        current = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        tracker = MockCalibrationTracker({"A": 0.10, "B": 0.15, "C": 0.20, "D": 0.25})

        result = updater.update_agent_weights(current, calibration_tracker=tracker)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)

    def test_bounds_enforced(self):
        updater = self._make_updater()
        # Extreme disparity: one agent very accurate, others poor
        current = {"Star": 0.25, "Bad1": 0.25, "Bad2": 0.25, "Bad3": 0.25}
        tracker = MockCalibrationTracker({"Star": 0.01, "Bad1": 0.49, "Bad2": 0.49, "Bad3": 0.49})

        result = updater.update_agent_weights(current, calibration_tracker=tracker)
        for w in result.values():
            assert w >= MIN_WEIGHT - 1e-9
            assert w <= MAX_WEIGHT + 1e-9

    def test_zero_brier_uses_epsilon(self):
        """Brier = 0 (perfect) shouldn't cause division by zero."""
        updater = self._make_updater()
        current = {"Perfect": 0.50, "OK": 0.50}
        tracker = MockCalibrationTracker({"Perfect": 0.0, "OK": 0.20})

        result = updater.update_agent_weights(current, calibration_tracker=tracker)
        assert result["Perfect"] > result["OK"]
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)


class TestFallbackToPnL:
    """Test PnL-based fallback when no calibration data."""

    def _make_updater(self):
        engine = MagicMock()
        updater = AgentWeightUpdater(engine)
        return updater

    def test_no_calibration_tracker(self):
        updater = self._make_updater()
        current = {"A": 0.50, "B": 0.50}

        # Mock DB to return no rows (PnL fallback returns current weights)
        with patch("weatheredge.learning.agent_weight_updater.get_session") as mock_gs:
            mock_session = MagicMock()
            mock_session.execute.return_value.all.return_value = []
            mock_gs.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_gs.return_value.__exit__ = MagicMock(return_value=False)

            result = updater.update_agent_weights(current, calibration_tracker=None)
        assert result == current

    def test_empty_brier_scores_falls_back(self):
        updater = self._make_updater()
        current = {"A": 0.50, "B": 0.50}
        tracker = MockCalibrationTracker({})  # Empty = insufficient data

        with patch("weatheredge.learning.agent_weight_updater.get_session") as mock_gs:
            mock_session = MagicMock()
            mock_session.execute.return_value.all.return_value = []
            mock_gs.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_gs.return_value.__exit__ = MagicMock(return_value=False)

            result = updater.update_agent_weights(current, calibration_tracker=tracker)
        assert result == current

    def test_brier_exception_falls_back(self):
        """If get_agent_brier_scores raises, should fall back to PnL."""
        updater = self._make_updater()
        current = {"A": 0.50, "B": 0.50}
        tracker = MagicMock()
        tracker.get_agent_brier_scores.side_effect = RuntimeError("DB error")

        with patch("weatheredge.learning.agent_weight_updater.get_session") as mock_gs:
            mock_session = MagicMock()
            mock_session.execute.return_value.all.return_value = []
            mock_gs.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_gs.return_value.__exit__ = MagicMock(return_value=False)

            result = updater.update_agent_weights(current, calibration_tracker=tracker)
        assert result == current

    def test_partial_brier_data_uses_available(self):
        """If only some agents have Brier, use those and keep others at current."""
        updater = self._make_updater()
        current = {"A": 0.40, "B": 0.30, "C": 0.30}
        # Only A and B have Brier data
        tracker = MockCalibrationTracker({"A": 0.10, "B": 0.30})

        result = updater.update_agent_weights(current, calibration_tracker=tracker)
        assert result["A"] > result["B"]  # A more accurate
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)


class TestGetAgentBrierScores:
    """Test CalibrationTracker.get_agent_brier_scores()."""

    def test_returns_empty_when_no_data(self):
        from weatheredge.learning.calibration import CalibrationTracker
        engine = MagicMock()
        tracker = CalibrationTracker(engine)

        with patch("weatheredge.learning.calibration.get_session") as mock_gs:
            mock_session = MagicMock()
            mock_session.execute.return_value.all.return_value = []
            mock_gs.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_gs.return_value.__exit__ = MagicMock(return_value=False)

            result = tracker.get_agent_brier_scores()
        assert result == {}

    def test_returns_empty_when_insufficient_records(self):
        from weatheredge.learning.calibration import CalibrationTracker
        engine = MagicMock()
        tracker = CalibrationTracker(engine)

        # Agent has only 10 records (< 20 threshold)
        mock_row = MagicMock()
        mock_row.agent_name = "Conservative"
        mock_row.avg_brier = 0.15
        mock_row.count = 10

        with patch("weatheredge.learning.calibration.get_session") as mock_gs:
            mock_session = MagicMock()
            mock_session.execute.return_value.all.return_value = [mock_row]
            mock_gs.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_gs.return_value.__exit__ = MagicMock(return_value=False)

            result = tracker.get_agent_brier_scores()
        assert result == {}

    def test_returns_scores_with_sufficient_data(self):
        from weatheredge.learning.calibration import CalibrationTracker
        engine = MagicMock()
        tracker = CalibrationTracker(engine)

        rows = []
        for name, brier, count in [("Conservative", 0.10, 50), ("Aggressive", 0.20, 30)]:
            row = MagicMock()
            row.agent_name = name
            row.avg_brier = brier
            row.count = count
            rows.append(row)

        with patch("weatheredge.learning.calibration.get_session") as mock_gs:
            mock_session = MagicMock()
            mock_session.execute.return_value.all.return_value = rows
            mock_gs.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_gs.return_value.__exit__ = MagicMock(return_value=False)

            result = tracker.get_agent_brier_scores()
        assert result == {"Conservative": 0.10, "Aggressive": 0.20}


class TestUpdateCountTracking:
    """Test that update count is tracked after Brier-based update."""

    def test_trade_count_updated_after_brier_weights(self):
        engine = MagicMock()
        updater = AgentWeightUpdater(engine)
        updater._trade_count = 100

        current = {"A": 0.50, "B": 0.50}
        tracker = MockCalibrationTracker({"A": 0.10, "B": 0.20})

        updater.update_agent_weights(current, calibration_tracker=tracker)
        assert updater._last_update_count == 100
