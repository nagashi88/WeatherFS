"""Tests for ModelWeightUpdater and AgentWeightUpdater."""

from __future__ import annotations

import pytest
from weatheredge.db.database import init_db, make_engine
from weatheredge.db.models import AgentPerformance, CalibrationRecord
from weatheredge.learning.weight_updater import ModelWeightUpdater
from weatheredge.learning.agent_weight_updater import AgentWeightUpdater


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def model_updater(engine):
    return ModelWeightUpdater(engine)


@pytest.fixture()
def agent_updater(engine):
    return AgentWeightUpdater(engine)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_calibration(engine, model: str, station: str, lead_time_hours: int, brier: float):
    """Insert a single CalibrationRecord row directly."""
    from weatheredge.db.database import get_session
    with get_session(engine) as session:
        session.add(CalibrationRecord(
            model=model,
            station=station,
            lead_time_hours=lead_time_hours,
            predicted_probability=0.5,
            actual_outcome=1,
            brier_score=brier,
        ))


def _insert_agent_perf(engine, agent_name: str, total_pnl: float, period: str = "all_time"):
    """Insert a single AgentPerformance row directly."""
    from weatheredge.db.database import get_session
    with get_session(engine) as session:
        session.add(AgentPerformance(
            agent_name=agent_name,
            period=period,
            total_pnl=total_pnl,
        ))


# ---------------------------------------------------------------------------
# ModelWeightUpdater tests
# ---------------------------------------------------------------------------

def test_model_lower_brier_gets_higher_weight(engine, model_updater):
    """Model A (brier=0.1) should have higher weight than Model B (brier=0.4)."""
    _insert_calibration(engine, "modelA", "KNYC", 24, 0.1)
    _insert_calibration(engine, "modelB", "KNYC", 24, 0.4)

    model_updater.update_model_weights()

    w_a = model_updater.get_model_weight("KNYC", "modelA", 1)
    w_b = model_updater.get_model_weight("KNYC", "modelB", 1)
    assert w_a > w_b, f"Expected modelA weight ({w_a}) > modelB weight ({w_b})"


def test_model_weights_sum_to_one(engine, model_updater):
    """Weights for all models in a (station, lead_day) group must sum to 1.0."""
    _insert_calibration(engine, "modelA", "KNYC", 24, 0.1)
    _insert_calibration(engine, "modelB", "KNYC", 24, 0.3)
    _insert_calibration(engine, "modelC", "KNYC", 24, 0.2)

    model_updater.update_model_weights()

    total = sum(
        model_updater.get_model_weight("KNYC", m, 1)
        for m in ("modelA", "modelB", "modelC")
    )
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"


def test_model_should_update_at_interval(model_updater):
    """should_update returns False at 49 trades and True at 50 trades."""
    for _ in range(49):
        model_updater.record_trade()
    assert not model_updater.should_update(interval=50)

    model_updater.record_trade()  # 50th trade
    assert model_updater.should_update(interval=50)


def test_model_default_weight_when_no_data(model_updater):
    """With no calibration data, get_model_weight returns 1.0 (equal weight default)."""
    assert model_updater.get_model_weight("KNYC", "gfs", 1) == 1.0


def test_model_update_resets_should_update(engine, model_updater):
    """After update_model_weights(), should_update returns False until interval trades more."""
    for _ in range(50):
        model_updater.record_trade()
    assert model_updater.should_update(interval=50)

    model_updater.update_model_weights()
    assert not model_updater.should_update(interval=50)


# ---------------------------------------------------------------------------
# AgentWeightUpdater tests
# ---------------------------------------------------------------------------

def test_agent_better_pnl_gets_higher_weight(engine, agent_updater):
    """Agent with PnL=0.20 should receive higher weight than agent with PnL=0.05."""
    _insert_agent_perf(engine, "Conservative", 0.20)
    _insert_agent_perf(engine, "Aggressive", 0.05)

    current = {"Conservative": 0.50, "Aggressive": 0.50}
    new_weights = agent_updater.update_agent_weights(current)

    assert new_weights["Conservative"] > new_weights["Aggressive"], (
        f"Expected Conservative ({new_weights['Conservative']}) > "
        f"Aggressive ({new_weights['Aggressive']})"
    )


def test_agent_weight_floor_ceiling(engine, agent_updater):
    """No weight should be below MIN_WEIGHT (0.05) or above MAX_WEIGHT (0.50)."""
    # Extreme PnL difference to stress-test clamping
    _insert_agent_perf(engine, "Conservative", 100.0)
    _insert_agent_perf(engine, "Aggressive", 0.001)
    _insert_agent_perf(engine, "MeanReversion", 0.001)
    _insert_agent_perf(engine, "Qualitative", 0.001)

    current = {"Conservative": 0.25, "Aggressive": 0.25, "MeanReversion": 0.25, "Qualitative": 0.25}
    new_weights = agent_updater.update_agent_weights(current)

    from weatheredge.learning.agent_weight_updater import MIN_WEIGHT, MAX_WEIGHT
    for name, w in new_weights.items():
        assert w >= MIN_WEIGHT - 1e-9, f"{name} weight {w} below MIN_WEIGHT {MIN_WEIGHT}"
        assert w <= MAX_WEIGHT + 1e-9, f"{name} weight {w} above MAX_WEIGHT {MAX_WEIGHT}"


def test_agent_weights_sum_to_one(engine, agent_updater):
    """Updated agent weights must always sum to 1.0."""
    _insert_agent_perf(engine, "Conservative", 0.10)
    _insert_agent_perf(engine, "Aggressive", 0.05)
    _insert_agent_perf(engine, "MeanReversion", 0.20)
    _insert_agent_perf(engine, "Qualitative", -0.03)

    current = {"Conservative": 0.30, "Aggressive": 0.30, "MeanReversion": 0.20, "Qualitative": 0.20}
    new_weights = agent_updater.update_agent_weights(current)

    total = sum(new_weights.values())
    assert abs(total - 1.0) < 1e-9, f"Agent weights sum to {total}, expected 1.0"


def test_agent_update_at_interval(agent_updater):
    """should_update returns False at 99 trades and True at 100 trades."""
    for _ in range(99):
        agent_updater.record_trade()
    assert not agent_updater.should_update(interval=100)

    agent_updater.record_trade()  # 100th trade
    assert agent_updater.should_update(interval=100)


def test_agent_no_data_returns_current(agent_updater):
    """Empty DB: update_agent_weights returns the existing weights unchanged."""
    current = {"Conservative": 0.30, "Aggressive": 0.30, "MeanReversion": 0.20, "Qualitative": 0.20}
    result = agent_updater.update_agent_weights(current)
    assert result == current


def test_agent_unknown_agents_ignored(engine, agent_updater):
    """Agents in DB but not in current_weights are ignored; weights still sum to 1.0."""
    _insert_agent_perf(engine, "UnknownAgent", 999.0)
    _insert_agent_perf(engine, "Conservative", 0.10)

    current = {"Conservative": 0.50, "Aggressive": 0.50}
    new_weights = agent_updater.update_agent_weights(current)

    assert "UnknownAgent" not in new_weights
    assert abs(sum(new_weights.values()) - 1.0) < 1e-9
