"""Tests for ErrorAnalyzer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from weatheredge.db.database import init_db, make_engine, get_session
from weatheredge.db.models import Trade
from weatheredge.learning.error_analysis import ErrorAnalyzer, ErrorReport, LLMErrorReport


@pytest.fixture()
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = make_engine("sqlite:///:memory:")
    init_db(engine=eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def analyzer(engine):
    return ErrorAnalyzer(engine)


def _make_trade(
    status: str = "lost",
    model_probability: float = 0.5,
    edge_at_entry: float = 0.15,
    pnl: float = -10.0,
    agent_name: str = "test_agent",
    side: str = "buy_yes",
    created_at: datetime | None = None,
    resolved_at: datetime | None = None,
) -> Trade:
    """Helper to build a Trade row."""
    return Trade(
        platform="polymarket",
        mode="paper",
        side=side,
        price=0.5,
        amount=20.0,
        cost=10.0,
        status=status,
        model_probability=model_probability,
        edge_at_entry=edge_at_entry,
        pnl=pnl,
        agent_name=agent_name,
        created_at=created_at or datetime.now(timezone.utc),
        resolved_at=resolved_at,
    )


# ---------------------------------------------------------------------------
# test_model_error_categorized
# ---------------------------------------------------------------------------

def test_model_error_categorized(engine, analyzer):
    """High-probability loss (0.85) → model_errors incremented."""
    with get_session(engine) as session:
        session.add(_make_trade(model_probability=0.85, edge_at_entry=0.15))

    report = analyzer.analyze_errors(days=30)

    assert report.model_errors == 1
    assert report.timing_errors == 0
    assert report.edge_compression == 0
    assert report.total_losses == 1


def test_model_error_categorized_for_buy_no_side(engine, analyzer):
    """buy_no loss should use NO-side confidence (1 - model_probability)."""
    with get_session(engine) as session:
        session.add(_make_trade(
            status="lost",
            side="buy_no",
            model_probability=0.20,  # NO-side confidence = 0.80
            edge_at_entry=0.15,
            pnl=-8.0,
        ))

    report = analyzer.analyze_errors(days=30)

    assert report.total_losses == 1
    assert report.model_errors == 1
    assert report.timing_errors == 0
    assert report.edge_compression == 0


# ---------------------------------------------------------------------------
# test_edge_compression_categorized
# ---------------------------------------------------------------------------

def test_edge_compression_categorized(engine, analyzer):
    """Low-edge loss (0.05) → edge_compression incremented."""
    with get_session(engine) as session:
        session.add(_make_trade(model_probability=0.55, edge_at_entry=0.05))

    report = analyzer.analyze_errors(days=30)

    assert report.edge_compression == 1
    assert report.model_errors == 0
    assert report.timing_errors == 0
    assert report.total_losses == 1


def test_early_exit_negative_pnl_included(engine, analyzer):
    """Negative early-exit trades (edge_stop/capital_protect/...) must be analysed."""
    with get_session(engine) as session:
        session.add(_make_trade(
            status="edge_stop",
            model_probability=0.55,
            edge_at_entry=0.05,
            pnl=-3.0,
        ))

    report = analyzer.analyze_errors(days=30)

    assert report.total_losses == 1
    assert report.edge_compression == 1


def test_resolved_at_used_for_recency_filter(engine, analyzer):
    """Old created_at but recent resolved_at should still be in current analysis window."""
    now = datetime.now(timezone.utc)
    old_created = now - timedelta(days=40)
    with get_session(engine) as session:
        session.add(_make_trade(
            status="lost",
            model_probability=0.75,
            edge_at_entry=0.15,
            pnl=-4.0,
            created_at=old_created,
            resolved_at=now,
        ))

    report = analyzer.analyze_errors(days=7)
    assert report.total_losses == 1
    assert report.model_errors == 1


# ---------------------------------------------------------------------------
# test_worst_trades_sorted
# ---------------------------------------------------------------------------

def test_worst_trades_sorted(engine, analyzer):
    """5 losses → worst_trades contains top 3 by most negative pnl."""
    pnls = [-5.0, -50.0, -20.0, -100.0, -1.0]
    with get_session(engine) as session:
        for p in pnls:
            session.add(_make_trade(pnl=p))

    report = analyzer.analyze_errors(days=30)

    assert len(report.worst_trades) == 3
    worst_pnls = [t["pnl"] for t in report.worst_trades]
    # Should be the 3 most negative: -100, -50, -20
    assert worst_pnls == [-100.0, -50.0, -20.0]


# ---------------------------------------------------------------------------
# test_no_losses_returns_empty
# ---------------------------------------------------------------------------

def test_no_losses_returns_empty(engine, analyzer):
    """No losing trades → all fields zero, worst_trades empty list."""
    # Insert a won trade — should not appear in results
    with get_session(engine) as session:
        session.add(_make_trade(status="won", pnl=15.0))

    report = analyzer.analyze_errors(days=30)

    assert report.total_losses == 0
    assert report.model_errors == 0
    assert report.timing_errors == 0
    assert report.edge_compression == 0
    assert report.adverse_selection == 0
    assert report.worst_trades == []


# ---------------------------------------------------------------------------
# test_report_contains_all_categories
# ---------------------------------------------------------------------------

def test_report_contains_all_categories():
    """ErrorReport dataclass has all required fields with correct defaults."""
    report = ErrorReport()

    assert hasattr(report, "model_errors")
    assert hasattr(report, "timing_errors")
    assert hasattr(report, "edge_compression")
    assert hasattr(report, "adverse_selection")
    assert hasattr(report, "total_losses")
    assert hasattr(report, "worst_trades")

    assert report.model_errors == 0
    assert report.timing_errors == 0
    assert report.edge_compression == 0
    assert report.adverse_selection == 0
    assert report.total_losses == 0
    assert report.worst_trades == []


# ---------------------------------------------------------------------------
# test_llm_prompt_includes_trade_data
# ---------------------------------------------------------------------------

def test_llm_prompt_includes_trade_data(analyzer):
    """analyze_with_llm constructs a prompt containing city, forecast, and pnl."""
    trades = [
        {"city": "NYC", "forecast": "rain", "probability": 0.72, "actual": "no_rain", "pnl": -15.0},
        {"city": "Miami", "forecast": "sunny", "probability": 0.65, "actual": "rain", "pnl": -8.0},
    ]

    captured_prompt = {}

    def fake_chat_completion(client, model, prompt, max_tokens=500):
        captured_prompt["prompt"] = prompt
        return '{"findings": [], "suggested_adjustments": [], "city_biases": {}}'

    with patch("weatheredge.learning.error_analysis.chat_completion", side_effect=fake_chat_completion):
        with patch("weatheredge.learning.error_analysis.get_llm_client", return_value=MagicMock()):
            analyzer.analyze_with_llm(trades, api_key="test-key")

    prompt_text = captured_prompt["prompt"]
    assert "NYC" in prompt_text
    assert "rain" in prompt_text
    assert "-15.0" in prompt_text


# ---------------------------------------------------------------------------
# test_llm_returns_structured_report
# ---------------------------------------------------------------------------

def test_llm_returns_structured_report(analyzer):
    """When API returns JSON findings, LLMErrorReport is populated correctly."""
    trades = [
        {"city": "Chicago", "forecast": "snow", "probability": 0.80, "actual": "clear", "pnl": -25.0},
    ]
    json_response = (
        '{"findings": ["Overconfident on snow forecasts"], '
        '"suggested_adjustments": ["Reduce Chicago snow probability by 10%"], '
        '"city_biases": {"Chicago": "overestimates winter precipitation"}}'
    )

    with patch("weatheredge.learning.error_analysis.chat_completion", return_value=json_response):
        with patch("weatheredge.learning.error_analysis.get_llm_client", return_value=MagicMock()):
            report = analyzer.analyze_with_llm(trades, api_key="test-key")

    assert isinstance(report, LLMErrorReport)
    assert report.findings == ["Overconfident on snow forecasts"]
    assert report.suggested_adjustments == ["Reduce Chicago snow probability by 10%"]
    assert report.city_biases == {"Chicago": "overestimates winter precipitation"}
    assert json_response in report.raw_response


# ---------------------------------------------------------------------------
# test_llm_api_failure_returns_empty
# ---------------------------------------------------------------------------

def test_llm_api_failure_returns_empty(analyzer):
    """When the API returns None (error), an empty LLMErrorReport is returned."""
    trades = [
        {"city": "Denver", "forecast": "hail", "probability": 0.60, "actual": "clear", "pnl": -5.0},
    ]

    with patch("weatheredge.learning.error_analysis.chat_completion", return_value=None):
        with patch("weatheredge.learning.error_analysis.get_llm_client", return_value=MagicMock()):
            report = analyzer.analyze_with_llm(trades, api_key="bad-key")

    assert isinstance(report, LLMErrorReport)
    assert report.findings == []
    assert report.suggested_adjustments == []
    assert report.city_biases == {}
    assert report.raw_response == ""
