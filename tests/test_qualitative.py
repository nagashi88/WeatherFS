"""Tests for the QualitativeAgent (LLM-powered)."""
from __future__ import annotations
import json
import pytest
from unittest.mock import MagicMock, patch
from weatheredge.agents.base import Signal, MarketOpportunity
from weatheredge.agents.qualitative import QualitativeAgent, WeatherContext
from weatheredge.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_settings(**kwargs):
    s = MagicMock(spec=Settings)
    s.enable_qualitative_agent = kwargs.get("enable_qualitative_agent", True)
    s.openrouter_api_key = kwargs.get("openrouter_api_key", "test-key-123")
    s.llm_model_decision = kwargs.get("llm_model_decision", "anthropic/claude-sonnet-4-20250514")
    return s


def make_opportunity(**kwargs):
    defaults = dict(
        market_id="MKT-1", platform="kalshi", city="NYC", station="KNYC",
        threshold=80.0, direction="above", market_price=0.40,
        model_probability=0.70, net_edge=0.25, forecast_temp=85.0,
        rmse=2.5, lead_time_days=1,
    )
    defaults.update(kwargs)
    return MarketOpportunity(**defaults)


def make_api_response(text: str):
    """Build a mock OpenAI-compatible chat completion response object."""
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def valid_json_response(adjustment=0.05, confidence=0.8, reasoning="test reasoning"):
    return json.dumps({"adjustment": adjustment, "confidence": confidence, "reasoning": reasoning})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_prompt_construction():
    """Verify the prompt contains expected fields."""
    agent = QualitativeAgent(make_settings())
    opp = make_opportunity(city="Chicago", station="KORD", threshold=75.0, direction="below",
                           forecast_temp=72.0, model_probability=0.60, market_price=0.35)
    ctx = WeatherContext(forecast=72.0, ensemble_spread=1.5, recent_bias=-0.3,
                         current_conditions="clear", notable_factors="lake effect")

    prompt = agent._build_prompt(opp, ctx)

    assert "Chicago" in prompt
    assert "KORD" in prompt
    assert "75.0" in prompt
    assert "72.0" in prompt


def test_adjustment_clamped():
    """API returns adjustment=0.50 -> clamped to 0.10."""
    agent = QualitativeAgent(make_settings())
    opp = make_opportunity()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = make_api_response(
        valid_json_response(adjustment=0.50, confidence=0.9)
    )
    agent._client = mock_client

    signal = agent.analyze(opp)

    assert signal.side == "buy_yes"
    assert signal.confidence == pytest.approx(0.10)
    assert signal.agent_name == "Qualitative"


def test_adjustment_negative():
    """API returns adjustment=-0.05 -> side=buy_no."""
    agent = QualitativeAgent(make_settings())
    opp = make_opportunity()

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = make_api_response(
        valid_json_response(adjustment=-0.05, confidence=0.7)
    )
    agent._client = mock_client

    signal = agent.analyze(opp)

    assert signal.side == "buy_no"
    assert signal.confidence == pytest.approx(0.05)


def test_api_error_returns_no_trade():
    """Mock API raising exception -> no_trade signal."""
    agent = QualitativeAgent(make_settings())
    opp = make_opportunity()

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("connection timeout")
    agent._client = mock_client

    signal = agent.analyze(opp)

    assert signal.side == "no_trade"
    assert signal.confidence == 0.0
    assert "API error" in signal.reasoning


def test_disabled_agent_returns_no_trade():
    """settings.enable_qualitative_agent=False -> no_trade without calling API."""
    agent = QualitativeAgent(make_settings(enable_qualitative_agent=False))
    opp = make_opportunity()

    mock_client = MagicMock()
    agent._client = mock_client

    signal = agent.analyze(opp)

    assert signal.side == "no_trade"
    assert signal.confidence == 0.0
    assert "disabled" in signal.reasoning.lower()
    mock_client.chat.completions.create.assert_not_called()


def test_no_api_key_returns_no_trade():
    """Empty API key -> no_trade without calling API."""
    agent = QualitativeAgent(make_settings(openrouter_api_key=""))
    opp = make_opportunity()

    mock_client = MagicMock()
    agent._client = mock_client

    signal = agent.analyze(opp)

    assert signal.side == "no_trade"
    assert signal.confidence == 0.0
    mock_client.chat.completions.create.assert_not_called()


def test_parse_valid_json():
    """Valid JSON response -> correct Signal fields."""
    agent = QualitativeAgent(make_settings())
    market_id = "MKT-42"
    text = valid_json_response(adjustment=0.07, confidence=0.85, reasoning="lake effect cooling")

    signal = agent._parse_response(market_id, text)

    assert signal.market_id == market_id
    assert signal.side == "buy_yes"
    assert signal.confidence == pytest.approx(0.07)
    assert signal.reasoning == "lake effect cooling"
    assert signal.agent_name == "Qualitative"


def test_parse_invalid_json():
    """Non-JSON response -> no_trade signal."""
    agent = QualitativeAgent(make_settings())

    signal = agent._parse_response("MKT-1", "Sorry, I cannot answer that.")

    assert signal.side == "no_trade"
    assert signal.confidence == 0.0
    assert "Parse error" in signal.reasoning
