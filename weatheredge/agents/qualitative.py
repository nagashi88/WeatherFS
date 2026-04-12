from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from weatheredge.agents.base import Signal, MarketOpportunity
from weatheredge.config import Settings
from weatheredge.llm import get_llm_client, chat_completion

log = logging.getLogger(__name__)

@dataclass
class WeatherContext:
    forecast: float
    ensemble_spread: float | None
    recent_bias: float | None
    current_conditions: str
    notable_factors: str

class QualitativeAgent:
    """LLM-powered qualitative factor agent. Evaluates factors quantitative models miss."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = get_llm_client(self.settings.openrouter_api_key)
        return self._client

    def analyze(self, opportunity: MarketOpportunity, context: WeatherContext | None = None) -> Signal:
        """
        Ask Claude to evaluate qualitative factors that quantitative models miss.
        Returns Signal with confidence as adjustment factor clamped to [-0.10, +0.10].
        """
        if not self.settings.enable_qualitative_agent:
            return Signal(market_id=opportunity.market_id, side="no_trade", confidence=0.0,
                         agent_name="Qualitative", reasoning="Agent disabled")

        if not self.settings.openrouter_api_key:
            log.warning("Qualitative agent: no API key configured")
            return Signal(market_id=opportunity.market_id, side="no_trade", confidence=0.0,
                         agent_name="Qualitative", reasoning="No API key")

        prompt = self._build_prompt(opportunity, context)

        try:
            client = self._get_client()
            text = chat_completion(client, self.settings.llm_model_decision, prompt, max_tokens=500)
            if text is None:
                return Signal(market_id=opportunity.market_id, side="no_trade", confidence=0.0,
                             agent_name="Qualitative", reasoning="API error: no response")
            return self._parse_response(opportunity.market_id, text)
        except Exception as e:
            log.warning("Qualitative agent API error: %s", e)
            return Signal(market_id=opportunity.market_id, side="no_trade", confidence=0.0,
                         agent_name="Qualitative", reasoning=f"API error: {e}")

    def _build_prompt(self, opportunity: MarketOpportunity, context: WeatherContext | None) -> str:
        ctx = context or WeatherContext(
            forecast=opportunity.forecast_temp,
            ensemble_spread=None, recent_bias=None,
            current_conditions="unknown", notable_factors="none"
        )
        return f"""You are a weather market analyst. Evaluate whether quantitative forecast models are likely to be wrong for this specific situation.

City: {opportunity.city}
Station: {opportunity.station}
Date: today
Current forecast: {ctx.forecast}°F
Model ensemble spread: {ctx.ensemble_spread or 'N/A'}
Recent model bias (last 7 days): {ctx.recent_bias or 'N/A'}
Current conditions: {ctx.current_conditions}
Notable factors: {ctx.notable_factors}

Market threshold: {opportunity.threshold}°F ({opportunity.direction})
Current model probability: {opportunity.model_probability:.2f}
Current market price: {opportunity.market_price:.2f}

Consider:
1. Are there microclimate factors this station is known for?
2. Is there unusual weather (smoke, inversion, unusual wind)?
3. Has the model been systematically biased recently?
4. Are there timing issues (DST/LST, observation schedule)?

Return ONLY a JSON object with:
- "adjustment": float between -0.10 and 0.10 (negative = model is too warm)
- "confidence": float between 0.0 and 1.0
- "reasoning": string explaining your assessment"""

    def _parse_response(self, market_id: str, text: str) -> Signal:
        """Parse LLM response into a Signal."""
        try:
            # Try to extract JSON from response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
            else:
                data = json.loads(text)

            adjustment = max(-0.10, min(0.10, float(data.get("adjustment", 0))))
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0))))
            reasoning = str(data.get("reasoning", ""))

            # Determine side from adjustment
            if abs(adjustment) < 0.01:
                side = "no_trade"
                confidence = 0.0
            elif adjustment > 0:
                side = "buy_yes"  # Model is too cold, actual likely warmer
            else:
                side = "buy_no"   # Model is too warm, actual likely cooler

            return Signal(
                market_id=market_id, side=side,
                confidence=abs(adjustment),  # Use adjustment magnitude as confidence
                agent_name="Qualitative", reasoning=reasoning
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            log.warning("Failed to parse qualitative response: %s", e)
            return Signal(market_id=market_id, side="no_trade", confidence=0.0,
                         agent_name="Qualitative", reasoning=f"Parse error: {e}")
