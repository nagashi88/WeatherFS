from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from weatheredge.db.models import Trade
from weatheredge.db.database import get_session
from weatheredge.llm import get_llm_client, chat_completion

log = logging.getLogger(__name__)


@dataclass
class LLMErrorReport:
    findings: list[str] = field(default_factory=list)
    suggested_adjustments: list[str] = field(default_factory=list)
    city_biases: dict[str, str] = field(default_factory=dict)
    raw_response: str = ""


@dataclass
class ErrorReport:
    model_errors: int = 0         # Wrong probability estimate
    timing_errors: int = 0        # Resolved before forecast updated
    edge_compression: int = 0     # Edge disappeared by resolution
    adverse_selection: int = 0    # VPIN-related loss
    total_losses: int = 0
    worst_trades: list = field(default_factory=list)  # Top 3 by abs(pnl)


class ErrorAnalyzer:
    """Analyzes losing trades to identify error patterns. Runs weekly."""

    def __init__(self, engine):
        self.engine = engine

    def analyze_errors(self, days: int = 7) -> ErrorReport:
        """Examine losing resolved trades from the last N days and categorize error types."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        with get_session(self.engine) as session:
            losses = session.execute(
                select(Trade).where(
                    Trade.pnl.is_not(None),
                    Trade.pnl < 0.0,
                    Trade.status.notin_(["open", "cancelled", "abandoned"]),
                )
            ).scalars().all()

            # Detach from session for processing.
            # Use resolution time when available (otherwise fallback to created_at)
            # so long-held trades are analysed in the week they actually lost.
            loss_data = []
            for t in losses:
                ts = t.resolved_at or t.created_at
                if ts is None:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    continue
                loss_data.append({
                    "id": t.id,
                    "market_id": t.market_id,
                    "side": t.side,
                    "status": t.status,
                    "model_probability": t.model_probability,
                    "edge_at_entry": t.edge_at_entry,
                    "cost": t.cost,
                    "pnl": t.pnl,
                    "agent_name": t.agent_name,
                    "resolved_at": t.resolved_at,
                    "created_at": t.created_at,
                })

        if not loss_data:
            return ErrorReport()

        report = ErrorReport(total_losses=len(loss_data))

        for trade in loss_data:
            model_prob_yes = float(trade.get("model_probability") or 0.0)
            side = trade.get("side") or "buy_yes"
            # Use position-side confidence so buy_no trades are scored correctly.
            trade_side_prob = (1.0 - model_prob_yes) if side == "buy_no" else model_prob_yes
            edge = abs(float(trade.get("edge_at_entry") or 0.0))

            # Categorize: high probability loss = model error
            if trade_side_prob > 0.70:
                report.model_errors += 1
            # Low edge loss = edge compression
            elif edge < 0.10:
                report.edge_compression += 1
            # Otherwise: timing or general
            else:
                report.timing_errors += 1

        # Sort by worst P&L
        sorted_losses = sorted(loss_data, key=lambda t: t.get("pnl", 0) or 0)
        report.worst_trades = sorted_losses[:3]

        log.info(
            "Error analysis: %d losses — model:%d, timing:%d, edge_compression:%d, adverse:%d",
            report.total_losses, report.model_errors, report.timing_errors,
            report.edge_compression, report.adverse_selection,
        )

        return report

    def analyze_with_llm(
        self,
        losing_trades: list[dict],
        api_key: str,
        model: str = "anthropic/claude-sonnet-4",
    ) -> LLMErrorReport:
        """Call LLM API via OpenRouter to identify patterns in losing trades.

        Returns an empty LLMErrorReport on any API error.
        """
        if not losing_trades:
            return LLMErrorReport()

        trade_lines = []
        for t in losing_trades:
            city = t.get("city", "unknown")
            forecast = t.get("forecast", "unknown")
            probability = t.get("probability", t.get("model_probability", "?"))
            actual = t.get("actual", "unknown")
            pnl = t.get("pnl", 0)
            trade_lines.append(
                f"  city={city}, forecast={forecast}, probability={probability}, "
                f"actual={actual}, pnl={pnl}"
            )

        trade_block = "\n".join(trade_lines)
        prompt = (
            "You are a trading analyst. Analyze the following losing weather-market trades "
            "and identify patterns.\n\n"
            f"Losing trades:\n{trade_block}\n\n"
            "Respond with a JSON object containing:\n"
            '- "findings": list of strings describing identified error patterns\n'
            '- "suggested_adjustments": list of strings with recommended changes\n'
            '- "city_biases": dict mapping city names to bias descriptions\n\n'
            "Return only valid JSON, no extra text."
        )

        client = get_llm_client(api_key)
        raw = chat_completion(client, model, prompt, max_tokens=1024)
        if raw is None:
            log.warning("LLM error analysis failed: no response")
            return LLMErrorReport()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("LLM response was not valid JSON: %s", raw[:200])
            return LLMErrorReport(raw_response=raw)

        return LLMErrorReport(
            findings=data.get("findings", []),
            suggested_adjustments=data.get("suggested_adjustments", []),
            city_biases=data.get("city_biases", {}),
            raw_response=raw,
        )
