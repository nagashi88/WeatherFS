"""Telegram notification layer for WeatherEdge bot."""

from __future__ import annotations

import logging

from weatheredge.config import Settings

logger = logging.getLogger(__name__)

ALERT_TEMPLATES = {
    "trade_executed": "TRADE: {side} {amount} shares @ ${price} on {market}. Edge: {edge}%",
    "market_resolved": "RESOLVED: {market} -> {outcome}. P&L: ${pnl}",
    "daily_summary": "Daily: {trades} trades, P&L: ${pnl}, Win rate: {wr}%",
    "circuit_breaker": "CIRCUIT BREAKER: {reason}. Trading paused.",
    "kill_switch": "KILL SWITCH: {reason}. ALL TRADING HALTED. Manual restart required.",
    "error": "ERROR: {error_message}",
    "degraded_mode": "DEGRADED MODE: {mode}. {details}",
}


class TelegramBot:
    """Sends Telegram notifications for trade events and system alerts."""

    def __init__(self, settings: Settings) -> None:
        self._token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id

    async def send_alert(self, message: str) -> None:
        """Send a message to the configured chat_id. Logs and swallows errors."""
        if not self._token:
            logger.warning("Telegram bot token not configured; skipping alert.")
            return
        if not self._chat_id:
            logger.warning("Telegram chat_id not configured; skipping alert.")
            return
        try:
            from telegram import Bot  # python-telegram-bot

            bot = Bot(token=self._token)
            await bot.send_message(chat_id=self._chat_id, text=message)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to send Telegram alert: %s", exc)

    # ------------------------------------------------------------------
    # Formatters
    # ------------------------------------------------------------------

    def format_trade_alert(self, trade_data: dict) -> str:
        """Format a trade-executed alert from trade_data dict."""
        return ALERT_TEMPLATES["trade_executed"].format(
            side=trade_data.get("side", ""),
            amount=trade_data.get("amount", ""),
            price=trade_data.get("price", ""),
            market=trade_data.get("market", ""),
            edge=trade_data.get("edge", ""),
        )

    def format_resolution_alert(self, resolution_data: dict) -> str:
        """Format a market-resolved alert from resolution_data dict."""
        return ALERT_TEMPLATES["market_resolved"].format(
            market=resolution_data.get("market", ""),
            outcome=resolution_data.get("outcome", ""),
            pnl=resolution_data.get("pnl", ""),
        )

    def format_daily_summary(self, summary_data: dict) -> str:
        """Format a daily P&L summary alert."""
        return ALERT_TEMPLATES["daily_summary"].format(
            trades=summary_data.get("trades", ""),
            pnl=summary_data.get("pnl", ""),
            wr=summary_data.get("wr", ""),
        )

    def format_circuit_breaker_alert(self, reason: str) -> str:
        """Format a circuit-breaker alert."""
        return ALERT_TEMPLATES["circuit_breaker"].format(reason=reason)

    def format_kill_switch_alert(self, reason: str) -> str:
        """Format a kill-switch alert."""
        return ALERT_TEMPLATES["kill_switch"].format(reason=reason)

    def format_error_alert(self, error: str) -> str:
        """Format an error alert."""
        return ALERT_TEMPLATES["error"].format(error_message=error)

    def format_degraded_mode_alert(self, mode: str, details: str) -> str:
        """Format a degraded-mode alert (Addendum Gap 12)."""
        return ALERT_TEMPLATES["degraded_mode"].format(mode=mode, details=details)
