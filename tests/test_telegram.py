"""Tests for Telegram notification layer."""

from __future__ import annotations

import logging

import pytest

from weatheredge.config import Settings
from weatheredge.execution.paper_trader import PerformanceReport
from weatheredge.notifications.formatters import (
    format_agent_performance,
    format_competitor_alert,
    format_edge_decay_alert,
    format_kill_confirmation,
    format_morning_briefing,
    format_performance_dashboard,
    format_performance_report,
    format_positions_list,
    format_status,
    format_vpin_status,
)
from weatheredge.notifications.telegram import ALERT_TEMPLATES, TelegramBot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings(**overrides) -> Settings:
    """Build a Settings instance with test-friendly defaults (no .env required)."""
    defaults = dict(
        polymarket_private_key="",
        polymarket_proxy_address="",
        kalshi_api_key="",
        kalshi_private_key="",
        anthropic_api_key="",
        telegram_bot_token="",
        telegram_chat_id="",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_bot(**overrides) -> TelegramBot:
    return TelegramBot(make_settings(**overrides))


# ---------------------------------------------------------------------------
# format_trade_alert
# ---------------------------------------------------------------------------


def test_format_trade_alert_produces_expected_string():
    bot = make_bot()
    result = bot.format_trade_alert(
        {"side": "YES", "amount": 100, "price": "0.65", "market": "NYC-TEMP", "edge": "12.5"}
    )
    assert result == "TRADE: YES 100 shares @ $0.65 on NYC-TEMP. Edge: 12.5%"


def test_format_trade_alert_matches_template_keys():
    bot = make_bot()
    data = {"side": "NO", "amount": 50, "price": "0.40", "market": "CHI-RAIN", "edge": "8.0"}
    result = bot.format_trade_alert(data)
    assert "NO" in result
    assert "50" in result
    assert "CHI-RAIN" in result
    assert "8.0%" in result


# ---------------------------------------------------------------------------
# format_resolution_alert
# ---------------------------------------------------------------------------


def test_format_resolution_alert_win():
    bot = make_bot()
    result = bot.format_resolution_alert(
        {"market": "NYC-TEMP", "outcome": "YES", "pnl": "35.00"}
    )
    assert result == "RESOLVED: NYC-TEMP -> YES. P&L: $35.00"


def test_format_resolution_alert_loss():
    bot = make_bot()
    result = bot.format_resolution_alert(
        {"market": "MIA-RAIN", "outcome": "NO", "pnl": "-20.00"}
    )
    assert result == "RESOLVED: MIA-RAIN -> NO. P&L: $-20.00"


# ---------------------------------------------------------------------------
# format_daily_summary
# ---------------------------------------------------------------------------


def test_format_daily_summary():
    bot = make_bot()
    result = bot.format_daily_summary({"trades": 5, "pnl": "42.10", "wr": "60.0"})
    assert result == "Daily: 5 trades, P&L: $42.10, Win rate: 60.0%"


# ---------------------------------------------------------------------------
# format_circuit_breaker_alert
# ---------------------------------------------------------------------------


def test_format_circuit_breaker_alert():
    bot = make_bot()
    result = bot.format_circuit_breaker_alert("Daily loss limit hit")
    assert result == "CIRCUIT BREAKER: Daily loss limit hit. Trading paused."


# ---------------------------------------------------------------------------
# format_kill_switch_alert
# ---------------------------------------------------------------------------


def test_format_kill_switch_alert():
    bot = make_bot()
    result = bot.format_kill_switch_alert("Total drawdown exceeded -40%")
    assert (
        result
        == "KILL SWITCH: Total drawdown exceeded -40%. ALL TRADING HALTED. Manual restart required."
    )


# ---------------------------------------------------------------------------
# format_degraded_mode_alert
# ---------------------------------------------------------------------------


def test_format_degraded_mode_alert():
    bot = make_bot()
    result = bot.format_degraded_mode_alert("STALE_WEATHER", "Last update 3h ago")
    assert result == "DEGRADED MODE: STALE_WEATHER. Last update 3h ago"


# ---------------------------------------------------------------------------
# send_alert — handles missing token gracefully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_alert_missing_token_logs_warning_no_crash(caplog):
    """send_alert should log a warning and not raise when token is empty."""
    bot = make_bot(telegram_bot_token="", telegram_chat_id="12345")
    with caplog.at_level(logging.WARNING, logger="weatheredge.notifications.telegram"):
        await bot.send_alert("test message")
    assert any("token" in record.message.lower() for record in caplog.records)


@pytest.mark.asyncio
async def test_send_alert_missing_chat_id_logs_warning_no_crash(caplog):
    """send_alert should log a warning and not raise when chat_id is empty."""
    bot = make_bot(telegram_bot_token="fake-token", telegram_chat_id="")
    with caplog.at_level(logging.WARNING, logger="weatheredge.notifications.telegram"):
        await bot.send_alert("test message")
    assert any("chat_id" in record.message.lower() for record in caplog.records)


# ---------------------------------------------------------------------------
# format_performance_report
# ---------------------------------------------------------------------------


def test_format_performance_report():
    report = PerformanceReport(
        total_trades=20,
        winning_trades=12,
        losing_trades=8,
        win_rate=0.60,
        total_pnl=150.50,
        roi=0.1505,
        sharpe_ratio=1.25,
        max_drawdown=0.08,
        avg_edge_at_entry=0.135,
    )
    result = format_performance_report(report)
    assert "20" in result
    assert "60.0%" in result
    assert "$150.50" in result
    assert "1.25" in result


def test_format_performance_report_zero_trades():
    report = PerformanceReport(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    result = format_performance_report(report)
    assert "0" in result


# ---------------------------------------------------------------------------
# format_positions_list
# ---------------------------------------------------------------------------


def test_format_positions_list_empty():
    result = format_positions_list([])
    assert result == "No open positions."


def test_format_positions_list_with_positions():
    positions = [
        {
            "market": "NYC-TEMP",
            "side": "YES",
            "amount": 100,
            "entry_price": "0.65",
            "mark_price": "0.70",
            "unrealized_pnl": "5.00",
        }
    ]
    result = format_positions_list(positions)
    assert "NYC-TEMP" in result
    assert "YES" in result
    assert "100" in result


# ---------------------------------------------------------------------------
# format_status
# ---------------------------------------------------------------------------


def test_format_status_positive_pnl():
    result = format_status(bankroll=1050.00, open_count=3, daily_pnl=50.00)
    assert "$1050.00" in result
    assert "3" in result
    assert "+$50.00" in result


def test_format_status_negative_pnl():
    result = format_status(bankroll=950.00, open_count=1, daily_pnl=-50.00)
    assert "$950.00" in result
    assert "$-50.00" in result


# ---------------------------------------------------------------------------
# format_agent_performance
# ---------------------------------------------------------------------------


def test_format_agent_performance():
    agent_stats = {
        "Conservative": {"trades": 50, "win_rate": 0.62, "pnl": 45.0, "weight": 0.30},
        "Aggressive": {"trades": 80, "win_rate": 0.58, "pnl": 120.0, "weight": 0.30},
    }
    result = format_agent_performance(agent_stats)
    assert "Conservative" in result
    assert "Aggressive" in result
    assert "62%" in result
    assert "58%" in result
    assert "0.30" in result


def test_format_agent_performance_empty():
    result = format_agent_performance({})
    assert result == "Agent Performance:"


# ---------------------------------------------------------------------------
# format_vpin_status
# ---------------------------------------------------------------------------


def test_format_vpin_status_normal():
    result = format_vpin_status(vpin=0.30, is_toxic=False)
    assert "0.30" in result
    assert "Normal" in result
    assert "TOXIC" not in result


def test_format_vpin_status_toxic():
    result = format_vpin_status(vpin=0.80, is_toxic=True)
    assert "0.80" in result
    assert "TOXIC" in result


# ---------------------------------------------------------------------------
# format_kill_confirmation
# ---------------------------------------------------------------------------


def test_format_kill_confirmation():
    result = format_kill_confirmation()
    assert "KILL SWITCH" in result
    assert "Manual restart" in result


# ---------------------------------------------------------------------------
# format_performance_dashboard
# ---------------------------------------------------------------------------


def test_format_performance_dashboard():
    """Dashboard output contains bankroll, P&L, win rate, Sharpe, and Brier score."""
    result = format_performance_dashboard(
        bankroll=1250.00,
        total_pnl=250.00,
        win_rate=0.63,
        sharpe=1.45,
        max_drawdown=0.07,
        brier_score=0.1823,
        agent_stats={},
    )
    assert "$1250.00" in result
    assert "$250.00" in result
    assert "63.0%" in result
    assert "1.45" in result
    assert "0.1823" in result


def test_format_performance_dashboard_with_agents():
    """Per-agent breakdown appears when agent_stats is provided."""
    agent_stats = {
        "Conservative": {"trades": 40, "win_rate": 0.65, "pnl": 80.0, "weight": 0.35},
        "Aggressive": {"trades": 60, "win_rate": 0.55, "pnl": 170.0, "weight": 0.25},
    }
    result = format_performance_dashboard(
        bankroll=1500.00,
        total_pnl=300.00,
        win_rate=0.60,
        sharpe=1.20,
        max_drawdown=0.10,
        brier_score=0.2100,
        agent_stats=agent_stats,
    )
    assert "Conservative" in result
    assert "Aggressive" in result
    assert "65%" in result
    assert "55%" in result
    assert "$80.00" in result
    assert "$170.00" in result


# ---------------------------------------------------------------------------
# format_morning_briefing
# ---------------------------------------------------------------------------


def test_format_morning_briefing():
    result = format_morning_briefing("Prices moved up overnight.")
    assert result == "MORNING BRIEFING:\nPrices moved up overnight."


# ---------------------------------------------------------------------------
# format_edge_decay_alert
# ---------------------------------------------------------------------------


def test_format_edge_decay_alert():
    result = format_edge_decay_alert(old_sharpe=2.10, new_sharpe=1.40, assessment="ALERT: Edge may be eroding")
    assert "EDGE DECAY CHECK:" in result
    assert "2.10" in result
    assert "1.40" in result
    assert "ALERT: Edge may be eroding" in result


def test_format_edge_decay_alert_no_old():
    result = format_edge_decay_alert(old_sharpe=None, new_sharpe=1.20, assessment="Stable")
    assert "N/A" in result
    assert "1.20" in result
    assert "Stable" in result


# ---------------------------------------------------------------------------
# format_competitor_alert
# ---------------------------------------------------------------------------


def test_format_competitor_alert():
    wallets = [
        {"address": "0xABCDEF1234567890", "trades": 75, "sharpe": 1.85},
        {"address": "0x1122334455667788", "trades": 120, "sharpe": 2.30},
    ]
    result = format_competitor_alert(wallets)
    assert "NEW WEATHER WALLETS (2):" in result
    assert "0xABCDEF12" in result
    assert "75 trades" in result
    assert "1.85" in result


def test_format_competitor_alert_empty():
    result = format_competitor_alert([])
    assert result == "No new weather wallets detected this week."
