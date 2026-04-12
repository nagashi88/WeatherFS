"""Tests for the Wallet Intelligence system (WalletCollector, WalletProfiler, WalletQueryEngine)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from weatheredge.db.database import init_db
from weatheredge.db.models import Base, WalletProfile
from weatheredge.intelligence.wallet_profiler import (
    WalletCollector,
    WalletProfiler,
    WalletQueryEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine():
    """In-memory SQLite engine for isolation."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return engine


def make_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()


def make_api_response(text: str):
    """Build a minimal mock OpenAI-compatible chat completion response."""
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def make_leaderboard_entry(wallet: str = "0xABC", sharpe: float = 2.5):
    return {
        "wallet": wallet,
        "sharpe": sharpe,
        "max_drawdown": 0.10,
        "categories": ["weather"],
        "is_automated": True,
    }


def make_trade(pnl: float = 10.0, is_weather: bool = True):
    return {
        "pnl": pnl,
        "market": "Will NYC temperature exceed 80F on 2026-07-01?" if is_weather else "Will BTC hit $100k?",
    }


# ---------------------------------------------------------------------------
# WalletCollector tests
# ---------------------------------------------------------------------------

class TestWalletCollector:

    def test_collect_constructs_correct_commands(self):
        """subprocess.run is called with the expected CLI command args."""
        collector = WalletCollector()

        leaderboard_data = [make_leaderboard_entry("0xDEAD")]
        weather_trades = [make_trade(pnl=5.0, is_weather=True) for _ in range(60)]

        leaderboard_result = MagicMock()
        leaderboard_result.stdout = json.dumps(leaderboard_data)

        trades_result = MagicMock()
        trades_result.stdout = json.dumps(weather_trades)

        call_results = [leaderboard_result, trades_result]

        with patch("subprocess.run", side_effect=call_results) as mock_run:
            collector.collect_weather_wallets(min_trades=50)

        assert mock_run.call_count == 2

        # First call: leaderboard
        first_args = mock_run.call_args_list[0][0][0]
        assert first_args[0] == "polymarket"
        assert "leaderboard" in first_args
        assert "--json" in first_args

        # Second call: trades for the wallet
        second_args = mock_run.call_args_list[1][0][0]
        assert second_args[0] == "polymarket"
        assert "trades" in second_args
        assert "0xDEAD" in second_args
        assert "--json" in second_args

    def test_collect_filters_by_min_trades(self):
        """Wallets with fewer than min_trades weather trades are excluded."""
        collector = WalletCollector()

        leaderboard_data = [
            make_leaderboard_entry("0xRICH"),
            make_leaderboard_entry("0xPOOR"),
        ]

        rich_trades = [make_trade(is_weather=True) for _ in range(60)]
        poor_trades = [make_trade(is_weather=True) for _ in range(10)]  # below threshold

        leaderboard_result = MagicMock()
        leaderboard_result.stdout = json.dumps(leaderboard_data)

        rich_trades_result = MagicMock()
        rich_trades_result.stdout = json.dumps(rich_trades)

        poor_trades_result = MagicMock()
        poor_trades_result.stdout = json.dumps(poor_trades)

        with patch("subprocess.run", side_effect=[leaderboard_result, rich_trades_result, poor_trades_result]):
            wallets = collector.collect_weather_wallets(min_trades=50)

        assert len(wallets) == 1
        assert wallets[0]["address"] == "0xRICH"

    def test_collect_handles_missing_cli(self):
        """FileNotFoundError from subprocess -> empty list, no crash."""
        collector = WalletCollector()

        with patch("subprocess.run", side_effect=FileNotFoundError("polymarket not found")):
            result = collector.collect_weather_wallets(min_trades=50)

        assert result == []


# ---------------------------------------------------------------------------
# WalletProfiler tests
# ---------------------------------------------------------------------------

class TestWalletProfiler:

    def _make_profiler(self, mock_client=None):
        engine = make_engine()
        profiler = WalletProfiler(engine=engine, api_key="test-key")
        if mock_client is not None:
            profiler._client = mock_client
        return profiler, engine

    def test_profile_prompt_includes_stats(self):
        """The prompt sent to the LLM includes the wallet address, trades, and sharpe."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = make_api_response("This wallet is a weather specialist.")

        profiler, _ = self._make_profiler(mock_client)

        stats = {
            "address": "0xTEST123",
            "total_trades": 150,
            "win_rate": 0.65,
            "sharpe_ratio": 2.8,
            "total_pnl": 3200.0,
            "max_drawdown": 0.12,
            "categories": ["weather"],
        }
        profiler.profile_wallet(stats)

        call_kwargs = mock_client.chat.completions.create.call_args
        prompt_text = call_kwargs.kwargs.get("messages", [{}])[0].get("content", "")

        assert "0xTEST123" in prompt_text
        assert "150" in prompt_text
        assert "2.80" in prompt_text

    def test_profile_stores_to_db(self):
        """store_profile writes a WalletProfile row to the database."""
        engine = make_engine()
        profiler = WalletProfiler(engine=engine, api_key="test-key")
        session = make_session(engine)

        stats = {
            "total_trades": 75,
            "win_rate": 0.60,
            "sharpe_ratio": 1.9,
            "total_pnl": 800.0,
            "max_drawdown": 0.08,
            "categories": ["weather", "politics"],
            "is_automated": False,
        }

        profiler.store_profile(session, "0xSTORE", "Great weather trader.", stats)
        session.commit()

        row = session.get(WalletProfile, "0xSTORE")
        assert row is not None
        assert row.wallet_address == "0xSTORE"
        assert row.profile_text == "Great weather trader."
        assert row.total_trades == 75
        assert row.win_rate == pytest.approx(0.60)
        assert row.sharpe_ratio == pytest.approx(1.9)

        session.close()

    def test_build_index_creates_json(self):
        """build_index writes a JSON file with the correct structure."""
        profiler, _ = self._make_profiler()

        profiles = [
            {
                "address": "0xAAA",
                "profile_text": "Aggressive weather arb bot.",
                "total_trades": 200,
                "win_rate": 0.70,
                "sharpe_ratio": 3.1,
            },
            {
                "address": "0xBBB",
                "profile_text": "Conservative manual trader.",
                "total_trades": 80,
                "win_rate": 0.55,
                "sharpe_ratio": 1.5,
            },
        ]

        index_path = profiler.build_index(profiles)

        assert Path(index_path).exists()

        data = json.loads(Path(index_path).read_text())
        assert isinstance(data, list)
        assert len(data) == 2

        first = data[0]
        assert first["address"] == "0xAAA"
        assert first["profile_text"] == "Aggressive weather arb bot."
        assert "stats" in first


# ---------------------------------------------------------------------------
# WalletQueryEngine tests
# ---------------------------------------------------------------------------

class TestWalletQueryEngine:

    def _make_index(self, profiles: list[dict], tmp_dir: str) -> str:
        """Write a wallet index JSON to a temp file and return its path."""
        index_path = os.path.join(tmp_dir, "wallet_index.json")
        entries = [
            {
                "address": p.get("address", ""),
                "profile_text": p.get("profile_text", ""),
                "stats": {k: v for k, v in p.items() if k not in ("address", "profile_text")},
            }
            for p in profiles
        ]
        Path(index_path).write_text(json.dumps(entries), encoding="utf-8")
        return index_path

    def test_query_returns_answer(self):
        """query() returns the LLM response text when index has data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            profiles = [
                {"address": "0xSOPH", "profile_text": "Top weather arb.", "sharpe_ratio": 3.2},
            ]
            index_path = self._make_index(profiles, tmp_dir)

            engine = WalletQueryEngine(index_path=index_path, api_key="test-key")

            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = make_api_response("0xSOPH has Sharpe 3.2.")
            engine._client = mock_client

            answer = engine.query("Who is the best weather trader?")

            assert answer == "0xSOPH has Sharpe 3.2."
            mock_client.chat.completions.create.assert_called_once()

    def test_query_empty_index(self):
        """query() returns 'No wallet data available' when index is empty."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = self._make_index([], tmp_dir)
            engine = WalletQueryEngine(index_path=index_path, api_key="test-key")
            mock_client = MagicMock()
            engine._client = mock_client

            answer = engine.query("Who is the best weather trader?")

            assert answer == "No wallet data available"
            mock_client.chat.completions.create.assert_not_called()

    def test_query_api_error(self):
        """query() returns an error message string when the LLM call fails."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            profiles = [
                {"address": "0xERR", "profile_text": "Some trader.", "sharpe_ratio": 1.0},
            ]
            index_path = self._make_index(profiles, tmp_dir)

            engine = WalletQueryEngine(index_path=index_path, api_key="test-key")

            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = RuntimeError("API timeout")
            engine._client = mock_client

            answer = engine.query("What strategies are most profitable?")

            assert "error" in answer.lower() or "Error" in answer
