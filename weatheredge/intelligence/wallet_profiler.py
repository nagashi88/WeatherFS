"""Wallet Intelligence system for WeatherEdge.

Three components:
- WalletCollector: shells out to polymarket-cli to gather wallet data
- WalletProfiler: uses Claude Haiku to generate analytical profiles
- WalletQueryEngine: natural language queries against a JSON index
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from weatheredge.llm import get_llm_client, chat_completion

log = logging.getLogger(__name__)

# Directory where the wallet index is stored (relative to this file's package root)
_INTELLIGENCE_DIR = Path(__file__).resolve().parent


class WalletCollector:
    """Collects Polymarket wallet data via polymarket-cli."""

    def collect_weather_wallets(self, min_trades: int = 50) -> list[dict]:
        """Collect wallets active in weather markets with at least min_trades trades.

        Shells out to polymarket-cli for leaderboard, holder, and trade data.
        Returns list of wallet stat dicts. Returns empty list if CLI not found.
        """
        wallets: list[dict] = []

        # Step 1: get monthly leaderboard sorted by PnL
        try:
            result = subprocess.run(
                ["polymarket", "data", "leaderboard", "--period", "month", "--order-by", "pnl", "--json"],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            log.warning("polymarket-cli not found; skipping wallet collection")
            return []

        try:
            leaderboard = json.loads(result.stdout) if result.stdout else []
        except json.JSONDecodeError:
            log.warning("Failed to parse leaderboard JSON: %s", result.stdout[:200])
            leaderboard = []

        if not leaderboard:
            return []

        # Step 2: for each leaderboard wallet, fetch trades and filter by weather activity
        for entry in leaderboard:
            wallet_address = entry.get("wallet") or entry.get("address") or entry.get("id", "")
            if not wallet_address:
                continue

            try:
                trade_result = subprocess.run(
                    ["polymarket", "data", "trades", wallet_address, "--limit", "500", "--json"],
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError:
                log.warning("polymarket-cli not found; skipping wallet collection")
                return []

            try:
                trades = json.loads(trade_result.stdout) if trade_result.stdout else []
            except json.JSONDecodeError:
                log.warning("Failed to parse trades JSON for %s", wallet_address)
                continue

            # Filter for weather market trades
            weather_trades = [
                t for t in trades
                if _is_weather_trade(t)
            ]

            if len(weather_trades) < min_trades:
                continue

            wallet_stats = _compute_stats(wallet_address, weather_trades, entry)
            wallets.append(wallet_stats)

        return wallets


def _is_weather_trade(trade: dict) -> bool:
    """Return True if the trade appears to be in a weather market."""
    text_fields = [
        str(trade.get("market", "")),
        str(trade.get("market_slug", "")),
        str(trade.get("question", "")),
        str(trade.get("category", "")),
    ]
    weather_keywords = {"weather", "temperature", "rain", "snow", "wind", "humidity", "precipitation", "forecast"}
    combined = " ".join(text_fields).lower()
    return any(kw in combined for kw in weather_keywords)


def _compute_stats(wallet_address: str, weather_trades: list[dict], leaderboard_entry: dict) -> dict:
    """Compute summary statistics from trade list and leaderboard entry."""
    total_trades = len(weather_trades)
    wins = sum(1 for t in weather_trades if float(t.get("pnl", 0)) > 0)
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    total_pnl = sum(float(t.get("pnl", 0)) for t in weather_trades)

    return {
        "address": wallet_address,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "sharpe_ratio": float(leaderboard_entry.get("sharpe", 0.0)),
        "max_drawdown": float(leaderboard_entry.get("max_drawdown", 0.0)),
        "categories": leaderboard_entry.get("categories", ["weather"]),
        "is_automated": leaderboard_entry.get("is_automated"),
    }


class WalletProfiler:
    """Generates Claude Haiku analytical profiles for wallets and stores them."""

    def __init__(self, engine, api_key: str) -> None:
        self._engine = engine
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = get_llm_client(self._api_key)
        return self._client

    def profile_wallet(self, wallet_stats: dict, model: str | None = None) -> str:
        """Call LLM to generate a 3-4 sentence analytical profile.

        Returns the profile text string.
        """
        from weatheredge.config import Settings
        _model = model or Settings().llm_model_cheap

        address = wallet_stats.get("address", "unknown")
        trades = wallet_stats.get("total_trades", 0)
        win_rate = wallet_stats.get("win_rate", 0.0)
        sharpe = wallet_stats.get("sharpe_ratio", 0.0)
        pnl = wallet_stats.get("total_pnl", 0.0)
        max_dd = wallet_stats.get("max_drawdown", 0.0)
        categories = wallet_stats.get("categories", [])

        prompt = (
            "Write a 3-4 sentence analytical profile of this Polymarket wallet. "
            "Focus on: strategy type, market category specialization, edge sustainability, "
            "and whether behavior is automated or manual.\n\n"
            f"Address: {address}\n"
            f"Trades: {trades} | Win rate: {win_rate:.1%}\n"
            f"Sharpe: {sharpe:.2f} | Total P&L: ${pnl:.2f}\n"
            f"Max drawdown: {max_dd:.1%} | Categories: {categories}"
        )

        client = self._get_client()
        result = chat_completion(client, _model, prompt, max_tokens=300)
        return result or ""

    def store_profile(self, db_session, wallet_address: str, profile_text: str, stats: dict) -> None:
        """Write or update a wallet profile row in the wallet_profiles table."""
        from weatheredge.db.models import WalletProfile

        existing = db_session.get(WalletProfile, wallet_address)
        if existing is None:
            record = WalletProfile(
                wallet_address=wallet_address,
                profile_text=profile_text,
                total_trades=stats.get("total_trades"),
                win_rate=stats.get("win_rate"),
                sharpe_ratio=stats.get("sharpe_ratio"),
                total_pnl=stats.get("total_pnl"),
                max_drawdown=stats.get("max_drawdown"),
                categories=json.dumps(stats.get("categories", [])),
                is_automated=stats.get("is_automated"),
            )
            db_session.add(record)
        else:
            existing.profile_text = profile_text
            existing.total_trades = stats.get("total_trades")
            existing.win_rate = stats.get("win_rate")
            existing.sharpe_ratio = stats.get("sharpe_ratio")
            existing.total_pnl = stats.get("total_pnl")
            existing.max_drawdown = stats.get("max_drawdown")
            existing.categories = json.dumps(stats.get("categories", []))
            existing.is_automated = stats.get("is_automated")

    def build_index(self, profiles: list[dict]) -> str:
        """Save profiles to intelligence/wallet_index.json and return the path."""
        index_path = _INTELLIGENCE_DIR / "wallet_index.json"
        index_entries = [
            {
                "address": p.get("address", ""),
                "profile_text": p.get("profile_text", ""),
                "stats": {
                    k: v for k, v in p.items()
                    if k not in ("address", "profile_text")
                },
            }
            for p in profiles
        ]
        index_path.write_text(json.dumps(index_entries, indent=2), encoding="utf-8")
        log.info("Wallet index written to %s (%d entries)", index_path, len(index_entries))
        return str(index_path)


class WalletQueryEngine:
    """Natural language query engine over the wallet profile index."""

    def __init__(self, index_path: str, api_key: str) -> None:
        self._index_path = index_path
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = get_llm_client(self._api_key)
        return self._client

    def query(self, question: str, model: str = "anthropic/claude-sonnet-4") -> str:
        """Answer a natural language question about the wallet index.

        Loads the JSON index, builds a context prompt, and calls the LLM.
        Returns "No wallet data available" for an empty index.
        Returns an error message string on API failure.
        """
        try:
            index_text = Path(self._index_path).read_text(encoding="utf-8")
            profiles = json.loads(index_text)
        except (FileNotFoundError, json.JSONDecodeError):
            profiles = []

        if not profiles:
            return "No wallet data available"

        # Build a compact context block (truncate if very large)
        context_lines = []
        for entry in profiles:
            addr = entry.get("address", "")
            profile = entry.get("profile_text", "")
            stats = entry.get("stats", {})
            context_lines.append(
                f"Wallet: {addr}\nProfile: {profile}\nStats: {json.dumps(stats)}\n"
            )
        context = "\n---\n".join(context_lines)

        prompt = (
            "You are a Polymarket analyst with access to wallet intelligence data. "
            "Answer the following question using the wallet profiles below.\n\n"
            f"WALLET DATA:\n{context}\n\n"
            f"QUESTION: {question}"
        )

        try:
            client = self._get_client()
            result = chat_completion(client, model, prompt, max_tokens=1000)
            if result is None:
                raise RuntimeError("No response from LLM")
            return result
        except Exception as e:
            log.warning("WalletQueryEngine API error: %s", e)
            return f"Error querying wallet intelligence: {e}"
