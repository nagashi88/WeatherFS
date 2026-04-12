"""Exit strategy manager for WeatherEdge.

Evaluates 3 exit rules on every open position each decision cycle:
  1. Edge-Based Trailing Stop  — sell if model flips against us by >3%
  2. Asymmetric Capital Protection — stop-loss with model-confidence gate
  3. Time-Decay Exit — lock profits / force-exit near resolution

Returns a list of (position, reason) tuples for positions that should be closed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from weatheredge.data.station_time import hours_until_market_close

log = logging.getLogger(__name__)

# --- Thresholds ---
EDGE_STOP_THRESHOLD: float = -0.03          # Exit 1: model edge < -3%
SOFT_LOSS_PCT: float = -0.05               # Exit 2: -5% P&L soft stop
HARD_LOSS_PCT: float = -0.15              # Exit 2: -15% P&L hard stop
MODEL_CONF_FLOOR: float = 0.70            # Exit 2: below this = low confidence
TIME_PROFIT_HOURS: float = 3.0            # Exit 3: take profit inside 3h
TIME_PROFIT_MIN_PNL: float = 0.05         # Exit 3: profit threshold for 3h window
TIME_FORCE_EXIT_HOURS: float = 1.0        # Exit 3: force-close inside 1h


class ExitManager:
    """Evaluates exit rules for all open positions each decision cycle."""

    def evaluate(
        self,
        positions: list[dict],
        opportunities: dict,          # market_id → MarketOpportunity
        market_metadata: dict[str, dict],
    ) -> list[tuple[dict, str]]:
        """Return list of (position, exit_reason) for positions that should be closed."""
        to_close: list[tuple[dict, str]] = []
        for pos in positions:
            reason = self._check_position(pos, opportunities, market_metadata)
            if reason:
                to_close.append((pos, reason))
        return to_close

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_position(
        self,
        pos: dict,
        opportunities: dict,
        market_metadata: dict[str, dict],
    ) -> str | None:
        """Return exit reason string or None if position should stay open."""
        market_id = pos["market_id"]
        opp = opportunities.get(market_id)
        if opp is None:
            # No current market data this cycle — cannot evaluate, keep open
            return None

        side = pos["side"]
        entry_price = pos["price"]
        amount = pos["amount"]
        cost = pos["cost"]
        entry_edge = pos.get("edge_at_entry", 0.0)
        model_prob = opp.model_probability
        current_price = opp.market_price  # current YES price

        # --- Unrealized P&L ---
        if side == "buy_yes":
            current_value = current_price * amount
        else:  # buy_no — we hold NO contracts worth (1 - yes_price)
            current_value = (1.0 - current_price) * amount
        unrealized_pnl = current_value - cost
        pnl_pct = unrealized_pnl / cost if cost > 0 else 0.0

        # --- Current edge (model vs market, from our position's perspective) ---
        if side == "buy_yes":
            current_edge = model_prob - current_price
        else:
            # buy_no: we profit when YES price falls; edge = (1-model) - (1-price) = price - model
            current_edge = (1.0 - model_prob) - (1.0 - current_price)

        # --- Lead time ---
        lead_time_hours = self._lead_time_hours(market_id, market_metadata, pos)

        # ================================================================
        # EXIT 1: Edge-Based Trailing Stop
        # Gated by MODULES flag — disabled in Phase 1+ in favour of EV-based exit.
        # ================================================================
        from weatheredge.config import MODULES
        if MODULES.get("exit_edge_flip", True) and current_edge < EDGE_STOP_THRESHOLD:
            log.info(
                "EXIT edge_stop: market=%s entry_edge=%.1f%% current_edge=%.1f%% → selling",
                market_id, entry_edge * 100, current_edge * 100,
            )
            return "edge_stop"

        # ================================================================
        # EXIT 2: Asymmetric Capital Protection
        # PRD v3.1: STRUCTURAL KILL on P&L stop-loss — gated by MODULES flag.
        # On prediction markets, P&L drawdown without model change = better odds.
        # ================================================================
        if MODULES.get("exit_pnl_stoploss", False) and pnl_pct < HARD_LOSS_PCT:
            log.info(
                "EXIT capital_protect: market=%s pnl=%.1f%% model_prob=%.1f%% → selling (hard stop)",
                market_id, pnl_pct * 100, model_prob * 100,
            )
            return "capital_protect"

        if MODULES.get("exit_pnl_stoploss", False) and pnl_pct < SOFT_LOSS_PCT:
            # BugFix H1: For buy_no, model confidence in our position is (1 - model_prob)
            model_confidence = (1.0 - model_prob) if side == "buy_no" else model_prob
            if model_confidence < MODEL_CONF_FLOOR:
                log.info(
                    "EXIT capital_protect: market=%s pnl=%.1f%% model_confidence=%.1f%% → selling",
                    market_id, pnl_pct * 100, model_confidence * 100,
                )
                return "capital_protect"
            # model confidence >= 70% → HOLD (dip, model still confident)
            log.debug(
                "exit_manager: market=%s pnl=%.1f%% but model_confidence=%.1f%% >= %.0f%% → holding dip",
                market_id, pnl_pct * 100, model_confidence * 100, MODEL_CONF_FLOOR * 100,
            )

        # ================================================================
        # EXIT 2b: EV-Based Exit (Phase 1 — replaces P&L stop-loss)
        # ================================================================
        if MODULES.get("exit_ev_based", False):
            ev_exit = self._evaluate_exit_ev(pos, current_price, model_prob)
            if ev_exit:
                log.info(
                    "EXIT ev_based: market=%s pnl=%.1f%% → EV of closing > EV of holding",
                    market_id, pnl_pct * 100,
                )
                return "ev_based"

        # ================================================================
        # EXIT 3: Time-Decay Exit
        # ================================================================
        if lead_time_hours is not None:
            if lead_time_hours < TIME_FORCE_EXIT_HOURS:
                log.info(
                    "EXIT time_decay: market=%s lead_time=%.2fh pnl=%.1f%% → selling (wire)",
                    market_id, lead_time_hours, pnl_pct * 100,
                )
                return "time_decay"

            if lead_time_hours < TIME_PROFIT_HOURS and pnl_pct > TIME_PROFIT_MIN_PNL:
                log.info(
                    "EXIT time_decay: market=%s lead_time=%.2fh pnl=%.1f%% → selling (profit lock)",
                    market_id, lead_time_hours, pnl_pct * 100,
                )
                return "time_decay"

        return None

    def _evaluate_exit_ev(self, pos: dict, current_price: float, model_prob: float) -> bool:
        """Phase 1 EV-based exit: exit only if closing has higher EV than holding.

        On prediction markets, P&L drawdown without model change means better odds.
        Only exit when the expected value of holding is negative vs closing now.
        """
        entry_price = pos.get("price", 0.0)
        side = pos.get("side", "buy_yes")
        cost = pos.get("cost", 0.0)
        amount = pos.get("amount", 0.0)

        # Estimate spread cost to close (~2% round-trip)
        exit_cost = 0.02 * amount

        # EV of holding to resolution
        if side == "buy_yes":
            ev_hold = model_prob * (amount - cost) - (1 - model_prob) * cost
        else:
            ev_hold = (1 - model_prob) * (amount - cost) - model_prob * cost

        # EV of closing now (mark-to-market minus exit cost)
        if side == "buy_yes":
            current_value = current_price * amount
        else:
            current_value = (1.0 - current_price) * amount
        ev_close = current_value - cost - exit_cost

        # Require meaningful EV advantage to close — avoid churn on tiny differences.
        # Buffer = 5% of cost: e.g. on a $10 position, need $0.50 EV advantage to exit.
        ev_buffer = 0.05 * cost
        return ev_close > ev_hold + ev_buffer

    @staticmethod
    def _lead_time_hours(
        market_id: str,
        market_metadata: dict[str, dict],
        pos: dict | None = None,
    ) -> float | None:
        """Compute hours until market resolution.

        Tries parsed metadata first (Polymarket path), then falls back to the
        position's own market_date/station fields so Kalshi markets also get
        time-decay exit coverage.
        """
        meta = market_metadata.get(market_id, {})
        parsed = meta.get("parsed")

        # --- Extract date and station from best available source ---
        market_date: str | None = None
        station: str | None = None

        if parsed and hasattr(parsed, "date") and parsed.date:
            market_date = parsed.date
            resolution_source = meta.get("resolution_source")
            if isinstance(resolution_source, dict):
                station = resolution_source.get("station")
            if not station and hasattr(parsed, "station"):
                station = parsed.station

        # Fallback: use position's persisted metadata (covers Kalshi)
        if not market_date and pos:
            market_date = pos.get("market_date") or None
        if not station and pos:
            station = pos.get("station") or None

        if not market_date:
            return None

        # Compute lead time with station-aware timezone if possible
        if station:
            delta = hours_until_market_close(market_date, station)
            if delta is not None:
                return max(0.0, delta)

        # Fallback for unknown station timezone: UTC end-of-day.
        try:
            target = datetime.strptime(market_date, "%Y-%m-%d").replace(
                hour=23, minute=59, tzinfo=timezone.utc
            )
            delta = (target - datetime.now(timezone.utc)).total_seconds() / 3600.0
            return max(0.0, delta)
        except (ValueError, TypeError):
            pass
        return None
