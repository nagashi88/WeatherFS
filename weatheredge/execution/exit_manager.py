"""Exit strategy manager for WeatherEdge.

Evaluates 5 exit rules per open position each decision cycle, in priority:
  0. Structural bucket invalidation — buy_yes on physically-impossible bucket
  1. EV-Based Exit (Phase 1) — close if EV(close) > EV(hold) + buffer
  2. Edge-Based Trailing Stop — fallback if model flips against us by >3%
  3. Asymmetric Capital Protection — P&L stop-loss (gated)
  4. Time-Decay Exit — lock profits / force-exit near resolution

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
MARKET_AGAINST_BUY_NO_YES_PRICE: float = 0.95
MARKET_AGAINST_BUY_YES_PRICE: float = 0.05
BELOW_MARKET_INTRADAY_YES_PRICE: float = 0.80


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

        # --- Unrealized P&L (net of exit taker fee) ---
        # pnl_pct drives time_decay profit-lock and future capital_protect.
        # It must reflect *what we would realize if we closed now*, so
        # subtract the Kalshi taker fee at the current price. Entry fee is
        # already baked into `cost` at entry time (paper_trader). Symmetric
        # in price so the same fee(price) value applies to YES and NO sides.
        from weatheredge.probability.edge import kalshi_taker_fee
        if side == "buy_yes":
            current_value = current_price * amount
        else:  # buy_no — we hold NO contracts worth (1 - yes_price)
            current_value = (1.0 - current_price) * amount
        platform = pos.get("platform", "polymarket")
        exit_fee = kalshi_taker_fee(current_price) * amount if platform == "kalshi" else 0.0
        unrealized_pnl = current_value - exit_fee - cost
        pnl_pct = unrealized_pnl / cost if cost > 0 else 0.0

        # --- Current edge (model vs market, from our position's perspective) ---
        # CCG-bughunt2 G8: carry a side-aware NET current edge alongside the
        # gross value so the edge_stop log shows units that match the stored
        # entry_edge (which is net). Decision still uses gross (threshold is
        # calibrated against gross), but operators reading logs need apples-
        # to-apples values — mixing net entry_edge with gross current_edge
        # silently understates how much edge has actually decayed.
        if side == "buy_yes":
            current_edge = model_prob - current_price
            current_edge_net = getattr(opp, "net_edge", current_edge)
        else:
            # buy_no: we profit when YES price falls; edge = (1-model) - (1-price) = price - model
            current_edge = (1.0 - model_prob) - (1.0 - current_price)
            current_edge_net = getattr(opp, "net_edge_no", current_edge)

        # --- Lead time ---
        lead_time_hours = self._lead_time_hours(market_id, market_metadata, pos)

        # ================================================================
        # EXIT 0: Structural bucket invalidation (pre-empts all other rules)
        # When METAR observes daily_high > bucket_high, YES resolution becomes
        # physically impossible (daily_high is monotone non-decreasing intra-day).
        # Standard exits (edge_flip, ev_based, time_decay) fail to close buy_yes
        # positions in this state because market≈0 and model≈0 leave no edge to
        # trigger on and make "hold" marginally better than "close" in EV terms.
        # This rule is intentionally asymmetric: buy_no does NOT need it because
        # when METAR shows temp in-range the daily_high can still climb out, and
        # the large (1-model) shift handles buy_no via edge_flip naturally.
        # ================================================================
        if (
            getattr(opp, "bucket_invalidated_for_yes", False)
            and side == "buy_yes"
        ):
            log.info(
                "EXIT bucket_invalidated: market=%s side=buy_yes metar_high=%.1f "
                "bucket_high=%s → YES physically impossible",
                market_id,
                opp.bucket_metar_high if opp.bucket_metar_high is not None else -999.0,
                f"{opp.bucket_high:.0f}" if opp.bucket_high is not None else "?",
            )
            return "bucket_invalidated"

        from weatheredge.config import MODULES

        # Same-day "below" markets can become dangerous before final resolution:
        # if observed high is still below threshold and the tradable market is
        # already strongly YES, a buy_no is fighting both tape and live weather.
        if (
            MODULES.get("exit_edge_flip", True)
            and side == "buy_no"
            and getattr(opp, "direction", "") == "below"
            and getattr(opp, "market_type", "daily_high") == "daily_high"
            and current_price >= BELOW_MARKET_INTRADAY_YES_PRICE
            and getattr(opp, "metar_daily_high", None) is not None
            and getattr(opp, "threshold", None) is not None
            and opp.metar_daily_high < opp.threshold
            and lead_time_hours is not None
            and lead_time_hours <= 12.0
        ):
            log.info(
                "EXIT edge_stop: market=%s below-market intraday stop "
                "(YES=%.3f >= %.2f, daily_high=%.1f < threshold=%.1f, lead=%.1fh) → selling",
                market_id,
                current_price,
                BELOW_MARKET_INTRADAY_YES_PRICE,
                opp.metar_daily_high,
                opp.threshold,
                lead_time_hours,
            )
            return "edge_stop"

        # ================================================================
        # EXIT 1: EV-Based Exit (Phase 1 primary decision rule)
        # STEP 2: moved before edge_flip. With EV at check #1, closes that
        # optimal EV would author get labeled ev_based instead of being
        # shadowed by the edge_flip heuristic. When EV says hold (returns
        # False), execution falls through to edge_flip as a safety net.
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
        # EXIT 2: Edge-Based Trailing Stop (fallback when EV says hold but
        # current edge has flipped hard against our position).
        # CCG-bughunt2 G8: EDGE_STOP_THRESHOLD is calibrated against GROSS
        # current_edge, so the decision intentionally uses gross. But the
        # prior log line pasted `entry_edge` (net) next to `current_edge`
        # (gross) with identical "%%" formatting, silently understating
        # edge decay by the round-trip fee. Log now shows both the gross
        # value that triggered and the NET value that's comparable to
        # entry_edge — operators can read the real delta at a glance.
        # ================================================================
        if MODULES.get("exit_edge_flip", True) and current_edge < EDGE_STOP_THRESHOLD:
            log.info(
                "EXIT edge_stop: market=%s entry_edge_net=%.1f%% "
                "current_edge_net=%.1f%% (gross=%.1f%% < %.0f%% threshold) → selling",
                market_id,
                entry_edge * 100,
                current_edge_net * 100,
                current_edge * 100,
                EDGE_STOP_THRESHOLD * 100,
            )
            return "edge_stop"

        # ================================================================
        # EXIT 3: Asymmetric Capital Protection
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
        # EXIT 2b: Market-consensus kill switch
        # If the tradable market has moved to the tails against our side, the
        # position is functionally invalidated even if our model still reports
        # stale edge. Keep this after capital_protect so phases that explicitly
        # enable P&L stops retain their more specific exit reason.
        # ================================================================
        if MODULES.get("exit_edge_flip", True):
            if side == "buy_no" and current_price >= MARKET_AGAINST_BUY_NO_YES_PRICE:
                log.info(
                    "EXIT edge_stop: market=%s market consensus against buy_no "
                    "(YES=%.3f >= %.2f) → selling",
                    market_id, current_price, MARKET_AGAINST_BUY_NO_YES_PRICE,
                )
                return "edge_stop"
            if side == "buy_yes" and current_price <= MARKET_AGAINST_BUY_YES_PRICE:
                log.info(
                    "EXIT edge_stop: market=%s market consensus against buy_yes "
                    "(YES=%.3f <= %.2f) → selling",
                    market_id, current_price, MARKET_AGAINST_BUY_YES_PRICE,
                )
                return "edge_stop"

        # ================================================================
        # EXIT 3: Time-Decay Exit
        # CCG-bughunt2 Codex F8: honor MODULES["exit_time_decay"] so phases
        # can turn the rule off without patching the code.
        # ================================================================
        if MODULES.get("exit_time_decay", True) and lead_time_hours is not None:
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

        Per-contract exit cost is price-dependent on Kalshi (taker fee follows
        ceil(0.07·p·(1-p)·100)/100, symmetric in p, plus one half-spread). A
        flat 0.02 was accurate only at floor-bound prices (p≲0.10); at contested
        prices (p≈0.5) it underestimated by ~33%, making EV-exit fire too
        eagerly on Phase-1 trades entering at contested markets.
        """
        from weatheredge.probability.edge import kalshi_taker_fee, PLATFORM_FEES

        side = pos.get("side", "buy_yes")
        cost = pos.get("cost", 0.0)
        amount = pos.get("amount", 0.0)
        platform = pos.get("platform", "kalshi")

        # One-leg cost to close now. Kalshi: taker fee at current price
        # (symmetric in YES/NO because fee(p) == fee(1-p)) + half-spread.
        # Non-Kalshi platforms retain the pre-existing flat 0.02 per contract
        # — this bug is Kalshi-specific and non-Kalshi paths are out of scope.
        if platform == "kalshi":
            per_contract_exit_cost = kalshi_taker_fee(current_price) + (
                PLATFORM_FEES["kalshi"]["avg_spread"] / 2.0
            )
        else:
            per_contract_exit_cost = 0.02
        exit_cost = per_contract_exit_cost * amount

        if side == "buy_yes":
            ev_hold = model_prob * (amount - cost) - (1 - model_prob) * cost
        else:
            ev_hold = (1 - model_prob) * (amount - cost) - model_prob * cost

        if side == "buy_yes":
            current_value = current_price * amount
        else:
            current_value = (1.0 - current_price) * amount
        ev_close = current_value - cost - exit_cost

        # Require meaningful EV advantage to close — avoid churn on tiny
        # differences. Buffer = 5% of cost.
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
