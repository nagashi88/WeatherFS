"""Paper trading engine for WeatherEdge.

PRD Section 6.1 — paper trading mode (hold-to-resolution lifecycle).
Addendum Gap 9 — unfilled limit orders expire after ORDER_TTL_HOURS (2h).
Trades are persisted to the DB ``trades`` table so they survive restarts.
"""

from __future__ import annotations

import logging

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

from weatheredge.probability.edge import kalshi_taker_fee

log = logging.getLogger(__name__)

ORDER_TTL_HOURS = 2  # Addendum Gap 9: unfilled orders expire after 2h


def _kalshi_fee(price: float, amount: float, platform: str) -> float:
    """Per-execution Kalshi taker fee (price-dependent, symmetric in p).

    Kalshi charges ceil(0.07·p·(1-p)·100)/100 per contract at every
    execution (entry and early exit); settlement to $0/$1 is fee-free.
    Polymarket has no explicit fee (spread cost lives in edge calc).
    """
    if platform != "kalshi":
        return 0.0
    return kalshi_taker_fee(price) * amount


@dataclass
class Order:
    market_id: str
    side: str        # 'buy_yes' or 'buy_no'
    price: float
    amount: float
    agent_name: str = ""
    model_probability: float = 0.0
    edge_at_entry: float = 0.0
    city: str = ""
    platform: str = "polymarket"
    station: str = ""
    market_date: str = ""
    gaussian_prob: float | None = None
    ensemble_prob: float | None = None
    blended_prob: float | None = None
    raw_model_prob: float | None = None  # Pre-clamp blended (audit only)
    order_type: str = "market"
    forecast_timestamp: datetime | None = None
    market_price_timestamp: datetime | None = None
    signal_ids: list[int] | None = None
    cash_capped: bool = False  # propagated from TradeOrder; persisted on Trade row
    regime: str | None = None  # strategy regime tag, persisted on Trade row


@dataclass
class PerformanceReport:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    roi: float
    sharpe_ratio: float
    max_drawdown: float
    avg_edge_at_entry: float


class PaperTrader:
    """Simulates paper trading with hold-to-resolution lifecycle.

    Positions are held until market resolution (won/lost) per Addendum Gap 9.
    Unfilled limit orders are auto-cancelled after ORDER_TTL_HOURS.
    When a DB engine is connected via ``connect_db()``, all trades are persisted
    to the ``trades`` table and restored on restart.
    """

    def __init__(self, initial_bankroll: float = 1000.0):
        self.bankroll = initial_bankroll
        self.initial_bankroll = initial_bankroll
        self._engine = None
        self.open_orders: list[dict] = []      # Unfilled limit orders
        self.open_positions: list[dict] = []   # Filled positions awaiting resolution
        self.last_limit_fills: list[dict] = []  # Filled positions from latest check_limit_fills()
        self.trade_history: list[dict] = []
        self.daily_pnl: dict[str, float] = {}  # date_str -> pnl
        # Re-entry guard: market_id -> {date_str -> entry_count}
        self._daily_entries: dict[str, dict[str, int]] = {}

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def connect_db(self, engine) -> None:
        """Attach a SQLAlchemy engine and restore state from the trades table.

        Called after ``init_db()`` in main so the PaperTrader can be created
        before the database is ready.
        """
        self._engine = engine
        self._load_from_db()

    def _load_from_db(self) -> None:
        """Restore open positions, trade history, daily P&L, and bankroll.

        BugFix H7: Reconstruct bankroll as initial + resolved_pnl - open_costs
        so that resolved trade returns are not lost on restart.
        """
        from sqlalchemy import select
        from weatheredge.db.database import get_session
        from weatheredge.db.models import Trade

        resolved_pnl_total = 0.0
        open_cost_total = 0.0

        with get_session(self._engine) as session:
            # Open positions
            open_trades = session.execute(
                select(Trade).where(Trade.mode == "paper", Trade.status == "open")
            ).scalars().all()

            for t in open_trades:
                self.open_positions.append(self._trade_to_dict(t))
                open_cost_total += t.cost

            pending_trades = session.execute(
                select(Trade).where(Trade.mode == "paper", Trade.status == "pending")
            ).scalars().all()

            for t in pending_trades:
                self.open_orders.append(self._trade_to_dict(t))

            # Resolved trades — reconstruct history and daily P&L
            resolved = session.execute(
                select(Trade).where(
                    Trade.mode == "paper",
                    Trade.status.in_(["won", "lost", "cancelled", "abandoned",
                                      "edge_stop", "capital_protect", "time_decay", "ev_based",
                                      "bucket_invalidated"]),
                )
            ).scalars().all()

            for t in resolved:
                self.trade_history.append(self._trade_to_dict(t))
                if t.pnl:
                    # CCG-bughunt2 G3: match the live booking path (close_position /
                    # resolution) which keys daily_pnl on market_date, not UTC
                    # resolved_at. A cross-midnight resolution otherwise buckets
                    # differently pre- vs post-restart, silently distorting the
                    # daily circuit breaker and P&L attribution.
                    day = (
                        t.market_date
                        or (t.resolved_at.strftime("%Y-%m-%d") if t.resolved_at else None)
                    )
                    if day:
                        self.daily_pnl[day] = self.daily_pnl.get(day, 0) + t.pnl
                resolved_pnl_total += (t.pnl or 0.0)

        # Reconstruct bankroll: initial + net resolved P&L - open position costs
        self.bankroll = self.initial_bankroll + resolved_pnl_total - open_cost_total

        # Restore daily entry counters keyed on market_date (station-local resolution
        # date). Prevents the re-entry guard from resetting at UTC midnight when
        # the market's local day is still active. Legacy trades without a stored
        # market_date fall back to the created_at date.
        today_dt = datetime.now(timezone.utc)
        today_utc = today_dt.strftime("%Y-%m-%d")
        # CCG-bughunt2 G5: only rebuild buckets whose market_date is still
        # potentially active. Past markets have already resolved and cannot
        # be re-entered, so their counters are dead memory. Without this,
        # _daily_entries grows to O(total_trades) on every restart because
        # trade_history is never pruned (see append paths at lines 499/533/571).
        # Cutoff = today-1 to cover overnight UTC windows where station-local
        # market_date still lags behind today_utc by a few hours.
        recent_floor = (today_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        for t in list(self.open_positions) + list(self.trade_history):
            created = t.get("created_at")
            created_day: str | None = None
            if created is not None:
                if isinstance(created, str):
                    try:
                        created = datetime.fromisoformat(created)
                    except ValueError:
                        created = None
                if created is not None:
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    created_day = created.strftime("%Y-%m-%d")
            raw_md = t.get("market_date")
            # DB _trade_to_dict coerces None → "" for market_date, so treat both
            # as "missing" for the legacy-fallback path.
            has_market_date = bool(raw_md)
            market_day = raw_md if has_market_date else created_day
            if market_day is None:
                continue
            if not has_market_date and market_day != today_utc:
                # Legacy row with no market_date that didn't happen "today UTC"
                # → skip; counting it would inflate today's limit incorrectly.
                continue
            # Open positions always count (still active regardless of market_date).
            # Resolved trades only count if market_date is recent enough to still
            # be tradeable today.
            is_open = t.get("status") == "open"
            if has_market_date and not is_open and market_day < recent_floor:
                continue
            key = self._market_id_key(t.get("market_id"))
            bucket = self._daily_entries.setdefault(key, {})
            bucket[market_day] = bucket.get(market_day, 0) + 1

        log.info(
            "PaperTrader loaded from DB: %d open positions, %d resolved, bankroll=$%.2f",
            len(self.open_positions), len(self.trade_history), self.bankroll,
        )

    @staticmethod
    def _market_id_key(market_id: object) -> str:
        """Normalize market ids for robust comparisons across int/str sources."""
        if market_id is None:
            return ""
        return str(market_id).strip()

    @staticmethod
    def _coerce_utc_datetime(value: object) -> datetime | None:
        """Return an aware UTC datetime for DB/in-memory timestamps.

        SQLite drops timezone info on DateTime columns, so persisted pending
        orders can come back as naive datetimes after restart.
        """
        if value is None:
            return None
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _trade_to_dict(t) -> dict:
        """Convert a DB Trade row to the in-memory dict format."""
        return {
            "db_id": t.id,
            "market_id": PaperTrader._market_id_key(t.market_id),
            "platform": t.platform,
            "side": t.side,
            "price": t.price,
            "amount": t.amount,
            "cost": t.cost,
            "agent_name": t.agent_name,
            "model_probability": t.model_probability,
            "edge_at_entry": t.edge_at_entry,
            "raw_model_prob": t.raw_model_prob,
            "city": "",
            "station": t.station or "",
            "market_date": t.market_date or "",
            "order_type": t.order_type or "market",
            "limit_price": t.limit_price,
            "forecast_timestamp": t.forecast_timestamp,
            "market_price_timestamp": t.market_price_timestamp,
            "cash_capped": bool(t.cash_capped),
            "status": t.status,
            "created_at": PaperTrader._coerce_utc_datetime(t.created_at),
            "pnl": t.pnl,
            "regime": t.regime,
        }

    def _persist_new_trade(self, trade_dict: dict) -> int | None:
        """Write a new open trade to the DB. Returns the DB row id."""
        if self._engine is None:
            return None

        from weatheredge.db.database import get_session
        from weatheredge.db.models import Trade

        with get_session(self._engine) as session:
            db_trade = Trade(
                market_id=str(trade_dict["market_id"]),
                platform=trade_dict.get("platform", "polymarket"),
                mode="paper",
                side=trade_dict["side"],
                price=trade_dict["price"],
                amount=trade_dict["amount"],
                cost=trade_dict["cost"],
                agent_name=trade_dict.get("agent_name"),
                model_probability=trade_dict.get("model_probability"),
                edge_at_entry=trade_dict.get("edge_at_entry"),
                station=trade_dict.get("station"),
                market_date=trade_dict.get("market_date"),
                gaussian_prob=trade_dict.get("gaussian_prob"),
                ensemble_prob=trade_dict.get("ensemble_prob"),
                blended_prob=trade_dict.get("blended_prob"),
                raw_model_prob=trade_dict.get("raw_model_prob"),
                cash_capped=bool(trade_dict.get("cash_capped", False)),
                order_type=trade_dict.get("order_type", "market"),
                limit_price=trade_dict.get("limit_price"),
                forecast_timestamp=trade_dict.get("forecast_timestamp"),
                market_price_timestamp=trade_dict.get("market_price_timestamp"),
                status=trade_dict.get("status", "open"),
                regime=trade_dict.get("regime"),
            )
            session.add(db_trade)
            session.flush()
            db_id = db_trade.id

        return db_id

    def _update_trade_in_db(self, db_id: int | None, status: str, pnl: float) -> None:
        """Update an existing trade's status and pnl in the DB."""
        if self._engine is None or db_id is None:
            return

        from weatheredge.db.database import get_session
        from weatheredge.db.models import Trade

        with get_session(self._engine) as session:
            trade = session.get(Trade, db_id)
            if trade:
                trade.status = status
                trade.pnl = pnl
                trade.resolved_at = datetime.now(timezone.utc)

    def _mark_pending_order_filled(self, db_id: int | None, trade_dict: dict) -> int | None:
        """Update a persisted pending limit order into an open position."""
        if self._engine is None or db_id is None:
            return None

        from weatheredge.db.database import get_session
        from weatheredge.db.models import Trade

        with get_session(self._engine) as session:
            trade = session.get(Trade, db_id)
            if trade is None:
                return None
            trade.status = "open"
            trade.price = trade_dict["price"]
            trade.amount = trade_dict["amount"]
            trade.cost = trade_dict["cost"]
            trade.agent_name = trade_dict.get("agent_name")
            trade.model_probability = trade_dict.get("model_probability")
            trade.edge_at_entry = trade_dict.get("edge_at_entry")
            trade.gaussian_prob = trade_dict.get("gaussian_prob")
            trade.ensemble_prob = trade_dict.get("ensemble_prob")
            trade.blended_prob = trade_dict.get("blended_prob")
            trade.raw_model_prob = trade_dict.get("raw_model_prob")
            trade.cash_capped = bool(trade_dict.get("cash_capped", False))
            trade.order_type = "limit"
            trade.limit_price = trade_dict.get("limit_price")
            trade.forecast_timestamp = trade_dict.get("forecast_timestamp")
            trade.market_price_timestamp = trade_dict.get("market_price_timestamp")
        return db_id

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> dict:
        """Place a paper order. Market orders fill immediately; limit orders go to open_orders."""
        from weatheredge.config import MODULES

        # US-TG: Temporal guard — reject trades with lookahead bias
        if MODULES.get("temporal_guard", False):
            from weatheredge.temporal_guard import assert_no_lookahead, TemporalLeakageError, record_leakage_violation
            if order.forecast_timestamp is None or order.market_price_timestamp is None:
                error = (
                    "MISSING TEMPORAL TIMESTAMPS: forecast_timestamp and "
                    "market_price_timestamp are required before trading"
                )
                log.critical("Temporal guard rejected %s: %s", order.market_id, error)
                record_leakage_violation(order.market_id, error, order.__dict__)
                return {"filled": False, "reason": "missing_temporal_timestamps", "error": error}
            try:
                assert_no_lookahead(
                    forecast_timestamp=order.forecast_timestamp,
                    market_price_timestamp=order.market_price_timestamp,
                    trade_decision_timestamp=datetime.now(timezone.utc),
                    forecast_source=order.agent_name,
                    market_id=order.market_id,
                )
            except TemporalLeakageError as exc:
                log.critical("Temporal leakage detected for %s: %s", order.market_id, exc)
                record_leakage_violation(order.market_id, str(exc), order.__dict__)
                return {"filled": False, "reason": "temporal_leakage", "error": str(exc)}

        market_id = self._market_id_key(order.market_id)

        if order.order_type == "limit":
            from weatheredge.config import Settings
            settings = Settings()
            if order.side == "buy_yes":
                limit_price = order.price - settings.limit_order_offset
            else:  # buy_no
                limit_price = (1.0 - order.price) - settings.limit_order_offset

            pending = {
                "market_id": market_id,
                "side": order.side,
                "price": order.price,
                "amount": order.amount,
                "cost": 0.0,
                "limit_price": limit_price,
                "agent_name": order.agent_name,
                "model_probability": order.model_probability,
                "edge_at_entry": order.edge_at_entry,
                "city": order.city,
                "platform": order.platform,
                "station": order.station,
                "market_date": order.market_date,
                "gaussian_prob": order.gaussian_prob,
                "ensemble_prob": order.ensemble_prob,
                "blended_prob": order.blended_prob,
                "raw_model_prob": order.raw_model_prob,
                "cash_capped": order.cash_capped,
                "forecast_timestamp": order.forecast_timestamp,
                "market_price_timestamp": order.market_price_timestamp,
                "order_type": "limit",
                "status": "pending",
                "created_at": datetime.now(timezone.utc),
                "pnl": None,
                "regime": order.regime,
            }
            try:
                db_id = self._persist_new_trade(pending)
            except Exception as exc:
                log.error(
                    "_persist_new_trade failed for pending limit %s: %s — kept in-memory",
                    market_id, exc,
                )
                db_id = None
            pending["db_id"] = db_id
            pending["signal_ids"] = list(order.signal_ids or [])
            self.open_orders.append(pending)
            return {"filled": False, "pending": True, "limit_price": limit_price, "db_id": db_id}

        # BugFix C1: buy_no cost is (1 - yes_price) * amount, not yes_price * amount
        if order.side == "buy_no":
            stake = (1.0 - order.price) * order.amount
        else:
            stake = order.price * order.amount
        # Kalshi taker fee is paid per contract at entry (symmetric in price),
        # so cost captures the real cash outlay and resolve/close PnL stays
        # consistent with (amount - cost) on a win.
        entry_fee = _kalshi_fee(order.price, order.amount, order.platform)
        cost = stake + entry_fee
        if cost > self.bankroll:
            return {"filled": False, "reason": "Insufficient bankroll"}

        self.bankroll -= cost
        trade = {
            "market_id": market_id,
            "side": order.side,
            "price": order.price,
            "amount": order.amount,
            "cost": cost,
            "agent_name": order.agent_name,
            "model_probability": order.model_probability,
            "edge_at_entry": order.edge_at_entry,
            "city": order.city,
            "platform": order.platform,
            "station": order.station,
            "market_date": order.market_date,
            "gaussian_prob": order.gaussian_prob,
            "ensemble_prob": order.ensemble_prob,
            "blended_prob": order.blended_prob,
            "raw_model_prob": order.raw_model_prob,
            "cash_capped": order.cash_capped,
            "forecast_timestamp": order.forecast_timestamp,
            "market_price_timestamp": order.market_price_timestamp,
            "order_type": "market",
            "status": "open",
            "created_at": datetime.now(timezone.utc),
            "pnl": None,
            "regime": order.regime,
        }

        try:
            db_id = self._persist_new_trade(trade)
        except Exception as exc:
            log.error(
                "_persist_new_trade failed for %s: %s — trade kept in-memory",
                market_id, exc,
            )
            db_id = None
        trade["db_id"] = db_id

        self.open_positions.append(trade)
        self._record_daily_entry(market_id, market_date=order.market_date or None)
        return {"filled": True, "trade": trade}

    def check_limit_fills(self, current_prices: dict[str, float]) -> int:
        """Check pending limit orders against current prices and fill qualifying orders.

        For buy_yes: fills when current_prices[market_id] <= limit_price.
        For buy_no: fills when (1 - current_prices[market_id]) <= limit_price.
        Filled orders are moved to open_positions and cost deducted from bankroll.
        Returns the number of orders filled.
        """
        filled_count = 0
        self.last_limit_fills = []
        still_pending = []

        for o in self.open_orders:
            if o.get("status") != "pending":
                still_pending.append(o)
                continue

            market_id = self._market_id_key(o["market_id"])
            current_yes = current_prices.get(market_id)
            if current_yes is None:
                still_pending.append(o)
                continue

            limit_price = o["limit_price"]
            if o["side"] == "buy_yes":
                triggered = current_yes <= limit_price
            else:  # buy_no
                triggered = (1.0 - current_yes) <= limit_price

            if not triggered:
                still_pending.append(o)
                continue

            # Fill at limit_price.
            # For buy_yes the limit_price is on the YES-leg (already canonical).
            # For buy_no the limit_price is on the NO-leg — convert to canonical
            # YES-price before persisting so downstream readers of pos["price"]
            # (exit_manager entry_price, main.py close fallback) see a consistent
            # YES-price across both market and limit fills.
            fill_price = limit_price
            if o["side"] == "buy_no":
                stake = fill_price * o["amount"]
                stored_price = 1.0 - fill_price
                fill_no_price: float | None = fill_price
            else:
                stake = fill_price * o["amount"]
                stored_price = fill_price
                fill_no_price = None

            platform = o.get("platform", "polymarket")
            entry_fee = _kalshi_fee(stored_price, o["amount"], platform)
            cost = stake + entry_fee

            if cost > self.bankroll:
                still_pending.append(o)
                continue

            self.bankroll -= cost
            trade = {
                "market_id": market_id,
                "side": o["side"],
                "price": stored_price,
                "fill_no_price": fill_no_price,
                "amount": o["amount"],
                "cost": cost,
                "agent_name": o.get("agent_name", ""),
                "model_probability": o.get("model_probability", 0.0),
                "edge_at_entry": o.get("edge_at_entry", 0.0),
                "city": o.get("city", ""),
                "platform": o.get("platform", "polymarket"),
                "station": o.get("station", ""),
                "market_date": o.get("market_date", ""),
                "gaussian_prob": o.get("gaussian_prob"),
                "ensemble_prob": o.get("ensemble_prob"),
                "blended_prob": o.get("blended_prob"),
                "raw_model_prob": o.get("raw_model_prob"),
                "cash_capped": bool(o.get("cash_capped", False)),
                "limit_price": limit_price,
                "forecast_timestamp": o.get("forecast_timestamp"),
                "market_price_timestamp": o.get("market_price_timestamp"),
                "order_type": "limit",
                "status": "open",
                "created_at": o.get("created_at") or datetime.now(timezone.utc),
                "pnl": None,
                "regime": o.get("regime"),
            }

            db_id = o.get("db_id")
            if db_id is not None:
                try:
                    db_id = self._mark_pending_order_filled(int(db_id), trade)
                except Exception as exc:
                    log.error(
                        "_mark_pending_order_filled failed for %s: %s — kept in-memory",
                        market_id, exc,
                    )
                    db_id = None
            if db_id is None:
                try:
                    db_id = self._persist_new_trade(trade)
                except Exception as exc:
                    log.error(
                        "_persist_new_trade failed for limit fill %s: %s — kept in-memory",
                        market_id, exc,
                    )
                    db_id = None
            trade["db_id"] = db_id
            trade["signal_ids"] = list(o.get("signal_ids") or [])

            self.open_positions.append(trade)
            self.last_limit_fills.append(trade)
            self._record_daily_entry(market_id, market_date=o.get("market_date") or None)
            filled_count += 1
            log.info(
                "check_limit_fills: filled %s %s at %.4f",
                o["side"], market_id, fill_price,
            )

        self.open_orders = still_pending
        return filled_count

    def cancel_expired_orders(self) -> int:
        """Cancel orders older than ORDER_TTL_HOURS per Addendum Gap 9.

        Returns the number of orders cancelled.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=ORDER_TTL_HOURS)
        expired = []
        still_open = []
        for o in self.open_orders:
            created_at = self._coerce_utc_datetime(o.get("created_at"))
            if created_at is None:
                created_at = now
            o["created_at"] = created_at
            if created_at < cutoff:
                expired.append(o)
            else:
                still_open.append(o)
        for o in expired:
            o["status"] = "cancelled"
            self._update_trade_in_db(o.get("db_id"), "cancelled", 0.0)
            self.trade_history.append(o)
        self.open_orders = still_open
        return len(expired)

    def resolve_position(self, market_id: str, outcome: str) -> None:
        """Resolve a position. outcome = 'yes' or 'no'.

        Hold-to-resolution: positions remain open until market resolves.
        Win: collect $1 per contract minus entry cost.
        Loss: lose entire cost.
        """
        target_market_id = self._market_id_key(market_id)
        for pos in self.open_positions[:]:
            if self._market_id_key(pos.get("market_id")) == target_market_id:
                won = (
                    (pos["side"] == "buy_yes" and outcome == "yes")
                    or (pos["side"] == "buy_no" and outcome == "no")
                )
                if won:
                    # Kalshi/Polymarket settle $0/$1 fee-free; the entry fee
                    # was already booked into cost, so win pnl is simply
                    # amount - cost (what cash actually lands after payout).
                    pnl = pos["amount"] - pos["cost"]
                    pos["status"] = "won"
                else:
                    pnl = -pos["cost"]  # lose entire stake (fee included)
                    pos["status"] = "lost"
                pos["pnl"] = pnl
                self.bankroll += pos["cost"] + pnl  # return cost ± pnl
                # M3 fix: book P&L to the market's resolution date, not today.
                # Prevents cross-midnight resolutions from distorting daily circuit breakers.
                day = pos.get("market_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
                self.daily_pnl[day] = self.daily_pnl.get(day, 0) + pnl
                self._update_trade_in_db(pos.get("db_id"), pos["status"], pnl)
                self.trade_history.append(pos)
                self.open_positions.remove(pos)

    def close_position(self, market_id: str, current_yes_price: float, reason: str = "edge_stop") -> bool:
        """Close ALL open positions for a market early at the current market price.

        BugFix N8: Closes all matching positions, not just the first.
        Computes mark-to-market P&L and records each position as closed with
        the given reason (e.g. 'edge_stop', 'capital_protect', 'time_decay', 'ev_based').
        Returns True if any position was found and closed.
        """
        closed_any = False
        target_market_id = self._market_id_key(market_id)
        for pos in self.open_positions[:]:
            if self._market_id_key(pos.get("market_id")) == target_market_id:
                if pos["side"] == "buy_yes":
                    current_value = current_yes_price * pos["amount"]
                else:  # buy_no
                    current_value = (1.0 - current_yes_price) * pos["amount"]
                # Early close pays a second taker fee at exit price
                # (symmetric in p on Kalshi). Cost already includes entry fee.
                exit_fee = _kalshi_fee(
                    current_yes_price,
                    pos["amount"],
                    pos.get("platform", "polymarket"),
                )
                pnl = current_value - exit_fee - pos["cost"]
                pos["status"] = reason
                pos["pnl"] = pnl
                self.bankroll += pos["cost"] + pnl  # return cost ± pnl
                # PRD M3 consistency: book early-exit P&L to the market's
                # station-local resolution date, not UTC today. Without this, an
                # early exit taken across UTC midnight contaminates the next
                # day's circuit-breaker budget instead of the day the position
                # was actually carried on.
                day = pos.get("market_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
                self.daily_pnl[day] = self.daily_pnl.get(day, 0) + pnl
                self._update_trade_in_db(pos.get("db_id"), reason, pnl)
                self.trade_history.append(pos)
                self.open_positions.remove(pos)
                log.info(
                    "close_position: market=%s reason=%s pnl=%.4f bankroll=%.2f",
                    target_market_id, reason, pnl, self.bankroll,
                )
                closed_any = True
        return closed_any

    def has_open_position(self, market_id: str) -> bool:
        """Return True if an open or pending position exists for the given market."""
        target_market_id = self._market_id_key(market_id)
        return any(
            self._market_id_key(p.get("market_id")) == target_market_id
            for p in self.open_positions
        ) or any(
            self._market_id_key(o.get("market_id")) == target_market_id
            for o in self.open_orders
        )

    def has_reached_daily_entry_limit(
        self,
        market_id: str,
        max_entries: int = 2,
        market_date: str | None = None,
    ) -> bool:
        """Return True if we have already entered this market max_entries times today.

        Prevents re-entry traps where the bot repeatedly enters and gets stopped
        out on the same market within a single day (e.g. 1953479 pattern: 4 entries,
        all edge_stop, cumulative -1.03).

        When ``market_date`` (station-local YYYY-MM-DD) is provided, the counter
        keys on that market's resolution date rather than UTC "today" — so a
        market trading during the overnight UTC window (after UTC midnight but
        before local midnight) still counts against the same local-day bucket.
        """
        day_key = market_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = self._market_id_key(market_id)
        return self._daily_entries.get(key, {}).get(day_key, 0) >= max_entries

    def _record_daily_entry(self, market_id: str, market_date: str | None = None) -> None:
        """Increment the daily entry counter for a market keyed on market_date.

        Falls back to UTC today when market_date is not supplied (legacy callers
        without market-date context).
        """
        day_key = market_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = self._market_id_key(market_id)
        bucket = self._daily_entries.setdefault(key, {})
        bucket[day_key] = bucket.get(day_key, 0) + 1
        log.debug(
            "daily_entry_count: market=%s date=%s count=%d",
            key, day_key, bucket[day_key],
        )

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    def get_performance_report(self) -> PerformanceReport:
        """Calculate performance metrics from trade history.

        Returns zeros for all metrics when no resolved trades exist.

        Includes both resolution outcomes (won/lost) and early exits
        (edge_stop/capital_protect/time_decay/ev_based/etc.) as long as they carry
        realised P&L. Cancelled/abandoned/open entries are excluded.
        """
        resolved = [
            t for t in self.trade_history
            if t.get("status") not in ("open", "cancelled", "abandoned")
            and t.get("pnl") is not None
        ]
        if not resolved:
            return PerformanceReport(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # P&L-sign based wins/losses so early exits are reflected in outcomes.
        wins = [t for t in resolved if (t.get("pnl") or 0.0) > 0.0]
        losses = [t for t in resolved if (t.get("pnl") or 0.0) < 0.0]
        total_pnl = sum(float(t.get("pnl") or 0.0) for t in resolved)
        pnls = [float(t.get("pnl") or 0.0) for t in resolved]

        # Sharpe: annualized from per-trade PnLs (ddof=1 for sample std)
        # Weather markets resolve ~daily; assume ~250 trading days/year for annualization.
        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
        per_trade_sharpe = mean_pnl / std_pnl if std_pnl > 0.0 else 0.0
        sharpe = per_trade_sharpe * np.sqrt(min(len(pnls), 250))

        # Max drawdown as fraction of initial bankroll
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = float(np.max(drawdowns)) / self.initial_bankroll if len(drawdowns) > 0 else 0.0

        edges = [t.get("edge_at_entry", 0.0) for t in resolved]

        return PerformanceReport(
            total_trades=len(resolved),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / len(resolved),
            total_pnl=total_pnl,
            roi=total_pnl / self.initial_bankroll,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            avg_edge_at_entry=float(np.mean(edges)) if edges else 0.0,
        )
