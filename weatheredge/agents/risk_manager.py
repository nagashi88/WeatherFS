"""Risk Manager for WeatherEdge.

Implements:
- Corrected Kelly Criterion (Addendum Gap 1): b derived from market_price, not win_probability
- Volatility targeting (PRD Section 5.7)
- Tiered risk scaling (Addendum Gap 8)
- Same-city correlation grouping 16% cap (Addendum Gap 11)
- VPIN toxic flow detection (PRD Section 5.7)
- Regional exposure caps (US-203): 30% per geographic region
- Slippage modeling (US-205): volume-based Kelly scaling and slippage cap
"""

from __future__ import annotations

from weatheredge.config import Settings

REGION_MAP: dict[str, list[str]] = {
    "northeast": ["KNYC", "KBOS"],
    "southeast": ["KMIA", "KATL", "KIAH"],
    "midwest": ["KMDW", "KDFW", "KDEN"],
    "west": ["KLAX", "KSFO", "KSEA", "KPHX", "KAUS"],
    "international": ["EGLC", "SAEZ", "LTAC", "RKSS"],
}


def station_to_region(station: str) -> str:
    for region, stations in REGION_MAP.items():
        if station in stations:
            return region
    return "unknown"


class RiskManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._resolved_trade_count: int = 0

    def set_resolved_trade_count(self, count: int) -> None:
        """Update resolved trade count for warm-up Kelly scaling."""
        self._resolved_trade_count = count

    def _effective_kelly_fraction(self) -> float:
        """Return Kelly fraction — reduced during warm-up period."""
        min_trades = getattr(self.settings, 'min_trades_for_full_kelly', 50)
        warmup_frac = getattr(self.settings, 'warmup_kelly_fraction', 0.10)
        if self._resolved_trade_count < min_trades:
            return warmup_frac
        return self.settings.kelly_fraction

    def calculate_position_size(
        self,
        win_probability: float,
        market_price: float,
        bankroll: float,
        contract_vol: float | None = None,
        side: str = "buy_yes",
        volume: float | None = None,
    ) -> float:
        """Two-layer sizing per addendum Gap 1.

        Layer 1: Corrected Kelly — b = (1-market_price)/market_price
        Layer 2: Volatility targeting (if vol data available)
        Layer 3: Slippage modeling (US-205) — volume-based scaling and cap
        Final = min(kelly_size, vol_size, hard_caps)

        For buy_no trades, flip to NO-side probabilities:
        win_prob_no = 1 - win_probability, price_no = 1 - market_price.
        """
        # Flip to NO-side perspective for buy_no trades
        if side == "buy_no":
            win_probability = 1.0 - win_probability
            market_price = 1.0 - market_price

        if win_probability <= market_price:
            return 0.0

        if market_price <= 0.0:
            return 0.0

        # Layer 1: Corrected Kelly
        b = (1.0 - market_price) / market_price
        p = win_probability
        q = 1.0 - p
        kelly = (b * p - q) / b
        kelly = max(0.0, kelly)
        kelly_size = kelly * self._effective_kelly_fraction() * bankroll

        # Layer 2: Volatility targeting
        if contract_vol and contract_vol > 0:
            raw_leverage = self.settings.target_volatility / contract_vol
            leverage = min(raw_leverage, self.settings.max_leverage)
            leverage = max(leverage, self.settings.min_allocation)
            vol_size = leverage * bankroll * (1 / self.settings.max_open_positions)
        else:
            vol_size = kelly_size

        # Layer 3: Slippage modeling (US-205)
        if volume is not None:
            if volume < 100:
                kelly_size *= 0.25
                vol_size *= 0.25
            elif volume < 1000:
                kelly_size *= 0.50
                vol_size *= 0.50

        position = min(kelly_size, vol_size)
        position = min(position, bankroll * self.settings.max_single_market_exposure)

        # BugFix P0-2: Cap dollar cost per trade.
        # position * market_price = dollar cost (market_price is already the "bet side" price
        # after any buy_no flip, so cost = contracts * no_price or contracts * yes_price).
        max_cost = bankroll * getattr(self.settings, "max_cost_per_trade_fraction", 0.025)
        cost = position * market_price
        if cost > max_cost and market_price > 0:
            position = max_cost / market_price

        # Slippage cap: estimated_slippage must be < edge * 0.5
        if volume is not None:
            edge = win_probability - market_price
            if edge > 0:
                estimated_slippage = (position * 0.005) / max(volume, 1)
                max_slippage = edge * 0.5
                if estimated_slippage > max_slippage:
                    # Cap position so slippage == max_slippage (never grow position)
                    position = min(position, (max_slippage * max(volume, 1)) / 0.005)

        # Minimum trade cost floor (€100 bankroll Option 2):
        # Kelly on small bankrolls produces sub-€1 trades below Polymarket's practical
        # minimum. Bump approved trades up to the floor. Upstream filters
        # (MIN_EDGE_PERCENT, conservative_min_edge 22%, aggressive_min_edge 12%) already
        # block weak edges; this safety threshold only catches catastrophic overbets
        # where Kelly wanted <20% of the floor — meaning Kelly itself didn't think the
        # trade was worth meaningful capital.
        min_cost = getattr(self.settings, "min_trade_cost", 0.0)
        if position > 0 and min_cost > 0 and market_price > 0:
            current_cost = position * market_price
            if current_cost < min_cost:
                if current_cost < min_cost * 0.2:
                    return 0.0
                if min_cost <= max_cost:
                    position = min_cost / market_price

        return round(position, 2)

    def calculate_vpin(self, recent_fills: list[dict]) -> float:
        """VPIN = |V_buy - V_sell| / (V_buy + V_sell)"""
        v_buy = sum(f["size"] for f in recent_fills if f["side"] == "buy")
        v_sell = sum(f["size"] for f in recent_fills if f["side"] == "sell")
        total = v_buy + v_sell
        if total == 0:
            return 0.0
        return abs(v_buy - v_sell) / total

    def check_circuit_breakers(
        self, daily_pnl: float, total_pnl: float, bankroll: float
    ) -> dict:
        """Tiered risk per addendum Gap 8."""
        risk = self.settings.risk_params  # Returns tier-appropriate params
        if total_pnl < 0 and abs(total_pnl) > bankroll * risk["max_total_drawdown"]:
            return {
                "halt": True,
                "reason": f"KILL SWITCH: Total drawdown exceeded -{risk['max_total_drawdown']*100:.0f}%",
                "severity": "critical",
            }
        if daily_pnl < 0 and abs(daily_pnl) > bankroll * risk["max_daily_loss"]:
            return {
                "halt": True,
                "reason": f"Daily loss limit exceeded -{risk['max_daily_loss']*100:.0f}%",
                "severity": "high",
            }
        return {"halt": False}

    def check_city_exposure(
        self,
        city: str,
        proposed_size: float,
        open_positions: list,
        bankroll: float,
    ) -> dict:
        """Per addendum Gap 11: 16% per-city cap."""
        # BugFix C3: open_positions are dicts, not objects — use .get() not getattr()
        current = sum(
            (t.get("cost", 0) if isinstance(t, dict) else getattr(t, "cost", 0))
            for t in open_positions
            if (t.get("city") if isinstance(t, dict) else getattr(t, "city", None)) == city
        )
        total = current + proposed_size
        max_allowed = bankroll * 0.16
        if total > max_allowed:
            return {
                "blocked": True,
                "reason": f"City {city} exposure {total:.2f} exceeds 16% cap ({max_allowed:.2f})",
            }
        return {"blocked": False}

    def check_region_exposure(
        self,
        station: str,
        proposed_size: float,
        open_positions: list,
        bankroll: float,
    ) -> dict:
        """Per US-203: 30% per-region cap across stations in the same geographic region."""
        region = station_to_region(station)
        current = sum(
            (t.get("cost", 0) if isinstance(t, dict) else getattr(t, "cost", 0))
            for t in open_positions
            if station_to_region(
                t.get("station", "") if isinstance(t, dict) else getattr(t, "station", "")
            ) == region
        )
        total = current + proposed_size
        max_allowed = bankroll * self.settings.max_single_region_exposure
        if total > max_allowed:
            return {
                "blocked": True,
                "reason": (
                    f"Region {region} exposure {total:.2f} exceeds "
                    f"{self.settings.max_single_region_exposure:.0%} cap ({max_allowed:.2f})"
                ),
            }
        return {"blocked": False}

    def check_adverse_selection(self, recent_fills: list[dict]) -> dict:
        """VPIN toxic flow detection."""
        vpin = self.calculate_vpin(recent_fills)
        if vpin > self.settings.vpin_danger_threshold:
            return {"toxic": True, "vpin": vpin, "action": "REDUCE_SIZE_50_PCT"}
        return {"toxic": False, "vpin": vpin}

    def check_date_exposure(
        self,
        market_date: str,
        proposed_size: float,
        open_positions: list,
        bankroll: float,
    ) -> dict:
        """Per-date exposure cap: prevent correlated losses from a single weather event."""
        if not market_date:
            return {"blocked": False}
        current = sum(
            (t.get("cost", 0) if isinstance(t, dict) else getattr(t, "cost", 0))
            for t in open_positions
            if (t.get("market_date") if isinstance(t, dict) else getattr(t, "market_date", None)) == market_date
        )
        total = current + proposed_size
        max_allowed = bankroll * self.settings.max_single_date_exposure
        if total > max_allowed:
            return {
                "blocked": True,
                "reason": f"Date {market_date} exposure {total:.2f} exceeds {self.settings.max_single_date_exposure:.0%} cap ({max_allowed:.2f})",
            }
        return {"blocked": False}

    def pre_trade_check(
        self,
        city: str,
        proposed_size: float,
        open_positions: list,
        bankroll: float,
        daily_pnl: float,
        total_pnl: float,
        recent_fills: list[dict] | None = None,
        market_date: str = "",
        station: str = "",
    ) -> dict:
        """Combined pre-trade risk check."""
        def _open_market_count(positions: list) -> int:
            keys: set[str] = set()
            for idx, pos in enumerate(positions):
                if isinstance(pos, dict):
                    raw_market_id = pos.get("market_id")
                else:
                    raw_market_id = getattr(pos, "market_id", None)
                if raw_market_id is None:
                    keys.add(f"__row__{idx}")
                    continue
                market_id = str(raw_market_id).strip()
                if not market_id:
                    keys.add(f"__row__{idx}")
                    continue
                keys.add(market_id)
            return len(keys)

        # Circuit breakers
        cb = self.check_circuit_breakers(daily_pnl, total_pnl, bankroll)
        if cb["halt"]:
            return {"blocked": True, "reason": cb["reason"]}

        # Position count
        open_market_count = _open_market_count(open_positions)
        if open_market_count >= self.settings.max_open_positions:
            return {
                "blocked": True,
                "reason": (
                    f"Max open positions ({self.settings.max_open_positions}) reached "
                    f"(unique markets={open_market_count})"
                ),
            }

        # City exposure
        city_check = self.check_city_exposure(city, proposed_size, open_positions, bankroll)
        if city_check["blocked"]:
            return {"blocked": True, "reason": city_check["reason"]}

        # Regional exposure cap (US-203)
        if station:
            region_check = self.check_region_exposure(station, proposed_size, open_positions, bankroll)
            if region_check["blocked"]:
                return {"blocked": True, "reason": region_check["reason"]}

        # Date exposure — prevent correlated losses from a single weather event
        if market_date:
            date_check = self.check_date_exposure(market_date, proposed_size, open_positions, bankroll)
            if date_check["blocked"]:
                return {"blocked": True, "reason": date_check["reason"]}

        # VPIN (if fill data available)
        if recent_fills:
            vpin_check = self.check_adverse_selection(recent_fills)
            if vpin_check["toxic"]:
                return {
                    "blocked": False,
                    "reduce_size": True,
                    "reason": f"VPIN={vpin_check['vpin']:.2f} — reduce size 50%",
                }

        return {"blocked": False}
