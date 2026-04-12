"""Trading P&L backtester for WeatherEdge — STEP 4.

Downloads historical forecast and observation data, simulates trading
decisions using the same probability engine as the live bot, and calculates
P&L metrics: win rate, ROI, Sharpe ratio, max drawdown, per-city breakdown.

Data sources:
- Open-Meteo Archive API: historical GFS daily high temperature forecasts
- Iowa Environmental Mesonet (IEM): actual observed daily high temperatures
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import aiohttp

from weatheredge.backtesting.market_prices import HistoricalMarketPriceBook
from weatheredge.probability.gaussian import gaussian_probability

log = logging.getLogger(__name__)

# Station coordinates for Open-Meteo queries
STATION_COORDS: dict[str, tuple[float, float]] = {
    "KNYC": (40.7128, -74.0060),
    "KMDW": (41.8781, -87.6298),
    "KMIA": (25.7617, -80.1918),
    "KDEN": (39.7392, -104.9903),
    "KLAX": (33.9425, -118.4081),
}

# IEM station identifiers and network mappings
# Network = {state}_ASOS (state where the station is located)
STATION_IEM: dict[str, tuple[str, str]] = {
    # station: (iem_id, network)
    "KNYC": ("NYC", "NY_ASOS"),
    "KMDW": ("MDW", "IL_ASOS"),
    "KMIA": ("MIA", "FL_ASOS"),
    "KDEN": ("DEN", "CO_ASOS"),
    "KLAX": ("LAX", "CA_ASOS"),
}

# Typical RMSE per station for Gaussian probability (°F)
STATION_RMSE: dict[str, float] = {
    "KNYC": 3.5,
    "KMDW": 4.0,
    "KMIA": 3.0,
    "KDEN": 4.5,
    "KLAX": 3.0,
}

# Fee estimate per round-trip trade (2% proxy as per instructions)
FEE_PER_TRADE = 0.02


@dataclass
class DailyRecord:
    """One day of historical data for a station."""
    station: str
    date: date
    forecast_high_f: float  # GFS predicted daily high (°F)
    actual_high_f: float    # Observed daily high (°F)


@dataclass
class SimulatedTrade:
    """A single simulated trade in the backtest."""
    station: str
    trade_date: date
    threshold: float
    direction: str           # "above" or "below"
    model_prob: float
    market_proxy: float      # Climatological probability as market proxy
    side: str                # "buy_yes" or "buy_no"
    entry_price: float
    won: bool
    pnl: float               # Net P&L after fees


@dataclass
class BacktestMetrics:
    """Aggregate metrics from a backtest run."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    gross_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    roi_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    per_city: dict[str, dict] = field(default_factory=dict)
    edge_distribution: list[float] = field(default_factory=list)


class HistoricalDataFetcher:
    """Downloads historical forecast and observation data."""

    async def fetch_archive_forecasts(
        self,
        station: str,
        start_date: date,
        end_date: date,
        session: aiohttp.ClientSession,
    ) -> dict[date, float]:
        """Fetch historical GFS daily max temperature from Open-Meteo Archive API.

        Returns dict of date -> forecast_high_fahrenheit.
        """
        coords = STATION_COORDS.get(station)
        if not coords:
            log.warning("No coords for station %s", station)
            return {}

        lat, lon = coords
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "timezone": "UTC",
        }

        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    log.warning("Open-Meteo Archive returned %d for %s", resp.status, station)
                    return {}
                data = await resp.json()
        except Exception as exc:
            log.warning("Open-Meteo Archive fetch failed for %s: %s", station, exc)
            return {}

        daily = data.get("daily", {})
        times = daily.get("time", [])
        temps = daily.get("temperature_2m_max", [])

        result: dict[date, float] = {}
        for t, temp in zip(times, temps):
            if temp is not None:
                d = date.fromisoformat(t)
                result[d] = temp
        log.info("Fetched %d archive forecast days for %s", len(result), station)
        return result

    async def fetch_iem_observations(
        self,
        station: str,
        start_date: date,
        end_date: date,
        session: aiohttp.ClientSession,
    ) -> dict[date, float]:
        """Fetch actual observed daily max temperature from Iowa Mesonet.

        Returns dict of date -> actual_high_fahrenheit.
        Uses the IEM JSON API endpoint.
        """
        iem_entry = STATION_IEM.get(station)
        if not iem_entry:
            log.warning("No IEM mapping for station %s", station)
            return {}

        iem_id, network = iem_entry

        # Use the IEM JSON API (returns all data; we filter client-side by date)
        url = "https://mesonet.agron.iastate.edu/api/1/daily.json"
        params = {
            "station": iem_id,
            "network": network,
        }

        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    log.warning("IEM API returned %d for %s", resp.status, station)
                    return {}
                data = await resp.json(content_type=None)  # Accept any content type
        except Exception as exc:
            log.warning("IEM API fetch failed for %s: %s", station, exc)
            return {}

        result: dict[date, float] = {}
        # API returns {"data": [...]} with objects containing "date" and "max_tmpf"
        rows = data.get("data", []) if isinstance(data, dict) else data
        for row in rows:
            try:
                d_str = row.get("date") or row.get("day") or row.get("valid")
                max_tmpf = row.get("max_tmpf")
                if d_str and max_tmpf is not None:
                    d = date.fromisoformat(str(d_str)[:10])
                    if start_date <= d <= end_date:
                        result[d] = float(max_tmpf)
            except (ValueError, TypeError):
                continue
        log.info("Fetched %d IEM observation days for %s", len(result), station)
        return result


class TradingBacktester:
    """Simulates trading decisions on historical data and calculates P&L."""

    def __init__(
        self,
        min_edge: float = 0.10,
        max_edge: float = 0.50,
        min_price: float = 0.10,
        bet_size: float = 10.0,
        fee_rate: float = FEE_PER_TRADE,
        max_trades_per_station_day: int = 3,
        require_real_market_price: bool = False,
    ):
        self.min_edge = min_edge
        self.max_edge = max_edge
        self.min_price = min_price
        self.bet_size = bet_size
        self.fee_rate = fee_rate
        # Risk realism: cap daily opportunities consumed per station/day.
        self.max_trades_per_station_day = max_trades_per_station_day
        # When True and a market_price_book is provided, skip trades without
        # a real historical price instead of using climatology fallback.
        self.require_real_market_price = require_real_market_price

    def build_daily_records(
        self,
        forecasts: dict[date, float],
        observations: dict[date, float],
        station: str,
    ) -> list[DailyRecord]:
        """Join forecast and observation data into DailyRecords."""
        records = []
        for d in sorted(forecasts.keys()):
            if d in observations:
                records.append(DailyRecord(
                    station=station,
                    date=d,
                    forecast_high_f=forecasts[d],
                    actual_high_f=observations[d],
                ))
        return records

    def compute_climatological_prob(
        self,
        records: list[DailyRecord],
        threshold: float,
        direction: str,
        window_days: int = 30,
        current_idx: int = 0,
    ) -> float:
        """Compute climatological probability as market proxy.

        Uses a rolling window of historical data (prior to current day)
        to estimate what fraction of days the temp exceeded the threshold.
        """
        # Use all records before current_idx as historical reference
        lookback = records[max(0, current_idx - window_days):current_idx]
        if len(lookback) < 5:
            return 0.50  # Not enough history, assume 50/50

        if direction == "above":
            # Match live resolution convention: "above" includes equality.
            count = sum(1 for r in lookback if r.actual_high_f >= threshold)
        else:
            count = sum(1 for r in lookback if r.actual_high_f < threshold)
        return count / len(lookback)

    @staticmethod
    def _default_thresholds(records: list[DailyRecord]) -> list[float]:
        """Build default threshold ladder from forecast range.

        Uses forecast highs (not realised highs) to avoid lookahead leakage.
        """
        import math

        if not records:
            return []
        temps = [r.forecast_high_f for r in records]
        # Snap to the same 5°F grid used by market builders so exact-key joins work.
        min_t = int(math.floor((min(temps) - 5.0) / 5.0) * 5)
        max_t = int(math.ceil((max(temps) + 5.0) / 5.0) * 5)
        if min_t > max_t:
            return []
        return list(range(min_t, max_t + 1, 5))

    def simulate_trading(
        self,
        records: list[DailyRecord],
        thresholds: list[float] | None = None,
        market_price_book: HistoricalMarketPriceBook | None = None,
    ) -> list[SimulatedTrade]:
        """Simulate trading on historical data.

        For each day and threshold, calculates model probability vs
        market proxy, and trades when edge exceeds minimum.

        If ``market_price_book`` is provided, uses real historical YES prices
        for (station, date, threshold, direction) when available; otherwise
        falls back to the climatological proxy.
        """
        if not records:
            return []

        station = records[0].station
        rmse = STATION_RMSE.get(station, 4.0)

        # Generate thresholds from data range if not provided
        if thresholds is None:
            thresholds = self._default_thresholds(records)
        if not thresholds:
            return []

        trades: list[SimulatedTrade] = []

        for i, record in enumerate(records):
            if i < 30:  # Need at least 30 days of history for climatological proxy
                continue

            # Keep only the strongest representation for each threshold/day.
            # This avoids double-counting complementary above/below contracts.
            day_best_by_threshold: dict[float, tuple[float, SimulatedTrade]] = {}

            for threshold in thresholds:
                for direction in ["above", "below"]:
                    # Model probability from Gaussian CDF
                    model_prob = gaussian_probability(
                        record.forecast_high_f, threshold, rmse, direction
                    )

                    # Market proxy: climatological probability
                    market_proxy = None
                    if market_price_book is not None:
                        market_proxy = market_price_book.get_yes_price(
                            station=station,
                            trade_date=record.date,
                            threshold=threshold,
                            direction=direction,
                        )
                        if market_proxy is None and self.require_real_market_price:
                            continue

                    if market_proxy is None:
                        market_proxy = self.compute_climatological_prob(
                            records, threshold, direction, window_days=60, current_idx=i,
                        )

                    # Clamp market proxy to valid range
                    market_proxy = max(0.01, min(0.99, market_proxy))

                    # Calculate edge
                    gross_edge = model_prob - market_proxy

                    # Apply filters (same as live bot)
                    if abs(gross_edge) > self.max_edge:
                        continue  # Unrealistic edge

                    # Determine trade side
                    if gross_edge >= self.min_edge:
                        side = "buy_yes"
                        entry_price = market_proxy
                    elif gross_edge <= -self.min_edge:
                        side = "buy_no"
                        entry_price = 1.0 - market_proxy
                    else:
                        continue  # Not enough edge

                    # Price floor applies to whichever side we actually buy.
                    if entry_price < self.min_price:
                        continue

                    # Determine outcome
                    # BugFix N9: Match live ensemble logic (>= for above, < for below)
                    if direction == "above":
                        event_occurred = record.actual_high_f >= threshold
                    else:
                        event_occurred = record.actual_high_f < threshold

                    if side == "buy_yes":
                        won = event_occurred
                    else:
                        won = not event_occurred

                    # P&L calculation
                    if won:
                        gross_pnl = (1.0 - entry_price) * self.bet_size
                    else:
                        gross_pnl = -entry_price * self.bet_size

                    # Apply fee only on winnings (live paper_trader behavior).
                    fee = self.fee_rate * gross_pnl if won else 0.0
                    net_pnl = gross_pnl - fee

                    trade = SimulatedTrade(
                        station=station,
                        trade_date=record.date,
                        threshold=threshold,
                        direction=direction,
                        model_prob=model_prob,
                        market_proxy=market_proxy,
                        side=side,
                        entry_price=entry_price,
                        won=won,
                        pnl=net_pnl,
                    )

                    edge_score = abs(gross_edge)
                    current = day_best_by_threshold.get(float(threshold))
                    if current is None or edge_score > current[0]:
                        day_best_by_threshold[float(threshold)] = (edge_score, trade)

            if not day_best_by_threshold:
                continue

            selected = sorted(
                day_best_by_threshold.values(),
                key=lambda item: item[0],
                reverse=True,
            )
            if self.max_trades_per_station_day > 0:
                selected = selected[: self.max_trades_per_station_day]
            trades.extend(trade for _, trade in selected)

        return trades

    def calculate_metrics(self, trades: list[SimulatedTrade]) -> BacktestMetrics:
        """Calculate aggregate metrics from simulated trades."""
        if not trades:
            return BacktestMetrics()

        wins = sum(1 for t in trades if t.won)
        losses = len(trades) - wins
        gross_pnl = sum(
            ((1.0 - t.entry_price) if t.won else -t.entry_price) * self.bet_size
            for t in trades
        )
        # Keep aggregate metrics consistent with per-trade simulated pnl values.
        net_pnl = sum(float(t.pnl) for t in trades)
        total_fees = max(0.0, gross_pnl - net_pnl)
        total_risked = sum(t.entry_price * self.bet_size for t in trades)
        roi_pct = (net_pnl / total_risked * 100) if total_risked > 0 else 0.0

        # Sharpe ratio (daily P&L series)
        daily_pnls: dict[date, float] = {}
        for t in trades:
            daily_pnls[t.trade_date] = daily_pnls.get(t.trade_date, 0.0) + t.pnl
        pnl_series = list(daily_pnls.values())
        if len(pnl_series) > 1:
            mean_pnl = sum(pnl_series) / len(pnl_series)
            std_pnl = math.sqrt(sum((p - mean_pnl) ** 2 for p in pnl_series) / (len(pnl_series) - 1))
            sharpe = (mean_pnl / std_pnl * math.sqrt(252)) if std_pnl > 0 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for d in sorted(daily_pnls.keys()):
            cumulative += daily_pnls[d]
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        # Per-city breakdown
        per_city: dict[str, dict] = {}
        for t in trades:
            city = t.station
            if city not in per_city:
                per_city[city] = {"trades": 0, "wins": 0, "pnl": 0.0}
            per_city[city]["trades"] += 1
            if t.won:
                per_city[city]["wins"] += 1
            per_city[city]["pnl"] += t.pnl
        for city_data in per_city.values():
            city_data["win_rate"] = city_data["wins"] / city_data["trades"] if city_data["trades"] > 0 else 0.0

        # Edge distribution
        edges = [abs(t.model_prob - t.market_proxy) for t in trades]

        return BacktestMetrics(
            total_trades=len(trades),
            wins=wins,
            losses=losses,
            win_rate=wins / len(trades),
            gross_pnl=gross_pnl,
            total_fees=total_fees,
            net_pnl=net_pnl,
            roi_pct=roi_pct,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            per_city=per_city,
            edge_distribution=edges,
        )

    def save_trades_csv(self, trades: list[SimulatedTrade], path: Path) -> None:
        """Save simulated trades to CSV."""
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "station", "date", "threshold", "direction",
                "model_prob", "market_proxy", "side", "entry_price",
                "won", "pnl",
            ])
            for t in trades:
                writer.writerow([
                    t.station, t.trade_date.isoformat(), t.threshold,
                    t.direction, f"{t.model_prob:.4f}", f"{t.market_proxy:.4f}",
                    t.side, f"{t.entry_price:.4f}",
                    t.won, f"{t.pnl:.2f}",
                ])
        log.info("Saved %d trades to %s", len(trades), path)

    def print_report(self, metrics: BacktestMetrics) -> str:
        """Print a formatted backtest report. Returns the report string."""
        lines = [
            "=" * 60,
            "WEATHEREDGE TRADING BACKTEST REPORT",
            "=" * 60,
            f"Total trades:    {metrics.total_trades}",
            f"Wins:            {metrics.wins}",
            f"Losses:          {metrics.losses}",
            f"Win rate:        {metrics.win_rate:.1%}",
            f"Gross P&L:       ${metrics.gross_pnl:.2f}",
            f"Total fees:      ${metrics.total_fees:.2f}",
            f"Net P&L:         ${metrics.net_pnl:.2f}",
            f"ROI:             {metrics.roi_pct:.1f}%",
            f"Sharpe ratio:    {metrics.sharpe_ratio:.2f}",
            f"Max drawdown:    ${metrics.max_drawdown:.2f}",
            "",
            "PER-CITY BREAKDOWN:",
            "-" * 40,
        ]
        for city, data in sorted(metrics.per_city.items()):
            lines.append(
                f"  {city}: {data['trades']} trades, "
                f"WR={data['win_rate']:.1%}, "
                f"P&L=${data['pnl']:.2f}"
            )

        if metrics.edge_distribution:
            avg_edge = sum(metrics.edge_distribution) / len(metrics.edge_distribution)
            lines.extend([
                "",
                f"Avg edge:        {avg_edge:.1%}",
                f"Edge range:      {min(metrics.edge_distribution):.1%} - {max(metrics.edge_distribution):.1%}",
            ])

        lines.append("=" * 60)
        report = "\n".join(lines)
        print(report)
        return report


async def run_full_backtest(
    stations: list[str] | None = None,
    months_back: int = 6,
    min_edge: float = 0.10,
    market_prices_csv: str | Path | None = None,
) -> BacktestMetrics:
    """Run a full backtest across all stations.

    Downloads historical data, simulates trading, and returns metrics.
    """
    if stations is None:
        stations = list(STATION_COORDS.keys())

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=months_back * 30)

    fetcher = HistoricalDataFetcher()
    backtester = TradingBacktester(min_edge=min_edge)
    all_trades: list[SimulatedTrade] = []
    market_price_book: HistoricalMarketPriceBook | None = None

    if market_prices_csv is not None:
        try:
            market_price_book = HistoricalMarketPriceBook.from_csv(market_prices_csv)
            log.info("Loaded historical market prices from %s (%d keyed entries)", market_prices_csv, len(market_price_book))
        except Exception as exc:
            log.warning("Failed to load market_prices_csv=%s: %s", market_prices_csv, exc)
            market_price_book = None

    async with aiohttp.ClientSession() as session:
        for station in stations:
            log.info("Fetching historical data for %s (%s to %s)", station, start, end)
            forecasts = await fetcher.fetch_archive_forecasts(station, start, end, session)
            observations = await fetcher.fetch_iem_observations(station, start, end, session)

            records = backtester.build_daily_records(forecasts, observations, station)
            log.info("Built %d daily records for %s", len(records), station)

            if len(records) < 60:
                log.warning("Not enough data for %s (%d records, need 60+)", station, len(records))
                continue

            trades = backtester.simulate_trading(records, market_price_book=market_price_book)
            log.info("Simulated %d trades for %s", len(trades), station)
            all_trades.extend(trades)

    metrics = backtester.calculate_metrics(all_trades)

    # Save CSV
    output_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    output_dir.mkdir(exist_ok=True)
    backtester.save_trades_csv(all_trades, output_dir / "backtest_results.csv")

    # Print report
    backtester.print_report(metrics)

    return metrics
