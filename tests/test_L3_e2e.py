import pytest
from unittest.mock import MagicMock
from weatheredge.agents.base import Signal, MarketOpportunity
from weatheredge.agents.strategies.conservative import conservative_strategy
from weatheredge.agents.strategies.aggressive import aggressive_strategy
from weatheredge.agents.strategies.mean_reversion import mean_reversion_strategy
from weatheredge.agents.portfolio_manager import PortfolioManager
from weatheredge.agents.risk_manager import RiskManager
from weatheredge.execution.paper_trader import PaperTrader, Order
from weatheredge.notifications.telegram import TelegramBot
from weatheredge.config import Settings

def make_settings():
    s = MagicMock(spec=Settings)
    s.conservative_min_edge = 0.20
    s.aggressive_min_edge = 0.08
    s.mean_reversion_lookback_hours = 168
    s.mean_reversion_min_sigma = 3.0
    s.kelly_fraction = 0.15
    s.target_volatility = 0.10
    s.max_leverage = 2.0
    s.min_allocation = 0.20
    s.max_open_positions = 20
    s.max_single_market_exposure = 0.08
    s.max_single_platform_exposure = 0.60
    s.max_single_region_exposure = 0.30
    s.vpin_danger_threshold = 0.70
    s.current_risk_tier = 100
    s.risk_params = {"max_total_drawdown": 0.15, "max_daily_loss": 0.10}
    s.openrouter_api_key = ""
    s.enable_qualitative_agent = False
    s.telegram_bot_token = ""
    s.telegram_chat_id = ""
    return s

@pytest.mark.L3
class TestL3EndToEnd:
    def test_full_decision_cycle(self):
        """Complete pipeline: opportunity -> strategies -> portfolio manager -> paper trade."""
        settings = make_settings()
        risk_manager = RiskManager(settings)
        portfolio_manager = PortfolioManager(settings, risk_manager)
        paper_trader = PaperTrader(initial_bankroll=1000.0)

        # Build opportunity with strong edge (should trigger Conservative)
        opp = MarketOpportunity(
            market_id="KXHIGHNY-2026-03-30",
            platform="kalshi", city="NYC", station="KNYC",
            threshold=80.0, direction="above",
            market_price=0.40, model_probability=0.75,
            net_edge=0.32,  # Big edge
            forecast_temp=88.0, rmse=2.5, lead_time_days=1,
            gaussian_prob=0.75, ensemble_prob=0.74,
        )

        # Run strategies
        conservative_sig = conservative_strategy(opp, settings)
        aggressive_sigs = aggressive_strategy(opp, [], settings)

        # Collect signals by market (Conservative + Aggressive + manual Qualitative to hit threshold)
        from weatheredge.agents.base import Signal
        all_signals = {"Conservative": [], "Aggressive": [], "Qualitative": []}
        if conservative_sig.side != "no_trade":
            all_signals["Conservative"].append(conservative_sig)
        for sig in aggressive_sigs:
            all_signals["Aggressive"].append(sig)
        # Add a qualitative signal to ensure weighted score crosses threshold with 5-agent weights
        all_signals["Qualitative"].append(
            Signal(market_id=opp.market_id, side="buy_yes", confidence=0.90, agent_name="Qualitative")
        )

        # Portfolio manager aggregates
        signals_by_market = {}
        for agent_name, sigs in all_signals.items():
            for sig in sigs:
                if sig.market_id not in signals_by_market:
                    signals_by_market[sig.market_id] = []
                signals_by_market[sig.market_id].append(sig)

        orders = portfolio_manager.aggregate_signals(
            signals_by_market,
            {opp.market_id: opp},
            bankroll=1000.0,
            open_positions=[],
            daily_pnl=0.0,
            total_pnl=0.0,
        )

        # Should generate at least 1 order (strong edge, both agents agree)
        assert len(orders) >= 1, f"Expected orders, got {orders}"

        # Execute paper trade
        order = orders[0]
        paper_order = Order(
            market_id=order.market_id,
            side=order.side,
            price=opp.market_price,
            amount=order.amount,
            agent_name="Portfolio",
            model_probability=opp.model_probability,
            edge_at_entry=opp.net_edge,
            city=opp.city,
        )
        result = paper_trader.place_order(paper_order)
        assert result["filled"] is True
        assert len(paper_trader.open_positions) == 1

    def test_decision_cycle_with_resolution(self):
        """Place trade, resolve it, verify P&L and performance report."""
        paper_trader = PaperTrader(initial_bankroll=1000.0)
        order = Order(market_id="MKT-1", side="buy_yes", price=0.40, amount=100.0,
                      model_probability=0.75, edge_at_entry=0.32, city="NYC")
        paper_trader.place_order(order)
        assert paper_trader.bankroll < 1000.0  # Cost deducted

        # Resolve as win
        paper_trader.resolve_position("MKT-1", "yes")
        assert len(paper_trader.open_positions) == 0
        assert len(paper_trader.trade_history) == 1
        assert paper_trader.trade_history[0]["status"] == "won"
        assert paper_trader.trade_history[0]["pnl"] > 0

        # Performance report
        report = paper_trader.get_performance_report()
        assert report.total_trades == 1
        assert report.win_rate == 1.0
        assert report.total_pnl > 0

    def test_risk_veto_blocks_trade(self):
        """City exposure at 16% cap -> no new trade for that city."""
        settings = make_settings()
        risk_manager = RiskManager(settings)

        # Simulate existing exposure: $160 in NYC on $1000 bankroll = 16%
        class FakePosition:
            def __init__(self, city, cost):
                self.city = city
                self.cost = cost
        existing = [FakePosition("NYC", 80.0), FakePosition("NYC", 80.0)]

        check = risk_manager.check_city_exposure("NYC", 10.0, existing, 1000.0)
        assert check["blocked"] is True
        assert "16%" in check["reason"]

    def test_telegram_alert_formatted(self):
        """Trade alert format produces non-empty string with market data."""
        settings = make_settings()
        bot = TelegramBot(settings)
        alert = bot.format_trade_alert({
            "side": "buy_yes", "amount": 50, "price": 0.40,
            "market": "NYC High Temp >80F", "edge": 32,
        })
        assert len(alert) > 0
        assert "buy_yes" in alert or "TRADE" in alert
