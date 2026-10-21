# WeatherEdge (WeatherFS)

Autonomous trading bot for weather-based prediction markets (Polymarket, Kalshi).

It forecasts weather outcomes for target cities, compares them against live market prices, and places trades when the model's edge over the market exceeds a configurable threshold.

## Highlights

- Multiple forecasting agents (conservative, aggressive, mean-reversion, qualitative) combined into a blended probability estimate
- Tiered risk management: position sizing via fractional Kelly, volatility targeting, per-market and per-platform exposure caps
- Backtesting and walk-forward simulation against historical market prices
- Paper-trading mode by default; live trading gated behind configurable performance thresholds (min trades, win rate, Sharpe, max drawdown)
- Telegram notifications for trade decisions and daily summaries

## Status

Active development. Core pipeline (forecast → edge → sizing → execution) is implemented and covered by a layered test suite (`tests/`, levels L0–L4: smoke, unit, integration, e2e, validation).

## Setup

```bash
cp .env.example .env   # fill in your API keys
pip install -e .
./launcher.sh
```

See `.env.example` for all configuration options (trading parameters, risk tiers, target cities, scheduling).

## License

MIT — see [LICENSE](LICENSE).
