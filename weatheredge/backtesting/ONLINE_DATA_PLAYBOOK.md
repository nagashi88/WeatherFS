# Online Data Playbook (Backtest Inputs)

Date checked: 2026-04-01

## Local status

Current workspace scan found **no local raw dataset files** (`.parquet`, `.tar.zst`, `.xz`, `.duckdb`), so real market-history inputs are not present yet.

## New local fast path (recommended first)

Use the bot DB snapshots directly:

```bash
cd /Users/naka/Documents/Weather
.venv/bin/python -m weatheredge.backtesting.db_market_price_builder \
  --db /Users/naka/Documents/Weather/weatheredge.db \
  --output /Users/naka/Documents/Weather/logs/localdb_market_prices.csv
```

Notes:
- Works best once `markets` metadata is populated by recent `fetch_market_data` runs.
- If your DB only has very recent open markets, historical overlap with backtest windows may be zero.

## Verified online sources

### 1) Becker dataset (best fit for current Kalshi pipeline)

- URL: `https://s3.jbecker.dev/data.tar.zst`
- HTTP status: `200`
- Size: `36,020,641,508` bytes (~33.5 GiB compressed)
- Notes:
  - Source repository: `Jon-Becker/prediction-market-analysis`
  - Contains Kalshi/Polymarket parquet data and schemas.
  - Best source for our `market_price_builder.py` because it is parquet-native.

### 2) warproxxx/poly_data snapshots (useful for Polymarket research/microstructure)

- URL A: `https://polydata-archive.s3.us-east-1.amazonaws.com/orderFilled_complete.csv.xz`
  - HTTP status: `200`
  - Size: `6,239,202,792` bytes (~5.8 GiB)
- URL B: `https://polydata-archive.s3.us-east-1.amazonaws.com/archive.tar.xz`
  - HTTP status: `200`
  - Size: `9,151,960,868` bytes (~8.5 GiB)
- Notes:
  - Good for Polymarket fills/orderflow analysis.
  - Less direct for our current Kalshi weather threshold backtest path.
  - `archive.tar.xz` starts with a very large `goldsky/orderFilled.csv` member (~39 GB uncompressed), so random extraction of later members can be slow.

## Important capacity constraint

Available disk on this machine at check time: **~31 GiB free**.

This is below Becker compressed size (~33.5 GiB), so full local download + extract is not feasible without:

1. freeing disk, or
2. using external storage, or
3. streaming selective extraction (advanced, slower, but possible).

## Recommended path (for our project now)

1. Prefer Becker dataset.
2. Store/download to external volume or after freeing space.
3. Build normalized prices CSV:

```bash
cd /Users/naka/Documents/Weather
.venv/bin/python -m weatheredge.backtesting.market_price_builder \
  --input /PATH/TO/becker/data/kalshi/markets \
  --output /Users/naka/Documents/Weather/logs/kalshi_market_prices.csv \
  --start-date 2025-10-01 --end-date 2026-03-31
```

4. Run backtest with real prices (`market_prices_csv`), then walk-forward:

```bash
cd /Users/naka/Documents/Weather
.venv/bin/python -m weatheredge.backtesting.walkforward \
  --input /Users/naka/Documents/Weather/logs/backtest_results.csv \
  --output /Users/naka/Documents/Weather/logs/walkforward_results.csv \
  --train-days 60 --test-days 30 --step-days 30 --min-test-trades 25
```

## Quick header checks (already validated)

```bash
curl -I https://s3.jbecker.dev/data.tar.zst
curl -I https://polydata-archive.s3.us-east-1.amazonaws.com/orderFilled_complete.csv.xz
curl -I https://polydata-archive.s3.us-east-1.amazonaws.com/archive.tar.xz
```

## Project-local fallback path (already prepared)

Inside this repo we now use:

- download dir: `/Users/naka/Documents/Weather/datasets/remote_snapshots`
- work dir: `/Users/naka/Documents/Weather/datasets/work`

If Becker parquet is unavailable locally, we can still generate a real-price CSV
from `poly_data` snapshots:

```bash
cd /Users/naka/Documents/Weather/datasets/remote_snapshots
curl -L --continue-at - -o archive.tar.xz \
  https://polydata-archive.s3.us-east-1.amazonaws.com/archive.tar.xz
```

Then:

```bash
cd /Users/naka/Documents/Weather
.venv/bin/python -m weatheredge.backtesting.prepare_polydata_snapshot \
  --archive /Users/naka/Documents/Weather/datasets/remote_snapshots/archive.tar.xz \
  --work-dir /Users/naka/Documents/Weather/datasets/work/polydata \
  --output /Users/naka/Documents/Weather/logs/polydata_market_prices.csv \
  --start-date 2025-10-01 --end-date 2026-03-31
```

This outputs `station,date,threshold,direction,yes_price` usable by the backtester.
