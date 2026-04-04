# TradeNoJutsu Agent -- AI Scalping Trading Bot

An autonomous AI-powered scalping agent for forex markets. The system combines MetaTrader 5 execution, Claude-driven decision making, technical analysis, and a real-time monitoring dashboard to identify and execute short-term trading opportunities.

> **WARNING:** This software trades real money. Use at your own risk. Past performance does not guarantee future results. See the [Risk Warnings](#risk-warnings) section before running in production.

---

## Architecture Overview

The agent is organized into several cooperating layers:

| Layer | Description |
|---|---|
| **Data Ingestion** | Pulls live tick and OHLCV data from MetaTrader 5, normalizes it, and publishes to Redis streams for downstream consumers. |
| **Signal Generation** | Computes technical indicators (RSI, MACD, Bollinger Bands, VWAP, ATR) and detects candlestick patterns. Emits candidate trade signals. |
| **AI Brain** | Claude evaluates candidate signals against market context, news sentiment, and portfolio state. Decides whether to enter, skip, or adjust parameters. |
| **Execution** | Translates brain decisions into MT5 orders with proper lot sizing, stop-loss, and take-profit. Manages open positions and implements trailing stops. |
| **Risk Management** | Enforces per-trade and daily drawdown limits, maximum position counts, and cooldown periods after consecutive losses. |
| **Monitoring & API** | FastAPI server exposes REST endpoints and real-time metrics. A WebSocket server pushes live updates to the dashboard. |
| **Dashboard** | Next.js + TailwindCSS frontend with live P&L charts, trade log, signal feed, and system health indicators. |
| **Persistence** | PostgreSQL stores trades, signals, and performance history. Redis provides caching, pub/sub, and rate-limit state. |
| **Backtesting** | Replay historical data through the signal and brain pipeline to evaluate strategy performance before going live. |

---

## Quick Start (Docker Compose)

The fastest way to run the full stack:

```bash
# 1. Clone the repository
git clone <repo-url> && cd TradeNoJutsuAgent

# 2. Create your environment file
cp .env.example .env
# Edit .env with your API keys and MT5 credentials

# 3. Start everything
docker compose up -d

# 4. Check health
python scripts/health_check.py

# 5. Open the dashboard
open http://localhost:3000
```

Docker Compose starts four services:

- **db** -- PostgreSQL 15 on port 5432
- **redis** -- Redis 7 on port 6379
- **agent** -- The Python trading agent (API on port 8000, WebSocket on port 8765)
- **dashboard** -- Next.js frontend on port 3000

---

## Manual Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- PostgreSQL 15+
- Redis 7+
- MetaTrader 5 terminal (Windows or Wine on Linux)

### Backend (Agent)

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run database migrations
alembic upgrade head

# Seed historical data (optional, for backtesting)
python scripts/seed_historical.py

# Start the agent
python -m agent.main
```

### Frontend (Dashboard)

```bash
cd dashboard
npm install
npm run dev
# Dashboard available at http://localhost:3000
```

---

## Configuration Reference

All configuration is via environment variables (or a `.env` file in the project root).

### Required

| Variable | Description | Example |
|---|---|---|
| `ANTHROPIC_API_KEY` | Claude API key for the AI brain | `sk-ant-...` |
| `MT5_LOGIN` | MetaTrader 5 account number | `12345678` |
| `MT5_PASSWORD` | MetaTrader 5 password | `...` |
| `MT5_SERVER` | MT5 broker server name | `MetaQuotes-Demo` |

### Database & Cache

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://agent:password@localhost:5432/trading_agent` | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |

### Trading Parameters

| Variable | Default | Description |
|---|---|---|
| `SYMBOLS` | `EURUSD,GBPUSD,USDJPY` | Comma-separated list of forex pairs to trade |
| `TIMEFRAME` | `M5` | Primary chart timeframe |
| `MAX_POSITION_SIZE` | `0.1` | Maximum lot size per trade |
| `MAX_DAILY_LOSS` | `100.0` | Daily loss limit in account currency |
| `MAX_OPEN_TRADES` | `3` | Maximum concurrent open positions |
| `RISK_PER_TRADE` | `0.01` | Risk as fraction of account balance (1%) |

### Ports

| Variable | Default | Description |
|---|---|---|
| `API_PORT` | `8000` | FastAPI REST server port |
| `WS_PORT` | `8765` | WebSocket server port |
| `DASHBOARD_PORT` | `3000` | Next.js dashboard port |

### Logging

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Python log level |
| `LOG_FORMAT` | `json` | Log output format (`json` or `console`) |

---

## Dashboard

The web dashboard provides real-time visibility into the agent:

- **P&L Chart** -- Equity curve and daily profit/loss over time
- **Trade Log** -- Scrollable table of recent trades with entry/exit prices, P&L, and duration
- **Signal Feed** -- Live stream of detected signals and brain decisions
- **System Health** -- Status of all components (MT5 connection, database, Redis, API)

<!-- Screenshots: replace these placeholders with actual screenshots -->
_Screenshots coming soon._

---

## Health Check

Run the built-in health monitor to verify all components are reachable:

```bash
# Human-readable output
python scripts/health_check.py

# JSON output (for CI or monitoring tools)
python scripts/health_check.py --json
```

The script checks PostgreSQL, Redis, the API server, and the dashboard, reporting status and latency for each.

---

## Backtesting

Replay historical data to evaluate strategy performance:

```bash
# Seed historical candles first
python scripts/seed_historical.py

# Run a backtest
python scripts/run_backtest.py --symbol EURUSD --start 2024-01-01 --end 2024-06-30
```

---

## Risk Warnings

**Read this section carefully before running the agent with real funds.**

- **No guarantee of profit.** Algorithmic trading involves substantial risk of loss. The agent may lose part or all of the capital allocated to it.
- **Market risk.** Forex markets are volatile. Rapid price movements, gaps, and slippage can cause losses exceeding stop-loss levels.
- **Technology risk.** Software bugs, network outages, broker API failures, or MT5 disconnections can result in missed exits or duplicate orders.
- **AI model risk.** Claude's decisions are based on pattern recognition, not certainty. Model hallucinations or misinterpretations can lead to poor trades.
- **Overfitting risk.** Backtesting results may not reflect live performance. Strategies that appear profitable on historical data can fail in real markets.
- **Latency risk.** Scalping strategies are sensitive to execution speed. Running the agent on a high-latency connection may degrade performance.
- **Regulatory risk.** Automated trading may be subject to regulations in your jurisdiction. Ensure compliance before deploying.

**Recommendations:**
- Start with a demo account.
- Set conservative daily loss limits.
- Monitor the agent actively during the first weeks of operation.
- Never allocate funds you cannot afford to lose.

---

## Estimated Monthly Cost Breakdown

| Component | Estimated Cost | Notes |
|---|---|---|
| Anthropic API (Claude) | $30 -- $150 | Depends on trade frequency and context size |
| VPS (2 vCPU, 4 GB RAM) | $20 -- $40 | Cloud VM to host the agent 24/7 |
| PostgreSQL (managed) | $0 -- $15 | Free tier or small instance; self-hosted is free |
| Redis (managed) | $0 -- $10 | Free tier or small instance; self-hosted is free |
| MT5 broker | $0 | Most brokers provide MT5 free; spreads/commissions apply per trade |
| Domain + TLS (optional) | $0 -- $5 | Only if exposing the dashboard publicly |
| **Total** | **$50 -- $220/mo** | Excluding trading capital and broker commissions |

Self-hosting PostgreSQL and Redis on the same VPS brings the lower bound closer to $50/month.

---

## License

This project is for educational and personal use. See `LICENSE` for details.
