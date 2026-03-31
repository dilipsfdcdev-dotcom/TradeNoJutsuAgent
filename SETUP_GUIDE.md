# TradeNoJutsu Agent — Step-by-Step Setup Guide

## Prerequisites

- **Windows PC or VPS** (MetaTrader 5 requires Windows — or Linux with Wine)
- **Docker & Docker Compose** installed
- **MT5 broker account** (demo account recommended for initial testing)
- **Anthropic API key** from [console.anthropic.com](https://console.anthropic.com)
- **Node.js 18+** (for dashboard development — Docker handles production)
- **Python 3.11+** (for agent development — Docker handles production)

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/dilipsfdcdev-dotcom/TradeNoJutsuAgent.git
cd TradeNoJutsuAgent
```

## Step 2: Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:

```env
# MT5 — Get these from your broker
MT5_LOGIN=12345678           # Your MT5 account number
MT5_PASSWORD=YourPassword    # Your MT5 password
MT5_SERVER=YourBroker-Demo   # e.g., "MetaQuotes-Demo" or "ICMarkets-Demo"
MT5_PATH=C:/Program Files/MetaTrader 5/terminal64.exe

# Claude API — Get from console.anthropic.com
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxx

# Database (default works with Docker Compose)
DATABASE_URL=postgresql://agent:password@localhost:5432/trading_agent
REDIS_URL=redis://localhost:6379/0

# News API (optional) — Get from newsapi.org
NEWS_API_KEY=your_newsapi_key

# Risk Settings (start conservative!)
MAX_RISK_PER_TRADE_PCT=0.5    # Start at 0.5% per trade
MAX_DAILY_LOSS_PCT=2.0         # Stop trading after 2% daily loss
MAX_OPEN_TRADES=2              # Max 2 simultaneous positions
MAX_DRAWDOWN_PCT=5.0           # Emergency shutdown at 5% drawdown
```

## Step 3: Start Infrastructure with Docker Compose

```bash
# Start PostgreSQL and Redis
docker compose up -d db redis

# Wait for databases to be ready
sleep 5

# Verify they're running
docker compose ps
```

## Step 4: Install Python Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -e ".[dev]"

# Install ML dependencies (optional, for v2/v3 features)
pip install xgboost torch scikit-learn
```

## Step 5: Initialize the Database

```bash
# Run Alembic migrations
alembic upgrade head
```

## Step 6: Install and Configure MetaTrader 5

1. **Download MT5** from your broker or [metatrader5.com](https://www.metatrader5.com)
2. **Install and login** with your broker credentials
3. **Enable Algo Trading**: Tools → Options → Expert Advisors → Check "Allow automated trading"
4. **Verify terminal path** matches `MT5_PATH` in your `.env`
5. **Test connection**:
```bash
python -c "
from agent.data.mt5_feed import init_mt5, get_account_info
if init_mt5():
    print('Connected!', get_account_info())
else:
    print('Connection failed')
"
```

## Step 7: Set Up the Dashboard

```bash
cd dashboard
npm install
npm run dev    # Starts on http://localhost:3000
```

Open http://localhost:3000/setup in your browser to configure credentials from the UI.

## Step 8: Seed Historical Data (Optional but Recommended)

```bash
# Download historical candle data for backtesting
python scripts/seed_historical.py --symbol XAUUSD --days 90
python scripts/seed_historical.py --symbol BTCUSD --days 90
python scripts/seed_historical.py --symbol XAGUSD --days 90
```

## Step 9: Run a Backtest (Recommended Before Live)

```bash
python scripts/run_backtest.py \
  --symbol XAUUSD \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --mode rules \
  --balance 10000
```

Review the generated HTML report in the output.

## Step 10: Start the Trading Agent

```bash
# Start the agent (connects to MT5, starts analysis loop)
python -m agent.main
```

The agent will:
1. Connect to MT5 and verify account
2. Connect to PostgreSQL and Redis
3. Start the FastAPI WebSocket server on port 8000
4. Begin the analysis loop on every 1-minute candle close

## Step 11: Monitor via Dashboard

Open http://localhost:3000 to see:
- **Live price charts** for all symbols
- **Active positions** with real-time P&L
- **AI Signal Log** showing every analysis decision
- **Equity curve** tracking your balance over time
- **MTF Status** showing multi-timeframe alignment
- **ML Scores** showing XGBoost/LSTM pre-filter results
- **Veto Panel** showing Claude's risk oversight decisions

---

## Production Deployment with Docker

For 24/7 operation, use Docker Compose:

```bash
# Build and start all services
docker compose up -d

# Check logs
docker compose logs -f agent
docker compose logs -f dashboard

# Stop everything
docker compose down
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| MT5 won't connect | Ensure MT5 terminal is running and Algo Trading is enabled |
| "No module named MetaTrader5" | MT5 Python package only works on Windows. Use Wine on Linux. |
| Database connection refused | Run `docker compose up -d db` first |
| Claude API errors | Check your API key at console.anthropic.com |
| Dashboard won't load | Run `npm install` in the dashboard/ directory |
| Agent stops trading | Check circuit breaker — may have hit daily loss limit |
| High API costs | Use "signal-only" mode — set `SIGNAL_ONLY_MODE=true` in .env |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FAST LOOP (<6ms)                      │
│  MTF Analysis → XGBoost Filter → LSTM Confidence        │
│  → Rule Engine → Veto Check → Execute                   │
├─────────────────────────────────────────────────────────┤
│                    SLOW LOOP (async)                     │
│  Claude Veto Scanner (30s) │ Trade Reviewer │ Monitor   │
├─────────────────────────────────────────────────────────┤
│  MT5 ←→ Agent ←→ PostgreSQL + Redis ←→ Dashboard       │
└─────────────────────────────────────────────────────────┘
```

- **Fast loop**: Deterministic, zero API calls, runs every 1m candle
- **Slow loop**: Claude reviews and vetoes asynchronously, never blocks trades
- **Dashboard**: Real-time WebSocket updates for monitoring

---

## Safety Checklist

- [ ] Start with a **demo account** — never real money first
- [ ] Set `MAX_DRAWDOWN_PCT` to 5% or less
- [ ] Set `MAX_DAILY_LOSS_PCT` to 2-3%
- [ ] Set `MAX_RISK_PER_TRADE_PCT` to 0.5-1%
- [ ] Run backtests on at least 3 months of data
- [ ] Paper trade for at least 2 weeks before going live
- [ ] Monitor the dashboard daily during initial live trading
- [ ] Never disable the circuit breaker
