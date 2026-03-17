# XAUUSD Autonomous Self-Learning Trading Agent v2.0

Production-grade, fully autonomous, self-learning XAUUSD (Gold) trading agent for MetaTrader 5.

## Architecture

```
Entry:      M3 + M5 (3-min primary trigger, 5-min confirmation)
Direction:  D1/H4/H1 Higher Timeframe Bias Engine
Protection: Dynamic Ratchet SL (profit-locking, never backward)
Lot Sizing: Live balance-based, never hardcoded
Learning:   Weekly LightGBM + Monthly PPO + Continuous MemoryAgent
```

## Four Design Principles

1. **Dynamic Profit-Locking SL** — Incremental ratchet that locks increasing % of profit as trade moves favorably. A trade at +$500 can never reverse to a full loss.

2. **Balance-Based Dynamic Lot Sizing** — Lot size calculated from live account balance on every trade. Account compounds automatically.

3. **M3/M5 Entry with HTF Direction** — Six timeframes, each with a specific role. HTF focuses trades directionally without blocking signals.

4. **Self-Learning Agent** — Continuously learns from live trade outcomes. Updates parameters, retrains models, all fully auditable.

## Setup Guide (Step by Step)

### Step 1: Prerequisites

Install on your machine before starting:
- **Python 3.11+** — [python.org/downloads](https://www.python.org/downloads/)
- **MetaTrader 5** — installed on a Windows machine or Windows VPS
- **PostgreSQL 16** — or use Docker (Step 4)
- **Docker** (optional) — for running PostgreSQL easily

### Step 2: Clone the Repository

```bash
git clone <repo-url>
cd TradeNoJutsuAgent
```

### Step 3: Create Python Virtual Environment

```bash
python3.11 -m venv venv

# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Start PostgreSQL Database

**Option A — Docker (recommended):**
```bash
docker-compose up -d postgres
```

**Option B — Local PostgreSQL:**
```bash
sudo -u postgres psql -c "CREATE USER agent WITH PASSWORD 'changeme';"
sudo -u postgres psql -c "CREATE DATABASE xauusd_agent OWNER agent;"
psql -U agent -d xauusd_agent -f scripts/init_db.sql
```

### Step 5: Configure Environment Variables

```bash
cp xauusd_agent/config/.env.example xauusd_agent/config/.env
```

Edit `xauusd_agent/config/.env` with your credentials:
- **MT5_LOGIN** — Your MetaTrader 5 account number
- **MT5_PASSWORD** — MT5 password
- **MT5_SERVER** — Broker server name (e.g., `ICMarketsSC-Demo`)
- **MT5_PATH** — Path to `terminal64.exe` on Windows
- **DB_PASSWORD** — PostgreSQL password (default: `changeme`)
- **TELEGRAM_BOT_TOKEN** — From [@BotFather](https://t.me/BotFather) on Telegram
- **TELEGRAM_CHAT_ID** — Your Telegram user/group ID
- **ANTHROPIC_API_KEY** — From [console.anthropic.com](https://console.anthropic.com/)
- **NEWSAPI_KEY** — From [newsapi.org](https://newsapi.org/)

### Step 6: Verify MT5 Connection

Open a Python shell and test:
```python
import MetaTrader5 as mt5
mt5.initialize()
mt5.login(your_login, password="your_password", server="your_server")
print(mt5.account_info())
print(mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 5))
mt5.shutdown()
```

If `TIMEFRAME_M3` returns empty, the agent will auto-synthesize from M1 data.

### Step 7: Review Settings

Open `xauusd_agent/config/settings.yaml`. Key settings to review:
- `risk.risk_pct: 1.0` — Risk per trade (1% of balance). Start conservative.
- `risk.max_concurrent: 3` — Max simultaneous open trades
- `signals.entry_threshold: 62` — Signal score needed to trade (52-78 range)

### Step 8: Run in Demo Mode First

```bash
python -m xauusd_agent.main
```

The agent starts two independent loops:
- **Signal loop** — every 3 minutes, scans for trade setups
- **Ratchet loop** — every 15 seconds, protects open trades

Monitor via Telegram commands: `/status`, `/trades`, `/pnl`

### Step 9: Backtest Before Going Live

```python
from xauusd_agent.backtest.data_loader import BacktestDataLoader
from xauusd_agent.backtest.nautilus_backtest import XAUUSDBacktester
from xauusd_agent.backtest.walk_forward import WalkForwardValidator
from xauusd_agent.backtest.report_generator import BacktestReportGenerator
import yaml

# Load settings
with open("xauusd_agent/config/settings.yaml") as f:
    settings = yaml.safe_load(f)

# Load data
loader = BacktestDataLoader()
data = loader.load_all_timeframes("XAUUSD")

# Run walk-forward validation
validator = WalkForwardValidator(settings)
results = validator.run_walk_forward(data)

# Generate HTML report
report = BacktestReportGenerator()
report.generate_html_report(results)
print("Pass/fail:", results["passed"])
```

**Go-live thresholds** (all must pass):
- Win rate > 50%
- Profit factor > 1.4
- Max drawdown < 18%
- Sharpe ratio > 1.0
- Minimum 300 trades across folds
- Ratchet saves > 15% of trades

### Step 10: Demo Trading (4 Weeks Minimum)

Run on a **demo account** for at least 4 weeks. Verify:
- [ ] Ratchet SL fires every 15 seconds (check `/trades`)
- [ ] Lot sizes change when balance changes
- [ ] `/bias` returns sensible HTF scores
- [ ] `/learned` shows learning cycles after 10+ trades
- [ ] Signal count is 5-12 per day in normal markets
- [ ] News blackout activates during FOMC/NFP

### Step 11: Go Live

1. Fund with minimum test amount ($500-$1,000)
2. Watch first 10 trades manually — verify lots match formula
3. Verify ratchet moves SL in live account
4. After 1 month: compare live vs backtest win rate (within 5%)
5. Do NOT increase risk until 30-trade baseline is established

## Docker Deployment (Full Stack)

```bash
docker-compose up -d
```
Starts PostgreSQL + the trading agent. Logs in `./logs/`.

## VPS Deployment (Systemd)

```bash
chmod +x setup_vps.sh
./setup_vps.sh
# Then: sudo systemctl start xauusd-agent
# Logs: journalctl -u xauusd-agent -f
```

## Project Structure

```
xauusd_agent/
├── config/          # Settings + secrets
├── data/            # MT5, news, macro data fetchers
├── features/        # 140+ feature engineering pipeline
├── models/          # PPO, LightGBM, ensemble, memory agent
├── brain/           # HTF bias, signal engine, LLM reasoner, combiner
├── execution/       # MT5 connector, executor, lot calc, ratchet SL
├── learning/        # Trade labelling, model retraining, regime detection
├── backtest/        # NautilusTrader harness, walk-forward validation
├── infra/           # Database, logging, watchdog, scheduler
├── telegram_bot/    # Async Telegram bot with full command suite
└── main.py          # Dual async loop orchestrator
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Bot state, balance, equity, regime, HTF bias |
| `/trades` | Open trades with P&L and locked profit |
| `/stop` | Pause signal loop (ratchet keeps running) |
| `/resume` | Resume signal loop |
| `/halt` | Emergency: close ALL positions + stop |
| `/risk X` | Set risk percentage |
| `/pnl` | Today/week/month P&L breakdown |
| `/bias` | Current HTF bias scores |
| `/regime` | Current market regime |
| `/threshold` | Current signal entry threshold |
| `/learned` | Last learning cycle results |
| `/scores` | Last signal component scores |

## Go-Live Checklist

See Phase 14 in the build prompt for the complete go-live verification checklist covering backtesting, demo trading, risk verification, and infrastructure checks.

## Technology Stack

- **Runtime**: Python 3.11+, asyncio
- **Broker**: MetaTrader 5
- **ML**: Stable-Baselines3 (PPO), LightGBM, ONNX Runtime
- **LLM**: Claude Haiku (news risk filter only)
- **Database**: PostgreSQL 16 + asyncpg
- **Alerts**: python-telegram-bot v20
- **Infrastructure**: Docker, systemd, structured JSON logging
