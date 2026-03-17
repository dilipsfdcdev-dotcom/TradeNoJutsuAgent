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

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd xauusd_agent

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp xauusd_agent/config/.env.example xauusd_agent/config/.env
# Edit .env with your MT5, DB, Telegram, and API credentials

# 4. Start PostgreSQL (via Docker)
docker-compose up -d postgres

# 5. Run
python -m xauusd_agent.main
```

## Docker Deployment

```bash
docker-compose up -d
```

## VPS Deployment

```bash
chmod +x setup_vps.sh
./setup_vps.sh
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
