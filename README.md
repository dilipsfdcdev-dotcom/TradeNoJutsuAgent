# TradeNoJutsu - Self-Thinking, Self-Learning AI Trading Agent

An autonomous AI trading agent that **thinks** through market conditions using LLM-powered chain-of-thought reasoning and **learns** from its own trading history using reinforcement learning. Combines institutional-grade features (159 indicators + Smart Money Concepts) with a clean, modular Python architecture.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation (Step by Step)](#installation-step-by-step)
3. [Configuration](#configuration)
4. [Quick Start — Your First Analysis](#quick-start--your-first-analysis)
5. [Running a Backtest](#running-a-backtest)
6. [Walk-Forward Validation](#walk-forward-validation)
7. [Paper Trading (Simulated)](#paper-trading-simulated)
8. [Live Trading with MetaTrader 5](#live-trading-with-metatrader-5)
9. [Web Dashboard](#web-dashboard)
10. [Telegram Bot Setup](#telegram-bot-setup)
11. [News Sentiment Analysis](#news-sentiment-analysis)
12. [How the AI Brain Works](#how-the-ai-brain-works)
13. [How Self-Learning Works](#how-self-learning-works)
14. [Project Structure](#project-structure)
15. [All CLI Commands](#all-cli-commands)
16. [Troubleshooting](#troubleshooting)
17. [Disclaimer](#disclaimer)

---

## Prerequisites

- **Python 3.11+** — [Download Python](https://www.python.org/downloads/)
- **pip** — comes with Python
- **Git** — to clone the repo
- **Anthropic API Key** — for AI reasoning (get one at [console.anthropic.com](https://console.anthropic.com))
- **(Optional)** MetaTrader 5 — for live trading (Windows only)
- **(Optional)** Telegram account — for alerts and remote control

---

## Installation (Step by Step)

### Step 1: Clone the repository

```bash
git clone https://github.com/dilipsfdcdev-dotcom/TradeNoJutsuAgent.git
cd TradeNoJutsuAgent
```

### Step 2: Create a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install the package

```bash
# Basic install
pip install -e .

# With Telegram support
pip install -e ".[telegram]"

# With everything (Telegram + dev tools)
pip install -e ".[all,dev]"
```

### Step 4: Verify installation

```bash
tradenojutsu --version
tradenojutsu --help
```

You should see:
```
Usage: tradenojutsu [OPTIONS] COMMAND [ARGS]...

  TradeNoJutsu - Self-thinking, self-learning AI trading agent.

Commands:
  analyze    Run a single analysis cycle (no trading).
  backtest   Run a backtest on historical data.
  dashboard  Launch the web dashboard for monitoring.
  run        Run the trading agent in live/paper mode.
  sentiment  Analyze news sentiment for a symbol.
  status     Show agent status and recent performance.
```

---

## Configuration

### Step 1: Set up your API key

```bash
# Copy the example env file
cp tradenojutsu/config/.env.example tradenojutsu/config/.env
```

Edit `tradenojutsu/config/.env`:
```env
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
TRADENOJUTSU_MODE=paper
TRADENOJUTSU_LOG_LEVEL=INFO
```

### Step 2: Customize settings (optional)

Edit `tradenojutsu/config/settings.yaml`:

```yaml
agent:
  mode: "paper"              # paper | live | backtest
  llm_model: "claude-sonnet-4-6"   # AI model for reasoning

trading:
  symbols: ["GC=F"]         # Symbols to trade
  # Use these for different markets:
  #   Gold:    ["GC=F"]
  #   Stocks:  ["AAPL", "MSFT", "TSLA"]
  #   S&P 500: ["SPY"]
  #   Crypto:  ["BTC/USDT"] (set data_source to "ccxt")
  data_source: "yfinance"   # yfinance | ccxt

risk:
  risk_per_trade_pct: 1.0   # 1% of capital per trade
  max_concurrent_positions: 3
  daily_drawdown_max_pct: 3.0

backtest:
  initial_capital: 10000.0
```

---

## Quick Start — Your First Analysis

This works immediately, no API key needed:

```bash
# Analyze Gold
tradenojutsu analyze --symbol GC=F --timeframe 1d

# Analyze Apple stock
tradenojutsu analyze --symbol AAPL --timeframe 1d

# Analyze S&P 500
tradenojutsu analyze --symbol SPY --timeframe 1d
```

**What you'll see:**
```
=== Market State ===
Symbol: GC=F
Price: 4524.30
Regime: trending_down
Trend: short
Volatility (ATR%): 3.43%
Volume ratio: 18.55x average
Key Levels:
  sma_20: 4884.97
  sma_50: 4941.67
  ...

=== Technical Score (short) ===
  rsi: 0.0/20
  macd: 20.0/20
  trend: 20.0/20
  volume: 10.7/20
  TOTAL: 50.7/100
```

---

## Running a Backtest

Test strategies against historical data (no API key needed):

```bash
# Backtest Gold for 1 year
tradenojutsu backtest --symbol GC=F --days 365

# Backtest Apple for 2 years with $25,000 capital
tradenojutsu backtest --symbol AAPL --days 730 --capital 25000

# Backtest Tesla
tradenojutsu backtest --symbol TSLA --days 365

# Backtest S&P 500
tradenojutsu backtest --symbol SPY --days 365
```

**What you'll see:**
```
Running backtest: GC=F (1d) - 365 days
Capital: $10,000.00

Using full features: 159 columns
=== BACKTEST RESULTS ===
Trades: 15 | Win Rate: 80.0% | PF: 7.37 | Sharpe: 13.76 | Max DD: 2.07% | PnL: 1496.93
Total trades: 15
  smc: Trades: 15 | Win Rate: 80.0% | PnL: +1496.93

Final equity: $11,496.93
```

### Backtest from Python (advanced)

```python
from tradenojutsu.data.market_data import MarketDataFetcher
from tradenojutsu.backtest.engine import BacktestEngine

fetcher = MarketDataFetcher(source="yfinance")
df = fetcher.fetch_ohlcv("GC=F", "1d", lookback_days=365)

engine = BacktestEngine(
    initial_capital=10000,
    use_full_features=True,   # 159 features
    use_ratchet_sl=True,      # Profit-locking stop loss
)
result = engine.run(df, "GC=F")
print(result.summary())
print(f"Equity curve: {result.equity_curve[-5:]}")
```

---

## Walk-Forward Validation

Validate strategies across multiple time periods before going live:

```python
from tradenojutsu.data.market_data import MarketDataFetcher
from tradenojutsu.backtest.walk_forward import WalkForwardOptimizer

fetcher = MarketDataFetcher(source="yfinance")
df = fetcher.fetch_ohlcv("GC=F", "1d", lookback_days=730)  # 2 years

optimizer = WalkForwardOptimizer(
    initial_capital=10000,
    train_months=12,    # 12 months training
    test_months=2,      # 2 months out-of-sample test
)

result = optimizer.run_walk_forward(df, "GC=F")

print(f"Overall PASSED: {result.passed}")
print(f"Aggregate: {result.aggregate.summary()}")
for fold in result.fold_results:
    print(f"  Fold: {fold.metrics.summary()}")
```

**Pass/fail thresholds:**
- Win rate > 50%
- Profit factor > 1.4
- Max drawdown < 18%
- Sharpe ratio > 1.0
- Minimum 50 trades across all folds

---

## Paper Trading (Simulated)

Run the full AI agent loop without risking real money:

```bash
# Run with 60-second intervals
tradenojutsu run --interval 60

# Run with custom config
tradenojutsu run --interval 30 --config path/to/custom_settings.yaml
```

**What happens each cycle:**
1. Fetches latest market data
2. Computes 159 technical indicators + SMC features
3. Checks economic calendar for blackout windows
4. Fetches and analyzes news sentiment
5. Runs 6 trading strategies
6. AI Brain thinks through the decision (chain-of-thought)
7. Combines all signals with confidence scoring
8. Risk manager approves/rejects
9. Executes paper trade if approved
10. Ratchet SL manages open positions
11. Self-learning adjusts parameters after every 10 trades

**Press Ctrl+C to stop.**

### Check status while running

In another terminal:
```bash
tradenojutsu status
```

---

## Live Trading with MetaTrader 5

> **WARNING:** Live trading involves real money. Only use after extensive backtesting and walk-forward validation.

### Step 1: Install MetaTrader 5 (Windows only)

```bash
pip install MetaTrader5
```

### Step 2: Configure MT5 credentials

Add to your `.env` file:
```env
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-Server
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
```

### Step 3: Update settings.yaml

```yaml
agent:
  mode: "live"    # Change from "paper" to "live"

trading:
  symbols: ["XAUUSD"]
  data_source: "mt5"
```

### Step 4: Run

```bash
tradenojutsu run --interval 180   # 3-minute cycles for live trading
```

---

## Web Dashboard

Launch a real-time monitoring dashboard:

```bash
tradenojutsu dashboard --port 8080
```

Then open **http://localhost:8080** in your browser.

**Dashboard shows:**
- Portfolio status and open positions
- Equity curve chart
- Recent trades with PnL
- AI thinking logs (chain-of-thought reasoning)
- Learning cycle history (parameter changes)
- Auto-refreshes every 30 seconds

**API endpoints** (for custom integrations):
```
GET /api/status       — Current positions
GET /api/trades       — Trade history
GET /api/performance  — Metrics + equity curve
GET /api/thinking     — AI reasoning logs
GET /api/learning     — Parameter adjustment history
GET /api/signals      — Signal history
GET /api/snapshots    — Performance over time
GET /docs             — Swagger API documentation
```

---

## Telegram Bot Setup

### Step 1: Create a Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Choose a name (e.g., "TradeNoJutsu Bot")
4. Choose a username (e.g., "tradenojutsu_bot")
5. Copy the **bot token** (looks like `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)

### Step 2: Get your Chat ID

1. Search for **@userinfobot** on Telegram
2. Send it any message
3. Copy your **Chat ID** (a number like `123456789`)

### Step 3: Configure

Edit `tradenojutsu/config/settings.yaml`:
```yaml
telegram:
  bot_token: "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
  chat_id: "123456789"
```

### Step 4: Install Telegram dependency

```bash
pip install -e ".[telegram]"
```

### Step 5: Run the agent

```bash
tradenojutsu run --interval 60
```

The bot will now send you alerts automatically.

### Available Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Open positions, capital, mode |
| `/trades` | Recent closed trades with PnL |
| `/pnl` | Today / this week / all-time PnL |
| `/performance` | Win rate, Sharpe, profit factor, drawdown |
| `/regime` | Current market regime for each symbol |
| `/risk 1.5` | View or change risk per trade % |
| `/threshold 70` | View or change signal entry threshold |
| `/learned` | Last learning cycle results |
| `/stop` | Pause the agent |
| `/resume` | Resume the agent |
| `/halt` | **EMERGENCY:** Close all positions and stop |
| `/help` | List all commands |

### Alert Types

- **Trade opened** — entry price, SL, TP, strategy, reasoning
- **Trade closed** — PnL, exit reason
- **Learning cycle** — parameter changes and performance
- **Risk warnings** — drawdown limits approaching
- **AI thinking** — high-confidence decision reasoning

---

## News Sentiment Analysis

Analyze market news sentiment for any symbol:

```bash
# Check sentiment (fetches headlines, no API key needed)
tradenojutsu sentiment --symbol GC=F
tradenojutsu sentiment --symbol AAPL
tradenojutsu sentiment --symbol SPY
```

**With API key** (for LLM-powered analysis):
```
Found 15 headlines:
  1. [Reuters] Gold prices drop as dollar strengthens...
  2. [Bloomberg] Federal Reserve signals rate pause...

News Sentiment for GC=F:
  Overall: bearish (-0.35)
  Confidence: 72%
  Headlines analyzed: 15
  Key Headlines:
    [-0.6] Gold tumbles 2% on strong jobs data
    [-0.4] Fed hawkish stance weighs on precious metals
    [+0.3] Central bank gold buying hits record
  Analysis: Bearish near-term due to dollar strength and Fed rhetoric.
```

### From Python

```python
from tradenojutsu.data.news_sentiment import NewsSentimentAnalyzer

analyzer = NewsSentimentAnalyzer()

# Fetch headlines (free, no API key)
headlines = analyzer.fetch_headlines("GC=F")
for h in headlines[:5]:
    print(f"  [{h.source}] {h.headline}")

# Full LLM analysis (requires ANTHROPIC_API_KEY)
report = analyzer.analyze_sentiment("GC=F")
print(report.to_prompt_context())
```

---

## How the AI Brain Works

The agent uses Claude as its reasoning engine. Before every trading decision, it performs **chain-of-thought analysis**:

```
=== AI THINKING ===

Step 1 - MARKET CONTEXT:
Gold is in a downtrend on the daily chart. Price has broken below the
20-day SMA and is testing the 50-day SMA. The regime is "trending_down"
with elevated volatility (ATR 3.4%).

Step 2 - TECHNICAL ANALYSIS:
RSI at 38.9 suggests oversold conditions approaching but not extreme.
MACD histogram is deeply negative and declining. Bollinger Band width
is expanding, confirming volatility expansion. ADX at 27.2 confirms
a strong trend.

Step 3 - RISK ASSESSMENT:
Entering a counter-trend long here carries significant risk as the trend
is firmly bearish. The risk/reward is unfavorable for longs. A short
position would align with the trend but RSI oversold conditions suggest
a bounce may be imminent.

Step 4 - PATTERN RECOGNITION:
SMC shows a bullish order block near current price, and a fair value gap
above. No liquidity sweep yet, so downside may continue.

Step 5 - DECISION:
WAIT. The setup is unclear — bearish trend conflicts with approaching
oversold conditions. Capital preservation takes priority.

Confidence: 35%
```

Every thinking session is logged to the database and visible in the dashboard.

---

## How Self-Learning Works

After every 10 closed trades, the agent runs a **learning cycle**:

1. **Calculates performance** — Win rate, Sharpe ratio, profit factor
2. **Computes reward signal** — How well are we doing vs. last cycle?
3. **Updates Q-table** — Which parameter adjustments improve performance?
4. **Selects changes** — Epsilon-greedy (90% exploit best known, 10% explore)
5. **Applies adjustments** — Updates settings.yaml with new values
6. **AI reflects** — Asks the LLM for qualitative insights

**Parameters the agent can tune itself:**

| Parameter | Range | What it controls |
|-----------|-------|------------------|
| `risk_per_trade_pct` | 0.5% – 2.0% | How much capital to risk per trade |
| `max_concurrent_positions` | 1 – 5 | Max simultaneous open trades |
| `stop_loss_atr_mult` | 0.8 – 2.5 | Stop-loss distance (ATR multiples) |
| `take_profit_rr_ratio` | 1.0 – 4.0 | Reward:risk ratio for TP |
| `entry_threshold` | 50 – 85 | Minimum signal score to trade |
| `model_weights.technical` | 0.1 – 0.6 | Weight of technical analysis |
| `model_weights.ml_ensemble` | 0.1 – 0.6 | Weight of ML models |
| `model_weights.llm_reasoning` | 0.1 – 0.6 | Weight of AI reasoning |
| `exploration_rate` | 0.01 – 0.30 | How much to explore vs exploit |

---

## Project Structure

```
TradeNoJutsuAgent/
├── pyproject.toml                    # Package definition
├── README.md                         # This file
├── tradenojutsu/
│   ├── __init__.py
│   ├── agent.py                      # Main orchestrator
│   ├── cli.py                        # CLI commands
│   ├── config/
│   │   ├── __init__.py               # Config loader
│   │   ├── settings.yaml             # All settings
│   │   └── .env.example              # API key template
│   ├── analysis/
│   │   ├── technical.py              # Basic indicators (25)
│   │   ├── features_150.py           # Advanced features (149)
│   │   ├── smc_features.py           # Smart Money Concepts (10)
│   │   └── multi_timeframe.py        # Multi-TF alignment
│   ├── brain/
│   │   ├── reasoner.py               # LLM chain-of-thought (self-thinking)
│   │   └── signal_combiner.py        # Multi-source signal fusion
│   ├── learning/
│   │   ├── self_learner.py           # Q-learning optimizer (self-learning)
│   │   └── ml_ensemble.py            # LightGBM + PPO models
│   ├── strategy/
│   │   └── strategies.py             # 6 strategies
│   ├── risk/
│   │   └── manager.py                # Position sizing & limits
│   ├── execution/
│   │   ├── paper_trader.py           # Paper trading
│   │   ├── ratchet_sl.py             # 6-level ratchet stop-loss
│   │   └── mt5_executor.py           # MetaTrader 5 live trading
│   ├── backtest/
│   │   ├── engine.py                 # Backtest engine (159 features)
│   │   └── walk_forward.py           # Walk-forward validation
│   ├── data/
│   │   ├── models.py                 # Core data types
│   │   ├── market_data.py            # yfinance/ccxt/MT5 data
│   │   ├── news_sentiment.py         # LLM news analysis
│   │   └── economic_calendar.py      # FOMC/NFP blackout detection
│   ├── dashboard/
│   │   └── server.py                 # FastAPI web dashboard
│   ├── telegram_bot/
│   │   ├── alerts.py                 # One-way trade alerts
│   │   └── commands.py               # 12+ interactive commands
│   └── infra/
│       ├── database.py               # SQLite persistence
│       └── logger.py                 # Rich console logging
└── tests/
    └── test_core.py                  # Unit tests (14 tests)
```

---

## All CLI Commands

```bash
# Analysis (no API key needed)
tradenojutsu analyze --symbol GC=F --timeframe 1d
tradenojutsu sentiment --symbol AAPL

# Backtesting (no API key needed)
tradenojutsu backtest --symbol GC=F --days 365 --capital 10000
tradenojutsu backtest --symbol MSFT --days 730 --capital 25000

# Paper trading (needs ANTHROPIC_API_KEY)
tradenojutsu run --interval 60

# Dashboard
tradenojutsu dashboard --port 8080

# Status
tradenojutsu status
```

---

## Troubleshooting

### "No module named 'tradenojutsu'"
Make sure you installed with `pip install -e .` from the project root.

### "Could not resolve authentication method"
Set your `ANTHROPIC_API_KEY` in `tradenojutsu/config/.env`.

### "No data returned for XAUUSD from yfinance"
Use `GC=F` for gold on yfinance. `XAUUSD` is for MT5/broker feeds.

### "ModuleNotFoundError: MetaTrader5"
MT5 is only needed for live trading. Install with `pip install MetaTrader5` (Windows only).

### "pip install fails with build errors"
Try upgrading pip and setuptools first:
```bash
pip install --upgrade pip setuptools
pip install -e .
```

### Tests
```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

---

## Disclaimer

This is an experimental AI trading agent for **educational and research purposes**.

- **Do NOT use with real money** without extensive backtesting and walk-forward validation
- **Past performance** does not guarantee future results
- **Trading involves substantial risk** of loss
- The AI reasoning is probabilistic and can be wrong
- Always use proper risk management and position sizing
- Start with paper trading and small amounts

The authors are not responsible for any financial losses incurred from using this software.
