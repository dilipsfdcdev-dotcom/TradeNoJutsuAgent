# TradeNoJutsu - Self-Thinking, Self-Learning AI Trading Agent

An autonomous AI trading agent that **thinks** through market conditions using LLM-powered chain-of-thought reasoning and **learns** from its own trading history using reinforcement learning.

## Architecture

```
tradenojutsu/
├── agent.py                 # Main orchestrator - the "nervous system"
├── cli.py                   # CLI entry point
├── config/                  # Settings & environment
│   ├── settings.yaml        # All tunable parameters
│   └── .env.example         # API keys template
├── data/
│   ├── models.py            # Core data types (Signal, Trade, MarketState)
│   └── market_data.py       # Market data fetching (yfinance, ccxt)
├── analysis/
│   └── technical.py         # Technical indicators & market state
├── brain/                   # "Self-Thinking" Engine
│   ├── reasoner.py          # LLM chain-of-thought reasoning
│   └── signal_combiner.py   # Multi-source signal fusion
├── learning/                # "Self-Learning" Engine
│   ├── self_learner.py      # RL-based parameter optimization
│   └── ml_ensemble.py       # LightGBM + PPO models
├── strategy/
│   └── strategies.py        # Trading strategies (trend, mean-rev, breakout, momentum)
├── risk/
│   └── manager.py           # Position sizing, stop-loss, portfolio risk
├── execution/
│   └── paper_trader.py      # Paper trading simulator
├── backtest/
│   └── engine.py            # Historical backtesting
└── infra/
    ├── logger.py            # Structured logging
    └── database.py          # SQLite persistence
```

## How It Works

### Self-Thinking (AI Brain)
The agent uses Claude as its reasoning engine. Before every trade decision, it performs **chain-of-thought analysis**:
1. Evaluates market regime (trending, ranging, volatile)
2. Analyzes technical indicator confluence
3. Assesses risk/reward favorability
4. Makes a reasoned decision with transparent rationale

Every thought process is logged to the database for review and learning.

### Self-Learning (RL Feedback Loop)
After every N trades, the agent runs a **learning cycle**:
1. Calculates performance metrics (Sharpe ratio, win rate, profit factor)
2. Computes a reward signal for Q-learning updates
3. Asks the AI Brain for qualitative self-reflection
4. Uses epsilon-greedy exploration to adjust parameters
5. Tracks what changes helped and what didn't

Tunable parameters include risk sizing, entry thresholds, signal weights, and more.

## Quick Start

```bash
# Install
pip install -e .

# Configure
cp tradenojutsu/config/.env.example tradenojutsu/config/.env
# Edit .env with your ANTHROPIC_API_KEY

# Run analysis (no trading)
tradenojutsu analyze --symbol "GC=F"

# Run backtest
tradenojutsu backtest --symbol "GC=F" --days 365

# Run paper trading
tradenojutsu run --interval 60

# Check status
tradenojutsu status
```

## Strategies

| Strategy | Type | Description |
|----------|------|-------------|
| Trend Following | Trend | EMA crossover with ADX confirmation |
| Mean Reversion | Counter-trend | Bollinger Band + RSI extremes in ranging markets |
| Breakout | Momentum | Range breakout with volume confirmation |
| Momentum | Multi-indicator | RSI + MACD + Stochastic alignment |

## Configuration

All parameters live in `tradenojutsu/config/settings.yaml`. Parameters marked `[AGENT]` can be auto-tuned by the self-learning module. Parameters marked `[FIXED]` require manual changes.

## Disclaimer

This is an experimental AI trading agent for educational and research purposes. **Do not use with real money without extensive testing.** Past performance does not guarantee future results. Trading involves substantial risk of loss.
