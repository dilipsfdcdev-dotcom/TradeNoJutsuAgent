"""CLI entry point for TradeNoJutsu Agent."""

from __future__ import annotations

import asyncio
import sys

import click

from tradenojutsu.infra.logger import setup_logging


@click.group()
@click.version_option(version="1.0.0")
def main():
    """TradeNoJutsu - Self-thinking, self-learning AI trading agent."""
    pass


@main.command()
@click.option("--interval", default=60, help="Seconds between analysis cycles")
@click.option("--config", default=None, help="Path to config YAML")
def run(interval: int, config: str | None):
    """Run the trading agent in live/paper mode."""
    setup_logging()
    from tradenojutsu.agent import TradeNoJutsuAgent

    agent = TradeNoJutsuAgent(config_path=config)
    click.echo(f"Starting TradeNoJutsu Agent (mode={agent.mode})...")
    click.echo(f"Symbols: {agent.symbols}")
    click.echo(f"Interval: {interval}s")
    click.echo("Press Ctrl+C to stop.\n")

    try:
        asyncio.run(agent.run(interval_seconds=interval))
    except KeyboardInterrupt:
        agent.stop()
        click.echo("\nAgent stopped.")


@main.command()
@click.option("--symbol", default="GC=F", help="Symbol to backtest (use GC=F for gold on yfinance)")
@click.option("--timeframe", default="1d", help="Timeframe (1d, 1h, etc)")
@click.option("--days", default=365, help="Lookback period in days")
@click.option("--capital", default=10000.0, help="Initial capital")
def backtest(symbol: str, timeframe: str, days: int, capital: float):
    """Run a backtest on historical data."""
    setup_logging()
    from tradenojutsu.agent import TradeNoJutsuAgent

    agent = TradeNoJutsuAgent()
    click.echo(f"Running backtest: {symbol} ({timeframe}) - {days} days")
    click.echo(f"Capital: ${capital:,.2f}\n")

    result = agent.run_backtest(symbol, timeframe, days)
    if result:
        click.echo(result.summary())
        click.echo(f"\nEquity curve points: {len(result.equity_curve)}")
    else:
        click.echo("Backtest failed - no data available.")


@main.command()
@click.option("--symbol", default="GC=F", help="Symbol to analyze")
@click.option("--timeframe", default="1d", help="Timeframe")
def analyze(symbol: str, timeframe: str):
    """Run a single analysis cycle (no trading)."""
    setup_logging()
    from tradenojutsu.agent import TradeNoJutsuAgent

    agent = TradeNoJutsuAgent()
    click.echo(f"Analyzing {symbol} ({timeframe})...\n")

    df = agent.data_fetcher.fetch_ohlcv(symbol, timeframe, lookback_days=180)
    if df.empty:
        click.echo("No data available.")
        return

    df = agent.analyzer.compute_indicators(df)
    state = agent.analyzer.build_market_state(symbol, df)

    click.echo("=== Market State ===")
    click.echo(state.to_prompt_context())

    direction = state.trend_direction
    score, components = agent.analyzer.score_signal(df, direction)
    click.echo(f"\n=== Technical Score ({direction.value}) ===")
    for name, val in components.items():
        click.echo(f"  {name}: {val:.1f}/20")
    click.echo(f"  TOTAL: {score:.1f}/100")


@main.command()
def status():
    """Show agent status and recent performance."""
    setup_logging()
    from tradenojutsu.infra.database import get_open_trades, get_recent_trades

    open_trades = get_open_trades()
    recent = get_recent_trades(20)

    click.echo("=== TradeNoJutsu Status ===")
    click.echo(f"Open positions: {len(open_trades)}")
    for t in open_trades:
        click.echo(f"  {t['direction']} {t['symbol']} @ {t['entry_price']:.4f}")

    click.echo(f"\nRecent closed trades: {len(recent)}")
    total_pnl = sum(t.get("pnl", 0) or 0 for t in recent)
    wins = sum(1 for t in recent if (t.get("pnl") or 0) > 0)
    click.echo(f"  Win rate: {wins}/{len(recent)} ({wins/len(recent)*100:.0f}%)" if recent else "  No trades yet")
    click.echo(f"  Total PnL: {total_pnl:+.2f}")


if __name__ == "__main__":
    main()
