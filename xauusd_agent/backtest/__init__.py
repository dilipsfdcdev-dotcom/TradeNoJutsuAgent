"""Backtesting infrastructure for the XAUUSD trading agent."""

from xauusd_agent.backtest.data_loader import BacktestDataLoader
from xauusd_agent.backtest.nautilus_backtest import (
    BacktestPosition,
    BacktestResult,
    XAUUSDBacktester,
)
from xauusd_agent.backtest.report_generator import BacktestReportGenerator
from xauusd_agent.backtest.walk_forward import WalkForwardOptimizer

__all__ = [
    "BacktestDataLoader",
    "BacktestPosition",
    "BacktestResult",
    "XAUUSDBacktester",
    "BacktestReportGenerator",
    "WalkForwardOptimizer",
]
