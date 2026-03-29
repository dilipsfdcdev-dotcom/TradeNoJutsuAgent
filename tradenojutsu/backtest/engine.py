"""Backtesting engine - test strategies against historical data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tradenojutsu.analysis.technical import TechnicalAnalyzer
from tradenojutsu.data.models import Direction, PerformanceMetrics, Signal, Trade, TradeStatus
from tradenojutsu.infra.logger import get_logger
from tradenojutsu.risk.manager import RiskManager, RiskParams
from tradenojutsu.strategy.strategies import BaseStrategy, get_all_strategies

logger = get_logger("backtest.engine")


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    strategy_breakdown: dict[str, PerformanceMetrics] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== BACKTEST RESULTS ===",
            self.metrics.summary(),
            f"Total trades: {len(self.trades)}",
        ]
        for name, m in self.strategy_breakdown.items():
            lines.append(f"  {name}: {m.summary()}")
        return "\n".join(lines)


class BacktestEngine:
    """Runs backtests against historical data.

    Simulates the full trading pipeline: indicators -> strategy -> risk -> execution.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_pct: float = 0.1,
        slippage_pct: float = 0.05,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.analyzer = TechnicalAnalyzer()

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategies: list[BaseStrategy] | None = None,
        risk_params: dict[str, Any] | None = None,
    ) -> BacktestResult:
        """Run a backtest on historical data.

        Args:
            df: OHLCV DataFrame
            symbol: Asset symbol
            strategies: Strategies to test (defaults to all)
            risk_params: Override risk parameters
        """
        if strategies is None:
            strategies = get_all_strategies()

        # Compute indicators
        df = self.analyzer.compute_indicators(df)

        # Setup risk manager
        rm_kwargs = {"capital": self.initial_capital}
        if risk_params:
            rm_kwargs.update(risk_params)
        risk_mgr = RiskManager(**rm_kwargs)

        capital = self.initial_capital
        equity_curve = [capital]
        all_trades: list[Trade] = []
        open_trades: list[Trade] = []

        # Walk through bars
        for i in range(50, len(df)):  # Skip first 50 for indicator warmup
            bar = df.iloc[i]
            price = bar["close"]
            atr = bar.get("atr", price * 0.02)
            lookback = df.iloc[:i + 1]

            # Check open trades for exit
            for trade in list(open_trades):
                should_exit, reason = risk_mgr.check_exit_conditions(trade, price)
                if should_exit:
                    trade = self._close_trade(trade, price, reason)
                    capital += trade.pnl
                    all_trades.append(trade)
                    open_trades.remove(trade)

            # Check for new signals
            for strategy in strategies:
                signal = strategy.evaluate(lookback, symbol)
                if signal is None or not signal.is_actionable:
                    continue

                # Risk check
                allowed, _ = risk_mgr.can_trade(signal)
                if not allowed:
                    continue

                # Apply slippage
                entry_price = self._apply_slippage(price, signal.direction)

                # Calculate risk params
                regime = self.analyzer.detect_regime(lookback)
                rp = risk_mgr.calculate_risk_params(signal, entry_price, atr, regime)

                trade = Trade(
                    symbol=symbol,
                    direction=signal.direction,
                    entry_price=entry_price,
                    stop_loss=rp.stop_loss,
                    take_profit=rp.take_profit,
                    quantity=rp.position_size,
                    status=TradeStatus.OPEN,
                    entry_time=df.index[i] if hasattr(df.index[i], 'isoformat') else None,
                    strategy=strategy.name,
                )
                open_trades.append(trade)
                risk_mgr.daily_trade_count += 1

            equity_curve.append(capital)

        # Close remaining open trades at last price
        last_price = df["close"].iloc[-1]
        for trade in open_trades:
            trade = self._close_trade(trade, last_price, "End of backtest")
            capital += trade.pnl
            all_trades.append(trade)

        # Calculate metrics
        metrics = self._compute_metrics(all_trades, equity_curve)
        strategy_breakdown = self._strategy_breakdown(all_trades)

        result = BacktestResult(
            trades=all_trades,
            equity_curve=equity_curve,
            metrics=metrics,
            strategy_breakdown=strategy_breakdown,
        )

        logger.info(f"Backtest complete: {result.summary()}")
        return result

    def _close_trade(self, trade: Trade, price: float, reason: str) -> Trade:
        exit_price = self._apply_slippage(price, trade.direction, exit=True)
        commission = exit_price * trade.quantity * (self.commission_pct / 100)

        if trade.direction == Direction.LONG:
            pnl = (exit_price - trade.entry_price) * trade.quantity - commission
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity - commission

        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_pct = (pnl / (trade.entry_price * trade.quantity)) * 100
        trade.status = TradeStatus.CLOSED
        return trade

    def _apply_slippage(self, price: float, direction: Direction, exit: bool = False) -> float:
        slip = price * (self.slippage_pct / 100)
        if (direction == Direction.LONG and not exit) or (direction == Direction.SHORT and exit):
            return price + slip  # Worse fill for entry long / exit short
        return price - slip

    def _compute_metrics(self, trades: list[Trade], equity_curve: list[float]) -> PerformanceMetrics:
        closed = [t for t in trades if t.status == TradeStatus.CLOSED]
        if not closed:
            return PerformanceMetrics()

        pnls = [t.pnl for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(closed) if closed else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(dd))

        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))
        else:
            sharpe = 0

        return PerformanceMetrics(
            total_trades=len(closed),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            profit_factor=float(pf),
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_pnl=float(sum(pnls)),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            expectancy=float(win_rate * avg_win - (1 - win_rate) * avg_loss),
        )

    def _strategy_breakdown(self, trades: list[Trade]) -> dict[str, PerformanceMetrics]:
        breakdown = {}
        strats = set(t.strategy for t in trades)
        for strat in strats:
            strat_trades = [t for t in trades if t.strategy == strat]
            breakdown[strat] = self._compute_metrics(strat_trades, [self.initial_capital])
        return breakdown
