"""Walk-forward optimization engine.

Splits historical data into rolling train/test windows, optimizes strategy
parameters on the training set, then validates on the unseen test set.
A strategy ensemble only passes if it meets minimum performance thresholds
across *all* out-of-sample folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from tradenojutsu.backtest.engine import BacktestEngine, BacktestResult
from tradenojutsu.data.models import PerformanceMetrics
from tradenojutsu.infra.logger import get_logger
from tradenojutsu.strategy.strategies import BaseStrategy, get_all_strategies

logger = get_logger("backtest.walk_forward")

# ---------------------------------------------------------------------------
# Pass / fail thresholds (out-of-sample)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: dict[str, float] = {
    "win_rate": 0.50,
    "profit_factor": 1.4,
    "max_drawdown": 0.18,
    "sharpe_ratio": 1.0,
    "min_trades": 50,
}


@dataclass
class FoldResult:
    """Result for a single train/test fold."""

    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_result: BacktestResult
    test_result: BacktestResult
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward results across all folds."""

    folds: list[FoldResult] = field(default_factory=list)
    aggregate_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    passed: bool = False
    failure_reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"=== WALK-FORWARD RESULTS ({status}) ===",
            f"Folds: {len(self.folds)}",
            f"Passed folds: {sum(1 for f in self.folds if f.passed)}/{len(self.folds)}",
            self.aggregate_metrics.summary(),
        ]
        if self.failure_reasons:
            lines.append("Failures: " + "; ".join(self.failure_reasons))
        return "\n".join(lines)


class WalkForwardOptimizer:
    """Rolling walk-forward backtester.

    Splits data into overlapping (train, test) folds and runs the
    ``BacktestEngine`` on each.  Aggregates out-of-sample metrics and
    checks pass/fail thresholds.

    Parameters
    ----------
    train_months : int
        Length of the in-sample training window in months.
    test_months : int
        Length of the out-of-sample test window in months.
    step_months : int | None
        How many months to advance between folds.  Defaults to
        ``test_months`` (non-overlapping test windows).
    thresholds : dict[str, float] | None
        Override the default pass/fail thresholds.
    initial_capital : float
        Starting equity for each fold.
    commission_pct : float
        Round-trip commission as a percentage of trade value.
    slippage_pct : float
        Simulated slippage as a percentage of price.
    """

    def __init__(
        self,
        train_months: int = 12,
        test_months: int = 2,
        step_months: int | None = None,
        thresholds: dict[str, float] | None = None,
        initial_capital: float = 10_000.0,
        commission_pct: float = 0.1,
        slippage_pct: float = 0.05,
    ) -> None:
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months or test_months
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    # ------------------------------------------------------------------
    # Fold generation
    # ------------------------------------------------------------------

    def generate_folds(
        self,
        df: pd.DataFrame,
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Split *df* into ``(train, test)`` pairs using a rolling window.

        The DataFrame **must** have a ``DatetimeIndex``.

        Returns
        -------
        list[tuple[pd.DataFrame, pd.DataFrame]]
            Each element is ``(train_df, test_df)``.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex for walk-forward splitting.")

        df = df.sort_index()
        start = df.index.min()
        end = df.index.max()

        folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        cursor = start

        while True:
            train_end = cursor + pd.DateOffset(months=self.train_months)
            test_end = train_end + pd.DateOffset(months=self.test_months)

            if test_end > end:
                break

            train_df = df.loc[cursor:train_end]  # type: ignore[misc]
            test_df = df.loc[train_end:test_end]  # type: ignore[misc]

            # Only include folds that have meaningful data in both windows
            if len(train_df) >= 50 and len(test_df) >= 10:
                folds.append((train_df, test_df))

            cursor += pd.DateOffset(months=self.step_months)

        logger.info(
            "Generated %d walk-forward folds (train=%d mo, test=%d mo, step=%d mo)",
            len(folds),
            self.train_months,
            self.test_months,
            self.step_months,
        )
        return folds

    # ------------------------------------------------------------------
    # Run walk-forward
    # ------------------------------------------------------------------

    def run_walk_forward(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategies: list[BaseStrategy] | None = None,
        risk_params: dict[str, Any] | None = None,
    ) -> WalkForwardResult:
        """Execute the full walk-forward optimization.

        Parameters
        ----------
        df : pd.DataFrame
            Full OHLCV history with ``DatetimeIndex``.
        symbol : str
            Asset symbol (e.g. ``"EURUSD"``).
        strategies : list[BaseStrategy] | None
            Strategies to evaluate.  Defaults to all registered strategies.
        risk_params : dict | None
            Risk manager overrides forwarded to ``BacktestEngine.run``.

        Returns
        -------
        WalkForwardResult
            Contains per-fold results, aggregated metrics, and a boolean
            ``passed`` flag.
        """
        if strategies is None:
            strategies = get_all_strategies()

        folds_data = self.generate_folds(df)
        if not folds_data:
            logger.warning("No valid folds could be generated from the supplied data.")
            return WalkForwardResult(
                passed=False,
                failure_reasons=["Insufficient data to generate any folds"],
            )

        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            slippage_pct=self.slippage_pct,
        )

        fold_results: list[FoldResult] = []

        for idx, (train_df, test_df) in enumerate(folds_data):
            logger.info(
                "Fold %d/%d  train: %s -> %s  test: %s -> %s",
                idx + 1,
                len(folds_data),
                train_df.index.min().date(),
                train_df.index.max().date(),
                test_df.index.min().date(),
                test_df.index.max().date(),
            )

            # --- In-sample (train) ---
            train_result = engine.run(
                train_df, symbol, strategies=strategies, risk_params=risk_params,
            )
            logger.info(
                "  Train: %d trades | WR %.1f%% | PF %.2f | Sharpe %.2f",
                train_result.metrics.total_trades,
                train_result.metrics.win_rate * 100,
                train_result.metrics.profit_factor,
                train_result.metrics.sharpe_ratio,
            )

            # --- Out-of-sample (test) ---
            test_result = engine.run(
                test_df, symbol, strategies=strategies, risk_params=risk_params,
            )
            logger.info(
                "  Test:  %d trades | WR %.1f%% | PF %.2f | Sharpe %.2f",
                test_result.metrics.total_trades,
                test_result.metrics.win_rate * 100,
                test_result.metrics.profit_factor,
                test_result.metrics.sharpe_ratio,
            )

            # --- Evaluate thresholds on OOS ---
            passed, reasons = self._evaluate_fold(test_result.metrics)

            fold_results.append(
                FoldResult(
                    fold_index=idx,
                    train_start=train_df.index.min().to_pydatetime(),
                    train_end=train_df.index.max().to_pydatetime(),
                    test_start=test_df.index.min().to_pydatetime(),
                    test_end=test_df.index.max().to_pydatetime(),
                    train_result=train_result,
                    test_result=test_result,
                    passed=passed,
                    failure_reasons=reasons,
                )
            )

            if passed:
                logger.info("  Fold %d PASSED", idx + 1)
            else:
                logger.warning("  Fold %d FAILED: %s", idx + 1, "; ".join(reasons))

        # --- Aggregate ---
        agg_metrics = self._aggregate_oos_metrics(fold_results)
        overall_passed, overall_reasons = self._evaluate_fold(agg_metrics)

        # Also fail if any individual fold failed
        failed_folds = [f for f in fold_results if not f.passed]
        if failed_folds:
            overall_passed = False
            overall_reasons.append(
                f"{len(failed_folds)}/{len(fold_results)} fold(s) failed OOS thresholds"
            )

        result = WalkForwardResult(
            folds=fold_results,
            aggregate_metrics=agg_metrics,
            passed=overall_passed,
            failure_reasons=overall_reasons,
        )

        logger.info("Walk-forward complete: %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_fold(self, metrics: PerformanceMetrics) -> tuple[bool, list[str]]:
        """Check a single fold's OOS metrics against thresholds."""
        reasons: list[str] = []

        if metrics.total_trades < self.thresholds["min_trades"]:
            reasons.append(
                f"trades {metrics.total_trades} < {int(self.thresholds['min_trades'])}"
            )
        if metrics.win_rate < self.thresholds["win_rate"]:
            reasons.append(
                f"win_rate {metrics.win_rate:.2%} < {self.thresholds['win_rate']:.0%}"
            )
        if metrics.profit_factor < self.thresholds["profit_factor"]:
            reasons.append(
                f"profit_factor {metrics.profit_factor:.2f} < {self.thresholds['profit_factor']:.1f}"
            )
        if metrics.max_drawdown > self.thresholds["max_drawdown"]:
            reasons.append(
                f"max_drawdown {metrics.max_drawdown:.2%} > {self.thresholds['max_drawdown']:.0%}"
            )
        if metrics.sharpe_ratio < self.thresholds["sharpe_ratio"]:
            reasons.append(
                f"sharpe {metrics.sharpe_ratio:.2f} < {self.thresholds['sharpe_ratio']:.1f}"
            )

        return len(reasons) == 0, reasons

    def _aggregate_oos_metrics(
        self,
        folds: list[FoldResult],
    ) -> PerformanceMetrics:
        """Combine out-of-sample results across all folds."""
        all_trades = []
        all_equity: list[float] = []

        for fold in folds:
            all_trades.extend(fold.test_result.trades)
            all_equity.extend(fold.test_result.equity_curve)

        if not all_trades:
            return PerformanceMetrics()

        pnls = [t.pnl for t in all_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / len(all_trades) if all_trades else 0.0
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(abs(np.mean(losses))) if losses else 0.0
        pf = (
            float(sum(wins) / abs(sum(losses)))
            if losses and sum(losses) != 0
            else float("inf")
        )

        # Drawdown from combined equity curve
        if all_equity:
            eq = np.array(all_equity)
            peak = np.maximum.accumulate(eq)
            dd = (peak - eq) / np.where(peak > 0, peak, 1)
            max_dd = float(np.max(dd))
        else:
            max_dd = 0.0

        # Sharpe from trade PnLs
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))
        else:
            sharpe = 0.0

        return PerformanceMetrics(
            total_trades=len(all_trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            profit_factor=pf,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_pnl=float(sum(pnls)),
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=float(win_rate * avg_win - (1 - win_rate) * avg_loss),
        )
