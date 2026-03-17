"""
Walk-forward optimisation and validation for the XAUUSD trading agent.

Implements a rolling 15-month train / 2-month test scheme that produces
at least 6 non-overlapping test folds across a 2+ year dataset.  Each fold
is backtested independently, and aggregate pass/fail thresholds are applied
to gate the strategy for live deployment.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import timedelta
from typing import Any

import pandas as pd

from xauusd_agent.backtest.nautilus_backtest import BacktestResult, XAUUSDBacktester
from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default pass/fail thresholds
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "win_rate": 0.50,          # > 50 %
    "profit_factor": 1.4,      # > 1.4
    "max_drawdown_pct": 0.18,  # < 18 %
    "sharpe_ratio": 1.0,       # > 1.0
    "min_trades": 300,         # across all folds
    "ratchet_save_pct": 0.15,  # > 15 %
}


class WalkForwardOptimizer:
    """15-month train / 2-month test walk-forward validation."""

    def __init__(
        self,
        settings: dict,
        initial_balance: float = 10_000.0,
    ) -> None:
        self.settings = settings
        self.initial_balance = initial_balance
        self.thresholds = {
            **_DEFAULT_THRESHOLDS,
            **settings.get("thresholds", {}),
        }

    # ------------------------------------------------------------------
    # Fold generation
    # ------------------------------------------------------------------

    def generate_folds(
        self,
        all_tf_data: dict[str, pd.DataFrame],
        train_months: int = 15,
        test_months: int = 2,
    ) -> list[dict]:
        """Generate walk-forward folds from the data.

        Each fold is a dict::

            {
                "fold_num": int,
                "train_start": pd.Timestamp,
                "train_end": pd.Timestamp,
                "test_start": pd.Timestamp,
                "test_end": pd.Timestamp,
            }

        The folds are anchored on the M3 timeframe date range and step
        forward by ``test_months`` each iteration, ensuring a minimum of
        6 non-overlapping test windows.

        Parameters
        ----------
        all_tf_data:
            Multi-TF data dict (must contain ``"M3"``).
        train_months:
            Length of the training window in months.
        test_months:
            Length of the out-of-sample test window in months.

        Returns
        -------
        list[dict]
            Ordered list of fold descriptors.
        """
        m3 = all_tf_data.get("M3")
        if m3 is None or m3.empty:
            logger.error("Cannot generate folds: no M3 data")
            return []

        data_start = m3.index.min()
        data_end = m3.index.max()

        train_delta = timedelta(days=train_months * 30)
        test_delta = timedelta(days=test_months * 30)

        folds: list[dict] = []
        fold_num = 1
        cursor = data_start

        while True:
            train_start = cursor
            train_end = train_start + train_delta
            test_start = train_end
            test_end = test_start + test_delta

            # Stop if test window extends beyond available data
            if test_end > data_end:
                break

            folds.append(
                {
                    "fold_num": fold_num,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )

            fold_num += 1
            cursor += test_delta  # step forward by test window size

        logger.info(
            "Generated %d walk-forward folds (train=%dm, test=%dm) "
            "from %s to %s",
            len(folds),
            train_months,
            test_months,
            data_start.date(),
            data_end.date(),
        )

        if len(folds) < 6:
            logger.warning(
                "Only %d folds generated (minimum recommended: 6). "
                "Consider using a longer dataset.",
                len(folds),
            )

        return folds

    # ------------------------------------------------------------------
    # Walk-forward execution
    # ------------------------------------------------------------------

    def run_walk_forward(
        self,
        all_tf_data: dict[str, pd.DataFrame],
    ) -> dict[str, Any]:
        """Run backtest on each fold and aggregate results.

        Returns
        -------
        dict
            ``folds``       – list of :class:`BacktestResult` (one per fold)
            ``aggregate``   – combined metrics across all folds
            ``passed``      – bool, True when all thresholds met
            ``thresholds``  – the threshold dict used for evaluation
            ``fold_details`` – per-fold summary dicts
        """
        folds = self.generate_folds(all_tf_data)
        if not folds:
            logger.error("No folds generated; aborting walk-forward")
            return {
                "folds": [],
                "aggregate": BacktestResult(),
                "passed": False,
                "thresholds": self.thresholds,
                "fold_details": [],
            }

        fold_results: list[BacktestResult] = []
        fold_details: list[dict] = []

        for fold in folds:
            logger.info(
                "Running fold %d: test %s -> %s",
                fold["fold_num"],
                fold["test_start"].date(),
                fold["test_end"].date(),
            )

            # We backtest on the TEST window, but supply full data up to
            # test_end so that HTF indicators have sufficient warm-up from
            # the training period.
            test_data = self._split_data(
                all_tf_data,
                fold["train_start"],
                fold["test_end"],
            )

            # Mark where the test period starts so we only count trades
            # opened during the test window.
            backtester = XAUUSDBacktester(
                settings=copy.deepcopy(self.settings),
                initial_balance=self.initial_balance,
            )

            result = backtester.run(test_data)

            # Filter trades to only those within the test window
            test_start_ts = fold["test_start"]
            test_trades = [
                t for t in result.trades
                if pd.Timestamp(t["open_time"]) >= test_start_ts
            ]
            result.total_trades = len(test_trades)
            result.wins = sum(1 for t in test_trades if t["pnl"] > 0)
            result.losses = sum(1 for t in test_trades if t["pnl"] <= 0)
            result.win_rate = (
                result.wins / result.total_trades
                if result.total_trades > 0
                else 0.0
            )

            fold_results.append(result)
            fold_details.append(
                {
                    "fold_num": fold["fold_num"],
                    "test_start": str(fold["test_start"].date()),
                    "test_end": str(fold["test_end"].date()),
                    "trades": result.total_trades,
                    "win_rate": round(result.win_rate, 4),
                    "profit_factor": round(result.profit_factor, 2),
                    "total_profit": round(result.total_profit, 2),
                    "max_drawdown_pct": round(result.max_drawdown_pct, 4),
                    "sharpe_ratio": round(result.sharpe_ratio, 2),
                    "ratchet_save_pct": round(result.ratchet_save_pct, 4),
                }
            )

            logger.info(
                "Fold %d complete: %d trades, WR=%.1f%%, PF=%.2f, DD=%.1f%%",
                fold["fold_num"],
                result.total_trades,
                result.win_rate * 100,
                result.profit_factor,
                result.max_drawdown_pct * 100,
            )

        # Aggregate metrics
        aggregate = self._aggregate_results(fold_results)

        # Pass/fail evaluation
        passed = self._evaluate_thresholds(aggregate)

        logger.info(
            "Walk-forward %s: %d total trades, WR=%.1f%%, PF=%.2f, "
            "DD=%.1f%%, Sharpe=%.2f, Ratchet saves=%.1f%%",
            "PASSED" if passed else "FAILED",
            aggregate.total_trades,
            aggregate.win_rate * 100,
            aggregate.profit_factor,
            aggregate.max_drawdown_pct * 100,
            aggregate.sharpe_ratio,
            aggregate.ratchet_save_pct * 100,
        )

        return {
            "folds": fold_results,
            "aggregate": aggregate,
            "passed": passed,
            "thresholds": self.thresholds,
            "fold_details": fold_details,
        }

    # ------------------------------------------------------------------
    # Data slicing
    # ------------------------------------------------------------------

    def _split_data(
        self,
        all_tf: dict[str, pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        """Slice all TF DataFrames to the ``[start, end]`` date range."""
        sliced: dict[str, pd.DataFrame] = {}
        for tf_name, df in all_tf.items():
            if df is None or df.empty:
                sliced[tf_name] = pd.DataFrame()
                continue
            mask = (df.index >= start) & (df.index <= end)
            sliced[tf_name] = df.loc[mask].copy()
        return sliced

    # ------------------------------------------------------------------
    # Result aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_results(results: list[BacktestResult]) -> BacktestResult:
        """Combine multiple fold results into a single aggregate."""
        agg = BacktestResult()

        if not results:
            return agg

        all_trades: list[dict] = []
        all_equity: list[dict] = []

        for r in results:
            all_trades.extend(r.trades)
            all_equity.extend(r.equity_curve)

        agg.total_trades = len(all_trades)
        agg.wins = sum(1 for t in all_trades if t["pnl"] > 0)
        agg.losses = sum(1 for t in all_trades if t["pnl"] <= 0)
        agg.win_rate = agg.wins / agg.total_trades if agg.total_trades > 0 else 0.0

        agg.total_profit = sum(t["pnl"] for t in all_trades)

        gross_wins = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
        gross_losses = abs(sum(t["pnl"] for t in all_trades if t["pnl"] <= 0))
        agg.profit_factor = (
            gross_wins / gross_losses if gross_losses > 0 else float("inf")
        )

        # Weighted average Sharpe
        sharpes = [r.sharpe_ratio for r in results if r.total_trades > 0]
        agg.sharpe_ratio = sum(sharpes) / len(sharpes) if sharpes else 0.0

        # Worst-case drawdown across all folds
        agg.max_drawdown_pct = max(
            (r.max_drawdown_pct for r in results), default=0.0
        )

        # Average R for winners and losers
        winner_rs = [t["R"] for t in all_trades if t["R"] > 0]
        loser_rs = [t["R"] for t in all_trades if t["R"] <= 0]
        if winner_rs:
            agg.avg_winner_R = sum(winner_rs) / len(winner_rs)
        if loser_rs:
            agg.avg_loser_R = sum(loser_rs) / len(loser_rs)

        # Ratchet stats
        ratchet_saves = sum(r.ratchet_saves for r in results)
        agg.ratchet_saves = ratchet_saves
        agg.ratchet_save_pct = (
            ratchet_saves / agg.total_trades if agg.total_trades > 0 else 0.0
        )

        # Equity curve (concatenated)
        agg.equity_curve = all_equity
        agg.trades = all_trades

        # Win rate by bias / regime (aggregate)
        bias_wins: dict[str, int] = {}
        bias_totals: dict[str, int] = {}
        regime_wins: dict[str, int] = {}
        regime_totals: dict[str, int] = {}

        for t in all_trades:
            d = t.get("htf_direction", "NEUTRAL")
            bias_totals[d] = bias_totals.get(d, 0) + 1
            if t["pnl"] > 0:
                bias_wins[d] = bias_wins.get(d, 0) + 1

            r = t.get("regime", "NORMAL")
            regime_totals[r] = regime_totals.get(r, 0) + 1
            if t["pnl"] > 0:
                regime_wins[r] = regime_wins.get(r, 0) + 1

        agg.win_rate_by_bias = {
            d: bias_wins.get(d, 0) / n if n > 0 else 0.0
            for d, n in bias_totals.items()
        }
        agg.win_rate_by_regime = {
            r: regime_wins.get(r, 0) / n if n > 0 else 0.0
            for r, n in regime_totals.items()
        }

        # Monthly P&L
        monthly: dict[str, float] = {}
        for t in all_trades:
            ct = t.get("close_time", "")
            if ct and len(ct) >= 7:
                month_key = ct[:7]
                monthly[month_key] = monthly.get(month_key, 0.0) + t["pnl"]
        agg.monthly_pnl = monthly

        return agg

    # ------------------------------------------------------------------
    # Threshold evaluation
    # ------------------------------------------------------------------

    def _evaluate_thresholds(self, aggregate: BacktestResult) -> bool:
        """Check whether the aggregate result passes all thresholds.

        Threshold rules:
            - win_rate > threshold (default 50 %)
            - profit_factor > threshold (default 1.4)
            - max_drawdown_pct < threshold (default 18 %)
            - sharpe_ratio > threshold (default 1.0)
            - total_trades >= min_trades (default 300)
            - ratchet_save_pct > threshold (default 15 %)
        """
        checks: list[tuple[str, bool]] = [
            (
                "win_rate",
                aggregate.win_rate > self.thresholds["win_rate"],
            ),
            (
                "profit_factor",
                aggregate.profit_factor > self.thresholds["profit_factor"],
            ),
            (
                "max_drawdown",
                aggregate.max_drawdown_pct < self.thresholds["max_drawdown_pct"],
            ),
            (
                "sharpe_ratio",
                aggregate.sharpe_ratio > self.thresholds["sharpe_ratio"],
            ),
            (
                "min_trades",
                aggregate.total_trades >= self.thresholds["min_trades"],
            ),
            (
                "ratchet_saves",
                aggregate.ratchet_save_pct > self.thresholds["ratchet_save_pct"],
            ),
        ]

        all_passed = True
        for name, ok in checks:
            if not ok:
                logger.warning("Threshold FAILED: %s", name)
                all_passed = False
            else:
                logger.info("Threshold passed: %s", name)

        return all_passed
