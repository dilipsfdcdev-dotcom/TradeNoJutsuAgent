"""
Walk-forward validation for the XAUUSD trading agent.

Implements a rolling 15-month train / 2-month test scheme that produces
at least 6 non-overlapping test folds across a 2+ year dataset.  Each fold
is backtested independently, and aggregate pass/fail thresholds are applied
to gate the strategy for live deployment.
"""

from __future__ import annotations

import copy
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from xauusd_agent.backtest.nautilus_backtest import XAUUSDBacktester
from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default go-live thresholds
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "win_rate": 50.0,           # > 50 %
    "profit_factor": 1.4,       # > 1.4
    "max_drawdown_pct": 18.0,   # < 18 %
    "sharpe_ratio": 1.0,        # > 1.0
    "min_trades": 300,          # across all folds
    "ratchet_saves_pct": 15.0,  # > 15 %
}


class WalkForwardValidator:
    """15-month train / 2-month test walk-forward validation."""

    def __init__(self, settings: dict) -> None:
        self.settings = settings
        self.train_months: int = settings.get("train_months", 15)
        self.test_months: int = settings.get("test_months", 2)
        self.initial_balance: float = settings.get("initial_balance", 10_000.0)

    # ------------------------------------------------------------------
    # Fold generation
    # ------------------------------------------------------------------

    def generate_folds(
        self, start_date: str, end_date: str,
    ) -> list[dict]:
        """Generate walk-forward folds.

        Each fold is a dict::

            {
                "fold_num": int,
                "train_start": pd.Timestamp,
                "train_end": pd.Timestamp,
                "test_start": pd.Timestamp,
                "test_end": pd.Timestamp,
            }

        The folds step forward by ``test_months`` each iteration,
        ensuring a minimum of 6 non-overlapping test windows when the
        dataset is long enough.

        Parameters
        ----------
        start_date / end_date:
            ISO-8601 date strings (``"YYYY-MM-DD"``) bounding the full
            dataset.

        Returns
        -------
        list[dict]
            Ordered list of fold descriptors.
        """
        data_start = pd.Timestamp(start_date, tz="UTC")
        data_end = pd.Timestamp(end_date, tz="UTC")

        train_delta = timedelta(days=self.train_months * 30)
        test_delta = timedelta(days=self.test_months * 30)

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

            folds.append({
                "fold_num": fold_num,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })

            fold_num += 1
            cursor += test_delta  # step forward by test window size

        logger.info(
            "Generated %d walk-forward folds (train=%dm, test=%dm) "
            "from %s to %s",
            len(folds), self.train_months, self.test_months,
            start_date, end_date,
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
        self, all_tf_data: dict[str, pd.DataFrame],
    ) -> dict:
        """Run backtest on each fold and aggregate results.

        For each fold:

        1. Split data into train/test periods (train data is included
           for indicator warm-up, but only test-period trades count).
        2. Run backtest on the combined window.
        3. Filter trades to only those opened during the test window.
        4. Collect results per fold.

        Parameters
        ----------
        all_tf_data:
            Multi-TF data dict as returned by
            :meth:`BacktestDataLoader.load_all_timeframes`.

        Returns
        -------
        dict
            ``folds``          -- list of per-fold result dicts
            ``aggregate``      -- combined metrics across all folds
            ``go_live``        -- output of :meth:`check_go_live_thresholds`
        """
        # Determine date range from M3 data
        m3 = all_tf_data.get("M3", pd.DataFrame())
        if m3.empty:
            logger.error("No M3 data; cannot run walk-forward")
            return {
                "folds": [],
                "aggregate": {},
                "go_live": {"passed": False, "checks": {}},
            }

        start_date = str(m3.index.min().date())
        end_date = str(m3.index.max().date())

        folds = self.generate_folds(start_date, end_date)
        if not folds:
            logger.error("No folds generated; aborting walk-forward")
            return {
                "folds": [],
                "aggregate": {},
                "go_live": {"passed": False, "checks": {}},
            }

        fold_results: list[dict] = []
        all_trades: list[dict] = []
        all_equity: list[dict] = []

        for fold in folds:
            logger.info(
                "Running fold %d: test %s -> %s",
                fold["fold_num"],
                fold["test_start"].date(),
                fold["test_end"].date(),
            )

            # Supply full data from train_start to test_end so HTF
            # indicators have sufficient warm-up from training period.
            fold_data = self._split_data(
                all_tf_data, fold["train_start"], fold["test_end"],
            )

            backtester = XAUUSDBacktester(
                settings=copy.deepcopy(self.settings),
                initial_balance=self.initial_balance,
            )

            result = backtester.run(fold_data)

            # Filter trades to only those within the test window
            test_start_ts = fold["test_start"]
            test_trades = [
                t for t in result.get("trades", [])
                if pd.Timestamp(t["open_time"]) >= test_start_ts
            ]

            # Recompute metrics for test-only trades
            test_winners = [t for t in test_trades if t["profit_usd"] > 0]
            test_losers = [t for t in test_trades if t["profit_usd"] <= 0]
            gross_profit = sum(t["profit_usd"] for t in test_winners)
            gross_loss = abs(sum(t["profit_usd"] for t in test_losers))

            fold_result = {
                "fold_num": fold["fold_num"],
                "test_start": str(fold["test_start"].date()),
                "test_end": str(fold["test_end"].date()),
                "total_trades": len(test_trades),
                "winners": len(test_winners),
                "losers": len(test_losers),
                "win_rate": round(
                    len(test_winners) / len(test_trades) * 100, 2,
                ) if test_trades else 0.0,
                "profit_factor": round(
                    gross_profit / gross_loss, 3,
                ) if gross_loss > 0 else float("inf"),
                "total_pnl": round(sum(t["profit_usd"] for t in test_trades), 2),
                "max_drawdown_pct": result.get("max_drawdown_pct", 0.0),
                "sharpe_ratio": result.get("sharpe_ratio", 0.0),
                "ratchet_saves_pct": result.get("ratchet_saves_pct", 0.0),
                "trades": test_trades,
                "equity_curve": result.get("equity_curve", []),
            }

            fold_results.append(fold_result)
            all_trades.extend(test_trades)
            all_equity.extend(result.get("equity_curve", []))

            logger.info(
                "Fold %d complete: %d trades, WR=%.1f%%, PF=%.2f",
                fold["fold_num"],
                fold_result["total_trades"],
                fold_result["win_rate"],
                fold_result["profit_factor"],
            )

        # Aggregate metrics across all folds
        aggregate = self._aggregate_results(fold_results, all_trades, all_equity)

        # Go-live evaluation
        go_live = self.check_go_live_thresholds(aggregate)

        logger.info(
            "Walk-forward %s: %d total trades, WR=%.1f%%, PF=%.2f, "
            "DD=%.1f%%, Sharpe=%.2f",
            "PASSED" if go_live["passed"] else "FAILED",
            aggregate.get("total_trades", 0),
            aggregate.get("win_rate", 0),
            aggregate.get("profit_factor", 0),
            aggregate.get("max_drawdown_pct", 0),
            aggregate.get("sharpe_ratio", 0),
        )

        return {
            "folds": fold_results,
            "aggregate": aggregate,
            "go_live": go_live,
        }

    # ------------------------------------------------------------------
    # Go-live threshold check
    # ------------------------------------------------------------------

    def check_go_live_thresholds(self, results: dict) -> dict:
        """Check aggregate results against go-live criteria.

        Criteria:
            - win_rate > 50%
            - profit_factor > 1.4
            - max_drawdown < 18%
            - sharpe > 1.0
            - min_trades > 300
            - ratchet_saves > 15%

        Parameters
        ----------
        results:
            Aggregate results dict from :meth:`run_walk_forward`.

        Returns
        -------
        dict
            ``passed``  -- bool, True when all thresholds are met.
            ``checks``  -- dict of ``{criterion: {value, threshold, passed}}``.
        """
        thresholds = {**_DEFAULT_THRESHOLDS, **self.settings.get("thresholds", {})}

        checks: dict[str, dict] = {}

        # win_rate > 50%
        wr = results.get("win_rate", 0.0)
        checks["win_rate"] = {
            "value": wr,
            "threshold": thresholds["win_rate"],
            "passed": wr > thresholds["win_rate"],
        }

        # profit_factor > 1.4
        pf = results.get("profit_factor", 0.0)
        checks["profit_factor"] = {
            "value": pf,
            "threshold": thresholds["profit_factor"],
            "passed": pf > thresholds["profit_factor"],
        }

        # max_drawdown < 18%
        dd = results.get("max_drawdown_pct", 100.0)
        checks["max_drawdown_pct"] = {
            "value": dd,
            "threshold": thresholds["max_drawdown_pct"],
            "passed": dd < thresholds["max_drawdown_pct"],
        }

        # sharpe > 1.0
        sr = results.get("sharpe_ratio", 0.0)
        checks["sharpe_ratio"] = {
            "value": sr,
            "threshold": thresholds["sharpe_ratio"],
            "passed": sr > thresholds["sharpe_ratio"],
        }

        # min_trades > 300
        trades = results.get("total_trades", 0)
        checks["min_trades"] = {
            "value": trades,
            "threshold": thresholds["min_trades"],
            "passed": trades > thresholds["min_trades"],
        }

        # ratchet_saves > 15%
        rs = results.get("ratchet_saves_pct", 0.0)
        checks["ratchet_saves_pct"] = {
            "value": rs,
            "threshold": thresholds["ratchet_saves_pct"],
            "passed": rs > thresholds["ratchet_saves_pct"],
        }

        all_passed = all(c["passed"] for c in checks.values())

        for name, check in checks.items():
            status = "passed" if check["passed"] else "FAILED"
            logger.info(
                "Go-live %s: %s = %.2f (threshold: %.2f)",
                status, name, check["value"], check["threshold"],
            )

        return {"passed": all_passed, "checks": checks}

    # ------------------------------------------------------------------
    # Data slicing
    # ------------------------------------------------------------------

    @staticmethod
    def _split_data(
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
    def _aggregate_results(
        fold_results: list[dict],
        all_trades: list[dict],
        all_equity: list[dict],
    ) -> dict:
        """Combine multiple fold results into a single aggregate dict."""
        if not all_trades:
            return {
                "total_trades": 0,
                "winners": 0,
                "losers": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_pnl": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "ratchet_saves_pct": 0.0,
                "equity_curve": [],
                "trades": [],
                "fold_count": len(fold_results),
            }

        total = len(all_trades)
        winners = [t for t in all_trades if t["profit_usd"] > 0]
        losers = [t for t in all_trades if t["profit_usd"] <= 0]

        gross_profit = sum(t["profit_usd"] for t in winners)
        gross_loss = abs(sum(t["profit_usd"] for t in losers))

        # Max drawdown: worst across any fold
        max_dd = max(
            (f.get("max_drawdown_pct", 0.0) for f in fold_results), default=0.0,
        )

        # Average Sharpe across folds (weighted by trade count)
        sharpe_weighted = 0.0
        total_fold_trades = 0
        for f in fold_results:
            ft = f.get("total_trades", 0)
            sharpe_weighted += f.get("sharpe_ratio", 0.0) * ft
            total_fold_trades += ft
        avg_sharpe = (
            sharpe_weighted / total_fold_trades if total_fold_trades > 0 else 0.0
        )

        # Ratchet saves aggregation
        ratchet_saves = sum(
            1 for t in all_trades
            if t.get("ratchet_triggered", False)
            and t.get("profit_usd", 0) >= 0
            and t.get("close_reason", "") == "stop_loss"
        )
        ratchet_saves_pct = ratchet_saves / total * 100 if total > 0 else 0.0

        return {
            "total_trades": total,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round(len(winners) / total * 100, 2),
            "profit_factor": round(
                gross_profit / gross_loss, 3,
            ) if gross_loss > 0 else float("inf"),
            "total_pnl": round(sum(t["profit_usd"] for t in all_trades), 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(avg_sharpe, 3),
            "ratchet_saves_pct": round(ratchet_saves_pct, 2),
            "equity_curve": all_equity,
            "trades": all_trades,
            "fold_count": len(fold_results),
        }
