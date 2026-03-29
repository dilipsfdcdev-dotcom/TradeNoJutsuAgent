"""Self-Learning Module - Reinforcement learning that adapts strategy parameters.

This is the "self-learning" core. After every N trades, it:
1. Evaluates recent performance metrics
2. Asks the AI Brain for qualitative reflection
3. Uses a Q-learning approach to adjust parameters
4. Tracks what changes worked and what didn't
"""

from __future__ import annotations

import json
import random
from typing import Any

import numpy as np

from tradenojutsu.brain.reasoner import AIReasoner
from tradenojutsu.config import load_settings, save_settings
from tradenojutsu.data.models import PerformanceMetrics
from tradenojutsu.infra.database import (
    get_recent_trades,
    insert_learning_log,
    insert_performance_snapshot,
)
from tradenojutsu.infra.logger import get_logger

logger = get_logger("learning.self_learner")


# Parameters that the agent can tune, with their valid ranges
TUNABLE_PARAMS = {
    "risk.risk_per_trade_pct": (0.5, 2.0, 0.1),    # (min, max, step)
    "risk.max_concurrent_positions": (1, 5, 1),
    "risk.max_daily_trades": (5, 25, 1),
    "risk.stop_loss_atr_mult": (0.8, 2.5, 0.1),
    "risk.take_profit_rr_ratio": (1.0, 4.0, 0.25),
    "risk.trailing_stop_atr_mult": (0.5, 2.0, 0.1),
    "signals.entry_threshold": (50, 85, 2.5),
    "signals.model_weights.technical": (0.1, 0.6, 0.05),
    "signals.model_weights.ml_ensemble": (0.1, 0.6, 0.05),
    "signals.model_weights.llm_reasoning": (0.1, 0.6, 0.05),
    "signals.min_confluence": (1, 4, 1),
    "learning.exploration_rate": (0.01, 0.3, 0.01),
}


def _get_nested(d: dict, key: str) -> Any:
    """Get a nested dict value using dot notation."""
    parts = key.split(".")
    for part in parts:
        d = d[part]
    return d


def _set_nested(d: dict, key: str, value: Any) -> None:
    """Set a nested dict value using dot notation."""
    parts = key.split(".")
    for part in parts[:-1]:
        d = d[part]
    d[parts[-1]] = value


class SelfLearner:
    """Autonomous self-learning module that improves trading over time.

    Uses a hybrid approach:
    - Q-learning for parameter adjustments (which direction to tune)
    - LLM reflection for qualitative strategy insights
    - Performance tracking to measure impact of changes
    """

    def __init__(
        self,
        reasoner: AIReasoner | None = None,
        cycle_every_n: int = 10,
        max_changes_per_cycle: int = 3,
        reward_metric: str = "sharpe",
    ):
        self.reasoner = reasoner
        self.cycle_every_n = cycle_every_n
        self.max_changes_per_cycle = max_changes_per_cycle
        self.reward_metric = reward_metric
        self.cycle_count = 0
        self.trade_count_since_cycle = 0

        # Q-table: param -> direction (up/down) -> estimated value
        self.q_table: dict[str, dict[str, float]] = {
            param: {"up": 0.0, "down": 0.0} for param in TUNABLE_PARAMS
        }
        self.exploration_rate = 0.1
        self.learning_rate = 0.1
        self.discount = 0.95

        # Track previous performance for reward calculation
        self._prev_metrics: PerformanceMetrics | None = None

    def on_trade_closed(self) -> dict[str, Any] | None:
        """Called after each trade closes. Triggers learning cycle if threshold met.

        Returns learning cycle results if a cycle ran, None otherwise.
        """
        self.trade_count_since_cycle += 1
        if self.trade_count_since_cycle >= self.cycle_every_n:
            return self.run_learning_cycle()
        return None

    def run_learning_cycle(self) -> dict[str, Any]:
        """Execute a full learning cycle.

        1. Calculate current performance metrics
        2. Compute reward signal
        3. Get AI reflection (qualitative insights)
        4. Select and apply parameter adjustments
        5. Log everything
        """
        self.cycle_count += 1
        self.trade_count_since_cycle = 0
        logger.info(f"=== Learning Cycle #{self.cycle_count} ===")

        # 1. Evaluate current performance
        trades = get_recent_trades(100)
        metrics = self._compute_metrics(trades)
        insert_performance_snapshot({
            "total_trades": metrics.total_trades,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "total_pnl": metrics.total_pnl,
        })

        # 2. Calculate reward
        reward = self._compute_reward(metrics)
        logger.info(f"Performance: {metrics.summary()} | Reward: {reward:.3f}")

        # 3. Get AI reflection
        reflection = ""
        if self.reasoner and trades:
            reflection = self.reasoner.reflect_on_trades(trades, metrics)
            logger.info(f"AI Reflection: {reflection[:200]}...")

        # 4. Update Q-values based on reward
        self._update_q_values(reward)

        # 5. Select and apply parameter changes
        changes = self._select_changes(reflection)
        settings = load_settings()

        applied_changes = []
        for param, direction in changes[:self.max_changes_per_cycle]:
            old_value = _get_nested(settings, param)
            new_value = self._compute_new_value(param, old_value, direction)

            if new_value != old_value:
                _set_nested(settings, param, new_value)
                applied_changes.append({
                    "param": param,
                    "old": old_value,
                    "new": new_value,
                    "direction": direction,
                })
                insert_learning_log(
                    cycle_num=self.cycle_count,
                    param=param,
                    old_val=str(old_value),
                    new_val=str(new_value),
                    reason=f"Q-value: {self.q_table[param][direction]:.3f}",
                    perf_before={"reward": reward},
                    perf_after={},  # Will be filled next cycle
                )
                logger.info(f"  Adjusted {param}: {old_value} -> {new_value} ({direction})")

        if applied_changes:
            save_settings(settings)

        self._prev_metrics = metrics

        result = {
            "cycle": self.cycle_count,
            "metrics": metrics.summary(),
            "reward": reward,
            "changes": applied_changes,
            "reflection_excerpt": reflection[:500] if reflection else "",
        }
        logger.info(f"Learning cycle complete: {len(applied_changes)} changes applied")
        return result

    def _compute_metrics(self, trades: list[dict]) -> PerformanceMetrics:
        """Calculate performance metrics from trade history."""
        if not trades:
            return PerformanceMetrics()

        closed = [t for t in trades if t.get("status") == "closed" and t.get("pnl") is not None]
        if not closed:
            return PerformanceMetrics()

        pnls = [t["pnl"] for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_trades = len(closed)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")

        # Sharpe ratio (simplified)
        if len(pnls) > 1:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = np.max(drawdowns) / (running_max[-1] if running_max[-1] > 0 else 1)

        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            profit_factor=float(profit_factor),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            total_pnl=float(sum(pnls)),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            expectancy=float(expectancy),
        )

    def _compute_reward(self, metrics: PerformanceMetrics) -> float:
        """Compute scalar reward for the RL update."""
        if self.reward_metric == "sharpe":
            return metrics.sharpe_ratio
        elif self.reward_metric == "profit":
            return metrics.total_pnl
        elif self.reward_metric == "calmar":
            if metrics.max_drawdown > 0:
                return metrics.total_pnl / metrics.max_drawdown
            return metrics.total_pnl
        return 0.0

    def _update_q_values(self, reward: float) -> None:
        """Update Q-table based on the observed reward."""
        prev_reward = 0.0
        if self._prev_metrics:
            prev_reward = self._compute_reward(self._prev_metrics)

        delta = reward - prev_reward  # improvement signal

        # Update all Q-values with temporal difference
        for param in self.q_table:
            for direction in ("up", "down"):
                old_q = self.q_table[param][direction]
                self.q_table[param][direction] = (
                    old_q + self.learning_rate * (delta - old_q)
                )

    def _select_changes(self, reflection: str = "") -> list[tuple[str, str]]:
        """Select which parameters to adjust and in which direction.

        Uses epsilon-greedy strategy with AI reflection as a tiebreaker.
        """
        candidates = []

        for param in TUNABLE_PARAMS:
            if random.random() < self.exploration_rate:
                # Explore: random direction
                direction = random.choice(["up", "down"])
            else:
                # Exploit: pick best Q-value direction
                q_up = self.q_table[param]["up"]
                q_down = self.q_table[param]["down"]
                direction = "up" if q_up >= q_down else "down"

            q_val = self.q_table[param][direction]
            candidates.append((param, direction, abs(q_val)))

        # Sort by Q-value magnitude (most impactful changes first)
        candidates.sort(key=lambda x: x[2], reverse=True)

        return [(p, d) for p, d, _ in candidates[:self.max_changes_per_cycle]]

    def _compute_new_value(self, param: str, current: Any, direction: str) -> Any:
        """Calculate the new parameter value within bounds."""
        min_val, max_val, step = TUNABLE_PARAMS[param]

        if direction == "up":
            new_val = current + step
        else:
            new_val = current - step

        # Clamp to bounds
        new_val = max(min_val, min(max_val, new_val))

        # Match type of original
        if isinstance(current, int):
            return int(round(new_val))
        return round(new_val, 4)
