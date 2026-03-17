"""
Monthly PPO fine-tuner.

Loads the latest PPO checkpoint, constructs a mini-environment from recent
trade data, fine-tunes for a fixed number of steps, and promotes the new
checkpoint only if cumulative reward improves.

NEVER trains from scratch — always starts from an existing checkpoint.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

_FINETUNE_STEPS = 100_000
_EVAL_EPISODES = 20


class _ReplayTradeEnv(gym.Env):
    """Minimal gym environment that replays historical trade sequences.

    Each step presents a feature observation and the agent chooses
    buy / hold / sell.  Reward is the actual ``profit_r`` of the trade if
    the action matches the recorded direction, else 0.
    """

    metadata = {"render_modes": []}

    def __init__(self, trades: list[dict], feature_dim: int) -> None:
        super().__init__()
        self.trades = trades
        self._idx = 0

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)  # 0=sell, 1=hold, 2=buy

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._idx = 0
        return self._obs(), {}

    def step(self, action: int):
        trade = self.trades[self._idx % len(self.trades)]
        direction = trade.get("direction", 0)  # 1=long, -1=short
        profit_r = trade.get("profit_r", 0.0)

        # Map action: 0→-1 (sell), 1→0 (hold), 2→1 (buy)
        agent_dir = action - 1

        # Reward: full profit_r if direction matched, penalty for wrong direction
        if agent_dir == direction:
            reward = profit_r
        elif agent_dir == 0:
            reward = 0.0
        else:
            reward = -abs(profit_r) * 0.5

        self._idx += 1
        terminated = self._idx >= len(self.trades)
        truncated = False
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self) -> np.ndarray:
        trade = self.trades[self._idx % len(self.trades)]
        snapshot = trade.get("feature_snapshot", [])
        if isinstance(snapshot, str):
            import json
            try:
                snapshot = json.loads(snapshot)
            except (json.JSONDecodeError, TypeError):
                snapshot = []
        arr = np.array(snapshot, dtype=np.float32)
        expected = self.observation_space.shape[0]
        if arr.shape[0] < expected:
            arr = np.pad(arr, (0, expected - arr.shape[0]))
        elif arr.shape[0] > expected:
            arr = arr[:expected]
        return arr


class PPOFineTuner:
    """Fine-tunes a PPO checkpoint on recent trade data once a month."""

    def __init__(self, db_pool: Any, checkpoint_dir: str = "models/") -> None:
        self.db = db_pool
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def monthly_finetune(self) -> tuple[bool, str]:
        """Run the monthly fine-tuning pipeline.

        Returns ``(success, summary_message)``.
        """
        # 1. Locate latest checkpoint
        checkpoint_path = self._latest_checkpoint()
        if checkpoint_path is None:
            msg = "No existing PPO checkpoint found; cannot fine-tune."
            logger.warning(msg)
            return False, msg

        # 2. Fetch last 30 days of closed trades
        trades = await self._fetch_recent_trades(days=30)
        if len(trades) < 30:
            msg = f"Only {len(trades)} trades in last 30 days; need >= 30 for fine-tuning."
            logger.info(msg)
            return False, msg

        feature_dim = self._infer_feature_dim(trades)
        if feature_dim == 0:
            msg = "Could not infer feature dimension from trades."
            logger.warning(msg)
            return False, msg

        # 3. Evaluate baseline
        baseline_reward = self._evaluate(checkpoint_path, trades, feature_dim)
        logger.info("Baseline cumulative reward: %.4f", baseline_reward)

        # 4. Fine-tune
        logger.info(
            "Fine-tuning PPO for %d steps on %d trades (feature_dim=%d)",
            _FINETUNE_STEPS, len(trades), feature_dim,
        )
        env = DummyVecEnv([lambda: _ReplayTradeEnv(trades, feature_dim)])
        model = PPO.load(checkpoint_path, env=env)
        model.learn(total_timesteps=_FINETUNE_STEPS, reset_num_timesteps=False)

        # 5. Save candidate
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        candidate_path = str(self.checkpoint_dir / f"ppo_candidate_{ts}")
        model.save(candidate_path)

        # 6. Evaluate candidate
        candidate_reward = self._evaluate(candidate_path, trades, feature_dim)
        logger.info(
            "Candidate cumulative reward: %.4f (baseline: %.4f)",
            candidate_reward, baseline_reward,
        )

        # 7. Promote or revert
        if candidate_reward > baseline_reward:
            promoted_path = str(self.checkpoint_dir / f"ppo_checkpoint_{ts}")
            os.rename(candidate_path + ".zip", promoted_path + ".zip")

            # Update "latest" symlink
            latest_link = self.checkpoint_dir / "ppo_latest.zip"
            latest_link.unlink(missing_ok=True)
            try:
                latest_link.symlink_to(os.path.abspath(promoted_path + ".zip"))
            except OSError:
                logger.warning("Could not create PPO latest symlink")

            msg = (
                f"PPO fine-tuned: reward {baseline_reward:.4f} → {candidate_reward:.4f}. "
                f"Saved to {promoted_path}"
            )
            logger.info(msg)
            await self._log_finetune(True, msg)
            return True, msg
        else:
            # Clean up candidate
            cand_file = Path(candidate_path + ".zip")
            cand_file.unlink(missing_ok=True)

            msg = (
                f"PPO fine-tune did not improve: "
                f"{candidate_reward:.4f} <= {baseline_reward:.4f}. Reverted."
            )
            logger.info(msg)
            await self._log_finetune(False, msg)
            return False, msg

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self, model_path: str, trades: list[dict], feature_dim: int
    ) -> float:
        """Evaluate a PPO checkpoint over *_EVAL_EPISODES* episodes."""
        env = DummyVecEnv([lambda: _ReplayTradeEnv(trades, feature_dim)])
        try:
            model = PPO.load(model_path, env=env)
        except Exception:
            logger.warning("Could not load model at %s for evaluation", model_path)
            return float("-inf")

        total_reward = 0.0
        for _ in range(_EVAL_EPISODES):
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done_arr, _info = env.step(action)
                total_reward += float(reward[0])
                done = bool(done_arr[0])

        return total_reward / _EVAL_EPISODES

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _latest_checkpoint(self) -> str | None:
        """Locate the latest PPO checkpoint in the checkpoint directory."""
        latest_link = self.checkpoint_dir / "ppo_latest.zip"
        if latest_link.exists():
            return str(latest_link).removesuffix(".zip")

        # Fallback: find most recent ppo_checkpoint_*.zip
        candidates = sorted(self.checkpoint_dir.glob("ppo_checkpoint_*.zip"))
        if candidates:
            return str(candidates[-1]).removesuffix(".zip")
        return None

    @staticmethod
    def _infer_feature_dim(trades: list[dict]) -> int:
        """Infer observation dimension from the first valid trade snapshot."""
        import json as _json

        for t in trades:
            snapshot = t.get("feature_snapshot")
            if snapshot is None:
                continue
            if isinstance(snapshot, str):
                try:
                    snapshot = _json.loads(snapshot)
                except (ValueError, TypeError):
                    continue
            if isinstance(snapshot, (list, tuple)) and len(snapshot) > 0:
                return len(snapshot)
        return 0

    async def _fetch_recent_trades(self, days: int = 30) -> list[dict]:
        query = """
            SELECT *
            FROM   trades
            WHERE  status = 'closed'
              AND  closed_at >= NOW() - INTERVAL '$1 days'
            ORDER  BY closed_at
        """
        async with self.db.acquire() as conn:
            rows = await conn.fetch(query.replace("$1", str(days)))
        return [dict(r) for r in rows]

    async def _log_finetune(self, success: bool, summary: str) -> None:
        query = """
            INSERT INTO learning_log
                (trigger_type, param_name, old_value, new_value, reason, trades_analysed, win_rate)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        try:
            async with self.db.acquire() as conn:
                await conn.execute(
                    query,
                    "ppo_finetune",
                    "ppo_model",
                    0.0,
                    1.0 if success else 0.0,
                    summary,
                    0,
                    0.0,
                )
        except Exception:
            logger.exception("Failed to log PPO fine-tune event")
