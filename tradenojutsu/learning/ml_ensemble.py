"""ML Ensemble - LightGBM + PPO models for signal generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tradenojutsu.data.models import Direction
from tradenojutsu.infra.logger import get_logger

logger = get_logger("learning.ml_ensemble")

MODEL_DIR = Path("data/models")


class MLEnsemble:
    """Machine learning ensemble combining gradient boosting and RL.

    Components:
    - LightGBM: Supervised model trained on labeled historical trades
    - PPO (Stable-Baselines3): RL agent trained on market environment

    Both models output a directional signal + confidence score.
    """

    def __init__(self):
        self.lgbm_model = None
        self.ppo_model = None
        self._feature_columns: list[str] = []
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def predict(self, df: pd.DataFrame) -> tuple[float, Direction]:
        """Get ensemble prediction from available models.

        Returns (score 0-100, direction).
        Falls back to technical baseline if no models trained.
        """
        scores = []
        directions = []

        # LightGBM prediction
        if self.lgbm_model is not None:
            lgbm_score, lgbm_dir = self._predict_lgbm(df)
            scores.append(lgbm_score)
            directions.append(lgbm_dir)

        # PPO prediction
        if self.ppo_model is not None:
            ppo_score, ppo_dir = self._predict_ppo(df)
            scores.append(ppo_score)
            directions.append(ppo_dir)

        if not scores:
            # No trained models yet - return neutral
            return 50.0, Direction.FLAT

        # Average scores
        avg_score = np.mean(scores)

        # Majority direction
        long_count = sum(1 for d in directions if d == Direction.LONG)
        short_count = sum(1 for d in directions if d == Direction.SHORT)
        if long_count > short_count:
            direction = Direction.LONG
        elif short_count > long_count:
            direction = Direction.SHORT
        else:
            direction = Direction.FLAT

        return float(avg_score), direction

    def train_lgbm(self, features: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
        """Train/retrain the LightGBM model on labeled trade data.

        Args:
            features: Technical indicator features
            labels: 1 for profitable trades, 0 for losing trades

        Returns metrics dict.
        """
        import lightgbm as lgb
        from sklearn.model_selection import TimeSeriesSplit

        self._feature_columns = list(features.columns)

        tscv = TimeSeriesSplit(n_splits=5)
        metrics = {"accuracy": [], "auc": []}

        for train_idx, val_idx in tscv.split(features):
            X_train = features.iloc[train_idx]
            y_train = labels.iloc[train_idx]
            X_val = features.iloc[val_idx]
            y_val = labels.iloc[val_idx]

            train_set = lgb.Dataset(X_train, y_train)
            val_set = lgb.Dataset(X_val, y_val, reference=train_set)

            params = {
                "objective": "binary",
                "metric": "auc",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "verbose": -1,
            }

            model = lgb.train(
                params, train_set, num_boost_round=500,
                valid_sets=[val_set],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            preds = model.predict(X_val)
            acc = np.mean((preds > 0.5) == y_val)
            metrics["accuracy"].append(acc)

        self.lgbm_model = model
        model.save_model(str(MODEL_DIR / "lgbm_signal.txt"))

        avg_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
        logger.info(f"LightGBM trained: {avg_metrics}")
        return avg_metrics

    def train_ppo(self, env, total_timesteps: int = 100_000) -> dict[str, float]:
        """Train/finetune the PPO agent on a trading environment.

        Args:
            env: Gymnasium environment wrapping the trading simulation
            total_timesteps: Number of training steps

        Returns metrics dict.
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback

        model_path = MODEL_DIR / "ppo_agent"

        if self.ppo_model is not None:
            # Continue training existing model
            self.ppo_model.set_env(env)
        else:
            self.ppo_model = PPO(
                "MlpPolicy", env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=0,
            )

        self.ppo_model.learn(total_timesteps=total_timesteps)
        self.ppo_model.save(str(model_path))

        logger.info(f"PPO trained for {total_timesteps} steps")
        return {"timesteps": total_timesteps}

    def load_models(self) -> None:
        """Load pre-trained models from disk."""
        lgbm_path = MODEL_DIR / "lgbm_signal.txt"
        if lgbm_path.exists():
            import lightgbm as lgb
            self.lgbm_model = lgb.Booster(model_file=str(lgbm_path))
            logger.info("Loaded LightGBM model")

        ppo_path = MODEL_DIR / "ppo_agent.zip"
        if ppo_path.exists():
            from stable_baselines3 import PPO
            self.ppo_model = PPO.load(str(ppo_path))
            logger.info("Loaded PPO model")

    def _predict_lgbm(self, df: pd.DataFrame) -> tuple[float, Direction]:
        """Get LightGBM prediction."""
        try:
            # Use latest row of features
            features = df[self._feature_columns].iloc[-1:] if self._feature_columns else df.iloc[-1:]
            prob = self.lgbm_model.predict(features)[0]
            score = prob * 100
            direction = Direction.LONG if prob > 0.5 else Direction.SHORT
            return float(score), direction
        except Exception as e:
            logger.warning(f"LightGBM prediction failed: {e}")
            return 50.0, Direction.FLAT

    def _predict_ppo(self, df: pd.DataFrame) -> tuple[float, Direction]:
        """Get PPO prediction."""
        try:
            # Build observation from latest data
            obs = self._df_to_observation(df)
            action, _ = self.ppo_model.predict(obs, deterministic=True)
            # action: 0=hold, 1=long, 2=short
            if action == 1:
                return 70.0, Direction.LONG
            elif action == 2:
                return 70.0, Direction.SHORT
            return 50.0, Direction.FLAT
        except Exception as e:
            logger.warning(f"PPO prediction failed: {e}")
            return 50.0, Direction.FLAT

    def _df_to_observation(self, df: pd.DataFrame) -> np.ndarray:
        """Convert DataFrame to a flat observation vector for PPO."""
        cols = ["close", "rsi", "macd", "adx", "atr_pct", "volume_ratio", "bb_width"]
        available = [c for c in cols if c in df.columns]
        if not available:
            return np.zeros(10, dtype=np.float32)
        obs = df[available].iloc[-1].values.astype(np.float32)
        # Normalize
        obs = np.nan_to_num(obs, nan=0.0)
        return obs
