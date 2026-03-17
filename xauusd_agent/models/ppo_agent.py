"""
PPO reinforcement-learning agent wrapper around a Stable-Baselines3 checkpoint.

Provides a thin predict/load/save API consumed by the ensemble decision layer.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# SB3 discrete action mapping → trade direction
_ACTION_MAP: dict[int, int] = {0: -1, 1: 0, 2: 1}  # sell / hold / buy


class PPOAgent:
    """Wrapper for a pre-trained PPO model stored as a ``.zip`` checkpoint."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model: PPO | None = None
        self.model_path: str | None = model_path

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self, path: str | None = None) -> None:
        """Load a PPO checkpoint from *path* (or ``self.model_path``).

        Raises nothing on missing file — logs a warning and leaves
        ``self.model`` as ``None`` so callers can degrade gracefully.
        """
        load_path = path or self.model_path
        if load_path is None:
            logger.warning("No PPO model path provided; skipping load")
            return

        try:
            self.model = PPO.load(load_path)
            self.model_path = load_path
            logger.info("PPO model loaded from %s", load_path)
        except FileNotFoundError:
            logger.warning("PPO checkpoint not found at %s", load_path)
        except Exception:
            logger.exception("Failed to load PPO model from %s", load_path)

    def save(self, path: str | None = None) -> None:
        """Persist the current model to disk."""
        save_path = path or self.model_path
        if self.model is None:
            logger.warning("No PPO model in memory; nothing to save")
            return
        if save_path is None:
            logger.warning("No save path specified for PPO model")
            return

        self.model.save(save_path)
        logger.info("PPO model saved to %s", save_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, observation: np.ndarray) -> tuple[int, float]:
        """Return ``(action, confidence)`` for a single observation.

        Parameters
        ----------
        observation:
            1-D feature vector matching the model's observation space.

        Returns
        -------
        action:
            ``-1`` (sell), ``0`` (hold), or ``1`` (buy).
        confidence:
            Probability assigned to the chosen action (0.0 – 1.0).
        """
        if self.model is None:
            logger.warning("PPO model not loaded; returning hold with zero confidence")
            return 0, 0.0

        # Ensure observation is 2-D (batch dim required by SB3)
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        # Deterministic greedy action
        raw_action, _states = self.model.predict(observation, deterministic=True)
        action_idx = int(raw_action[0]) if np.ndim(raw_action) > 0 else int(raw_action)

        # Confidence from the policy's action distribution
        confidence = self._action_confidence(observation, action_idx)

        mapped_action = _ACTION_MAP.get(action_idx, 0)
        logger.debug(
            "PPO predict: raw=%d mapped=%d confidence=%.4f",
            action_idx,
            mapped_action,
            confidence,
        )
        return mapped_action, confidence

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _action_confidence(self, observation: np.ndarray, action_idx: int) -> float:
        """Compute softmax probability for *action_idx*."""
        try:
            import torch

            obs_tensor = torch.as_tensor(observation, dtype=torch.float32).to(
                self.model.policy.device
            )
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.detach().cpu().numpy().flatten()
            return float(probs[action_idx])
        except Exception:
            logger.debug("Could not compute action probabilities; defaulting to 1.0")
            return 1.0
