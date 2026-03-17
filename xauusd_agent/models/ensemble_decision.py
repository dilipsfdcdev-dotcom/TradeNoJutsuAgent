"""
Ensemble decision layer that fuses PPO and LightGBM predictions.

Keeps the combination logic in one place so callers get a single dict
with all model outputs.
"""

from __future__ import annotations

import numpy as np

from xauusd_agent.infra.logger import get_logger
from xauusd_agent.models.lgbm_signal import LGBMSignal
from xauusd_agent.models.ppo_agent import PPOAgent

logger = get_logger(__name__)


def ensemble_predict(
    ppo_agent: PPOAgent,
    lgbm_signal: LGBMSignal,
    features: np.ndarray,
    observation: np.ndarray,
) -> dict:
    """Run both models and return a combined prediction dictionary.

    Parameters
    ----------
    ppo_agent:
        Loaded :class:`PPOAgent` instance.
    lgbm_signal:
        Loaded :class:`LGBMSignal` instance.
    features:
        1-D feature vector for the LightGBM model.
    observation:
        1-D observation vector for the PPO model.

    Returns
    -------
    dict with keys:
        * ``ppo_action``     – int (-1 / 0 / 1)
        * ``ppo_confidence`` – float 0.0–1.0
        * ``lgbm_prob``      – float 0.0–1.0 (buy probability)
    """
    # --- PPO prediction ---
    try:
        ppo_action, ppo_confidence = ppo_agent.predict(observation)
    except Exception:
        logger.exception("PPO prediction failed; defaulting to hold")
        ppo_action, ppo_confidence = 0, 0.0

    # --- LightGBM prediction ---
    try:
        lgbm_prob = lgbm_signal.predict_proba(features)
    except Exception:
        logger.exception("LightGBM prediction failed; defaulting to 0.5")
        lgbm_prob = 0.5

    result = {
        "ppo_action": ppo_action,
        "ppo_confidence": round(ppo_confidence, 4),
        "lgbm_prob": round(lgbm_prob, 4),
    }

    logger.debug("Ensemble result: %s", result)
    return result
