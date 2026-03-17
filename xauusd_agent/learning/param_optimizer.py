"""
Parameter weight optimizer.

Analyses which predictive signal (signal score, PPO, LightGBM) has the
best track record and shifts ensemble weights accordingly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

_WEIGHT_STEP = 0.05
_MIN_WEIGHT = 0.10  # no model can go below 10 %


class ParamOptimizer:
    """Auto-tune ensemble weights from trade history."""

    def optimize_weights(self, trades: list[dict]) -> dict[str, float] | None:
        """Compute updated model weights based on predictive accuracy.

        For each completed trade the method checks whether the model's
        prediction was directionally correct:

        * **signal_score** — ``>= 65`` predicted buy, ``< 65`` predicted sell.
        * **ppo_action** — ``1`` buy, ``-1`` sell.
        * **lgbm_prob** — ``>= 0.5`` buy, ``< 0.5`` sell.

        The model with the highest hit rate receives ``+_WEIGHT_STEP``
        and the weakest receives ``-_WEIGHT_STEP``.  Weights are clamped
        to ``[_MIN_WEIGHT, 1.0]`` and normalised to sum to 1.0.

        Parameters
        ----------
        trades:
            List of trade dicts, each containing at minimum
            ``signal_score``, ``ppo_action``, ``lgbm_prob``, ``direction``,
            and ``profit_r``.

        Returns
        -------
        ``dict`` with keys ``signal_weight``, ``ppo_weight``, ``lgbm_weight``
        or ``None`` if there is insufficient data.
        """
        if len(trades) < 20:
            logger.info("Need >= 20 trades to optimise weights; got %d", len(trades))
            return None

        # --- Compute hit rates ---
        hits: dict[str, int] = {"signal": 0, "ppo": 0, "lgbm": 0}
        total = 0

        for t in trades:
            direction = t.get("direction", 0)
            profit_r = t.get("profit_r", 0.0)
            if direction == 0:
                continue

            won = profit_r > 0
            total += 1

            # Signal score
            score = t.get("signal_score", 65)
            sig_pred = 1 if score >= 65 else -1
            if (sig_pred == direction) == won:
                hits["signal"] += 1

            # PPO
            ppo_act = t.get("ppo_action", 0)
            if ppo_act != 0 and (ppo_act == direction) == won:
                hits["ppo"] += 1

            # LightGBM
            lgbm_p = t.get("lgbm_prob", 0.5)
            lgbm_pred = 1 if lgbm_p >= 0.5 else -1
            if (lgbm_pred == direction) == won:
                hits["lgbm"] += 1

        if total == 0:
            return None

        rates = {k: v / total for k, v in hits.items()}
        logger.info("Model hit rates: %s", rates)

        # --- Determine best and worst ---
        sorted_models = sorted(rates, key=rates.get, reverse=True)
        best = sorted_models[0]
        worst = sorted_models[-1]

        if rates[best] == rates[worst]:
            logger.info("All models have equal hit rate; no weight change")
            return None

        # --- Current weights (default equal) ---
        weights = {"signal": 1 / 3, "ppo": 1 / 3, "lgbm": 1 / 3}

        weights[best] += _WEIGHT_STEP
        weights[worst] -= _WEIGHT_STEP

        # Clamp
        for k in weights:
            weights[k] = max(_MIN_WEIGHT, weights[k])

        # Normalise to 1.0
        total_w = sum(weights.values())
        weights = {k: round(v / total_w, 4) for k, v in weights.items()}

        result = {
            "signal_weight": weights["signal"],
            "ppo_weight": weights["ppo"],
            "lgbm_weight": weights["lgbm"],
        }
        logger.info("Optimised weights: %s", result)
        return result
