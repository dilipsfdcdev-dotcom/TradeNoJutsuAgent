"""
Trade labeller — converts completed trades into labelled training samples
for the LightGBM signal model.
"""

from __future__ import annotations

import json

import numpy as np

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)


class TradeLabeller:
    """Assigns quality labels to completed trades and builds training datasets."""

    def label_trade(self, trade: dict) -> dict:
        """Attach a ``label`` field to a completed trade dict.

        Labelling scheme
        ----------------
        * **1.0** — ``profit_R >= 1.0`` (full winner).
        * **0.5** — ``0 < profit_R < 1.0`` (partial winner).
        * **0.0** — ``profit_R <= 0`` (loser).

        Returns the same *trade* dict with the ``label`` key added.
        """
        profit_r = trade.get("profit_r", 0.0)

        if profit_r >= 1.0:
            label = 1.0
        elif profit_r > 0:
            label = 0.5
        else:
            label = 0.0

        trade["label"] = label
        return trade

    def get_training_dataset(
        self, trades: list[dict]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract ``(X, y)`` arrays from a list of completed trades.

        Each trade must contain a ``feature_snapshot`` (JSON string or list/array)
        and will be labelled via :meth:`label_trade` if not already labelled.

        Returns
        -------
        X : np.ndarray
            Feature matrix of shape ``(n_trades, n_features)``.
        y : np.ndarray
            Label vector of shape ``(n_trades,)``.
        """
        features_list: list[list[float]] = []
        labels: list[float] = []

        for trade in trades:
            snapshot = trade.get("feature_snapshot")
            if snapshot is None:
                logger.debug(
                    "Skipping trade %s — no feature_snapshot",
                    trade.get("id", "?"),
                )
                continue

            # Parse snapshot
            if isinstance(snapshot, str):
                try:
                    snapshot = json.loads(snapshot)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Invalid feature_snapshot JSON for trade %s",
                        trade.get("id", "?"),
                    )
                    continue

            if isinstance(snapshot, np.ndarray):
                snapshot = snapshot.tolist()

            if not isinstance(snapshot, (list, tuple)) or len(snapshot) == 0:
                logger.debug(
                    "Empty/invalid feature_snapshot for trade %s",
                    trade.get("id", "?"),
                )
                continue

            # Ensure label exists
            if "label" not in trade:
                self.label_trade(trade)

            features_list.append([float(v) for v in snapshot])
            labels.append(float(trade["label"]))

        if not features_list:
            logger.warning("No valid trades for training dataset")
            return np.empty((0, 0)), np.empty((0,))

        X = np.array(features_list, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        logger.info(
            "Training dataset built: X=%s  y=%s  positive_rate=%.2f",
            X.shape,
            y.shape,
            float((y > 0).mean()) if len(y) else 0.0,
        )
        return X, y
