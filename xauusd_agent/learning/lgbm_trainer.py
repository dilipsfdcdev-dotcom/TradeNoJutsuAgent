"""
Weekly LightGBM retrainer.

Pulls recent trades, merges with a rolling historical window, trains a new
binary classifier, validates against a held-out set, and (if improved)
exports both native and ONNX checkpoints.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np

from xauusd_agent.infra.logger import get_logger
from xauusd_agent.learning.trade_labeller import TradeLabeller

logger = get_logger(__name__)

# Default LightGBM hyper-parameters
_DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

_MIN_TRADES_FOR_RETRAIN = 50
_HISTORICAL_WINDOW = 500
_HOLDOUT_SIZE = 20
_IMPROVEMENT_THRESHOLD = 0.02  # 2 % accuracy lift required


class LGBMTrainer:
    """Orchestrates weekly LightGBM retraining from trade history."""

    def __init__(self, db_pool, model_dir: str = "models/") -> None:
        self.db = db_pool
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._labeller = TradeLabeller()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def weekly_retrain(self) -> tuple[bool, str]:
        """Run the full retrain pipeline.

        Returns ``(success, summary_message)``.
        """
        # 1. Fetch recent trades since last retrain
        recent = await self._fetch_trades_since_last_retrain()
        if len(recent) < _MIN_TRADES_FOR_RETRAIN:
            msg = (
                f"Only {len(recent)} trades since last retrain "
                f"(need {_MIN_TRADES_FOR_RETRAIN}); skipping."
            )
            logger.info(msg)
            return False, msg

        # 2. Merge with rolling historical window
        historical = await self._fetch_historical_trades(limit=_HISTORICAL_WINDOW)
        all_trades = self._deduplicate(historical + recent)

        # 3. Build training dataset
        X, y = self._labeller.get_training_dataset(all_trades)
        if X.shape[0] < _MIN_TRADES_FOR_RETRAIN:
            msg = f"Insufficient labelled samples ({X.shape[0]}); skipping."
            logger.info(msg)
            return False, msg

        # 4. Hold out last N for validation
        X_train, y_train = X[:-_HOLDOUT_SIZE], y[:-_HOLDOUT_SIZE]
        X_val, y_val = X[-_HOLDOUT_SIZE:], y[-_HOLDOUT_SIZE:]

        # 5. Train
        logger.info(
            "Training LightGBM: %d train / %d val samples, %d features",
            len(y_train), len(y_val), X_train.shape[1],
        )
        train_ds = lgb.Dataset(X_train, label=(y_train >= 0.5).astype(float))
        val_ds = lgb.Dataset(X_val, label=(y_val >= 0.5).astype(float), reference=train_ds)

        model = lgb.train(
            _DEFAULT_PARAMS,
            train_ds,
            num_boost_round=300,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)],
        )

        # 6. Evaluate new model
        new_preds = model.predict(X_val)
        new_acc = float(((new_preds >= 0.5) == (y_val >= 0.5)).mean())

        # 7. Compare with previous model
        prev_acc = self._evaluate_previous_model(X_val, y_val)
        improvement = new_acc - prev_acc

        logger.info(
            "Validation accuracy: new=%.4f  prev=%.4f  delta=%.4f",
            new_acc, prev_acc, improvement,
        )

        if improvement < _IMPROVEMENT_THRESHOLD and prev_acc > 0:
            msg = (
                f"New model ({new_acc:.2%}) did not beat previous "
                f"({prev_acc:.2%}) by >= {_IMPROVEMENT_THRESHOLD:.0%}; keeping old."
            )
            logger.info(msg)
            await self._log_retrain(success=False, summary=msg)
            return False, msg

        # 8. Save new model + ONNX export
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        native_path = str(self.model_dir / f"lgbm_signal_{ts}.txt")
        onnx_path = str(self.model_dir / f"lgbm_signal_{ts}.onnx")

        model.save_model(native_path)
        self._export_onnx(model, X_train.shape[1], onnx_path)

        # Symlink "latest"
        latest_native = self.model_dir / "lgbm_signal_latest.txt"
        latest_onnx = self.model_dir / "lgbm_signal_latest.onnx"
        self._safe_symlink(native_path, latest_native)
        self._safe_symlink(onnx_path, latest_onnx)

        msg = (
            f"Retrained LightGBM: acc {new_acc:.2%} (+{improvement:.2%}). "
            f"Saved to {native_path}"
        )
        logger.info(msg)
        await self._log_retrain(success=True, summary=msg)
        return True, msg

    # ------------------------------------------------------------------
    # ONNX export
    # ------------------------------------------------------------------

    def _export_onnx(self, model: lgb.Booster, feature_count: int, output_path: str) -> None:
        """Export a LightGBM Booster to ONNX via ``onnxmltools``."""
        try:
            import onnxmltools
            from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
                convert_lightgbm,  # noqa: F401 – imported for side-effect registration
            )
            from skl2onnx.common.data_types import FloatTensorType

            initial_type = [("input", FloatTensorType([None, feature_count]))]
            onnx_model = onnxmltools.convert_lightgbm(
                model, initial_types=initial_type, target_opset=11
            )
            onnxmltools.utils.save_model(onnx_model, output_path)
            logger.info("ONNX model exported to %s", output_path)
        except ImportError:
            logger.warning(
                "onnxmltools / skl2onnx not installed; skipping ONNX export"
            )
        except Exception:
            logger.exception("ONNX export failed for %s", output_path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate_previous_model(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Load the latest native model and compute accuracy on *X_val*."""
        latest = self.model_dir / "lgbm_signal_latest.txt"
        if not latest.exists():
            return 0.0
        try:
            prev = lgb.Booster(model_file=str(latest))
            preds = prev.predict(X_val)
            return float(((preds >= 0.5) == (y_val >= 0.5)).mean())
        except Exception:
            logger.warning("Could not evaluate previous model", exc_info=True)
            return 0.0

    async def _fetch_trades_since_last_retrain(self) -> list[dict]:
        query = """
            SELECT *
            FROM   trades
            WHERE  status = 'closed'
              AND  closed_at > COALESCE(
                       (SELECT MAX(created_at) FROM learning_log
                        WHERE trigger_type = 'lgbm_retrain'), '1970-01-01')
            ORDER  BY closed_at
        """
        async with self.db.acquire() as conn:
            rows = await conn.fetch(query)
        return [dict(r) for r in rows]

    async def _fetch_historical_trades(self, limit: int) -> list[dict]:
        query = """
            SELECT *
            FROM   trades
            WHERE  status = 'closed'
            ORDER  BY closed_at DESC
            LIMIT  $1
        """
        async with self.db.acquire() as conn:
            rows = await conn.fetch(query, limit)
        return [dict(r) for r in reversed(rows)]

    async def _log_retrain(self, success: bool, summary: str) -> None:
        query = """
            INSERT INTO learning_log
                (trigger_type, param_name, old_value, new_value, reason, trades_analysed, win_rate)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """
        try:
            async with self.db.acquire() as conn:
                await conn.execute(
                    query,
                    "lgbm_retrain",
                    "lgbm_model",
                    0.0,
                    1.0 if success else 0.0,
                    summary,
                    0,
                    0.0,
                )
        except Exception:
            logger.exception("Failed to log retrain event")

    @staticmethod
    def _deduplicate(trades: list[dict]) -> list[dict]:
        seen: set = set()
        unique: list[dict] = []
        for t in trades:
            tid = t.get("id")
            if tid and tid not in seen:
                seen.add(tid)
                unique.append(t)
        return unique

    @staticmethod
    def _safe_symlink(target: str, link: Path) -> None:
        try:
            link.unlink(missing_ok=True)
            link.symlink_to(os.path.abspath(target))
        except OSError:
            logger.warning("Could not create symlink %s → %s", link, target)
