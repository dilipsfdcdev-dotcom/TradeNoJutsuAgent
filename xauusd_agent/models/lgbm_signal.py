"""
LightGBM signal model with optional ONNX-accelerated inference.

Produces a buy-probability score consumed by the ensemble decision layer.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnxruntime as ort

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)


class LGBMSignal:
    """Wrapper around a LightGBM binary-classification model.

    Supports two inference backends:
    * **ONNX** — preferred for latency-sensitive paths.
    * **Native LightGBM Booster** — fallback when no ONNX export is available.
    """

    def __init__(
        self,
        model_path: str | None = None,
        onnx_path: str | None = None,
    ) -> None:
        self.model: lgb.Booster | None = None
        self.onnx_session: ort.InferenceSession | None = None
        self._model_path = model_path
        self._onnx_path = onnx_path

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self, path: str | None = None) -> None:
        """Load a model — tries ONNX first, then falls back to native LightGBM.

        Parameters
        ----------
        path:
            Override for the native LightGBM model file.  ONNX path is taken
            from ``self._onnx_path`` supplied at construction time.
        """
        native_path = path or self._model_path

        # --- Attempt ONNX ---
        if self._onnx_path and Path(self._onnx_path).exists():
            try:
                self.onnx_session = ort.InferenceSession(
                    self._onnx_path,
                    providers=["CPUExecutionProvider"],
                )
                logger.info("LightGBM ONNX session loaded from %s", self._onnx_path)
                return
            except Exception:
                logger.warning(
                    "ONNX load failed for %s; falling back to native",
                    self._onnx_path,
                    exc_info=True,
                )

        # --- Attempt native Booster ---
        if native_path is None:
            logger.warning("No LGBM model path provided; skipping load")
            return

        try:
            self.model = lgb.Booster(model_file=native_path)
            logger.info("Native LightGBM model loaded from %s", native_path)
        except FileNotFoundError:
            logger.warning("LightGBM model not found at %s", native_path)
        except Exception:
            logger.exception("Failed to load LightGBM model from %s", native_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, features: np.ndarray) -> float:
        """Return buy probability in ``[0.0, 1.0]``.

        Parameters
        ----------
        features:
            1-D or 2-D float array of input features.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # --- ONNX path ---
        if self.onnx_session is not None:
            return self._predict_onnx(features)

        # --- Native path ---
        if self.model is not None:
            return self._predict_native(features)

        logger.warning("No LGBM model loaded; returning 0.5 (neutral)")
        return 0.5

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict[str, float]:
        """Return ``{feature_name: importance}`` from the native Booster.

        Only available when a native LightGBM model is loaded; returns an
        empty dict otherwise.
        """
        if self.model is None:
            logger.warning("Native model not loaded; cannot provide feature importance")
            return {}

        names = self.model.feature_name()
        importances = self.model.feature_importance(importance_type="gain")
        total = float(importances.sum()) or 1.0
        return {
            name: round(float(imp) / total, 6)
            for name, imp in zip(names, importances)
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _predict_onnx(self, features: np.ndarray) -> float:
        """Run ONNX inference and return buy probability."""
        try:
            input_name = self.onnx_session.get_inputs()[0].name
            input_data = features.astype(np.float32)
            outputs = self.onnx_session.run(None, {input_name: input_data})
            # ONNX classifiers typically output [labels, probabilities]
            if len(outputs) >= 2:
                probas = outputs[1]
                # probas shape: (batch, n_classes) — class-1 is "buy"
                if isinstance(probas, list):
                    return float(probas[0].get(1, 0.5))
                return float(probas[0][1]) if probas.ndim == 2 else float(probas[0])
            # Single output — treat as raw probability
            return float(np.clip(outputs[0].flat[0], 0.0, 1.0))
        except Exception:
            logger.exception("ONNX inference failed; returning 0.5")
            return 0.5

    def _predict_native(self, features: np.ndarray) -> float:
        """Run native Booster prediction (returns raw score for binary objective)."""
        try:
            raw = self.model.predict(features)
            prob = float(raw.flat[0])
            return float(np.clip(prob, 0.0, 1.0))
        except Exception:
            logger.exception("Native LightGBM inference failed; returning 0.5")
            return 0.5
