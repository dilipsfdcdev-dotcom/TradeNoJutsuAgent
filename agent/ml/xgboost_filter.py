"""Brain 1: XGBoost pre-filter — fast binary classifier."""

import json
import numpy as np
import pandas as pd
import structlog
from pathlib import Path

logger = structlog.get_logger()


class XGBoostFilter:

    def __init__(self, symbol: str, model_path: str | None = None):
        self.symbol = symbol
        self.model = None
        self.threshold = 0.55
        self.feature_names: list[str] = []
        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load saved XGBoost model."""
        import xgboost as xgb
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        # Load feature names and threshold from metadata
        meta_path = Path(path).with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.threshold = meta.get("threshold", 0.55)
            self.feature_names = meta.get("feature_names", [])

    def predict(self, features: dict) -> dict:
        """
        Input: xgb_features dict
        Output: {"score": 0.73, "pass": True, "top_features": [...]}
        Takes <1ms.
        """
        if self.model is None:
            # No model trained yet — pass everything through
            return {"score": 0.5, "pass": True, "top_features": []}

        # Convert features dict to array in correct order
        feature_array = np.array([[features.get(f, 0.0) for f in self.feature_names]])

        # Predict probability
        proba = self.model.predict_proba(feature_array)[0]
        score = float(proba[1])  # probability of positive class

        # Get feature importance for this prediction (top 5)
        top_features = self._get_top_features(features)

        return {
            "score": round(score, 4),
            "pass": score >= self.threshold,
            "top_features": top_features,
        }

    def _get_top_features(self, features: dict) -> list[tuple[str, float]]:
        """Get top 5 most important features."""
        if self.model is None:
            return []
        importances = self.model.feature_importances_
        pairs = list(zip(self.feature_names, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [(name, round(float(imp), 4)) for name, imp in pairs[:5]]

    def retrain(self, features_df: pd.DataFrame, labels: pd.Series, validation_split: float = 0.2) -> dict:
        """
        Weekly retraining with walk-forward validation.
        - Split: last validation_split fraction is test set (time-ordered, NOT random)
        - Model: XGBClassifier with n_estimators=200, max_depth=6, lr=0.05
        - Returns: {"auc": float, "precision": float, "recall": float, "deployed": bool}
        """
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score, precision_score, recall_score

        split_idx = int(len(features_df) * (1 - validation_split))
        X_train, X_val = features_df.iloc[:split_idx], features_df.iloc[split_idx:]
        y_train, y_val = labels.iloc[:split_idx], labels.iloc[split_idx:]

        # Handle class imbalance
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr", use_label_encoder=False,
            early_stopping_rounds=20,
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Evaluate
        val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba) if len(y_val.unique()) > 1 else 0.5

        # Auto-tune threshold
        new_threshold = self.auto_tune_threshold(X_val, y_val, model)
        val_preds = (val_proba >= new_threshold).astype(int)
        precision = precision_score(y_val, val_preds, zero_division=0)
        recall = recall_score(y_val, val_preds, zero_division=0)

        # Regression protection: deploy only if not significantly worse
        old_auc = self._get_current_auc() if self.model else 0.0
        deployed = auc >= old_auc - 0.02

        if deployed:
            self.model = model
            self.threshold = new_threshold
            self.feature_names = list(features_df.columns)

        result = {"auc": round(auc, 4), "precision": round(precision, 4),
                  "recall": round(recall, 4), "threshold": round(new_threshold, 4),
                  "deployed": deployed}
        logger.info("xgb_retrain", symbol=self.symbol, **result)
        return result

    def auto_tune_threshold(self, X_val, y_val, model=None) -> float:
        """Find threshold maximizing precision * recall, favoring precision > 0.60."""
        m = model or self.model
        if m is None:
            return 0.55
        proba = m.predict_proba(X_val)[:, 1]
        from sklearn.metrics import precision_score, recall_score
        best_score, best_thresh = 0.0, 0.55
        for t in np.arange(0.40, 0.75, 0.01):
            preds = (proba >= t).astype(int)
            if preds.sum() == 0:
                continue
            p = precision_score(y_val, preds, zero_division=0)
            r = recall_score(y_val, preds, zero_division=0)
            # Favor precision > 0.60
            score = p * r * (1.2 if p >= 0.60 else 0.8)
            if score > best_score:
                best_score, best_thresh = score, t
        return best_thresh

    def save_model(self, path: str):
        if self.model:
            self.model.save_model(path)
            meta = {"threshold": self.threshold, "feature_names": self.feature_names}
            Path(path).with_suffix(".meta.json").write_text(json.dumps(meta))

    def _get_current_auc(self) -> float:
        return 0.5  # placeholder, would load from metrics DB
