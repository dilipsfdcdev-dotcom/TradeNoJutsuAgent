"""Tests for ML models — XGBoost filter and LSTM confidence scorer."""

import pytest
import numpy as np
import pandas as pd

from agent.ml.xgboost_filter import XGBoostFilter
from agent.ml.lstm_model import LSTMConfidence, MultiTFLSTM


class TestXGBoostFilter:
    def test_init_no_model(self):
        f = XGBoostFilter("XAUUSD")
        assert f.symbol == "XAUUSD"
        assert f.model is None
        assert f.threshold == 0.55

    def test_predict_no_model_passes_through(self):
        f = XGBoostFilter("XAUUSD")
        result = f.predict({"feature1": 0.5, "feature2": 0.3})
        assert result["pass"] is True
        assert result["score"] == 0.5
        assert result["top_features"] == []

    def test_retrain_basic(self):
        f = XGBoostFilter("XAUUSD")
        np.random.seed(42)
        n = 200
        features = pd.DataFrame({
            f"f{i}": np.random.randn(n) for i in range(10)
        })
        labels = pd.Series(np.random.randint(0, 2, n))

        result = f.retrain(features, labels, validation_split=0.2)

        assert "auc" in result
        assert "precision" in result
        assert "recall" in result
        assert "deployed" in result
        assert isinstance(result["auc"], float)
        assert f.model is not None

    def test_predict_after_retrain(self):
        f = XGBoostFilter("XAUUSD")
        np.random.seed(42)
        n = 200
        features = pd.DataFrame({
            f"f{i}": np.random.randn(n) for i in range(10)
        })
        labels = pd.Series(np.random.randint(0, 2, n))
        f.retrain(features, labels)

        sample = {f"f{i}": 0.5 for i in range(10)}
        result = f.predict(sample)
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0
        assert isinstance(result["pass"], bool)
        assert isinstance(result["top_features"], list)

    def test_save_and_load(self, tmp_path):
        f = XGBoostFilter("XAUUSD")
        np.random.seed(42)
        features = pd.DataFrame({f"f{i}": np.random.randn(100) for i in range(5)})
        labels = pd.Series(np.random.randint(0, 2, 100))
        f.retrain(features, labels)

        path = str(tmp_path / "model.json")
        f.save_model(path)

        f2 = XGBoostFilter("XAUUSD", model_path=path)
        assert f2.model is not None
        assert f2.feature_names == f.feature_names

    def test_auto_tune_threshold(self):
        f = XGBoostFilter("XAUUSD")
        np.random.seed(42)
        features = pd.DataFrame({f"f{i}": np.random.randn(200) for i in range(5)})
        labels = pd.Series(np.random.randint(0, 2, 200))
        f.retrain(features, labels)

        thresh = f.auto_tune_threshold(
            features.iloc[-40:], labels.iloc[-40:]
        )
        assert 0.4 <= thresh <= 0.75


class TestMultiTFLSTM:
    def test_model_forward(self):
        import torch
        model = MultiTFLSTM()
        seq_1m = torch.randn(2, 30, 5)
        seq_3m = torch.randn(2, 20, 5)
        seq_15m = torch.randn(2, 10, 5)

        direction, confidence, regime = model(seq_1m, seq_3m, seq_15m)

        assert direction.shape == (2, 3)
        assert confidence.shape == (2, 1)
        assert regime.shape == (2, 3)
        # Softmax outputs should sum to ~1
        assert torch.allclose(direction.sum(dim=1), torch.ones(2), atol=1e-5)
        assert torch.allclose(regime.sum(dim=1), torch.ones(2), atol=1e-5)
        # Sigmoid should be between 0 and 1
        assert (confidence >= 0).all() and (confidence <= 1).all()

    def test_model_parameter_count(self):
        model = MultiTFLSTM()
        total_params = sum(p.numel() for p in model.parameters())
        # Should be lightweight (~50K-100K params)
        assert total_params < 200_000


class TestLSTMConfidence:
    def test_init_no_model(self):
        m = LSTMConfidence("XAUUSD")
        assert m.symbol == "XAUUSD"
        assert m.confidence_threshold == 0.60

    def test_predict(self):
        m = LSTMConfidence("XAUUSD")
        sequences = {
            "seq_1m": np.random.randn(30, 5).astype(np.float32),
            "seq_3m": np.random.randn(20, 5).astype(np.float32),
            "seq_15m": np.random.randn(10, 5).astype(np.float32),
        }
        result = m.predict(sequences)

        assert "direction" in result
        assert result["direction"] in ("buy", "sell", "wait")
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert "regime" in result
        assert result["regime"] in ("trending", "ranging", "volatile")
        assert "pass" in result
        assert isinstance(result["pass"], bool)

    def test_predict_deterministic_with_eval(self):
        m = LSTMConfidence("XAUUSD")
        sequences = {
            "seq_1m": np.random.randn(30, 5).astype(np.float32),
            "seq_3m": np.random.randn(20, 5).astype(np.float32),
            "seq_15m": np.random.randn(10, 5).astype(np.float32),
        }
        r1 = m.predict(sequences)
        r2 = m.predict(sequences)
        assert r1["confidence"] == r2["confidence"]
        assert r1["direction"] == r2["direction"]

    def test_save_and_load(self, tmp_path):
        m = LSTMConfidence("XAUUSD")
        path = str(tmp_path / "lstm.pt")
        m.save_model(path)

        m2 = LSTMConfidence("XAUUSD", model_path=path)
        sequences = {
            "seq_1m": np.random.randn(30, 5).astype(np.float32),
            "seq_3m": np.random.randn(20, 5).astype(np.float32),
            "seq_15m": np.random.randn(10, 5).astype(np.float32),
        }
        r1 = m.predict(sequences)
        r2 = m2.predict(sequences)
        assert r1["direction"] == r2["direction"]
