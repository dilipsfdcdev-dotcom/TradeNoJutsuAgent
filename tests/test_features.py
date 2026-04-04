"""Tests for the ML feature engineering pipeline."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from agent.signals.mtf_analyzer import MTFState
from agent.ml.features import compute_features, compute_label


def _make_candles(n, base_price, seed=42):
    np.random.seed(seed)
    dates = [datetime(2024, 1, 15, 10, 0) + timedelta(minutes=i) for i in range(n)]
    prices = [base_price]
    for _ in range(n - 1):
        prices.append(prices[-1] + np.random.randn() * 1.5)
    rows = []
    for dt, p in zip(dates, prices):
        o = p + np.random.randn() * 0.5
        h = max(o, p) + abs(np.random.randn()) * 1.0
        l = min(o, p) - abs(np.random.randn()) * 1.0
        rows.append({
            "time": dt, "open": o, "high": h, "low": l,
            "close": p, "volume": int(np.random.uniform(100, 1000)),
            "spread": 20,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def candles_1m():
    return _make_candles(100, 2050.0, seed=42)


@pytest.fixture
def candles_3m():
    return _make_candles(100, 2050.0, seed=43)


@pytest.fixture
def candles_15m():
    return _make_candles(100, 2050.0, seed=44)


@pytest.fixture
def candles_1h():
    return _make_candles(200, 2050.0, seed=45)


@pytest.fixture
def mtf_state():
    return MTFState(symbol="XAUUSD")


@pytest.fixture
def account_state():
    return {
        "balance": 10000.0, "equity": 10050.0,
        "daily_pnl_pct": 0.5, "recent_win_streak": 2,
        "minutes_since_last_trade": 30, "spread": 20, "avg_spread": 18,
    }


class TestComputeFeatures:
    def test_returns_correct_structure(
        self, mtf_state, candles_1m, candles_3m, candles_15m, candles_1h, account_state
    ):
        result = compute_features(
            mtf_state, candles_1m, candles_3m, candles_15m,
            candles_1h, account_state, sentiment=0.3,
        )
        assert "xgb_features" in result
        assert "lstm_sequences" in result
        assert isinstance(result["xgb_features"], dict)
        assert isinstance(result["lstm_sequences"], dict)

    def test_xgb_features_not_empty(
        self, mtf_state, candles_1m, candles_3m, candles_15m, candles_1h, account_state
    ):
        result = compute_features(
            mtf_state, candles_1m, candles_3m, candles_15m,
            candles_1h, account_state, sentiment=0.3,
        )
        assert len(result["xgb_features"]) >= 30

    def test_xgb_features_no_nan(
        self, mtf_state, candles_1m, candles_3m, candles_15m, candles_1h, account_state
    ):
        result = compute_features(
            mtf_state, candles_1m, candles_3m, candles_15m,
            candles_1h, account_state, sentiment=0.3,
        )
        for key, val in result["xgb_features"].items():
            assert not np.isnan(val), f"NaN found in feature: {key}"

    def test_lstm_sequences_shapes(
        self, mtf_state, candles_1m, candles_3m, candles_15m, candles_1h, account_state
    ):
        result = compute_features(
            mtf_state, candles_1m, candles_3m, candles_15m,
            candles_1h, account_state, sentiment=0.3,
        )
        seqs = result["lstm_sequences"]
        assert "seq_1m" in seqs
        assert "seq_3m" in seqs
        assert "seq_15m" in seqs
        assert seqs["seq_1m"].shape == (30, 5)
        assert seqs["seq_3m"].shape == (20, 5)
        assert seqs["seq_15m"].shape == (10, 5)

    def test_lstm_sequences_normalized(
        self, mtf_state, candles_1m, candles_3m, candles_15m, candles_1h, account_state
    ):
        result = compute_features(
            mtf_state, candles_1m, candles_3m, candles_15m,
            candles_1h, account_state, sentiment=0.3,
        )
        for key in ("seq_1m", "seq_3m", "seq_15m"):
            seq = result["lstm_sequences"][key]
            # Normalized sequences should have values roughly in [-3, 3] range
            assert np.abs(seq).max() < 50, f"Sequence {key} appears unnormalized"


class TestComputeLabel:
    def test_winning_trade(self):
        trade = {
            "pnl": 50.0,
            "rr_actual": 2.0,
            "trade_quality": 8,
            "entry_price": 2050.0,
            "direction": "buy",
        }
        label = compute_label(trade)
        assert label["direction_correct"] == 1
        assert label["rr_achieved"] == 2.0

    def test_losing_trade(self):
        trade = {
            "pnl": -30.0,
            "rr_actual": -0.8,
            "trade_quality": 3,
            "entry_price": 2050.0,
            "direction": "sell",
        }
        label = compute_label(trade)
        assert label["direction_correct"] == 0
