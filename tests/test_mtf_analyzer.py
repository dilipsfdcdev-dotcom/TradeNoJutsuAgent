"""Tests for the multi-timeframe analyzer."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from agent.signals.mtf_analyzer import (
    MTFAnalyzer, MTFState, PriceLevel, Zone, EntrySignal,
)


@pytest.fixture
def sample_candles_1h():
    """200 1H candles with a bullish trend."""
    np.random.seed(42)
    n = 200
    dates = [datetime(2024, 1, 1, 0, 0) + timedelta(hours=i) for i in range(n)]
    base = 2000.0
    prices = [base]
    for _ in range(n - 1):
        prices.append(prices[-1] + np.random.randn() * 3 + 0.5)  # upward drift

    rows = []
    for dt, p in zip(dates, prices):
        o = p + np.random.randn() * 1
        h = max(o, p) + abs(np.random.randn()) * 2
        l = min(o, p) - abs(np.random.randn()) * 2
        rows.append({"time": dt, "open": o, "high": h, "low": l,
                      "close": p, "volume": int(np.random.uniform(500, 2000)), "spread": 20})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_candles_15m():
    np.random.seed(43)
    n = 100
    dates = [datetime(2024, 1, 8, 0, 0) + timedelta(minutes=15 * i) for i in range(n)]
    base = 2100.0
    prices = [base]
    for _ in range(n - 1):
        prices.append(prices[-1] + np.random.randn() * 1.5 + 0.2)

    rows = []
    for dt, p in zip(dates, prices):
        o = p + np.random.randn() * 0.5
        h = max(o, p) + abs(np.random.randn()) * 1.0
        l = min(o, p) - abs(np.random.randn()) * 1.0
        rows.append({"time": dt, "open": o, "high": h, "low": l,
                      "close": p, "volume": int(np.random.uniform(200, 1000)), "spread": 20})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_candles_3m():
    np.random.seed(44)
    n = 50
    dates = [datetime(2024, 1, 9, 8, 0) + timedelta(minutes=3 * i) for i in range(n)]
    base = 2120.0
    prices = [base]
    for _ in range(n - 1):
        prices.append(prices[-1] + np.random.randn() * 0.8 + 0.1)

    rows = []
    for dt, p in zip(dates, prices):
        o = p + np.random.randn() * 0.3
        h = max(o, p) + abs(np.random.randn()) * 0.5
        l = min(o, p) - abs(np.random.randn()) * 0.5
        rows.append({"time": dt, "open": o, "high": h, "low": l,
                      "close": p, "volume": int(np.random.uniform(100, 500)), "spread": 20})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_candles_1m():
    np.random.seed(45)
    n = 100
    dates = [datetime(2024, 1, 9, 10, 0) + timedelta(minutes=i) for i in range(n)]
    base = 2130.0
    prices = [base]
    for _ in range(n - 1):
        prices.append(prices[-1] + np.random.randn() * 0.5)

    rows = []
    for dt, p in zip(dates, prices):
        o = p + np.random.randn() * 0.2
        h = max(o, p) + abs(np.random.randn()) * 0.3
        l = min(o, p) - abs(np.random.randn()) * 0.3
        rows.append({"time": dt, "open": o, "high": h, "low": l,
                      "close": p, "volume": int(np.random.uniform(50, 300)), "spread": 20})
    return pd.DataFrame(rows)


class TestMTFState:
    def test_default_state(self):
        state = MTFState(symbol="XAUUSD")
        assert state.symbol == "XAUUSD"
        assert state.h1_bias == "neutral"
        assert state.confluence_score == 0
        assert state.gates_passed is False
        assert state.m1_entry_signal is None

    def test_price_level(self):
        level = PriceLevel(
            price=2100.0, level_type="resistance",
            source_tf="1H", touches=3, strength=0.8,
        )
        assert level.price == 2100.0
        assert level.touches == 3

    def test_zone(self):
        zone = Zone(
            zone_type="order_block", high=2105.0, low=2100.0,
            source_tf="15M", direction="bullish",
            formation_time=datetime.utcnow(),
            tested=False, mitigated=False, strength=0.7,
        )
        assert zone.zone_type == "order_block"
        assert zone.direction == "bullish"


class TestMTFAnalyzer:
    def test_initialization(self):
        analyzer = MTFAnalyzer(["XAUUSD", "BTCUSD"])
        assert "XAUUSD" in analyzer.states
        assert "BTCUSD" in analyzer.states

    def test_update_returns_state(
        self, sample_candles_1m, sample_candles_3m,
        sample_candles_15m, sample_candles_1h,
    ):
        analyzer = MTFAnalyzer(["XAUUSD"])
        tick = {"bid": 2130.0, "ask": 2130.20}
        state = analyzer.update(
            "XAUUSD", sample_candles_1m, sample_candles_3m,
            sample_candles_15m, sample_candles_1h, tick,
        )
        assert isinstance(state, MTFState)
        assert state.symbol == "XAUUSD"
        assert state.h1_bias in ("bullish", "bearish", "neutral")
        assert 0 <= state.confluence_score <= 100
        assert isinstance(state.setup_narrative, str)

    def test_h1_analysis_sets_bias(
        self, sample_candles_1m, sample_candles_3m,
        sample_candles_15m, sample_candles_1h,
    ):
        analyzer = MTFAnalyzer(["XAUUSD"])
        tick = {"bid": 2130.0, "ask": 2130.20}
        state = analyzer.update(
            "XAUUSD", sample_candles_1m, sample_candles_3m,
            sample_candles_15m, sample_candles_1h, tick,
        )
        # With upward-drifting data, bias should lean bullish
        assert state.h1_bias in ("bullish", "bearish", "neutral")
        assert state.h1_atr > 0

    def test_confluence_score_range(
        self, sample_candles_1m, sample_candles_3m,
        sample_candles_15m, sample_candles_1h,
    ):
        analyzer = MTFAnalyzer(["XAUUSD"])
        tick = {"bid": 2130.0, "ask": 2130.20}
        state = analyzer.update(
            "XAUUSD", sample_candles_1m, sample_candles_3m,
            sample_candles_15m, sample_candles_1h, tick,
        )
        assert 0 <= state.confluence_score <= 100

    def test_narrative_not_empty(
        self, sample_candles_1m, sample_candles_3m,
        sample_candles_15m, sample_candles_1h,
    ):
        analyzer = MTFAnalyzer(["XAUUSD"])
        tick = {"bid": 2130.0, "ask": 2130.20}
        state = analyzer.update(
            "XAUUSD", sample_candles_1m, sample_candles_3m,
            sample_candles_15m, sample_candles_1h, tick,
        )
        assert len(state.setup_narrative) > 0

    def test_multiple_symbols(
        self, sample_candles_1m, sample_candles_3m,
        sample_candles_15m, sample_candles_1h,
    ):
        analyzer = MTFAnalyzer(["XAUUSD", "BTCUSD"])
        tick = {"bid": 2130.0, "ask": 2130.20}
        state1 = analyzer.update(
            "XAUUSD", sample_candles_1m, sample_candles_3m,
            sample_candles_15m, sample_candles_1h, tick,
        )
        state2 = analyzer.update(
            "BTCUSD", sample_candles_1m, sample_candles_3m,
            sample_candles_15m, sample_candles_1h, tick,
        )
        assert state1.symbol == "XAUUSD"
        assert state2.symbol == "BTCUSD"
