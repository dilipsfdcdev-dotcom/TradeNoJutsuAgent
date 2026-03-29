"""Core tests for TradeNoJutsu components."""

import numpy as np
import pandas as pd
import pytest

from tradenojutsu.analysis.technical import TechnicalAnalyzer
from tradenojutsu.data.models import (
    Direction, MarketRegime, MarketState, PerformanceMetrics, Signal, SignalStrength, Trade,
)
from tradenojutsu.risk.manager import RiskManager, RiskParams
from tradenojutsu.strategy.strategies import (
    BreakoutStrategy, MeanReversionStrategy, MomentumStrategy, TrendFollowingStrategy,
)


def _make_ohlcv(n: int = 200, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    base = 100.0
    returns = np.random.normal(0.001 if trend == "up" else -0.001, 0.02, n)
    prices = base * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "open": prices * (1 - np.random.uniform(0, 0.01, n)),
        "high": prices * (1 + np.random.uniform(0, 0.02, n)),
        "low": prices * (1 - np.random.uniform(0, 0.02, n)),
        "close": prices,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="D")
    return df


class TestTechnicalAnalyzer:
    def test_compute_indicators(self):
        df = _make_ohlcv(200)
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_indicators(df)

        assert "rsi" in result.columns
        assert "macd" in result.columns
        assert "bb_upper" in result.columns
        assert "atr" in result.columns
        assert "adx" in result.columns
        assert len(result) == len(df)

    def test_detect_regime(self):
        df = _make_ohlcv(200)
        analyzer = TechnicalAnalyzer()
        df = analyzer.compute_indicators(df)
        regime = analyzer.detect_regime(df)
        assert isinstance(regime, MarketRegime)

    def test_detect_trend(self):
        df = _make_ohlcv(200, trend="up")
        analyzer = TechnicalAnalyzer()
        df = analyzer.compute_indicators(df)
        trend = analyzer.detect_trend(df)
        assert isinstance(trend, Direction)

    def test_score_signal(self):
        df = _make_ohlcv(200)
        analyzer = TechnicalAnalyzer()
        df = analyzer.compute_indicators(df)
        score, components = analyzer.score_signal(df, Direction.LONG)
        assert 0 <= score <= 100
        assert isinstance(components, dict)

    def test_build_market_state(self):
        df = _make_ohlcv(200)
        analyzer = TechnicalAnalyzer()
        df = analyzer.compute_indicators(df)
        state = analyzer.build_market_state("TEST", df)
        assert state.symbol == "TEST"
        assert state.price > 0
        assert isinstance(state.regime, MarketRegime)


class TestDataModels:
    def test_signal_actionable(self):
        sig = Signal(
            symbol="TEST", direction=Direction.LONG, score=75,
            strength=SignalStrength.STRONG, strategy="test", reasoning="test",
        )
        assert sig.is_actionable

        sig_flat = Signal(
            symbol="TEST", direction=Direction.FLAT, score=30,
            strength=SignalStrength.NONE, strategy="test", reasoning="test",
        )
        assert not sig_flat.is_actionable

    def test_performance_summary(self):
        m = PerformanceMetrics(total_trades=100, win_rate=0.55, sharpe_ratio=1.5)
        assert "100" in m.summary()
        assert "55.0%" in m.summary()

    def test_market_state_prompt(self):
        state = MarketState(
            symbol="XAUUSD", price=2000.0, regime=MarketRegime.TRENDING_UP,
            trend_direction=Direction.LONG, volatility=1.5, volume_ratio=1.2,
        )
        prompt = state.to_prompt_context()
        assert "XAUUSD" in prompt
        assert "2000" in prompt


class TestRiskManager:
    def test_calculate_risk_params(self):
        rm = RiskManager(capital=10000, risk_per_trade_pct=1.0)
        signal = Signal(
            symbol="TEST", direction=Direction.LONG, score=70,
            strength=SignalStrength.MODERATE, strategy="test", reasoning="test",
        )
        params = rm.calculate_risk_params(signal, price=100.0, atr=2.0)
        assert params.position_size > 0
        assert params.stop_loss < 100.0
        assert params.take_profit > 100.0
        assert params.risk_amount == 100.0  # 1% of 10000

    def test_check_exit_conditions(self):
        rm = RiskManager()
        trade = Trade(
            direction=Direction.LONG, entry_price=100, stop_loss=95, take_profit=110,
        )
        # No exit yet
        should, _ = rm.check_exit_conditions(trade, 102)
        assert not should
        # Stop loss hit
        should, reason = rm.check_exit_conditions(trade, 94)
        assert should
        assert "Stop-loss" in reason
        # Take profit hit
        should, reason = rm.check_exit_conditions(trade, 111)
        assert should
        assert "Take-profit" in reason


class TestStrategies:
    def test_trend_following(self):
        df = _make_ohlcv(200, trend="up")
        analyzer = TechnicalAnalyzer()
        df = analyzer.compute_indicators(df)
        strategy = TrendFollowingStrategy()
        # May or may not produce signal depending on data
        sig = strategy.evaluate(df, "TEST")
        if sig is not None:
            assert sig.strategy == "trend_following"
            assert isinstance(sig.direction, Direction)

    def test_mean_reversion(self):
        df = _make_ohlcv(200)
        analyzer = TechnicalAnalyzer()
        df = analyzer.compute_indicators(df)
        strategy = MeanReversionStrategy()
        sig = strategy.evaluate(df, "TEST")
        if sig is not None:
            assert sig.strategy == "mean_reversion"

    def test_breakout(self):
        df = _make_ohlcv(200)
        analyzer = TechnicalAnalyzer()
        df = analyzer.compute_indicators(df)
        strategy = BreakoutStrategy()
        sig = strategy.evaluate(df, "TEST")
        if sig is not None:
            assert sig.strategy == "breakout"

    def test_momentum(self):
        df = _make_ohlcv(200)
        analyzer = TechnicalAnalyzer()
        df = analyzer.compute_indicators(df)
        strategy = MomentumStrategy()
        sig = strategy.evaluate(df, "TEST")
        if sig is not None:
            assert sig.strategy == "momentum"
