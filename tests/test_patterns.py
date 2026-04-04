"""Tests for agent.signals.patterns."""

import pandas as pd
import pytest

from agent.signals.patterns import (
    PatternSignal,
    detect_all_patterns,
    detect_engulfing,
    detect_pin_bars,
)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of row dicts."""
    from datetime import datetime, timedelta

    for i, row in enumerate(rows):
        row.setdefault("time", datetime(2024, 1, 15, 10, 0) + timedelta(minutes=i))
        row.setdefault("volume", 500)
        row.setdefault("spread", 20)
    return pd.DataFrame(rows)


class TestDetectEngulfing:
    def test_detect_bullish_engulfing(self):
        """Bearish candle followed by a larger bullish candle that engulfs it."""
        rows = [
            {"open": 2050.0, "high": 2051.0, "low": 2047.0, "close": 2048.0},  # bearish
            {"open": 2047.5, "high": 2053.0, "low": 2047.0, "close": 2052.0},  # bullish engulfs
        ]
        df = _make_df(rows)
        signals = detect_engulfing(df)
        assert len(signals) >= 1
        assert any(s.type == "bullish_engulfing" for s in signals)

    def test_detect_bearish_engulfing(self):
        """Bullish candle followed by a larger bearish candle that engulfs it."""
        rows = [
            {"open": 2048.0, "high": 2051.0, "low": 2047.0, "close": 2050.0},  # bullish
            {"open": 2051.0, "high": 2052.0, "low": 2045.0, "close": 2046.0},  # bearish engulfs
        ]
        df = _make_df(rows)
        signals = detect_engulfing(df)
        assert len(signals) >= 1
        assert any(s.type == "bearish_engulfing" for s in signals)

    def test_no_engulfing_on_same_direction(self):
        """Two bullish candles should not produce an engulfing signal."""
        rows = [
            {"open": 2048.0, "high": 2051.0, "low": 2047.0, "close": 2050.0},  # bullish
            {"open": 2050.0, "high": 2055.0, "low": 2049.0, "close": 2054.0},  # bullish
        ]
        df = _make_df(rows)
        signals = detect_engulfing(df)
        assert len(signals) == 0


class TestDetectPinBars:
    def test_detect_bullish_pin_bar(self):
        """Candle with long lower wick and small body near top (hammer)."""
        rows = [
            {
                "open": 2050.0,
                "high": 2051.0,
                "low": 2040.0,  # long lower wick
                "close": 2050.5,  # body near top
            },
        ]
        df = _make_df(rows)
        signals = detect_pin_bars(df)
        assert len(signals) >= 1
        assert any(s.type == "bullish_pin_bar" for s in signals)

    def test_detect_bearish_pin_bar(self):
        """Candle with long upper wick and small body near bottom (shooting star)."""
        rows = [
            {
                "open": 2050.0,
                "high": 2060.0,  # long upper wick
                "low": 2049.0,
                "close": 2049.5,  # body near bottom
            },
        ]
        df = _make_df(rows)
        signals = detect_pin_bars(df)
        assert len(signals) >= 1
        assert any(s.type == "bearish_pin_bar" for s in signals)

    def test_no_pin_bar_on_large_body(self):
        """A candle with a large body and short wicks is not a pin bar."""
        rows = [
            {
                "open": 2040.0,
                "high": 2051.0,
                "low": 2039.0,
                "close": 2050.0,  # large body, short wicks
            },
        ]
        df = _make_df(rows)
        signals = detect_pin_bars(df)
        # Should not detect a pin bar since lower wick (1) < 2 * body (10)
        assert len(signals) == 0


class TestDetectAllPatterns:
    def test_returns_list_of_pattern_signal(self, sample_candles):
        signals = detect_all_patterns(sample_candles)
        assert isinstance(signals, list)
        for s in signals:
            assert isinstance(s, PatternSignal)

    def test_pattern_signal_fields(self, sample_candles):
        signals = detect_all_patterns(sample_candles)
        if signals:
            s = signals[0]
            assert isinstance(s.type, str)
            assert isinstance(s.strength, float)
            assert 0 <= s.strength <= 1
            assert isinstance(s.price_level, float)
            assert isinstance(s.candle_index, int)

    def test_sorted_by_candle_index_descending(self, sample_candles):
        signals = detect_all_patterns(sample_candles)
        if len(signals) > 1:
            indices = [s.candle_index for s in signals]
            assert indices == sorted(indices, reverse=True)


class TestNoPatterns:
    def test_flat_data_no_patterns(self):
        """Flat price data with identical candles should produce no engulfing/pin bar signals."""
        rows = []
        for i in range(20):
            rows.append({
                "open": 2050.0,
                "high": 2050.0,
                "low": 2050.0,
                "close": 2050.0,
            })
        df = _make_df(rows)
        engulfing = detect_engulfing(df)
        pin_bars = detect_pin_bars(df)
        assert len(engulfing) == 0
        assert len(pin_bars) == 0
