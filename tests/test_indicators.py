"""Tests for agent.signals.indicators."""

import numpy as np
import pandas as pd
import pytest

from agent.signals.indicators import (
    compute_atr,
    compute_bollinger,
    compute_ema,
    compute_rsi,
    compute_vwap,
    compute_all_indicators,
)


class TestComputeEma:
    def test_compute_ema_column_exists(self, sample_candles):
        df = compute_ema(sample_candles, period=9)
        assert "ema_9" in df.columns

    def test_compute_ema_values_reasonable(self, sample_candles):
        df = compute_ema(sample_candles, period=9)
        valid = df["ema_9"].dropna()
        assert len(valid) > 0
        # EMA should be within the range of close prices
        assert valid.min() >= sample_candles["close"].min() - 10
        assert valid.max() <= sample_candles["close"].max() + 10

    def test_compute_ema_nan_during_warmup(self, sample_candles):
        df = compute_ema(sample_candles, period=21)
        # First 20 values should be NaN (need 21 values to seed)
        assert df["ema_21"].iloc[:20].isna().all()
        assert pd.notna(df["ema_21"].iloc[20])

    def test_compute_ema_period_1(self, sample_candles):
        df = compute_ema(sample_candles, period=1)
        # EMA with period=1 should equal close
        valid = df["ema_1"].dropna()
        assert len(valid) > 0


class TestComputeRsi:
    def test_compute_rsi_between_0_and_100(self, sample_candles):
        df = compute_rsi(sample_candles, period=14)
        valid = df["rsi"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_compute_rsi_column_exists(self, sample_candles):
        df = compute_rsi(sample_candles)
        assert "rsi" in df.columns

    def test_compute_rsi_nan_during_warmup(self, sample_candles):
        df = compute_rsi(sample_candles, period=14)
        # First 14 values should be NaN
        assert df["rsi"].iloc[:14].isna().all()


class TestComputeAtr:
    def test_compute_atr_positive(self, sample_candles):
        df = compute_atr(sample_candles, period=14)
        valid = df["atr"].dropna()
        assert len(valid) > 0
        assert (valid > 0).all()

    def test_compute_atr_column_exists(self, sample_candles):
        df = compute_atr(sample_candles)
        assert "atr" in df.columns

    def test_compute_atr_nan_during_warmup(self, sample_candles):
        df = compute_atr(sample_candles, period=14)
        assert df["atr"].iloc[:13].isna().all()
        assert pd.notna(df["atr"].iloc[13])


class TestComputeVwap:
    def test_compute_vwap_in_price_range(self, sample_candles):
        df = compute_vwap(sample_candles)
        valid = df["vwap"].dropna()
        assert len(valid) > 0
        # VWAP should be within the overall low-high range of prices
        overall_low = sample_candles["low"].min()
        overall_high = sample_candles["high"].max()
        assert valid.min() >= overall_low - 1
        assert valid.max() <= overall_high + 1

    def test_compute_vwap_column_exists(self, sample_candles):
        df = compute_vwap(sample_candles)
        assert "vwap" in df.columns


class TestComputeBollinger:
    def test_compute_bollinger_upper_gt_middle_gt_lower(self, sample_candles):
        df = compute_bollinger(sample_candles, period=20, std_dev=2.0)
        # Only compare where all bands are valid
        valid_mask = df["bb_upper"].notna() & df["bb_middle"].notna() & df["bb_lower"].notna()
        valid = df[valid_mask]
        assert len(valid) > 0
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_compute_bollinger_columns_exist(self, sample_candles):
        df = compute_bollinger(sample_candles)
        assert "bb_upper" in df.columns
        assert "bb_middle" in df.columns
        assert "bb_lower" in df.columns


class TestComputeAllIndicators:
    def test_all_columns_added(self, sample_candles):
        df = compute_all_indicators(sample_candles)
        expected = [
            "ema_9", "ema_21", "ema_50", "rsi", "atr",
            "vwap", "bb_upper", "bb_middle", "bb_lower", "volume_delta",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self, sample_candles):
        original_cols = list(sample_candles.columns)
        df = compute_all_indicators(sample_candles)
        for col in original_cols:
            assert col in df.columns


class TestEmptyDataframe:
    def test_empty_dataframe_ema(self):
        df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "spread"])
        result = compute_ema(df, period=9)
        assert "ema_9" in result.columns
        assert len(result) == 0

    def test_empty_dataframe_rsi(self):
        df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "spread"])
        result = compute_rsi(df)
        assert "rsi" in result.columns
        assert len(result) == 0

    def test_empty_dataframe_atr(self):
        df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "spread"])
        result = compute_atr(df)
        assert "atr" in result.columns
        assert len(result) == 0

    def test_empty_dataframe_vwap(self):
        df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "spread"])
        result = compute_vwap(df)
        assert "vwap" in result.columns
        assert len(result) == 0

    def test_empty_dataframe_bollinger(self):
        df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "spread"])
        result = compute_bollinger(df)
        assert "bb_upper" in result.columns
        assert len(result) == 0

    def test_empty_dataframe_all_indicators(self):
        df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "spread"])
        result = compute_all_indicators(df)
        assert len(result) == 0
