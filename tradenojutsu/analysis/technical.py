"""Technical analysis engine - calculates indicators and identifies patterns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from tradenojutsu.data.models import Direction, MarketRegime, MarketState, SignalStrength
from tradenojutsu.infra.logger import get_logger

logger = get_logger("analysis.technical")


class TechnicalAnalyzer:
    """Computes technical indicators and derives market state."""

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to a price DataFrame.

        Expects columns: open, high, low, close, volume
        """
        if len(df) < 50:
            logger.warning("Insufficient data for reliable indicators")
            return df

        df = df.copy()

        # Trend indicators
        df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
        df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
        df["sma_200"] = ta.trend.sma_indicator(df["close"], window=200)
        df["ema_9"] = ta.trend.ema_indicator(df["close"], window=9)
        df["ema_21"] = ta.trend.ema_indicator(df["close"], window=21)

        # MACD
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # RSI
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)
        df["rsi_6"] = ta.momentum.rsi(df["close"], window=6)

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["close"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # ATR
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
        df["atr_pct"] = (df["atr"] / df["close"]) * 100

        # ADX
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()

        # Volume
        df["volume_sma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma"].replace(0, np.nan)

        # OBV
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])

        # Support/Resistance via pivots
        df["pivot"] = (df["high"].shift(1) + df["low"].shift(1) + df["close"].shift(1)) / 3
        df["r1"] = 2 * df["pivot"] - df["low"].shift(1)
        df["s1"] = 2 * df["pivot"] - df["high"].shift(1)

        # Ichimoku
        ichi = ta.trend.IchimokuIndicator(df["high"], df["low"])
        df["ichi_conv"] = ichi.ichimoku_conversion_line()
        df["ichi_base"] = ichi.ichimoku_base_line()
        df["ichi_a"] = ichi.ichimoku_a()
        df["ichi_b"] = ichi.ichimoku_b()

        return df

    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Classify current market regime from indicators."""
        if len(df) < 50 or "adx" not in df.columns:
            return MarketRegime.RANGING

        latest = df.iloc[-1]
        adx = latest.get("adx", 20)
        atr_pct = latest.get("atr_pct", 1.0)
        avg_atr_pct = df["atr_pct"].rolling(50).mean().iloc[-1] if "atr_pct" in df.columns else 1.0

        # Volatility regime
        vol_ratio = atr_pct / avg_atr_pct if avg_atr_pct > 0 else 1.0
        if vol_ratio > 2.0:
            return MarketRegime.HIGH_VOLATILITY
        if vol_ratio < 0.5:
            return MarketRegime.LOW_VOLATILITY

        # Trend regime
        if adx > 25:
            sma_20 = latest.get("sma_20", 0)
            sma_50 = latest.get("sma_50", 0)
            if sma_20 > sma_50:
                return MarketRegime.TRENDING_UP
            return MarketRegime.TRENDING_DOWN

        return MarketRegime.RANGING

    def detect_trend(self, df: pd.DataFrame) -> Direction:
        """Determine overall trend direction."""
        if len(df) < 50 or "sma_20" not in df.columns:
            return Direction.FLAT

        latest = df.iloc[-1]
        price = latest["close"]
        sma_20 = latest.get("sma_20", price)
        sma_50 = latest.get("sma_50", price)
        ema_9 = latest.get("ema_9", price)
        ema_21 = latest.get("ema_21", price)

        bullish_signals = sum([
            price > sma_20,
            price > sma_50,
            sma_20 > sma_50,
            ema_9 > ema_21,
            latest.get("macd", 0) > latest.get("macd_signal", 0),
        ])

        if bullish_signals >= 4:
            return Direction.LONG
        if bullish_signals <= 1:
            return Direction.SHORT
        return Direction.FLAT

    def score_signal(self, df: pd.DataFrame, direction: Direction) -> tuple[float, dict[str, float]]:
        """Score a directional signal (0-100) based on technical confluence.

        Returns (score, component_scores).
        """
        if len(df) < 50 or "rsi" not in df.columns:
            return 0.0, {}

        latest = df.iloc[-1]
        components: dict[str, float] = {}
        is_long = direction == Direction.LONG

        # RSI (0-20 points)
        rsi = latest.get("rsi", 50)
        if is_long:
            components["rsi"] = max(0, min(20, (70 - rsi) / 2)) if rsi < 70 else 0
        else:
            components["rsi"] = max(0, min(20, (rsi - 30) / 2)) if rsi > 30 else 0

        # MACD (0-20 points)
        macd_hist = latest.get("macd_hist", 0)
        prev_hist = df["macd_hist"].iloc[-2] if len(df) > 1 and "macd_hist" in df.columns else 0
        if is_long:
            components["macd"] = min(20, max(0, macd_hist * 100)) if macd_hist > 0 else 0
            if prev_hist < 0 < macd_hist:
                components["macd"] = min(20, components["macd"] + 10)  # crossover bonus
        else:
            components["macd"] = min(20, max(0, -macd_hist * 100)) if macd_hist < 0 else 0
            if prev_hist > 0 > macd_hist:
                components["macd"] = min(20, components["macd"] + 10)

        # Trend alignment (0-20 points)
        trend = self.detect_trend(df)
        if trend == direction:
            components["trend"] = 20
        elif trend == Direction.FLAT:
            components["trend"] = 10
        else:
            components["trend"] = 0

        # Bollinger Bands (0-20 points)
        bb_pos = 0
        if "bb_upper" in df.columns and "bb_lower" in df.columns:
            bb_range = latest["bb_upper"] - latest["bb_lower"]
            if bb_range > 0:
                bb_pos = (latest["close"] - latest["bb_lower"]) / bb_range
            if is_long:
                components["bollinger"] = max(0, min(20, (1 - bb_pos) * 25)) if bb_pos < 0.8 else 0
            else:
                components["bollinger"] = max(0, min(20, bb_pos * 25)) if bb_pos > 0.2 else 0

        # Volume confirmation (0-20 points)
        vol_ratio = latest.get("volume_ratio", 1.0)
        components["volume"] = min(20, max(0, (vol_ratio - 0.5) * 20))

        total = sum(components.values())
        return min(100, total), components

    def build_market_state(self, symbol: str, df: pd.DataFrame) -> MarketState:
        """Build a complete market state snapshot."""
        if df.empty:
            return MarketState(
                symbol=symbol, price=0, regime=MarketRegime.RANGING,
                trend_direction=Direction.FLAT, volatility=0, volume_ratio=1.0,
            )

        latest = df.iloc[-1]
        regime = self.detect_regime(df)
        trend = self.detect_trend(df)

        key_levels = {}
        for level_name in ["sma_20", "sma_50", "sma_200", "bb_upper", "bb_lower", "r1", "s1", "pivot"]:
            if level_name in df.columns and pd.notna(latest.get(level_name)):
                key_levels[level_name] = float(latest[level_name])

        indicators = {}
        for ind_name in ["rsi", "macd", "macd_hist", "adx", "stoch_k", "atr_pct", "bb_width"]:
            if ind_name in df.columns and pd.notna(latest.get(ind_name)):
                indicators[ind_name] = float(latest[ind_name])

        return MarketState(
            symbol=symbol,
            price=float(latest["close"]),
            regime=regime,
            trend_direction=trend,
            volatility=float(latest.get("atr_pct", 0)),
            volume_ratio=float(latest.get("volume_ratio", 1.0)),
            key_levels=key_levels,
            indicators=indicators,
        )
