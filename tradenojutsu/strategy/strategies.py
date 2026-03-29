"""Built-in trading strategies that the AI agent can select and blend."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from tradenojutsu.data.models import Direction, Signal, SignalStrength
from tradenojutsu.infra.logger import get_logger

logger = get_logger("strategy")


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    name: str = "base"

    @abstractmethod
    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        """Evaluate strategy on current data. Returns Signal or None."""
        ...


class TrendFollowingStrategy(BaseStrategy):
    """Classic trend following using moving average crossovers + ADX confirmation."""

    name = "trend_following"

    def __init__(self, fast_period: int = 9, slow_period: int = 21, adx_threshold: float = 25):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.adx_threshold = adx_threshold

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if len(df) < self.slow_period + 10:
            return None

        ema_fast = f"ema_{self.fast_period}"
        ema_slow = f"ema_{self.slow_period}"

        if ema_fast not in df.columns or ema_slow not in df.columns:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        adx = latest.get("adx", 0)

        # EMA crossover
        cross_up = prev[ema_fast] <= prev[ema_slow] and latest[ema_fast] > latest[ema_slow]
        cross_down = prev[ema_fast] >= prev[ema_slow] and latest[ema_fast] < latest[ema_slow]

        if not (cross_up or cross_down):
            return None

        if adx < self.adx_threshold:
            return None  # No trend confirmation

        direction = Direction.LONG if cross_up else Direction.SHORT
        score = min(100, 50 + (adx - self.adx_threshold) * 2)
        strength = SignalStrength.STRONG if adx > 40 else SignalStrength.MODERATE

        return Signal(
            symbol=symbol,
            direction=direction,
            score=score,
            strength=strength,
            strategy=self.name,
            reasoning=f"EMA {self.fast_period}/{self.slow_period} crossover with ADX={adx:.1f}",
        )


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion using Bollinger Bands + RSI extremes."""

    name = "mean_reversion"

    def __init__(self, rsi_oversold: float = 30, rsi_overbought: float = 70):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if len(df) < 50:
            return None

        required = ["rsi", "bb_lower", "bb_upper", "close", "adx"]
        if not all(c in df.columns for c in required):
            return None

        latest = df.iloc[-1]
        price = latest["close"]
        rsi = latest["rsi"]
        adx = latest.get("adx", 30)

        # Only mean revert in ranging markets
        if adx > 30:
            return None

        direction = None
        score = 0

        if rsi < self.rsi_oversold and price <= latest["bb_lower"]:
            direction = Direction.LONG
            score = min(100, 50 + (self.rsi_oversold - rsi) * 2)
        elif rsi > self.rsi_overbought and price >= latest["bb_upper"]:
            direction = Direction.SHORT
            score = min(100, 50 + (rsi - self.rsi_overbought) * 2)

        if direction is None:
            return None

        return Signal(
            symbol=symbol,
            direction=direction,
            score=score,
            strength=SignalStrength.MODERATE,
            strategy=self.name,
            reasoning=f"RSI={rsi:.1f} at Bollinger Band extreme (ADX={adx:.1f} = ranging)",
        )


class BreakoutStrategy(BaseStrategy):
    """Breakout detection from consolidation zones."""

    name = "breakout"

    def __init__(self, lookback: int = 20, vol_expansion: float = 1.5):
        self.lookback = lookback
        self.vol_expansion = vol_expansion

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if len(df) < self.lookback + 10:
            return None

        if "atr" not in df.columns or "volume_ratio" not in df.columns:
            return None

        latest = df.iloc[-1]
        recent = df.iloc[-self.lookback:]

        # Check for range breakout
        range_high = recent["high"].max()
        range_low = recent["low"].min()
        price = latest["close"]

        # Volume expansion
        vol_ratio = latest.get("volume_ratio", 1.0)
        if vol_ratio < self.vol_expansion:
            return None

        direction = None
        if price > range_high:
            direction = Direction.LONG
        elif price < range_low:
            direction = Direction.SHORT

        if direction is None:
            return None

        score = min(100, 60 + (vol_ratio - self.vol_expansion) * 20)
        return Signal(
            symbol=symbol,
            direction=direction,
            score=score,
            strength=SignalStrength.STRONG if vol_ratio > 2.0 else SignalStrength.MODERATE,
            strategy=self.name,
            reasoning=(
                f"{'Upside' if direction == Direction.LONG else 'Downside'} breakout "
                f"from {self.lookback}-bar range with {vol_ratio:.1f}x volume"
            ),
        )


class MomentumStrategy(BaseStrategy):
    """Multi-indicator momentum strategy."""

    name = "momentum"

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if len(df) < 50:
            return None

        required = ["rsi", "macd_hist", "stoch_k", "adx"]
        if not all(c in df.columns for c in required):
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        # Score momentum signals
        bull_points = 0
        bear_points = 0

        # RSI momentum
        rsi = latest["rsi"]
        if 50 < rsi < 80:
            bull_points += 1
        elif 20 < rsi < 50:
            bear_points += 1

        # MACD histogram increasing
        if latest["macd_hist"] > prev["macd_hist"] and latest["macd_hist"] > 0:
            bull_points += 1
        elif latest["macd_hist"] < prev["macd_hist"] and latest["macd_hist"] < 0:
            bear_points += 1

        # Stochastic momentum
        stoch = latest["stoch_k"]
        if stoch > 50 and stoch < 80:
            bull_points += 1
        elif stoch < 50 and stoch > 20:
            bear_points += 1

        # ADX confirms trend
        if latest["adx"] > 20:
            if bull_points > bear_points:
                bull_points += 1
            elif bear_points > bull_points:
                bear_points += 1

        if bull_points >= 3:
            direction = Direction.LONG
            score = min(100, 50 + bull_points * 10)
        elif bear_points >= 3:
            direction = Direction.SHORT
            score = min(100, 50 + bear_points * 10)
        else:
            return None

        return Signal(
            symbol=symbol,
            direction=direction,
            score=score,
            strength=SignalStrength.MODERATE,
            strategy=self.name,
            reasoning=f"Momentum: {bull_points} bull / {bear_points} bear indicators aligned",
        )


class SMCStrategy(BaseStrategy):
    """Smart Money Concepts strategy using institutional order flow patterns.

    Looks for confluences between SMC signals (order blocks, FVG, BOS, CHoCH,
    liquidity sweeps) and trend alignment.
    """

    name = "smc"

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if len(df) < 50:
            return None

        # Check if SMC features are available
        smc_cols = ["ob_bullish_near", "ob_bearish_near", "fvg_bullish_near",
                     "fvg_bearish_near", "bos_bullish", "bos_bearish",
                     "choch_bullish", "choch_bearish", "liq_sweep_high", "liq_sweep_low"]
        if not all(c in df.columns for c in smc_cols):
            return None

        latest = df.iloc[-1]

        # Count bullish SMC signals
        bull_signals = sum([
            latest.get("ob_bullish_near", 0) > 0,
            latest.get("fvg_bullish_near", 0) > 0,
            latest.get("bos_bullish", 0) > 0,
            latest.get("choch_bullish", 0) > 0,
            latest.get("liq_sweep_low", 0) > 0,  # Sweep of lows = bullish reversal
        ])

        # Count bearish SMC signals
        bear_signals = sum([
            latest.get("ob_bearish_near", 0) > 0,
            latest.get("fvg_bearish_near", 0) > 0,
            latest.get("bos_bearish", 0) > 0,
            latest.get("choch_bearish", 0) > 0,
            latest.get("liq_sweep_high", 0) > 0,  # Sweep of highs = bearish reversal
        ])

        # Need at least 2 confluent SMC signals
        if bull_signals < 2 and bear_signals < 2:
            return None

        # Check trend alignment for extra confirmation
        trend_bull = 0
        trend_bear = 0
        if "ema_9" in df.columns and "ema_21" in df.columns:
            if latest.get("ema_9", 0) > latest.get("ema_21", 0):
                trend_bull += 1
            else:
                trend_bear += 1
        if "rsi_14" in df.columns:
            rsi = latest.get("rsi_14", 50)
            if rsi > 50:
                trend_bull += 1
            elif rsi < 50:
                trend_bear += 1

        if bull_signals > bear_signals:
            direction = Direction.LONG
            confluence = bull_signals + trend_bull
            score = min(100, 50 + confluence * 10)
        elif bear_signals > bull_signals:
            direction = Direction.SHORT
            confluence = bear_signals + trend_bear
            score = min(100, 50 + confluence * 10)
        else:
            return None

        strength = SignalStrength.STRONG if confluence >= 4 else SignalStrength.MODERATE

        return Signal(
            symbol=symbol,
            direction=direction,
            score=score,
            strength=strength,
            strategy=self.name,
            reasoning=(
                f"SMC: {bull_signals} bullish / {bear_signals} bearish signals "
                f"(OB/FVG/BOS/CHoCH/Sweep) with {confluence} total confluence"
            ),
        )


class DivergenceStrategy(BaseStrategy):
    """Divergence-based strategy using RSI and MACD divergence features."""

    name = "divergence"

    def evaluate(self, df: pd.DataFrame, symbol: str) -> Signal | None:
        if len(df) < 50:
            return None

        div_cols = ["rsi_bullish_div", "rsi_bearish_div", "macd_bullish_div", "macd_bearish_div"]
        if not all(c in df.columns for c in div_cols):
            return None

        latest = df.iloc[-1]

        bull_divs = sum([
            latest.get("rsi_bullish_div", 0) > 0,
            latest.get("macd_bullish_div", 0) > 0,
        ])

        bear_divs = sum([
            latest.get("rsi_bearish_div", 0) > 0,
            latest.get("macd_bearish_div", 0) > 0,
        ])

        if bull_divs == 0 and bear_divs == 0:
            return None

        # Confirm with oversold/overbought
        rsi = latest.get("rsi_14", 50)

        if bull_divs > 0 and rsi < 40:
            return Signal(
                symbol=symbol, direction=Direction.LONG,
                score=min(100, 55 + bull_divs * 15),
                strength=SignalStrength.MODERATE,
                strategy=self.name,
                reasoning=f"Bullish divergence ({bull_divs} indicators) with RSI={rsi:.0f}",
            )
        elif bear_divs > 0 and rsi > 60:
            return Signal(
                symbol=symbol, direction=Direction.SHORT,
                score=min(100, 55 + bear_divs * 15),
                strength=SignalStrength.MODERATE,
                strategy=self.name,
                reasoning=f"Bearish divergence ({bear_divs} indicators) with RSI={rsi:.0f}",
            )

        return None


# Registry of all built-in strategies
STRATEGY_REGISTRY: dict[str, type[BaseStrategy]] = {
    "trend_following": TrendFollowingStrategy,
    "mean_reversion": MeanReversionStrategy,
    "breakout": BreakoutStrategy,
    "momentum": MomentumStrategy,
    "smc": SMCStrategy,
    "divergence": DivergenceStrategy,
}


def get_all_strategies() -> list[BaseStrategy]:
    """Instantiate all registered strategies with default params."""
    return [cls() for cls in STRATEGY_REGISTRY.values()]
