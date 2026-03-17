"""
Higher-Time-Frame bias engine.

Reads D1, H4, H1 candle DataFrames and produces a single directional bias
score in the range ``[-1.0, +1.0]``.  The score is consumed downstream by the
signal combiner to gate and weight entry decisions.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _swing_highs(series: pd.Series, lookback: int) -> pd.Series:
    """Return a boolean Series that is True at swing-high bars."""
    rolled_max = series.rolling(window=2 * lookback + 1, center=True).max()
    return series == rolled_max


def _swing_lows(series: pd.Series, lookback: int) -> pd.Series:
    """Return a boolean Series that is True at swing-low bars."""
    rolled_min = series.rolling(window=2 * lookback + 1, center=True).min()
    return series == rolled_min


def _last_swing_value(
    series: pd.Series, is_swing: pd.Series, default: float
) -> float:
    """Return the most recent swing value, or *default* if none found."""
    idx = is_swing.iloc[:-1]  # exclude the current bar
    if idx.any():
        return float(series.loc[idx].iloc[-1])
    return default


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# HTFBiasEngine
# ---------------------------------------------------------------------------


class HTFBiasEngine:
    """Reads D1, H4, H1 and outputs a single bias score: -1.0 to +1.0."""

    def __init__(self, settings: dict) -> None:
        self.d1_weight: float = settings.get("d1_weight", 0.40)
        self.h4_weight: float = settings.get("h4_weight", 0.35)
        self.h1_weight: float = settings.get("h1_weight", 0.25)
        self._cache: dict | None = None
        self._cache_time: float = 0.0

    # ---- D1 -----------------------------------------------------------------

    def d1_bias(self, d1: pd.DataFrame) -> float:
        """
        Scoring rubric (sums then clamps to ``[-1, +1]``):

        * +0.30 / -0.30  price above / below EMA(200)
        * +0.20 / -0.20  price above / below EMA(50)
        * +0.30 / -0.30  MACD histogram positive / negative
        * +0.20 / -0.20  RSI(14) > 55 / < 45  (0 if 45-55)
        """
        if len(d1) < 200:
            logger.warning("D1 DataFrame has fewer than 200 rows; bias may be unreliable")
            if len(d1) < 2:
                return 0.0

        close = d1["close"]
        last_close = float(close.iloc[-1])

        ema200 = float(_ema(close, 200).iloc[-1])
        ema50 = float(_ema(close, 50).iloc[-1])
        _, _, hist = _macd(close)
        last_hist = float(hist.iloc[-1])
        last_rsi = float(_rsi(close, 14).iloc[-1])

        score = 0.0
        # EMA 200
        score += 0.30 if last_close > ema200 else -0.30
        # EMA 50
        score += 0.20 if last_close > ema50 else -0.20
        # MACD histogram
        score += 0.30 if last_hist > 0 else -0.30
        # RSI
        if last_rsi > 55:
            score += 0.20
        elif last_rsi < 45:
            score -= 0.20
        # else 0

        return _clamp(score)

    # ---- H4 -----------------------------------------------------------------

    def h4_bias(self, h4: pd.DataFrame) -> float:
        """
        Scoring rubric (sums then clamps to ``[-1, +1]``):

        * +0.40 / -0.40  last swing high broken upward / swing low broken downward
        * +0.30 / -0.30  price above / below EMA(50)
        * +0.30 / -0.30  ATR rising in trend direction / falling against direction
        """
        if len(h4) < 21:
            logger.warning("H4 DataFrame too short for reliable bias")
            return 0.0

        close = h4["close"]
        high = h4["high"]
        low = h4["low"]
        last_close = float(close.iloc[-1])

        # Swing detection with 10-bar lookback
        sh = _swing_highs(high, lookback=10)
        sl = _swing_lows(low, lookback=10)
        last_sh = _last_swing_value(high, sh, default=last_close)
        last_sl = _last_swing_value(low, sl, default=last_close)

        score = 0.0

        # Swing break
        if last_close > last_sh:
            score += 0.40
        elif last_close < last_sl:
            score -= 0.40

        # EMA 50
        ema50 = float(_ema(close, 50).iloc[-1])
        score += 0.30 if last_close > ema50 else -0.30

        # ATR momentum: ATR(7) vs ATR(20)
        atr7 = float(_atr(h4, 7).iloc[-1])
        atr20 = float(_atr(h4, 20).iloc[-1])
        atr_rising = atr7 > atr20
        trend_up = last_close > ema50
        if atr_rising and trend_up:
            score += 0.30
        elif atr_rising and not trend_up:
            score -= 0.30
        elif not atr_rising and trend_up:
            score -= 0.30
        else:
            score += 0.30  # ATR contracting in a downtrend (consolidation)

        return _clamp(score)

    # ---- H1 -----------------------------------------------------------------

    def h1_bias(self, h1: pd.DataFrame) -> float:
        """
        Scoring rubric (sums then clamps to ``[-1, +1]``):

        * +0.40 / -0.40  EMA(21) slope rising / falling (last bar vs 5 bars ago)
        * +0.30 / -0.30  RSI(14) > 55 / < 45
        * +0.30 / -0.30  price above / below session VWAP approximation
        """
        if len(h1) < 22:
            logger.warning("H1 DataFrame too short for reliable bias")
            return 0.0

        close = h1["close"]
        last_close = float(close.iloc[-1])

        # EMA(21) slope
        ema21 = _ema(close, 21)
        ema_now = float(ema21.iloc[-1])
        ema_5ago = float(ema21.iloc[-6]) if len(ema21) >= 6 else ema_now
        score = 0.0
        score += 0.40 if ema_now > ema_5ago else -0.40

        # RSI
        last_rsi = float(_rsi(close, 14).iloc[-1])
        if last_rsi > 55:
            score += 0.30
        elif last_rsi < 45:
            score -= 0.30

        # Session VWAP approximation
        # Use typical price * volume cumulative ratio
        tp = (h1["high"] + h1["low"] + h1["close"]) / 3.0
        if "volume" in h1.columns and h1["volume"].sum() > 0:
            cum_tpv = (tp * h1["volume"]).cumsum()
            cum_vol = h1["volume"].cumsum().replace(0.0, np.nan)
            vwap = cum_tpv / cum_vol
            last_vwap = float(vwap.iloc[-1])
        else:
            # Fallback: simple mean of typical prices as pseudo-VWAP
            last_vwap = float(tp.mean())

        score += 0.30 if last_close > last_vwap else -0.30

        return _clamp(score)

    # ---- Main entry point ---------------------------------------------------

    def get_bias(self, all_tf: dict) -> dict:
        """
        Compute the composite bias and M15 proximity flags.

        Parameters
        ----------
        all_tf : dict
            Must contain keys ``"D1"``, ``"H4"``, ``"H1"``, ``"M15"`` each
            mapping to a ``pd.DataFrame`` with OHLCV columns.

        Returns
        -------
        dict
            ``score``          – float in [-1, +1]
            ``direction``      – one of STRONG_BULL | BULL | NEUTRAL | BEAR | STRONG_BEAR
            ``trade_longs``    – bool, True when score > -0.15
            ``trade_shorts``   – bool, True when score < +0.15
            ``d1`` / ``h4`` / ``h1`` – individual bias floats
            ``m15_at_support`` – bool, M15 price within 0.5 ATR of swing low
            ``m15_at_resist``  – bool, M15 price within 0.5 ATR of swing high
        """
        d1_val = self.d1_bias(all_tf["D1"])
        h4_val = self.h4_bias(all_tf["H4"])
        h1_val = self.h1_bias(all_tf["H1"])

        score = _clamp(
            d1_val * self.d1_weight
            + h4_val * self.h4_weight
            + h1_val * self.h1_weight
        )

        # Direction label
        if score > 0.50:
            direction = "STRONG_BULL"
        elif score > 0.15:
            direction = "BULL"
        elif score >= -0.15:
            direction = "NEUTRAL"
        elif score >= -0.50:
            direction = "BEAR"
        else:
            direction = "STRONG_BEAR"

        # M15 support / resistance proximity
        m15 = all_tf.get("M15")
        m15_at_support = False
        m15_at_resist = False

        if m15 is not None and len(m15) >= 41:
            m15_close = float(m15["close"].iloc[-1])
            m15_atr = float(_atr(m15, 14).iloc[-1])
            threshold = 0.5 * m15_atr

            sh = _swing_highs(m15["high"], lookback=20)
            sl = _swing_lows(m15["low"], lookback=20)

            last_sh = _last_swing_value(m15["high"], sh, default=m15_close + threshold + 1)
            last_sl = _last_swing_value(m15["low"], sl, default=m15_close - threshold - 1)

            m15_at_resist = abs(m15_close - last_sh) <= threshold
            m15_at_support = abs(m15_close - last_sl) <= threshold
        elif m15 is not None:
            logger.warning("M15 DataFrame too short for support/resistance detection")

        result = {
            "score": round(score, 4),
            "direction": direction,
            "trade_longs": score > -0.15,
            "trade_shorts": score < 0.15,
            "d1": round(d1_val, 4),
            "h4": round(h4_val, 4),
            "h1": round(h1_val, 4),
            "m15_at_support": m15_at_support,
            "m15_at_resist": m15_at_resist,
        }

        self._cache = result
        self._cache_time = time.time()

        logger.info(
            "HTF bias computed",
            extra={"score": result["score"], "direction": direction},
        )
        return result
