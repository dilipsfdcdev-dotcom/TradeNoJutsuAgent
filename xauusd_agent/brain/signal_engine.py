"""
M3 / M5 signal scoring engine.

Independently scores BUY and SELL opportunities from 0-100 based on
short-time-frame technical conditions, then applies an HTF directional
multiplier to reward with-trend entries and penalise counter-trend ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lightweight indicator helpers (duplicated from htf_analyzer to keep each
# module independently importable without circular deps)
# ---------------------------------------------------------------------------


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    return macd_line - signal_line


def _swing_lows(series: pd.Series, lookback: int) -> pd.Series:
    rolled_min = series.rolling(window=2 * lookback + 1, center=True).min()
    return series == rolled_min


def _swing_highs(series: pd.Series, lookback: int) -> pd.Series:
    rolled_max = series.rolling(window=2 * lookback + 1, center=True).max()
    return series == rolled_max


def _last_n_swing_values(series: pd.Series, is_swing: pd.Series, n: int = 3) -> list[float]:
    """Return the last *n* swing values (excluding current bar)."""
    mask = is_swing.iloc[:-1]
    vals = series.loc[mask]
    return [float(v) for v in vals.iloc[-n:]] if len(vals) >= n else [float(v) for v in vals]


# ---------------------------------------------------------------------------
# M3M5SignalEngine
# ---------------------------------------------------------------------------


class M3M5SignalEngine:
    """Scores BUY and SELL from 0-100 independently."""

    def __init__(self, settings: dict) -> None:
        self.entry_threshold: float = settings.get("entry_threshold", 62)
        self.htf_multipliers: dict[str, float] = {
            "STRONG_BULL": settings.get("strong_bull_mult", 1.15),
            "BULL": settings.get("bull_mult", 1.05),
            "NEUTRAL": settings.get("neutral_mult", 1.00),
            "BEAR": settings.get("bear_mult", 0.82),
            "STRONG_BEAR": settings.get("strong_bear_mult", 0.62),
        }

    # --------------------------------------------------------------------- #
    #  BUY scoring                                                           #
    # --------------------------------------------------------------------- #

    def score_buy(
        self,
        m3: pd.DataFrame,
        m5: pd.DataFrame,
        m15: pd.DataFrame,
        htf_bias: dict,
        spread_pips: float,
    ) -> tuple[float, dict]:
        """
        Score a potential BUY from 0-100.

        Returns
        -------
        (final_score, components)
            *components* maps component name to ``(fired: bool, points: int)``.
        """
        components: dict[str, tuple[bool, int]] = {}
        raw = 0

        m3_close = m3["close"]
        m5_close = m5["close"]

        # 1. M3 RSI(7) oversold  (< 40) ................................ 12 pts
        m3_rsi7 = float(_rsi(m3_close, 7).iloc[-1])
        fired = m3_rsi7 < 40
        components["m3_rsi_oversold"] = (fired, 12 if fired else 0)
        if fired:
            raw += 12

        # 2. M5 RSI(14) confirming  (< 47) ............................. 10 pts
        m5_rsi14 = float(_rsi(m5_close, 14).iloc[-1])
        fired = m5_rsi14 < 47
        components["m5_rsi_confirming"] = (fired, 10 if fired else 0)
        if fired:
            raw += 10

        # 3. M3 MACD histogram bullish (> 0 or crossover) .............. 12 pts
        m3_hist = _macd_hist(m3_close)
        last_hist = float(m3_hist.iloc[-1])
        prev_hist = float(m3_hist.iloc[-2]) if len(m3_hist) >= 2 else 0.0
        fired = last_hist > 0 or (prev_hist < 0 and last_hist >= 0)
        components["m3_macd_bull"] = (fired, 12 if fired else 0)
        if fired:
            raw += 12

        # 4. M5 MACD histogram > 0 ..................................... 8 pts
        m5_hist = float(_macd_hist(m5_close).iloc[-1])
        fired = m5_hist > 0
        components["m5_macd_bull"] = (fired, 8 if fired else 0)
        if fired:
            raw += 8

        # 5. M3 close > EMA(9) ......................................... 8 pts
        m3_ema9 = float(_ema(m3_close, 9).iloc[-1])
        fired = float(m3_close.iloc[-1]) > m3_ema9
        components["m3_above_ema9"] = (fired, 8 if fired else 0)
        if fired:
            raw += 8

        # 6. M5 close > EMA(21) ........................................ 8 pts
        m5_ema21 = float(_ema(m5_close, 21).iloc[-1])
        fired = float(m5_close.iloc[-1]) > m5_ema21
        components["m5_above_ema21"] = (fired, 8 if fired else 0)
        if fired:
            raw += 8

        # 7. M15 at support (from HTF bias) ............................ 10 pts
        fired = bool(htf_bias.get("m15_at_support", False))
        components["m15_at_support"] = (fired, 10 if fired else 0)
        if fired:
            raw += 10

        # 8. M3 & M5 making higher lows ................................ 10 pts
        fired = self._higher_lows(m3, m5)
        components["m3m5_higher_lows"] = (fired, 10 if fired else 0)
        if fired:
            raw += 10

        # 9. M3 bullish candle (body > 55% of range) ................... 7 pts
        fired = self._bullish_candle(m3)
        components["m3_bullish_candle"] = (fired, 7 if fired else 0)
        if fired:
            raw += 7

        # 10. Volume spike (M3 vol > 1.4x 10-bar avg) .................. 7 pts
        fired = self._volume_spike(m3)
        components["volume_spike"] = (fired, 7 if fired else 0)
        if fired:
            raw += 7

        # 11. Spread OK (< 6 pips) ..................................... 8 pts
        fired = spread_pips < 6.0
        components["spread_ok"] = (fired, 8 if fired else 0)
        if fired:
            raw += 8

        # Apply HTF multiplier
        direction = htf_bias.get("direction", "NEUTRAL")
        multiplier = self.htf_multipliers.get(direction, 1.0)
        final = min(float(raw) * multiplier, 100.0)

        return final, components

    # --------------------------------------------------------------------- #
    #  SELL scoring                                                          #
    # --------------------------------------------------------------------- #

    def score_sell(
        self,
        m3: pd.DataFrame,
        m5: pd.DataFrame,
        m15: pd.DataFrame,
        htf_bias: dict,
        spread_pips: float,
    ) -> tuple[float, dict]:
        """
        Score a potential SELL from 0-100 (mirror of :meth:`score_buy`).
        """
        components: dict[str, tuple[bool, int]] = {}
        raw = 0

        m3_close = m3["close"]
        m5_close = m5["close"]

        # 1. M3 RSI(7) overbought (> 60) ............................... 12 pts
        m3_rsi7 = float(_rsi(m3_close, 7).iloc[-1])
        fired = m3_rsi7 > 60
        components["m3_rsi_overbought"] = (fired, 12 if fired else 0)
        if fired:
            raw += 12

        # 2. M5 RSI(14) confirming (> 53) .............................. 10 pts
        m5_rsi14 = float(_rsi(m5_close, 14).iloc[-1])
        fired = m5_rsi14 > 53
        components["m5_rsi_confirming"] = (fired, 10 if fired else 0)
        if fired:
            raw += 10

        # 3. M3 MACD histogram bearish (< 0 or crossover down) ........ 12 pts
        m3_hist = _macd_hist(m3_close)
        last_hist = float(m3_hist.iloc[-1])
        prev_hist = float(m3_hist.iloc[-2]) if len(m3_hist) >= 2 else 0.0
        fired = last_hist < 0 or (prev_hist > 0 and last_hist <= 0)
        components["m3_macd_bear"] = (fired, 12 if fired else 0)
        if fired:
            raw += 12

        # 4. M5 MACD histogram < 0 ..................................... 8 pts
        m5_hist = float(_macd_hist(m5_close).iloc[-1])
        fired = m5_hist < 0
        components["m5_macd_bear"] = (fired, 8 if fired else 0)
        if fired:
            raw += 8

        # 5. M3 close < EMA(9) ......................................... 8 pts
        m3_ema9 = float(_ema(m3_close, 9).iloc[-1])
        fired = float(m3_close.iloc[-1]) < m3_ema9
        components["m3_below_ema9"] = (fired, 8 if fired else 0)
        if fired:
            raw += 8

        # 6. M5 close < EMA(21) ........................................ 8 pts
        m5_ema21 = float(_ema(m5_close, 21).iloc[-1])
        fired = float(m5_close.iloc[-1]) < m5_ema21
        components["m5_below_ema21"] = (fired, 8 if fired else 0)
        if fired:
            raw += 8

        # 7. M15 at resistance (from HTF bias) ......................... 10 pts
        fired = bool(htf_bias.get("m15_at_resist", False))
        components["m15_at_resist"] = (fired, 10 if fired else 0)
        if fired:
            raw += 10

        # 8. M3 & M5 making lower highs ................................ 10 pts
        fired = self._lower_highs(m3, m5)
        components["m3m5_lower_highs"] = (fired, 10 if fired else 0)
        if fired:
            raw += 10

        # 9. M3 bearish candle (body > 55% of range, close < open) ..... 7 pts
        fired = self._bearish_candle(m3)
        components["m3_bearish_candle"] = (fired, 7 if fired else 0)
        if fired:
            raw += 7

        # 10. Volume spike (M3 vol > 1.4x 10-bar avg) .................. 7 pts
        fired = self._volume_spike(m3)
        components["volume_spike"] = (fired, 7 if fired else 0)
        if fired:
            raw += 7

        # 11. Spread OK (< 6 pips) ..................................... 8 pts
        fired = spread_pips < 6.0
        components["spread_ok"] = (fired, 8 if fired else 0)
        if fired:
            raw += 8

        # Apply HTF multiplier (inverted: bearish bias helps sells)
        direction = htf_bias.get("direction", "NEUTRAL")
        sell_mult_map = {
            "STRONG_BEAR": self.htf_multipliers.get("STRONG_BULL", 1.15),
            "BEAR": self.htf_multipliers.get("BULL", 1.05),
            "NEUTRAL": self.htf_multipliers.get("NEUTRAL", 1.00),
            "BULL": self.htf_multipliers.get("BEAR", 0.82),
            "STRONG_BULL": self.htf_multipliers.get("STRONG_BEAR", 0.62),
        }
        multiplier = sell_mult_map.get(direction, 1.0)
        final = min(float(raw) * multiplier, 100.0)

        return final, components

    # --------------------------------------------------------------------- #
    #  Composite signal                                                      #
    # --------------------------------------------------------------------- #

    def get_signal(
        self,
        all_tf: dict,
        htf_bias: dict,
        spread_pips: float,
    ) -> dict:
        """
        Return the composite signal decision.

        Returns
        -------
        dict
            ``action``           – ``"BUY"`` | ``"SELL"`` | ``"HOLD"``
            ``buy_score``        – float (0-100)
            ``sell_score``       – float (0-100)
            ``threshold``        – float
            ``is_counter_trend`` – bool
            ``components``       – dict of scored components
            ``htf_bias``         – the bias dict passed in
        """
        m3 = all_tf["M3"]
        m5 = all_tf["M5"]
        m15 = all_tf["M15"]

        buy_score, buy_comp = self.score_buy(m3, m5, m15, htf_bias, spread_pips)
        sell_score, sell_comp = self.score_sell(m3, m5, m15, htf_bias, spread_pips)

        components = {"buy": buy_comp, "sell": sell_comp}

        buy_above = buy_score >= self.entry_threshold
        sell_above = sell_score >= self.entry_threshold

        if buy_above and sell_above:
            action = "BUY" if buy_score >= sell_score else "SELL"
        elif buy_above:
            action = "BUY"
        elif sell_above:
            action = "SELL"
        else:
            action = "HOLD"

        direction = htf_bias.get("direction", "NEUTRAL")
        is_counter_trend = (
            (action == "BUY" and direction in ("BEAR", "STRONG_BEAR"))
            or (action == "SELL" and direction in ("BULL", "STRONG_BULL"))
        )

        result = {
            "action": action,
            "buy_score": round(buy_score, 2),
            "sell_score": round(sell_score, 2),
            "threshold": self.entry_threshold,
            "is_counter_trend": is_counter_trend,
            "components": components,
            "htf_bias": htf_bias,
        }

        logger.info(
            "Signal scored",
            extra={
                "action": action,
                "buy_score": result["buy_score"],
                "sell_score": result["sell_score"],
                "counter_trend": is_counter_trend,
            },
        )
        return result

    # --------------------------------------------------------------------- #
    #  Private helpers                                                       #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _higher_lows(m3: pd.DataFrame, m5: pd.DataFrame) -> bool:
        """True if both M3 and M5 show ascending swing lows (last 3)."""
        for df in (m3, m5):
            sl = _swing_lows(df["low"], lookback=5)
            vals = _last_n_swing_values(df["low"], sl, n=3)
            if len(vals) < 3:
                return False
            if not (vals[-1] > vals[-2] > vals[-3]):
                return False
        return True

    @staticmethod
    def _lower_highs(m3: pd.DataFrame, m5: pd.DataFrame) -> bool:
        """True if both M3 and M5 show descending swing highs (last 3)."""
        for df in (m3, m5):
            sh = _swing_highs(df["high"], lookback=5)
            vals = _last_n_swing_values(df["high"], sh, n=3)
            if len(vals) < 3:
                return False
            if not (vals[-1] < vals[-2] < vals[-3]):
                return False
        return True

    @staticmethod
    def _bullish_candle(m3: pd.DataFrame) -> bool:
        """Last M3 candle body > 55 % of range and close > open."""
        row = m3.iloc[-1]
        rng = row["high"] - row["low"]
        if rng == 0:
            return False
        body = abs(row["close"] - row["open"])
        return (body / rng) > 0.55 and row["close"] > row["open"]

    @staticmethod
    def _bearish_candle(m3: pd.DataFrame) -> bool:
        """Last M3 candle body > 55 % of range and close < open."""
        row = m3.iloc[-1]
        rng = row["high"] - row["low"]
        if rng == 0:
            return False
        body = abs(row["close"] - row["open"])
        return (body / rng) > 0.55 and row["close"] < row["open"]

    @staticmethod
    def _volume_spike(m3: pd.DataFrame) -> bool:
        """Last bar volume > 1.4x the 10-bar average."""
        if "volume" not in m3.columns or len(m3) < 11:
            return False
        avg_vol = float(m3["volume"].iloc[-11:-1].mean())
        if avg_vol <= 0:
            return False
        return float(m3["volume"].iloc[-1]) > 1.4 * avg_vol
