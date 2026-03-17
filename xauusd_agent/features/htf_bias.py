"""
Higher-timeframe bias features for XAUUSD.

Distills multi-timeframe state into a compact feature vector that
summarises the directional bias from D1, H4, H1, and M15.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import talib

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

_EPS = 1e-10


def _ema(series: np.ndarray, period: int) -> np.ndarray:
    return talib.EMA(series, timeperiod=period)


def _rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    return talib.RSI(series, timeperiod=period)


def _atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    return talib.ATR(h, l, c, timeperiod=period)


def _trend_from_ema(close: np.ndarray, fast: int = 50, slow: int = 200) -> int:
    """Return +1 (bullish), -1 (bearish), or 0 (neutral) from EMA position."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    if ema_fast is None or ema_slow is None:
        return 0
    f_val = ema_fast[-1]
    s_val = ema_slow[-1]
    if np.isnan(f_val) or np.isnan(s_val):
        return 0
    if f_val > s_val:
        return 1
    elif f_val < s_val:
        return -1
    return 0


def _latest_rsi(close: np.ndarray, period: int = 14) -> float:
    """Return the most recent RSI value."""
    vals = _rsi(close, period)
    if vals is None or len(vals) == 0:
        return 50.0
    v = vals[-1]
    return float(v) if not np.isnan(v) else 50.0


def _swing_break(df: pd.DataFrame, lookback: int = 20) -> int:
    """Check if recent price broke swing high (+1) or swing low (-1)."""
    if len(df) < lookback + 1:
        return 0

    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)  # noqa: E741
    c = df["close"].values.astype(float)

    recent_h = h[-(lookback + 1) : -1]
    recent_l = l[-(lookback + 1) : -1]

    swing_high = np.nanmax(recent_h)
    swing_low = np.nanmin(recent_l)

    if c[-1] > swing_high:
        return 1
    elif c[-1] < swing_low:
        return -1
    return 0


def _ema_slope(close: np.ndarray, ema_period: int = 21, slope_bars: int = 5) -> float:
    """Slope of EMA over the last *slope_bars* bars, normalised by ATR."""
    ema_vals = _ema(close, ema_period)
    if ema_vals is None or len(ema_vals) < slope_bars + 1:
        return 0.0
    recent = ema_vals[-slope_bars:]
    if np.isnan(recent).any():
        return 0.0
    return float(recent[-1] - recent[0])


def _vwap_approx(df: pd.DataFrame) -> float:
    """Approximate session VWAP (cumulative typical * vol / cumvol)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["tick_volume"].astype(float)
    cum_tpv = (typical * vol).sum()
    cum_vol = vol.sum()
    if cum_vol < _EPS:
        return float(df["close"].iloc[-1])
    return float(cum_tpv / cum_vol)


def _support_resistance_distance(
    df: pd.DataFrame,
    lookback: int = 50,
) -> tuple[float, float]:
    """Distance to nearest support / resistance in ATR units."""
    if len(df) < lookback:
        return 0.0, 0.0

    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)  # noqa: E741
    c = df["close"].values.astype(float)

    atr_vals = _atr(h, l, c, period=14)
    curr_atr = atr_vals[-1] if not np.isnan(atr_vals[-1]) else 1.0
    curr_close = c[-1]

    # Use swing highs/lows from recent history as S/R levels
    window_h = h[-lookback:]
    window_l = l[-lookback:]

    # Find local peaks as resistance candidates
    resistances = []
    supports = []
    for i in range(2, len(window_h) - 2):
        if window_h[i] > window_h[i - 1] and window_h[i] > window_h[i - 2] and \
           window_h[i] > window_h[i + 1] and window_h[i] > window_h[i + 2]:
            if window_h[i] > curr_close:
                resistances.append(window_h[i])

        if window_l[i] < window_l[i - 1] and window_l[i] < window_l[i - 2] and \
           window_l[i] < window_l[i + 1] and window_l[i] < window_l[i + 2]:
            if window_l[i] < curr_close:
                supports.append(window_l[i])

    support_dist = 0.0
    resistance_dist = 0.0

    if supports:
        nearest_support = max(supports)  # closest below
        support_dist = (curr_close - nearest_support) / (curr_atr + _EPS)

    if resistances:
        nearest_resistance = min(resistances)  # closest above
        resistance_dist = (nearest_resistance - curr_close) / (curr_atr + _EPS)

    return support_dist, resistance_dist


# ── Public API ────────────────────────────────────────────────────────────

def compute_htf_bias_features(all_tf: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Compute higher-timeframe bias feature vector.

    Parameters
    ----------
    all_tf : dict[str, pd.DataFrame]
        Mapping of timeframe name to OHLCV DataFrame.  Expected keys
        include some subset of ``D1``, ``H4``, ``H1``, ``M15``.

    Returns
    -------
    dict
        Flat dictionary of bias features suitable for model input.
    """
    logger.info(
        "Computing HTF bias features from timeframes: %s",
        list(all_tf.keys()),
    )

    features: Dict[str, Any] = {}

    # ── D1 ──────────────────────────────────────────────────────────
    if "D1" in all_tf:
        d1 = all_tf["D1"]
        d1_c = d1["close"].values.astype(float)
        features["d1_trend"] = _trend_from_ema(d1_c, fast=50, slow=200)
        features["d1_rsi"] = _latest_rsi(d1_c, 14)

        macd_line, macd_signal, macd_hist = talib.MACD(
            d1_c, fastperiod=12, slowperiod=26, signalperiod=9,
        )
        features["d1_macd_hist"] = (
            float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else 0.0
        )
    else:
        features["d1_trend"] = 0
        features["d1_rsi"] = 50.0
        features["d1_macd_hist"] = 0.0

    # ── H4 ──────────────────────────────────────────────────────────
    if "H4" in all_tf:
        h4 = all_tf["H4"]
        h4_c = h4["close"].values.astype(float)
        features["h4_trend"] = _trend_from_ema(h4_c, fast=50, slow=200)
        features["h4_rsi"] = _latest_rsi(h4_c, 14)
        features["h4_swing_break"] = _swing_break(h4, lookback=20)
    else:
        features["h4_trend"] = 0
        features["h4_rsi"] = 50.0
        features["h4_swing_break"] = 0

    # ── H1 ──────────────────────────────────────────────────────────
    if "H1" in all_tf:
        h1 = all_tf["H1"]
        h1_c = h1["close"].values.astype(float)
        features["h1_ema_slope"] = _ema_slope(h1_c, ema_period=21, slope_bars=5)
        features["h1_rsi"] = _latest_rsi(h1_c, 14)
        vwap_val = _vwap_approx(h1)
        features["h1_above_vwap"] = bool(h1_c[-1] > vwap_val) if len(h1_c) > 0 else False
    else:
        features["h1_ema_slope"] = 0.0
        features["h1_rsi"] = 50.0
        features["h1_above_vwap"] = False

    # ── M15 ─────────────────────────────────────────────────────────
    if "M15" in all_tf:
        m15 = all_tf["M15"]
        sup_dist, res_dist = _support_resistance_distance(m15, lookback=50)
        features["m15_support_distance"] = sup_dist
        features["m15_resistance_distance"] = res_dist
    else:
        features["m15_support_distance"] = 0.0
        features["m15_resistance_distance"] = 0.0

    logger.info("HTF bias features: %s", features)
    return features
