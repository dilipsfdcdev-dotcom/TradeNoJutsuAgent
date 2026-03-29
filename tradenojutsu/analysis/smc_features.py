"""
Smart Money Concepts (SMC) feature engineering.

Implements institutional-style price action concepts:
  - Order blocks (OB)
  - Fair value gaps (FVG)
  - Break of structure (BOS)
  - Change of character (CHoCH)
  - Liquidity sweeps

Inspired by andywarui/xaubot methodology.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from ta.volatility import average_true_range

from tradenojutsu.infra.logger import get_logger

logger = get_logger(__name__)

_EPS = 1e-10


# ── Swing-point detection ─────────────────────────────────────────────────

def find_swing_points(
    df: pd.DataFrame,
    lookback: int = 10,
) -> Tuple[pd.Series, pd.Series]:
    """Identify swing highs and swing lows.

    A swing high at bar *i* means ``high[i]`` is the highest high in the
    window ``[i - lookback, i + lookback]``.  Swing lows are analogous.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``high`` and ``low`` columns.
    lookback : int
        Number of bars on each side to confirm a swing point.

    Returns
    -------
    swing_highs : pd.Series
        Float series – swing high price where confirmed, else ``NaN``.
    swing_lows : pd.Series
        Float series – swing low price where confirmed, else ``NaN``.
    """
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(high)

    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        window_h = high[i - lookback : i + lookback + 1]
        if high[i] == window_h.max() and np.sum(window_h == high[i]) == 1:
            sh[i] = high[i]

        window_l = low[i - lookback : i + lookback + 1]
        if low[i] == window_l.min() and np.sum(window_l == low[i]) == 1:
            sl[i] = low[i]

    return (
        pd.Series(sh, index=df.index, name="swing_high"),
        pd.Series(sl, index=df.index, name="swing_low"),
    )


# ── Order blocks ──────────────────────────────────────────────────────────

def _detect_order_blocks(
    df: pd.DataFrame,
    atr: np.ndarray,
    impulse_mult: float = 1.5,
    proximity_mult: float = 0.5,
    max_ob_age: int = 50,
) -> Tuple[pd.Series, pd.Series]:
    """Detect bullish and bearish order blocks and proximity signals.

    Bullish OB: last bearish candle before an impulsive bullish move
    (body > impulse_mult * ATR).
    Bearish OB: last bullish candle before an impulsive bearish move.
    """
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)  # noqa: E741
    c = df["close"].values.astype(float)
    n = len(c)

    ob_bull_near = np.zeros(n, dtype=float)
    ob_bear_near = np.zeros(n, dtype=float)

    # Collect OBs as (index, ob_high, ob_low)
    bull_obs: list[tuple[int, float, float]] = []
    bear_obs: list[tuple[int, float, float]] = []

    for i in range(1, n):
        body = abs(c[i] - o[i])
        curr_atr = atr[i] if not np.isnan(atr[i]) else 1.0

        # Bullish impulse at bar i → look for bearish candle at i-1
        if c[i] > o[i] and body > impulse_mult * curr_atr:
            # Find the last bearish candle before this impulse
            for j in range(i - 1, max(i - 6, -1), -1):
                if c[j] < o[j]:  # bearish candle
                    bull_obs.append((j, h[j], l[j]))
                    break

        # Bearish impulse
        if c[i] < o[i] and body > impulse_mult * curr_atr:
            for j in range(i - 1, max(i - 6, -1), -1):
                if c[j] > o[j]:  # bullish candle
                    bear_obs.append((j, h[j], l[j]))
                    break

    # Check proximity for each bar
    for i in range(n):
        curr_atr = atr[i] if not np.isnan(atr[i]) else 1.0
        threshold = proximity_mult * curr_atr

        # Bullish OBs – price near OB low zone
        for ob_idx, ob_high, ob_low in bull_obs:
            if ob_idx >= i:
                continue
            if i - ob_idx > max_ob_age:
                continue
            if abs(c[i] - ob_low) <= threshold or (ob_low <= c[i] <= ob_high):
                ob_bull_near[i] = 1.0
                break

        # Bearish OBs – price near OB high zone
        for ob_idx, ob_high, ob_low in bear_obs:
            if ob_idx >= i:
                continue
            if i - ob_idx > max_ob_age:
                continue
            if abs(c[i] - ob_high) <= threshold or (ob_low <= c[i] <= ob_high):
                ob_bear_near[i] = 1.0
                break

    return (
        pd.Series(ob_bull_near, index=df.index),
        pd.Series(ob_bear_near, index=df.index),
    )


# ── Fair value gaps ───────────────────────────────────────────────────────

def _detect_fvg(
    df: pd.DataFrame,
    atr: np.ndarray,
    proximity_mult: float = 0.5,
    max_fvg_age: int = 50,
) -> Tuple[pd.Series, pd.Series]:
    """Detect bullish and bearish fair value gaps (3-candle pattern)."""
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)  # noqa: E741
    c = df["close"].values.astype(float)
    n = len(c)

    fvg_bull_near = np.zeros(n, dtype=float)
    fvg_bear_near = np.zeros(n, dtype=float)

    # Collect FVGs: (bar_index, gap_high, gap_low)
    bull_fvgs: list[tuple[int, float, float]] = []
    bear_fvgs: list[tuple[int, float, float]] = []

    for i in range(2, n):
        # Bullish FVG: candle_3 low > candle_1 high  (gap up)
        if l[i] > h[i - 2]:
            bull_fvgs.append((i, l[i], h[i - 2]))

        # Bearish FVG: candle_1 low > candle_3 high  (gap down)
        if l[i - 2] > h[i]:
            bear_fvgs.append((i, l[i - 2], h[i]))

    for i in range(n):
        curr_atr = atr[i] if not np.isnan(atr[i]) else 1.0
        threshold = proximity_mult * curr_atr

        for fvg_idx, gap_hi, gap_lo in bull_fvgs:
            if fvg_idx >= i:
                continue
            if i - fvg_idx > max_fvg_age:
                continue
            if gap_lo <= c[i] <= gap_hi or abs(c[i] - gap_lo) <= threshold:
                fvg_bull_near[i] = 1.0
                break

        for fvg_idx, gap_hi, gap_lo in bear_fvgs:
            if fvg_idx >= i:
                continue
            if i - fvg_idx > max_fvg_age:
                continue
            if gap_lo <= c[i] <= gap_hi or abs(c[i] - gap_hi) <= threshold:
                fvg_bear_near[i] = 1.0
                break

    return (
        pd.Series(fvg_bull_near, index=df.index),
        pd.Series(fvg_bear_near, index=df.index),
    )


# ── Break of structure ────────────────────────────────────────────────────

def _detect_bos(
    df: pd.DataFrame,
    swing_highs: pd.Series,
    swing_lows: pd.Series,
    lookback: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """Break of structure: price breaks the most recent swing high/low."""
    c = df["close"].values.astype(float)
    sh = swing_highs.values
    sl = swing_lows.values
    n = len(c)

    bos_bull = np.zeros(n, dtype=float)
    bos_bear = np.zeros(n, dtype=float)

    last_sh = np.nan
    last_sl = np.nan

    for i in range(n):
        # Track latest confirmed swing points
        if not np.isnan(sh[i]):
            last_sh = sh[i]
        if not np.isnan(sl[i]):
            last_sl = sl[i]

        # Check if recent bars broke structure
        if not np.isnan(last_sh) and c[i] > last_sh:
            bos_bull[i] = 1.0

        if not np.isnan(last_sl) and c[i] < last_sl:
            bos_bear[i] = 1.0

    # Apply lookback window: mark True if BOS happened within last N bars
    bos_bull_s = pd.Series(bos_bull, index=df.index)
    bos_bear_s = pd.Series(bos_bear, index=df.index)
    bos_bull_recent = bos_bull_s.rolling(lookback, min_periods=1).max()
    bos_bear_recent = bos_bear_s.rolling(lookback, min_periods=1).max()

    return bos_bull_recent, bos_bear_recent


# ── Change of character ───────────────────────────────────────────────────

def _detect_choch(
    df: pd.DataFrame,
    swing_highs: pd.Series,
    swing_lows: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """Change of character: first BOS against the prevailing trend.

    If the market was making higher highs (bullish) and then breaks a
    swing low → CHoCH bearish.  Vice versa for bullish CHoCH.
    """
    c = df["close"].values.astype(float)
    sh = swing_highs.values
    sl = swing_lows.values
    n = len(c)

    choch_bull = np.zeros(n, dtype=float)
    choch_bear = np.zeros(n, dtype=float)

    # Track trend direction based on consecutive swing highs
    prev_sh = np.nan
    prev_sl = np.nan
    trend = 0  # +1 bullish, -1 bearish, 0 neutral

    last_sh = np.nan
    last_sl = np.nan

    for i in range(n):
        if not np.isnan(sh[i]):
            if not np.isnan(prev_sh):
                if sh[i] > prev_sh:
                    trend = 1
                elif sh[i] < prev_sh:
                    trend = -1
            prev_sh = sh[i]
            last_sh = sh[i]

        if not np.isnan(sl[i]):
            if not np.isnan(prev_sl):
                if sl[i] > prev_sl:
                    trend = 1
                elif sl[i] < prev_sl:
                    trend = -1
            prev_sl = sl[i]
            last_sl = sl[i]

        # CHoCH: break against trend
        if trend == 1 and not np.isnan(last_sl) and c[i] < last_sl:
            choch_bear[i] = 1.0
        elif trend == -1 and not np.isnan(last_sh) and c[i] > last_sh:
            choch_bull[i] = 1.0

    return (
        pd.Series(choch_bull, index=df.index),
        pd.Series(choch_bear, index=df.index),
    )


# ── Liquidity sweeps ─────────────────────────────────────────────────────

def _detect_liquidity_sweeps(
    df: pd.DataFrame,
    swing_highs: pd.Series,
    swing_lows: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """Liquidity sweeps: wick beyond swing level but close back inside.

    Bearish sweep: high > swing_high but close < swing_high.
    Bullish sweep: low < swing_low but close > swing_low.
    """
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)  # noqa: E741
    c = df["close"].values.astype(float)
    sh = swing_highs.values
    sl = swing_lows.values
    n = len(c)

    sweep_high = np.zeros(n, dtype=float)
    sweep_low = np.zeros(n, dtype=float)

    last_sh = np.nan
    last_sl = np.nan

    for i in range(n):
        if not np.isnan(sh[i]):
            last_sh = sh[i]
        if not np.isnan(sl[i]):
            last_sl = sl[i]

        # Bearish sweep: spike above swing high then close below
        if not np.isnan(last_sh) and h[i] > last_sh and c[i] < last_sh:
            sweep_high[i] = 1.0

        # Bullish sweep: spike below swing low then close above
        if not np.isnan(last_sl) and l[i] < last_sl and c[i] > last_sl:
            sweep_low[i] = 1.0

    return (
        pd.Series(sweep_high, index=df.index),
        pd.Series(sweep_low, index=df.index),
    )


# ── Public API ────────────────────────────────────────────────────────────

def compute_smc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Smart Money Concepts features from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume.

    Returns
    -------
    pd.DataFrame
        Boolean / float SMC feature columns aligned to the input index.
    """
    logger.info("Computing SMC features (%d rows)", len(df))

    # ATR for proximity thresholds (using ta library)
    atr_series = average_true_range(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )
    atr = atr_series.values.astype(float)

    # Swing points
    swing_highs, swing_lows = find_swing_points(df, lookback=10)

    # Order blocks
    ob_bull_near, ob_bear_near = _detect_order_blocks(df, atr)

    # Fair value gaps
    fvg_bull_near, fvg_bear_near = _detect_fvg(df, atr)

    # Break of structure
    bos_bull, bos_bear = _detect_bos(df, swing_highs, swing_lows, lookback=5)

    # Change of character
    choch_bull, choch_bear = _detect_choch(df, swing_highs, swing_lows)

    # Liquidity sweeps
    liq_sweep_high, liq_sweep_low = _detect_liquidity_sweeps(
        df, swing_highs, swing_lows,
    )

    result = pd.DataFrame(
        {
            "ob_bullish_near": ob_bull_near,
            "ob_bearish_near": ob_bear_near,
            "fvg_bullish_near": fvg_bull_near,
            "fvg_bearish_near": fvg_bear_near,
            "bos_bullish": bos_bull,
            "bos_bearish": bos_bear,
            "choch_bullish": choch_bull,
            "choch_bearish": choch_bear,
            "liq_sweep_high": liq_sweep_high,
            "liq_sweep_low": liq_sweep_low,
        },
        index=df.index,
    )

    logger.info("SMC features computed: %d columns", result.shape[1])
    return result
