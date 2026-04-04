"""
Technical indicators for scalping strategies.

All indicators operate on a pd.DataFrame with columns:
    time, open, high, low, close, volume

Each function adds column(s) to the dataframe and returns it.
Implemented from scratch using numpy/pandas for reliability.
"""

import numpy as np
import pandas as pd


def compute_ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.DataFrame:
    """Compute Exponential Moving Average.

    Adds column ``ema_{period}`` to *df*.

    EMA formula:
        multiplier = 2 / (period + 1)
        EMA_t = close_t * mult + EMA_{t-1} * (1 - mult)

    The first ``period`` values use an expanding SMA as the seed so the warm-up
    is identical to most charting platforms.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    if period < 1:
        raise ValueError("Period must be >= 1")

    col_name = f"ema_{period}"
    src = df[column].astype(float).values
    n = len(src)
    ema = np.full(n, np.nan)

    if n == 0 or period > n:
        df[col_name] = ema
        return df

    multiplier = 2.0 / (period + 1)

    # Seed: SMA of first `period` valid values
    first_valid = 0
    while first_valid < n and np.isnan(src[first_valid]):
        first_valid += 1

    if first_valid + period > n:
        df[col_name] = ema
        return df

    seed_slice = src[first_valid : first_valid + period]
    if np.any(np.isnan(seed_slice)):
        df[col_name] = ema
        return df

    sma_seed = np.mean(seed_slice)
    idx = first_valid + period - 1
    ema[idx] = sma_seed

    for i in range(idx + 1, n):
        val = src[i]
        if np.isnan(val):
            ema[i] = ema[i - 1]
        else:
            ema[i] = val * multiplier + ema[i - 1] * (1 - multiplier)

    df[col_name] = ema
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Relative Strength Index.

    Adds column ``rsi`` to *df*.

    Uses Wilder's smoothing (exponential) for average gain / average loss:
        RS  = avg_gain / avg_loss
        RSI = 100 - 100 / (1 + RS)
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")
    if period < 1:
        raise ValueError("Period must be >= 1")

    close = df["close"].astype(float).values
    n = len(close)
    rsi = np.full(n, np.nan)

    if n < period + 1:
        df["rsi"] = rsi
        return df

    deltas = np.diff(close)  # length n-1
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # First average: simple mean of the first `period` changes
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder's smoothing for subsequent values
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    df["rsi"] = rsi
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Average True Range.

    Adds column ``atr`` to *df*.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR is seeded with the SMA of the first ``period`` TR values,
    then smoothed using Wilder's method: ATR_t = (ATR_{t-1}*(period-1) + TR_t) / period
    """
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{col}' column")
    if period < 1:
        raise ValueError("Period must be >= 1")

    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    n = len(high)
    atr = np.full(n, np.nan)

    if n < 2:
        df["atr"] = atr
        return df

    # True Range array (index 0 has no previous close, use high-low)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    if n < period:
        df["atr"] = atr
        return df

    # Seed with SMA of first `period` TR values
    atr[period - 1] = np.mean(tr[:period])

    # Wilder smoothing
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    df["atr"] = atr
    return df


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Volume Weighted Average Price.

    Adds column ``vwap`` to *df*.

    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    typical_price = (high + low + close) / 3

    Computes a running (session-level) VWAP across the entire dataframe.
    """
    for col in ("high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{col}' column")

    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    close = df["close"].astype(float).values
    volume = df["volume"].astype(float).values

    typical_price = (high + low + close) / 3.0
    cum_tp_vol = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(cum_vol != 0, cum_tp_vol / cum_vol, np.nan)

    df["vwap"] = vwap
    return df


def compute_bollinger(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """Compute Bollinger Bands.

    Adds columns ``bb_upper``, ``bb_middle``, ``bb_lower`` to *df*.

    Middle = SMA(close, period)
    Upper  = Middle + std_dev * rolling_std(close, period)
    Lower  = Middle - std_dev * rolling_std(close, period)
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")
    if period < 1:
        raise ValueError("Period must be >= 1")

    close = df["close"].astype(float)

    middle = close.rolling(window=period, min_periods=period).mean()
    rolling_std = close.rolling(window=period, min_periods=period).std(ddof=0)

    df["bb_middle"] = middle.values
    df["bb_upper"] = (middle + std_dev * rolling_std).values
    df["bb_lower"] = (middle - std_dev * rolling_std).values

    return df


def compute_volume_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute approximate volume delta (buying vs selling pressure).

    Adds column ``volume_delta`` to *df*.

    Heuristic: if close > open the bar is bullish  -> +volume
               if close < open the bar is bearish  -> -volume
               if close == open                    ->  0
    """
    for col in ("open", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{col}' column")

    open_ = df["open"].astype(float).values
    close = df["close"].astype(float).values
    volume = df["volume"].astype(float).values

    sign = np.sign(close - open_)
    df["volume_delta"] = (volume * sign).astype(float)
    return df


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every indicator used by the scalping strategy.

    Adds the following columns:
        ema_9, ema_21, ema_50, rsi, atr, vwap,
        bb_upper, bb_middle, bb_lower, volume_delta

    Returns the dataframe with all new columns appended.
    """
    df = compute_ema(df, period=9)
    df = compute_ema(df, period=21)
    df = compute_ema(df, period=50)
    df = compute_rsi(df, period=14)
    df = compute_atr(df, period=14)
    df = compute_vwap(df)
    df = compute_bollinger(df, period=20, std_dev=2.0)
    df = compute_volume_delta(df)
    return df
