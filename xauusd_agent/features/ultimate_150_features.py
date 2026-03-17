"""
Ultimate 140+ feature engineering from OHLCV data for XAUUSD.

Uses TA-Lib (C-based) for core indicator calculations, with manual
implementations for indicators not covered by the library.

All functions expect a DataFrame with columns:
    open, high, low, close, tick_volume
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

_EPS = 1e-10


# ── helpers ────────────────────────────────────────────────────────────────

def _col(prefix: str, name: str) -> str:
    """Build a column name with an optional timeframe prefix."""
    return f"{prefix}_{name}" if prefix else name


def _linear_regression_slope(series: pd.Series, period: int) -> pd.Series:
    """Rolling OLS slope over *period* bars."""
    out = np.full(len(series), np.nan)
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    vals = series.values.astype(float)
    for i in range(period - 1, len(vals)):
        y = vals[i - period + 1 : i + 1]
        if np.isnan(y).any():
            continue
        y_mean = y.mean()
        out[i] = ((x - x_mean) * (y - y_mean)).sum() / (x_var + _EPS)
    return pd.Series(out, index=series.index)


def _supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """Supertrend indicator.  Returns (supertrend_line, direction ±1)."""
    atr = pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
    # Fill leading NaN ATR with a simple high-low range fallback
    atr = atr.fillna((high - low).rolling(period, min_periods=1).mean())
    hl2 = (high + low) / 2.0

    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    direction = pd.Series(np.ones(len(close)), index=close.index)  # 1 = up

    for i in range(1, len(close)):
        # Upper band
        if upper_basic.iat[i] < upper_band.iat[i - 1] or close.iat[i - 1] > upper_band.iat[i - 1]:
            upper_band.iat[i] = upper_basic.iat[i]
        else:
            upper_band.iat[i] = upper_band.iat[i - 1]

        # Lower band
        if lower_basic.iat[i] > lower_band.iat[i - 1] or close.iat[i - 1] < lower_band.iat[i - 1]:
            lower_band.iat[i] = lower_basic.iat[i]
        else:
            lower_band.iat[i] = lower_band.iat[i - 1]

        # Direction
        prev_dir = direction.iat[i - 1]
        if prev_dir == 1:  # was up
            if close.iat[i] < lower_band.iat[i]:
                direction.iat[i] = -1
            else:
                direction.iat[i] = 1
        else:  # was down
            if close.iat[i] > upper_band.iat[i]:
                direction.iat[i] = 1
            else:
                direction.iat[i] = -1

    supertrend = pd.Series(np.where(direction == 1, lower_band, upper_band), index=close.index)
    return supertrend, direction


def _ichimoku(
    high: pd.Series,
    low: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Ichimoku cloud components (no displacement)."""
    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2.0
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2.0
    senkou_a = (tenkan + kijun) / 2.0
    senkou_b = (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2.0
    return tenkan, kijun, senkou_a, senkou_b


# ── main function ──────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Compute 140+ technical features from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, tick_volume.
    prefix : str, optional
        Timeframe prefix prepended to every column name (e.g. ``"M15"``).

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned to the input index, NaN-handled.
    """
    logger.debug("Computing features (prefix=%s, rows=%d)", prefix, len(df))

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)  # noqa: E741
    c = df["close"].astype(float)
    v = df["tick_volume"].astype(float)

    feat: dict[str, pd.Series] = {}

    def _c(name: str) -> str:
        return _col(prefix, name)

    # ── TREND ──────────────────────────────────────────────────────────

    for p in (9, 21, 50, 100, 200):
        feat[_c(f"ema_{p}")] = pd.Series(talib.EMA(c.values, timeperiod=p), index=c.index)

    for p in (20, 50):
        feat[_c(f"sma_{p}")] = pd.Series(talib.SMA(c.values, timeperiod=p), index=c.index)

    # MACD
    macd_line, macd_signal, macd_hist = talib.MACD(c.values, fastperiod=12, slowperiod=26, signalperiod=9)
    feat[_c("macd_line")] = pd.Series(macd_line, index=c.index)
    feat[_c("macd_signal")] = pd.Series(macd_signal, index=c.index)
    feat[_c("macd_hist")] = pd.Series(macd_hist, index=c.index)

    # ADX
    feat[_c("adx_14")] = pd.Series(talib.ADX(h.values, l.values, c.values, timeperiod=14), index=c.index)
    feat[_c("plus_di_14")] = pd.Series(talib.PLUS_DI(h.values, l.values, c.values, timeperiod=14), index=c.index)
    feat[_c("minus_di_14")] = pd.Series(talib.MINUS_DI(h.values, l.values, c.values, timeperiod=14), index=c.index)

    # Supertrend
    st_line, st_dir = _supertrend(h, l, c, period=10, multiplier=3.0)
    feat[_c("supertrend")] = st_line
    feat[_c("supertrend_dir")] = st_dir

    # Ichimoku
    tenkan, kijun, senkou_a, senkou_b = _ichimoku(h, l)
    feat[_c("ichi_tenkan")] = tenkan
    feat[_c("ichi_kijun")] = kijun
    feat[_c("ichi_senkou_a")] = senkou_a
    feat[_c("ichi_senkou_b")] = senkou_b

    # Linear regression slope
    feat[_c("linreg_slope_20")] = _linear_regression_slope(c, 20)

    # ── MOMENTUM ───────────────────────────────────────────────────────

    for p in (7, 14, 21):
        feat[_c(f"rsi_{p}")] = pd.Series(talib.RSI(c.values, timeperiod=p), index=c.index)

    # Stochastic
    slowk, slowd = talib.STOCH(
        h.values, l.values, c.values,
        fastk_period=14, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0,
    )
    feat[_c("stoch_k")] = pd.Series(slowk, index=c.index)
    feat[_c("stoch_d")] = pd.Series(slowd, index=c.index)

    # CCI
    feat[_c("cci_20")] = pd.Series(talib.CCI(h.values, l.values, c.values, timeperiod=20), index=c.index)

    # Williams %R
    feat[_c("willr_14")] = pd.Series(talib.WILLR(h.values, l.values, c.values, timeperiod=14), index=c.index)

    # MFI (requires volume)
    if v.sum() > 0:
        feat[_c("mfi_14")] = pd.Series(talib.MFI(h.values, l.values, c.values, v.values, timeperiod=14), index=c.index)
    else:
        feat[_c("mfi_14")] = pd.Series(np.nan, index=c.index)

    # ROC
    for p in (9, 14):
        feat[_c(f"roc_{p}")] = pd.Series(talib.ROC(c.values, timeperiod=p), index=c.index)

    # ── VOLATILITY ─────────────────────────────────────────────────────

    for p in (7, 14, 20):
        feat[_c(f"atr_{p}")] = pd.Series(talib.ATR(h.values, l.values, c.values, timeperiod=p), index=c.index)

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = talib.BBANDS(c.values, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    bb_upper = pd.Series(bb_upper, index=c.index)
    bb_mid = pd.Series(bb_mid, index=c.index)
    bb_lower = pd.Series(bb_lower, index=c.index)
    feat[_c("bb_upper")] = bb_upper
    feat[_c("bb_mid")] = bb_mid
    feat[_c("bb_lower")] = bb_lower
    feat[_c("bb_bandwidth")] = (bb_upper - bb_lower) / (bb_mid + _EPS)
    feat[_c("bb_pct_b")] = (c - bb_lower) / (bb_upper - bb_lower + _EPS)

    # Keltner Channel (EMA20 ± 2*ATR20)
    kc_mid = pd.Series(talib.EMA(c.values, timeperiod=20), index=c.index)
    atr_20 = pd.Series(talib.ATR(h.values, l.values, c.values, timeperiod=20), index=c.index)
    feat[_c("kc_upper")] = kc_mid + 2.0 * atr_20
    feat[_c("kc_mid")] = kc_mid
    feat[_c("kc_lower")] = kc_mid - 2.0 * atr_20

    # ATR ratio (volatility momentum)
    atr_7 = feat[_c("atr_7")]
    atr_20_s = feat[_c("atr_20")]
    feat[_c("atr_ratio_7_20")] = atr_7 / (atr_20_s + _EPS)

    # ── VOLUME ─────────────────────────────────────────────────────────

    # OBV
    feat[_c("obv")] = pd.Series(talib.OBV(c.values, v.values), index=c.index)

    # VWAP (intraday approximation: cumulative typical_price*vol / cumvol)
    typical = (h + l + c) / 3.0
    cum_tpv = (typical * v).cumsum()
    cum_vol = v.cumsum()
    feat[_c("vwap")] = cum_tpv / (cum_vol + _EPS)

    # Volume SMA ratio
    vol_sma_20 = pd.Series(talib.SMA(v.values, timeperiod=20), index=c.index)
    feat[_c("vol_sma_ratio")] = v / (vol_sma_20 + _EPS)

    # Volume spike (bool)
    vol_sma_10 = pd.Series(talib.SMA(v.values, timeperiod=10), index=c.index)
    feat[_c("vol_spike")] = (v > 1.4 * vol_sma_10).astype(float)

    # ── PRICE ACTION ───────────────────────────────────────────────────

    body = (c - o).abs()
    candle_range = h - l + _EPS

    feat[_c("candle_body_ratio")] = body / candle_range

    upper_shadow = h - pd.concat([c, o], axis=1).max(axis=1)
    lower_shadow = pd.concat([c, o], axis=1).min(axis=1) - l
    feat[_c("upper_shadow_ratio")] = upper_shadow / candle_range
    feat[_c("lower_shadow_ratio")] = lower_shadow / candle_range

    # Higher highs / higher lows (3-bar)
    hh = (h > h.shift(1)) & (h.shift(1) > h.shift(2))
    hl = (l > l.shift(1)) & (l.shift(1) > l.shift(2))
    lh = (h < h.shift(1)) & (h.shift(1) < h.shift(2))
    ll = (l < l.shift(1)) & (l.shift(1) < l.shift(2))
    feat[_c("higher_highs")] = hh.astype(float)
    feat[_c("higher_lows")] = hl.astype(float)
    feat[_c("lower_highs")] = lh.astype(float)
    feat[_c("lower_lows")] = ll.astype(float)

    # Distance from recent swing high/low (20-bar)
    swing_high_20 = h.rolling(20).max()
    swing_low_20 = l.rolling(20).min()
    atr_14 = feat[_c("atr_14")]
    feat[_c("dist_swing_high_20")] = (swing_high_20 - c) / (atr_14 + _EPS)
    feat[_c("dist_swing_low_20")] = (c - swing_low_20) / (atr_14 + _EPS)

    # ── CROSS-FEATURE ──────────────────────────────────────────────────

    # Price vs EMA distance normalised by ATR(14)
    for p in (9, 21, 50, 100, 200):
        ema_col = feat[_c(f"ema_{p}")]
        feat[_c(f"price_ema{p}_dist")] = (c - ema_col) / (atr_14 + _EPS)

    # RSI divergence: price makes new 5-bar high but RSI does not
    price_hi_5 = c.rolling(5).max()
    rsi_14 = feat[_c("rsi_14")]
    rsi_hi_5 = rsi_14.rolling(5).max()
    feat[_c("rsi_bearish_div")] = ((c >= price_hi_5) & (rsi_14 < rsi_hi_5)).astype(float)

    price_lo_5 = c.rolling(5).min()
    rsi_lo_5 = rsi_14.rolling(5).min()
    feat[_c("rsi_bullish_div")] = ((c <= price_lo_5) & (rsi_14 > rsi_lo_5)).astype(float)

    # MACD-price divergence
    macd_line_s = feat[_c("macd_line")]
    macd_hi_5 = macd_line_s.rolling(5).max()
    macd_lo_5 = macd_line_s.rolling(5).min()
    feat[_c("macd_bearish_div")] = ((c >= price_hi_5) & (macd_line_s < macd_hi_5)).astype(float)
    feat[_c("macd_bullish_div")] = ((c <= price_lo_5) & (macd_line_s > macd_lo_5)).astype(float)

    # ── Additional cross / derived features to reach 140+ ──────────────

    # EMA crosses (binary)
    feat[_c("ema9_above_ema21")] = (feat[_c("ema_9")] > feat[_c("ema_21")]).astype(float)
    feat[_c("ema21_above_ema50")] = (feat[_c("ema_21")] > feat[_c("ema_50")]).astype(float)
    feat[_c("ema50_above_ema200")] = (feat[_c("ema_50")] > feat[_c("ema_200")]).astype(float)

    # Price relative to Ichimoku cloud
    feat[_c("price_above_cloud")] = (c > pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)).astype(float)
    feat[_c("price_below_cloud")] = (c < pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)).astype(float)
    feat[_c("cloud_thickness")] = (senkou_a - senkou_b).abs() / (atr_14 + _EPS)

    # Price relative to BB / KC
    feat[_c("price_above_bb_upper")] = (c > bb_upper).astype(float)
    feat[_c("price_below_bb_lower")] = (c < bb_lower).astype(float)
    feat[_c("squeeze")] = (
        (bb_upper < feat[_c("kc_upper")]) & (bb_lower > feat[_c("kc_lower")])
    ).astype(float)

    # RSI zones
    feat[_c("rsi_overbought")] = (rsi_14 > 70).astype(float)
    feat[_c("rsi_oversold")] = (rsi_14 < 30).astype(float)

    # Stochastic cross
    feat[_c("stoch_k_above_d")] = (feat[_c("stoch_k")] > feat[_c("stoch_d")]).astype(float)

    # ADX strength
    feat[_c("adx_strong")] = (feat[_c("adx_14")] > 25).astype(float)
    feat[_c("di_diff")] = feat[_c("plus_di_14")] - feat[_c("minus_di_14")]

    # Supertrend agreement with EMA
    feat[_c("supertrend_ema_agree")] = (
        (st_dir > 0) & (feat[_c("ema_9")] > feat[_c("ema_21")])
    ).astype(float)

    # VWAP distance
    feat[_c("price_vwap_dist")] = (c - feat[_c("vwap")]) / (atr_14 + _EPS)

    # Candle direction
    feat[_c("candle_bullish")] = (c > o).astype(float)

    # Momentum acceleration (RSI rate of change)
    feat[_c("rsi_14_roc")] = rsi_14.diff(3)

    # MACD histogram momentum
    macd_h = feat[_c("macd_hist")]
    feat[_c("macd_hist_rising")] = (macd_h > macd_h.shift(1)).astype(float)
    feat[_c("macd_hist_roc")] = macd_h.diff(3)

    # CCI zones
    cci = feat[_c("cci_20")]
    feat[_c("cci_overbought")] = (cci > 100).astype(float)
    feat[_c("cci_oversold")] = (cci < -100).astype(float)

    # Williams %R zones
    wr = feat[_c("willr_14")]
    feat[_c("willr_overbought")] = (wr > -20).astype(float)
    feat[_c("willr_oversold")] = (wr < -80).astype(float)

    # OBV slope
    obv = feat[_c("obv")]
    feat[_c("obv_slope_10")] = obv.diff(10) / (obv.rolling(10).std() + _EPS)

    # Return features
    for p in (1, 3, 5, 10):
        feat[_c(f"return_{p}")] = c.pct_change(p)

    # Volatility of returns
    feat[_c("return_std_10")] = c.pct_change().rolling(10).std()
    feat[_c("return_std_20")] = c.pct_change().rolling(20).std()

    # High-low range normalised
    feat[_c("range_atr_ratio")] = candle_range / (atr_14 + _EPS)

    # Gap (open vs previous close)
    feat[_c("gap")] = (o - c.shift(1)) / (atr_14 + _EPS)

    # Close position in daily range
    feat[_c("close_position")] = (c - l) / (candle_range)

    # Rolling skew and kurtosis of returns (20-bar)
    rets = c.pct_change()
    feat[_c("return_skew_20")] = rets.rolling(20).skew()
    feat[_c("return_kurt_20")] = rets.rolling(20).kurt()

    # Heikin-Ashi close (smoothed)
    ha_close = (o + h + l + c) / 4.0
    feat[_c("ha_close")] = ha_close
    feat[_c("ha_bullish")] = (ha_close > ha_close.shift(1)).astype(float)

    # Price vs SMA distance
    for p in (20, 50):
        sma_col = feat[_c(f"sma_{p}")]
        feat[_c(f"price_sma{p}_dist")] = (c - sma_col) / (atr_14 + _EPS)

    # EMA slopes (normalised)
    for p in (9, 21, 50):
        ema_col = feat[_c(f"ema_{p}")]
        feat[_c(f"ema{p}_slope_5")] = ema_col.diff(5) / (atr_14 + _EPS)

    # Tenkan-Kijun cross
    feat[_c("tenkan_above_kijun")] = (tenkan > kijun).astype(float)

    # ── Additional derived features ────────────────────────────────────

    # Donchian Channel (20-bar)
    dc_upper = h.rolling(20).max()
    dc_lower = l.rolling(20).min()
    dc_mid = (dc_upper + dc_lower) / 2.0
    feat[_c("dc_upper")] = dc_upper
    feat[_c("dc_lower")] = dc_lower
    feat[_c("dc_mid")] = dc_mid
    feat[_c("dc_width")] = (dc_upper - dc_lower) / (atr_14 + _EPS)
    feat[_c("price_dc_position")] = (c - dc_lower) / (dc_upper - dc_lower + _EPS)

    # TRIX (triple EMA oscillator)
    feat[_c("trix_15")] = pd.Series(talib.TRIX(c.values, timeperiod=15), index=c.index)

    # Aroon
    aroon_dn, aroon_up = talib.AROON(h.values, l.values, timeperiod=14)
    feat[_c("aroon_up")] = pd.Series(aroon_up, index=c.index)
    feat[_c("aroon_dn")] = pd.Series(aroon_dn, index=c.index)
    feat[_c("aroon_osc")] = pd.Series(aroon_up - aroon_dn, index=c.index)

    # Ultimate Oscillator
    feat[_c("ultosc")] = pd.Series(
        talib.ULTOSC(h.values, l.values, c.values, timeperiod1=7, timeperiod2=14, timeperiod3=28),
        index=c.index,
    )

    # Parabolic SAR direction
    sar = pd.Series(talib.SAR(h.values, l.values, acceleration=0.02, maximum=0.2), index=c.index)
    feat[_c("sar")] = sar
    feat[_c("price_above_sar")] = (c > sar).astype(float)

    # Chaikin Money Flow (approximation via AD)
    feat[_c("ad")] = pd.Series(talib.AD(h.values, l.values, c.values, v.values), index=c.index)

    # Normalised ATR (ATR / close)
    feat[_c("natr_14")] = pd.Series(talib.NATR(h.values, l.values, c.values, timeperiod=14), index=c.index)

    # Consecutive bullish / bearish candles
    bullish = (c > o).astype(float)
    bearish = (c < o).astype(float)
    # Count consecutive via groupby trick
    bull_groups = bullish.ne(bullish.shift()).cumsum()
    bear_groups = bearish.ne(bearish.shift()).cumsum()
    feat[_c("consec_bull")] = bullish.groupby(bull_groups).cumsum()
    feat[_c("consec_bear")] = bearish.groupby(bear_groups).cumsum()

    # Body size in ATR
    feat[_c("body_atr_ratio")] = body / (atr_14 + _EPS)

    # Wicks relative to ATR
    feat[_c("upper_wick_atr")] = upper_shadow / (atr_14 + _EPS)
    feat[_c("lower_wick_atr")] = lower_shadow / (atr_14 + _EPS)

    # Inside bar (high < prev high AND low > prev low)
    feat[_c("inside_bar")] = ((h < h.shift(1)) & (l > l.shift(1))).astype(float)

    # Outside bar (high > prev high AND low < prev low)
    feat[_c("outside_bar")] = ((h > h.shift(1)) & (l < l.shift(1))).astype(float)

    # Doji (body < 10% of range)
    feat[_c("doji")] = (body < 0.1 * (h - l + _EPS)).astype(float)

    # Pin bar detection (long lower wick bullish, long upper wick bearish)
    feat[_c("pin_bar_bull")] = ((lower_shadow > 2.0 * body) & (upper_shadow < body)).astype(float)
    feat[_c("pin_bar_bear")] = ((upper_shadow > 2.0 * body) & (lower_shadow < body)).astype(float)

    # Engulfing patterns
    prev_body = (c.shift(1) - o.shift(1)).abs()
    feat[_c("bull_engulf")] = (
        (c > o) & (c.shift(1) < o.shift(1)) &
        (body > prev_body) & (o <= c.shift(1)) & (c >= o.shift(1))
    ).astype(float)
    feat[_c("bear_engulf")] = (
        (c < o) & (c.shift(1) > o.shift(1)) &
        (body > prev_body) & (o >= c.shift(1)) & (c <= o.shift(1))
    ).astype(float)

    # Volume-price divergence: price up but volume declining
    feat[_c("vol_price_div_bull")] = (
        (c > c.shift(3)) & (v < v.shift(3))
    ).astype(float)
    feat[_c("vol_price_div_bear")] = (
        (c < c.shift(3)) & (v < v.shift(3))
    ).astype(float)

    # EMA ribbon spread (max EMA - min EMA) / ATR
    ema_vals = pd.DataFrame({
        p: feat[_c(f"ema_{p}")] for p in (9, 21, 50, 100, 200)
    })
    feat[_c("ema_ribbon_spread")] = (ema_vals.max(axis=1) - ema_vals.min(axis=1)) / (atr_14 + _EPS)

    # EMA alignment score: +1 for each pair in correct bull order, -1 for bear
    ema_order = [9, 21, 50, 100, 200]
    alignment = pd.Series(0.0, index=c.index)
    for idx_i in range(len(ema_order) - 1):
        faster = feat[_c(f"ema_{ema_order[idx_i]}")]
        slower = feat[_c(f"ema_{ema_order[idx_i + 1]}")]
        alignment = alignment + np.where(faster > slower, 1.0, -1.0)
    feat[_c("ema_alignment")] = alignment / len(ema_order)

    # RSI slope (5-bar)
    feat[_c("rsi_7_slope")] = feat[_c("rsi_7")].diff(5)
    feat[_c("rsi_21_slope")] = feat[_c("rsi_21")].diff(5)

    # MFI zones
    mfi = feat[_c("mfi_14")]
    feat[_c("mfi_overbought")] = (mfi > 80).astype(float)
    feat[_c("mfi_oversold")] = (mfi < 20).astype(float)

    # Stochastic zones
    feat[_c("stoch_overbought")] = (feat[_c("stoch_k")] > 80).astype(float)
    feat[_c("stoch_oversold")] = (feat[_c("stoch_k")] < 20).astype(float)

    # ── Assemble ───────────────────────────────────────────────────────

    result = pd.DataFrame(feat, index=df.index)

    logger.info(
        "Features computed: %d columns (prefix=%s)",
        result.shape[1],
        prefix or "<none>",
    )

    # NaN handling: forward-fill then backfill remaining leading NaNs
    result = result.ffill().bfill()

    return result
