"""Ultimate 150+ feature engineering using the `ta` Python library.

Ports all features from the legacy TA-Lib-based ``ultimate_150_features.py``
to the pure-Python ``ta`` library, supplemented by NumPy / Pandas for manual
calculations (candle patterns, Heikin-Ashi, linear regression, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from tradenojutsu.infra.logger import get_logger

logger = get_logger("analysis.features_150")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise a/b replacing inf/nan with 0."""
    return a.div(b.replace(0, np.nan)).fillna(0.0)


def _linreg_slope(series: pd.Series, window: int = 14) -> pd.Series:
    """Rolling linear-regression slope (OLS) over *window* bars."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(vals):
        if len(vals) < window:
            return np.nan
        y = vals
        return np.dot(y - y.mean(), x - x_mean) / x_var

    return series.rolling(window).apply(_slope, raw=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Compute 140+ features from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ``open``, ``high``, ``low``, ``close``, ``volume``.
    prefix : str, optional
        String prepended to every output column name (useful for multi-
        timeframe stacking).

    Returns
    -------
    pd.DataFrame
        Copy of *df* enriched with feature columns.
    """
    if len(df) < 50:
        logger.warning("DataFrame has fewer than 50 rows – indicators may be unreliable")

    out = df.copy()
    o, h, l, c, v = out["open"], out["high"], out["low"], out["close"], out["volume"]

    # ------------------------------------------------------------------ #
    #  1. MOVING AVERAGES                                                 #
    # ------------------------------------------------------------------ #
    for w in (9, 21, 50, 100, 200):
        out[f"ema_{w}"] = ta.trend.ema_indicator(c, window=w)
    for w in (20, 50):
        out[f"sma_{w}"] = ta.trend.sma_indicator(c, window=w)

    # Price distance to key EMAs (normalised by close)
    for w in (9, 21, 50, 100, 200):
        out[f"close_vs_ema_{w}"] = _safe_div(c - out[f"ema_{w}"], c) * 100

    # ------------------------------------------------------------------ #
    #  2. MACD                                                            #
    # ------------------------------------------------------------------ #
    macd_ind = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd_ind.macd()
    out["macd_signal"] = macd_ind.macd_signal()
    out["macd_hist"] = macd_ind.macd_diff()
    out["macd_hist_diff"] = out["macd_hist"].diff()

    # ------------------------------------------------------------------ #
    #  3. ADX / DMI                                                       #
    # ------------------------------------------------------------------ #
    adx_ind = ta.trend.ADXIndicator(h, l, c, window=14)
    out["adx"] = adx_ind.adx()
    out["adx_pos"] = adx_ind.adx_pos()
    out["adx_neg"] = adx_ind.adx_neg()
    out["adx_diff"] = out["adx_pos"] - out["adx_neg"]

    # ------------------------------------------------------------------ #
    #  4. RSI (multiple periods)                                          #
    # ------------------------------------------------------------------ #
    for w in (7, 14, 21):
        out[f"rsi_{w}"] = ta.momentum.rsi(c, window=w)
    out["rsi_diff_7_14"] = out["rsi_7"] - out["rsi_14"]

    # ------------------------------------------------------------------ #
    #  5. STOCHASTIC OSCILLATOR                                           #
    # ------------------------------------------------------------------ #
    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()
    out["stoch_diff"] = out["stoch_k"] - out["stoch_d"]

    # ------------------------------------------------------------------ #
    #  6. CCI                                                             #
    # ------------------------------------------------------------------ #
    out["cci"] = ta.trend.cci(h, l, c, window=20)

    # ------------------------------------------------------------------ #
    #  7. WILLIAMS %R                                                     #
    # ------------------------------------------------------------------ #
    out["williams_r"] = ta.momentum.williams_r(h, l, c, lbp=14)

    # ------------------------------------------------------------------ #
    #  8. MFI                                                             #
    # ------------------------------------------------------------------ #
    out["mfi"] = ta.volume.money_flow_index(h, l, c, v, window=14)

    # ------------------------------------------------------------------ #
    #  9. ROC (Rate of Change)                                            #
    # ------------------------------------------------------------------ #
    out["roc_10"] = ta.momentum.roc(c, window=10)
    out["roc_20"] = ta.momentum.roc(c, window=20)

    # ------------------------------------------------------------------ #
    # 10. ATR (multiple periods)                                          #
    # ------------------------------------------------------------------ #
    for w in (7, 14, 20):
        out[f"atr_{w}"] = ta.volatility.average_true_range(h, l, c, window=w)
    out["atr_pct"] = _safe_div(out["atr_14"], c) * 100
    out["atr_ratio_7_14"] = _safe_div(out["atr_7"], out["atr_14"])

    # ------------------------------------------------------------------ #
    # 11. BOLLINGER BANDS                                                 #
    # ------------------------------------------------------------------ #
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_middle"] = bb.bollinger_mavg()
    out["bb_lower"] = bb.bollinger_lband()
    bb_range = out["bb_upper"] - out["bb_lower"]
    out["bb_width"] = _safe_div(bb_range, out["bb_middle"])
    out["bb_pct_b"] = _safe_div(c - out["bb_lower"], bb_range)

    # ------------------------------------------------------------------ #
    # 12. KELTNER CHANNEL                                                 #
    # ------------------------------------------------------------------ #
    kc = ta.volatility.KeltnerChannel(h, l, c, window=20, window_atr=10)
    out["kc_upper"] = kc.keltner_channel_hband()
    out["kc_middle"] = kc.keltner_channel_mband()
    out["kc_lower"] = kc.keltner_channel_lband()
    out["kc_width"] = _safe_div(out["kc_upper"] - out["kc_lower"], out["kc_middle"])

    # ------------------------------------------------------------------ #
    # 13. SUPERTREND (manual – ATR-based)                                 #
    # ------------------------------------------------------------------ #
    _atr_st = out["atr_14"].copy()
    hl2 = (h + l) / 2
    _upper_band = hl2 + 3 * _atr_st
    _lower_band = hl2 - 3 * _atr_st
    supertrend = pd.Series(np.nan, index=out.index)
    st_dir = pd.Series(1, index=out.index)  # 1 = bullish

    for i in range(1, len(out)):
        # Upper band
        if _upper_band.iloc[i] < _upper_band.iloc[i - 1] or c.iloc[i - 1] > _upper_band.iloc[i - 1]:
            pass  # keep current
        else:
            _upper_band.iloc[i] = _upper_band.iloc[i - 1]
        # Lower band
        if _lower_band.iloc[i] > _lower_band.iloc[i - 1] or c.iloc[i - 1] < _lower_band.iloc[i - 1]:
            pass
        else:
            _lower_band.iloc[i] = _lower_band.iloc[i - 1]
        # Direction
        if supertrend.iloc[i - 1] == _upper_band.iloc[i - 1]:
            if c.iloc[i] <= _upper_band.iloc[i]:
                supertrend.iloc[i] = _upper_band.iloc[i]
                st_dir.iloc[i] = -1
            else:
                supertrend.iloc[i] = _lower_band.iloc[i]
                st_dir.iloc[i] = 1
        else:
            if c.iloc[i] >= _lower_band.iloc[i]:
                supertrend.iloc[i] = _lower_band.iloc[i]
                st_dir.iloc[i] = 1
            else:
                supertrend.iloc[i] = _upper_band.iloc[i]
                st_dir.iloc[i] = -1

    out["supertrend"] = supertrend
    out["supertrend_dir"] = st_dir
    out["supertrend_dist"] = _safe_div(c - supertrend, c) * 100

    # ------------------------------------------------------------------ #
    # 14. ICHIMOKU                                                        #
    # ------------------------------------------------------------------ #
    ichi = ta.trend.IchimokuIndicator(h, l, window1=9, window2=26, window3=52)
    out["ichi_conv"] = ichi.ichimoku_conversion_line()
    out["ichi_base"] = ichi.ichimoku_base_line()
    out["ichi_a"] = ichi.ichimoku_a()
    out["ichi_b"] = ichi.ichimoku_b()
    out["ichi_conv_base_diff"] = _safe_div(out["ichi_conv"] - out["ichi_base"], c) * 100
    out["ichi_cloud_top"] = out[["ichi_a", "ichi_b"]].max(axis=1)
    out["ichi_cloud_bot"] = out[["ichi_a", "ichi_b"]].min(axis=1)
    out["ichi_cloud_width"] = _safe_div(out["ichi_cloud_top"] - out["ichi_cloud_bot"], c) * 100
    out["close_vs_cloud"] = np.where(
        c > out["ichi_cloud_top"], 1,
        np.where(c < out["ichi_cloud_bot"], -1, 0),
    )

    # ------------------------------------------------------------------ #
    # 15. LINEAR REGRESSION SLOPE                                         #
    # ------------------------------------------------------------------ #
    out["linreg_slope_14"] = _linreg_slope(c, window=14)
    out["linreg_slope_28"] = _linreg_slope(c, window=28)

    # ------------------------------------------------------------------ #
    # 16. OBV & VOLUME                                                    #
    # ------------------------------------------------------------------ #
    out["obv"] = ta.volume.on_balance_volume(c, v)
    out["obv_ema_21"] = ta.trend.ema_indicator(out["obv"], window=21)
    out["obv_diff"] = out["obv"] - out["obv_ema_21"]

    # VWAP approximation (cumulative within session is typical; here we use
    # a rolling proxy since we lack intraday session boundaries).
    typical_price = (h + l + c) / 3
    cum_tp_vol = (typical_price * v).cumsum()
    cum_vol = v.cumsum()
    out["vwap_approx"] = _safe_div(cum_tp_vol, cum_vol)
    out["close_vs_vwap"] = _safe_div(c - out["vwap_approx"], c) * 100

    # Volume ratios
    vol_sma_20 = v.rolling(20).mean()
    vol_sma_50 = v.rolling(50).mean()
    out["volume_ratio_20"] = _safe_div(v, vol_sma_20)
    out["volume_ratio_50"] = _safe_div(v, vol_sma_50)
    out["volume_sma_ratio"] = _safe_div(vol_sma_20, vol_sma_50)
    out["volume_change"] = v.pct_change()

    # ------------------------------------------------------------------ #
    # 17. DONCHIAN CHANNEL                                                #
    # ------------------------------------------------------------------ #
    dc = ta.volatility.DonchianChannel(h, l, c, window=20)
    out["dc_upper"] = dc.donchian_channel_hband()
    out["dc_lower"] = dc.donchian_channel_lband()
    out["dc_middle"] = dc.donchian_channel_mband()
    out["dc_width"] = _safe_div(out["dc_upper"] - out["dc_lower"], out["dc_middle"])
    out["dc_pct"] = _safe_div(c - out["dc_lower"], (out["dc_upper"] - out["dc_lower"]))

    # ------------------------------------------------------------------ #
    # 18. AROON                                                           #
    # ------------------------------------------------------------------ #
    aroon = ta.trend.AroonIndicator(h, l, window=25)
    out["aroon_up"] = aroon.aroon_indicator()  # aroon_up - aroon_down
    # Manually compute aroon_up and aroon_down for full feature set
    out["aroon_up_raw"] = h.rolling(26).apply(lambda x: x.argmax() / 25 * 100, raw=True)
    out["aroon_down_raw"] = l.rolling(26).apply(lambda x: x.argmin() / 25 * 100, raw=True)
    out["aroon_osc"] = out["aroon_up_raw"] - out["aroon_down_raw"]

    # ------------------------------------------------------------------ #
    # 19. PARABOLIC SAR                                                   #
    # ------------------------------------------------------------------ #
    psar = ta.trend.PSARIndicator(h, l, c, step=0.02, max_step=0.2)
    out["psar"] = psar.psar()
    out["psar_up"] = psar.psar_up()
    out["psar_down"] = psar.psar_down()
    out["psar_dir"] = np.where(c > out["psar"].fillna(c), 1, -1)
    out["psar_dist"] = _safe_div(c - out["psar"].fillna(c), c) * 100

    # ------------------------------------------------------------------ #
    # 20. RETURNS & STATISTICAL MOMENTS                                   #
    # ------------------------------------------------------------------ #
    out["return_1"] = c.pct_change(1)
    out["return_5"] = c.pct_change(5)
    out["return_10"] = c.pct_change(10)
    out["return_20"] = c.pct_change(20)
    out["log_return_1"] = np.log(c / c.shift(1))

    out["rolling_std_20"] = out["return_1"].rolling(20).std()
    out["rolling_skew_20"] = out["return_1"].rolling(20).skew()
    out["rolling_kurt_20"] = out["return_1"].rolling(20).kurt()

    # ------------------------------------------------------------------ #
    # 21. PRICE ACTION FEATURES                                           #
    # ------------------------------------------------------------------ #
    # Defragment the DataFrame to avoid PerformanceWarning from many inserts
    out = out.copy()
    body = c - o
    abs_body = body.abs()
    candle_range = h - l
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l

    out["body_pct"] = _safe_div(body, candle_range)
    out["upper_shadow_pct"] = _safe_div(upper_shadow, candle_range)
    out["lower_shadow_pct"] = _safe_div(lower_shadow, candle_range)
    out["candle_range_pct"] = _safe_div(candle_range, c) * 100

    avg_body = abs_body.rolling(20).mean()

    # Doji: tiny body relative to range
    out["is_doji"] = (abs_body < candle_range * 0.1).astype(int)

    # Pin bar: long wick one side, small body
    out["is_pin_bar_bull"] = (
        (lower_shadow > 2 * abs_body) & (upper_shadow < abs_body)
    ).astype(int)
    out["is_pin_bar_bear"] = (
        (upper_shadow > 2 * abs_body) & (lower_shadow < abs_body)
    ).astype(int)

    # Engulfing
    prev_body = body.shift(1)
    out["is_bullish_engulfing"] = (
        (body > 0) & (prev_body < 0) & (abs_body > prev_body.abs())
    ).astype(int)
    out["is_bearish_engulfing"] = (
        (body < 0) & (prev_body > 0) & (abs_body > prev_body.abs())
    ).astype(int)

    # Inside bar: current range entirely within previous range
    out["is_inside_bar"] = (
        (h < h.shift(1)) & (l > l.shift(1))
    ).astype(int)

    # Outside bar: current range engulfs previous range
    out["is_outside_bar"] = (
        (h > h.shift(1)) & (l < l.shift(1))
    ).astype(int)

    # Hammer (bullish) / Shooting star (bearish)
    out["is_hammer"] = (
        (lower_shadow > 2 * abs_body) &
        (upper_shadow < 0.3 * candle_range) &
        (body > 0)
    ).astype(int)
    out["is_shooting_star"] = (
        (upper_shadow > 2 * abs_body) &
        (lower_shadow < 0.3 * candle_range) &
        (body < 0)
    ).astype(int)

    # Morning / Evening star (3-bar patterns)
    is_small = abs_body < avg_body * 0.5
    out["is_morning_star"] = (
        (body.shift(2) < 0) &
        is_small.shift(1) &
        (body > 0) &
        (c > (o.shift(2) + c.shift(2)) / 2)
    ).astype(int)
    out["is_evening_star"] = (
        (body.shift(2) > 0) &
        is_small.shift(1) &
        (body < 0) &
        (c < (o.shift(2) + c.shift(2)) / 2)
    ).astype(int)

    # Big candle (body > 2x average)
    out["is_big_candle"] = (abs_body > 2 * avg_body).astype(int)

    # ------------------------------------------------------------------ #
    # 22. HEIKIN-ASHI                                                     #
    # ------------------------------------------------------------------ #
    ha_close = (o + h + l + c) / 4
    ha_open = pd.Series(np.nan, index=out.index, dtype=float)
    ha_open.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2
    for i in range(1, len(out)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2
    ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([l, ha_open, ha_close], axis=1).min(axis=1)
    ha_body = ha_close - ha_open

    out["ha_body_pct"] = _safe_div(ha_body, (ha_high - ha_low))
    out["ha_direction"] = np.sign(ha_body).fillna(0).astype(int)
    # Consecutive HA bars in same direction
    _dir = out["ha_direction"]
    _group = (_dir != _dir.shift(1)).cumsum()
    out["ha_streak"] = _dir.groupby(_group).cumsum()

    # ------------------------------------------------------------------ #
    # 23. CROSS-FEATURES: EMA CROSSES & ALIGNMENT                        #
    # ------------------------------------------------------------------ #
    out["ema_cross_9_21"] = np.sign(out["ema_9"] - out["ema_21"]).fillna(0).astype(int)
    out["ema_cross_21_50"] = np.sign(out["ema_21"] - out["ema_50"]).fillna(0).astype(int)
    out["ema_cross_50_200"] = np.sign(out["ema_50"] - out["ema_200"]).fillna(0).astype(int)

    # Full alignment score: +4 if perfectly bullish, -4 if perfectly bearish
    ema_cols = [out[f"ema_{w}"] for w in (9, 21, 50, 100, 200)]
    alignment = sum(
        np.sign(ema_cols[i] - ema_cols[i + 1]) for i in range(len(ema_cols) - 1)
    )
    out["ema_alignment"] = alignment.fillna(0).astype(int)

    # ------------------------------------------------------------------ #
    # 24. SQUEEZE (BB inside KC)                                          #
    # ------------------------------------------------------------------ #
    out["squeeze_on"] = (
        (out["bb_upper"] < out["kc_upper"]) & (out["bb_lower"] > out["kc_lower"])
    ).astype(int)
    out["squeeze_off"] = 1 - out["squeeze_on"]

    # ------------------------------------------------------------------ #
    # 25. DIVERGENCES (price vs RSI / MACD hist)                          #
    # ------------------------------------------------------------------ #
    _rsi = out["rsi_14"]
    _close_ch = c.diff(5)
    _rsi_ch = _rsi.diff(5)
    out["rsi_bull_div"] = ((_close_ch < 0) & (_rsi_ch > 0)).astype(int)
    out["rsi_bear_div"] = ((_close_ch > 0) & (_rsi_ch < 0)).astype(int)

    _macd_ch = out["macd_hist"].diff(5)
    out["macd_bull_div"] = ((_close_ch < 0) & (_macd_ch > 0)).astype(int)
    out["macd_bear_div"] = ((_close_ch > 0) & (_macd_ch < 0)).astype(int)

    # ------------------------------------------------------------------ #
    # 26. PIVOT POINTS                                                    #
    # ------------------------------------------------------------------ #
    pivot = (h.shift(1) + l.shift(1) + c.shift(1)) / 3
    out["pivot"] = pivot
    out["pivot_r1"] = 2 * pivot - l.shift(1)
    out["pivot_s1"] = 2 * pivot - h.shift(1)
    out["close_vs_pivot"] = _safe_div(c - pivot, c) * 100

    # ------------------------------------------------------------------ #
    # 27. HIGH/LOW DISTANCE                                               #
    # ------------------------------------------------------------------ #
    out["high_20"] = h.rolling(20).max()
    out["low_20"] = l.rolling(20).min()
    out["dist_high_20"] = _safe_div(c - out["high_20"], c) * 100
    out["dist_low_20"] = _safe_div(c - out["low_20"], c) * 100

    # ------------------------------------------------------------------ #
    # 28. MISC MOMENTUM                                                   #
    # ------------------------------------------------------------------ #
    # TSI (True Strength Index)
    out["tsi"] = ta.momentum.tsi(c, window_slow=25, window_fast=13)

    # Ultimate Oscillator
    out["ult_osc"] = ta.momentum.ultimate_oscillator(h, l, c)

    # Awesome Oscillator
    out["ao"] = ta.momentum.awesome_oscillator(h, l)

    # Kama
    out["kama"] = ta.momentum.kama(c, window=10)
    out["close_vs_kama"] = _safe_div(c - out["kama"], c) * 100

    # ------------------------------------------------------------------ #
    # 29. VOLATILITY EXTRAS                                               #
    # ------------------------------------------------------------------ #
    # Ulcer Index
    out["ulcer_index"] = ta.volatility.ulcer_index(c, window=14)

    # ------------------------------------------------------------------ #
    # 30. GAP & RANGE                                                     #
    # ------------------------------------------------------------------ #
    out["gap_pct"] = _safe_div(o - c.shift(1), c.shift(1)) * 100
    out["true_range"] = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # ------------------------------------------------------------------ #
    # 31. ADDITIONAL CROSS / DERIVED FEATURES                             #
    # ------------------------------------------------------------------ #
    # EMA cross signals (1 on cross-up bar, -1 on cross-down bar, 0 else)
    out["ema_9_21_xup"] = ((out["ema_9"] > out["ema_21"]) &
                           (out["ema_9"].shift(1) <= out["ema_21"].shift(1))).astype(int)
    out["ema_9_21_xdn"] = ((out["ema_9"] < out["ema_21"]) &
                           (out["ema_9"].shift(1) >= out["ema_21"].shift(1))).astype(int)

    # RSI zones (overbought / oversold / neutral encoded)
    out["rsi_14_zone"] = np.where(out["rsi_14"] > 70, 1,
                          np.where(out["rsi_14"] < 30, -1, 0))

    # ADX strength category
    out["adx_strong"] = (out["adx"] > 25).astype(int)

    # Stochastic zones
    out["stoch_zone"] = np.where(out["stoch_k"] > 80, 1,
                         np.where(out["stoch_k"] < 20, -1, 0))

    # Momentum composite: normalized sum of RSI + Stoch + Williams
    out["momentum_composite"] = (
        (out["rsi_14"] - 50) / 50 +
        (out["stoch_k"] - 50) / 50 +
        (out["williams_r"] + 50) / 50
    ) / 3

    # Volatility ratio: current ATR vs rolling mean ATR
    atr_mean_50 = out["atr_14"].rolling(50).mean()
    out["atr_expansion"] = _safe_div(out["atr_14"], atr_mean_50)

    # Close position within daily range
    out["close_position"] = _safe_div(c - l, h - l)

    # ------------------------------------------------------------------ #
    # Apply prefix & final NaN handling                                   #
    # ------------------------------------------------------------------ #
    # Identify only the new feature columns (not original OHLCV)
    original_cols = set(df.columns)
    feature_cols = [col for col in out.columns if col not in original_cols]

    if prefix:
        rename_map = {col: f"{prefix}{col}" for col in feature_cols}
        out.rename(columns=rename_map, inplace=True)
        feature_cols = list(rename_map.values())

    # Forward-fill then back-fill residual NaN in feature columns
    out[feature_cols] = out[feature_cols].ffill().bfill()

    n_features = len(feature_cols)
    logger.info("Computed %d features (prefix=%r) on %d bars", n_features, prefix, len(out))

    return out
