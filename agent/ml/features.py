"""Feature engineering pipeline for XGBoost + LSTM."""
import numpy as np
import pandas as pd
import structlog
from datetime import datetime
import math

logger = structlog.get_logger()

def compute_features(mtf_state, candles_1m, candles_3m, candles_15m, candles_1h, account_state, sentiment, **kwargs):
    """
    Returns: {"xgb_features": dict (~50 features), "lstm_sequences": dict (3 numpy arrays)}

    Optional kwargs:
        historical_ctx: HistoricalContext instance for Tier 1 context features
    """
    xgb = {}

    # GROUP 1: Price action features per TF
    for suffix, df in [("_1m", candles_1m), ("_3m", candles_3m), ("_15m", candles_15m), ("_1h", candles_1h)]:
        if df is None or df.empty:
            xgb.update({f"body_pct{suffix}": 0, f"direction{suffix}": 0, f"upper_wick{suffix}": 0,
                        f"lower_wick{suffix}": 0, f"body_vs_atr{suffix}": 0, f"close_pos{suffix}": 0.5})
            continue
        last = df.iloc[-1]
        rng = last["high"] - last["low"]
        body = abs(last["close"] - last["open"])
        xgb[f"body_pct{suffix}"] = body / rng if rng > 0 else 0
        xgb[f"direction{suffix}"] = 1 if last["close"] > last["open"] else -1
        upper_wick = last["high"] - max(last["close"], last["open"])
        lower_wick = min(last["close"], last["open"]) - last["low"]
        xgb[f"upper_wick{suffix}"] = upper_wick / rng if rng > 0 else 0
        xgb[f"lower_wick{suffix}"] = lower_wick / rng if rng > 0 else 0
        atr = last.get("atr", rng) if "atr" in df.columns and pd.notna(last.get("atr")) else rng
        xgb[f"body_vs_atr{suffix}"] = body / atr if atr > 0 else 0
        xgb[f"close_pos{suffix}"] = (last["close"] - last["low"]) / rng if rng > 0 else 0.5

    # GROUP 2: Indicator features from 1M
    for suffix, df in [("_1m", candles_1m), ("_3m", candles_3m)]:
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        xgb[f"rsi{suffix}"] = float(last.get("rsi", 50)) if pd.notna(last.get("rsi")) else 50.0
        rsi = xgb[f"rsi{suffix}"]
        xgb[f"rsi_zone{suffix}"] = -1 if rsi < 30 else (1 if rsi > 70 else 0)
        close = float(last["close"])
        atr = float(last.get("atr", 1)) if "atr" in df.columns and pd.notna(last.get("atr")) else 1.0
        for ema_p in [9, 21, 50]:
            col = f"ema_{ema_p}"
            ema_val = float(last.get(col, close)) if col in df.columns and pd.notna(last.get(col)) else close
            xgb[f"ema_{ema_p}_dist{suffix}"] = (close - ema_val) / atr if atr > 0 else 0

        # EMA stack score
        e9 = last.get("ema_9") if "ema_9" in df.columns else None
        e21 = last.get("ema_21") if "ema_21" in df.columns else None
        e50 = last.get("ema_50") if "ema_50" in df.columns else None
        score = 0
        if e9 is not None and e21 is not None and pd.notna(e9) and pd.notna(e21):
            score += 1 if e9 > e21 else -1
        if e21 is not None and e50 is not None and pd.notna(e21) and pd.notna(e50):
            score += 1 if e21 > e50 else -1
        if e9 is not None and e50 is not None and pd.notna(e9) and pd.notna(e50):
            score += 1 if e9 > e50 else -1
        xgb[f"ema_stack{suffix}"] = score

        # Bollinger position
        bb_u = last.get("bb_upper") if "bb_upper" in df.columns else None
        bb_l = last.get("bb_lower") if "bb_lower" in df.columns else None
        if bb_u is not None and bb_l is not None and pd.notna(bb_u) and pd.notna(bb_l):
            bb_range = bb_u - bb_l
            xgb[f"bb_pos{suffix}"] = (close - bb_l) / bb_range if bb_range > 0 else 0.5
        else:
            xgb[f"bb_pos{suffix}"] = 0.5

        # ATR percentile
        if "atr" in df.columns:
            atr_series = df["atr"].dropna()
            if len(atr_series) > 10:
                xgb[f"atr_pctile{suffix}"] = float((atr_series < atr).sum()) / len(atr_series)
            else:
                xgb[f"atr_pctile{suffix}"] = 0.5

        # Volume ratio
        if "volume" in df.columns:
            vol = float(last["volume"])
            avg_vol = float(df["volume"].tail(20).mean())
            xgb[f"vol_ratio{suffix}"] = vol / avg_vol if avg_vol > 0 else 1.0

    # GROUP 3: Structure features from MTF state
    # Access mtf_state attributes (handle both old and new format)
    h1_bias = getattr(mtf_state, "h1_bias", "neutral")
    if h1_bias == "neutral" and hasattr(mtf_state, "timeframes"):
        h1_bias = mtf_state.timeframes.get("1H", {}).get("trend", "neutral")
    bias_map = {"bullish": 1, "bearish": -1, "neutral": 0}
    xgb["h1_bias"] = bias_map.get(h1_bias, 0)
    xgb["h1_trend_strength"] = getattr(mtf_state, "h1_trend_strength", 0.5)
    xgb["confluence"] = getattr(mtf_state, "confluence_score", 50) / 100.0
    xgb["m3_confirmed"] = 1 if getattr(mtf_state, "m3_confirmed", False) else 0
    m3_mom = getattr(mtf_state, "m3_momentum", "flat")
    xgb["m3_momentum"] = {"bullish": 1, "bearish": -1, "flat": 0}.get(m3_mom, 0)
    xgb["gates_passed"] = 1 if getattr(mtf_state, "gates_passed", False) else 0

    # POI distance
    m15_poi = getattr(mtf_state, "m15_poi", None)
    if m15_poi and candles_1m is not None and not candles_1m.empty:
        price = float(candles_1m.iloc[-1]["close"])
        atr_1m = float(candles_1m.iloc[-1].get("atr", 1)) if "atr" in candles_1m.columns else 1.0
        poi_mid = (m15_poi.high + m15_poi.low) / 2 if hasattr(m15_poi, "high") else 0
        xgb["dist_to_poi"] = abs(price - poi_mid) / atr_1m if atr_1m > 0 else 999
        xgb["has_poi"] = 1
    else:
        xgb["dist_to_poi"] = 999
        xgb["has_poi"] = 0

    # GROUP 4: Pattern features
    m1_patterns = getattr(mtf_state, "m1_active_patterns", [])
    xgb["pat_bull_engulf"] = 1 if any("bullish_engulfing" in str(p) for p in m1_patterns) else 0
    xgb["pat_bear_engulf"] = 1 if any("bearish_engulfing" in str(p) for p in m1_patterns) else 0
    xgb["pat_pin_bar"] = 1 if any("pin" in str(p).lower() or "hammer" in str(p).lower() for p in m1_patterns) else 0
    xgb["num_bull_patterns"] = sum(1 for p in m1_patterns if "bullish" in str(p))
    xgb["num_bear_patterns"] = sum(1 for p in m1_patterns if "bearish" in str(p))

    # GROUP 5: Context features
    from agent.signals.market_structure import detect_session
    session = detect_session()
    session_map = {"asian": 0, "london": 1, "ny": 2, "overlap": 3}
    xgb["session"] = session_map.get(session, 0)
    now = datetime.utcnow()
    xgb["hour_sin"] = math.sin(2 * math.pi * now.hour / 24)
    xgb["hour_cos"] = math.cos(2 * math.pi * now.hour / 24)
    xgb["day_of_week"] = now.weekday()
    xgb["sentiment"] = float(sentiment) if sentiment else 0.0
    xgb["daily_pnl_pct"] = account_state.get("daily_pnl_pct", 0.0)
    xgb["recent_streak"] = account_state.get("recent_win_streak", 0)

    # GROUP 6: Historical context features (Tier 1) — if available
    historical_ctx = kwargs.get("historical_ctx", None)
    if historical_ctx is not None:
        current_atr = xgb.get("body_vs_atr_1m", 0)  # proxy; real ATR from candles
        if candles_1m is not None and not candles_1m.empty and "atr" in candles_1m.columns:
            last_atr = candles_1m.iloc[-1].get("atr")
            if pd.notna(last_atr):
                current_atr = float(last_atr)
        current_vol = 0.0
        if candles_1m is not None and not candles_1m.empty and "volume" in candles_1m.columns:
            current_vol = float(candles_1m.iloc[-1]["volume"])
        current_spread = 0.0
        if candles_1m is not None and not candles_1m.empty and "spread" in candles_1m.columns:
            current_spread = float(candles_1m.iloc[-1].get("spread", 0))
        current_price = 0.0
        if candles_1m is not None and not candles_1m.empty:
            current_price = float(candles_1m.iloc[-1]["close"])

        ctx_feats = compute_context_features(
            historical_ctx=historical_ctx,
            current_atr=current_atr,
            current_vol=current_vol,
            current_spread=current_spread,
            current_price=current_price,
            session=session,
            day_of_week=xgb["day_of_week"],
        )
        xgb.update(ctx_feats)

    # Replace any NaN
    for k, v in xgb.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            xgb[k] = 0.0

    # LSTM sequences
    lstm_seqs = _prepare_sequences(candles_1m, candles_3m, candles_15m)

    return {"xgb_features": xgb, "lstm_sequences": lstm_seqs}


def _prepare_sequences(candles_1m, candles_3m, candles_15m):
    """Prepare normalized OHLCV sequences for LSTM."""
    def extract(df, length):
        if df is None or df.empty:
            return np.zeros((length, 5), dtype=np.float32)
        cols = ["open", "high", "low", "close", "volume"]
        avail = [c for c in cols if c in df.columns]
        if len(avail) < 5:
            return np.zeros((length, 5), dtype=np.float32)
        data = df[cols].tail(length).values.astype(np.float32)
        if len(data) < length:
            pad = np.zeros((length - len(data), 5), dtype=np.float32)
            data = np.vstack([pad, data])
        # Log transform volume
        data[:, 4] = np.log1p(data[:, 4])
        # Normalize per window
        for col_idx in range(5):
            col_data = data[:, col_idx]
            mean = col_data.mean()
            std = col_data.std()
            if std > 0:
                data[:, col_idx] = (col_data - mean) / std
        return data

    return {
        "seq_1m": extract(candles_1m, 30),
        "seq_3m": extract(candles_3m, 20),
        "seq_15m": extract(candles_15m, 10),
    }


def compute_context_features(historical_ctx: "HistoricalContext", current_atr: float,
                              current_vol: float, current_spread: float, current_price: float,
                              session: str, day_of_week: int) -> dict:
    """Compute Tier 1 context features from 5yr historical stats.

    These features normalise live market conditions against long-term
    distributions so the model can gauge whether *now* is unusual.

    Args:
        historical_ctx: Pre-computed HistoricalContext for the symbol.
        current_atr: Current ATR value (e.g. 14-bar ATR on the 15-min chart).
        current_vol: Current bar volume.
        current_spread: Current spread.
        current_price: Current close price.
        session: Current session name (asian/london/overlap/ny).
        day_of_week: 0=Mon ... 4=Fri.

    Returns:
        dict of ctx_* feature values.
    """
    ctx = historical_ctx
    feats: dict[str, float] = {}

    # --- ATR vs session average ---
    sess_atr_mean = ctx.atr_mean_by_session.get(session, 0.0)
    sess_atr_std = ctx.atr_std_by_session.get(session, 1.0)
    feats["ctx_atr_vs_session_avg"] = (
        (current_atr - sess_atr_mean) / sess_atr_std if sess_atr_std > 0 else 0.0
    )

    # --- ATR percentile within 5yr distribution ---
    pctiles = ctx.atr_percentiles  # [p10, p25, p50, p75, p90]
    if pctiles and pctiles[-1] > 0:
        if current_atr <= pctiles[0]:
            feats["ctx_atr_percentile_5yr"] = 0.10
        elif current_atr >= pctiles[-1]:
            feats["ctx_atr_percentile_5yr"] = 0.90
        else:
            # Linear interpolation between known percentiles
            anchors = [10, 25, 50, 75, 90]
            for i in range(len(pctiles) - 1):
                if pctiles[i] <= current_atr <= pctiles[i + 1]:
                    span = pctiles[i + 1] - pctiles[i]
                    frac = (current_atr - pctiles[i]) / span if span > 0 else 0.0
                    feats["ctx_atr_percentile_5yr"] = (
                        anchors[i] + frac * (anchors[i + 1] - anchors[i])
                    ) / 100.0
                    break
            else:
                feats["ctx_atr_percentile_5yr"] = 0.50
    else:
        feats["ctx_atr_percentile_5yr"] = 0.50

    # --- Volume vs session average ---
    sess_vol_mean = ctx.vol_mean_by_session.get(session, 0.0)
    feats["ctx_vol_vs_session_avg"] = (
        current_vol / sess_vol_mean if sess_vol_mean > 0 else 1.0
    )

    # --- Volume percentile within 5yr distribution ---
    vol_pctiles = ctx.vol_percentiles
    if vol_pctiles and vol_pctiles[-1] > 0:
        if current_vol <= vol_pctiles[0]:
            feats["ctx_vol_percentile_5yr"] = 0.10
        elif current_vol >= vol_pctiles[-1]:
            feats["ctx_vol_percentile_5yr"] = 0.90
        else:
            anchors = [10, 25, 50, 75, 90]
            for i in range(len(vol_pctiles) - 1):
                if vol_pctiles[i] <= current_vol <= vol_pctiles[i + 1]:
                    span = vol_pctiles[i + 1] - vol_pctiles[i]
                    frac = (current_vol - vol_pctiles[i]) / span if span > 0 else 0.0
                    feats["ctx_vol_percentile_5yr"] = (
                        anchors[i] + frac * (anchors[i + 1] - anchors[i])
                    ) / 100.0
                    break
            else:
                feats["ctx_vol_percentile_5yr"] = 0.50
    else:
        feats["ctx_vol_percentile_5yr"] = 0.50

    # --- Spread vs historical ---
    feats["ctx_spread_vs_avg"] = (
        current_spread / ctx.spread_mean if ctx.spread_mean > 0 else 1.0
    )

    # --- Price position within 5yr range ---
    price_range = ctx.price_high - ctx.price_low
    feats["ctx_price_position_5yr"] = (
        (current_price - ctx.price_low) / price_range if price_range > 0 else 0.5
    )

    # --- Session win-rate bias ---
    long_wr = ctx.session_long_wr.get(session, 0.5)
    short_wr = ctx.session_short_wr.get(session, 0.5)
    feats["ctx_session_long_wr"] = long_wr
    feats["ctx_session_short_wr"] = short_wr
    feats["ctx_session_wr_edge"] = long_wr - short_wr

    # --- Day-of-week average range ratio ---
    dow_range = ctx.dow_avg_range.get(day_of_week, 0.0)
    avg_dow_range = np.mean(list(ctx.dow_avg_range.values())) if ctx.dow_avg_range else 0.0
    feats["ctx_dow_range_ratio"] = (
        dow_range / avg_dow_range if avg_dow_range > 0 else 1.0
    )

    return feats


def compute_label(trade_data: dict) -> dict:
    """Generate training labels from a closed trade."""
    pnl = trade_data.get("pnl", 0)
    rr = trade_data.get("rr_actual", 0)
    quality = trade_data.get("trade_quality", 5)
    return {
        "direction_correct": 1 if pnl > 0 else 0,
        "quality_score": (quality or 5) / 10.0,
        "rr_achieved": float(rr) if rr else 0.0,
        "optimal_entry": 1 if pnl > 0 and abs(rr or 0) >= 1.5 else 0,
    }
