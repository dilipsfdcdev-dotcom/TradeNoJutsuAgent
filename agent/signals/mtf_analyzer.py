"""Multi-timeframe (MTF) analyzer -- computes alignment across 1H/15M/3M/1M.

Provides the ``MTFAnalyzer`` class that produces an ``MTFState`` on every
candle update, and the ``compute_features`` helper that extracts feature
vectors for the XGBoost and LSTM models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MTFState:
    """Snapshot of multi-timeframe alignment for a single symbol."""

    symbol: str
    gates_passed: bool = False
    gate_details: str = ""
    confluence_score: float = 0.0
    setup_narrative: str = ""
    timeframes: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MTF Analyzer
# ---------------------------------------------------------------------------

class MTFAnalyzer:
    """Stateless multi-timeframe gate checker.

    On each call to ``update()`` it computes per-timeframe structure
    (trend, EMA position, RSI, ATR) and checks alignment gates.
    """

    def update(
        self,
        symbol: str,
        candles_1m: pd.DataFrame,
        candles_3m: pd.DataFrame,
        candles_15m: pd.DataFrame,
        candles_1h: pd.DataFrame,
        tick: dict | None,
    ) -> MTFState:
        """Compute MTF state from the latest candle data.

        Parameters
        ----------
        symbol : str
            Trading instrument.
        candles_1m, candles_3m, candles_15m, candles_1h : pd.DataFrame
            OHLCV DataFrames with indicators already computed.
        tick : dict | None
            Latest tick data (bid/ask/spread).

        Returns
        -------
        MTFState
            The computed multi-timeframe state including gate results.
        """
        state = MTFState(symbol=symbol)

        # Build per-timeframe summaries
        state.timeframes = {
            "1H": self._summarise_tf(candles_1h),
            "15M": self._summarise_tf(candles_15m),
            "3M": self._summarise_tf(candles_3m),
            "1M": self._summarise_tf(candles_1m),
        }

        # Gate check: 1H and 15M must agree on trend direction
        h1_trend = state.timeframes["1H"].get("trend", "neutral")
        m15_trend = state.timeframes["15M"].get("trend", "neutral")
        m3_trend = state.timeframes["3M"].get("trend", "neutral")

        gate_reasons: list[str] = []

        # Gate 1: Higher-timeframe alignment (1H + 15M must not conflict)
        if h1_trend == "neutral" and m15_trend == "neutral":
            gate_reasons.append("No clear trend on 1H or 15M")
        elif h1_trend != "neutral" and m15_trend != "neutral" and h1_trend != m15_trend:
            gate_reasons.append(f"1H ({h1_trend}) vs 15M ({m15_trend}) conflict")

        # Gate 2: RSI extremes on 1H (overbought/oversold = wait)
        h1_rsi = state.timeframes["1H"].get("rsi")
        if h1_rsi is not None and (h1_rsi > 80 or h1_rsi < 20):
            gate_reasons.append(f"1H RSI extreme ({h1_rsi:.1f})")

        # Gate 3: 3M must show some structure (not completely flat)
        m3_atr = state.timeframes["3M"].get("atr")
        if m3_atr is not None and m3_atr <= 0:
            gate_reasons.append("3M ATR is zero -- no volatility")

        state.gates_passed = len(gate_reasons) == 0
        state.gate_details = "; ".join(gate_reasons) if gate_reasons else "All gates passed"

        # Confluence score (0-100)
        state.confluence_score = self._compute_confluence(state.timeframes)

        # Setup narrative
        direction = h1_trend if h1_trend != "neutral" else m15_trend
        state.setup_narrative = (
            f"{direction.upper() if direction != 'neutral' else 'FLAT'} bias from 1H, "
            f"15M structure is {m15_trend}, "
            f"3M confirmation is {m3_trend}. "
            f"Confluence: {state.confluence_score:.0f}/100."
        )

        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarise_tf(df: pd.DataFrame) -> dict[str, Any]:
        """Extract a compact summary dict from a candle DataFrame."""
        if df is None or df.empty:
            return {}

        last = df.iloc[-1]

        # Trend from EMA alignment
        ema_9 = last.get("ema_9") if "ema_9" in df.columns else None
        ema_21 = last.get("ema_21") if "ema_21" in df.columns else None
        ema_50 = last.get("ema_50") if "ema_50" in df.columns else None
        close = last.get("close", 0)

        trend = "neutral"
        if ema_9 is not None and ema_21 is not None:
            if pd.notna(ema_9) and pd.notna(ema_21):
                if ema_9 > ema_21:
                    trend = "bullish"
                elif ema_9 < ema_21:
                    trend = "bearish"

        # EMA position: where price sits relative to EMAs
        ema_position = "at_ema"
        if ema_21 is not None and pd.notna(ema_21) and close:
            diff_pct = (close - ema_21) / ema_21 * 100 if ema_21 else 0
            if diff_pct > 0.1:
                ema_position = "above"
            elif diff_pct < -0.1:
                ema_position = "below"

        rsi = float(last["rsi"]) if "rsi" in df.columns and pd.notna(last.get("rsi")) else None
        atr = float(last["atr"]) if "atr" in df.columns and pd.notna(last.get("atr")) else None

        # Simple structure label
        structure = "neutral"
        if rsi is not None:
            if rsi > 60 and trend == "bullish":
                structure = "strong_bull"
            elif rsi < 40 and trend == "bearish":
                structure = "strong_bear"
            elif 40 <= rsi <= 60:
                structure = "ranging"

        return {
            "trend": trend,
            "ema_position": ema_position,
            "rsi": round(rsi, 2) if rsi is not None else None,
            "atr": round(atr, 6) if atr is not None else None,
            "structure": structure,
            "close": float(close) if pd.notna(close) else None,
        }

    @staticmethod
    def _compute_confluence(timeframes: dict[str, dict]) -> float:
        """Compute a 0-100 confluence score based on timeframe alignment."""
        score = 50.0  # neutral baseline

        trends = [timeframes[tf].get("trend", "neutral") for tf in ("1H", "15M", "3M", "1M")]

        # All same direction: +30
        non_neutral = [t for t in trends if t != "neutral"]
        if non_neutral and all(t == non_neutral[0] for t in non_neutral):
            score += 30.0
        elif non_neutral:
            # Check for conflicts
            dirs = set(non_neutral)
            if len(dirs) > 1:
                score -= 20.0

        # 1H trend clarity bonus
        h1_rsi = timeframes.get("1H", {}).get("rsi")
        if h1_rsi is not None:
            if 55 <= h1_rsi <= 70 or 30 <= h1_rsi <= 45:
                score += 10.0  # good trending zone

        # 3M structure confirmation
        m3_struct = timeframes.get("3M", {}).get("structure", "neutral")
        if m3_struct in ("strong_bull", "strong_bear"):
            score += 10.0

        return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# Feature computation for ML models
# ---------------------------------------------------------------------------

def compute_features(
    mtf_state: MTFState,
    candles_1m: pd.DataFrame,
    candles_3m: pd.DataFrame,
    candles_15m: pd.DataFrame,
    candles_1h: pd.DataFrame,
    account: dict,
    sentiment_score: float,
) -> dict:
    """Compute feature vectors for XGBoost and LSTM models.

    Returns
    -------
    dict
        ``{"xgb_features": dict, "lstm_sequences": dict}``
    """
    xgb_features = _compute_xgb_features(
        mtf_state, candles_1m, candles_3m, candles_15m, candles_1h,
        account, sentiment_score,
    )
    lstm_sequences = _compute_lstm_sequences(candles_1m, candles_3m, candles_15m)

    return {
        "xgb_features": xgb_features,
        "lstm_sequences": lstm_sequences,
    }


def _compute_xgb_features(
    mtf_state: MTFState,
    candles_1m: pd.DataFrame,
    candles_3m: pd.DataFrame,
    candles_15m: pd.DataFrame,
    candles_1h: pd.DataFrame,
    account: dict,
    sentiment_score: float,
) -> dict:
    """Build a flat feature dict for XGBoost."""

    def _safe_last(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
        if df is None or df.empty or col not in df.columns:
            return default
        val = df.iloc[-1].get(col)
        return float(val) if pd.notna(val) else default

    features: dict[str, float] = {}

    # Per-timeframe features
    for label, df in [("1m", candles_1m), ("3m", candles_3m),
                      ("15m", candles_15m), ("1h", candles_1h)]:
        features[f"rsi_{label}"] = _safe_last(df, "rsi", 50.0)
        features[f"atr_{label}"] = _safe_last(df, "atr")
        features[f"ema9_{label}"] = _safe_last(df, "ema_9")
        features[f"ema21_{label}"] = _safe_last(df, "ema_21")
        features[f"close_{label}"] = _safe_last(df, "close")
        features[f"volume_{label}"] = _safe_last(df, "volume")

    # MTF alignment features
    tf = mtf_state.timeframes
    trend_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
    for tf_label in ("1H", "15M", "3M", "1M"):
        tf_data = tf.get(tf_label, {})
        features[f"trend_{tf_label}"] = trend_map.get(tf_data.get("trend", "neutral"), 0.0)

    features["confluence_score"] = mtf_state.confluence_score
    features["sentiment_score"] = sentiment_score

    # Account features
    balance = account.get("balance", 0.0)
    equity = account.get("equity", 0.0)
    features["equity_balance_ratio"] = equity / balance if balance > 0 else 1.0

    return features


def _compute_lstm_sequences(
    candles_1m: pd.DataFrame,
    candles_3m: pd.DataFrame,
    candles_15m: pd.DataFrame,
) -> dict:
    """Build normalised OHLCV sequences for the LSTM model.

    Returns dict with keys ``seq_1m`` (30, 5), ``seq_3m`` (20, 5),
    ``seq_15m`` (10, 5) as numpy arrays.
    """

    def _extract_seq(df: pd.DataFrame, length: int) -> np.ndarray:
        cols = ["open", "high", "low", "close", "volume"]
        if df is None or df.empty:
            return np.zeros((length, 5), dtype=np.float32)

        available = [c for c in cols if c in df.columns]
        if len(available) < 5:
            return np.zeros((length, 5), dtype=np.float32)

        data = df[cols].tail(length).values.astype(np.float32)

        # Pad if not enough rows
        if len(data) < length:
            pad = np.zeros((length - len(data), 5), dtype=np.float32)
            data = np.vstack([pad, data])

        # Normalise: divide by the last close to get relative values
        last_close = data[-1, 3]  # close column
        if last_close > 0:
            data[:, :4] = data[:, :4] / last_close  # normalise OHLC
            vol_max = data[:, 4].max()
            if vol_max > 0:
                data[:, 4] = data[:, 4] / vol_max  # normalise volume

        return data

    return {
        "seq_1m": _extract_seq(candles_1m, 30),
        "seq_3m": _extract_seq(candles_3m, 20),
        "seq_15m": _extract_seq(candles_15m, 10),
    }
