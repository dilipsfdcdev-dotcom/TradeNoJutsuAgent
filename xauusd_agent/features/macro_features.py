"""
Macro-economic feature engineering for XAUUSD.

Transforms raw macro data (VIX, DXY, Oil, BTC) into directional signals
for gold trading decisions.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce a value to float, returning *default* on failure."""
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _vix_level(vix: float) -> str:
    """Classify VIX into regime buckets."""
    if vix < 15:
        return "LOW"
    elif vix < 25:
        return "NORMAL"
    elif vix < 35:
        return "HIGH"
    else:
        return "EXTREME"


def compute_macro_features(macro_data: dict) -> Dict[str, Any]:
    """Derive macro feature vector from raw market data.

    Parameters
    ----------
    macro_data : dict
        Expected keys (all optional – missing keys degrade gracefully):

        - ``"vix"``          : current VIX level (float)
        - ``"vix_prev"``     : previous VIX level (float)
        - ``"dxy"``          : current DXY value (float)
        - ``"dxy_prev"``     : previous DXY value (float)
        - ``"oil"``          : current crude-oil price (float)
        - ``"oil_prev"``     : previous crude-oil price (float)
        - ``"btc"``          : current BTC price (float)
        - ``"btc_prev"``     : previous BTC price (float)

    Returns
    -------
    dict
        Feature dictionary with macro signals for gold.
    """
    logger.info("Computing macro features from keys: %s", list(macro_data.keys()))

    features: Dict[str, Any] = {}

    # ── VIX ─────────────────────────────────────────────────────────
    vix = _safe_float(macro_data.get("vix"), default=20.0)
    vix_prev = _safe_float(macro_data.get("vix_prev"), default=vix)
    features["vix_level"] = _vix_level(vix)
    vix_change_pct = ((vix - vix_prev) / (abs(vix_prev) + 1e-10)) * 100.0
    features["vix_rising"] = bool(vix_change_pct > 5.0)
    features["vix_value"] = vix
    features["vix_change_pct"] = round(vix_change_pct, 2)

    # ── DXY (USD index) ────────────────────────────────────────────
    # Rising DXY is typically bearish for gold; falling is bullish.
    dxy = _safe_float(macro_data.get("dxy"), default=0.0)
    dxy_prev = _safe_float(macro_data.get("dxy_prev"), default=dxy)
    if dxy > 0 and dxy_prev > 0:
        dxy_change = (dxy - dxy_prev) / (abs(dxy_prev) + 1e-10)
        if dxy_change > 0.001:
            features["dxy_direction"] = 1   # rising → bearish for gold
        elif dxy_change < -0.001:
            features["dxy_direction"] = -1  # falling → bullish for gold
        else:
            features["dxy_direction"] = 0
    else:
        features["dxy_direction"] = 0

    # ── Oil ─────────────────────────────────────────────────────────
    # Oil and gold often move together (inflation proxy).
    oil = _safe_float(macro_data.get("oil"), default=0.0)
    oil_prev = _safe_float(macro_data.get("oil_prev"), default=oil)
    if oil > 0 and oil_prev > 0:
        oil_change = (oil - oil_prev) / (abs(oil_prev) + 1e-10)
        if oil_change > 0.005:
            features["oil_correlation"] = 1   # rising oil → bullish gold signal
        elif oil_change < -0.005:
            features["oil_correlation"] = -1  # falling oil → bearish gold signal
        else:
            features["oil_correlation"] = 0
    else:
        features["oil_correlation"] = 0

    # ── BTC (risk sentiment) ───────────────────────────────────────
    # BTC rallying > 2% = risk-on → may be bearish for safe-haven gold.
    btc = _safe_float(macro_data.get("btc"), default=0.0)
    btc_prev = _safe_float(macro_data.get("btc_prev"), default=btc)
    if btc > 0 and btc_prev > 0:
        btc_change_pct = ((btc - btc_prev) / (abs(btc_prev) + 1e-10)) * 100.0
        features["btc_risk_on"] = bool(btc_change_pct > 2.0)
        features["btc_change_pct"] = round(btc_change_pct, 2)
    else:
        features["btc_risk_on"] = False
        features["btc_change_pct"] = 0.0

    # ── Composite gold bias ────────────────────────────────────────
    # Weighted sum from -1 (bearish) to +1 (bullish).
    #
    # Weights:
    #   VIX rising    → +0.25 (fear → gold bullish)
    #   VIX extreme   → +0.15
    #   DXY direction → -0.25 * dxy_dir  (inverted: rising DXY = bearish gold)
    #   Oil           → +0.15 * oil_cor
    #   BTC risk-on   → -0.10 if risk_on else 0
    #   VIX level     → +0.10 if HIGH/EXTREME else 0

    bias = 0.0

    # VIX component
    if features["vix_rising"]:
        bias += 0.25
    vix_lev = features["vix_level"]
    if vix_lev == "HIGH":
        bias += 0.10
    elif vix_lev == "EXTREME":
        bias += 0.15

    # DXY component (inverted)
    bias -= 0.25 * features["dxy_direction"]

    # Oil component
    bias += 0.15 * features["oil_correlation"]

    # BTC risk-on component
    if features["btc_risk_on"]:
        bias -= 0.10

    # Clamp to [-1, +1]
    features["macro_gold_bias"] = round(max(-1.0, min(1.0, bias)), 4)

    logger.info(
        "Macro features computed: vix=%s dxy_dir=%s oil=%s btc_risk=%s bias=%.4f",
        features["vix_level"],
        features["dxy_direction"],
        features["oil_correlation"],
        features["btc_risk_on"],
        features["macro_gold_bias"],
    )

    return features
