"""
Market regime classifier.

Runs every 30 minutes and categorises the current market state into one of
five regimes that downstream components use to adjust behaviour.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)


class RegimeDetector:
    """Classify the current XAUUSD market regime from multi-timeframe data."""

    REGIMES = [
        "TRENDING_BULL",
        "TRENDING_BEAR",
        "RANGING",
        "HIGH_VOLATILE",
        "LOW_VOLATILE",
    ]

    def detect(self, all_tf: dict[str, Any], htf_bias: dict[str, str]) -> str:
        """Determine the current regime.

        Parameters
        ----------
        all_tf:
            Timeframe data keyed by label (e.g. ``"D1"``, ``"H4"``, ``"M5"``).
            Each entry should carry at least ``atr_14`` and an ``atr_avg_20``
            (20-period ATR average) value.
        htf_bias:
            Higher-timeframe directional bias, e.g.
            ``{"D1": "bull", "H4": "bear"}``.

        Returns
        -------
        One of :attr:`REGIMES`.
        """
        # --- Volatility checks (highest priority) ---
        m5 = all_tf.get("M5", {})
        atr_m5 = m5.get("atr_14", 0.0)
        atr_avg = m5.get("atr_avg_20", atr_m5 or 1.0)  # avoid div-by-zero

        if atr_avg > 0:
            vol_ratio = atr_m5 / atr_avg
        else:
            vol_ratio = 1.0

        if vol_ratio > 2.0:
            regime = "HIGH_VOLATILE"
            logger.info(
                "Regime: HIGH_VOLATILE (ATR ratio %.2f > 2.0)", vol_ratio
            )
            return regime

        if vol_ratio < 0.5:
            regime = "LOW_VOLATILE"
            logger.info(
                "Regime: LOW_VOLATILE (ATR ratio %.2f < 0.5)", vol_ratio
            )
            return regime

        # --- Trend checks ---
        d1_bias = htf_bias.get("D1", "").lower()
        h4_bias = htf_bias.get("H4", "").lower()

        # Need ATR above average for a convincing trend
        trending = vol_ratio >= 1.0

        if d1_bias == "bull" and h4_bias == "bull" and trending:
            regime = "TRENDING_BULL"
            logger.info("Regime: TRENDING_BULL (D1+H4 bull, vol_ratio=%.2f)", vol_ratio)
            return regime

        if d1_bias == "bear" and h4_bias == "bear" and trending:
            regime = "TRENDING_BEAR"
            logger.info("Regime: TRENDING_BEAR (D1+H4 bear, vol_ratio=%.2f)", vol_ratio)
            return regime

        # --- Default: ranging ---
        regime = "RANGING"
        logger.info(
            "Regime: RANGING (D1=%s, H4=%s, vol_ratio=%.2f)",
            d1_bias, h4_bias, vol_ratio,
        )
        return regime
