"""
Position-sizing and SL/TP computation for the XAUUSD trading agent.

All calculations use live broker data (tick value, tick size, volume
constraints) so that lot sizes are always valid for the connected
account.
"""

from __future__ import annotations

from typing import Any

import MetaTrader5 as mt5
import pandas as pd

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default settings (overridable via the *settings* dict)
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "counter_trend_lot_mult": 0.50,
    "high_volatile_lot_mult": 0.75,
    "low_volatile_lot_mult": 1.10,
    "max_lot_cap": 10.0,
    # SL multipliers
    "sl_mult_trend": 1.2,
    "sl_mult_counter": 0.8,
    "sl_mult_volatile": 1.5,
    # TP / SL ratios
    "tp_ratio_trend": 2.2,
    "tp_ratio_counter": 1.5,
    "tp_ratio_volatile": 2.5,
    # Hard SL limits (pips)
    "sl_min_pips": 8.0,
    "sl_max_pips": 35.0,
    # XAUUSD pip size (price movement per pip)
    "pip_size": 0.1,
}


def _s(settings: dict | None, key: str) -> Any:
    """Resolve a setting with fallback to ``_DEFAULTS``."""
    if settings and key in settings:
        return settings[key]
    return _DEFAULTS[key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def round_to_step(value: float, step: float) -> float:
    """Round *value* down to the nearest *step* increment.

    >>> round_to_step(0.137, 0.01)
    0.13
    """
    if step <= 0:
        return value
    return round(round(value / step) * step, 10)


# ---------------------------------------------------------------------------
# Lot calculator
# ---------------------------------------------------------------------------


def calculate_lot(
    live_balance: float,
    risk_pct: float,
    sl_pips: float,
    is_counter_trend: bool,
    current_regime: str,
    symbol: str = "XAUUSD",
    settings: dict | None = None,
) -> tuple[float, float]:
    """Compute the position size and dollar risk for a trade.

    Args:
        live_balance:     Account balance in deposit currency.
        risk_pct:         Risk percentage (e.g. ``1.0`` for 1 %).
        sl_pips:          Distance to stop-loss in pips.
        is_counter_trend: ``True`` for counter-trend entries.
        current_regime:   One of ``"NORMAL"``, ``"HIGH_VOLATILE"``,
                          ``"LOW_VOLATILE"``.
        symbol:           Trading instrument.
        settings:         Optional overrides for default multipliers.

    Returns:
        ``(lot_size, risk_usd)``
    """
    # Step 1 — Real pip value from broker ----------------------------------
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error("symbol_info returned None for %s", symbol)
        return 0.0, 0.0

    pip_size: float = _s(settings, "pip_size")
    pip_value: float = info.trade_tick_value / info.trade_tick_size * pip_size

    # Step 2 — Base lot from risk budget -----------------------------------
    risk_usd: float = live_balance * (risk_pct / 100.0)
    if sl_pips <= 0 or pip_value <= 0:
        logger.error(
            "Invalid sl_pips (%.4f) or pip_value (%.6f)", sl_pips, pip_value
        )
        return 0.0, 0.0

    lot: float = risk_usd / (sl_pips * pip_value)

    # Step 3 — Regime / direction adjustments ------------------------------
    if is_counter_trend:
        lot *= _s(settings, "counter_trend_lot_mult")

    regime = current_regime.upper()
    if regime == "HIGH_VOLATILE":
        lot *= _s(settings, "high_volatile_lot_mult")
    elif regime == "LOW_VOLATILE":
        lot *= _s(settings, "low_volatile_lot_mult")

    # Step 4 — Clamp to broker limits --------------------------------------
    max_lot_cap: float = _s(settings, "max_lot_cap")
    lot = max(info.volume_min, min(lot, min(info.volume_max, max_lot_cap)))
    lot = round_to_step(lot, info.volume_step)

    logger.debug(
        "calculate_lot: balance=%.2f risk_pct=%.2f sl_pips=%.2f → lot=%.2f risk_usd=%.2f",
        live_balance,
        risk_pct,
        sl_pips,
        lot,
        risk_usd,
        extra={
            "lot": lot,
            "risk_usd": risk_usd,
            "is_counter_trend": is_counter_trend,
            "regime": regime,
        },
    )
    return lot, risk_usd


# ---------------------------------------------------------------------------
# SL / TP calculator
# ---------------------------------------------------------------------------


def compute_sl_tp(
    m3: pd.DataFrame,
    m5: pd.DataFrame,
    direction: str,
    is_counter_trend: bool,
    regime: str,
    settings: dict | None = None,
) -> tuple[float, float, float] | None:
    """Derive stop-loss and take-profit distances from ATR.

    Args:
        m3:               3-minute OHLC DataFrame (must contain a ``high``
                          and ``low`` column at minimum).
        m5:               5-minute OHLC DataFrame.
        direction:        ``"BUY"`` or ``"SELL"`` (unused for distance but
                          kept for symmetry with caller).
        is_counter_trend: ``True`` for counter-trend signals.
        regime:           Market regime string.
        settings:         Optional overrides.

    Returns:
        ``(sl_pips, tp_pips, atr_value)`` or ``None`` if the resulting SL
        exceeds the hard maximum (too volatile to trade).
    """
    pip_size: float = _s(settings, "pip_size")

    # ATR blend: average of ATR(7) on M3 and ATR(14) on M5 ----------------
    atr7_m3 = _atr(m3, period=7)
    atr14_m5 = _atr(m5, period=14)

    if atr7_m3 is None or atr14_m5 is None:
        logger.warning("Insufficient data for ATR computation")
        return None

    atr_value: float = (atr7_m3 + atr14_m5) / 2.0
    atr_pips: float = atr_value / pip_size

    # SL multiplier --------------------------------------------------------
    regime_upper = regime.upper()
    if regime_upper == "HIGH_VOLATILE":
        sl_mult = _s(settings, "sl_mult_volatile")
    elif is_counter_trend:
        sl_mult = _s(settings, "sl_mult_counter")
    else:
        sl_mult = _s(settings, "sl_mult_trend")

    sl_pips: float = atr_pips * sl_mult

    # Hard limits ----------------------------------------------------------
    sl_min: float = _s(settings, "sl_min_pips")
    sl_max: float = _s(settings, "sl_max_pips")

    if sl_pips > sl_max:
        logger.warning(
            "SL %.2f pips exceeds hard max %.2f — skipping trade",
            sl_pips,
            sl_max,
        )
        return None

    sl_pips = max(sl_pips, sl_min)

    # TP / SL ratio --------------------------------------------------------
    if regime_upper == "HIGH_VOLATILE":
        tp_ratio = _s(settings, "tp_ratio_volatile")
    elif is_counter_trend:
        tp_ratio = _s(settings, "tp_ratio_counter")
    else:
        tp_ratio = _s(settings, "tp_ratio_trend")

    tp_pips: float = sl_pips * tp_ratio

    logger.debug(
        "compute_sl_tp: atr=%.5f sl=%.2f tp=%.2f regime=%s counter=%s",
        atr_value,
        sl_pips,
        tp_pips,
        regime,
        is_counter_trend,
    )
    return sl_pips, tp_pips, atr_value


# ---------------------------------------------------------------------------
# Internal ATR helper
# ---------------------------------------------------------------------------


def _atr(df: pd.DataFrame, period: int) -> float | None:
    """Compute the latest ATR value for a DataFrame.

    Expects columns ``high``, ``low``, and ``close``.
    """
    if df is None or len(df) < period + 1:
        return None

    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_series = tr.ewm(span=period, adjust=False).mean()
    latest = atr_series.iloc[-1]
    if pd.isna(latest):
        return None
    return float(latest)
