"""Higher timeframe market structure context for the scalper."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd


@dataclass
class MarketContext:
    trend: str                          # "bullish", "bearish", "ranging"
    key_levels: list[float] = field(default_factory=list)
    session: str = ""                   # "asian", "london", "ny", "overlap"
    atr: float = 0.0
    volatility_rank: str = "normal"     # "low", "normal", "high"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_swing_highs(df: pd.DataFrame, lookback: int = 5) -> list[tuple[int, float]]:
    """Return (index, price) for swing highs using *lookback*-candle pivots."""
    swings: list[tuple[int, float]] = []
    for i in range(lookback, len(df) - lookback):
        high_i = df.iloc[i]["high"]
        if all(high_i >= df.iloc[j]["high"] for j in range(i - lookback, i + lookback + 1) if j != i):
            swings.append((i, high_i))
    return swings


def _find_swing_lows(df: pd.DataFrame, lookback: int = 5) -> list[tuple[int, float]]:
    """Return (index, price) for swing lows using *lookback*-candle pivots."""
    swings: list[tuple[int, float]] = []
    for i in range(lookback, len(df) - lookback):
        low_i = df.iloc[i]["low"]
        if all(low_i <= df.iloc[j]["low"] for j in range(i - lookback, i + lookback + 1) if j != i):
            swings.append((i, low_i))
    return swings


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute the current Average True Range over *period* candles."""
    if len(df) < period + 1:
        if len(df) < 2:
            return 0.0
        period = len(df) - 1

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    tr_values: list[float] = []
    for i in range(1, len(df)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_values.append(tr)

    # Simple average of the last *period* true ranges
    recent = tr_values[-period:]
    return sum(recent) / len(recent) if recent else 0.0


def _deduplicate_levels(levels: list[float], threshold_pct: float = 0.1) -> list[float]:
    """Remove levels that are within *threshold_pct*% of each other, keeping the first."""
    if not levels:
        return []
    sorted_levels = sorted(levels)
    deduped = [sorted_levels[0]]
    for lvl in sorted_levels[1:]:
        if deduped[-1] == 0 or abs(lvl - deduped[-1]) / abs(deduped[-1]) * 100 > threshold_pct:
            deduped.append(lvl)
    return deduped


def _round_number_levels(price_range_mid: float, symbol: str) -> list[float]:
    """Generate round-number levels near *price_range_mid* for the given symbol."""
    symbol_upper = symbol.upper()

    # Determine rounding granularity
    if "XAU" in symbol_upper or "GOLD" in symbol_upper:
        step = 10.0
    elif "BTC" in symbol_upper:
        step = 1000.0
    elif "XAG" in symbol_upper or "SILVER" in symbol_upper:
        step = 1.0
    else:
        # Generic: pick a step roughly 0.5% of the price
        step = max(1.0, round(price_range_mid * 0.005, -int(len(str(int(price_range_mid))) - 2)))

    base = round(price_range_mid / step) * step
    return [base - 2 * step, base - step, base, base + step, base + 2 * step]


# ---------------------------------------------------------------------------
# 1. Trend detection
# ---------------------------------------------------------------------------

def detect_trend(df_3m: pd.DataFrame) -> str:
    """Determine trend from swing-point sequence on the 3-minute (or similar) chart.

    - Higher highs + higher lows = "bullish"
    - Lower highs + lower lows  = "bearish"
    - Otherwise                 = "ranging"
    """
    swing_highs = _find_swing_highs(df_3m)
    swing_lows = _find_swing_lows(df_3m)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "ranging"

    # Use the last 3 swing highs / lows (or fewer if not available)
    recent_highs = [p for _, p in swing_highs[-3:]]
    recent_lows = [p for _, p in swing_lows[-3:]]

    higher_highs = all(recent_highs[i] > recent_highs[i - 1] for i in range(1, len(recent_highs)))
    higher_lows = all(recent_lows[i] > recent_lows[i - 1] for i in range(1, len(recent_lows)))

    lower_highs = all(recent_highs[i] < recent_highs[i - 1] for i in range(1, len(recent_highs)))
    lower_lows = all(recent_lows[i] < recent_lows[i - 1] for i in range(1, len(recent_lows)))

    if higher_highs and higher_lows:
        return "bullish"
    if lower_highs and lower_lows:
        return "bearish"
    return "ranging"


# ---------------------------------------------------------------------------
# 2. Key levels
# ---------------------------------------------------------------------------

def find_key_levels(df_15m: pd.DataFrame, symbol: str = "XAUUSD") -> list[float]:
    """Extract key support/resistance levels from 15-minute data.

    Combines recent swing highs/lows with round-number levels, then
    deduplicates levels within 0.1% of each other.
    """
    levels: list[float] = []

    swing_highs = _find_swing_highs(df_15m, lookback=5)
    swing_lows = _find_swing_lows(df_15m, lookback=5)

    for _, price in swing_highs:
        levels.append(round(price, 5))
    for _, price in swing_lows:
        levels.append(round(price, 5))

    # Round-number levels centred on the mid-price of the data
    if len(df_15m) > 0:
        mid = (df_15m["high"].max() + df_15m["low"].min()) / 2
        levels.extend(_round_number_levels(mid, symbol))

    return _deduplicate_levels(levels)


# ---------------------------------------------------------------------------
# 3. Session detection
# ---------------------------------------------------------------------------

def detect_session(now: datetime | None = None) -> str:
    """Return the current trading session based on UTC hour.

    - Asian:   00:00 - 08:00 UTC
    - London:  08:00 - 12:00 UTC
    - Overlap: 12:00 - 17:00 UTC  (London + NY)
    - NY:      17:00 - 22:00 UTC
    - Outside 22:00-00:00 maps to the closest session (Asian).
    """
    if now is None:
        now = datetime.now(timezone.utc)

    hour = now.hour

    if 0 <= hour < 8:
        return "asian"
    if 8 <= hour < 12:
        return "london"
    if 12 <= hour < 17:
        return "overlap"
    if 17 <= hour < 22:
        return "ny"
    # 22-24: closest session is Asian (wraps around midnight)
    return "asian"


# ---------------------------------------------------------------------------
# 4. Full market context
# ---------------------------------------------------------------------------

def get_market_context(
    symbol: str,
    df_3m: pd.DataFrame,
    df_15m: pd.DataFrame | None = None,
) -> MarketContext:
    """Build a complete :class:`MarketContext` snapshot.

    Parameters
    ----------
    symbol:  Trading instrument identifier (e.g. "XAUUSD", "BTCUSD").
    df_3m:   3-minute OHLCV DataFrame used for trend and ATR.
    df_15m:  15-minute OHLCV DataFrame used for key levels (optional).
    """
    trend = detect_trend(df_3m)

    key_levels: list[float] = []
    if df_15m is not None and len(df_15m) > 0:
        key_levels = find_key_levels(df_15m, symbol=symbol)

    session = detect_session()

    # ATR (current, 14-period)
    atr = _compute_atr(df_3m, period=14)

    # Volatility rank: compare current ATR to 50-period ATR average
    volatility_rank = "normal"
    if len(df_3m) >= 51:
        atr_values: list[float] = []
        for end in range(15, len(df_3m) + 1):
            start = max(0, end - 15)
            chunk = df_3m.iloc[start:end]
            atr_values.append(_compute_atr(chunk, period=14))

        if len(atr_values) >= 50:
            avg_atr = sum(atr_values[-50:]) / 50
            if avg_atr > 0:
                ratio = atr / avg_atr
                if ratio < 0.7:
                    volatility_rank = "low"
                elif ratio > 1.3:
                    volatility_rank = "high"

    return MarketContext(
        trend=trend,
        key_levels=key_levels,
        session=session,
        atr=round(atr, 6),
        volatility_rank=volatility_rank,
    )
