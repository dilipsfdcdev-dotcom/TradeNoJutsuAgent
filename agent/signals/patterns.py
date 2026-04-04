"""Candlestick and price action pattern detection."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class PatternSignal:
    type: str          # e.g. "bullish_engulfing", "bearish_fvg"
    strength: float    # 0-1 confidence
    price_level: float  # Where the pattern formed
    candle_index: int   # Index in the dataframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _body_size(row: pd.Series) -> float:
    return abs(row["close"] - row["open"])


def _is_bullish(row: pd.Series) -> bool:
    return row["close"] > row["open"]


def _is_bearish(row: pd.Series) -> bool:
    return row["close"] < row["open"]


def _upper_wick(row: pd.Series) -> float:
    return row["high"] - max(row["open"], row["close"])


def _lower_wick(row: pd.Series) -> float:
    return min(row["open"], row["close"]) - row["low"]


def _find_swing_highs(df: pd.DataFrame, lookback: int = 5) -> list[tuple[int, float]]:
    """Return list of (index, price) for swing highs over *lookback* candles."""
    swings: list[tuple[int, float]] = []
    for i in range(lookback, len(df) - lookback):
        high_i = df.iloc[i]["high"]
        if all(high_i >= df.iloc[j]["high"] for j in range(i - lookback, i + lookback + 1) if j != i):
            swings.append((i, high_i))
    return swings


def _find_swing_lows(df: pd.DataFrame, lookback: int = 5) -> list[tuple[int, float]]:
    """Return list of (index, price) for swing lows over *lookback* candles."""
    swings: list[tuple[int, float]] = []
    for i in range(lookback, len(df) - lookback):
        low_i = df.iloc[i]["low"]
        if all(low_i <= df.iloc[j]["low"] for j in range(i - lookback, i + lookback + 1) if j != i):
            swings.append((i, low_i))
    return swings


# ---------------------------------------------------------------------------
# 1. Engulfing
# ---------------------------------------------------------------------------

def detect_engulfing(df: pd.DataFrame) -> list[PatternSignal]:
    """Detect bullish and bearish engulfing patterns."""
    signals: list[PatternSignal] = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        prev_body = _body_size(prev)
        curr_body = _body_size(curr)

        if prev_body == 0 or curr_body == 0:
            continue

        # Bullish engulfing: prior bearish candle engulfed by current bullish candle
        if _is_bearish(prev) and _is_bullish(curr):
            if curr["open"] <= prev["close"] and curr["close"] >= prev["open"]:
                strength = min(curr_body / prev_body / 3.0, 1.0)
                signals.append(PatternSignal(
                    type="bullish_engulfing",
                    strength=round(strength, 4),
                    price_level=curr["close"],
                    candle_index=i,
                ))

        # Bearish engulfing: prior bullish candle engulfed by current bearish candle
        if _is_bullish(prev) and _is_bearish(curr):
            if curr["open"] >= prev["close"] and curr["close"] <= prev["open"]:
                strength = min(curr_body / prev_body / 3.0, 1.0)
                signals.append(PatternSignal(
                    type="bearish_engulfing",
                    strength=round(strength, 4),
                    price_level=curr["close"],
                    candle_index=i,
                ))

    return signals


# ---------------------------------------------------------------------------
# 2. Pin bars (hammer / shooting star)
# ---------------------------------------------------------------------------

def detect_pin_bars(df: pd.DataFrame) -> list[PatternSignal]:
    """Detect hammer and shooting star pin-bar patterns."""
    signals: list[PatternSignal] = []
    for i in range(len(df)):
        row = df.iloc[i]
        body = _body_size(row)
        if body == 0:
            continue

        upper = _upper_wick(row)
        lower = _lower_wick(row)
        total_range = row["high"] - row["low"]
        if total_range == 0:
            continue

        # Hammer: small body near top, long lower wick >= 2x body
        if lower >= 2.0 * body and upper <= body:
            ratio = lower / body
            strength = min(ratio / 5.0, 1.0)
            signals.append(PatternSignal(
                type="bullish_pin_bar",
                strength=round(strength, 4),
                price_level=row["close"],
                candle_index=i,
            ))

        # Shooting star: small body near bottom, long upper wick >= 2x body
        if upper >= 2.0 * body and lower <= body:
            ratio = upper / body
            strength = min(ratio / 5.0, 1.0)
            signals.append(PatternSignal(
                type="bearish_pin_bar",
                strength=round(strength, 4),
                price_level=row["close"],
                candle_index=i,
            ))

    return signals


# ---------------------------------------------------------------------------
# 3. Fair Value Gap (FVG)
# ---------------------------------------------------------------------------

def detect_fvg(df: pd.DataFrame) -> list[PatternSignal]:
    """Detect bullish and bearish fair value gaps."""
    signals: list[PatternSignal] = []
    for i in range(2, len(df)):
        c0 = df.iloc[i - 2]  # candle[i-2]
        c1 = df.iloc[i - 1]  # candle[i-1] (middle)
        c2 = df.iloc[i]      # candle[i]

        # Bullish FVG: gap up — candle[i].low > candle[i-2].high
        if c2["low"] > c0["high"]:
            gap_size = c2["low"] - c0["high"]
            mid_cover = max(0, c1["high"] - c0["high"])  # how much the middle candle covers
            remaining_gap = gap_size - mid_cover
            if remaining_gap > 0:
                strength = min(remaining_gap / c1["high"] * 100, 1.0)
                signals.append(PatternSignal(
                    type="bullish_fvg",
                    strength=round(strength, 4),
                    price_level=(c0["high"] + c2["low"]) / 2,
                    candle_index=i,
                ))

        # Bearish FVG: gap down — candle[i].high < candle[i-2].low
        if c2["high"] < c0["low"]:
            gap_size = c0["low"] - c2["high"]
            mid_cover = max(0, c0["low"] - c1["low"])
            remaining_gap = gap_size - mid_cover
            if remaining_gap > 0:
                strength = min(remaining_gap / c1["low"] * 100, 1.0)
                signals.append(PatternSignal(
                    type="bearish_fvg",
                    strength=round(strength, 4),
                    price_level=(c0["low"] + c2["high"]) / 2,
                    candle_index=i,
                ))

    return signals


# ---------------------------------------------------------------------------
# 4. Break of Structure (BOS)
# ---------------------------------------------------------------------------

def detect_bos(df: pd.DataFrame) -> list[PatternSignal]:
    """Detect break of structure — price closing beyond a recent swing point."""
    signals: list[PatternSignal] = []
    swing_highs = _find_swing_highs(df)
    swing_lows = _find_swing_lows(df)

    for i in range(len(df)):
        row = df.iloc[i]

        # Bullish BOS: close above the most recent swing high that preceded this candle
        recent_sh = [s for s in swing_highs if s[0] < i]
        if recent_sh:
            last_sh_idx, last_sh_price = recent_sh[-1]
            if row["close"] > last_sh_price and i - last_sh_idx <= 20:
                distance = row["close"] - last_sh_price
                strength = min(distance / last_sh_price * 100, 1.0)
                signals.append(PatternSignal(
                    type="bullish_bos",
                    strength=round(strength, 4),
                    price_level=last_sh_price,
                    candle_index=i,
                ))

        # Bearish BOS: close below the most recent swing low
        recent_sl = [s for s in swing_lows if s[0] < i]
        if recent_sl:
            last_sl_idx, last_sl_price = recent_sl[-1]
            if row["close"] < last_sl_price and i - last_sl_idx <= 20:
                distance = last_sl_price - row["close"]
                strength = min(distance / last_sl_price * 100, 1.0)
                signals.append(PatternSignal(
                    type="bearish_bos",
                    strength=round(strength, 4),
                    price_level=last_sl_price,
                    candle_index=i,
                ))

    return signals


# ---------------------------------------------------------------------------
# 5. Order Blocks
# ---------------------------------------------------------------------------

def detect_order_blocks(df: pd.DataFrame) -> list[PatternSignal]:
    """Detect order blocks — last opposite candle before a BOS."""
    signals: list[PatternSignal] = []
    bos_signals = detect_bos(df)

    for bos in bos_signals:
        i = bos.candle_index

        if bos.type == "bullish_bos":
            # Look backward for the last bearish candle before this bullish BOS
            for j in range(i - 1, max(i - 20, -1), -1):
                if _is_bearish(df.iloc[j]):
                    ob_row = df.iloc[j]
                    signals.append(PatternSignal(
                        type="bullish_order_block",
                        strength=round(bos.strength, 4),
                        price_level=ob_row["open"],
                        candle_index=j,
                    ))
                    break

        elif bos.type == "bearish_bos":
            # Look backward for the last bullish candle before this bearish BOS
            for j in range(i - 1, max(i - 20, -1), -1):
                if _is_bullish(df.iloc[j]):
                    ob_row = df.iloc[j]
                    signals.append(PatternSignal(
                        type="bearish_order_block",
                        strength=round(bos.strength, 4),
                        price_level=ob_row["open"],
                        candle_index=j,
                    ))
                    break

    return signals


# ---------------------------------------------------------------------------
# 6. Liquidity Sweep
# ---------------------------------------------------------------------------

def detect_liquidity_sweep(df: pd.DataFrame) -> list[PatternSignal]:
    """Detect liquidity sweeps — wick beyond swing point, close back inside."""
    signals: list[PatternSignal] = []
    swing_highs = _find_swing_highs(df)
    swing_lows = _find_swing_lows(df)

    for i in range(len(df)):
        row = df.iloc[i]

        # Sweep of swing high: wick above, close below
        recent_sh = [s for s in swing_highs if s[0] < i and i - s[0] <= 20]
        for sh_idx, sh_price in recent_sh:
            if row["high"] > sh_price and row["close"] < sh_price:
                sweep_depth = row["high"] - sh_price
                strength = min(sweep_depth / sh_price * 200, 1.0)
                signals.append(PatternSignal(
                    type="bearish_liquidity_sweep",
                    strength=round(strength, 4),
                    price_level=sh_price,
                    candle_index=i,
                ))
                break  # one signal per candle per direction

        # Sweep of swing low: wick below, close above
        recent_sl = [s for s in swing_lows if s[0] < i and i - s[0] <= 20]
        for sl_idx, sl_price in recent_sl:
            if row["low"] < sl_price and row["close"] > sl_price:
                sweep_depth = sl_price - row["low"]
                strength = min(sweep_depth / sl_price * 200, 1.0)
                signals.append(PatternSignal(
                    type="bullish_liquidity_sweep",
                    strength=round(strength, 4),
                    price_level=sl_price,
                    candle_index=i,
                ))
                break

    return signals


# ---------------------------------------------------------------------------
# 7. Detect all patterns
# ---------------------------------------------------------------------------

def detect_all_patterns(df: pd.DataFrame) -> list[PatternSignal]:
    """Run every detector and return signals sorted by candle_index descending."""
    all_signals: list[PatternSignal] = []
    all_signals.extend(detect_engulfing(df))
    all_signals.extend(detect_pin_bars(df))
    all_signals.extend(detect_fvg(df))
    all_signals.extend(detect_bos(df))
    all_signals.extend(detect_order_blocks(df))
    all_signals.extend(detect_liquidity_sweep(df))
    all_signals.sort(key=lambda s: s.candle_index, reverse=True)
    return all_signals
