"""
Multi-timeframe feature assembly for XAUUSD.

Merges feature matrices from multiple timeframes into a single DataFrame
indexed on the lowest timeframe (M3) timestamps.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from xauusd_agent.features.ultimate_150_features import compute_features
from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# Timeframes treated as higher-timeframe (only latest row broadcast)
_HTF_NAMES = {"H1", "H4", "D1"}

# Timeframes whose full series are kept and aligned
_LTF_NAMES = {"M3", "M5", "M15"}


def build_multi_tf_features(
    all_tf: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a unified feature matrix from multiple timeframes.

    Parameters
    ----------
    all_tf : dict[str, pd.DataFrame]
        Mapping of timeframe name (e.g. ``"M3"``, ``"H4"``) to OHLCV
        DataFrames.  Each DataFrame must have a ``DatetimeIndex`` and
        columns: open, high, low, close, tick_volume.

    Returns
    -------
    pd.DataFrame
        Single DataFrame indexed on the M3 (or lowest available)
        timestamps with all features prefixed by their timeframe name.
    """
    if not all_tf:
        raise ValueError("all_tf dict is empty – need at least one timeframe")

    # Determine the base (lowest) timeframe index
    base_tf = None
    for candidate in ("M3", "M5", "M15"):
        if candidate in all_tf:
            base_tf = candidate
            break
    if base_tf is None:
        # Fall back to the timeframe with the most rows
        base_tf = max(all_tf, key=lambda k: len(all_tf[k]))
    base_index = all_tf[base_tf].index.copy()

    logger.info(
        "Building multi-TF features – base=%s (%d rows), timeframes=%s",
        base_tf,
        len(base_index),
        list(all_tf.keys()),
    )

    frames: list[pd.DataFrame] = []

    for tf_name, tf_df in all_tf.items():
        prefix = tf_name
        feats = compute_features(tf_df, prefix=prefix)

        if tf_name in _HTF_NAMES:
            # Higher timeframe: broadcast latest known value to each base row
            feats = _broadcast_htf(feats, base_index)
        else:
            # Lower / same timeframes: reindex to base with forward-fill
            feats = feats.reindex(base_index, method="ffill")

        frames.append(feats)

    result = pd.concat(frames, axis=1)

    # Final NaN cleanup
    result = result.ffill()
    first_valid = result.dropna(how="any").index.min()
    if first_valid is not None:
        result = result.loc[first_valid:]

    logger.info(
        "Multi-TF feature matrix: %d rows x %d cols",
        result.shape[0],
        result.shape[1],
    )
    return result


def _broadcast_htf(
    htf_features: pd.DataFrame,
    base_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Align higher-timeframe features to the base index.

    For each base timestamp, the most recent HTF row that is <= that
    timestamp is used (forward-fill / as-of merge logic).
    """
    # Sort both indices to ensure correctness
    htf_features = htf_features.sort_index()
    base_sorted = base_index.sort_values()

    # Use reindex with forward-fill (equivalent to as-of join)
    aligned = htf_features.reindex(base_sorted, method="ffill")
    return aligned
