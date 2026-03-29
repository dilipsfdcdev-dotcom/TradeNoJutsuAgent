"""Multi-timeframe feature builder.

Takes a dict of ``{timeframe_name: DataFrame}`` and produces a single
combined DataFrame where every row is aligned to the **base** timeframe.

* Higher-timeframe (HTF) features are forward-filled / broadcast down.
* Lower-timeframe (LTF) features are reindexed to the base.

Each timeframe's features are computed via
``tradenojutsu.analysis.features_150.compute_features`` and prefixed with
the timeframe name (e.g. ``1h_rsi_14``, ``4h_macd``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tradenojutsu.analysis.features_150 import compute_features
from tradenojutsu.infra.logger import get_logger

logger = get_logger("analysis.multi_timeframe")

# Timeframe ordering from lowest to highest resolution.
# Anything at or below the base is considered LTF; above is HTF.
_TF_ORDER: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}


def _tf_rank(tf: str) -> int:
    """Return numeric rank for a timeframe string."""
    return _TF_ORDER.get(tf.lower(), 0)


class MultiTimeframeBuilder:
    """Combine features across multiple timeframes into a single DataFrame.

    Parameters
    ----------
    base_tf : str
        The base (reference) timeframe.  All other timeframes will be
        aligned to this one.
    feature_kwargs : dict[str, Any] | None
        Extra keyword arguments forwarded to ``compute_features`` for every
        timeframe.
    """

    def __init__(
        self,
        base_tf: str = "15m",
        feature_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.base_tf = base_tf
        self.feature_kwargs = feature_kwargs or {}

    def build(
        self,
        tf_data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Compute and merge features from every supplied timeframe.

        Parameters
        ----------
        tf_data : dict[str, pd.DataFrame]
            Mapping of timeframe label (e.g. ``"1h"``) to its OHLCV
            DataFrame.  Each DataFrame **must** have a ``DatetimeIndex``.

        Returns
        -------
        pd.DataFrame
            A single DataFrame indexed to the base timeframe with columns
            prefixed by their source timeframe.
        """
        if self.base_tf not in tf_data:
            raise ValueError(
                f"Base timeframe '{self.base_tf}' not found in tf_data keys: "
                f"{list(tf_data.keys())}"
            )

        base_df = tf_data[self.base_tf].sort_index()
        self._validate_index(base_df, self.base_tf)

        base_rank = _tf_rank(self.base_tf)

        # Compute features for the base timeframe first
        logger.info("Computing features for base timeframe '%s'", self.base_tf)
        base_features = compute_features(base_df, **self.feature_kwargs)
        combined = self._prefix_columns(base_features, self.base_tf)

        # Process remaining timeframes
        for tf, df in tf_data.items():
            if tf == self.base_tf:
                continue

            self._validate_index(df, tf)
            df = df.sort_index()

            logger.info(
                "Computing features for timeframe '%s' (%d bars)",
                tf,
                len(df),
            )
            features = compute_features(df, **self.feature_kwargs)
            prefixed = self._prefix_columns(features, tf)

            tf_rank = _tf_rank(tf)

            if tf_rank > base_rank:
                # HTF: reindex to base and forward-fill
                merged = self._merge_htf(combined, prefixed)
            else:
                # LTF: reindex to base
                merged = self._merge_ltf(combined, prefixed)

            combined = merged

        logger.info(
            "Multi-timeframe build complete: %d rows x %d columns",
            len(combined),
            len(combined.columns),
        )
        return combined

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_index(df: pd.DataFrame, tf: str) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"DataFrame for timeframe '{tf}' must have a DatetimeIndex."
            )

    @staticmethod
    def _prefix_columns(df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Prefix every column with the timeframe label."""
        df = df.copy()
        df.columns = [f"{tf}_{col}" for col in df.columns]
        return df

    @staticmethod
    def _merge_htf(
        base: pd.DataFrame,
        htf: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge higher-timeframe features into the base DataFrame.

        HTF rows are reindexed to the base index using ``method='ffill'``
        so that each base bar sees the most recent HTF value without
        look-ahead bias.
        """
        htf_reindexed = htf.reindex(base.index, method="ffill")
        return pd.concat([base, htf_reindexed], axis=1)

    @staticmethod
    def _merge_ltf(
        base: pd.DataFrame,
        ltf: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge lower-timeframe features into the base DataFrame.

        LTF bars are reindexed to the base index (nearest previous
        timestamp).  Where multiple LTF bars fall into a single base bar,
        only the last is kept (the one whose timestamp is <= the base
        timestamp).
        """
        ltf_reindexed = ltf.reindex(base.index, method="ffill")
        return pd.concat([base, ltf_reindexed], axis=1)
