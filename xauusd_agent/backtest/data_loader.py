"""
Historical data loader for XAUUSD backtesting.

Downloads OHLCV data from MetaTrader 5 or loads cached parquet files.
Supports M1-to-M3 synthesis when native M3 bars are unavailable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MT5 timeframe name -> constant mapping
# ---------------------------------------------------------------------------

_TF_MAP: dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M3": mt5.TIMEFRAME_M3,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

# The six timeframes required for backtesting
_REQUIRED_TFS: list[str] = ["M3", "M5", "M15", "H1", "H4", "D1"]


class BacktestDataLoader:
    """Load historical data from MT5 or parquet cache for backtesting."""

    def __init__(self, data_dir: str = "backtest_data/") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # MT5 download
    # ------------------------------------------------------------------

    def download_mt5_data(
        self,
        symbol: str,
        timeframe_name: str,
        mt5_tf: int,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Download historical data from MT5.

        Converts the MT5 rates array to a DataFrame with a UTC
        ``DatetimeIndex`` and columns: open, high, low, close,
        tick_volume, spread.  The result is also saved to parquet for
        caching.

        Parameters
        ----------
        symbol:
            Instrument name, e.g. ``"XAUUSD"``.
        timeframe_name:
            Human-readable name used for the cache filename (e.g. ``"M5"``).
        mt5_tf:
            MT5 timeframe constant (e.g. ``mt5.TIMEFRAME_M5``).
        start_date / end_date:
            ISO-8601 date strings (``"YYYY-MM-DD"``).

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame indexed by datetime, empty on failure.
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        logger.info(
            "Downloading %s %s from %s to %s",
            symbol, timeframe_name, start_date, end_date,
        )

        rates = mt5.copy_rates_range(symbol, mt5_tf, start_dt, end_dt)

        if rates is None or len(rates) == 0:
            logger.warning(
                "No data returned for %s %s (%s -> %s): %s",
                symbol, timeframe_name, start_date, end_date, mt5.last_error(),
            )
            return pd.DataFrame()

        df = self._rates_to_dataframe(rates)

        # Cache to parquet
        filename = f"{symbol}_{timeframe_name}_{start_date}_{end_date}.parquet"
        self.save_to_parquet(df, filename)

        logger.info("Downloaded %d bars for %s %s", len(df), symbol, timeframe_name)
        return df

    # ------------------------------------------------------------------
    # M3 synthesis from M1
    # ------------------------------------------------------------------

    def synthesize_m3_from_m1(self, m1_df: pd.DataFrame) -> pd.DataFrame:
        """Resample M1 bars to M3 bars.

        Aggregation rules::

            open=first, high=max, low=min, close=last,
            tick_volume=sum, spread=last.

        Parameters
        ----------
        m1_df:
            M1 OHLCV DataFrame with a DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Resampled M3 DataFrame, empty if input is empty.
        """
        if m1_df.empty:
            logger.warning("Empty M1 DataFrame passed to synthesize_m3_from_m1")
            return pd.DataFrame()

        agg_map: dict[str, str] = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "tick_volume": "sum",
        }
        if "spread" in m1_df.columns:
            agg_map["spread"] = "last"

        m3_df = (
            m1_df.resample("3min")
            .agg(agg_map)
            .dropna(subset=["open"])
        )

        logger.info(
            "Synthesized %d M3 bars from %d M1 bars", len(m3_df), len(m1_df)
        )
        return m3_df

    # ------------------------------------------------------------------
    # Load all required timeframes
    # ------------------------------------------------------------------

    def load_all_timeframes(
        self,
        symbol: str = "XAUUSD",
        start: str = "2023-01-01",
        end: str = "2025-12-31",
    ) -> dict[str, pd.DataFrame]:
        """Load all six timeframes required for backtesting.

        Loading order for each timeframe:

        1. Try loading from parquet cache.
        2. If cache miss, download from MT5 and cache.
        3. For M3 specifically: attempt native download first; if that
           fails, download M1 and synthesize M3 via
           :meth:`synthesize_m3_from_m1`.

        Parameters
        ----------
        symbol:
            Trading instrument (default ``"XAUUSD"``).
        start / end:
            ISO-8601 date strings bounding the backtest period.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping ``{timeframe_name: ohlcv_dataframe}`` for the six
            required timeframes.
        """
        all_tf: dict[str, pd.DataFrame] = {}

        for tf_name in _REQUIRED_TFS:
            cache_file = f"{symbol}_{tf_name}_{start}_{end}.parquet"

            # 1. Try parquet cache
            df = self.load_from_parquet(cache_file)
            if df is not None:
                logger.info(
                    "Loaded %s %s from cache (%d bars)",
                    symbol, tf_name, len(df),
                )
                all_tf[tf_name] = df
                continue

            # 2. Download from MT5
            mt5_tf = _TF_MAP.get(tf_name)
            if mt5_tf is not None:
                df = self.download_mt5_data(symbol, tf_name, mt5_tf, start, end)

            # 3. M3 fallback: synthesize from M1
            if (df is None or df.empty) and tf_name == "M3":
                logger.info("M3 native download failed; synthesizing from M1")
                m1_cache = f"{symbol}_M1_{start}_{end}.parquet"
                m1_df = self.load_from_parquet(m1_cache)
                if m1_df is None:
                    m1_mt5_tf = _TF_MAP["M1"]
                    m1_df = self.download_mt5_data(
                        symbol, "M1", m1_mt5_tf, start, end,
                    )
                if m1_df is not None and not m1_df.empty:
                    df = self.synthesize_m3_from_m1(m1_df)
                    if not df.empty:
                        self.save_to_parquet(df, cache_file)

            if df is None or df.empty:
                logger.warning("No data available for %s %s", symbol, tf_name)
                all_tf[tf_name] = pd.DataFrame()
            else:
                all_tf[tf_name] = df

        loaded_counts = {
            tf: len(df) for tf, df in all_tf.items() if not df.empty
        }
        logger.info("Loaded timeframes: %s", loaded_counts)
        return all_tf

    # ------------------------------------------------------------------
    # Parquet persistence
    # ------------------------------------------------------------------

    def save_to_parquet(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to parquet in ``data_dir``.

        No-op if the DataFrame is empty.
        """
        if df.empty:
            return
        path = self.data_dir / filename
        df.to_parquet(path, engine="pyarrow")
        logger.debug("Saved %d rows to %s", len(df), path)

    def load_from_parquet(self, filename: str) -> pd.DataFrame | None:
        """Load DataFrame from parquet if the file exists.

        Returns ``None`` if the file is not found.
        """
        path = self.data_dir / filename
        if not path.exists():
            logger.debug("Parquet not found: %s", path)
            return None

        df = pd.read_parquet(path, engine="pyarrow")
        logger.info("Loaded %d rows from %s", len(df), path)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rates_to_dataframe(rates) -> pd.DataFrame:
        """Convert MT5 rates structured array to a pandas DataFrame."""
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        expected_cols = ["open", "high", "low", "close", "tick_volume", "spread"]
        available = [c for c in expected_cols if c in df.columns]
        return df[available]
