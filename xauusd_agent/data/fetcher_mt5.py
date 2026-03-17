"""
Multi-timeframe OHLCV fetcher for MetaTrader 5.

Provides cached access to M3, M5, M15, H1, H4, and D1 candles for XAUUSD,
live account information, spread, and current price data.
"""

from __future__ import annotations

import time
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Timeframe configuration
# ---------------------------------------------------------------------------

TIMEFRAMES: dict[str, int] = {
    "M3": mt5.TIMEFRAME_M3,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

CANDLE_COUNTS: dict[str, int] = {
    "M3": 300,
    "M5": 200,
    "M15": 100,
    "H1": 100,
    "H4": 60,
    "D1": 90,
}

CACHE_SECONDS: dict[str, int] = {
    "M3": 180,
    "M5": 300,
    "M15": 900,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
}


class MT5DataFetcher:
    """Cached multi-timeframe OHLCV fetcher backed by MetaTrader 5."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _is_cache_valid(self, tf: str) -> bool:
        """Return *True* if cached data for *tf* is still fresh."""
        if tf not in self._cache:
            return False
        cached_ts, _ = self._cache[tf]
        return (time.time() - cached_ts) < CACHE_SECONDS.get(tf, 0)

    # ------------------------------------------------------------------
    # Single-timeframe fetch
    # ------------------------------------------------------------------

    def _fetch_single_tf(self, symbol: str, tf_name: str) -> pd.DataFrame:
        """Fetch OHLCV bars for a single timeframe via MT5.

        Returns a :class:`~pandas.DataFrame` indexed by ``datetime`` with
        columns: open, high, low, close, tick_volume, spread.
        """
        mt5_tf = TIMEFRAMES[tf_name]
        count = CANDLE_COUNTS[tf_name]

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.warning(
                "No data returned for %s %s: %s",
                symbol,
                tf_name,
                mt5.last_error(),
            )
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "tick_volume": "tick_volume",
                "spread": "spread",
            },
            inplace=True,
        )
        df = df[["open", "high", "low", "close", "tick_volume", "spread"]]

        logger.debug(
            "Fetched %d bars for %s %s", len(df), symbol, tf_name
        )
        return df

    # ------------------------------------------------------------------
    # M3 synthesis from M1
    # ------------------------------------------------------------------

    def _synthesize_m3_from_m1(self, symbol: str) -> pd.DataFrame:
        """Build 3-minute bars by resampling 900 M1 candles.

        Used as a fallback when the broker does not natively support
        ``TIMEFRAME_M3``.
        """
        logger.info("Synthesizing M3 from M1 bars for %s", symbol)

        rates = mt5.copy_rates_from_pos(
            symbol, mt5.TIMEFRAME_M1, 0, 900
        )
        if rates is None or len(rates) == 0:
            logger.warning(
                "No M1 data for M3 synthesis: %s", mt5.last_error()
            )
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        df_m3 = (
            df.resample("3min")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "tick_volume": "sum",
                    "spread": "last",
                }
            )
            .dropna(subset=["open"])
        )

        logger.debug(
            "Synthesized %d M3 bars from M1 for %s", len(df_m3), symbol
        )
        return df_m3

    # ------------------------------------------------------------------
    # Fetch all timeframes
    # ------------------------------------------------------------------

    def fetch_all_timeframes(
        self, symbol: str = "XAUUSD"
    ) -> dict[str, pd.DataFrame]:
        """Fetch all configured timeframes, using cache where valid.

        M3 is always fetched fresh (never served from cache).  For M3,
        native ``TIMEFRAME_M3`` is attempted first; on failure the data
        is synthesized from M1 candles.

        Returns a mapping ``{tf_name: DataFrame}``.
        """
        result: dict[str, pd.DataFrame] = {}

        for tf_name in TIMEFRAMES:
            if tf_name == "M3":
                # M3 always fresh -- try native, fall back to synthesis.
                df = self._fetch_single_tf(symbol, "M3")
                if df.empty:
                    df = self._synthesize_m3_from_m1(symbol)
                self._cache["M3"] = (time.time(), df)
                result["M3"] = df
            elif self._is_cache_valid(tf_name):
                logger.debug("Serving %s from cache", tf_name)
                _, df = self._cache[tf_name]
                result[tf_name] = df
            else:
                df = self._fetch_single_tf(symbol, tf_name)
                self._cache[tf_name] = (time.time(), df)
                result[tf_name] = df

        return result

    # ------------------------------------------------------------------
    # Account / tick helpers
    # ------------------------------------------------------------------

    def get_live_account(self) -> dict:
        """Return live account summary from MT5.

        Keys: balance, equity, margin_free, profit, daily_pnl, open_count.
        """
        info = mt5.account_info()
        if info is None:
            logger.error("Failed to get account info: %s", mt5.last_error())
            return {}

        positions = mt5.positions_get()
        open_count = len(positions) if positions is not None else 0

        total_profit = 0.0
        if positions:
            total_profit = sum(pos.profit for pos in positions)

        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin_free": info.margin_free,
            "profit": info.profit,
            "daily_pnl": total_profit,
            "open_count": open_count,
        }

    def get_spread_pips(self, symbol: str = "XAUUSD") -> float:
        """Return current spread in pips.

        For XAUUSD, 1 pip = 0.1 (i.e. ``point * 10``).
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(
                "Failed to get tick for %s: %s", symbol, mt5.last_error()
            )
            return 0.0

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            logger.error(
                "Failed to get symbol info for %s: %s",
                symbol,
                mt5.last_error(),
            )
            return 0.0

        point = sym_info.point
        pip_size = point * 10  # XAUUSD: 1 pip = 0.1
        spread_price = tick.ask - tick.bid
        spread_pips = spread_price / pip_size

        return round(spread_pips, 2)

    def get_current_price(self, symbol: str = "XAUUSD") -> dict:
        """Return the current bid, ask, and server time for *symbol*."""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(
                "Failed to get tick for %s: %s", symbol, mt5.last_error()
            )
            return {}

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "time": tick.time,
        }
