"""Market data fetching from multiple sources (yfinance, ccxt, MT5).

Supports all timeframes including 1m, 3m (via resampling), 5m, 15m, 1h, 4h, 1d.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from tradenojutsu.infra.logger import get_logger

logger = get_logger("data.market")

# Timeframes that need resampling from a base timeframe
_RESAMPLE_MAP = {
    "3m": ("1m", "3min"),     # 3m from 1m
    "4h": ("1h", "4h"),       # 4h from 1h
    "2h": ("1h", "2h"),       # 2h from 1h
}


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe."""
    return df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()


class MarketDataFetcher:
    """Unified market data interface supporting multiple backends.

    Supports 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 1d, 1w timeframes.
    3m and 4h are built by resampling from 1m and 1h respectively.
    """

    def __init__(self, source: str = "yfinance", exchange: str = "binance"):
        self.source = source
        self.exchange_name = exchange
        self._cache: dict[str, pd.DataFrame] = {}

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1d",
        lookback_days: int = 365,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.

        Returns DataFrame with columns: open, high, low, close, volume.
        Handles 3m/4h by fetching base timeframe and resampling.
        """
        cache_key = f"{symbol}_{timeframe}_{lookback_days}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check if we need to resample
        needs_resample = timeframe in _RESAMPLE_MAP
        if needs_resample:
            base_tf, resample_rule = _RESAMPLE_MAP[timeframe]
            # Fetch more data at base timeframe
            extra_factor = int(resample_rule.replace("min", "").replace("h", ""))
            base_lookback = lookback_days
        else:
            base_tf = timeframe
            resample_rule = None

        if self.source == "yfinance":
            df = self._fetch_yfinance(symbol, base_tf, lookback_days, end_date)
        elif self.source == "ccxt":
            df = self._fetch_ccxt(symbol, base_tf if needs_resample else timeframe, lookback_days)
        elif self.source == "mt5":
            df = self._fetch_mt5(symbol, base_tf if needs_resample else timeframe, lookback_days)
        else:
            raise ValueError(f"Unknown data source: {self.source}")

        # Resample if needed (3m from 1m, 4h from 1h)
        if needs_resample and df is not None and not df.empty:
            df = _resample_ohlcv(df, resample_rule)
            logger.info(f"Resampled to {timeframe}: {len(df)} bars")

        if df is not None and not df.empty:
            self._cache[cache_key] = df
        return df if df is not None else pd.DataFrame()

    def _fetch_yfinance(
        self, symbol: str, timeframe: str, lookback_days: int, end_date: datetime | None
    ) -> pd.DataFrame:
        """Fetch from Yahoo Finance.

        yfinance limits:
        - 1m: max 7 days
        - 5m/15m/30m: max 59 days
        - 1h: max 729 days
        - 1d: unlimited
        """
        import yfinance as yf

        tf_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "1d": "1d", "1w": "1wk",
        }
        yf_interval = tf_map.get(timeframe, "1d")

        # yfinance intraday limits
        if timeframe == "1m":
            lookback_days = min(lookback_days, 7)
        elif timeframe in ("5m", "15m", "30m"):
            lookback_days = min(lookback_days, 59)
        elif timeframe == "1h":
            lookback_days = min(lookback_days, 729)

        period_end = end_date or datetime.utcnow()
        period_start = period_end - timedelta(days=lookback_days)

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=period_start.strftime("%Y-%m-%d"),
            end=period_end.strftime("%Y-%m-%d"),
            interval=yf_interval,
        )

        if df.empty:
            logger.warning(f"No data returned for {symbol} ({timeframe}) from yfinance")
            return pd.DataFrame()

        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index)

        logger.info(f"Fetched {len(df)} bars for {symbol} ({timeframe}) from yfinance")
        return df

    def _fetch_ccxt(self, symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
        """Fetch from crypto exchange via ccxt.

        ccxt supports 1m, 3m, 5m, 15m, 1h, 4h, 1d natively on most exchanges.
        """
        import ccxt

        exchange_class = getattr(ccxt, self.exchange_name)
        exchange = exchange_class({"enableRateLimit": True})

        since = int((datetime.utcnow() - timedelta(days=lookback_days)).timestamp() * 1000)

        all_ohlcv = []
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break

        if not all_ohlcv:
            logger.warning(f"No data returned for {symbol} from {self.exchange_name}")
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        logger.info(f"Fetched {len(df)} bars for {symbol} ({timeframe}) from {self.exchange_name}")
        return df

    def _fetch_mt5(self, symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
        """Fetch from MetaTrader 5 terminal.

        MT5 supports all timeframes natively including M1, M3, M5, etc.
        """
        try:
            import MetaTrader5 as mt5
        except ImportError:
            logger.error("MetaTrader5 not installed. Run: pip install MetaTrader5")
            return pd.DataFrame()

        # MT5 timeframe constants
        tf_map = {
            "1m": mt5.TIMEFRAME_M1,
            "3m": mt5.TIMEFRAME_M3,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "2h": mt5.TIMEFRAME_H2,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
            "1w": mt5.TIMEFRAME_W1,
        }

        mt5_tf = tf_map.get(timeframe)
        if mt5_tf is None:
            logger.error(f"Unknown MT5 timeframe: {timeframe}")
            return pd.DataFrame()

        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return pd.DataFrame()

        # Calculate number of bars needed
        bars_per_day = {
            "1m": 1440, "3m": 480, "5m": 288, "15m": 96, "30m": 48,
            "1h": 24, "2h": 12, "4h": 6, "1d": 1, "1w": 1,
        }
        num_bars = lookback_days * bars_per_day.get(timeframe, 1)
        num_bars = min(num_bars, 100000)  # MT5 limit

        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, num_bars)
        if rates is None or len(rates) == 0:
            logger.warning(f"No MT5 data for {symbol} ({timeframe}): {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].copy()

        logger.info(f"Fetched {len(df)} bars for {symbol} ({timeframe}) from MT5")
        return df

    def get_current_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        if self.source == "mt5":
            try:
                import MetaTrader5 as mt5
                if mt5.initialize():
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        return float((tick.bid + tick.ask) / 2)
            except Exception:
                pass

        df = self.fetch_ohlcv(symbol, timeframe="1d", lookback_days=5)
        if df.empty:
            raise ValueError(f"Cannot fetch price for {symbol}")
        return float(df["close"].iloc[-1])

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
