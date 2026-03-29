"""Market data fetching from multiple sources (yfinance, ccxt)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from tradenojutsu.infra.logger import get_logger

logger = get_logger("data.market")


class MarketDataFetcher:
    """Unified market data interface supporting multiple backends."""

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

        Returns DataFrame with columns: open, high, low, close, volume
        """
        cache_key = f"{symbol}_{timeframe}_{lookback_days}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.source == "yfinance":
            df = self._fetch_yfinance(symbol, timeframe, lookback_days, end_date)
        elif self.source == "ccxt":
            df = self._fetch_ccxt(symbol, timeframe, lookback_days)
        else:
            raise ValueError(f"Unknown data source: {self.source}")

        if df is not None and not df.empty:
            self._cache[cache_key] = df
        return df

    def _fetch_yfinance(
        self, symbol: str, timeframe: str, lookback_days: int, end_date: datetime | None
    ) -> pd.DataFrame:
        """Fetch from Yahoo Finance."""
        import yfinance as yf

        # Map timeframes
        tf_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "1h",  # yfinance doesn't have 4h, we'll resample
            "1d": "1d", "1w": "1wk",
        }
        yf_interval = tf_map.get(timeframe, "1d")

        # yfinance has limits on intraday data
        if timeframe in ("1m", "5m", "15m", "30m"):
            lookback_days = min(lookback_days, 59)
        elif timeframe in ("1h", "4h"):
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
            logger.warning(f"No data returned for {symbol} from yfinance")
            return pd.DataFrame()

        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index)

        # Resample to 4h if needed
        if timeframe == "4h":
            df = df.resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()

        logger.info(f"Fetched {len(df)} bars for {symbol} ({timeframe}) from yfinance")
        return df

    def _fetch_ccxt(self, symbol: str, timeframe: str, lookback_days: int) -> pd.DataFrame:
        """Fetch from crypto exchange via ccxt."""
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

    def get_current_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        df = self.fetch_ohlcv(symbol, timeframe="1d", lookback_days=5)
        if df.empty:
            raise ValueError(f"Cannot fetch price for {symbol}")
        return float(df["close"].iloc[-1])

    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache.clear()
