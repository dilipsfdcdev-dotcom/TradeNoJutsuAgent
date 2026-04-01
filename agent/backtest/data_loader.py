"""Load historical candle data for backtesting."""

import pandas as pd
import structlog
from pathlib import Path
from datetime import datetime

logger = structlog.get_logger()


def load_from_mt5(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load historical candles from MT5.

    Returns DataFrame with columns: time, open, high, low, close, volume, spread
    """
    from agent.data.mt5_feed import TF_MAP, init_mt5

    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.error("mt5.import_failed", hint="MetaTrader5 package not installed")
        return pd.DataFrame()

    init_mt5()
    tf = TF_MAP.get(timeframe, mt5.TIMEFRAME_M1)
    rates = mt5.copy_rates_range(symbol, tf, start, end)

    if rates is None or len(rates) == 0:
        logger.warning("no_data_from_mt5", symbol=symbol, start=start, end=end)
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    if "tick_volume" in df.columns:
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
    if "real_volume" in df.columns:
        df.drop(columns=["real_volume"], inplace=True, errors="ignore")

    return df[["time", "open", "high", "low", "close", "volume", "spread"]]


def load_from_csv(filepath: str | Path) -> pd.DataFrame:
    """Load candles from a CSV file.

    Expected columns: time/date/datetime, open, high, low, close, volume
    """
    df = pd.read_csv(filepath)
    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Find the time column
    for col in ["time", "date", "datetime", "timestamp"]:
        if col in df.columns:
            df["time"] = pd.to_datetime(df[col])
            break

    if "time" not in df.columns:
        raise ValueError(
            f"No time column found in {filepath}. "
            "Expected one of: time, date, datetime, timestamp"
        )

    if "spread" not in df.columns:
        df["spread"] = 0
    if "volume" not in df.columns:
        df["volume"] = 0

    return df[["time", "open", "high", "low", "close", "volume", "spread"]]
