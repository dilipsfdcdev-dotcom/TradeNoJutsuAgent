"""MT5 initialization, candle fetching, tick streaming, and Redis caching."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd
import structlog

from agent.config import settings

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Timeframe mapping
# ---------------------------------------------------------------------------
TF_MAP: dict[str, int] = {
    "M1": mt5.TIMEFRAME_M1,
    "M3": mt5.TIMEFRAME_M3,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
}

# How many candles to keep per symbol/tf in Redis
_ROLLING_BUFFER = settings.LOOKBACK_CANDLES  # default 100

# Reconnect back-off parameters (seconds)
_RECONNECT_BASE = 2
_RECONNECT_MAX = 60


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def init_mt5() -> bool:
    """Initialize the MT5 terminal connection.

    Returns True on success, False on failure.
    """
    kwargs: dict = {"path": settings.MT5_PATH}
    if settings.MT5_LOGIN:
        kwargs.update(
            login=settings.MT5_LOGIN,
            password=settings.MT5_PASSWORD,
            server=settings.MT5_SERVER,
        )

    if not mt5.initialize(**kwargs):
        log.error(
            "mt5.initialize failed",
            error_code=mt5.last_error(),
        )
        return False

    info = mt5.terminal_info()
    log.info(
        "MT5 connected",
        build=getattr(info, "build", None),
        company=getattr(info, "company", None),
        connected=getattr(info, "connected", None),
    )
    return True


def shutdown_mt5() -> None:
    """Shut down the MT5 terminal connection."""
    mt5.shutdown()
    log.info("MT5 shutdown complete")


def _ensure_connected() -> bool:
    """Return True if connected, attempting a reconnect if not."""
    info = mt5.terminal_info()
    if info is not None and getattr(info, "connected", False):
        return True

    log.warning("MT5 connection lost — attempting reconnect")
    mt5.shutdown()
    return init_mt5()


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

def get_candles(symbol: str, tf: str, count: int) -> pd.DataFrame:
    """Fetch the last *count* candles for *symbol* / *tf*.

    Parameters
    ----------
    symbol : str
        e.g. ``"XAUUSD"``
    tf : str
        One of the keys in :data:`TF_MAP` (``"M1"``, ``"M3"`` …).
    count : int
        Number of candles to retrieve.

    Returns
    -------
    pd.DataFrame
        Columns: time, open, high, low, close, volume, spread.
        Empty DataFrame on error.
    """
    mt5_tf = TF_MAP.get(tf)
    if mt5_tf is None:
        log.error("unknown timeframe", tf=tf, known=list(TF_MAP))
        return pd.DataFrame()

    if not _ensure_connected():
        return pd.DataFrame()

    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
    if rates is None or len(rates) == 0:
        log.warning(
            "no candle data returned",
            symbol=symbol,
            tf=tf,
            error=mt5.last_error(),
        )
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["time", "open", "high", "low", "close", "tick_volume", "spread"]].rename(
        columns={"tick_volume": "volume"},
    )


def get_tick(symbol: str) -> dict:
    """Return the latest tick for *symbol*.

    Returns
    -------
    dict
        ``{bid, ask, spread, time}``  — empty dict on error.
    """
    if not _ensure_connected():
        return {}

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.warning("no tick data", symbol=symbol, error=mt5.last_error())
        return {}

    return {
        "bid": tick.bid,
        "ask": tick.ask,
        "spread": round(tick.ask - tick.bid, 6),
        "time": datetime.fromtimestamp(tick.time, tz=timezone.utc).isoformat(),
    }


def get_account_info() -> dict:
    """Return core account metrics.

    Returns
    -------
    dict
        ``{balance, equity, margin, free_margin, profit}`` — empty dict
        on error.
    """
    if not _ensure_connected():
        return {}

    info = mt5.account_info()
    if info is None:
        log.warning("cannot fetch account info", error=mt5.last_error())
        return {}

    return {
        "balance": info.balance,
        "equity": info.equity,
        "margin": info.margin,
        "free_margin": info.margin_free,
        "profit": info.profit,
    }


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

def _redis_key(symbol: str, tf: str) -> str:
    return f"candles:{symbol}:{tf}"


def _candles_to_json(df: pd.DataFrame) -> str:
    """Serialize a candle DataFrame to a compact JSON string."""
    records = df.copy()
    records["time"] = records["time"].astype(str)
    return records.to_json(orient="records")


def _push_candles(redis_client, symbol: str, tf: str, df: pd.DataFrame) -> None:
    """Store the rolling buffer in Redis and publish an update event."""
    if df.empty:
        return

    key = _redis_key(symbol, tf)
    payload = _candles_to_json(df)

    redis_client.set(key, payload)

    # Pub/Sub notification for downstream consumers
    channel = f"candles_update:{symbol}:{tf}"
    redis_client.publish(channel, payload)

    log.debug(
        "redis candle update",
        symbol=symbol,
        tf=tf,
        rows=len(df),
        key=key,
    )


# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------

def stream_loop(redis_client) -> None:  # noqa: C901 – intentionally a long-running loop
    """Continuously fetch candles for all configured symbols/TFs and cache
    them in Redis.

    This function **runs forever** (blocking).  It polls MT5 on each
    iteration, stores the latest rolling buffer of candles in Redis, and
    publishes an update on the ``candles_update:{symbol}:{tf}`` channel.

    Parameters
    ----------
    redis_client
        A synchronous ``redis.Redis`` instance (or compatible).
    """
    symbols = settings.symbols_list
    timeframes = settings.timeframes_list
    backoff = _RECONNECT_BASE

    log.info(
        "stream_loop starting",
        symbols=symbols,
        timeframes=timeframes,
        buffer_size=_ROLLING_BUFFER,
    )

    if not _ensure_connected():
        log.error("initial MT5 connection failed — will retry in loop")

    while True:
        try:
            if not _ensure_connected():
                log.warning("MT5 unavailable, backing off", seconds=backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_MAX)
                continue

            # Reset back-off on success
            backoff = _RECONNECT_BASE

            for symbol in symbols:
                for tf in timeframes:
                    df = get_candles(symbol, tf, _ROLLING_BUFFER)
                    _push_candles(redis_client, symbol, tf, df)

            # Sleep ~1 s between polling cycles (M1 is the fastest TF)
            time.sleep(1)

        except KeyboardInterrupt:
            log.info("stream_loop interrupted by user")
            break
        except Exception:
            log.exception(
                "unhandled error in stream_loop — retrying",
                backoff=backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX)

    shutdown_mt5()
    log.info("stream_loop exited")
