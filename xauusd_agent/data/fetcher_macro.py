"""
Macro-economic data fetcher using yfinance.

Provides recent VIX, DXY, Oil, and BTC values along with their
percentage changes, cached for 15 minutes.
"""

from __future__ import annotations

import time

import yfinance as yf

from xauusd_agent.infra.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_TICKERS: dict[str, str] = {
    "vix": "^VIX",
    "dxy": "DX-Y.NYB",
    "oil": "CL=F",
    "btc": "BTC-USD",
}

_CACHE_TTL_SECONDS = 900  # 15 minutes
_cache: dict[str, tuple[float, dict]] = {}
_CACHE_KEY = "macro_data"


def _pct_change(series) -> float:
    """Return percentage change between last two available values."""
    clean = series.dropna()
    if len(clean) < 2:
        return 0.0
    prev = clean.iloc[-2]
    curr = clean.iloc[-1]
    if prev == 0:
        return 0.0
    return round(((curr - prev) / prev) * 100, 4)


def fetch_macro_data() -> dict:
    """Fetch latest macro indicator values and percentage changes.

    Returns a dict with keys:
        vix, dxy, oil, btc           -- latest closing prices
        vix_change_pct, dxy_change_pct -- hourly % change for VIX and DXY

    Data is cached for 15 minutes to reduce API calls.
    """
    # Serve from cache if still fresh.
    if _CACHE_KEY in _cache:
        cached_ts, cached_data = _cache[_CACHE_KEY]
        if (time.time() - cached_ts) < _CACHE_TTL_SECONDS:
            logger.debug("Serving macro data from cache")
            return cached_data

    result: dict = {
        "vix": None,
        "dxy": None,
        "oil": None,
        "btc": None,
        "vix_change_pct": 0.0,
        "dxy_change_pct": 0.0,
    }

    for key, ticker in _TICKERS.items():
        try:
            df = yf.download(
                ticker,
                period="5d",
                interval="1h",
                progress=False,
            )
            if df.empty:
                logger.warning("No data returned for %s (%s)", key, ticker)
                continue

            close = df["Close"].squeeze()
            latest = close.dropna().iloc[-1]
            result[key] = round(float(latest), 4)

            change_key = f"{key}_change_pct"
            if change_key in result:
                result[change_key] = _pct_change(close)

            logger.debug(
                "Macro %s = %.4f (change %.4f%%)",
                key,
                result[key],
                result.get(f"{key}_change_pct", 0.0),
            )
        except Exception:
            logger.exception("Failed to fetch macro data for %s", ticker)

    _cache[_CACHE_KEY] = (time.time(), result)
    logger.info("Macro data refreshed: %s", result)
    return result
