import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def sample_candles():
    """Generate 100 sample XAUUSD 1-minute candles."""
    np.random.seed(42)
    dates = [datetime(2024, 1, 15, 10, 0) + timedelta(minutes=i) for i in range(100)]
    base_price = 2050.0
    prices = [base_price]
    for i in range(99):
        prices.append(prices[-1] + np.random.randn() * 2)

    data = []
    for i, (dt, price) in enumerate(zip(dates, prices)):
        o = price
        h = price + abs(np.random.randn()) * 1.5
        l = price - abs(np.random.randn()) * 1.5
        c = price + np.random.randn() * 1.0
        v = int(np.random.uniform(100, 1000))
        data.append({
            "time": dt,
            "open": o,
            "high": max(o, h, c),
            "low": min(o, l, c),
            "close": c,
            "volume": v,
            "spread": 20,
        })

    return pd.DataFrame(data)


@pytest.fixture
def sample_account():
    return {
        "balance": 10000.0,
        "equity": 10050.0,
        "margin": 500.0,
        "free_margin": 9550.0,
        "profit": 50.0,
    }


@pytest.fixture
def mock_redis():
    r = AsyncMock()
    r.get = AsyncMock(return_value=None)
    r.setex = AsyncMock()
    r.publish = AsyncMock()
    return r
