"""Configuration loaded from environment variables with validation."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # MT5
    MT5_LOGIN: int = 0
    MT5_PASSWORD: str = ""
    MT5_SERVER: str = ""
    MT5_PATH: str = "C:/Program Files/MetaTrader 5/terminal64.exe"

    # Claude API
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"

    # Database
    DATABASE_URL: str = "postgresql://agent:password@localhost:5432/trading_agent"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Risk settings
    MAX_RISK_PER_TRADE_PCT: float = 1.0
    MAX_DAILY_LOSS_PCT: float = 3.0
    MAX_OPEN_TRADES: int = 3
    MAX_DRAWDOWN_PCT: float = 8.0

    # Scalping settings
    SYMBOLS: str = "XAUUSD,BTCUSD,XAGUSD"
    TIMEFRAMES: str = "M1,M3"
    LOOKBACK_CANDLES: int = 100
    MIN_RR_RATIO: float = 1.5
    SPREAD_FILTER_MULTIPLIER: float = 2.0

    # News
    NEWS_API_KEY: str = ""
    NEWS_CHECK_INTERVAL_SEC: int = 60

    # Dashboard
    DASHBOARD_PORT: int = 3000
    WS_PORT: int = 8765
    API_PORT: int = 8000

    @property
    def symbols_list(self) -> list[str]:
        return [s.strip() for s in self.SYMBOLS.split(",")]

    @property
    def timeframes_list(self) -> list[str]:
        return [t.strip() for t in self.TIMEFRAMES.split(",")]

    @property
    def async_database_url(self) -> str:
        return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


settings = Settings()
