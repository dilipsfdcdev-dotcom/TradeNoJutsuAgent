"""Database session management."""

import asyncio

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from agent.config import settings

# Convert postgresql:// to postgresql+asyncpg://
database_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Primary engine – used by the main event loop (agent tasks).
engine = create_async_engine(database_url, echo=False, pool_size=10, max_overflow=20)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ---------------------------------------------------------------------------
# Per-event-loop engine registry
# ---------------------------------------------------------------------------
# The WebSocket server runs in a background thread with its own event loop.
# asyncpg connections are bound to the loop that created them, so we must
# create a separate engine for each distinct loop.

_engines: dict[int, async_sessionmaker] = {}


def _session_factory_for_current_loop() -> async_sessionmaker:
    """Return (or create) a session factory bound to the running event loop."""
    loop = asyncio.get_running_loop()
    loop_id = id(loop)

    if loop_id not in _engines:
        eng = create_async_engine(
            database_url, echo=False, pool_size=5, max_overflow=10,
        )
        _engines[loop_id] = async_sessionmaker(
            eng, class_=AsyncSession, expire_on_commit=False,
        )

    return _engines[loop_id]


async def get_session() -> AsyncSession:  # type: ignore[misc]
    factory = _session_factory_for_current_loop()
    async with factory() as session:
        yield session
