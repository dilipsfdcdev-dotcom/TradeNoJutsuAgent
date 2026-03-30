"""Database session management."""

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from agent.config import settings

# Convert postgresql:// to postgresql+asyncpg://
database_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(database_url, echo=False, pool_size=10, max_overflow=20)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
