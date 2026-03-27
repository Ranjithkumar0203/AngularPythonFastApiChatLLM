import os

import asyncpg
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/ollama_chat",
)

engine = create_async_engine(DATABASE_URL, future=True, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


async def ensure_database_exists() -> None:
    url = make_url(DATABASE_URL)
    if not url.drivername.startswith("postgresql"):
        return

    database_name = url.database
    if not database_name:
        return

    admin_database = os.getenv("POSTGRES_ADMIN_DB", "postgres")
    conn = await asyncpg.connect(
        user=url.username,
        password=url.password,
        host=url.host or "localhost",
        port=url.port or 5432,
        database=admin_database,
    )

    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            database_name,
        )
        if not exists:
            escaped_name = database_name.replace('"', '""')
            await conn.execute(f'CREATE DATABASE "{escaped_name}"')
    finally:
        await conn.close()


async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
