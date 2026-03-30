import os

import asyncpg
from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from pgvector.asyncpg import register_vector

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/ollama_chat",
)
USE_PGVECTOR = os.getenv("USE_PGVECTOR", "false").strip().lower() in {"1", "true", "yes", "on"}


engine = create_async_engine(DATABASE_URL, future=True, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


if USE_PGVECTOR:
    @event.listens_for(engine.sync_engine, "connect")
    def _register_pgvector(dbapi_connection, _connection_record) -> None:
        try:
            dbapi_connection.run_async(register_vector)
        except ValueError as exc:
            if "unknown type: public.vector" not in str(exc):
                raise


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


async def ensure_vector_extension() -> None:
    if not USE_PGVECTOR:
        return False

    async with engine.begin() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except DBAPIError as exc:
            message = str(exc).lower()
            if "extension \"vector\" is not available" in message or "could not open extension control file" in message:
                return False
            raise
    await engine.dispose()
    return True


async def ensure_rag_embedding_vector_column() -> None:
    if not USE_PGVECTOR:
        return

    async with engine.begin() as conn:
        table_exists = await conn.scalar(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'rag_chunks'
                )
                """
            )
        )
        if not table_exists:
            return

        column_type = await conn.scalar(
            text(
                """
                SELECT udt_name
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'rag_chunks'
                  AND column_name = 'embedding'
                """
            )
        )

        if column_type == "vector":
            return

        await conn.execute(
            text(
                """
                ALTER TABLE rag_chunks
                ALTER COLUMN embedding TYPE vector
                USING embedding::vector
                """
            )
        )


async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
