"""PostgreSQL connection pool management using asyncpg.

Provides connection pooling, health checks, transaction support,
and query helpers for The Author Library's PostgreSQL + pgvector store.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import asyncpg
import structlog

from author_library.errors import StorageError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from author_library.config import DatabaseSettings

log = structlog.get_logger(__name__)


class PostgresPool:
    """Async connection pool for PostgreSQL with pgvector support."""

    def __init__(
        self, settings: DatabaseSettings, *, min_size: int = 2, max_size: int = 10
    ) -> None:
        self._dsn = settings.postgres_url
        self._min_size = min_size
        self._max_size = max_size
        self._pool: asyncpg.Pool[asyncpg.Record] | None = None

    @property
    def pool(self) -> asyncpg.Pool[asyncpg.Record]:
        """Return the connection pool, raising if not initialized."""
        if self._pool is None:
            raise StorageError("PostgreSQL pool not initialized — call connect() first")
        return self._pool

    async def connect(self) -> None:
        """Create the connection pool and initialize pgvector extension."""
        if self._pool is not None:
            return
        try:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=self._min_size,
                max_size=self._max_size,
            )
            # Initialize pgvector extension
            async with self.pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            log.info(
                "pg_pool_connected",
                dsn=self._dsn.split("@")[-1],  # Log only host/db, not creds
                min_size=self._min_size,
                max_size=self._max_size,
            )
        except Exception as exc:
            raise StorageError(
                "Failed to connect to PostgreSQL",
                context={"dsn": self._dsn.split("@")[-1]},
                cause=exc,
            ) from exc

    async def close(self) -> None:
        """Gracefully close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            log.info("pg_pool_closed")

    async def health_check(self) -> bool:
        """Verify connectivity with SELECT 1."""
        try:
            result = await self.fetch_val("SELECT 1")
            return bool(result == 1)
        except Exception:
            log.warning("pg_health_check_failed", exc_info=True)
            return False

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection[asyncpg.Record]]:
        """Provide a transactional connection context manager."""
        async with self.pool.acquire() as conn, conn.transaction():
            yield conn

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a query and return the status string."""
        async with self.pool.acquire() as conn:
            result: str = await conn.execute(query, *args)
            return result

    async def fetch_all(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """Execute a query and return all rows."""
        async with self.pool.acquire() as conn:
            rows: list[asyncpg.Record] = await conn.fetch(query, *args)
            return rows

    async def fetch_one(self, query: str, *args: Any) -> asyncpg.Record | None:
        """Execute a query and return a single row (or None)."""
        async with self.pool.acquire() as conn:
            row: asyncpg.Record | None = await conn.fetchrow(query, *args)
            return row

    async def fetch_val(self, query: str, *args: Any, column: int = 0) -> Any:
        """Execute a query and return a single value from the first row."""
        async with self.pool.acquire() as conn:
            value: Any = await conn.fetchval(query, *args, column=column)
            return value
