"""Tests for PostgreSQL connection pool management."""

from __future__ import annotations

import pytest

from author_library.config import DatabaseSettings
from author_library.errors import StorageError
from author_library.storage.postgres import PostgresPool


async def test_pool_connect_and_close(pg_pool: PostgresPool) -> None:
    """Pool connects and closes cleanly."""
    assert pg_pool._pool is not None
    await pg_pool.close()
    assert pg_pool._pool is None


async def test_pool_not_initialized_raises() -> None:
    """Accessing pool before connect() raises StorageError."""
    settings = DatabaseSettings()
    pool = PostgresPool(settings)
    with pytest.raises(StorageError, match="not initialized"):
        _ = pool.pool


async def test_health_check(pg_pool: PostgresPool) -> None:
    """Health check returns True when connected."""
    assert await pg_pool.health_check() is True


async def test_execute_and_fetch(pg_pool: PostgresPool) -> None:
    """Basic execute/fetch operations work."""
    val = await pg_pool.fetch_val("SELECT 42")
    assert val == 42

    rows = await pg_pool.fetch_all("SELECT generate_series(1, 3) AS n")
    assert len(rows) == 3
    assert [r["n"] for r in rows] == [1, 2, 3]


async def test_fetch_one(pg_pool: PostgresPool) -> None:
    """fetch_one returns a single row or None."""
    row = await pg_pool.fetch_one("SELECT 1 AS x")
    assert row is not None
    assert row["x"] == 1

    row = await pg_pool.fetch_one("SELECT 1 WHERE FALSE")
    assert row is None


async def test_transaction_commit(pg_pool: PostgresPool) -> None:
    """Transaction context manager commits on success."""
    await pg_pool.execute(
        "CREATE TABLE IF NOT EXISTS _test_tx (id SERIAL PRIMARY KEY, val TEXT)"
    )
    try:
        async with pg_pool.transaction() as conn:
            await conn.execute("INSERT INTO _test_tx (val) VALUES ($1)", "hello")
        row = await pg_pool.fetch_one("SELECT val FROM _test_tx WHERE val = $1", "hello")
        assert row is not None
        assert row["val"] == "hello"
    finally:
        await pg_pool.execute("DROP TABLE IF EXISTS _test_tx")


async def test_transaction_rollback(pg_pool: PostgresPool) -> None:
    """Transaction rolls back on exception."""
    await pg_pool.execute(
        "CREATE TABLE IF NOT EXISTS _test_tx (id SERIAL PRIMARY KEY, val TEXT)"
    )
    try:
        with pytest.raises(ValueError):
            async with pg_pool.transaction() as conn:
                await conn.execute("INSERT INTO _test_tx (val) VALUES ($1)", "rollme")
                raise ValueError("force rollback")
        row = await pg_pool.fetch_one("SELECT val FROM _test_tx WHERE val = $1", "rollme")
        assert row is None
    finally:
        await pg_pool.execute("DROP TABLE IF EXISTS _test_tx")


async def test_pgvector_extension(pg_pool: PostgresPool) -> None:
    """pgvector extension is available after connect."""
    row = await pg_pool.fetch_one(
        "SELECT extname FROM pg_extension WHERE extname = 'vector'"
    )
    assert row is not None
    assert row["extname"] == "vector"


async def test_connect_bad_dsn() -> None:
    """Connecting with bad DSN raises StorageError."""
    settings = DatabaseSettings(postgres_url="postgresql://bad:bad@localhost:19999/nope")
    pool = PostgresPool(settings)
    with pytest.raises(StorageError, match="Failed to connect"):
        await pool.connect()
