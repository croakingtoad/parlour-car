"""Shared fixtures for storage tests.

Tests connect to real PostgreSQL and Neo4j Docker containers.
Each test function gets a clean database state via transactional rollback
or explicit cleanup.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest

from author_library.config import DatabaseSettings
from author_library.storage.manager import StorageManager
from author_library.storage.neo4j import Neo4jConnection
from author_library.storage.postgres import PostgresPool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture(scope="session")
def db_settings() -> DatabaseSettings:
    """Real database settings pointing at Docker containers."""
    return DatabaseSettings(
        postgres_url="postgresql://author_library:author_library@localhost:5432/author_library",
        neo4j_url="bolt://localhost:7687",
        neo4j_user="neo4j",
    )


@pytest.fixture
async def pg_pool(db_settings: DatabaseSettings) -> AsyncIterator[PostgresPool]:
    """Provide a connected PostgreSQL pool, cleaned up after the test."""
    pool = PostgresPool(db_settings, min_size=1, max_size=3)
    await pool.connect()
    yield pool
    await pool.close()


@pytest.fixture
async def neo4j_conn(db_settings: DatabaseSettings) -> AsyncIterator[Neo4jConnection]:
    """Provide a connected Neo4j driver, cleaned up after the test."""
    conn = Neo4jConnection(db_settings)
    await conn.connect()
    yield conn
    await conn.close()


@pytest.fixture
async def storage(db_settings: DatabaseSettings) -> AsyncIterator[StorageManager]:
    """Provide a fully connected StorageManager with migrations applied."""
    mgr = StorageManager(db_settings, pg_min_pool=1, pg_max_pool=3)
    await mgr.connect(run_pg_migrations=True, init_neo4j_schema=True)
    yield mgr
    await mgr.close()


@pytest.fixture(autouse=True)
async def _cleanup_pg(pg_pool: PostgresPool) -> AsyncIterator[None]:
    """Clean up test data after each test. Runs after every test."""
    yield
    # Clean up in reverse dependency order
    for table in [
        "thematic_appearances",
        "thematic_entries",
        "voice_profiles",
        "chunk_embeddings",
        "chunks",
        "works",
        "authors",
    ]:
        with contextlib.suppress(Exception):
            await pg_pool.execute(f"DELETE FROM {table}")


@pytest.fixture(autouse=True)
async def _cleanup_neo4j(neo4j_conn: Neo4jConnection) -> AsyncIterator[None]:
    """Clean up Neo4j test data after each test."""
    yield
    with contextlib.suppress(Exception):
        await neo4j_conn.execute_write("MATCH (n) DETACH DELETE n", {})
