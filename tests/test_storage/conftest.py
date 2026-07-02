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
    """Database settings — reads from env vars set by root conftest.py.

    DB_POSTGRES_URL is forced to the test database by tests/conftest.py
    so these settings NEVER point at the production database.
    """
    return DatabaseSettings()


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
        "ingestion_lessons",
        "transcript_cache",
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
    """Clean up Neo4j test data after each test.

    IMPORTANT: Only deletes nodes with work_id starting with 'test--' to avoid
    wiping production data. The previous implementation used MATCH (n) DETACH
    DELETE n which destroyed ALL production graph data every test run.
    """
    yield
    with contextlib.suppress(Exception):
        await neo4j_conn.execute_write(
            "MATCH (c:Chunk) WHERE c.work_id STARTS WITH 'test--' DETACH DELETE c", {}
        )
        await neo4j_conn.execute_write(
            "MATCH (w:Work) WHERE w.work_id STARTS WITH 'test--' DETACH DELETE w", {}
        )
        # Clean test entity nodes BY NAME PREFIX, never by orphan heuristic:
        # production vocabulary nodes can legitimately have zero edges (e.g.
        # restored themes awaiting extraction backfill) — an orphan sweep
        # deletes them (td-aef7c5, 2026-07-02 incident).
        for label in ("Theme", "Person", "Concept", "Argument", "Author"):
            await neo4j_conn.execute_write(
                f"MATCH (n:{label}) WHERE n.canonical_name STARTS WITH 'test--' "
                "DETACH DELETE n",
                {},
            )
        await neo4j_conn.execute_write(
            "MATCH (a:Author {author_id: 'test-author'}) DETACH DELETE a", {}
        )
        # TestNode label is used only by connection smoke tests
        await neo4j_conn.execute_write("MATCH (t:TestNode) DETACH DELETE t", {})
