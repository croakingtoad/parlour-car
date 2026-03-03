"""Shared fixtures for integration tests.

Integration tests require real PostgreSQL + Neo4j (via Docker compose).
Tests are skipped if databases are not available.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio

from author_library.config import DatabaseSettings, Settings
from author_library.storage.manager import StorageManager

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import AsyncIterator


def _db_available() -> bool:
    """Check if Docker-based databases are reachable."""
    import socket

    for host, port in [("localhost", 5432), ("localhost", 7687)]:
        try:
            with socket.create_connection((host, port), timeout=2):
                pass
        except OSError:
            return False
    return True


SKIP_NO_DB = pytest.mark.skipif(
    not _db_available(),
    reason="PostgreSQL or Neo4j not available (run `make dev`)",
)

SKIP_NO_ANTHROPIC = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop for async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def integration_settings() -> Settings:
    """Settings configured for integration tests.

    DB_POSTGRES_URL is forced to the test database by tests/conftest.py
    so these settings NEVER point at the production database.
    """
    return Settings()


@pytest_asyncio.fixture
async def storage(integration_settings: Settings) -> AsyncIterator[StorageManager]:
    """Connected storage manager for integration tests.

    Runs migrations and initializes Neo4j schema on connect.
    """
    mgr = StorageManager(integration_settings.database)
    await mgr.connect(run_pg_migrations=True, init_neo4j_schema=True)
    yield mgr
    await mgr.close()


@pytest_asyncio.fixture
async def clean_storage(storage: StorageManager) -> AsyncIterator[StorageManager]:
    """Storage manager with test data cleaned before and after each test.

    Deletes all data from works-dependent tables, then works, then authors.
    """
    await _clean_all(storage)
    yield storage
    await _clean_all(storage)


async def _clean_all(storage: StorageManager) -> None:
    """Remove all test data from PG and Neo4j."""
    # PG: delete in dependency order
    await storage.pg.execute("DELETE FROM chunk_embeddings")
    await storage.pg.execute("DELETE FROM thematic_appearances")
    await storage.pg.execute("DELETE FROM thematic_entries")
    await storage.pg.execute("DELETE FROM voice_profiles")
    await storage.pg.execute("DELETE FROM chunks")
    await storage.pg.execute("DELETE FROM works")
    await storage.pg.execute("DELETE FROM authors")

    # Neo4j: clear all nodes and edges
    await storage.neo4j.execute_write("MATCH (n) DETACH DELETE n")
