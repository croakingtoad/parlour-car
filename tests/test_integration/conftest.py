"""Shared fixtures for integration tests.

Integration tests require real PostgreSQL + Neo4j (via Docker compose).
Tests are skipped if databases are not available.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from author_library.config import Settings
from author_library.storage.manager import StorageManager
from tests.conftest import TEST_NAMESPACE

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import AsyncIterator


def _db_available() -> bool:
    """Check whether PostgreSQL and the configured disposable Neo4j are reachable."""
    import socket

    neo4j_url = os.environ.get("TEST_NEO4J_URL", "bolt://localhost:7688")
    parsed_neo4j_url = urlparse(neo4j_url)
    neo4j_host = parsed_neo4j_url.hostname or "localhost"
    neo4j_port = parsed_neo4j_url.port or 7687

    for host, port in [("localhost", 5432), (neo4j_host, neo4j_port)]:
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

SKIP_NO_VOYAGE = pytest.mark.skipif(
    not os.environ.get("VOYAGE_API_KEY"),
    reason="VOYAGE_API_KEY not set (source .env)",
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
async def storage(
    integration_settings: Settings,
    assert_graph_is_disposable: None,
) -> AsyncIterator[StorageManager]:
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
    await storage.pg.execute("DELETE FROM acquisition_candidates")
    await storage.pg.execute("DELETE FROM chunks")
    await storage.pg.execute("DELETE FROM works")
    await storage.pg.execute("DELETE FROM authors")
    await storage.pg.execute("DELETE FROM ingestion_lessons")

    # Neo4j: scoped cleanup — only delete test-created nodes.
    #
    # IMPORTANT: every pattern here MUST be scoped to the test-- namespace.
    #   1. Never use unscoped MATCH (n) DETACH DELETE n.
    #   2. Never list a real author's prefix. The sibling fixture in
    #      test_intelligence carried the production Guite prefix and deleted
    #      all 5 production Guite works (3,495 chunks) on 2026-08-13.
    #   3. Never orphan-sweep (MATCH (n:Label) WHERE NOT (n)--()). It
    #      deletes unreferenced PRODUCTION entities; that is how three real
    #      Author nodes were lost.
    await storage.neo4j.execute_write(
        "MATCH (c:Chunk) WHERE c.work_id STARTS WITH $prefix DETACH DELETE c",
        {"prefix": TEST_NAMESPACE},
    )
    await storage.neo4j.execute_write(
        "MATCH (w:Work) WHERE w.work_id STARTS WITH $prefix DETACH DELETE w",
        {"prefix": TEST_NAMESPACE},
    )
    for label in ("Theme", "Person", "Concept", "Argument", "Author"):
        await storage.neo4j.execute_write(
            f"MATCH (n:{label}) "
            "WHERE n.canonical_name STARTS WITH $prefix "
            "   OR n.author_id STARTS WITH $prefix "
            "DETACH DELETE n",
            {"prefix": TEST_NAMESPACE},
        )
