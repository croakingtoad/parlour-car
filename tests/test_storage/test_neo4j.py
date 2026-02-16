"""Tests for Neo4j connection and schema management."""

from __future__ import annotations

import pytest

from author_library.config import DatabaseSettings
from author_library.errors import StorageError
from author_library.storage.neo4j import Neo4jConnection


async def test_neo4j_connect_and_close(neo4j_conn: Neo4jConnection) -> None:
    """Neo4j connects and closes cleanly."""
    assert neo4j_conn._driver is not None
    await neo4j_conn.close()
    assert neo4j_conn._driver is None


async def test_neo4j_not_initialized_raises() -> None:
    """Accessing driver before connect() raises StorageError."""
    settings = DatabaseSettings()
    conn = Neo4jConnection(settings)
    with pytest.raises(StorageError, match="not initialized"):
        _ = conn.driver


async def test_health_check(neo4j_conn: Neo4jConnection) -> None:
    """Health check returns True when connected."""
    assert await neo4j_conn.health_check() is True


async def test_init_schema(neo4j_conn: Neo4jConnection) -> None:
    """Schema initialization creates constraints and indexes."""
    await neo4j_conn.init_schema()
    # Verify a constraint exists
    results = await neo4j_conn.execute_read("SHOW CONSTRAINTS", {})
    constraint_names = [r.get("name", "") for r in results]
    assert any("work_id_unique" in name for name in constraint_names)


async def test_execute_write_and_read(neo4j_conn: Neo4jConnection) -> None:
    """Write and read operations work."""
    await neo4j_conn.execute_write(
        "CREATE (t:TestNode {name: $name})", {"name": "test_value"}
    )
    results = await neo4j_conn.execute_read(
        "MATCH (t:TestNode {name: $name}) RETURN t.name AS name",
        {"name": "test_value"},
    )
    assert len(results) == 1
    assert results[0]["name"] == "test_value"


async def test_connect_bad_uri() -> None:
    """Connecting with bad URI raises StorageError."""
    settings = DatabaseSettings(neo4j_url="bolt://localhost:19999")
    conn = Neo4jConnection(settings)
    with pytest.raises(StorageError, match="Failed to connect"):
        await conn.connect()
