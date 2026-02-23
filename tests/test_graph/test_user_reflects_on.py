"""Tests for USER_REFLECTS_ON edge (A2).

Covers:
- GraphRepository abstract interface for create_user_reflects_on_edge
- GraphRepository abstract interface for get_reflections_for_target
- Neo4jGraphRepository implementation (integration, requires Neo4j)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from author_library.chunking.models import ChunkGranularity
from author_library.storage.repositories import GraphRepository, Neo4jGraphRepository

if TYPE_CHECKING:
    from author_library.storage.neo4j import Neo4jConnection

requires_neo4j = pytest.mark.skipif(
    os.environ.get("SKIP_NEO4J", "0") == "1",
    reason="Neo4j not available (SKIP_NEO4J=1)",
)


# ---------------------------------------------------------------------------
# Interface tests (ensure abstract methods exist with correct signatures)
# ---------------------------------------------------------------------------


class TestGraphRepositoryInterface:
    def test_create_user_reflects_on_edge_is_abstract(self) -> None:
        """GraphRepository must declare create_user_reflects_on_edge."""
        assert hasattr(GraphRepository, "create_user_reflects_on_edge")
        # Verify it's in the abstract methods
        abstract_methods = getattr(GraphRepository, "__abstractmethods__", set())
        assert "create_user_reflects_on_edge" in abstract_methods

    def test_get_reflections_for_target_is_abstract(self) -> None:
        """GraphRepository must declare get_reflections_for_target."""
        assert hasattr(GraphRepository, "get_reflections_for_target")
        abstract_methods = getattr(GraphRepository, "__abstractmethods__", set())
        assert "get_reflections_for_target" in abstract_methods


# ---------------------------------------------------------------------------
# Neo4j integration tests (requires running Neo4j)
# ---------------------------------------------------------------------------


@requires_neo4j
class TestUserReflectsOnEdge:
    """Integration tests for USER_REFLECTS_ON edge via Neo4j."""

    async def _setup_nodes(
        self, neo4j_conn: Neo4jConnection, graph_repo: Neo4jGraphRepository
    ) -> dict[str, str]:
        """Create test nodes and return their IDs."""
        # Create a personal reflection chunk node
        await graph_repo.upsert_chunk_node({
            "chunk_id": "personal-reflection-001",
            "work_id": "marty--reflection-on-guite",
            "text_preview": "My thoughts on Guite's reading of Coleridge",
            "granularity": "meso",
            "source_class": "personal",
            "user_id": "marty",
        })

        # Create a target chunk node (the thing being reflected upon)
        await graph_repo.upsert_chunk_node({
            "chunk_id": "primary-chunk-001",
            "work_id": "guite--faith-hope-poetry",
            "text_preview": "The Primary Imagination is a living power",
            "granularity": "meso",
            "source_class": "primary",
        })

        return {
            "reflection_id": "personal-reflection-001",
            "target_id": "primary-chunk-001",
        }

    async def test_create_edge(self, neo4j_conn: Neo4jConnection) -> None:
        graph_repo = Neo4jGraphRepository(neo4j_conn)
        ids = await self._setup_nodes(neo4j_conn, graph_repo)

        await graph_repo.create_user_reflects_on_edge(
            reflection_chunk_id=ids["reflection_id"],
            target_id=ids["target_id"],
            target_type="capture",
            target_label="Chunk",
            target_key="chunk_id",
            date_created="2026-02-23",
        )

        # Verify edge exists
        results = await neo4j_conn.execute_read(
            """MATCH (c:Chunk {chunk_id: $chunk_id})-[r:USER_REFLECTS_ON]->(t:Chunk)
            RETURN r.target_type AS target_type, r.date_created AS date_created""",
            {"chunk_id": ids["reflection_id"]},
        )
        assert len(results) == 1
        assert results[0]["target_type"] == "capture"
        assert results[0]["date_created"] == "2026-02-23"

    async def test_create_edge_without_date(self, neo4j_conn: Neo4jConnection) -> None:
        graph_repo = Neo4jGraphRepository(neo4j_conn)
        ids = await self._setup_nodes(neo4j_conn, graph_repo)

        await graph_repo.create_user_reflects_on_edge(
            reflection_chunk_id=ids["reflection_id"],
            target_id=ids["target_id"],
            target_type="capture",
            target_label="Chunk",
            target_key="chunk_id",
        )

        results = await neo4j_conn.execute_read(
            """MATCH (c:Chunk {chunk_id: $chunk_id})-[r:USER_REFLECTS_ON]->(t:Chunk)
            RETURN r.target_type AS target_type""",
            {"chunk_id": ids["reflection_id"]},
        )
        assert len(results) == 1
        assert results[0]["target_type"] == "capture"

    async def test_get_reflections_for_target(
        self, neo4j_conn: Neo4jConnection
    ) -> None:
        graph_repo = Neo4jGraphRepository(neo4j_conn)
        ids = await self._setup_nodes(neo4j_conn, graph_repo)

        # Create the edge
        await graph_repo.create_user_reflects_on_edge(
            reflection_chunk_id=ids["reflection_id"],
            target_id=ids["target_id"],
            target_type="capture",
            target_label="Chunk",
            target_key="chunk_id",
            date_created="2026-02-23",
        )

        # Retrieve reflections
        reflections = await graph_repo.get_reflections_for_target(
            target_id=ids["target_id"],
            target_key="chunk_id",
            target_label="Chunk",
        )
        assert len(reflections) == 1
        assert reflections[0]["chunk_id"] == ids["reflection_id"]
        assert reflections[0]["source_class"] == "personal"
        assert reflections[0]["user_id"] == "marty"
        assert reflections[0]["target_type"] == "capture"

    async def test_get_reflections_empty(self, neo4j_conn: Neo4jConnection) -> None:
        graph_repo = Neo4jGraphRepository(neo4j_conn)

        # Create a chunk with no reflections
        await graph_repo.upsert_chunk_node({
            "chunk_id": "lonely-chunk",
            "work_id": "guite--sounding-seasons",
            "text_preview": "A sonnet about Advent",
            "granularity": "meso",
            "source_class": "primary",
        })

        reflections = await graph_repo.get_reflections_for_target(
            target_id="lonely-chunk",
            target_key="chunk_id",
            target_label="Chunk",
        )
        assert reflections == []

    async def test_personal_chunk_node_has_user_id(
        self, neo4j_conn: Neo4jConnection
    ) -> None:
        """Verify that personal chunk nodes store user_id property."""
        graph_repo = Neo4jGraphRepository(neo4j_conn)

        await graph_repo.upsert_chunk_node({
            "chunk_id": "personal-with-uid",
            "work_id": "marty--journal",
            "text_preview": "My daily reflection",
            "granularity": "meso",
            "source_class": "personal",
            "user_id": "marty",
        })

        results = await neo4j_conn.execute_read(
            """MATCH (c:Chunk {chunk_id: $chunk_id})
            RETURN c.user_id AS user_id, c.source_class AS source_class""",
            {"chunk_id": "personal-with-uid"},
        )
        assert len(results) == 1
        assert results[0]["user_id"] == "marty"
        assert results[0]["source_class"] == "personal"
