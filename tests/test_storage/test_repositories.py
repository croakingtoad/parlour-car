"""Tests for repository implementations against real databases."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from author_library.storage.migrations.runner import run_migrations
from author_library.storage.repositories import (
    Neo4jGraphRepository,
    PgChunkRepository,
    PgEmbeddingRepository,
    PgThematicRepository,
    PgVoiceProfileRepository,
    PgWorkRepository,
)

if TYPE_CHECKING:
    from author_library.storage.manager import StorageManager
    from author_library.storage.neo4j import Neo4jConnection
    from author_library.storage.postgres import PostgresPool


# -- Helpers -----------------------------------------------------------------

SAMPLE_AUTHOR: dict[str, str] = {
    "id": "malcolm-guite",
    "canonical_name": "Malcolm Guite",
}

SAMPLE_WORK: dict[str, Any] = {
    "work_id": "malcolm-guite--faith-hope-and-poetry",
    "title": "Faith, Hope and Poetry",
    "author": "malcolm-guite",
    "source_class": "primary",
    "source_class_note": (
        "Authored by Malcolm Guite, the subject author of this collection"
    ),
    "publication_year": 2010,
    "publisher": "Ashgate Publishing",
    "format_ingested": "epub",
    "word_count": 85000,
    "genre_tags": ["literary criticism", "theology"],
    "subject_headings": ["poetry", "imagination", "theology"],
    "source_metadata": {},
}

SAMPLE_CHUNK: dict[str, Any] = {
    "work_id": "malcolm-guite--faith-hope-and-poetry",
    "text": (
        "The imagination is not merely a faculty for producing images; "
        "it is the living power and prime agent of all human perception."
    ),
    "granularity": "meso",
    "source_class": "primary",
    "position": 1,
}


async def _insert_author(pool: PostgresPool) -> None:
    """Insert the sample author for FK references."""
    await pool.execute(
        "INSERT INTO authors (id, canonical_name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        SAMPLE_AUTHOR["id"],
        SAMPLE_AUTHOR["canonical_name"],
    )


# -- Work Repository Tests ---------------------------------------------------


async def test_work_crud(pg_pool: PostgresPool) -> None:
    """Full CRUD cycle for works."""
    await run_migrations(pg_pool)
    await _insert_author(pg_pool)
    repo = PgWorkRepository(pg_pool)

    # Create
    work_id = await repo.create(SAMPLE_WORK)
    assert work_id == SAMPLE_WORK["work_id"]

    # Read
    work = await repo.get(work_id)
    assert work is not None
    assert work["title"] == "Faith, Hope and Poetry"
    assert work["source_class"] == "primary"

    # List by author
    works = await repo.list_by_author("malcolm-guite")
    assert len(works) == 1

    # Update
    updated = await repo.update(work_id, {"title": "Faith, Hope and Poetry (2nd ed.)"})
    assert updated is True
    work = await repo.get(work_id)
    assert work is not None
    assert work["title"] == "Faith, Hope and Poetry (2nd ed.)"

    # Delete
    deleted = await repo.delete(work_id)
    assert deleted is True
    assert await repo.get(work_id) is None


# -- Chunk Repository Tests --------------------------------------------------


async def test_chunk_crud(pg_pool: PostgresPool) -> None:
    """Full CRUD cycle for chunks."""
    await run_migrations(pg_pool)
    await _insert_author(pg_pool)
    work_repo = PgWorkRepository(pg_pool)
    await work_repo.create(SAMPLE_WORK)

    chunk_repo = PgChunkRepository(pg_pool)

    # Create
    chunk_id = await chunk_repo.create(SAMPLE_CHUNK)
    assert isinstance(chunk_id, uuid.UUID)

    # Read
    chunk = await chunk_repo.get(chunk_id)
    assert chunk is not None
    assert chunk["granularity"] == "meso"

    # List by work
    chunks = await chunk_repo.list_by_work(SAMPLE_WORK["work_id"])
    assert len(chunks) == 1

    # List by work + granularity filter
    chunks = await chunk_repo.list_by_work(SAMPLE_WORK["work_id"], granularity="macro")
    assert len(chunks) == 0

    # Delete
    deleted = await chunk_repo.delete(chunk_id)
    assert deleted is True

    # Delete by work
    await chunk_repo.create(SAMPLE_CHUNK)
    count = await chunk_repo.delete_by_work(SAMPLE_WORK["work_id"])
    assert count == 1


# -- Embedding Repository Tests -----------------------------------------------


async def test_embedding_store_and_search(pg_pool: PostgresPool) -> None:
    """Store embeddings and run similarity search."""
    await run_migrations(pg_pool)
    await _insert_author(pg_pool)
    work_repo = PgWorkRepository(pg_pool)
    await work_repo.create(SAMPLE_WORK)

    chunk_repo = PgChunkRepository(pg_pool)
    chunk_id = await chunk_repo.create(SAMPLE_CHUNK)

    emb_repo = PgEmbeddingRepository(pg_pool)

    # Store a 1024-dim embedding
    test_embedding = [0.01 * i for i in range(1024)]
    emb_id = await emb_repo.store(
        chunk_id, test_embedding, "voyage", "voyage-3-large", 1024
    )
    assert isinstance(emb_id, uuid.UUID)

    # Retrieve by chunk
    embeddings = await emb_repo.get_by_chunk(chunk_id)
    assert len(embeddings) == 1
    assert embeddings[0]["provider"] == "voyage"

    # Similarity search
    query_emb = [0.01 * i + 0.001 for i in range(1024)]  # slightly different
    results = await emb_repo.similarity_search(
        query_emb, provider="voyage", model="voyage-3-large", limit=5
    )
    assert len(results) == 1
    assert results[0]["chunk_id"] == chunk_id


# -- Thematic Repository Tests ------------------------------------------------


async def test_thematic_crud(pg_pool: PostgresPool) -> None:
    """Full CRUD for thematic entries and appearances."""
    await run_migrations(pg_pool)
    await _insert_author(pg_pool)
    work_repo = PgWorkRepository(pg_pool)
    await work_repo.create(SAMPLE_WORK)

    thematic = PgThematicRepository(pg_pool)

    # Create entry
    entry_id = await thematic.create_entry({
        "author_id": "malcolm-guite",
        "theme": "Imagination and Perception",
        "author_stance": "Imagination is the primary faculty of perception",
        "related_themes": ["Coleridge", "Romanticism"],
    })
    assert isinstance(entry_id, uuid.UUID)

    # Get entry
    entry = await thematic.get_entry(entry_id)
    assert entry is not None
    assert entry["theme"] == "Imagination and Perception"

    # List entries
    entries = await thematic.list_entries("malcolm-guite")
    assert len(entries) == 1

    # Add appearance
    app_id = await thematic.add_appearance({
        "entry_id": entry_id,
        "work_id": SAMPLE_WORK["work_id"],
        "chapters": ["Chapter 3", "Chapter 7"],
        "treatment_summary": "Explored through Coleridge's lens",
    })
    assert isinstance(app_id, uuid.UUID)

    # Delete entry (cascades to appearances)
    deleted = await thematic.delete_entry(entry_id)
    assert deleted is True


# -- Voice Profile Repository Tests -------------------------------------------


async def test_voice_profile_store_and_get(pg_pool: PostgresPool) -> None:
    """Store and retrieve voice profiles."""
    await run_migrations(pg_pool)
    await _insert_author(pg_pool)

    profiles = PgVoiceProfileRepository(pg_pool)

    # Store version 1
    profile_data: dict[str, Any] = {
        "register": "scholarly yet accessible",
        "sentence_patterns": [
            "complex periodic sentences",
            "rhetorical questions",
        ],
        "vocabulary_level": "elevated",
    }
    v1_id = await profiles.store("malcolm-guite", profile_data, version=1)
    assert isinstance(v1_id, uuid.UUID)

    # Get current
    current = await profiles.get_current("malcolm-guite")
    assert current is not None
    assert current["version"] == 1
    assert current["is_current"] is True

    # Store version 2 — should supersede v1
    v2_data = {**profile_data, "register": "more conversational"}
    await profiles.store("malcolm-guite", v2_data, version=2)

    current = await profiles.get_current("malcolm-guite")
    assert current is not None
    assert current["version"] == 2

    # List all versions
    versions = await profiles.list_versions("malcolm-guite")
    assert len(versions) == 2


# -- Graph Repository Tests ---------------------------------------------------


async def test_graph_work_and_chunk_nodes(neo4j_conn: Neo4jConnection) -> None:
    """Upsert work and chunk nodes in Neo4j."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    await graph.upsert_work_node({
        "work_id": "malcolm-guite--faith-hope-and-poetry",
        "title": "Faith, Hope and Poetry",
        "author": "malcolm-guite",
        "source_class": "primary",
        "publication_year": 2010,
    })

    await graph.upsert_chunk_node({
        "chunk_id": "chunk-001",
        "work_id": "malcolm-guite--faith-hope-and-poetry",
        "text_preview": "The imagination is not merely a faculty...",
        "granularity": "meso",
        "source_class": "primary",
    })

    # Verify nodes exist
    works = await neo4j_conn.execute_read(
        "MATCH (w:Work) RETURN w.work_id AS wid", {}
    )
    assert len(works) == 1
    assert works[0]["wid"] == "malcolm-guite--faith-hope-and-poetry"


async def test_graph_create_edge_and_query(neo4j_conn: Neo4jConnection) -> None:
    """Create edges between nodes and query relationships."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    # Create two chunks
    for cid in ("chunk-a", "chunk-b"):
        await graph.upsert_chunk_node({
            "chunk_id": cid,
            "work_id": "work-1",
            "text_preview": f"Text for {cid}",
            "granularity": "meso",
            "source_class": "primary",
        })

    # Create a theme
    await neo4j_conn.execute_write(
        "MERGE (t:Theme {name: 'Imagination', canonical_name: 'imagination'})",
        {},
    )

    # Create edges
    await graph.create_edge(
        "Chunk", "chunk_id", "chunk-a",
        "ENGAGES_WITH",
        "Chunk", "chunk_id", "chunk-b",
        {
            "link_type": "explicit_citation",
            "confidence": 0.95,
            "evidence": "direct quote",
        },
    )

    await graph.create_edge(
        "Chunk", "chunk_id", "chunk-a",
        "EXPLORES_THEME",
        "Theme", "canonical_name", "imagination",
    )

    # Query related chunks
    related = await graph.get_related_chunks("chunk-a", "ENGAGES_WITH")
    assert len(related) == 1
    assert related[0]["chunk_id"] == "chunk-b"
    assert related[0]["rel_props"]["confidence"] == 0.95

    # Query themes
    themes = await graph.get_themes_for_chunk("chunk-a")
    assert len(themes) == 1
    assert themes[0]["canonical_name"] == "imagination"


async def test_chunk_creates_part_of_edge_to_work(neo4j_conn: Neo4jConnection) -> None:
    """Upserting a chunk node creates a PART_OF edge to its Work node."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    # Create work node first (as ingestion pipeline does)
    await graph.upsert_work_node({
        "work_id": "malcolm-guite--faith-hope-and-poetry",
        "title": "Faith, Hope and Poetry",
        "author": "malcolm-guite",
        "source_class": "primary",
        "publication_year": 2010,
    })

    # Create chunk node — should automatically create PART_OF edge
    await graph.upsert_chunk_node({
        "chunk_id": "chunk-001",
        "work_id": "malcolm-guite--faith-hope-and-poetry",
        "text_preview": "The imagination is not merely a faculty...",
        "granularity": "meso",
        "source_class": "primary",
    })

    # Verify PART_OF edge exists
    results = await neo4j_conn.execute_read(
        """MATCH (c:Chunk {chunk_id: $chunk_id})-[:PART_OF]->(w:Work)
        RETURN w.work_id AS work_id, w.title AS title""",
        {"chunk_id": "chunk-001"},
    )
    assert len(results) == 1
    assert results[0]["work_id"] == "malcolm-guite--faith-hope-and-poetry"
    assert results[0]["title"] == "Faith, Hope and Poetry"


async def test_all_chunks_connected_to_work_via_part_of(neo4j_conn: Neo4jConnection) -> None:
    """All chunks for a work are connected via PART_OF edges."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    work_id = "malcolm-guite--faith-hope-and-poetry"
    await graph.upsert_work_node({
        "work_id": work_id,
        "title": "Faith, Hope and Poetry",
        "author": "malcolm-guite",
        "source_class": "primary",
        "publication_year": 2010,
    })

    # Create multiple chunks
    chunk_ids = [f"chunk-{i:03d}" for i in range(1, 6)]
    for cid in chunk_ids:
        await graph.upsert_chunk_node({
            "chunk_id": cid,
            "work_id": work_id,
            "text_preview": f"Text for {cid}",
            "granularity": "meso",
            "source_class": "primary",
        })

    # Verify all chunks are connected to the work
    results = await neo4j_conn.execute_read(
        """MATCH (c:Chunk)-[:PART_OF]->(w:Work {work_id: $work_id})
        RETURN count(c) AS chunk_count""",
        {"work_id": work_id},
    )
    assert results[0]["chunk_count"] == 5

    # Verify the work node is NOT disconnected
    disconnected = await neo4j_conn.execute_read(
        """MATCH (w:Work {work_id: $work_id})
        WHERE NOT (w)--()
        RETURN count(w) AS count""",
        {"work_id": work_id},
    )
    assert disconnected[0]["count"] == 0


async def test_work_node_has_correct_properties(neo4j_conn: Neo4jConnection) -> None:
    """Work node is created with all expected properties."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    await graph.upsert_work_node({
        "work_id": "malcolm-guite--faith-hope-and-poetry",
        "title": "Faith, Hope and Poetry",
        "author": "malcolm-guite",
        "source_class": "primary",
        "publication_year": 2010,
    })

    results = await neo4j_conn.execute_read(
        """MATCH (w:Work {work_id: $work_id})
        RETURN w.title AS title, w.author AS author,
               w.source_class AS source_class,
               w.publication_year AS publication_year""",
        {"work_id": "malcolm-guite--faith-hope-and-poetry"},
    )
    assert len(results) == 1
    assert results[0]["title"] == "Faith, Hope and Poetry"
    assert results[0]["author"] == "malcolm-guite"
    assert results[0]["source_class"] == "primary"
    assert results[0]["publication_year"] == 2010


async def test_chunk_without_work_node_still_created(neo4j_conn: Neo4jConnection) -> None:
    """Chunk node is created even when Work node doesn't exist (no PART_OF edge)."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    # Create chunk WITHOUT creating work node first
    await graph.upsert_chunk_node({
        "chunk_id": "orphan-chunk",
        "work_id": "nonexistent-work",
        "text_preview": "Some text",
        "granularity": "meso",
        "source_class": "primary",
    })

    # Chunk node should exist
    chunks = await neo4j_conn.execute_read(
        "MATCH (c:Chunk {chunk_id: $cid}) RETURN c.chunk_id AS cid",
        {"cid": "orphan-chunk"},
    )
    assert len(chunks) == 1

    # But no PART_OF edge (work doesn't exist)
    edges = await neo4j_conn.execute_read(
        "MATCH (c:Chunk {chunk_id: $cid})-[:PART_OF]->() RETURN count(*) AS cnt",
        {"cid": "orphan-chunk"},
    )
    assert edges[0]["cnt"] == 0


async def test_part_of_is_idempotent(neo4j_conn: Neo4jConnection) -> None:
    """Upserting the same chunk twice doesn't create duplicate PART_OF edges."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    await graph.upsert_work_node({
        "work_id": "work-1",
        "title": "Test Work",
        "author": "test-author",
        "source_class": "primary",
        "publication_year": 2020,
    })

    chunk_data = {
        "chunk_id": "chunk-dup",
        "work_id": "work-1",
        "text_preview": "Some text",
        "granularity": "meso",
        "source_class": "primary",
    }

    # Upsert twice
    await graph.upsert_chunk_node(chunk_data)
    await graph.upsert_chunk_node(chunk_data)

    # Should have exactly one PART_OF edge
    results = await neo4j_conn.execute_read(
        """MATCH (c:Chunk {chunk_id: $cid})-[r:PART_OF]->(w:Work)
        RETURN count(r) AS edge_count""",
        {"cid": "chunk-dup"},
    )
    assert results[0]["edge_count"] == 1


# -- StorageManager Tests ----------------------------------------------------


async def test_storage_manager_lifecycle(storage: StorageManager) -> None:
    """StorageManager connects, reports health, and closes."""
    health = await storage.health_check()
    assert health["postgres"] is True
    assert health["neo4j"] is True
