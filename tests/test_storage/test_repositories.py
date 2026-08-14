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
    "id": "test--guite",
    "canonical_name": "Test Guite",
}

SAMPLE_WORK: dict[str, Any] = {
    "work_id": "test--faith-hope-and-poetry",
    "title": "Faith, Hope and Poetry",
    "author": "test--guite",
    "source_class": "primary",
    "source_class_note": (
        "Authored by Test Guite, the subject author of this collection"
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
    "work_id": "test--faith-hope-and-poetry",
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
    works = await repo.list_by_author("test--guite")
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


# -- Annotation Roundtrip Tests -----------------------------------------------


async def test_chunk_annotation_persisted(pg_pool: PostgresPool) -> None:
    """Chunk annotation field survives storage and retrieval."""
    await run_migrations(pg_pool)
    await _insert_author(pg_pool)
    work_repo = PgWorkRepository(pg_pool)
    await work_repo.create(SAMPLE_WORK)

    chunk_repo = PgChunkRepository(pg_pool)

    annotated_chunk = {
        **SAMPLE_CHUNK,
        "annotation": (
            '[PRIMARY] From "Faith, Hope and Poetry" (2010) by Test Guite.\n'
            "This meso covers: the role of imagination in perception."
        ),
    }
    chunk_id = await chunk_repo.create(annotated_chunk)

    # Verify annotation survives get()
    chunk = await chunk_repo.get(chunk_id)
    assert chunk is not None
    assert chunk["annotation"] is not None
    assert "[PRIMARY]" in chunk["annotation"]
    assert "imagination" in chunk["annotation"]

    # Verify annotation survives list_by_work()
    chunks = await chunk_repo.list_by_work(SAMPLE_WORK["work_id"])
    assert len(chunks) == 1
    assert chunks[0]["annotation"] == annotated_chunk["annotation"]


async def test_similarity_search_returns_annotation(pg_pool: PostgresPool) -> None:
    """Similarity search results include the annotation field."""
    await run_migrations(pg_pool)
    await _insert_author(pg_pool)
    work_repo = PgWorkRepository(pg_pool)
    await work_repo.create(SAMPLE_WORK)

    chunk_repo = PgChunkRepository(pg_pool)
    annotation_text = (
        '[PRIMARY] From "Faith, Hope and Poetry" (2010) by Test Guite.\n'
        "This meso covers: the role of imagination in perception."
    )
    annotated_chunk = {**SAMPLE_CHUNK, "annotation": annotation_text}
    chunk_id = await chunk_repo.create(annotated_chunk)

    emb_repo = PgEmbeddingRepository(pg_pool)
    test_embedding = [0.01 * i for i in range(1024)]
    await emb_repo.store(chunk_id, test_embedding, "voyage", "voyage-3-large", 1024)

    # Similarity search should return annotation
    query_emb = [0.01 * i + 0.001 for i in range(1024)]
    results = await emb_repo.similarity_search(
        query_emb, provider="voyage", model="voyage-3-large", limit=5,
    )
    assert len(results) == 1
    assert results[0]["annotation"] == annotation_text


async def test_similarity_search_annotation_null_when_absent(pg_pool: PostgresPool) -> None:
    """Similarity search returns None annotation when chunk has no annotation."""
    await run_migrations(pg_pool)
    await _insert_author(pg_pool)
    work_repo = PgWorkRepository(pg_pool)
    await work_repo.create(SAMPLE_WORK)

    chunk_repo = PgChunkRepository(pg_pool)
    chunk_id = await chunk_repo.create(SAMPLE_CHUNK)  # no annotation

    emb_repo = PgEmbeddingRepository(pg_pool)
    test_embedding = [0.01 * i for i in range(1024)]
    await emb_repo.store(chunk_id, test_embedding, "voyage", "voyage-3-large", 1024)

    query_emb = [0.01 * i + 0.001 for i in range(1024)]
    results = await emb_repo.similarity_search(
        query_emb, provider="voyage", model="voyage-3-large", limit=5,
    )
    assert len(results) == 1
    assert results[0]["annotation"] is None


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
        "author_id": "test--guite",
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
    entries = await thematic.list_entries("test--guite")
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
    v1_id = await profiles.store("test--guite", profile_data, version=1)
    assert isinstance(v1_id, uuid.UUID)

    # Get current
    current = await profiles.get_current("test--guite")
    assert current is not None
    assert current["version"] == 1
    assert current["is_current"] is True

    # Store version 2 — should supersede v1
    v2_data = {**profile_data, "register": "more conversational"}
    await profiles.store("test--guite", v2_data, version=2)

    current = await profiles.get_current("test--guite")
    assert current is not None
    assert current["version"] == 2

    # List all versions
    versions = await profiles.list_versions("test--guite")
    assert len(versions) == 2


# -- Graph Repository Tests ---------------------------------------------------


async def test_graph_work_and_chunk_nodes(neo4j_conn: Neo4jConnection) -> None:
    """Upsert work and chunk nodes in Neo4j."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    # test-- prefix: NEVER a production work_id — upsert_work_node MERGEs by
    # work_id and would overwrite the production node's properties. The Neo4j
    # instance is shared with production (see parlour-test-isolation-patterns).
    await graph.upsert_work_node({
        "work_id": "test--faith-hope-and-poetry",
        "title": "Faith, Hope and Poetry",
        "author": "test-author",
        "source_class": "primary",
        "publication_year": 2010,
    })

    await graph.upsert_chunk_node({
        "chunk_id": "chunk-001",
        "work_id": "test--faith-hope-and-poetry",
        "text_preview": "The imagination is not merely a faculty...",
        "granularity": "meso",
        "source_class": "primary",
    })

    # Verify nodes exist (scoped — shared instance holds production works)
    works = await neo4j_conn.execute_read(
        "MATCH (w:Work {work_id: 'test--faith-hope-and-poetry'}) "
        "RETURN w.work_id AS wid",
        {},
    )
    assert len(works) == 1
    assert works[0]["wid"] == "test--faith-hope-and-poetry"


async def test_graph_create_edge_and_query(neo4j_conn: Neo4jConnection) -> None:
    """Create edges between nodes and query relationships."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    # Create two chunks
    for cid in ("chunk-a", "chunk-b"):
        await graph.upsert_chunk_node({
            "chunk_id": cid,
            "work_id": "test--work-1",
            "text_preview": f"Text for {cid}",
            "granularity": "meso",
            "source_class": "primary",
        })

    # Create a theme
    await neo4j_conn.execute_write(
        "MERGE (t:Theme {name: 'Imagination', canonical_name: 'test--imagination'})",
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
        "Theme", "canonical_name", "test--imagination",
    )

    # Query related chunks
    related = await graph.get_related_chunks("chunk-a", "ENGAGES_WITH")
    assert len(related) == 1
    assert related[0]["chunk_id"] == "chunk-b"
    assert related[0]["rel_props"]["confidence"] == 0.95

    # Query themes
    themes = await graph.get_themes_for_chunk("chunk-a")
    assert len(themes) == 1
    assert themes[0]["canonical_name"] == "test--imagination"


async def test_chunk_creates_part_of_edge_to_work(neo4j_conn: Neo4jConnection) -> None:
    """Upserting a chunk node creates a PART_OF edge to its Work node."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    # Create work node first (as ingestion pipeline does)
    await graph.upsert_work_node({
        "work_id": "test--faith-hope-and-poetry",
        "title": "Faith, Hope and Poetry",
        "author": "test--guite",
        "source_class": "primary",
        "publication_year": 2010,
    })

    # Create chunk node — should automatically create PART_OF edge
    await graph.upsert_chunk_node({
        "chunk_id": "chunk-001",
        "work_id": "test--faith-hope-and-poetry",
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
    assert results[0]["work_id"] == "test--faith-hope-and-poetry"
    assert results[0]["title"] == "Faith, Hope and Poetry"


async def test_all_chunks_connected_to_work_via_part_of(neo4j_conn: Neo4jConnection) -> None:
    """All chunks for a work are connected via PART_OF edges."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    work_id = "test--faith-hope-and-poetry"
    await graph.upsert_work_node({
        "work_id": work_id,
        "title": "Faith, Hope and Poetry",
        "author": "test--guite",
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
        "work_id": "test--faith-hope-and-poetry",
        "title": "Faith, Hope and Poetry",
        "author": "test--guite",
        "source_class": "primary",
        "publication_year": 2010,
    })

    results = await neo4j_conn.execute_read(
        """MATCH (w:Work {work_id: $work_id})
        RETURN w.title AS title, w.author AS author,
               w.source_class AS source_class,
               w.publication_year AS publication_year""",
        {"work_id": "test--faith-hope-and-poetry"},
    )
    assert len(results) == 1
    assert results[0]["title"] == "Faith, Hope and Poetry"
    assert results[0]["author"] == "test--guite"
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
        "work_id": "test--work-1",
        "title": "Test Work",
        "author": "test-author",
        "source_class": "primary",
        "publication_year": 2020,
    })

    chunk_data = {
        "chunk_id": "chunk-dup",
        "work_id": "test--work-1",
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


# -- get_passage_links_for_work Tests ----------------------------------------


async def test_get_passage_links_for_work_empty(neo4j_conn: Neo4jConnection) -> None:
    """Returns empty list when no passage links exist for a work."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    # Create a work and chunk with no passage links
    await graph.upsert_work_node({
        "work_id": "test--work-no-links",
        "title": "No Links Work",
        "author": "test-author",
        "source_class": "primary",
        "publication_year": 2020,
    })
    await graph.upsert_chunk_node({
        "chunk_id": "chunk-lonely",
        "work_id": "test--work-no-links",
        "text_preview": "A lonely chunk with no connections",
        "granularity": "meso",
        "source_class": "primary",
    })

    result = await graph.get_passage_links_for_work("test--work-no-links")
    assert result == []


async def test_get_passage_links_for_work_engages_with(neo4j_conn: Neo4jConnection) -> None:
    """Returns ENGAGES_WITH edges for chunks belonging to a work."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    # Create two works with chunks
    await graph.upsert_work_node({
        "work_id": "test--work-primary",
        "title": "Primary Work",
        "author": "test-author",
        "source_class": "primary",
        "publication_year": 2020,
    })
    await graph.upsert_work_node({
        "work_id": "test--work-contextual",
        "title": "Contextual Work",
        "author": "test-author",
        "source_class": "contextual",
        "publication_year": 2019,
    })
    await graph.upsert_chunk_node({
        "chunk_id": "chunk-p1",
        "work_id": "test--work-primary",
        "text_preview": "Primary chunk text",
        "granularity": "meso",
        "source_class": "primary",
    })
    await graph.upsert_chunk_node({
        "chunk_id": "chunk-c1",
        "work_id": "test--work-contextual",
        "text_preview": "Contextual chunk text",
        "granularity": "meso",
        "source_class": "contextual",
    })

    # Create an ENGAGES_WITH edge
    await graph.create_edge(
        "Chunk", "chunk_id", "chunk-p1",
        "ENGAGES_WITH",
        "Chunk", "chunk_id", "chunk-c1",
        {
            "link_type": "explicit_citation",
            "confidence": "high",
            "detection_method": "footnote_reference",
            "evidence": "See Author, p. 42",
        },
    )

    # Query from the primary work
    result = await graph.get_passage_links_for_work("test--work-primary")
    assert len(result) == 1
    assert result[0]["source_chunk_id"] == "chunk-p1"
    assert result[0]["target_chunk_id"] == "chunk-c1"
    assert result[0]["rel_type"] == "ENGAGES_WITH"
    assert result[0]["link_type"] == "explicit_citation"
    assert result[0]["confidence"] == "high"

    # Query from the contextual work should also find the same edge
    result_ctx = await graph.get_passage_links_for_work("test--work-contextual")
    assert len(result_ctx) == 1
    assert result_ctx[0]["source_chunk_id"] == "chunk-c1"
    assert result_ctx[0]["target_chunk_id"] == "chunk-p1"
    assert result_ctx[0]["rel_type"] == "ENGAGES_WITH"


async def test_get_passage_links_for_work_thematic_parallel(neo4j_conn: Neo4jConnection) -> None:
    """Returns THEMATIC_PARALLEL edges for chunks belonging to a work."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    # Create two works with chunks
    for wid, title, sc in [
        ("test--work-a", "Work A", "primary"),
        ("test--work-b", "Work B", "primary"),
    ]:
        await graph.upsert_work_node({
            "work_id": wid,
            "title": title,
            "author": "test-author",
            "source_class": sc,
            "publication_year": 2020,
        })

    await graph.upsert_chunk_node({
        "chunk_id": "chunk-a1",
        "work_id": "test--work-a",
        "text_preview": "Chunk from work A",
        "granularity": "meso",
        "source_class": "primary",
    })
    await graph.upsert_chunk_node({
        "chunk_id": "chunk-b1",
        "work_id": "test--work-b",
        "text_preview": "Chunk from work B",
        "granularity": "meso",
        "source_class": "primary",
    })

    # Create a THEMATIC_PARALLEL edge
    await graph.create_edge(
        "Chunk", "chunk_id", "chunk-a1",
        "THEMATIC_PARALLEL",
        "Chunk", "chunk_id", "chunk-b1",
        {
            "link_type": "thematic_parallel",
            "confidence": "low",
            "detection_method": "semantic_similarity",
            "similarity_score": 0.91,
            "shared_themes": ["imagination", "perception"],
        },
    )

    result = await graph.get_passage_links_for_work("test--work-a")
    assert len(result) == 1
    assert result[0]["source_chunk_id"] == "chunk-a1"
    assert result[0]["target_chunk_id"] == "chunk-b1"
    assert result[0]["rel_type"] == "THEMATIC_PARALLEL"
    assert result[0]["link_type"] == "thematic_parallel"
    assert result[0]["confidence"] == "low"


async def test_get_passage_links_for_work_both_edge_types(neo4j_conn: Neo4jConnection) -> None:
    """Returns both ENGAGES_WITH and THEMATIC_PARALLEL edges together."""
    await neo4j_conn.init_schema()
    graph = Neo4jGraphRepository(neo4j_conn)

    for wid, title in [("test--work-1", "Work One"), ("test--work-2", "Work Two")]:
        await graph.upsert_work_node({
            "work_id": wid,
            "title": title,
            "author": "test-author",
            "source_class": "primary",
            "publication_year": 2020,
        })

    for cid, wid in [
        ("chunk-1a", "test--work-1"),
        ("chunk-1b", "test--work-1"),
        ("chunk-2a", "test--work-2"),
        ("chunk-2b", "test--work-2"),
    ]:
        await graph.upsert_chunk_node({
            "chunk_id": cid,
            "work_id": wid,
            "text_preview": f"Text for {cid}",
            "granularity": "meso",
            "source_class": "primary",
        })

    # ENGAGES_WITH between chunk-1a and chunk-2a
    await graph.create_edge(
        "Chunk", "chunk_id", "chunk-1a",
        "ENGAGES_WITH",
        "Chunk", "chunk_id", "chunk-2a",
        {"link_type": "explicit_citation", "confidence": "high"},
    )

    # THEMATIC_PARALLEL between chunk-1b and chunk-2b
    await graph.create_edge(
        "Chunk", "chunk_id", "chunk-1b",
        "THEMATIC_PARALLEL",
        "Chunk", "chunk_id", "chunk-2b",
        {"link_type": "thematic_parallel", "confidence": "low"},
    )

    result = await graph.get_passage_links_for_work("test--work-1")
    assert len(result) == 2

    rel_types = {r["rel_type"] for r in result}
    assert rel_types == {"ENGAGES_WITH", "THEMATIC_PARALLEL"}


async def test_connection_scanner_get_existing_links_uses_method(
    neo4j_conn: Neo4jConnection,
) -> None:
    """ConnectionScanner._get_existing_links calls get_passage_links_for_work without error."""
    await neo4j_conn.init_schema()

    # Build a minimal StorageManager-like object with a graph property
    graph = Neo4jGraphRepository(neo4j_conn)

    class _FakeStorage:
        """Minimal stand-in providing just the graph attribute."""

        def __init__(self, graph_repo: Neo4jGraphRepository) -> None:
            self.graph = graph_repo

    from author_library.surfacing.connection_scanner import ConnectionScanner

    fake_storage: Any = _FakeStorage(graph)
    scanner = ConnectionScanner.__new__(ConnectionScanner)
    scanner._storage = fake_storage

    # Should return empty set without raising AttributeError
    links = await scanner._get_existing_links("nonexistent-work")
    assert links == set()


# -- StorageManager Tests ----------------------------------------------------


async def test_storage_manager_lifecycle(storage: StorageManager) -> None:
    """StorageManager connects, reports health, and closes."""
    health = await storage.health_check()
    assert health["postgres"] is True
    assert health["neo4j"] is True
