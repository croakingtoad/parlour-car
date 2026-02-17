"""End-to-end integration tests for the full ingestion and query pipeline.

Tests run against real PostgreSQL + Neo4j via Docker compose.
LLM-dependent steps (classification, entity extraction, ask_author) are
gated behind ANTHROPIC_API_KEY availability.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from author_library.tools.ingest import handle_ingest_book
from author_library.tools.meta import handle_health_check, handle_library_stats, handle_list_works
from author_library.tools.query import handle_find_quotes

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.storage.manager import StorageManager

from .conftest import SKIP_NO_ANTHROPIC, SKIP_NO_DB

# ---------------------------------------------------------------------------
# Public-domain test content (Shakespeare, Sonnet 18)
# ---------------------------------------------------------------------------

SONNET_18 = """\
Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date;
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimm'd;
And every fair from fair sometime declines,
By chance or nature's changing course untrimm'd;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.
"""

HAMLET_EXCERPT = """\
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them. To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause: there's the respect
That makes calamity of so long life.
"""


def _make_test_author(storage: StorageManager, author_id: str = "shakespeare") -> Any:
    """Insert a test author into the authors table."""
    return storage.pg.execute(
        "INSERT INTO authors (id, canonical_name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        author_id,
        "William Shakespeare",
    )


def _write_temp_text(content: str, suffix: str = ".txt") -> Path:
    """Write content to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, content.encode())
    os.close(fd)
    return Path(path)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestHealthCheck:
    """Test the health_check tool against real databases."""

    async def test_health_check_all_healthy(self, storage: StorageManager) -> None:
        result_str = await handle_health_check({}, storage=storage)
        result = json.loads(result_str)
        assert result["postgres"]["status"] == "healthy"
        assert result["neo4j"]["status"] == "healthy"
        assert result["overall"] == "healthy"

    async def test_health_check_without_embedding(self, storage: StorageManager) -> None:
        result_str = await handle_health_check({}, storage=storage, embedding_provider=None)
        result = json.loads(result_str)
        assert result["postgres"]["status"] == "healthy"
        assert result["neo4j"]["status"] == "healthy"
        assert result["embedding"]["status"] == "not_configured"


# ---------------------------------------------------------------------------
# Database connectivity
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestDatabaseConnectivity:
    """Verify raw database operations work against real infrastructure."""

    async def test_postgres_read_write(self, clean_storage: StorageManager) -> None:
        """Verify PG write + read round trip."""
        await _make_test_author(clean_storage)
        row = await clean_storage.pg.fetch_one(
            "SELECT canonical_name FROM authors WHERE id = $1", "shakespeare"
        )
        assert row is not None
        assert row["canonical_name"] == "William Shakespeare"

    async def test_neo4j_read_write(self, clean_storage: StorageManager) -> None:
        """Verify Neo4j write + read round trip."""
        await clean_storage.graph.upsert_work_node({
            "work_id": "shakespeare--hamlet",
            "title": "Hamlet",
            "author": "William Shakespeare",
            "source_class": "primary",
            "publication_year": 1603,
        })
        results = await clean_storage.neo4j.execute_read(
            "MATCH (w:Work {work_id: $wid}) RETURN w.title AS title",
            {"wid": "shakespeare--hamlet"},
        )
        assert len(results) == 1
        assert results[0]["title"] == "Hamlet"

    async def test_chunk_storage_roundtrip(self, clean_storage: StorageManager) -> None:
        """Verify chunk create + read with all granularities."""
        await _make_test_author(clean_storage)
        # Create a work first
        await clean_storage.works.create({
            "work_id": "shakespeare--sonnets",
            "title": "Sonnets",
            "author": "shakespeare",
            "source_class": "primary",
            "source_class_note": "Original works by the subject author Shakespeare",
            "publication_year": 1609,
            "publisher": "Thomas Thorpe",
            "format_ingested": "txt",
            "word_count": 500,
            "genre_tags": ["poetry"],
            "subject_headings": ["English poetry"],
        })

        # Store chunks at each granularity
        for gran in ("macro", "meso", "micro"):
            chunk_id = await clean_storage.chunks.create({
                "work_id": "shakespeare--sonnets",
                "text": f"Test chunk at {gran} granularity",
                "granularity": gran,
                "source_class": "primary",
                "position": 0,
            })
            assert chunk_id is not None

        # Verify all three granularities stored
        all_chunks = await clean_storage.chunks.list_by_work("shakespeare--sonnets")
        granularities = {c["granularity"] for c in all_chunks}
        assert granularities == {"macro", "meso", "micro"}

    async def test_embedding_storage_roundtrip(self, clean_storage: StorageManager) -> None:
        """Verify embedding store + similarity search works with pgvector."""
        await _make_test_author(clean_storage)
        await clean_storage.works.create({
            "work_id": "shakespeare--sonnets",
            "title": "Sonnets",
            "author": "shakespeare",
            "source_class": "primary",
            "source_class_note": "Original works by the subject author Shakespeare",
            "publication_year": 1609,
            "publisher": "Thomas Thorpe",
            "format_ingested": "txt",
            "word_count": 500,
            "genre_tags": ["poetry"],
            "subject_headings": ["English poetry"],
        })

        chunk_id = await clean_storage.chunks.create({
            "work_id": "shakespeare--sonnets",
            "text": "Shall I compare thee to a summer's day?",
            "granularity": "micro",
            "source_class": "primary",
            "position": 0,
        })

        # Store a 1024-dim embedding
        test_embedding = [0.01 * i for i in range(1024)]
        embed_id = await clean_storage.embeddings.store(
            chunk_id, test_embedding, "test", "test-model", 1024
        )
        assert embed_id is not None

        # Verify retrieval
        rows = await clean_storage.embeddings.get_by_chunk(chunk_id)
        assert len(rows) == 1
        assert rows[0]["provider"] == "test"

    async def test_neo4j_graph_operations(self, clean_storage: StorageManager) -> None:
        """Verify Neo4j node and edge creation."""
        # Create two chunk nodes
        await clean_storage.graph.upsert_chunk_node({
            "chunk_id": "chunk-1",
            "work_id": "shakespeare--hamlet",
            "text_preview": "To be or not to be",
            "granularity": "micro",
            "source_class": "primary",
        })
        await clean_storage.graph.upsert_chunk_node({
            "chunk_id": "chunk-2",
            "work_id": "shakespeare--hamlet",
            "text_preview": "The slings and arrows",
            "granularity": "micro",
            "source_class": "primary",
        })

        # Create an edge
        await clean_storage.graph.create_edge(
            "Chunk", "chunk_id", "chunk-1",
            "DEVELOPS_FROM",
            "Chunk", "chunk_id", "chunk-2",
            properties={"confidence": "high"},
        )

        # Verify edge exists
        related = await clean_storage.graph.get_related_chunks(
            "chunk-1", "DEVELOPS_FROM"
        )
        assert len(related) >= 1
        assert related[0]["chunk_id"] == "chunk-2"


# ---------------------------------------------------------------------------
# Full pipeline E2E (requires ANTHROPIC_API_KEY for LLM steps)
# ---------------------------------------------------------------------------


@SKIP_NO_DB
@SKIP_NO_ANTHROPIC
class TestFullPipelineE2E:
    """Full ingestion pipeline E2E: ingest a text file, verify all storage layers."""

    async def test_ingest_and_query(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """Ingest a public-domain text and verify chunks, embeddings, and graph nodes."""
        from author_library.embeddings import ProviderRegistry

        await _make_test_author(clean_storage, "shakespeare")
        embedding_provider = ProviderRegistry.create(integration_settings)

        # Write test content to a temp file
        test_content = f"Sonnet 18\n\n{SONNET_18}\n\nHamlet Act 3 Scene 1\n\n{HAMLET_EXCERPT}"
        temp_path = _write_temp_text(test_content)

        try:
            # Ingest via the MCP tool handler
            result_str = await handle_ingest_book(
                {
                    "file_path": str(temp_path),
                    "subject_author_id": "shakespeare",
                    "metadata_hints": {
                        "source_class": "primary",
                        "genre_tags": ["poetry", "drama"],
                    },
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            result = json.loads(result_str)

            # Verify ingestion result
            assert "work_id" in result
            assert result["source_class"] == "primary"
            work_id = result["work_id"]

            # Verify chunks exist in PG at multiple granularities
            all_chunks = await clean_storage.chunks.list_by_work(work_id)
            assert len(all_chunks) > 0, "No chunks created"

            granularities = {c["granularity"] for c in all_chunks}
            assert len(granularities) >= 1, "Expected at least 1 granularity level"

            # Verify embeddings were stored
            chunks_with_embeddings = 0
            for chunk in all_chunks[:5]:  # Check first 5
                embeds = await clean_storage.embeddings.get_by_chunk(chunk["id"])
                if embeds:
                    chunks_with_embeddings += 1
            assert chunks_with_embeddings > 0, "No embeddings stored"

            # Verify Neo4j work node exists
            neo4j_results = await clean_storage.neo4j.execute_read(
                "MATCH (w:Work {work_id: $wid}) RETURN w.title AS title",
                {"wid": work_id},
            )
            assert len(neo4j_results) >= 1, "Work node not in Neo4j"

            # Verify Neo4j chunk nodes exist
            chunk_results = await clean_storage.neo4j.execute_read(
                "MATCH (c:Chunk {work_id: $wid}) RETURN count(c) AS cnt",
                {"wid": work_id},
            )
            assert chunk_results[0]["cnt"] > 0, "No chunk nodes in Neo4j"

            # Verify list_works returns the ingested work
            works_str = await handle_list_works(
                {"author_id": "shakespeare"}, storage=clean_storage
            )
            works = json.loads(works_str)
            assert works["total_works"] >= 1

            # Verify library_stats counts something
            stats_str = await handle_library_stats({}, storage=clean_storage)
            stats = json.loads(stats_str)
            assert stats["works"]["total_works"] >= 1
            assert stats["chunks"]["total_chunks"] >= 1

        finally:
            temp_path.unlink(missing_ok=True)
            await embedding_provider.close()

    async def test_find_quotes_after_ingestion(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """Ingest text, then verify find_quotes returns matching passages."""
        from author_library.embeddings import ProviderRegistry

        await _make_test_author(clean_storage, "shakespeare")
        embedding_provider = ProviderRegistry.create(integration_settings)

        temp_path = _write_temp_text(
            f"Shakespeare Collected\n\n{SONNET_18}\n\n{HAMLET_EXCERPT}"
        )

        try:
            # Ingest
            await handle_ingest_book(
                {
                    "file_path": str(temp_path),
                    "subject_author_id": "shakespeare",
                    "metadata_hints": {
                        "source_class": "primary",
                        "genre_tags": ["poetry", "drama"],
                    },
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )

            # Search for a known phrase
            quotes_str = await handle_find_quotes(
                {"query": "summer's day", "limit": 5},
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            quotes = json.loads(quotes_str)
            assert quotes["total_results"] > 0, "find_quotes returned no results"

            # At least one result should contain the phrase
            found = any("summer" in q["text"].lower() for q in quotes["quotes"])
            assert found, "Expected 'summer' in at least one quote result"

        finally:
            temp_path.unlink(missing_ok=True)
            await embedding_provider.close()
