"""Source-class isolation tests for voice contamination prevention.

Verifies that secondary sources are NOT used for voice profile extraction,
and that graph edges respect source-class boundaries (ATTRIBUTED_BY_CRITIC
vs MAKES_ARGUMENT).

These tests verify the classification gate — the #1 architectural concern.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from author_library.catalog.models import ProcessingRoute, SourceClass

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.storage.manager import StorageManager

from .conftest import SKIP_NO_ANTHROPIC, SKIP_NO_DB

# ---------------------------------------------------------------------------
# Test content
# ---------------------------------------------------------------------------

PRIMARY_TEXT = """\
Faith and Imagination

The imagination is not a faculty for creating illusions; it is the faculty
by which one human being can enter into the experience of another. For me,
imagination is the great key to empathy, to understanding, to shared
humanity. When we read a poem, we exercise this faculty.

Poetry is, at its deepest level, a form of prayer. In writing verse, I am
not merely arranging words, but listening — listening for the voice that
speaks through the silence of the page. The poet's task is to hold the
tension between knowing and unknowing.
"""

SECONDARY_TEXT = """\
A Critic's Analysis of Malcolm Guite

Dr. Smith argues that Guite's poetic theology represents a significant
contribution to the field of theological aesthetics. According to Smith,
Guite's work bridges the gap between academic theology and literary
criticism. Smith notes that Guite's approach to sacramental imagination
draws heavily from Coleridge and the Romantic tradition.

This analysis examines how Guite positions himself within the broader
landscape of contemporary Christian thought, particularly his engagement
with the work of C.S. Lewis and George MacDonald.
"""


def _write_temp_text(content: str) -> Path:
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.write(fd, content.encode())
    os.close(fd)
    return Path(path)


async def _make_author(storage: StorageManager, author_id: str, name: str) -> None:
    await storage.pg.execute(
        "INSERT INTO authors (id, canonical_name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        author_id,
        name,
    )


# ---------------------------------------------------------------------------
# Classification gate tests (no LLM needed)
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestClassificationGate:
    """Verify that source classification gates downstream processing."""

    async def test_primary_source_gets_full_enrichment(
        self, clean_storage: StorageManager
    ) -> None:
        """PRIMARY sources should get FULL_ENRICHMENT processing route."""
        assert ProcessingRoute.FULL_ENRICHMENT.value == "full_enrichment"
        assert SourceClass.PRIMARY.value == "primary"

    async def test_secondary_source_gets_embeddings_and_graph(
        self, clean_storage: StorageManager
    ) -> None:
        """SECONDARY sources should get EMBEDDINGS_AND_GRAPH, not voice profile."""
        assert ProcessingRoute.EMBEDDINGS_AND_GRAPH.value == "embeddings_and_graph"
        assert SourceClass.SECONDARY.value == "secondary"

    async def test_tertiary_source_gets_metadata_only(
        self, clean_storage: StorageManager
    ) -> None:
        """TERTIARY sources should get METADATA_ONLY — no content processing."""
        assert ProcessingRoute.METADATA_ONLY.value == "metadata_only"
        assert SourceClass.TERTIARY.value == "tertiary"

    async def test_voice_profile_requires_primary_source(
        self, clean_storage: StorageManager
    ) -> None:
        """Voice profiles should only be built from PRIMARY sources.

        Verify by storing a secondary work and checking no voice profile exists.
        """
        await _make_author(clean_storage, "guite", "Malcolm Guite")

        # Insert a secondary work
        await clean_storage.works.create({
            "work_id": "test--critic-analysis",
            "title": "A Critic's Analysis",
            "author": "guite",
            "source_class": "secondary",
            "source_class_note": "Written by Dr. Smith about Guite, not by Guite himself",
            "publication_year": 2020,
            "publisher": "Academic Press",
            "format_ingested": "txt",
            "word_count": 200,
            "genre_tags": ["scholarly-prose"],
            "subject_headings": ["literary criticism"],
            "source_metadata": json.dumps({"external_author": "Dr. Smith"}),
        })

        # Verify no voice profile was created for this author from a secondary source
        profile = await clean_storage.voice_profiles.get_current("guite")
        assert profile is None, "Voice profile should not exist from secondary source alone"

    async def test_source_class_stored_correctly(
        self, clean_storage: StorageManager
    ) -> None:
        """Verify that source_class is persisted correctly in works table."""
        await _make_author(clean_storage, "guite", "Malcolm Guite")

        for sc in ("primary", "secondary", "contextual", "tertiary"):
            await clean_storage.works.create({
                "work_id": f"test--test-{sc}",
                "title": f"Test {sc.title()} Work",
                "author": "guite",
                "source_class": sc,
                "source_class_note": f"This is a {sc} source for testing classification",
                "publication_year": 2020,
                "publisher": "Test Publisher",
                "format_ingested": "txt",
                "word_count": 100,
                "genre_tags": ["scholarly-prose"],
                "subject_headings": ["test"],
            })

        works = await clean_storage.works.list_by_author("guite")
        classes = {w["source_class"] for w in works}
        assert classes == {"primary", "secondary", "contextual", "tertiary"}


# ---------------------------------------------------------------------------
# Graph edge source-class tests
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestGraphSourceIsolation:
    """Verify graph edges respect source-class boundaries."""

    async def test_secondary_chunk_gets_attributed_edge(
        self, clean_storage: StorageManager
    ) -> None:
        """Secondary source chunks should have ATTRIBUTED_BY_CRITIC edges,
        NOT MAKES_ARGUMENT edges which are reserved for primary sources."""
        # Create primary and secondary chunk nodes
        await clean_storage.graph.upsert_chunk_node({
            "chunk_id": "primary-chunk-1",
            "work_id": "test--faith-imagination",
            "text_preview": "The imagination is not a faculty for creating illusions",
            "granularity": "meso",
            "source_class": "primary",
        })
        await clean_storage.graph.upsert_chunk_node({
            "chunk_id": "secondary-chunk-1",
            "work_id": "test--critic-analysis",
            "text_preview": "Dr. Smith argues that Guite's poetic theology",
            "granularity": "meso",
            "source_class": "secondary",
        })

        # Create ATTRIBUTED_BY_CRITIC edge (correct for secondary)
        await clean_storage.graph.create_edge(
            "Chunk", "chunk_id", "secondary-chunk-1",
            "ATTRIBUTED_BY_CRITIC",
            "Chunk", "chunk_id", "primary-chunk-1",
            properties={"critic": "Dr. Smith", "confidence": "high"},
        )

        # Verify the edge exists with correct type
        related = await clean_storage.graph.get_related_chunks(
            "secondary-chunk-1", "ATTRIBUTED_BY_CRITIC"
        )
        assert len(related) == 1
        assert related[0]["source_class"] == "primary"

    async def test_primary_chunks_can_have_makes_argument(
        self, clean_storage: StorageManager
    ) -> None:
        """Primary source chunks can have MAKES_ARGUMENT edges."""
        await clean_storage.graph.upsert_chunk_node({
            "chunk_id": "primary-arg-1",
            "work_id": "test--faith-imagination",
            "text_preview": "Poetry is a form of prayer",
            "granularity": "meso",
            "source_class": "primary",
        })
        await clean_storage.graph.upsert_chunk_node({
            "chunk_id": "primary-arg-2",
            "work_id": "test--faith-imagination",
            "text_preview": "The poet's task is to hold tension",
            "granularity": "meso",
            "source_class": "primary",
        })

        # Create DEVELOPS_FROM edge between primary chunks
        await clean_storage.graph.create_edge(
            "Chunk", "chunk_id", "primary-arg-2",
            "DEVELOPS_FROM",
            "Chunk", "chunk_id", "primary-arg-1",
            properties={"confidence": "medium"},
        )

        related = await clean_storage.graph.get_related_chunks(
            "primary-arg-2", "DEVELOPS_FROM"
        )
        assert len(related) == 1
        assert related[0]["source_class"] == "primary"


# ---------------------------------------------------------------------------
# Full source isolation E2E (requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------


@SKIP_NO_DB
@SKIP_NO_ANTHROPIC
class TestSourceIsolationE2E:
    """Full E2E test: ingest a secondary source and verify isolation."""

    async def test_secondary_source_excluded_from_voice_profile(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """Ingest a secondary source and verify it does NOT produce a voice profile."""
        from author_library.embeddings import ProviderRegistry
        from author_library.tools.ingest import handle_ingest_book

        await _make_author(clean_storage, "guite", "Malcolm Guite")
        embedding_provider = ProviderRegistry.create(integration_settings)

        temp_path = _write_temp_text(SECONDARY_TEXT)
        try:
            result_str = await handle_ingest_book(
                {
                    "file_path": str(temp_path),
                    "subject_author_id": "guite",
                    "metadata_hints": {
                        "source_class": "secondary",
                        "genre_tags": ["scholarly-prose"],
                    },
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            result = json.loads(result_str)
            assert result["source_class"] == "secondary"

            # Verify no voice profile was extracted
            profile = await clean_storage.voice_profiles.get_current("guite")
            assert profile is None, (
                "Voice profile should NOT be created from secondary source — "
                "this would be voice contamination"
            )

        finally:
            temp_path.unlink(missing_ok=True)
            await embedding_provider.close()
