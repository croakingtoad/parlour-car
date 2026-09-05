"""Live integration tests for the composable ingestion pipeline.

Tests each MCP tool in the composable flow:
  classify_source → catalog_source → chunk_source → detect_passage_links → flag_acquisition

Runs against real PostgreSQL + Neo4j. LLM-gated steps require ANTHROPIC_API_KEY.
Uses public-domain Shakespeare text as test content.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from author_library.errors import ClassificationError
from author_library.tools.composable_ingestion import (
    handle_catalog_source,
    handle_chunk_source,
    handle_classify_source,
    handle_detect_passage_links,
    handle_flag_acquisition,
)

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.storage.manager import StorageManager

from .conftest import SKIP_NO_ANTHROPIC, SKIP_NO_DB

# ---------------------------------------------------------------------------
# Public-domain test content
# ---------------------------------------------------------------------------

HAMLET_EXCERPT = """\
Hamlet: A Play in Five Acts
By William Shakespeare

Act III, Scene 1

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

Act II, Scene 2

What a piece of work is a man!
How noble in reason, how infinite in faculty!
In form and moving how express and admirable!
In action how like an Angel! in apprehension
how like a god! The beauty of the world!
The paragon of animals! And yet to me, what
is this quintessence of dust?
"""


def _write_temp_text(content: str, suffix: str = ".txt") -> Path:
    """Write content to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, content.encode())
    os.close(fd)
    return Path(path)


# ---------------------------------------------------------------------------
# B1: classify_source
# ---------------------------------------------------------------------------


@SKIP_NO_DB
@SKIP_NO_ANTHROPIC
class TestClassifySource:
    """classify_source parses a document and returns suggested classification."""

    async def test_classify_primary_source(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """classify_source on Shakespeare text returns primary with high confidence."""
        temp_path = _write_temp_text(HAMLET_EXCERPT)
        try:
            result_str = await handle_classify_source(
                {
                    "file_path": str(temp_path),
                    "subject_author": "shakespeare",
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=None,  # type: ignore[arg-type]
            )
            result = json.loads(result_str)

            assert "suggested_class" in result
            assert result["suggested_class"] == "primary"
            assert result["confidence"] >= 0.5, "Expected at least moderate confidence"
            assert isinstance(result["signals"], list)
            assert "reasoning" in result
            assert "suggested_work_type" in result
            assert "document_metadata" in result
            assert "word_count" in result["document_metadata"]
        finally:
            temp_path.unlink(missing_ok=True)

    async def test_classify_does_not_store_in_db(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """classify_source must NOT create any database records."""
        temp_path = _write_temp_text(HAMLET_EXCERPT)
        try:
            await handle_classify_source(
                {
                    "file_path": str(temp_path),
                    "subject_author": "shakespeare",
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=None,  # type: ignore[arg-type]
            )
            # Verify nothing was written to PG
            rows = await clean_storage.pg.fetch_all("SELECT work_id FROM works")
            assert len(rows) == 0, "classify_source must not write to the works table"
        finally:
            temp_path.unlink(missing_ok=True)

    async def test_classify_returns_human_judgment_flag(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """classify_source result includes requires_human_judgment flag."""
        temp_path = _write_temp_text(HAMLET_EXCERPT)
        try:
            result_str = await handle_classify_source(
                {
                    "file_path": str(temp_path),
                    "subject_author": "shakespeare",
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=None,  # type: ignore[arg-type]
            )
            result = json.loads(result_str)
            assert "requires_human_judgment" in result
            assert isinstance(result["requires_human_judgment"], bool)
        finally:
            temp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# B2: catalog_source
# ---------------------------------------------------------------------------


@SKIP_NO_DB
@SKIP_NO_ANTHROPIC
class TestCatalogSource:
    """catalog_source stores a catalog entry in PG and Neo4j."""

    async def test_catalog_creates_work_record(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """catalog_source creates a work record in PG and returns work_id."""
        temp_path = _write_temp_text(HAMLET_EXCERPT)
        try:
            result_str = await handle_catalog_source(
                {
                    "file_path": str(temp_path),
                    "source_class": "primary",
                    "work_type": "other",
                    "metadata_overrides": {
                        "author": "Shakespeare",
                        "title": "Hamlet Test",
                        "subject_author_id": "shakespeare",
                        "genre_tags": ["drama"],
                        "subject_headings": ["English drama"],
                        "publication_year": 1603,
                        "publisher": "Unknown",
                    },
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=None,  # type: ignore[arg-type]
            )
            result = json.loads(result_str)

            assert "work_id" in result
            work_id = result["work_id"]
            assert "shakespeare" in work_id, f"Expected 'shakespeare' in work_id, got: {work_id}"

            # Verify work exists in PG
            work = await clean_storage.works.get(work_id)
            assert work is not None, "Work must exist in PostgreSQL after catalog_source"
            assert work["source_class"] == "primary"

            # Verify work node exists in Neo4j
            neo4j_rows = await clean_storage.neo4j.execute_read(
                "MATCH (w:Work {work_id: $wid}) RETURN w.title AS title",
                {"wid": work_id},
            )
            assert len(neo4j_rows) >= 1, "Work node must exist in Neo4j after catalog_source"
        finally:
            temp_path.unlink(missing_ok=True)

    async def test_catalog_returns_catalog_record_structure(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """catalog_source result includes catalog_record, chapters_detected, table_of_contents."""
        temp_path = _write_temp_text(HAMLET_EXCERPT)
        try:
            result_str = await handle_catalog_source(
                {
                    "file_path": str(temp_path),
                    "source_class": "primary",
                    "work_type": "other",
                    "metadata_overrides": {
                        "author": "Shakespeare",
                        "title": "Hamlet Catalog Test",
                        "subject_author_id": "shakespeare",
                        "genre_tags": ["drama"],
                        "subject_headings": ["English drama"],
                        "publication_year": 1603,
                        "publisher": "Unknown",
                    },
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=None,  # type: ignore[arg-type]
            )
            result = json.loads(result_str)

            assert "catalog_record" in result
            catalog = result["catalog_record"]
            assert "title" in catalog
            assert "author" in catalog
            assert "source_class" in catalog
            assert catalog["source_class"] == "primary"

            assert "chapters_detected" in result
            assert isinstance(result["chapters_detected"], list)

            assert "table_of_contents" in result
            assert isinstance(result["table_of_contents"], list)
        finally:
            temp_path.unlink(missing_ok=True)


@SKIP_NO_DB
class TestReferenceCatalogSource:
    """A confirmed reference source catalogs without an LLM reclassification."""

    @pytest.mark.parametrize(
        "missing_field",
        ["external_author", "reference_type", "subject_domain"],
    )
    async def test_catalog_reference_rejects_missing_required_metadata(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
        missing_field: str,
    ) -> None:
        metadata_overrides = {
            "author": "Test Reference Author",
            "title": "Incomplete Reference Catalog",
            "external_author": "Test Reference Author",
            "reference_type": "craft-handbook",
            "subject_domain": "dramatic-verse",
        }
        metadata_overrides.pop(missing_field)
        if missing_field == "external_author":
            metadata_overrides.pop("author")

        temp_path = _write_temp_text("Standalone reference text without a byline.")
        try:
            works_before = await clean_storage.pg.fetch_val("SELECT count(*) FROM works")
            with pytest.raises(
                ClassificationError,
                match=rf'source_class="reference" requires metadata field "{missing_field}"',
            ):
                await handle_catalog_source(
                    {
                        "file_path": str(temp_path),
                        "source_class": "reference",
                        "metadata_overrides": metadata_overrides,
                    },
                    settings=integration_settings,
                    storage=clean_storage,
                    embedding_provider=None,
                )
            assert await clean_storage.pg.fetch_val("SELECT count(*) FROM works") == works_before
        finally:
            temp_path.unlink(missing_ok=True)

    async def test_catalog_reference_creates_reference_record(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        temp_path = _write_temp_text(HAMLET_EXCERPT)
        try:
            result = json.loads(await handle_catalog_source(
                {
                    "file_path": str(temp_path),
                    "source_class": "reference",
                    "metadata_overrides": {
                        "author": "Test",
                        "title": "Reference Catalog",
                        "external_author": "Test",
                        "reference_type": "craft-handbook",
                        "subject_domain": "dramatic-verse",
                        "publication_year": 1603,
                        "publisher": "Test Publisher",
                    },
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=None,
            ))

            work = await clean_storage.works.get(result["work_id"])
            assert work is not None
            assert work["source_class"] == "reference"
            source_metadata = work["source_metadata"]
            if isinstance(source_metadata, str):
                source_metadata = json.loads(source_metadata)
            assert source_metadata["reference_type"] == "craft-handbook"
            assert "voice_profile_eligible" not in source_metadata
        finally:
            temp_path.unlink(missing_ok=True)

    async def test_reference_passage_links_load_corpus_primary_counterparts(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        common_work = {
            "source_class_note": "Reference linking integration test",
            "publication_year": 2026,
            "publisher": "Test Publisher",
            "format_ingested": "txt",
            "word_count": 100,
            "genre_tags": ["craft"],
            "subject_headings": ["General"],
        }
        await clean_storage.works.create({
            **common_work,
            "work_id": "test--link-primary",
            "title": "Primary Counterpart",
            "author": "Test Subject",
            "source_class": "primary",
            "source_metadata": {"subject_author_id": "test-subject"},
        })
        await clean_storage.works.create({
            **common_work,
            "work_id": "test--link-reference",
            "title": "Reference Work",
            "author": "Test Reference Author",
            "source_class": "reference",
            "source_metadata": {
                "external_author": "Test Reference Author",
                "reference_type": "craft-handbook",
                "subject_domain": "prosody",
            },
        })
        for work_id, source_class in (
            ("test--link-primary", "primary"),
            ("test--link-reference", "reference"),
        ):
            await clean_storage.chunks.create({
                "work_id": work_id,
                "text": f"Meso passage from {work_id}",
                "annotation": None,
                "granularity": "meso",
                "source_class": source_class,
                "position": 0,
                "metadata": {},
            })

        result = json.loads(await handle_detect_passage_links(
            {
                "work_id": "test--link-reference",
                "scan_types": ["none"],
            },
            settings=integration_settings,
            storage=clean_storage,
            # scan_types=["none"] exercises routing without an embedding scan.
            embedding_provider=None,  # type: ignore[arg-type]
        ))

        assert [
            source["work_id"] for source in result["contextual_sources_referenced"]
        ] == ["test--link-primary"]


# ---------------------------------------------------------------------------
# B3: chunk_source (requires prior catalog)
# ---------------------------------------------------------------------------


@SKIP_NO_DB
@SKIP_NO_ANTHROPIC
class TestChunkSource:
    """chunk_source creates chunks from a previously cataloged work."""

    async def test_chunk_creates_pg_chunks(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """chunk_source creates chunks in PG with proper granularity distribution."""
        from author_library.embeddings import ProviderRegistry

        temp_path = _write_temp_text(HAMLET_EXCERPT)
        embedding_provider = ProviderRegistry.create(integration_settings)

        try:
            # Step 1: catalog_source
            catalog_str = await handle_catalog_source(
                {
                    "file_path": str(temp_path),
                    "source_class": "primary",
                    "work_type": "other",
                    "metadata_overrides": {
                        "author": "Shakespeare",
                        "title": "Hamlet Chunk Test",
                        "subject_author_id": "shakespeare",
                        "genre_tags": ["drama"],
                        "subject_headings": ["English drama"],
                        "publication_year": 1603,
                        "publisher": "Unknown",
                    },
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            catalog_result = json.loads(catalog_str)
            work_id = catalog_result["work_id"]

            # Step 2: Patch source_metadata with file_path so chunk_source can re-parse.
            # NOTE: catalog_source does not persist file_path because it is not a
            # CatalogEntry model field. This patch bridges that gap until the catalog
            # handler is updated to store the original file path.
            work = await clean_storage.works.get(work_id)
            assert work is not None
            source_meta_raw = work.get("source_metadata") or {}
            source_meta = json.loads(source_meta_raw) if isinstance(source_meta_raw, str) else dict(source_meta_raw)
            source_meta["file_path"] = str(temp_path)
            await clean_storage.works.update(work_id, {"source_metadata": source_meta})

            # Step 3: chunk_source
            chunk_str = await handle_chunk_source(
                {"work_id": work_id},
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            chunk_result = json.loads(chunk_str)

            assert chunk_result["work_id"] == work_id
            assert "chunks_created" in chunk_result
            total_chunks = sum(chunk_result["chunks_created"].values())
            assert total_chunks > 0, "Expected at least one chunk created"

            # Verify chunks exist in PG
            all_chunks = await clean_storage.chunks.list_by_work(work_id)
            assert len(all_chunks) > 0, "No chunks found in PG after chunk_source"
            pg_chunk_ids = {str(chunk["id"]) for chunk in all_chunks}
            graph_chunks = await clean_storage.neo4j.execute_read(
                "MATCH (c:Chunk {work_id: $wid}) RETURN c.chunk_id AS chunk_id",
                {"wid": work_id},
            )
            graph_chunk_ids = {row["chunk_id"] for row in graph_chunks}
            assert graph_chunk_ids == pg_chunk_ids

        finally:
            temp_path.unlink(missing_ok=True)
            await embedding_provider.close()

    async def test_chunk_returns_status(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """chunk_source returns status 'complete' or 'partial' (never absent)."""
        from author_library.embeddings import ProviderRegistry

        temp_path = _write_temp_text(HAMLET_EXCERPT)
        embedding_provider = ProviderRegistry.create(integration_settings)

        try:
            catalog_str = await handle_catalog_source(
                {
                    "file_path": str(temp_path),
                    "source_class": "primary",
                    "work_type": "other",
                    "metadata_overrides": {
                        "author": "Shakespeare",
                        "title": "Hamlet Status Test",
                        "subject_author_id": "shakespeare",
                        "genre_tags": ["drama"],
                        "subject_headings": ["English drama"],
                        "publication_year": 1603,
                        "publisher": "Unknown",
                    },
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            work_id = json.loads(catalog_str)["work_id"]

            work = await clean_storage.works.get(work_id)
            source_meta_raw = work.get("source_metadata") or {}
            source_meta = json.loads(source_meta_raw) if isinstance(source_meta_raw, str) else dict(source_meta_raw)
            source_meta["file_path"] = str(temp_path)
            await clean_storage.works.update(work_id, {"source_metadata": source_meta})

            chunk_str = await handle_chunk_source(
                {"work_id": work_id},
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            chunk_result = json.loads(chunk_str)

            assert "status" in chunk_result
            assert chunk_result["status"] in ("complete", "partial"), (
                f"Unexpected status: {chunk_result['status']}"
            )
            assert "embedding_provider" in chunk_result
        finally:
            temp_path.unlink(missing_ok=True)
            await embedding_provider.close()

    async def test_chunk_idempotent(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """Running chunk_source twice replaces chunks rather than duplicating them."""
        from author_library.embeddings import ProviderRegistry

        temp_path = _write_temp_text(HAMLET_EXCERPT)
        embedding_provider = ProviderRegistry.create(integration_settings)

        try:
            catalog_str = await handle_catalog_source(
                {
                    "file_path": str(temp_path),
                    "source_class": "primary",
                    "work_type": "other",
                    "metadata_overrides": {
                        "author": "Shakespeare",
                        "title": "Hamlet Idempotent Test",
                        "subject_author_id": "shakespeare",
                        "genre_tags": ["drama"],
                        "subject_headings": ["English drama"],
                        "publication_year": 1603,
                        "publisher": "Unknown",
                    },
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            work_id = json.loads(catalog_str)["work_id"]

            work = await clean_storage.works.get(work_id)
            source_meta_raw = work.get("source_metadata") or {}
            source_meta = json.loads(source_meta_raw) if isinstance(source_meta_raw, str) else dict(source_meta_raw)
            source_meta["file_path"] = str(temp_path)
            await clean_storage.works.update(work_id, {"source_metadata": source_meta})

            # First run
            await handle_chunk_source(
                {"work_id": work_id},
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            count_after_first = len(await clean_storage.chunks.list_by_work(work_id))

            # Second run (idempotent — replaces chunks)
            await handle_chunk_source(
                {"work_id": work_id},
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            count_after_second = len(await clean_storage.chunks.list_by_work(work_id))

            assert count_after_second == count_after_first, (
                f"chunk_source must be idempotent: first={count_after_first}, "
                f"second={count_after_second}"
            )
        finally:
            temp_path.unlink(missing_ok=True)
            await embedding_provider.close()


# ---------------------------------------------------------------------------
# B4: detect_passage_links (requires prior chunk)
# ---------------------------------------------------------------------------


@SKIP_NO_DB
@SKIP_NO_ANTHROPIC
class TestDetectPassageLinks:
    """detect_passage_links finds cross-resource links after chunking."""

    async def test_detect_returns_links_structure(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """detect_passage_links returns expected JSON structure (0 links is valid)."""
        from author_library.embeddings import ProviderRegistry

        temp_path = _write_temp_text(HAMLET_EXCERPT)
        embedding_provider = ProviderRegistry.create(integration_settings)

        try:
            # Catalog
            catalog_str = await handle_catalog_source(
                {
                    "file_path": str(temp_path),
                    "source_class": "primary",
                    "work_type": "other",
                    "metadata_overrides": {
                        "author": "Shakespeare",
                        "title": "Hamlet Links Test",
                        "subject_author_id": "shakespeare",
                        "genre_tags": ["drama"],
                        "subject_headings": ["English drama"],
                        "publication_year": 1603,
                        "publisher": "Unknown",
                    },
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            work_id = json.loads(catalog_str)["work_id"]

            # Patch file_path for chunk step
            work = await clean_storage.works.get(work_id)
            source_meta_raw = work.get("source_metadata") or {}
            source_meta = json.loads(source_meta_raw) if isinstance(source_meta_raw, str) else dict(source_meta_raw)
            source_meta["file_path"] = str(temp_path)
            await clean_storage.works.update(work_id, {"source_metadata": source_meta})

            # Chunk
            await handle_chunk_source(
                {"work_id": work_id},
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )

            # Detect passage links (explicit_citation only — fast, no embeddings needed)
            links_str = await handle_detect_passage_links(
                {
                    "work_id": work_id,
                    "scan_types": ["explicit_citation"],
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            links_result = json.loads(links_str)

            assert "work_id" in links_result
            assert links_result["work_id"] == work_id
            assert "links_created" in links_result
            assert "explicit_citation" in links_result["links_created"]
            assert isinstance(links_result["links_created"]["explicit_citation"], int)
            assert "contextual_sources_referenced" in links_result
        finally:
            temp_path.unlink(missing_ok=True)
            await embedding_provider.close()

    async def test_detect_with_no_chunks_returns_note(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """detect_passage_links on a work with no meso chunks returns a note, not an error."""
        from author_library.embeddings import ProviderRegistry

        temp_path = _write_temp_text(HAMLET_EXCERPT)
        embedding_provider = ProviderRegistry.create(integration_settings)

        try:
            # Catalog only — no chunk step
            catalog_str = await handle_catalog_source(
                {
                    "file_path": str(temp_path),
                    "source_class": "primary",
                    "work_type": "other",
                    "metadata_overrides": {
                        "author": "Shakespeare",
                        "title": "Hamlet No Chunks Test",
                        "subject_author_id": "shakespeare",
                        "genre_tags": ["drama"],
                        "subject_headings": ["English drama"],
                        "publication_year": 1603,
                        "publisher": "Unknown",
                    },
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            work_id = json.loads(catalog_str)["work_id"]

            links_str = await handle_detect_passage_links(
                {"work_id": work_id, "scan_types": ["explicit_citation"]},
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            links_result = json.loads(links_str)

            assert "note" in links_result, "Expected 'note' field when no chunks exist"
            assert "chunk" in links_result["note"].lower()
        finally:
            temp_path.unlink(missing_ok=True)
            await embedding_provider.close()


# ---------------------------------------------------------------------------
# B5: flag_acquisition (independent of prior pipeline steps)
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestFlagAcquisition:
    """flag_acquisition records unresolved citations as acquisition candidates."""

    async def test_flag_new_citation(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """flag_acquisition adds a new citation and returns updated count."""
        result_str = await handle_flag_acquisition(
            {
                "citations": [
                    {
                        "citation_text": "The Abolition of Man by C.S. Lewis",
                        "probable_work": "The Abolition of Man",
                        "priority": "high",
                        "note": "Frequently cited by Shakespeare in his theological works",
                    }
                ]
            },
            settings=integration_settings,
            storage=clean_storage,
            embedding_provider=None,  # type: ignore[arg-type]
        )
        result = json.loads(result_str)

        assert result["added"] == 1
        assert result["already_flagged"] == 0
        assert "acquisition_list_total" in result
        assert result["acquisition_list_total"] >= 1

    async def test_flag_duplicate_citation(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """Flagging the same citation twice increments already_flagged, not added."""
        citation = {
            "citation_text": "Mere Christianity by C.S. Lewis",
            "priority": "medium",
        }
        # First flag
        await handle_flag_acquisition(
            {"citations": [citation]},
            settings=integration_settings,
            storage=clean_storage,
            embedding_provider=None,  # type: ignore[arg-type]
        )
        # Second flag (duplicate)
        result_str = await handle_flag_acquisition(
            {"citations": [citation]},
            settings=integration_settings,
            storage=clean_storage,
            embedding_provider=None,  # type: ignore[arg-type]
        )
        result = json.loads(result_str)

        assert result["added"] == 0
        assert result["already_flagged"] == 1

    async def test_flag_multiple_citations(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """flag_acquisition handles multiple citations in one call."""
        result_str = await handle_flag_acquisition(
            {
                "citations": [
                    {"citation_text": "Orthodoxy by G.K. Chesterton", "priority": "high"},
                    {"citation_text": "The Everlasting Man by G.K. Chesterton", "priority": "medium"},
                    {"citation_text": "Surprised by Joy by C.S. Lewis", "priority": "low"},
                ]
            },
            settings=integration_settings,
            storage=clean_storage,
            embedding_provider=None,  # type: ignore[arg-type]
        )
        result = json.loads(result_str)

        assert result["added"] == 3
        assert result["acquisition_list_total"] >= 3


# ---------------------------------------------------------------------------
# Full composable pipeline (all 5 steps in sequence)
# ---------------------------------------------------------------------------


@SKIP_NO_DB
@SKIP_NO_ANTHROPIC
class TestFullComposablePipeline:
    """Tests classify→catalog→chunk→detect_links→flag_acquisition as a complete flow."""

    async def test_full_pipeline_sequential(
        self,
        clean_storage: StorageManager,
        integration_settings: Settings,
    ) -> None:
        """Runs all 5 composable tools in order and verifies each step's output."""
        from author_library.embeddings import ProviderRegistry

        temp_path = _write_temp_text(HAMLET_EXCERPT)
        embedding_provider = ProviderRegistry.create(integration_settings)

        try:
            # --- B1: classify_source ---
            classify_str = await handle_classify_source(
                {
                    "file_path": str(temp_path),
                    "subject_author": "shakespeare",
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            classify_result = json.loads(classify_str)
            assert "suggested_class" in classify_result
            suggested_class = classify_result["suggested_class"]
            assert suggested_class in ("primary", "secondary", "contextual", "tertiary", "personal")

            # --- B2: catalog_source (user confirms classification) ---
            catalog_str = await handle_catalog_source(
                {
                    "file_path": str(temp_path),
                    "source_class": "primary",  # user-confirmed
                    "work_type": "other",
                    "metadata_overrides": {
                        "author": "Shakespeare",
                        "title": "Hamlet Full Pipeline Test",
                        "subject_author_id": "shakespeare",
                        "genre_tags": ["drama"],
                        "subject_headings": ["English drama"],
                        "publication_year": 1603,
                        "publisher": "Unknown",
                    },
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            catalog_result = json.loads(catalog_str)
            assert "work_id" in catalog_result
            work_id = catalog_result["work_id"]

            # Verify work was stored
            work = await clean_storage.works.get(work_id)
            assert work is not None

            # --- B3: chunk_source (patch file_path into source_metadata) ---
            source_meta_raw = work.get("source_metadata") or {}
            source_meta = json.loads(source_meta_raw) if isinstance(source_meta_raw, str) else dict(source_meta_raw)
            source_meta["file_path"] = str(temp_path)
            await clean_storage.works.update(work_id, {"source_metadata": source_meta})

            chunk_str = await handle_chunk_source(
                {"work_id": work_id},
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            chunk_result = json.loads(chunk_str)
            assert chunk_result["status"] in ("complete", "partial")
            total_chunks = sum(chunk_result["chunks_created"].values())
            assert total_chunks > 0

            # --- B4: detect_passage_links ---
            links_str = await handle_detect_passage_links(
                {
                    "work_id": work_id,
                    "scan_types": ["explicit_citation"],
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            links_result = json.loads(links_str)
            assert "links_created" in links_result
            # 0 links is valid — no contextual counterparts in clean DB
            assert isinstance(links_result["links_created"]["explicit_citation"], int)

            # --- B5: flag_acquisition ---
            acq_str = await handle_flag_acquisition(
                {
                    "citations": [
                        {
                            "citation_text": "Henry V by William Shakespeare",
                            "probable_work": "Henry V",
                            "priority": "medium",
                            "note": "Referenced in Hamlet full pipeline test",
                        }
                    ]
                },
                settings=integration_settings,
                storage=clean_storage,
                embedding_provider=embedding_provider,
            )
            acq_result = json.loads(acq_str)
            assert acq_result["added"] >= 1 or acq_result["already_flagged"] >= 1

        finally:
            temp_path.unlink(missing_ok=True)
            await embedding_provider.close()
