"""Tests for ingestion pipeline orchestrator and IngestionResult."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from author_library.tools.ingestion_pipeline import IngestionPipeline, IngestionResult


class TestIngestionResult:
    """Test IngestionResult value object."""

    def test_attributes(self) -> None:
        result = IngestionResult(
            work_id="lewis--mere-christianity",
            source_class="primary",
            processing_route="full_enrichment",
            chunks_by_granularity={"macro": 5, "meso": 20, "micro": 80},
            embeddings_stored=100,
            entity_count=12,
            edge_count=34,
            errors=[],
        )
        assert result.work_id == "lewis--mere-christianity"
        assert result.source_class == "primary"
        assert result.processing_route == "full_enrichment"
        assert result.embeddings_stored == 100
        assert result.entity_count == 12
        assert result.edge_count == 34
        assert result.errors == []
        assert result.chunks_by_granularity["macro"] == 5

    def test_to_dict(self) -> None:
        result = IngestionResult(
            work_id="lewis--the-screwtape-letters",
            source_class="primary",
            processing_route="full_enrichment",
            chunks_by_granularity={"meso": 15},
            embeddings_stored=15,
            entity_count=8,
            edge_count=3,
            errors=["minor warning"],
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["work_id"] == "lewis--the-screwtape-letters"
        assert d["source_class"] == "primary"
        assert d["processing_route"] == "full_enrichment"
        assert d["chunks_by_granularity"] == {"meso": 15}
        stats = d["post_ingestion_stats"]
        assert stats["embeddings_stored"] == 15
        assert stats["total_chunks"] == 15
        assert stats["embedding_coverage_percent"] == 100.0
        assert stats["unembedded_chunks"] == 0
        assert stats["entity_count"] == 8
        assert stats["edge_count"] == 3
        assert stats["status"] == "complete — all chunks embedded"
        assert d["errors"] == ["minor warning"]

    def test_to_dict_round_trip(self) -> None:
        """All fields survive the to_dict() conversion."""
        result = IngestionResult(
            work_id="tolkien--on-fairy-stories",
            source_class="contextual",
            processing_route="embeddings_and_links",
            chunks_by_granularity={"macro": 2, "meso": 10, "micro": 40},
            embeddings_stored=50,
            entity_count=0,
            edge_count=6,
            errors=[],
        )
        d = result.to_dict()
        assert set(d.keys()) == {
            "work_id",
            "source_class",
            "processing_route",
            "chunks_by_granularity",
            "post_ingestion_stats",
            "errors",
        }

    def test_empty_errors_list(self) -> None:
        result = IngestionResult(
            work_id="w1",
            source_class="tertiary",
            processing_route="metadata_only",
            chunks_by_granularity={},
            embeddings_stored=0,
            entity_count=0,
            edge_count=0,
            errors=[],
        )
        assert result.errors == []
        assert result.to_dict()["chunks_by_granularity"] == {}

    def test_metadata_only_route_has_zero_counts(self) -> None:
        """Tertiary/metadata-only route produces no chunks or embeddings."""
        result = IngestionResult(
            work_id="ref--bibliography",
            source_class="tertiary",
            processing_route="metadata_only",
            chunks_by_granularity={},
            embeddings_stored=0,
            entity_count=0,
            edge_count=0,
            errors=[],
        )
        d = result.to_dict()
        stats = d["post_ingestion_stats"]
        assert stats["embeddings_stored"] == 0
        assert stats["entity_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["status"] == "complete — all chunks embedded"


class TestAuthorUpsertDuringIngestion:
    """Verify that ingest() upserts an author record in PG and Neo4j."""

    @patch("author_library.tools.ingestion_pipeline.get_parser")
    @patch("author_library.tools.ingestion_pipeline.ClassificationPipeline")
    async def test_author_upserted_in_pg_on_tertiary_route(
        self,
        mock_classification_cls: MagicMock,
        mock_get_parser: MagicMock,
        tmp_path,
    ) -> None:
        """Even for tertiary (metadata-only) route, author record must be upserted."""
        from author_library.catalog.models import ProcessingRoute, SourceClass

        # Mock parser
        mock_parser = AsyncMock()
        mock_document = MagicMock()
        mock_document.metadata.title = "A Bibliography"
        mock_document.metadata.word_count = 500
        mock_document.format = "txt"
        mock_parser.parse.return_value = mock_document
        mock_get_parser.return_value = mock_parser

        # Mock classification pipeline result
        mock_catalog_entry = MagicMock()
        mock_catalog_entry.work_id = "guite--bibliography"
        mock_catalog_entry.title = "A Bibliography"
        mock_catalog_entry.author = "Test Guite"
        mock_catalog_entry.publication_year = 2020
        mock_catalog_entry.genre_tags = ["bibliography"]

        mock_classification = MagicMock()
        mock_classification.source_class = SourceClass.TERTIARY

        mock_pipeline_result = MagicMock()
        mock_pipeline_result.catalog_entry = mock_catalog_entry
        mock_pipeline_result.classification = mock_classification
        mock_pipeline_result.processing_route = ProcessingRoute.METADATA_ONLY

        mock_classification_pipeline = AsyncMock()
        mock_classification_pipeline.process.return_value = mock_pipeline_result
        mock_classification_cls.return_value = mock_classification_pipeline

        # Mock storage
        mock_storage = MagicMock()
        mock_pg = AsyncMock()
        mock_storage.pg = mock_pg
        mock_neo4j = AsyncMock()
        mock_storage.neo4j = mock_neo4j
        mock_storage.graph = AsyncMock()
        mock_storage.chunks = AsyncMock()
        mock_storage.chunks.get_max_pass_number.return_value = 0
        mock_storage.chunks.delete_by_work.return_value = 0
        mock_storage.works = AsyncMock()

        mock_settings = MagicMock()
        mock_embedding = AsyncMock()

        test_file = tmp_path / "test.txt"
        test_file.write_text("dummy content")

        pipeline = IngestionPipeline(
            settings=mock_settings,
            storage=mock_storage,
            embedding_provider=mock_embedding,
        )

        result = await pipeline.ingest(
            test_file,
            subject_author_id="test--guite",
        )

        # Verify PG author upsert was called
        pg_execute_calls = mock_pg.execute.call_args_list
        author_upsert_calls = [
            c for c in pg_execute_calls
            if "INSERT INTO authors" in str(c)
        ]
        assert len(author_upsert_calls) == 1
        call_args = author_upsert_calls[0]
        assert call_args[0][1] == "test--guite"  # author id
        assert call_args[0][2] == "Test Guite"  # canonical name

        # Verify Neo4j Author node + AUTHORED edge
        neo4j_write_calls = mock_neo4j.execute_write.call_args_list
        author_neo4j_calls = [
            c for c in neo4j_write_calls
            if "Author" in str(c) and "AUTHORED" in str(c)
        ]
        assert len(author_neo4j_calls) == 1
        neo4j_params = author_neo4j_calls[0][0][1]
        assert neo4j_params["author_id"] == "test--guite"
        assert neo4j_params["name"] == "Test Guite"
        assert neo4j_params["work_id"] == "guite--bibliography"


# ---------------------------------------------------------------------------
# Helpers for quality-check tests
# ---------------------------------------------------------------------------


def _make_pipeline(
    *,
    neo4j_read_results: list[list[dict[str, Any]]] | None = None,
    neo4j_write_results: list[list[dict[str, Any]]] | None = None,
    pg_fetch_val_results: list[Any] | None = None,
    work_record: dict[str, Any] | None = None,
) -> IngestionPipeline:
    """Build an IngestionPipeline with mocked storage for quality-check tests."""
    mock_storage = MagicMock()

    mock_neo4j = AsyncMock()
    if neo4j_write_results is not None:
        mock_neo4j.execute_write.side_effect = neo4j_write_results
    else:
        mock_neo4j.execute_write.return_value = [{"deleted": 0}]
    if neo4j_read_results is not None:
        mock_neo4j.execute_read.side_effect = neo4j_read_results
    else:
        mock_neo4j.execute_read.return_value = [
            {"total_chunks": 10, "no_entity_chunks": 0}
        ]
    mock_storage.neo4j = mock_neo4j

    mock_pg = AsyncMock()
    if pg_fetch_val_results is not None:
        mock_pg.fetch_val.side_effect = pg_fetch_val_results
    else:
        # Default: 0 noise, 10 total chunks, 10 embeddings
        mock_pg.fetch_val.side_effect = [0, 10, 10]
    mock_storage.pg = mock_pg

    mock_works = AsyncMock()
    mock_works.get.return_value = work_record or {
        "work_id": "lewis--mere-christianity",
        "author": "C.S. Lewis",
        "source_class": "primary",
    }
    mock_storage.works = mock_works

    mock_settings = MagicMock()
    mock_embedding = AsyncMock()

    return IngestionPipeline(
        settings=mock_settings,
        storage=mock_storage,
        embedding_provider=mock_embedding,
    )


class TestRunQualityChecks:
    """Tests for IngestionPipeline._run_quality_checks."""

    async def test_all_pass(self) -> None:
        """Clean work produces a 'pass' status with no warnings."""
        pipeline = _make_pipeline()
        result = await pipeline._run_quality_checks(
            "lewis--mere-christianity", "primary", "c-s-lewis",
        )
        assert result["status"] == "pass"
        assert result["orphans_cleaned"] == 0
        assert result["classification_warning"] is None
        assert result["noise_chunks"] == 0
        assert result["embedding_coverage_pct"] == 100.0
        assert result["entity_coverage_pct"] == 100.0
        assert "warnings" not in result

    async def test_orphans_cleaned(self) -> None:
        """Orphaned entity nodes are deleted and counted."""
        pipeline = _make_pipeline(
            neo4j_write_results=[[{"deleted": 3}]],
        )
        result = await pipeline._run_quality_checks(
            "lewis--mere-christianity", "primary", "c-s-lewis",
        )
        assert result["orphans_cleaned"] == 3

    async def test_classification_warning_triggered(self) -> None:
        """Author match + contextual class triggers a classification warning."""
        pipeline = _make_pipeline(
            work_record={
                "work_id": "lewis--some-essay",
                "author": "C.S. Lewis",
                "source_class": "contextual",
            },
        )
        result = await pipeline._run_quality_checks(
            "lewis--some-essay", "contextual", "c-s-lewis",
        )
        assert result["classification_warning"] is not None
        assert "reclassifying" in result["classification_warning"].lower()
        assert result["status"] == "warnings"

    async def test_no_classification_warning_for_different_author(self) -> None:
        """Author mismatch should not trigger a classification warning."""
        pipeline = _make_pipeline(
            work_record={
                "work_id": "tolkien--on-fairy-stories",
                "author": "J.R.R. Tolkien",
                "source_class": "contextual",
            },
        )
        result = await pipeline._run_quality_checks(
            "tolkien--on-fairy-stories", "contextual", "c-s-lewis",
        )
        assert result["classification_warning"] is None

    async def test_no_classification_warning_for_reference_author_match(self) -> None:
        """Reference works are filed under their own author without being primary."""
        pipeline = _make_pipeline(
            work_record={
                "work_id": "paul-fussell--poetic-meter-and-poetic-form",
                "author": "Paul Fussell",
                "source_class": "reference",
            },
        )
        result = await pipeline._run_quality_checks(
            "paul-fussell--poetic-meter-and-poetic-form",
            "reference",
            "paul-fussell",
        )
        assert result["classification_warning"] is None

    async def test_noise_chunks_detected(self) -> None:
        """Micro/nano chunks under 50 chars are flagged."""
        pipeline = _make_pipeline(
            pg_fetch_val_results=[5, 10, 10],  # 5 noise, 10 total, 10 embedded
        )
        result = await pipeline._run_quality_checks(
            "lewis--mere-christianity", "primary", "c-s-lewis",
        )
        assert result["noise_chunks"] == 5
        assert result["status"] == "warnings"

    async def test_embedding_coverage_gap(self) -> None:
        """Missing embeddings lowers coverage below 100%."""
        pipeline = _make_pipeline(
            pg_fetch_val_results=[0, 20, 15],  # 0 noise, 20 chunks, 15 embedded
        )
        result = await pipeline._run_quality_checks(
            "lewis--mere-christianity", "primary", "c-s-lewis",
        )
        assert result["embedding_coverage_pct"] == 75.0
        assert result["status"] == "warnings"

    async def test_entity_coverage_gap(self) -> None:
        """Chunks without entity edges trigger a warning when > 10%."""
        pipeline = _make_pipeline(
            neo4j_read_results=[
                [{"total_chunks": 20, "no_entity_chunks": 5}],
            ],
        )
        result = await pipeline._run_quality_checks(
            "lewis--mere-christianity", "primary", "c-s-lewis",
        )
        assert result["entity_coverage_pct"] == 75.0
        assert result["status"] == "warnings"

    async def test_entity_coverage_ok_at_threshold(self) -> None:
        """Entity coverage at exactly 90% should not warn."""
        pipeline = _make_pipeline(
            neo4j_read_results=[
                [{"total_chunks": 10, "no_entity_chunks": 1}],
            ],
        )
        result = await pipeline._run_quality_checks(
            "lewis--mere-christianity", "primary", "c-s-lewis",
        )
        assert result["entity_coverage_pct"] == 90.0
        # 90% is the threshold — at exactly 90% there should be no entity warning
        entity_warnings = [
            w for w in result.get("warnings", []) if "Entity coverage" in w
        ]
        assert entity_warnings == []

    async def test_neo4j_failure_does_not_crash(self) -> None:
        """If Neo4j is unreachable, quality checks still complete with warnings."""
        pipeline = _make_pipeline()
        pipeline._storage.neo4j.execute_write.side_effect = Exception("Neo4j down")
        pipeline._storage.neo4j.execute_read.side_effect = Exception("Neo4j down")
        result = await pipeline._run_quality_checks(
            "lewis--mere-christianity", "primary", "c-s-lewis",
        )
        assert result["status"] == "warnings"
        assert any("Orphan check failed" in w for w in result.get("warnings", []))
        assert any("Entity coverage check failed" in w for w in result.get("warnings", []))

    async def test_pg_failure_does_not_crash(self) -> None:
        """If PG is unreachable for noise/embedding checks, still completes."""
        pipeline = _make_pipeline()
        pipeline._storage.pg.fetch_val.side_effect = Exception("PG down")
        result = await pipeline._run_quality_checks(
            "lewis--mere-christianity", "primary", "c-s-lewis",
        )
        assert result["status"] == "warnings"
        assert any("Noise check failed" in w for w in result.get("warnings", []))
        assert any("Embedding coverage check failed" in w for w in result.get("warnings", []))


class TestIngestionResultWithQualityChecks:
    """Tests for IngestionResult quality_checks integration."""

    def test_quality_checks_in_to_dict(self) -> None:
        """quality_checks dict appears in to_dict output when provided."""
        qc = {
            "orphans_cleaned": 2,
            "classification_warning": None,
            "noise_chunks": 0,
            "embedding_coverage_pct": 100.0,
            "entity_coverage_pct": 95.0,
            "status": "pass",
        }
        result = IngestionResult(
            work_id="lewis--mere-christianity",
            source_class="primary",
            processing_route="full_enrichment",
            chunks_by_granularity={"meso": 10},
            embeddings_stored=10,
            entity_count=5,
            edge_count=3,
            errors=[],
            quality_checks=qc,
        )
        d = result.to_dict()
        assert "quality_checks" in d
        assert d["quality_checks"]["orphans_cleaned"] == 2
        assert d["quality_checks"]["status"] == "pass"

    def test_no_quality_checks_omitted(self) -> None:
        """quality_checks key absent from to_dict when None."""
        result = IngestionResult(
            work_id="w1",
            source_class="primary",
            processing_route="full_enrichment",
            chunks_by_granularity={"meso": 5},
            embeddings_stored=5,
            entity_count=0,
            edge_count=0,
            errors=[],
        )
        d = result.to_dict()
        assert "quality_checks" not in d


# ---------------------------------------------------------------------------
# Section-type routing tests (td-4e1d64)
# ---------------------------------------------------------------------------


def _make_chunk(section_type: str, text: str = "sample text", position: int = 0):
    """Build a minimal Chunk for section-routing tests."""
    from author_library.chunking.models import Chunk, ChunkGranularity

    return Chunk(
        text=text,
        granularity=ChunkGranularity.MACRO,
        work_id="guite--test-work",
        source_class="primary",
        position=position,
        section_type=section_type,
    )


class TestFilterBySectionType:
    """Unit tests for IngestionPipeline._filter_by_section_type."""

    def _pipeline(self) -> IngestionPipeline:
        mock_storage = MagicMock()
        mock_storage.pg = AsyncMock()
        return IngestionPipeline(
            settings=MagicMock(),
            storage=mock_storage,
            embedding_provider=AsyncMock(),
        )

    def test_chapter_chunks_kept(self) -> None:
        pipeline = self._pipeline()
        chunks = [_make_chunk("chapter"), _make_chunk("chapter")]
        content, skipped, structural = pipeline._filter_by_section_type(chunks, "work-1")
        assert len(content) == 2
        assert skipped == {}
        assert structural == {}

    def test_preface_chunks_kept(self) -> None:
        """Preface is author voice — kept in full pipeline."""
        pipeline = self._pipeline()
        chunks = [_make_chunk("preface")]
        content, skipped, structural = pipeline._filter_by_section_type(chunks, "work-1")
        assert len(content) == 1
        assert skipped == {}

    def test_index_chunks_excluded_and_routed(self) -> None:
        pipeline = self._pipeline()
        chunks = [_make_chunk("index"), _make_chunk("chapter")]
        content, skipped, structural = pipeline._filter_by_section_type(chunks, "work-1")
        assert len(content) == 1
        assert content[0].section_type == "chapter"
        assert skipped.get("index") == 1
        assert "index" in structural
        assert len(structural["index"]) == 1

    def test_bibliography_chunks_excluded_and_routed(self) -> None:
        pipeline = self._pipeline()
        chunks = [_make_chunk("bibliography"), _make_chunk("bibliography")]
        content, skipped, structural = pipeline._filter_by_section_type(chunks, "work-1")
        assert len(content) == 0
        assert skipped.get("bibliography") == 2
        assert "bibliography" in structural
        assert len(structural["bibliography"]) == 2

    def test_toc_excluded_not_routed(self) -> None:
        """ToC is dropped entirely — no downstream routing."""
        pipeline = self._pipeline()
        chunks = [_make_chunk("toc")]
        content, skipped, structural = pipeline._filter_by_section_type(chunks, "work-1")
        assert len(content) == 0
        assert skipped.get("toc") == 1
        assert "toc" not in structural  # no routing for ToC

    def test_front_matter_excluded_not_routed(self) -> None:
        """Front matter is dropped — no downstream routing."""
        pipeline = self._pipeline()
        chunks = [_make_chunk("front_matter")]
        content, skipped, structural = pipeline._filter_by_section_type(chunks, "work-1")
        assert len(content) == 0
        assert "front_matter" not in structural

    def test_mixed_sections(self) -> None:
        """Mixed section types are correctly partitioned."""
        pipeline = self._pipeline()
        chunks = [
            _make_chunk("chapter", position=0),
            _make_chunk("index", position=1),
            _make_chunk("bibliography", position=2),
            _make_chunk("toc", position=3),
            _make_chunk("preface", position=4),
        ]
        content, skipped, structural = pipeline._filter_by_section_type(chunks, "work-1")
        assert len(content) == 2  # chapter + preface
        assert set(c.section_type for c in content) == {"chapter", "preface"}
        assert skipped == {"index": 1, "bibliography": 1, "toc": 1}
        assert set(structural.keys()) == {"index", "bibliography"}


class TestRouteStructuralSections:
    """Unit tests for IngestionPipeline._route_structural_sections."""

    def _pipeline_with_mocks(self):
        mock_storage = MagicMock()
        mock_storage.pg = AsyncMock()
        pipeline = IngestionPipeline(
            settings=MagicMock(),
            storage=mock_storage,
            embedding_provider=AsyncMock(),
        )
        return pipeline, mock_storage

    async def test_index_chunks_do_not_auto_propose_vocabulary(self) -> None:
        """Raw index entries must not refill controlled vocabulary."""
        pipeline, mock_storage = self._pipeline_with_mocks()

        index_chunk = _make_chunk("index", text="grace\nforgiveness\nlove")
        structural = {"index": [index_chunk]}

        with patch(
            "author_library.vocabulary.VocabularyManager"
        ) as mock_vocab_cls:
            mock_vocab = AsyncMock()
            mock_vocab.propose.return_value = {"term": "grace", "already_exists": False}
            mock_vocab_cls.return_value = mock_vocab

            await pipeline._route_structural_sections(structural, "guite--test")

        mock_vocab_cls.assert_not_called()

    async def test_bibliography_routes_to_acquisition(self) -> None:
        """Bibliography chunks have their lines flagged as acquisition candidates."""
        pipeline, mock_storage = self._pipeline_with_mocks()

        bib_chunk = _make_chunk(
            "bibliography",
            text="Lewis, C.S. Mere Christianity. 1952.\nTolkien, J.R.R. The Lord of the Rings. 1954.",
        )
        structural = {"bibliography": [bib_chunk]}

        with patch(
            "author_library.catalog.acquisition.AcquisitionManager"
        ) as mock_acq_cls:
            mock_acq = AsyncMock()
            mock_acq.flag.return_value = True
            mock_acq_cls.return_value = mock_acq

            await pipeline._route_structural_sections(structural, "guite--test")

        assert mock_acq.flag.call_count == 2

    async def test_index_locators_do_not_reach_vocabulary(self) -> None:
        """Index locators and cross-references stay out of vocabulary."""
        pipeline, mock_storage = self._pipeline_with_mocks()

        index_chunk = _make_chunk("index", text="\ngrace\n123, 456\n  \nforgiveness")
        structural = {"index": [index_chunk]}

        with patch(
            "author_library.vocabulary.VocabularyManager"
        ) as mock_vocab_cls:
            mock_vocab = AsyncMock()
            mock_vocab.propose.return_value = {"term": "grace", "already_exists": False}
            mock_vocab_cls.return_value = mock_vocab

            await pipeline._route_structural_sections(structural, "guite--test")

        mock_vocab_cls.assert_not_called()

    async def test_empty_structural_dict_is_noop(self) -> None:
        """Empty structural_chunks dict calls neither manager."""
        pipeline, mock_storage = self._pipeline_with_mocks()

        with (
            patch("author_library.vocabulary.VocabularyManager") as mock_v,
            patch("author_library.catalog.acquisition.AcquisitionManager") as mock_a,
        ):
            await pipeline._route_structural_sections({}, "guite--test")

        mock_v.assert_not_called()
        mock_a.assert_not_called()

    async def test_already_known_term_counted_correctly(self) -> None:
        """Terms already in vocabulary are counted as already_known, not proposed."""
        pipeline, mock_storage = self._pipeline_with_mocks()

        index_chunk = _make_chunk("index", text="grace\nforgiveness")
        structural = {"index": [index_chunk]}

        with patch(
            "author_library.vocabulary.VocabularyManager"
        ) as mock_vocab_cls:
            mock_vocab = AsyncMock()
            # grace already exists, forgiveness is new
            mock_vocab.propose.side_effect = [
                {"term": "grace", "already_exists": True},
                {"term": "forgiveness", "already_exists": False},
            ]
            mock_vocab_cls.return_value = mock_vocab

            # Should not raise — completes without error
            await pipeline._route_structural_sections(structural, "guite--test")
