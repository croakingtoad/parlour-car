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
        mock_catalog_entry.author = "Malcolm Guite"
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
            subject_author_id="malcolm-guite",
        )

        # Verify PG author upsert was called
        pg_execute_calls = mock_pg.execute.call_args_list
        author_upsert_calls = [
            c for c in pg_execute_calls
            if "INSERT INTO authors" in str(c)
        ]
        assert len(author_upsert_calls) == 1
        call_args = author_upsert_calls[0]
        assert call_args[0][1] == "malcolm-guite"  # author id
        assert call_args[0][2] == "Malcolm Guite"  # canonical name

        # Verify Neo4j Author node + AUTHORED edge
        neo4j_write_calls = mock_neo4j.execute_write.call_args_list
        author_neo4j_calls = [
            c for c in neo4j_write_calls
            if "Author" in str(c) and "AUTHORED" in str(c)
        ]
        assert len(author_neo4j_calls) == 1
        neo4j_params = author_neo4j_calls[0][0][1]
        assert neo4j_params["author_id"] == "malcolm-guite"
        assert neo4j_params["name"] == "Malcolm Guite"
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
