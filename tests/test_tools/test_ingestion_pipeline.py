"""Tests for ingestion pipeline orchestrator and IngestionResult."""

from __future__ import annotations

from author_library.tools.ingestion_pipeline import IngestionResult


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
