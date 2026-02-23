"""Tests for N1: Post-ingestion connection scanner."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from author_library.surfacing.connection_scanner import (
    ConnectionScanner,
    ScanResult,
    StagedConnection,
)
from author_library.surfacing.confidence import ConfidenceLevel
from author_library.surfacing.related_content import (
    ConnectionType,
    RelatedContentResult,
    RelatedItem,
)


@pytest.fixture()
def mock_settings():
    settings = MagicMock()
    settings.llm.query_model = "claude-sonnet-4-5-20250929"
    settings.api_keys.anthropic_api_key.get_secret_value.return_value = "test-key"
    return settings


@pytest.fixture()
def mock_storage():
    storage = MagicMock()
    storage.chunks = MagicMock()
    storage.graph = MagicMock()
    storage.works = MagicMock()
    return storage


@pytest.fixture()
def mock_embedding_provider():
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[0.1] * 1024)
    provider.close = AsyncMock()
    return provider


class TestConnectionScanner:
    """Tests for ConnectionScanner.scan_new_connections."""

    @pytest.mark.asyncio()
    async def test_no_chunks_returns_empty(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ) -> None:
        """Work with no chunks returns empty result."""
        mock_storage.chunks.list_by_work = AsyncMock(return_value=[])

        scanner = ConnectionScanner(
            mock_settings, mock_storage, mock_embedding_provider,
        )
        result = await scanner.scan_new_connections("test-work-id")

        assert result.total_found == 0
        assert result.connections == []

    @pytest.mark.asyncio()
    async def test_scan_finds_connections(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ) -> None:
        """Scanner discovers connections across existing works."""
        # Setup: one chunk in the new work
        mock_storage.chunks.list_by_work = AsyncMock(return_value=[
            {"id": "chunk-1", "text": "The imagination is the primary faculty..."},
        ])
        mock_storage.graph.get_passage_links_for_work = AsyncMock(return_value=[])

        staged_conn = StagedConnection(
            source_chunk_id="chunk-1",
            target_chunk_id="existing-chunk-1",
            source_work_id="test-work-id",
            target_work_id="existing-work-1",
            connection_type="thematic_parallel",
            confidence_level="medium",
            confidence_label="This appears to connect to",
            source_excerpt="The imagination is the primary faculty...",
            target_excerpt="Coleridge argued that imagination...",
            explanation="Both passages explore shared themes: imagination.",
        )

        scanner = ConnectionScanner(
            mock_settings, mock_storage, mock_embedding_provider,
        )
        scanner._scan_single_chunk = AsyncMock(return_value=[staged_conn])
        result = await scanner.scan_new_connections("test-work-id")

        assert result.total_found == 1
        assert result.connections[0].connection_type == "thematic_parallel"
        assert "imagination" in result.connections[0].explanation

    @pytest.mark.asyncio()
    async def test_skips_self_references(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ) -> None:
        """Scanner skips connections within the same work."""
        mock_storage.chunks.list_by_work = AsyncMock(return_value=[
            {"id": "chunk-1", "text": "Some text"},
        ])
        mock_storage.graph.get_passage_links_for_work = AsyncMock(return_value=[])

        # Self-referencing item should be filtered out
        self_item = RelatedItem(
            chunk_id="chunk-2",
            work_id="test-work-id",  # Same work!
            text="Self-referencing text",
            source_class="primary",
            granularity="meso",
            connection_type=ConnectionType.VECTOR_SIMILARITY,
            relevance_score=0.9,
        )
        mock_result = RelatedContentResult(
            context_chunk_id="chunk-1",
            context_work_id="test-work-id",
            items=[self_item],
            strategies_used=["vector_similarity"],
        )

        with patch(
            "author_library.surfacing.connection_scanner.RelatedContentFinder",
        ) as MockFinder:
            instance = MockFinder.return_value
            instance.find_related = AsyncMock(return_value=mock_result)

            scanner = ConnectionScanner(
                mock_settings, mock_storage, mock_embedding_provider,
            )
            result = await scanner.scan_new_connections("test-work-id")

        assert result.total_found == 0

    @pytest.mark.asyncio()
    async def test_skips_existing_links(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ) -> None:
        """Scanner skips already-linked chunk pairs."""
        mock_storage.chunks.list_by_work = AsyncMock(return_value=[
            {"id": "chunk-1", "text": "Some text"},
        ])
        # This link already exists
        mock_storage.graph.get_passage_links_for_work = AsyncMock(return_value=[
            {"source_chunk_id": "chunk-1", "target_chunk_id": "existing-chunk-1"},
        ])

        existing_item = RelatedItem(
            chunk_id="existing-chunk-1",
            work_id="other-work",
            text="Already linked text",
            source_class="secondary",
            granularity="meso",
            connection_type=ConnectionType.PASSAGE_LINK,
            relevance_score=0.95,
        )
        mock_result = RelatedContentResult(
            context_chunk_id="chunk-1",
            context_work_id="test-work-id",
            items=[existing_item],
            strategies_used=["passage_links"],
        )

        with patch(
            "author_library.surfacing.connection_scanner.RelatedContentFinder",
        ) as MockFinder:
            instance = MockFinder.return_value
            instance.find_related = AsyncMock(return_value=mock_result)

            scanner = ConnectionScanner(
                mock_settings, mock_storage, mock_embedding_provider,
            )
            result = await scanner.scan_new_connections("test-work-id")

        assert result.total_found == 0


class TestScanResultSerialization:
    """Test ScanResult serialization."""

    def test_to_dict(self) -> None:
        """ScanResult serializes to dict correctly."""
        conn = StagedConnection(
            source_chunk_id="src-1",
            target_chunk_id="tgt-1",
            source_work_id="work-1",
            target_work_id="work-2",
            connection_type="thematic_parallel",
            confidence_level="high",
            confidence_label="This directly engages with",
            source_excerpt="Source text...",
            target_excerpt="Target text...",
            explanation="Shared themes.",
        )
        result = ScanResult(
            work_id="work-1",
            connections=[conn],
            by_confidence={"high": [conn]},
            by_target_work={"work-2": [conn]},
            total_found=1,
        )
        d = result.to_dict()

        assert d["work_id"] == "work-1"
        assert d["total_found"] == 1
        assert len(d["by_confidence"]["high"]) == 1
        assert d["by_target_work"]["work-2"] == 1

    def test_empty_result_serializes(self) -> None:
        """Empty ScanResult serializes without errors."""
        result = ScanResult(work_id="empty-work")
        d = result.to_dict()
        assert d["total_found"] == 0
        assert d["errors"] == []
