"""Tests for N3: Batch surfacing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from author_library.surfacing.batch_surfacing import (
    BatchSurfacer,
    BatchSurfacingResult,
)
from author_library.surfacing.connection_scanner import ScanResult, StagedConnection
from author_library.surfacing.pr_content import PRContent


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
    return storage


@pytest.fixture()
def mock_embedding_provider():
    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[0.1] * 1024)
    provider.close = AsyncMock()
    return provider


def _make_scan_result(total: int = 3) -> ScanResult:
    """Create a ScanResult with test connections."""
    conns = []
    for i in range(total):
        conns.append(StagedConnection(
            source_chunk_id=f"src-{i}",
            target_chunk_id=f"tgt-{i}",
            source_work_id="new-work",
            target_work_id=f"existing-work-{i}",
            connection_type="thematic_parallel",
            confidence_level="medium",
            confidence_label="This appears to connect to",
            source_excerpt=f"Source text {i}",
            target_excerpt=f"Target text {i}",
            explanation=f"Thematic parallel {i}.",
        ))
    return ScanResult(
        work_id="new-work",
        connections=conns,
        by_confidence={"medium": conns},
        by_target_work={f"existing-work-{i}": [c] for i, c in enumerate(conns)},
        total_found=total,
    )


class TestBatchSurfacer:
    """Tests for BatchSurfacer pipeline."""

    @pytest.mark.asyncio()
    async def test_surface_after_ingestion_no_connections(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ) -> None:
        """No connections found returns empty result without PR."""
        empty_scan = ScanResult(work_id="new-work", total_found=0)

        surfacer = BatchSurfacer(
            mock_settings, mock_storage, mock_embedding_provider,
        )
        surfacer._scanner.scan_new_connections = AsyncMock(
            return_value=empty_scan,
        )

        result = await surfacer.surface_after_ingestion("new-work")

        assert result.pr_content is None
        assert result.pr_created is False

    @pytest.mark.asyncio()
    async def test_surface_after_ingestion_with_connections(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ) -> None:
        """Connections found generates PR content."""
        scan = _make_scan_result(total=3)

        surfacer = BatchSurfacer(
            mock_settings, mock_storage, mock_embedding_provider,
        )
        surfacer._scanner.scan_new_connections = AsyncMock(return_value=scan)

        result = await surfacer.surface_after_ingestion(
            "new-work",
            work_title="Faith, Hope and Poetry",
        )

        assert result.scan_result is not None
        assert result.scan_result.total_found == 3
        assert result.pr_content is not None
        assert "Faith, Hope and Poetry" in result.pr_content.title

    @pytest.mark.asyncio()
    async def test_below_threshold_skips_pr(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ) -> None:
        """Below min_connections_for_pr skips PR generation."""
        scan = _make_scan_result(total=1)

        surfacer = BatchSurfacer(
            mock_settings, mock_storage, mock_embedding_provider,
        )
        surfacer._scanner.scan_new_connections = AsyncMock(return_value=scan)

        result = await surfacer.surface_after_ingestion(
            "new-work",
            min_connections_for_pr=5,  # Higher than found
        )

        assert result.scan_result is not None
        assert result.pr_content is None

    @pytest.mark.asyncio()
    async def test_scan_error_captured(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ) -> None:
        """Scanner errors are captured in result, not raised."""
        surfacer = BatchSurfacer(
            mock_settings, mock_storage, mock_embedding_provider,
        )
        surfacer._scanner.scan_new_connections = AsyncMock(
            side_effect=RuntimeError("DB connection failed"),
        )

        result = await surfacer.surface_after_ingestion("new-work")
        assert len(result.errors) > 0
        assert "DB connection failed" in result.errors[0]


class TestBatchSurfacingResultSerialization:
    """Test result serialization."""

    def test_to_dict_with_pr(self) -> None:
        """Result with PR content serializes correctly."""
        result = BatchSurfacingResult(
            work_id="test-work",
            scan_result=_make_scan_result(2),
            pr_content=PRContent(
                title="Test PR",
                body="Body",
                affected_notes=["note-1"],
                pr_type="new_connection",
                labels=["parlour/new-connections"],
            ),
            pr_created=True,
            pr_id="123",
            pr_url="https://github.com/test/repo/pull/123",
        )

        d = result.to_dict()
        assert d["work_id"] == "test-work"
        assert d["pr_created"] is True
        assert d["pr_id"] == "123"
        assert d["total_connections"] == 2
        assert d["pr_title"] == "Test PR"

    def test_to_dict_without_pr(self) -> None:
        """Result without PR serializes correctly."""
        result = BatchSurfacingResult(work_id="test-work")
        d = result.to_dict()
        assert d["pr_created"] is False
        assert "pr_id" not in d
