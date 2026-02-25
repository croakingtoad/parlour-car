"""Tests for surfacing wiring — verify ConnectionScanner and BatchSurfacer
are called from the ingestion pipeline and composable ingestion tools.

Tests cover:
  - task_surface_connections arq task wraps BatchSurfacer correctly
  - IngestionPipeline.ingest() calls _surface_connections after passage linking
  - handle_detect_passage_links triggers surfacing after link detection
  - TaskQueue.enqueue_surface_connections enqueues correctly
  - Worker registers task_surface_connections
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from author_library.surfacing.batch_surfacing import BatchSurfacingResult
from author_library.surfacing.connection_scanner import ScanResult, StagedConnection
from author_library.surfacing.pr_content import PRContent
from author_library.tasks import task_surface_connections
from author_library.worker import WorkerSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _make_batch_result(total: int = 3) -> BatchSurfacingResult:
    """Create a BatchSurfacingResult with PR content."""
    scan = _make_scan_result(total)
    return BatchSurfacingResult(
        work_id="new-work",
        scan_result=scan,
        pr_content=PRContent(
            title="New connections found after ingesting Test Work",
            body="## New Connections\n\n3 connections found.",
            affected_notes=["existing-work-0", "existing-work-1", "existing-work-2"],
            pr_type="new_connection",
            labels=["parlour/new-connections"],
        ),
    )


# ---------------------------------------------------------------------------
# Worker registration
# ---------------------------------------------------------------------------


class TestWorkerRegistration:
    """Verify task_surface_connections is registered in the arq worker."""

    def test_surface_connections_registered_in_worker(self) -> None:
        func_names = [f.__name__ for f in WorkerSettings.functions]
        assert "task_surface_connections" in func_names

    def test_worker_has_all_four_tasks(self) -> None:
        func_names = [f.__name__ for f in WorkerSettings.functions]
        assert "task_ingest_book" in func_names
        assert "task_ingest_corpus" in func_names
        assert "task_process_capture" in func_names
        assert "task_surface_connections" in func_names


# ---------------------------------------------------------------------------
# task_surface_connections arq task
# ---------------------------------------------------------------------------


class TestTaskSurfaceConnections:
    """Tests for the arq task wrapper."""

    @pytest.mark.asyncio()
    async def test_task_calls_batch_surfacer(self) -> None:
        """task_surface_connections delegates to BatchSurfacer.surface_after_ingestion."""
        expected_result = _make_batch_result()

        ctx: dict[str, Any] = {
            "settings": MagicMock(),
            "storage": MagicMock(),
            "embedding_provider": MagicMock(),
        }

        with patch(
            "author_library.surfacing.batch_surfacing.BatchSurfacer",
        ) as MockSurfacer:
            instance = MockSurfacer.return_value
            instance.surface_after_ingestion = AsyncMock(return_value=expected_result)

            result = await task_surface_connections(
                ctx,
                work_id="new-work",
                work_title="Test Work",
                work_author="Test Author",
            )

        # Verify BatchSurfacer was instantiated with correct deps
        MockSurfacer.assert_called_once_with(
            settings=ctx["settings"],
            storage=ctx["storage"],
            embedding_provider=ctx["embedding_provider"],
        )

        # Verify surface_after_ingestion was called with correct args
        instance.surface_after_ingestion.assert_awaited_once_with(
            "new-work",
            work_title="Test Work",
            work_author="Test Author",
            confidence_threshold=0.4,
            min_connections_for_pr=1,
        )

        # Verify result is serialized
        assert result["work_id"] == "new-work"
        assert result["total_connections"] == 3

    @pytest.mark.asyncio()
    async def test_task_no_connections_found(self) -> None:
        """Task returns empty result when no connections found."""
        empty_result = BatchSurfacingResult(
            work_id="lonely-work",
            scan_result=ScanResult(work_id="lonely-work", total_found=0),
        )

        ctx: dict[str, Any] = {
            "settings": MagicMock(),
            "storage": MagicMock(),
            "embedding_provider": MagicMock(),
        }

        with patch(
            "author_library.surfacing.batch_surfacing.BatchSurfacer",
        ) as MockSurfacer:
            instance = MockSurfacer.return_value
            instance.surface_after_ingestion = AsyncMock(return_value=empty_result)

            result = await task_surface_connections(
                ctx,
                work_id="lonely-work",
            )

        assert result["work_id"] == "lonely-work"
        assert result["pr_created"] is False

    @pytest.mark.asyncio()
    async def test_task_passes_custom_thresholds(self) -> None:
        """Task forwards confidence_threshold and min_connections_for_pr."""
        empty_result = BatchSurfacingResult(
            work_id="test-work",
            scan_result=ScanResult(work_id="test-work", total_found=0),
        )

        ctx: dict[str, Any] = {
            "settings": MagicMock(),
            "storage": MagicMock(),
            "embedding_provider": MagicMock(),
        }

        with patch(
            "author_library.surfacing.batch_surfacing.BatchSurfacer",
        ) as MockSurfacer:
            instance = MockSurfacer.return_value
            instance.surface_after_ingestion = AsyncMock(return_value=empty_result)

            await task_surface_connections(
                ctx,
                work_id="test-work",
                confidence_threshold=0.7,
                min_connections_for_pr=5,
            )

        instance.surface_after_ingestion.assert_awaited_once_with(
            "test-work",
            work_title="",
            work_author="",
            confidence_threshold=0.7,
            min_connections_for_pr=5,
        )


# ---------------------------------------------------------------------------
# IngestionPipeline._surface_connections wiring
# ---------------------------------------------------------------------------


class TestIngestionPipelineSurfacingWiring:
    """Verify IngestionPipeline calls surfacing after passage linking."""

    @pytest.mark.asyncio()
    async def test_surface_connections_method_exists(self) -> None:
        """IngestionPipeline has _surface_connections method."""
        from author_library.tools.ingestion_pipeline import IngestionPipeline

        pipeline = IngestionPipeline(
            settings=MagicMock(),
            storage=MagicMock(),
            embedding_provider=MagicMock(),
        )
        assert hasattr(pipeline, "_surface_connections")
        assert callable(pipeline._surface_connections)

    @pytest.mark.asyncio()
    async def test_surface_connections_calls_batch_surfacer(self) -> None:
        """_surface_connections delegates to BatchSurfacer."""
        from author_library.tools.ingestion_pipeline import IngestionPipeline

        expected_result = _make_batch_result()

        pipeline = IngestionPipeline(
            settings=MagicMock(),
            storage=MagicMock(),
            embedding_provider=MagicMock(),
        )

        with patch(
            "author_library.surfacing.batch_surfacing.BatchSurfacer",
        ) as MockSurfacer:
            instance = MockSurfacer.return_value
            instance.surface_after_ingestion = AsyncMock(return_value=expected_result)

            result = await pipeline._surface_connections(
                work_id="test-work",
                work_title="Test Work",
                work_author="Test Author",
            )

        assert result.scan_result is not None
        assert result.scan_result.total_found == 3
        instance.surface_after_ingestion.assert_awaited_once_with(
            "test-work",
            work_title="Test Work",
            work_author="Test Author",
        )


# ---------------------------------------------------------------------------
# TaskQueue.enqueue_surface_connections
# ---------------------------------------------------------------------------


class TestTaskQueueSurfacing:
    """Verify TaskQueue has enqueue_surface_connections method."""

    def test_enqueue_method_exists(self) -> None:
        from author_library.queue import TaskQueue

        tq = TaskQueue()
        assert hasattr(tq, "enqueue_surface_connections")
        assert callable(tq.enqueue_surface_connections)

    @pytest.mark.asyncio()
    async def test_enqueue_returns_none_without_pool(self) -> None:
        from author_library.queue import TaskQueue

        tq = TaskQueue()
        result = await tq.enqueue_surface_connections(
            work_id="test-work",
            work_title="Test Work",
        )
        assert result is None

    @pytest.mark.asyncio()
    async def test_enqueue_with_pool_calls_enqueue_job(self) -> None:
        from author_library.queue import TaskQueue

        tq = TaskQueue()
        mock_pool = MagicMock()
        mock_job = MagicMock()
        mock_job.job_id = "test-job-123"
        mock_pool.enqueue_job = AsyncMock(return_value=mock_job)
        tq._pool = mock_pool

        result = await tq.enqueue_surface_connections(
            work_id="test-work",
            work_title="Test Work",
            work_author="Test Author",
            confidence_threshold=0.5,
            min_connections_for_pr=2,
        )

        assert result == "test-job-123"
        mock_pool.enqueue_job.assert_awaited_once_with(
            "task_surface_connections",
            work_id="test-work",
            work_title="Test Work",
            work_author="Test Author",
            confidence_threshold=0.5,
            min_connections_for_pr=2,
        )


# ---------------------------------------------------------------------------
# handle_detect_passage_links surfacing wiring
# ---------------------------------------------------------------------------


class TestDetectPassageLinksSurfacingWiring:
    """Verify handle_detect_passage_links triggers surfacing."""

    @pytest.mark.asyncio()
    async def test_surfacing_triggered_after_links_created(self) -> None:
        """When passage links are created, surfacing is triggered.

        To reach the surfacing code, we need: primary source + contextual
        counterpart chunks so the function doesn't return early. We patch
        the link detectors to avoid real Neo4j calls, and verify surfacing
        is called when total_links > 0.
        """
        from author_library.tools.composable_ingestion import handle_detect_passage_links

        mock_storage = MagicMock()
        mock_storage.works.get = AsyncMock(return_value={
            "work_id": "test--work",
            "source_class": "primary",
            "title": "Test Work",
            "author": "Test Author",
        })
        mock_storage.works.list_by_author = AsyncMock(return_value=[
            {"work_id": "test--work", "source_class": "primary"},
            {"work_id": "test--context", "source_class": "contextual"},
        ])
        # list_by_work is called multiple times:
        # 1st call: meso chunks for the work being linked
        # 2nd call: counterpart contextual chunks (must be non-empty!)
        mock_storage.chunks.list_by_work = AsyncMock(side_effect=[
            # First call: chunks for work being linked
            [{"id": "chunk-1", "text": "The imagination is the primary faculty...",
              "granularity": "meso", "work_id": "test--work",
              "source_class": "primary", "position": 0}],
            # Second call: counterpart contextual chunks (non-empty so we proceed)
            [{"id": "ctx-chunk-1", "text": "Coleridge argued that imagination...",
              "granularity": "meso", "work_id": "test--context",
              "source_class": "contextual", "position": 0}],
        ])

        mock_settings = MagicMock()
        mock_embedding = MagicMock()

        # Mock link detectors so we get some links_created > 0
        mock_link_result = MagicMock()
        mock_link_result.edges_created = 1
        mock_link_result.links = []

        with patch(
            "author_library.tools.composable_ingestion.ExplicitLinkDetector",
        ) as MockExplicit, patch(
            "author_library.tools.composable_ingestion.ImplicitEngagementDetector",
        ) as MockImplicit, patch(
            "author_library.tools.composable_ingestion.ThematicParallelDetector",
        ) as MockThematic, patch(
            "author_library.surfacing.batch_surfacing.BatchSurfacer",
        ) as MockSurfacer:
            # Setup link detectors to return some edges
            MockExplicit.return_value.detect_and_link = AsyncMock(
                return_value=mock_link_result,
            )
            MockImplicit.return_value.detect_and_link = AsyncMock(
                return_value=mock_link_result,
            )
            MockThematic.return_value.detect_and_link = AsyncMock(
                return_value=mock_link_result,
            )

            # Setup surfacing
            surfacing_result = BatchSurfacingResult(
                work_id="test--work",
                scan_result=ScanResult(work_id="test--work", total_found=2),
            )
            instance = MockSurfacer.return_value
            instance.surface_after_ingestion = AsyncMock(return_value=surfacing_result)

            import json
            result_str = await handle_detect_passage_links(
                {"work_id": "test--work"},
                settings=mock_settings,
                storage=mock_storage,
                embedding_provider=mock_embedding,
            )
            result = json.loads(result_str)

        # Surfacing was triggered (total_links > 0)
        instance.surface_after_ingestion.assert_awaited_once()
        assert "surfacing" in result
        assert result["surfacing"]["total_connections"] == 2
