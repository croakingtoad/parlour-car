"""Tests for arq task wrapping and TaskQueue (D2).

Tests cover:
  - task_ingest_book raises IngestionError for missing files
  - task_ingest_corpus handles missing files gracefully
  - TaskQueue.available reports correctly before/after connect
  - TaskQueue.enqueue_* returns None when pool not connected
  - TaskQueue.get_job_status returns queue_unavailable when pool not connected
  - Worker configuration constants and settings
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from author_library.errors import IngestionError
from author_library.queue import TaskQueue
from author_library.tasks import task_ingest_book, task_ingest_corpus
from author_library.worker import (
    QUEUE_DEFAULT,
    QUEUE_INGESTION,
    WorkerSettings,
    get_redis_settings,
)


# ---------------------------------------------------------------------------
# Worker configuration (D1)
# ---------------------------------------------------------------------------


class TestWorkerConfig:
    def test_queue_constants_are_defined(self) -> None:
        assert QUEUE_INGESTION == "parlour:ingestion"
        assert QUEUE_DEFAULT == "arq:queue"

    def test_redis_settings_parses_default_url(self) -> None:
        settings = get_redis_settings()
        assert settings.host == "localhost"
        assert settings.port == 6379

    def test_worker_settings_class_attributes(self) -> None:
        assert WorkerSettings.max_jobs == 5
        assert WorkerSettings.job_timeout == 1800
        assert WorkerSettings.keep_result == 3600
        assert WorkerSettings.health_check_interval == 30
        assert WorkerSettings.queue_name == QUEUE_DEFAULT

    def test_worker_settings_registers_task_functions(self) -> None:
        func_names = [f.__name__ for f in WorkerSettings.functions]
        assert "task_ingest_book" in func_names
        assert "task_ingest_corpus" in func_names

    def test_worker_settings_has_lifecycle_hooks(self) -> None:
        assert WorkerSettings.on_startup is not None
        assert WorkerSettings.on_shutdown is not None


# ---------------------------------------------------------------------------
# Task functions (D2) — direct invocation
# ---------------------------------------------------------------------------


class TestTaskIngestBook:
    async def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        ctx: dict[str, Any] = {
            "settings": None,
            "storage": None,
            "embedding_provider": None,
        }
        with pytest.raises(IngestionError, match="File not found"):
            await task_ingest_book(
                ctx,
                file_path=str(tmp_path / "nonexistent.epub"),
                subject_author_id="test-author",
            )


class TestTaskIngestCorpus:
    async def test_missing_files_are_recorded_as_errors(self, tmp_path: Path) -> None:
        ctx: dict[str, Any] = {
            "settings": None,
            "storage": None,
            "embedding_provider": None,
        }
        result = await task_ingest_corpus(
            ctx,
            file_paths=[str(tmp_path / "missing1.epub"), str(tmp_path / "missing2.epub")],
            subject_author_id="test-author",
            run_cross_work_analysis=False,
        )
        assert result["works_processed"] == 0
        assert result["works_failed"] == 2
        assert len(result["errors"]) == 2


# ---------------------------------------------------------------------------
# TaskQueue (D2) — without Redis
# ---------------------------------------------------------------------------


class TestTaskQueueWithoutRedis:
    """TaskQueue behavior when Redis is not available."""

    def test_not_available_before_connect(self) -> None:
        tq = TaskQueue()
        assert tq.available is False

    async def test_enqueue_ingest_book_returns_none_without_pool(self) -> None:
        tq = TaskQueue()
        result = await tq.enqueue_ingest_book(
            file_path="/tmp/test.epub",
            subject_author_id="test-author",
        )
        assert result is None

    async def test_enqueue_ingest_corpus_returns_none_without_pool(self) -> None:
        tq = TaskQueue()
        result = await tq.enqueue_ingest_corpus(
            file_paths=["/tmp/test.epub"],
            subject_author_id="test-author",
        )
        assert result is None

    async def test_get_job_status_without_pool_returns_unavailable(self) -> None:
        tq = TaskQueue()
        result = await tq.get_job_status("some-job-id")
        assert result["status"] == "queue_unavailable"
        assert result["job_id"] == "some-job-id"

    async def test_connect_gracefully_fails_without_redis(self) -> None:
        """connect() should not raise when Redis is unreachable."""
        tq = TaskQueue()
        # If Redis is not running this should log a warning and set pool to None
        await tq.connect()
        # May or may not be available depending on local Redis
        # The key test is that it doesn't raise
        await tq.close()

    async def test_close_is_safe_when_not_connected(self) -> None:
        tq = TaskQueue()
        # Should not raise
        await tq.close()
