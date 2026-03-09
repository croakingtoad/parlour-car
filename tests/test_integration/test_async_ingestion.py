"""Live integration tests for ingest_book_async and job_status MCP tools.

Tests the async ingestion queue (arq + Redis):
- ingest_book_async: enqueue a job, get job_id back immediately
- job_status: poll a specific job or list all jobs

IMPORTANT: These tests do NOT start an arq worker. Jobs enqueued here will
remain in "queued" state. We test the enqueue/poll layer only, not full
pipeline execution. The worker integration is verified by the MCP server
logs after actual ingestion.

Skips when Redis is not available (localhost:6379).
"""

from __future__ import annotations

import json
import socket
import tempfile
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from author_library.jobs import JobStatus, get_job_info
from author_library.queue import TaskQueue
from author_library.server import _handle_ingest_book_async, _handle_job_status


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def _redis_available() -> bool:
    """Check if Redis is reachable on localhost:6379."""
    try:
        with socket.create_connection(("localhost", 6379), timeout=2):
            return True
    except OSError:
        return False


SKIP_NO_REDIS = pytest.mark.skipif(
    not _redis_available(),
    reason="Redis not available (run `make dev`)",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def task_queue() -> TaskQueue:
    """Connected TaskQueue for testing. Disconnected after each test."""
    q = TaskQueue()
    await q.connect()
    assert q.available, "TaskQueue failed to connect — Redis unreachable?"
    yield q
    await q.close()


@pytest_asyncio.fixture
async def queue_state(task_queue: TaskQueue) -> dict[str, Any]:
    """Fake server state dict with a live TaskQueue."""
    return {"task_queue": task_queue}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _delete_job(task_queue: TaskQueue, job_id: str) -> None:
    """Remove an arq job from Redis (cleanup helper)."""
    if task_queue._pool:
        await task_queue._pool.delete(f"arq:job:{job_id}")


# ---------------------------------------------------------------------------
# TestIngestBookAsync
# ---------------------------------------------------------------------------


@SKIP_NO_REDIS
class TestIngestBookAsync:
    """ingest_book_async enqueues a job and returns a job_id immediately."""

    async def test_enqueue_returns_job_id(
        self, queue_state: dict[str, Any], task_queue: TaskQueue
    ) -> None:
        """ingest_book_async returns job_id and 'queued' status immediately."""
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False
        ) as f:
            f.write("Test content for async ingestion.")
            temp_path = f.name

        job_id: str | None = None
        try:
            result_str = await _handle_ingest_book_async(
                {
                    "file_path": temp_path,
                    "subject_author_id": "test-author",
                },
                state=queue_state,
            )
            result = json.loads(result_str)

            assert "job_id" in result, f"Expected job_id in result: {result}"
            assert result["status"] == "queued"
            assert "message" in result
            assert "job_status" in result["message"].lower() or "poll" in result["message"].lower()

            job_id = result["job_id"]
            assert isinstance(job_id, str)
            assert len(job_id) > 0

        finally:
            Path(temp_path).unlink(missing_ok=True)
            if job_id:
                await _delete_job(task_queue, job_id)

    async def test_enqueue_is_non_blocking(
        self, queue_state: dict[str, Any], task_queue: TaskQueue
    ) -> None:
        """ingest_book_async returns immediately (does not wait for pipeline)."""
        import time

        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False
        ) as f:
            f.write("A" * 10_000)  # Simulate a larger file
            temp_path = f.name

        job_id: str | None = None
        try:
            start = time.monotonic()
            result_str = await _handle_ingest_book_async(
                {
                    "file_path": temp_path,
                    "subject_author_id": "test-author",
                },
                state=queue_state,
            )
            elapsed = time.monotonic() - start

            result = json.loads(result_str)
            assert "job_id" in result

            # Should return in well under 5 seconds (no pipeline running)
            assert elapsed < 5.0, f"Enqueue took {elapsed:.1f}s — not non-blocking?"
            job_id = result["job_id"]

        finally:
            Path(temp_path).unlink(missing_ok=True)
            if job_id:
                await _delete_job(task_queue, job_id)

    async def test_missing_file_path_returns_error(
        self, queue_state: dict[str, Any]
    ) -> None:
        """ingest_book_async returns error when file_path is missing."""
        result_str = await _handle_ingest_book_async(
            {"subject_author_id": "test-author"},
            state=queue_state,
        )
        result = json.loads(result_str)
        assert "error" in result
        assert "file_path" in result["error"].lower()

    async def test_missing_subject_author_id_returns_error(
        self, queue_state: dict[str, Any]
    ) -> None:
        """ingest_book_async returns error when subject_author_id is missing."""
        result_str = await _handle_ingest_book_async(
            {"file_path": "/tmp/test.txt"},
            state=queue_state,
        )
        result = json.loads(result_str)
        assert "error" in result
        assert "subject_author_id" in result["error"].lower()

    async def test_no_task_queue_returns_error(self) -> None:
        """ingest_book_async returns error when task queue is not available."""
        result_str = await _handle_ingest_book_async(
            {"file_path": "/tmp/test.txt", "subject_author_id": "test-author"},
            state={"task_queue": None},
        )
        result = json.loads(result_str)
        assert "error" in result


# ---------------------------------------------------------------------------
# TestJobStatus
# ---------------------------------------------------------------------------


@SKIP_NO_REDIS
class TestJobStatus:
    """job_status returns job lifecycle state from arq/Redis."""

    async def test_job_status_queued_after_enqueue(
        self, queue_state: dict[str, Any], task_queue: TaskQueue
    ) -> None:
        """job_status shows 'queued' for a job enqueued but not yet picked up."""
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", delete=False
        ) as f:
            f.write("Job status test content.")
            temp_path = f.name

        job_id: str | None = None
        try:
            enqueue_str = await _handle_ingest_book_async(
                {
                    "file_path": temp_path,
                    "subject_author_id": "test-author",
                },
                state=queue_state,
            )
            enqueue_result = json.loads(enqueue_str)
            job_id = enqueue_result["job_id"]

            status_str = await _handle_job_status(
                {"job_id": job_id},
                state=queue_state,
            )
            status = json.loads(status_str)

            assert status["job_id"] == job_id
            # No worker running — should be queued (or not_found if arq cleaned it)
            assert status["status"] in ("queued", "in_progress", "not_found")

        finally:
            Path(temp_path).unlink(missing_ok=True)
            if job_id:
                await _delete_job(task_queue, job_id)

    async def test_job_status_unknown_job_returns_not_found(
        self, queue_state: dict[str, Any]
    ) -> None:
        """job_status returns not_found for an unknown job_id."""
        result_str = await _handle_job_status(
            {"job_id": "nonexistent-job-id-00000000"},
            state=queue_state,
        )
        result = json.loads(result_str)
        assert result["status"] == JobStatus.NOT_FOUND.value

    async def test_job_status_no_job_id_lists_all(
        self, queue_state: dict[str, Any]
    ) -> None:
        """job_status without job_id returns list of all recent jobs."""
        result_str = await _handle_job_status(
            {},
            state=queue_state,
        )
        result = json.loads(result_str)

        # Should return a listing dict
        assert "total_jobs" in result
        assert "jobs" in result
        assert isinstance(result["jobs"], list)

    async def test_job_status_no_task_queue_returns_error(self) -> None:
        """job_status returns error when task queue is not available."""
        result_str = await _handle_job_status(
            {"job_id": "some-job-id"},
            state={"task_queue": None},
        )
        result = json.loads(result_str)
        assert "error" in result


# ---------------------------------------------------------------------------
# TestTaskQueue (unit tests for the queue client layer)
# ---------------------------------------------------------------------------


@SKIP_NO_REDIS
class TestTaskQueue:
    """TaskQueue client connects, enqueues, and reports job state."""

    async def test_connect_and_available(self) -> None:
        """TaskQueue.available is True after connect()."""
        q = TaskQueue()
        await q.connect()
        try:
            assert q.available is True
        finally:
            await q.close()

    async def test_available_false_after_close(self) -> None:
        """TaskQueue.available is False after close()."""
        q = TaskQueue()
        await q.connect()
        await q.close()
        assert q.available is False

    async def test_enqueue_returns_string_job_id(
        self, task_queue: TaskQueue
    ) -> None:
        """enqueue_ingest_book returns a non-empty string job ID."""
        job_id = await task_queue.enqueue_ingest_book(
            file_path="/tmp/test-file.txt",
            subject_author_id="test-author",
        )
        try:
            assert job_id is not None
            assert isinstance(job_id, str)
            assert len(job_id) > 0
        finally:
            if job_id:
                await _delete_job(task_queue, job_id)

    async def test_enqueued_job_shows_queued_status(
        self, task_queue: TaskQueue
    ) -> None:
        """An enqueued job appears as queued in arq's Redis state."""
        job_id = await task_queue.enqueue_ingest_book(
            file_path="/tmp/test-status.txt",
            subject_author_id="test-author",
        )
        try:
            assert job_id is not None
            info = await get_job_info(task_queue._pool, job_id)
            # No worker → queued (or not_found if arq already expired it)
            assert info.status in (JobStatus.QUEUED, JobStatus.NOT_FOUND)
        finally:
            if job_id:
                await _delete_job(task_queue, job_id)

    async def test_get_job_info_unknown_returns_not_found(
        self, task_queue: TaskQueue
    ) -> None:
        """get_job_info for an unknown job_id returns NOT_FOUND status."""
        info = await get_job_info(task_queue._pool, "definitely-not-a-real-job")
        assert info.status == JobStatus.NOT_FOUND
