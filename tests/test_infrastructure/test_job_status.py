"""Tests for job status tracking (D3) and arq worker lifecycle.

Tests cover:
  - JobStatus enum values
  - JobInfo model serialization via to_dict()
  - JobInfo conditional field inclusion (task_name, result, error)
  - MCP tool handlers: _handle_job_status, _handle_ingest_book_async
  - arq worker auto-start/stop lifecycle
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from author_library.jobs import JobInfo, JobStatus
from author_library.server import (
    _handle_ingest_book_async,
    _handle_job_status,
    _start_arq_worker,
    _stop_arq_worker,
)


# ---------------------------------------------------------------------------
# JobStatus enum
# ---------------------------------------------------------------------------


class TestJobStatus:
    def test_enum_values(self) -> None:
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.IN_PROGRESS.value == "in_progress"
        assert JobStatus.COMPLETE.value == "complete"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.NOT_FOUND.value == "not_found"

    def test_is_string_enum(self) -> None:
        assert isinstance(JobStatus.QUEUED, str)
        assert JobStatus.QUEUED == "queued"


# ---------------------------------------------------------------------------
# JobInfo
# ---------------------------------------------------------------------------


class TestJobInfo:
    def test_to_dict_minimal(self) -> None:
        info = JobInfo(job_id="j-123", status=JobStatus.QUEUED)
        d = info.to_dict()
        assert d == {"job_id": "j-123", "status": "queued"}

    def test_to_dict_with_task_name(self) -> None:
        info = JobInfo(
            job_id="j-123",
            status=JobStatus.IN_PROGRESS,
            task_name="task_ingest_book",
        )
        d = info.to_dict()
        assert d["task_name"] == "task_ingest_book"

    def test_to_dict_with_result(self) -> None:
        result_data = {"work_id": "w-1", "chunks": 42}
        info = JobInfo(
            job_id="j-123",
            status=JobStatus.COMPLETE,
            task_name="task_ingest_book",
            result=result_data,
        )
        d = info.to_dict()
        assert d["result"] == result_data
        assert "error" not in d

    def test_to_dict_with_error(self) -> None:
        info = JobInfo(
            job_id="j-123",
            status=JobStatus.FAILED,
            task_name="task_ingest_book",
            error="File not found",
        )
        d = info.to_dict()
        assert d["error"] == "File not found"
        assert "result" not in d

    def test_to_dict_omits_empty_task_name(self) -> None:
        info = JobInfo(job_id="j-123", status=JobStatus.QUEUED, task_name="")
        d = info.to_dict()
        assert "task_name" not in d

    def test_slots_defined(self) -> None:
        info = JobInfo(job_id="j-123", status=JobStatus.QUEUED)
        assert hasattr(info, "__slots__")


# ---------------------------------------------------------------------------
# MCP tool handler: _handle_job_status
# ---------------------------------------------------------------------------


class TestHandleJobStatus:
    async def test_missing_job_id_returns_error(self) -> None:
        result = await _handle_job_status({}, state={})
        data = json.loads(result)
        assert "error" in data
        assert "error" in data  # No task_queue available

    async def test_no_task_queue_returns_error(self) -> None:
        result = await _handle_job_status({"job_id": "j-123"}, state={})
        data = json.loads(result)
        assert "error" in data
        assert "not available" in data["error"].lower()

    async def test_unavailable_queue_returns_error(self) -> None:
        class FakeQueue:
            available = False

        result = await _handle_job_status(
            {"job_id": "j-123"}, state={"task_queue": FakeQueue()}
        )
        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# MCP tool handler: _handle_ingest_book_async
# ---------------------------------------------------------------------------


class TestHandleIngestBookAsync:
    async def test_missing_file_path_returns_error(self) -> None:
        result = await _handle_ingest_book_async(
            {"subject_author_id": "test"},
            state={},
        )
        data = json.loads(result)
        assert "error" in data
        assert "file_path" in data["error"].lower()

    async def test_missing_subject_author_id_returns_error(self) -> None:
        result = await _handle_ingest_book_async(
            {"file_path": "/some/path.epub"},
            state={},
        )
        data = json.loads(result)
        assert "error" in data
        assert "subject_author_id" in data["error"].lower()

    async def test_no_task_queue_returns_error(self) -> None:
        result = await _handle_ingest_book_async(
            {"file_path": "/some/path.epub", "subject_author_id": "test"},
            state={},
        )
        data = json.loads(result)
        assert "error" in data
        assert "not available" in data["error"].lower()

    async def test_unavailable_queue_returns_error(self) -> None:
        class FakeQueue:
            available = False

        result = await _handle_ingest_book_async(
            {"file_path": "/some/path.epub", "subject_author_id": "test"},
            state={"task_queue": FakeQueue()},
        )
        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# arq worker auto-start lifecycle
# ---------------------------------------------------------------------------


class TestStartArqWorker:
    """Verify arq worker subprocess launch behaviour."""

    @pytest.mark.asyncio
    async def test_skips_when_redis_unavailable(self) -> None:
        """Worker is NOT started when Redis is down."""
        queue = MagicMock()
        queue.available = False

        result = await _start_arq_worker(queue)
        assert result is None

    @pytest.mark.asyncio
    async def test_starts_subprocess_when_redis_available(self) -> None:
        """Worker subprocess is launched when Redis is connected."""
        queue = MagicMock()
        queue.available = True

        fake_proc = AsyncMock()
        fake_proc.pid = 12345

        with patch(
            "author_library.server.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=fake_proc,
        ) as mock_exec:
            result = await _start_arq_worker(queue)

        assert result is fake_proc
        mock_exec.assert_called_once()
        # Verify arq module and worker settings are in the command
        call_args = mock_exec.call_args[0]
        assert "-m" in call_args
        assert "arq" in call_args
        assert "author_library.worker.WorkerSettings" in call_args

    @pytest.mark.asyncio
    async def test_returns_none_on_exec_failure(self) -> None:
        """If subprocess launch raises, returns None (graceful degradation)."""
        queue = MagicMock()
        queue.available = True

        with patch(
            "author_library.server.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            side_effect=OSError("exec failed"),
        ):
            result = await _start_arq_worker(queue)

        assert result is None


class TestStopArqWorker:
    """Verify arq worker subprocess shutdown behaviour."""

    @pytest.mark.asyncio
    async def test_noop_when_process_is_none(self) -> None:
        """No error when called with None (worker was never started)."""
        await _stop_arq_worker(None)  # Should not raise

    @pytest.mark.asyncio
    async def test_noop_when_process_already_exited(self) -> None:
        """No error when process has already exited."""
        proc = MagicMock()
        proc.returncode = 0  # Already exited
        await _stop_arq_worker(proc)
        proc.send_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_sigterm_and_waits(self) -> None:
        """Sends SIGTERM and waits for the process to exit."""
        proc = MagicMock()
        proc.returncode = None  # Still running
        proc.pid = 99999
        proc.send_signal = MagicMock()

        # Make wait() resolve immediately (process exits after SIGTERM)
        async def _wait() -> int:
            proc.returncode = 0
            return 0

        proc.wait = _wait

        await _stop_arq_worker(proc)
        proc.send_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_kills_on_timeout(self) -> None:
        """Falls back to SIGKILL if process doesn't exit within timeout."""
        proc = MagicMock()
        proc.returncode = None
        proc.pid = 99999
        proc.send_signal = MagicMock()
        proc.kill = MagicMock()
        proc.wait = AsyncMock(return_value=0)

        # Patch wait_for to raise TimeoutError (simulating slow shutdown)
        with patch("author_library.server.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            await _stop_arq_worker(proc)

        proc.kill.assert_called_once()
