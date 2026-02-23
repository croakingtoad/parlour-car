"""Tests for job status tracking (D3).

Tests cover:
  - JobStatus enum values
  - JobInfo model serialization via to_dict()
  - JobInfo conditional field inclusion (task_name, result, error)
  - MCP tool handlers: _handle_job_status, _handle_ingest_book_async
"""

from __future__ import annotations

import json

from author_library.jobs import JobInfo, JobStatus
from author_library.server import _handle_ingest_book_async, _handle_job_status


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
        assert "job_id" in data["error"].lower()

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
