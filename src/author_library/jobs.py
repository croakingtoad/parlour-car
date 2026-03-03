"""Job status tracking for long-running background operations.

Provides a unified interface for tracking job lifecycle (pending →
processing → complete/failed) built on arq's Redis-backed job state.
The MCP server returns job IDs for async operations, and clients
poll via the job_status tool.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class JobStatus(str, Enum):
    """Lifecycle states for a background job."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    NOT_FOUND = "not_found"


class JobInfo:
    """Structured job status information."""

    __slots__ = ("error", "job_id", "result", "status", "task_name")

    def __init__(
        self,
        *,
        job_id: str,
        status: JobStatus,
        task_name: str = "",
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.job_id = job_id
        self.status = status
        self.task_name = task_name
        self.result = result
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status.value,
        }
        if self.task_name:
            d["task_name"] = self.task_name
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        return d


async def get_job_info(pool: Any, job_id: str) -> JobInfo:
    """Retrieve job status from arq's Redis state.

    Maps arq's internal status values to our JobStatus enum.

    Args:
        pool: ArqRedis connection pool.
        job_id: The arq job identifier.

    Returns:
        JobInfo with current status and result (if complete).
    """
    from arq.jobs import Job, JobStatus as ArqJobStatus

    job = Job(job_id, pool)
    arq_status = await job.status()

    # Map arq statuses to our enum
    status_map = {
        ArqJobStatus.deferred: JobStatus.QUEUED,
        ArqJobStatus.queued: JobStatus.QUEUED,
        ArqJobStatus.in_progress: JobStatus.IN_PROGRESS,
        ArqJobStatus.complete: JobStatus.COMPLETE,
        ArqJobStatus.not_found: JobStatus.NOT_FOUND,
    }

    status = status_map.get(arq_status, JobStatus.NOT_FOUND)

    result_data = None
    error = None

    if status == JobStatus.COMPLETE:
        try:
            raw_result = await job.result(timeout=1)
            if isinstance(raw_result, dict):
                # Check if arq stored an error
                if raw_result.get("success") is False:
                    status = JobStatus.FAILED
                    error = raw_result.get("error", "Unknown error")
                else:
                    result_data = raw_result
            else:
                result_data = {"raw": str(raw_result)}
        except Exception as exc:
            status = JobStatus.FAILED
            error = str(exc)

    # Get job info for task name
    info = await job.info()
    task_name = info.function if info else ""

    return JobInfo(
        job_id=job_id,
        status=status,
        task_name=task_name,
        result=result_data,
        error=error,
    )
