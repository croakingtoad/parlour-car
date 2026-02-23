"""Task queue client for enqueuing background work via arq + Redis.

Provides a thin async interface over arq's Redis pool for enqueueing
ingestion jobs from the MCP server. The queue client is optional —
if Redis is unavailable, the server falls back to synchronous processing.
"""

from __future__ import annotations

from typing import Any

import structlog
from arq.connections import ArqRedis, create_pool

from author_library.worker import get_redis_settings

log = structlog.get_logger(__name__)


class TaskQueue:
    """Async client for enqueuing arq tasks.

    Maintains a connection pool to Redis and provides typed methods
    for enqueueing specific task types. Safe to use across the
    server's lifetime.
    """

    def __init__(self) -> None:
        self._pool: ArqRedis | None = None

    async def connect(self) -> None:
        """Initialize the Redis connection pool."""
        try:
            self._pool = await create_pool(get_redis_settings())
            log.info("task_queue_connected")
        except Exception as exc:
            log.warning("task_queue_connection_failed", error=str(exc))
            self._pool = None

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._pool:
            await self._pool.aclose()
            self._pool = None
            log.info("task_queue_closed")

    @property
    def available(self) -> bool:
        """Whether the queue is connected and available."""
        return self._pool is not None

    async def enqueue_ingest_book(
        self,
        *,
        file_path: str,
        subject_author_id: str,
        metadata_hints: dict[str, Any] | None = None,
    ) -> str | None:
        """Enqueue a single-work ingestion job.

        Returns:
            The arq job ID if enqueued, or None if the queue is unavailable.
        """
        if not self._pool:
            return None

        job = await self._pool.enqueue_job(
            "task_ingest_book",
            file_path=file_path,
            subject_author_id=subject_author_id,
            metadata_hints=metadata_hints,
        )

        if job is None:
            log.warning("task_enqueue_failed", task="ingest_book")
            return None

        log.info(
            "task_enqueued",
            task="ingest_book",
            job_id=job.job_id,
            file_path=file_path,
            subject_author=subject_author_id,
        )
        return job.job_id

    async def enqueue_ingest_corpus(
        self,
        *,
        file_paths: list[str],
        subject_author_id: str,
        metadata_hints: dict[str, Any] | None = None,
        run_cross_work_analysis: bool = True,
    ) -> str | None:
        """Enqueue a corpus ingestion job.

        Returns:
            The arq job ID if enqueued, or None if the queue is unavailable.
        """
        if not self._pool:
            return None

        job = await self._pool.enqueue_job(
            "task_ingest_corpus",
            file_paths=file_paths,
            subject_author_id=subject_author_id,
            metadata_hints=metadata_hints,
            run_cross_work_analysis=run_cross_work_analysis,
        )

        if job is None:
            log.warning("task_enqueue_failed", task="ingest_corpus")
            return None

        log.info(
            "task_enqueued",
            task="ingest_corpus",
            job_id=job.job_id,
            file_count=len(file_paths),
            subject_author=subject_author_id,
        )
        return job.job_id

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get the status of a queued job.

        Returns:
            dict with 'status' (queued/in_progress/complete/not_found),
            and optionally 'result' or 'error'.
        """
        if not self._pool:
            return {"status": "queue_unavailable", "job_id": job_id}

        from arq.jobs import Job

        job = Job(job_id, self._pool)
        status = await job.status()

        result: dict[str, Any] = {
            "job_id": job_id,
            "status": status.value,
        }

        if status.value == "complete":
            try:
                job_result = await job.result(timeout=1)
                result["result"] = job_result
            except Exception as exc:
                result["error"] = str(exc)

        return result
