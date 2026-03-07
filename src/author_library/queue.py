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

    async def enqueue_capture(
        self,
        *,
        payload: dict[str, Any],
    ) -> str | None:
        """Enqueue a capture event for background processing.

        Args:
            payload: Serialized CapturePayload dict.

        Returns:
            The arq job ID if enqueued, or None if the queue is unavailable.
        """
        if not self._pool:
            return None

        job = await self._pool.enqueue_job(
            "task_process_capture",
            payload=payload,
        )

        if job is None:
            log.warning("task_enqueue_failed", task="process_capture")
            return None

        log.info(
            "task_enqueued",
            task="process_capture",
            job_id=job.job_id,
            source_url=payload.get("source_url"),
            mode=payload.get("mode"),
        )
        return job.job_id

    async def enqueue_surface_connections(
        self,
        *,
        work_id: str,
        work_title: str = "",
        work_author: str = "",
        confidence_threshold: float = 0.4,
        min_connections_for_pr: int = 1,
    ) -> str | None:
        """Enqueue a post-ingestion connection surfacing job.

        Scans for new cross-work connections and generates PR content.
        Triggered automatically after passage link detection completes.

        Args:
            work_id: The newly ingested work to scan connections for.
            work_title: Title for PR readability.
            work_author: Author for PR readability.
            confidence_threshold: Minimum confidence to include.
            min_connections_for_pr: Skip PR if fewer connections found.

        Returns:
            The arq job ID if enqueued, or None if the queue is unavailable.
        """
        if not self._pool:
            return None

        job = await self._pool.enqueue_job(
            "task_surface_connections",
            work_id=work_id,
            work_title=work_title,
            work_author=work_author,
            confidence_threshold=confidence_threshold,
            min_connections_for_pr=min_connections_for_pr,
        )

        if job is None:
            log.warning("task_enqueue_failed", task="surface_connections")
            return None

        log.info(
            "task_enqueued",
            task="surface_connections",
            job_id=job.job_id,
            work_id=work_id,
        )
        return job.job_id


    async def enqueue_quality_gate(
        self,
        *,
        work_id: str,
        author_id: str,
    ) -> str | None:
        """Enqueue post-ingestion async quality checks.

        Runs theme dedup, PG-Neo4j consistency, cross-work linking,
        and entity coverage audit in the background.

        Args:
            work_id: The newly ingested work that triggered this gate.
            author_id: The subject author for cross-work analysis.

        Returns:
            The arq job ID if enqueued, or None if the queue is unavailable.
        """
        if not self._pool:
            return None

        job = await self._pool.enqueue_job(
            "task_quality_gate",
            work_id=work_id,
            author_id=author_id,
        )

        if job is None:
            log.warning("task_enqueue_failed", task="quality_gate")
            return None

        log.info(
            "task_enqueued",
            task="quality_gate",
            job_id=job.job_id,
            work_id=work_id,
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
