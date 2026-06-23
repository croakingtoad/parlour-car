"""arq worker configuration for background task processing.

Configures the arq worker with Redis connection settings, queue names,
and task function registration. Run with:

    arq author_library.worker.WorkerSettings
"""

from __future__ import annotations

from typing import Any

import structlog
from arq.connections import RedisSettings as ArqRedisSettings

from author_library.config import get_settings
from author_library.tasks import (
    task_ingest_book,
    task_ingest_corpus,
    task_process_capture,
    task_quality_gate,
    task_surface_connections,
)

log = structlog.get_logger(__name__)

# Queue names for task routing
QUEUE_INGESTION = "parlour:ingestion"
QUEUE_DEFAULT = "arq:queue"


def get_redis_settings() -> ArqRedisSettings:
    """Build arq RedisSettings from application configuration.

    Parses the REDIS_URL environment variable (or default) into
    the host/port/database/password components arq expects.
    """
    settings = get_settings()
    redis_url = settings.redis.redis_url

    # Parse redis://[:password@]host[:port][/database]
    from urllib.parse import urlparse

    parsed = urlparse(redis_url)
    return ArqRedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int(parsed.path.lstrip("/") or 0) if parsed.path and parsed.path != "/" else 0,
        password=parsed.password,
    )


async def startup(ctx: dict[str, Any]) -> None:
    """Worker startup hook — initializes storage and embedding provider."""
    from author_library.embeddings import ProviderRegistry
    from author_library.logging import setup_logging
    from author_library.storage import StorageManager

    settings = get_settings()
    setup_logging(level=settings.server.log_level, log_format=settings.server.log_format)

    storage = StorageManager(settings.database)
    await storage.connect()

    embedding_provider = ProviderRegistry.create(settings)

    ctx["settings"] = settings
    ctx["storage"] = storage
    ctx["embedding_provider"] = embedding_provider

    log.info("worker_started", redis_url=settings.redis.redis_url)


async def shutdown(ctx: dict[str, Any]) -> None:
    """Worker shutdown hook — closes storage and embedding connections."""
    embedding_provider = ctx.get("embedding_provider")
    if embedding_provider:
        await embedding_provider.close()

    storage = ctx.get("storage")
    if storage:
        await storage.close()

    log.info("worker_shutdown")


class WorkerSettings:
    """arq WorkerSettings class.

    arq discovers this class and uses its attributes to configure the worker.
    Task functions are registered here; they will be populated when
    ingestion tasks are implemented (D2).
    """

    functions = [task_ingest_book, task_ingest_corpus, task_process_capture, task_surface_connections, task_quality_gate]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = get_redis_settings()
    queue_name = QUEUE_DEFAULT
    max_jobs = 5
    job_timeout = 7200  # 2 hours — large OCR PDFs need extended time
    keep_result = 3600  # Keep results for 1 hour
    health_check_interval = 30
