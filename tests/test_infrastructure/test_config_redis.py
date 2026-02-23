"""Tests for Redis configuration settings (D1).

Tests cover:
  - RedisSettings defaults
  - RedisSettings included in root Settings
  - Worker entry point is importable
"""

from __future__ import annotations

from author_library.config import RedisSettings, Settings


class TestRedisSettings:
    def test_defaults(self) -> None:
        s = RedisSettings()
        assert s.redis_url == "redis://localhost:6379"

    def test_included_in_root_settings(self) -> None:
        s = Settings()
        assert isinstance(s.redis, RedisSettings)
        assert s.redis.redis_url == "redis://localhost:6379"


class TestWorkerImport:
    def test_worker_module_importable(self) -> None:
        from author_library import worker
        assert hasattr(worker, "WorkerSettings")
        assert hasattr(worker, "get_redis_settings")

    def test_tasks_module_importable(self) -> None:
        from author_library import tasks
        assert hasattr(tasks, "task_ingest_book")
        assert hasattr(tasks, "task_ingest_corpus")

    def test_queue_module_importable(self) -> None:
        from author_library import queue
        assert hasattr(queue, "TaskQueue")

    def test_jobs_module_importable(self) -> None:
        from author_library import jobs
        assert hasattr(jobs, "JobStatus")
        assert hasattr(jobs, "JobInfo")
        assert hasattr(jobs, "get_job_info")
