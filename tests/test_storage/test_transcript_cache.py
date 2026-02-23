"""Tests for PgTranscriptCacheRepository (D5).

Tests cover:
  - cache() stores a transcript keyed by source_url
  - get_cached() returns text when TTL is still valid
  - get_cached() returns None for unknown URLs
  - invalidate() removes a specific entry
  - invalidate() returns False for unknown URLs
  - invalidate_expired() bulk removes expired entries

These tests run against real PostgreSQL (same as other test_storage tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from author_library.storage.migrations.runner import run_migrations
from author_library.storage.repositories import PgTranscriptCacheRepository

if TYPE_CHECKING:
    from author_library.storage.postgres import PostgresPool


# ---------------------------------------------------------------------------
# Transcript cache CRUD tests
# ---------------------------------------------------------------------------


async def test_cache_and_get(pg_pool: PostgresPool) -> None:
    """Cache a transcript and retrieve it while TTL is valid."""
    await run_migrations(pg_pool)
    repo = PgTranscriptCacheRepository(pg_pool)

    url = "https://example.com/video/123"
    text = "This is the transcript text."

    await repo.cache(url, text, ttl_seconds=3600)
    result = await repo.get_cached(url)
    assert result == text


async def test_get_cached_missing_url_returns_none(pg_pool: PostgresPool) -> None:
    """get_cached returns None for URLs not in the cache."""
    await run_migrations(pg_pool)
    repo = PgTranscriptCacheRepository(pg_pool)

    result = await repo.get_cached("https://example.com/nonexistent")
    assert result is None


async def test_cache_upserts_on_conflict(pg_pool: PostgresPool) -> None:
    """Caching the same URL twice updates the transcript text."""
    await run_migrations(pg_pool)
    repo = PgTranscriptCacheRepository(pg_pool)

    url = "https://example.com/video/456"
    await repo.cache(url, "first version")
    await repo.cache(url, "second version")

    result = await repo.get_cached(url)
    assert result == "second version"


async def test_invalidate_removes_entry(pg_pool: PostgresPool) -> None:
    """invalidate() removes a specific cached transcript."""
    await run_migrations(pg_pool)
    repo = PgTranscriptCacheRepository(pg_pool)

    url = "https://example.com/video/789"
    await repo.cache(url, "some transcript")

    removed = await repo.invalidate(url)
    assert removed is True

    result = await repo.get_cached(url)
    assert result is None


async def test_invalidate_missing_url_returns_false(pg_pool: PostgresPool) -> None:
    """invalidate() returns False when the URL isn't cached."""
    await run_migrations(pg_pool)
    repo = PgTranscriptCacheRepository(pg_pool)

    removed = await repo.invalidate("https://example.com/nonexistent")
    assert removed is False


async def test_get_cached_returns_none_when_ttl_expired(pg_pool: PostgresPool) -> None:
    """get_cached returns None when the TTL has expired.

    Uses a very short TTL and manually backdates the cached_at timestamp
    to simulate expiration without needing to sleep.
    """
    await run_migrations(pg_pool)
    repo = PgTranscriptCacheRepository(pg_pool)

    url = "https://example.com/video/expired"
    await repo.cache(url, "expired transcript", ttl_seconds=1)

    # Manually backdate cached_at by 10 seconds to force expiration
    await pg_pool.execute(
        "UPDATE transcript_cache SET cached_at = NOW() - INTERVAL '10 seconds' WHERE source_url = $1",
        url,
    )

    result = await repo.get_cached(url)
    assert result is None


async def test_invalidate_expired_removes_old_entries(pg_pool: PostgresPool) -> None:
    """invalidate_expired() bulk removes entries past their TTL."""
    await run_migrations(pg_pool)
    repo = PgTranscriptCacheRepository(pg_pool)

    # Insert two entries
    await repo.cache("https://example.com/a", "transcript a", ttl_seconds=1)
    await repo.cache("https://example.com/b", "transcript b", ttl_seconds=3600)

    # Backdate entry 'a' to make it expired
    await pg_pool.execute(
        "UPDATE transcript_cache SET cached_at = NOW() - INTERVAL '10 seconds' WHERE source_url = $1",
        "https://example.com/a",
    )

    count = await repo.invalidate_expired()
    assert count == 1

    # 'a' should be gone, 'b' should remain
    assert await repo.get_cached("https://example.com/a") is None
    assert await repo.get_cached("https://example.com/b") == "transcript b"
