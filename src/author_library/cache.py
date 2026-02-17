"""In-memory LRU cache with TTL for The Author Library.

Provides query result, embedding, and graph query caching with:
  - Configurable TTL per cache instance
  - Configurable max size (LRU eviction)
  - Automatic invalidation on ingestion (new content)
  - Thread-safe via asyncio locks
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class _CacheEntry:
    """A single cache entry with expiration tracking."""

    __slots__ = ("created_at", "value")

    def __init__(self, value: Any, created_at: float) -> None:
        self.value = value
        self.created_at = created_at


class TTLCache:
    """In-memory LRU cache with per-entry TTL expiration.

    Args:
        name: Human-readable name for logging.
        max_size: Maximum number of entries before LRU eviction.
        ttl_seconds: Time-to-live for each entry in seconds.
    """

    def __init__(self, name: str, *, max_size: int = 1000, ttl_seconds: float = 300.0) -> None:
        self._name = name
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return len(self._store)

    @property
    def hits(self) -> int:
        """Total cache hits since creation."""
        return self._hits

    @property
    def misses(self) -> int:
        """Total cache misses since creation."""
        return self._misses

    async def get(self, key: str) -> Any | None:
        """Retrieve a value by key, returning None on miss or expiration.

        Promotes the entry to most-recently-used on hit.
        """
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            if (time.monotonic() - entry.created_at) > self._ttl:
                del self._store[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    async def put(self, key: str, value: Any) -> None:
        """Store a value, evicting the LRU entry if at capacity."""
        async with self._lock:
            if key in self._store:
                # Update existing entry
                self._store[key] = _CacheEntry(value, time.monotonic())
                self._store.move_to_end(key)
                return

            # Evict LRU if at capacity
            while len(self._store) >= self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                log.debug("cache_eviction", cache=self._name, evicted_key=evicted_key)

            self._store[key] = _CacheEntry(value, time.monotonic())

    async def invalidate(self, key: str) -> bool:
        """Remove a specific key. Returns True if it existed."""
        async with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def clear(self) -> int:
        """Clear all entries. Returns the number of entries removed."""
        async with self._lock:
            count = len(self._store)
            self._store.clear()
            log.info("cache_cleared", cache=self._name, entries_removed=count)
            return count

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "name": self._name,
            "size": len(self._store),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }


def _make_cache_key(*parts: str) -> str:
    """Create a deterministic cache key from string parts."""
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()


class CacheManager:
    """Manages multiple TTL caches for the application.

    Provides typed access to query, embedding, and graph caches,
    and bulk invalidation when new content is ingested.
    """

    def __init__(
        self,
        *,
        query_max_size: int = 500,
        query_ttl: float = 600.0,
        embedding_max_size: int = 2000,
        embedding_ttl: float = 3600.0,
        graph_max_size: int = 500,
        graph_ttl: float = 600.0,
    ) -> None:
        self.query_cache = TTLCache("query", max_size=query_max_size, ttl_seconds=query_ttl)
        self.embedding_cache = TTLCache(
            "embedding", max_size=embedding_max_size, ttl_seconds=embedding_ttl
        )
        self.graph_cache = TTLCache("graph", max_size=graph_max_size, ttl_seconds=graph_ttl)

    def query_key(self, question: str, **params: Any) -> str:
        """Build a cache key for a query result."""
        param_parts = [f"{k}={v}" for k, v in sorted(params.items()) if v is not None]
        return _make_cache_key("query", question, *param_parts)

    def embedding_key(self, text: str, provider: str, model: str) -> str:
        """Build a cache key for an embedding result."""
        return _make_cache_key("embedding", text, provider, model)

    def graph_key(self, query_name: str, **params: Any) -> str:
        """Build a cache key for a graph query result."""
        param_parts = [f"{k}={v}" for k, v in sorted(params.items()) if v is not None]
        return _make_cache_key("graph", query_name, *param_parts)

    async def invalidate_on_ingestion(self) -> dict[str, int]:
        """Invalidate all caches after new content is ingested.

        New content means query and graph caches are stale.
        Embedding cache is preserved since embeddings for identical
        text + provider + model will not change.
        """
        query_cleared = await self.query_cache.clear()
        graph_cleared = await self.graph_cache.clear()
        log.info(
            "cache_invalidated_on_ingestion",
            query_cleared=query_cleared,
            graph_cleared=graph_cleared,
        )
        return {"query_cleared": query_cleared, "graph_cleared": graph_cleared}

    def all_stats(self) -> dict[str, dict[str, Any]]:
        """Return statistics for all caches."""
        return {
            "query": self.query_cache.stats(),
            "embedding": self.embedding_cache.stats(),
            "graph": self.graph_cache.stats(),
        }
