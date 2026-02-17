"""Tests for the TTL cache and CacheManager."""

from __future__ import annotations

import asyncio

from author_library.cache import CacheManager, TTLCache


class TestTTLCache:
    """Unit tests for the TTLCache."""

    async def test_put_and_get(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=60)
        await cache.put("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    async def test_get_miss(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=60)
        result = await cache.get("nonexistent")
        assert result is None

    async def test_ttl_expiration(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=0.1)
        await cache.put("key1", "value1")
        await asyncio.sleep(0.15)
        result = await cache.get("key1")
        assert result is None, "Entry should have expired"

    async def test_lru_eviction(self) -> None:
        cache = TTLCache("test", max_size=3, ttl_seconds=60)
        await cache.put("a", 1)
        await cache.put("b", 2)
        await cache.put("c", 3)
        # Cache is full; adding d should evict a (LRU)
        await cache.put("d", 4)
        assert await cache.get("a") is None, "a should have been evicted"
        assert await cache.get("b") == 2
        assert await cache.get("c") == 3
        assert await cache.get("d") == 4

    async def test_lru_promotion_on_get(self) -> None:
        cache = TTLCache("test", max_size=3, ttl_seconds=60)
        await cache.put("a", 1)
        await cache.put("b", 2)
        await cache.put("c", 3)
        # Access a — promotes it to most recently used
        await cache.get("a")
        # Adding d should evict b (now LRU), not a
        await cache.put("d", 4)
        assert await cache.get("a") == 1, "a should be preserved (recently accessed)"
        assert await cache.get("b") is None, "b should have been evicted"

    async def test_update_existing_key(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=60)
        await cache.put("key1", "v1")
        await cache.put("key1", "v2")
        result = await cache.get("key1")
        assert result == "v2"
        assert cache.size == 1

    async def test_invalidate(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=60)
        await cache.put("key1", "value1")
        removed = await cache.invalidate("key1")
        assert removed is True
        assert await cache.get("key1") is None

    async def test_invalidate_nonexistent(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=60)
        removed = await cache.invalidate("nonexistent")
        assert removed is False

    async def test_clear(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=60)
        await cache.put("a", 1)
        await cache.put("b", 2)
        await cache.put("c", 3)
        count = await cache.clear()
        assert count == 3
        assert cache.size == 0

    async def test_stats(self) -> None:
        cache = TTLCache("test", max_size=100, ttl_seconds=60)
        await cache.put("a", 1)
        await cache.put("b", 2)
        await cache.get("a")  # hit
        await cache.get("c")  # miss

        stats = cache.stats()
        assert stats["name"] == "test"
        assert stats["size"] == 2
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    async def test_complex_values(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=60)
        complex_value = {"results": [1, 2, 3], "metadata": {"key": "val"}}
        await cache.put("key1", complex_value)
        result = await cache.get("key1")
        assert result == complex_value


class TestCacheManager:
    """Tests for the CacheManager that manages multiple caches."""

    async def test_query_cache(self) -> None:
        mgr = CacheManager()
        key = mgr.query_key("What is love?", author_id="guite")
        await mgr.query_cache.put(key, {"response": "Baby don't hurt me"})
        result = await mgr.query_cache.get(key)
        assert result is not None
        assert result["response"] == "Baby don't hurt me"

    async def test_embedding_cache(self) -> None:
        mgr = CacheManager()
        key = mgr.embedding_key("test text", "voyage", "voyage-3-large")
        await mgr.embedding_cache.put(key, [0.1, 0.2, 0.3])
        result = await mgr.embedding_cache.get(key)
        assert result == [0.1, 0.2, 0.3]

    async def test_graph_cache(self) -> None:
        mgr = CacheManager()
        key = mgr.graph_key("theme_subgraph", theme="joy")
        await mgr.graph_cache.put(key, {"nodes": 5, "edges": 10})
        result = await mgr.graph_cache.get(key)
        assert result == {"nodes": 5, "edges": 10}

    async def test_invalidate_on_ingestion(self) -> None:
        mgr = CacheManager()
        # Populate all caches
        qkey = mgr.query_key("test")
        await mgr.query_cache.put(qkey, "cached query")
        ekey = mgr.embedding_key("text", "voyage", "model")
        await mgr.embedding_cache.put(ekey, [0.1])
        gkey = mgr.graph_key("test_query")
        await mgr.graph_cache.put(gkey, "cached graph")

        # Invalidate on ingestion
        cleared = await mgr.invalidate_on_ingestion()
        assert cleared["query_cleared"] == 1
        assert cleared["graph_cleared"] == 1

        # Query and graph caches should be empty
        assert await mgr.query_cache.get(qkey) is None
        assert await mgr.graph_cache.get(gkey) is None

        # Embedding cache should be preserved
        assert await mgr.embedding_cache.get(ekey) == [0.1]

    async def test_all_stats(self) -> None:
        mgr = CacheManager()
        stats = mgr.all_stats()
        assert "query" in stats
        assert "embedding" in stats
        assert "graph" in stats
        assert stats["query"]["name"] == "query"

    async def test_deterministic_keys(self) -> None:
        mgr = CacheManager()
        k1 = mgr.query_key("question", author_id="guite", style="academic")
        k2 = mgr.query_key("question", author_id="guite", style="academic")
        k3 = mgr.query_key("question", author_id="lewis", style="academic")
        assert k1 == k2, "Same inputs should produce same key"
        assert k1 != k3, "Different inputs should produce different keys"
