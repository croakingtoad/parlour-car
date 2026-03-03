"""Tests for the in-memory TTL cache and CacheManager (D4).

Tests cover:
  - TTLCache LRU eviction behavior
  - TTLCache TTL expiration behavior
  - TTLCache hit/miss statistics
  - CacheManager cache-key generation (deterministic, distinct per type)
  - CacheManager invalidation on ingestion
  - CacheManager per-author invalidation for voice/thematic caches
  - CachedEmbeddingProvider cache-through for embed_text
  - CachedEmbeddingProvider cache-through for embed_query
  - CachedEmbeddingProvider cache-aware embed_batch
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

from author_library.cache import CacheManager, TTLCache, _make_cache_key
from author_library.embeddings.base import (
    BatchEmbeddingResult,
    EmbeddingResult,
)
from author_library.embeddings.cached import CachedEmbeddingProvider


# ---------------------------------------------------------------------------
# TTLCache unit tests
# ---------------------------------------------------------------------------


class TestTTLCache:
    """Core LRU/TTL cache behavior."""

    async def test_put_and_get(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=60.0)
        await cache.put("k1", "v1")
        result = await cache.get("k1")
        assert result == "v1"

    async def test_get_missing_key_returns_none(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=60.0)
        result = await cache.get("nonexistent")
        assert result is None

    async def test_hit_and_miss_stats(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=60.0)
        await cache.put("k1", "v1")

        # Hit
        await cache.get("k1")
        assert cache.hits == 1
        assert cache.misses == 0

        # Miss
        await cache.get("k2")
        assert cache.hits == 1
        assert cache.misses == 1

    async def test_lru_eviction(self) -> None:
        cache = TTLCache("test", max_size=3, ttl_seconds=60.0)
        await cache.put("a", 1)
        await cache.put("b", 2)
        await cache.put("c", 3)
        assert cache.size == 3

        # Adding a 4th entry should evict 'a' (least recently used)
        await cache.put("d", 4)
        assert cache.size == 3
        assert await cache.get("a") is None
        assert await cache.get("d") == 4

    async def test_lru_promotion_on_get(self) -> None:
        cache = TTLCache("test", max_size=3, ttl_seconds=60.0)
        await cache.put("a", 1)
        await cache.put("b", 2)
        await cache.put("c", 3)

        # Access 'a' to promote it
        await cache.get("a")

        # Now 'b' is the LRU — it should be evicted when we add 'd'
        await cache.put("d", 4)
        assert await cache.get("a") == 1  # promoted, still here
        assert await cache.get("b") is None  # evicted

    async def test_ttl_expiration(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=0.05)
        await cache.put("k1", "v1")
        assert await cache.get("k1") == "v1"

        # Wait for TTL to expire
        await asyncio.sleep(0.1)
        assert await cache.get("k1") is None

    async def test_put_updates_existing_entry(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=60.0)
        await cache.put("k1", "old")
        await cache.put("k1", "new")
        assert cache.size == 1
        assert await cache.get("k1") == "new"

    async def test_invalidate_existing_key(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=60.0)
        await cache.put("k1", "v1")
        removed = await cache.invalidate("k1")
        assert removed is True
        assert await cache.get("k1") is None

    async def test_invalidate_missing_key_returns_false(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=60.0)
        removed = await cache.invalidate("nope")
        assert removed is False

    async def test_clear_removes_all(self) -> None:
        cache = TTLCache("test", max_size=10, ttl_seconds=60.0)
        await cache.put("a", 1)
        await cache.put("b", 2)
        await cache.put("c", 3)
        count = await cache.clear()
        assert count == 3
        assert cache.size == 0

    async def test_stats_returns_expected_keys(self) -> None:
        cache = TTLCache("test-stats", max_size=100, ttl_seconds=300.0)
        await cache.put("a", 1)
        await cache.get("a")
        await cache.get("miss")
        s = cache.stats()
        assert s["name"] == "test-stats"
        assert s["size"] == 1
        assert s["max_size"] == 100
        assert s["ttl_seconds"] == 300.0
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert 0.0 <= s["hit_rate"] <= 1.0


# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    def test_deterministic(self) -> None:
        k1 = _make_cache_key("a", "b", "c")
        k2 = _make_cache_key("a", "b", "c")
        assert k1 == k2

    def test_different_inputs_different_keys(self) -> None:
        k1 = _make_cache_key("query", "hello")
        k2 = _make_cache_key("query", "world")
        assert k1 != k2

    def test_order_matters(self) -> None:
        k1 = _make_cache_key("a", "b")
        k2 = _make_cache_key("b", "a")
        assert k1 != k2


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------


class TestCacheManager:
    def test_creates_all_caches(self) -> None:
        mgr = CacheManager()
        assert mgr.query_cache is not None
        assert mgr.embedding_cache is not None
        assert mgr.graph_cache is not None
        assert mgr.voice_cache is not None
        assert mgr.thematic_cache is not None

    def test_custom_sizes_and_ttls(self) -> None:
        mgr = CacheManager(
            query_max_size=10,
            query_ttl=30.0,
            embedding_max_size=20,
            embedding_ttl=60.0,
            graph_max_size=5,
            graph_ttl=15.0,
        )
        assert mgr.query_cache._max_size == 10
        assert mgr.query_cache._ttl == 30.0
        assert mgr.embedding_cache._max_size == 20
        assert mgr.graph_cache._max_size == 5

    def test_key_generators_return_strings(self) -> None:
        mgr = CacheManager()
        assert isinstance(mgr.query_key("hello", author="test"), str)
        assert isinstance(mgr.embedding_key("text", "voyage", "v3"), str)
        assert isinstance(mgr.graph_key("themes", author="test"), str)
        assert isinstance(mgr.voice_key("author-1"), str)
        assert isinstance(mgr.thematic_key("author-1"), str)

    def test_distinct_keys_per_type(self) -> None:
        mgr = CacheManager()
        qk = mgr.query_key("test")
        ek = mgr.embedding_key("test", "voyage", "v3")
        gk = mgr.graph_key("test")
        vk = mgr.voice_key("test")
        tk = mgr.thematic_key("test")
        # All distinct from each other
        all_keys = {qk, ek, gk, vk, tk}
        assert len(all_keys) == 5

    async def test_invalidate_on_ingestion_clears_query_and_graph(self) -> None:
        mgr = CacheManager()
        await mgr.query_cache.put("q1", "result1")
        await mgr.graph_cache.put("g1", "graph_result")
        await mgr.embedding_cache.put("e1", "embed_result")

        result = await mgr.invalidate_on_ingestion(author_id="test-author")

        assert result["query_cleared"] == 1
        assert result["graph_cleared"] == 1
        # Embedding cache should be preserved
        assert await mgr.embedding_cache.get("e1") == "embed_result"

    async def test_invalidate_on_ingestion_clears_specific_author_voice(self) -> None:
        mgr = CacheManager()
        vk1 = mgr.voice_key("author-a")
        vk2 = mgr.voice_key("author-b")
        await mgr.voice_cache.put(vk1, "voice-a")
        await mgr.voice_cache.put(vk2, "voice-b")

        result = await mgr.invalidate_on_ingestion(author_id="author-a")

        assert result["voice_cleared"] == 1
        # author-b's voice should remain
        assert await mgr.voice_cache.get(vk2) == "voice-b"

    async def test_invalidate_on_ingestion_no_author_clears_all_voice_and_thematic(self) -> None:
        mgr = CacheManager()
        vk = mgr.voice_key("author-a")
        tk = mgr.thematic_key("author-a")
        await mgr.voice_cache.put(vk, "voice-a")
        await mgr.thematic_cache.put(tk, "thematic-a")

        result = await mgr.invalidate_on_ingestion(author_id=None)

        assert result["voice_cleared"] == 1
        assert result["thematic_cleared"] == 1

    async def test_invalidate_voice_profile(self) -> None:
        mgr = CacheManager()
        vk = mgr.voice_key("test-author")
        await mgr.voice_cache.put(vk, "profile")
        removed = await mgr.invalidate_voice_profile("test-author")
        assert removed is True
        assert await mgr.voice_cache.get(vk) is None

    async def test_invalidate_thematic_index(self) -> None:
        mgr = CacheManager()
        tk = mgr.thematic_key("test-author")
        await mgr.thematic_cache.put(tk, "thematic")
        removed = await mgr.invalidate_thematic_index("test-author")
        assert removed is True
        assert await mgr.thematic_cache.get(tk) is None

    def test_all_stats(self) -> None:
        mgr = CacheManager()
        stats = mgr.all_stats()
        assert "query" in stats
        assert "embedding" in stats
        assert "graph" in stats
        assert "voice_profile" in stats
        assert "thematic_index" in stats
        for s in stats.values():
            assert "name" in s
            assert "size" in s
            assert "hits" in s
            assert "misses" in s


# ---------------------------------------------------------------------------
# CachedEmbeddingProvider
# ---------------------------------------------------------------------------


def _make_inner_provider() -> AsyncMock:
    """Create a minimal AsyncMock that satisfies EmbeddingProvider interface."""
    inner = AsyncMock()
    inner.provider_name = "test-provider"
    inner.model_name = "test-model"
    inner.dimensions = 128
    return inner


class TestCachedEmbeddingProvider:
    async def test_embed_text_cache_miss_calls_inner(self) -> None:
        inner = _make_inner_provider()
        expected = EmbeddingResult(
            vector=[0.1] * 128,
            model="test-model",
            provider="test-provider",
            dimensions=128,
        )
        inner.embed_text.return_value = expected

        cache_mgr = CacheManager()
        provider = CachedEmbeddingProvider(inner, cache_mgr)
        result = await provider.embed_text("hello world")

        assert result == expected
        inner.embed_text.assert_awaited_once_with("hello world")

    async def test_embed_text_cache_hit_skips_inner(self) -> None:
        inner = _make_inner_provider()
        expected = EmbeddingResult(
            vector=[0.1] * 128,
            model="test-model",
            provider="test-provider",
            dimensions=128,
        )
        inner.embed_text.return_value = expected

        cache_mgr = CacheManager()
        provider = CachedEmbeddingProvider(inner, cache_mgr)

        # First call populates cache
        await provider.embed_text("hello world")
        # Second call should use cache
        result = await provider.embed_text("hello world")

        assert result == expected
        assert inner.embed_text.await_count == 1  # Only called once

    async def test_embed_query_uses_distinct_cache_key(self) -> None:
        inner = _make_inner_provider()
        text_result = EmbeddingResult(
            vector=[0.1] * 128, model="test-model", provider="test-provider", dimensions=128
        )
        query_result = EmbeddingResult(
            vector=[0.2] * 128, model="test-model", provider="test-provider", dimensions=128
        )
        inner.embed_text.return_value = text_result
        inner.embed_query.return_value = query_result

        cache_mgr = CacheManager()
        provider = CachedEmbeddingProvider(inner, cache_mgr)

        # These should use different cache keys
        tr = await provider.embed_text("same text")
        qr = await provider.embed_query("same text")

        assert tr.vector != qr.vector
        inner.embed_text.assert_awaited_once()
        inner.embed_query.assert_awaited_once()

    async def test_embed_batch_partial_cache(self) -> None:
        inner = _make_inner_provider()
        cache_mgr = CacheManager()
        provider = CachedEmbeddingProvider(inner, cache_mgr)

        # Pre-populate cache for "text-a"
        text_a_result = EmbeddingResult(
            vector=[0.1] * 128, model="test-model", provider="test-provider", dimensions=128
        )
        inner.embed_text.return_value = text_a_result
        await provider.embed_text("text-a")

        # Now batch embed ["text-a", "text-b", "text-c"]
        # Only "text-b" and "text-c" should go to the inner provider
        inner.embed_batch.return_value = BatchEmbeddingResult(
            vectors=[[0.2] * 128, [0.3] * 128],
            model="test-model",
            provider="test-provider",
            dimensions=128,
        )

        result = await provider.embed_batch(["text-a", "text-b", "text-c"])

        assert len(result.vectors) == 3
        assert result.vectors[0] == [0.1] * 128  # From cache
        assert result.vectors[1] == [0.2] * 128  # From provider
        assert result.vectors[2] == [0.3] * 128  # From provider
        inner.embed_batch.assert_awaited_once_with(["text-b", "text-c"])

    async def test_embed_batch_all_cached(self) -> None:
        inner = _make_inner_provider()
        cache_mgr = CacheManager()
        provider = CachedEmbeddingProvider(inner, cache_mgr)

        # Pre-populate cache for both texts
        for text, vec in [("text-a", [0.1] * 128), ("text-b", [0.2] * 128)]:
            inner.embed_text.return_value = EmbeddingResult(
                vector=vec, model="test-model", provider="test-provider", dimensions=128
            )
            await provider.embed_text(text)

        inner.embed_batch.reset_mock()
        result = await provider.embed_batch(["text-a", "text-b"])

        assert len(result.vectors) == 2
        # Should not call inner provider at all
        inner.embed_batch.assert_not_awaited()

    async def test_properties_delegate_to_inner(self) -> None:
        inner = _make_inner_provider()
        cache_mgr = CacheManager()
        provider = CachedEmbeddingProvider(inner, cache_mgr)

        assert provider.provider_name == "test-provider"
        assert provider.model_name == "test-model"
        assert provider.dimensions == 128

    async def test_close_delegates_to_inner(self) -> None:
        inner = _make_inner_provider()
        cache_mgr = CacheManager()
        provider = CachedEmbeddingProvider(inner, cache_mgr)

        await provider.close()
        inner.close.assert_awaited_once()
