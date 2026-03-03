"""Caching wrapper for embedding providers.

Intercepts embed_text and embed_query calls, checking the in-memory
TTL cache before invoking the underlying provider. Batch embeddings
are handled per-text: cached texts skip the API call, only uncached
texts are sent to the provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from author_library.embeddings.base import (
    BatchEmbeddingResult,
    EmbeddingProvider,
    EmbeddingResult,
)

if TYPE_CHECKING:
    from author_library.cache import CacheManager

log = structlog.get_logger(__name__)


class CachedEmbeddingProvider(EmbeddingProvider):
    """Wraps an EmbeddingProvider with in-memory TTL caching.

    Embedding results are keyed by (text, provider, model) and cached
    to avoid redundant API calls for repeated or unchanged content.
    """

    def __init__(self, inner: EmbeddingProvider, cache: CacheManager) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed_text(self, text: str) -> EmbeddingResult:
        key = self._cache.embedding_key(text, self.provider_name, self.model_name)
        cached = await self._cache.embedding_cache.get(key)
        if cached is not None:
            return cached

        result = await self._inner.embed_text(text)
        await self._cache.embedding_cache.put(key, result)
        return result

    async def embed_query(self, text: str) -> EmbeddingResult:
        # Query embeddings may differ from document embeddings (e.g. Voyage AI),
        # so use a distinct key prefix.
        key = self._cache.embedding_key(f"query:{text}", self.provider_name, self.model_name)
        cached = await self._cache.embedding_cache.get(key)
        if cached is not None:
            return cached

        result = await self._inner.embed_query(text)
        await self._cache.embedding_cache.put(key, result)
        return result

    async def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        """Cache-aware batch embedding.

        Checks cache for each text. Only uncached texts are sent to the
        provider. Results are reassembled in the original order.
        """
        keys = [
            self._cache.embedding_key(t, self.provider_name, self.model_name) for t in texts
        ]
        cached_results: list[EmbeddingResult | None] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, key in enumerate(keys):
            cached = await self._cache.embedding_cache.get(key)
            if cached is not None:
                cached_results.append(cached)
            else:
                cached_results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(texts[i])

        # Fetch uncached embeddings from provider
        if uncached_texts:
            batch_result = await self._inner.embed_batch(uncached_texts)
            for idx, vector in zip(uncached_indices, batch_result.vectors, strict=True):
                result = EmbeddingResult(
                    vector=vector,
                    model=batch_result.model,
                    provider=batch_result.provider,
                    dimensions=batch_result.dimensions,
                )
                cached_results[idx] = result
                await self._cache.embedding_cache.put(keys[idx], result)

        # Reassemble full vector list
        vectors = [r.vector for r in cached_results if r is not None]
        return BatchEmbeddingResult(
            vectors=vectors,
            model=self.model_name,
            provider=self.provider_name,
            dimensions=self.dimensions,
        )

    async def close(self) -> None:
        await self._inner.close()
