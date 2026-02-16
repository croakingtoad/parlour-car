"""Tests for vector similarity search."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from author_library.retrieval.vector_search import vector_search

if TYPE_CHECKING:
    from .conftest import (
        DeterministicEmbeddingProvider,
        InMemoryEmbeddingRepository,
    )


@pytest.mark.asyncio
async def test_vector_search_returns_results(
    embedding_provider: DeterministicEmbeddingProvider,
    embedding_repo: InMemoryEmbeddingRepository,
) -> None:
    """Vector search returns ranked results for a relevant query."""
    results = await vector_search(
        "Lewis sermon Weight of Glory",
        embedding_provider=embedding_provider,
        embedding_repo=embedding_repo,
        limit=5,
    )
    assert len(results) > 0
    assert all(r.source == "vector" for r in results)
    # Scores should be between 0 and 1
    assert all(0.0 <= r.score <= 1.0 for r in results)
    # Should be sorted by score descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_vector_search_respects_limit(
    embedding_provider: DeterministicEmbeddingProvider,
    embedding_repo: InMemoryEmbeddingRepository,
) -> None:
    """Vector search respects the limit parameter."""
    results = await vector_search(
        "moral law Christianity",
        embedding_provider=embedding_provider,
        embedding_repo=embedding_repo,
        limit=2,
    )
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_vector_search_score_threshold(
    embedding_provider: DeterministicEmbeddingProvider,
    embedding_repo: InMemoryEmbeddingRepository,
) -> None:
    """Vector search filters by score threshold."""
    results = await vector_search(
        "Lewis sermon Weight of Glory",
        embedding_provider=embedding_provider,
        embedding_repo=embedding_repo,
        limit=10,
        score_threshold=0.99,  # Very high threshold
    )
    # With a very high threshold, we might get few or no results
    assert all(r.score >= 0.99 for r in results)


@pytest.mark.asyncio
async def test_vector_search_source_class_filter(
    embedding_provider: DeterministicEmbeddingProvider,
    embedding_repo: InMemoryEmbeddingRepository,
) -> None:
    """Vector search filters by source_class."""
    results = await vector_search(
        "Lewis conversion biography",
        embedding_provider=embedding_provider,
        embedding_repo=embedding_repo,
        limit=10,
        source_class_filter="primary",
    )
    assert all(r.source_class == "primary" for r in results)


@pytest.mark.asyncio
async def test_vector_search_granularity_filter(
    embedding_provider: DeterministicEmbeddingProvider,
    embedding_repo: InMemoryEmbeddingRepository,
) -> None:
    """Vector search filters by granularity."""
    results = await vector_search(
        "Lewis Christianity published",
        embedding_provider=embedding_provider,
        embedding_repo=embedding_repo,
        limit=10,
        granularity_filter="micro",
    )
    assert all(r.granularity == "micro" for r in results)


@pytest.mark.asyncio
async def test_vector_search_result_fields(
    embedding_provider: DeterministicEmbeddingProvider,
    embedding_repo: InMemoryEmbeddingRepository,
) -> None:
    """Each result has all required fields populated."""
    results = await vector_search(
        "imagination Romantic tradition",
        embedding_provider=embedding_provider,
        embedding_repo=embedding_repo,
        limit=5,
    )
    for r in results:
        assert r.chunk_id is not None
        assert r.work_id
        assert r.text
        assert r.granularity
        assert r.source_class
        assert r.source == "vector"


@pytest.mark.asyncio
async def test_vector_search_empty_query(
    embedding_provider: DeterministicEmbeddingProvider,
    embedding_repo: InMemoryEmbeddingRepository,
) -> None:
    """Vector search handles empty-ish queries gracefully."""
    results = await vector_search(
        "a",
        embedding_provider=embedding_provider,
        embedding_repo=embedding_repo,
        limit=5,
    )
    # Should not raise, results may vary
    assert isinstance(results, list)
