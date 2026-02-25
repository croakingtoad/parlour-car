"""Tests for full-text and phrase search wrappers.

These tests verify the text_search module's wrapping of the storage.search
module. Integration tests against a real PostgreSQL instance with tsvector
indexes are in the storage test suite; these tests verify the wrapper
logic using a fake PostgresPool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pytest

from author_library.retrieval.text_search import keyword_search, phrase_search

# ---------------------------------------------------------------------------
# Fake PostgresPool that returns controlled results
# ---------------------------------------------------------------------------


@dataclass
class FakeRow:
    """Simulates an asyncpg.Record with dict-like access."""

    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class FakePostgresPool:
    """A fake pool that returns pre-configured search results."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def fetch_all(self, sql: str, *args: Any) -> list[FakeRow]:
        return [FakeRow(r) for r in self._rows]

    async def fetch_one(self, sql: str, *args: Any) -> FakeRow | None:
        return FakeRow(self._rows[0]) if self._rows else None


SAMPLE_ROWS = [
    {
        "chunk_id": uuid4(),
        "work_id": "lewis--mere-christianity",
        "rank": 0.85,
        "snippet": "In **Mere Christianity** Lewis argues that the **moral law**...",
        "granularity": "meso",
        "source_class": "primary",
        "pass_number": 1,
        "speaker": None,
    },
    {
        "chunk_id": uuid4(),
        "work_id": "lewis--weight-of-glory",
        "rank": 0.72,
        "snippet": "The **Weight** of **Glory** is perhaps Lewis's most eloquent sermon...",
        "granularity": "meso",
        "source_class": "primary",
        "pass_number": 1,
        "speaker": None,
    },
]


@pytest.mark.asyncio
async def test_keyword_search_returns_retrieval_results() -> None:
    """keyword_search wraps fulltext results as RetrievalResult objects."""
    pool = FakePostgresPool(SAMPLE_ROWS)  # type: ignore[arg-type]
    results = await keyword_search(pool, "moral law Christianity")  # type: ignore[arg-type]

    assert len(results) == 2
    assert results[0].source == "fulltext"
    assert results[0].work_id == "lewis--mere-christianity"
    assert results[0].score == 0.85
    assert results[0].granularity == "meso"
    assert results[0].source_class == "primary"


@pytest.mark.asyncio
async def test_keyword_search_preserves_rank_order() -> None:
    """Results preserve the rank order from PostgreSQL."""
    pool = FakePostgresPool(SAMPLE_ROWS)  # type: ignore[arg-type]
    results = await keyword_search(pool, "Lewis sermon")  # type: ignore[arg-type]

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_phrase_search_returns_retrieval_results() -> None:
    """phrase_search wraps phrase results as RetrievalResult objects."""
    pool = FakePostgresPool(SAMPLE_ROWS[:1])  # type: ignore[arg-type]
    results = await phrase_search(pool, "moral law")  # type: ignore[arg-type]

    assert len(results) == 1
    assert results[0].source == "phrase"
    assert results[0].work_id == "lewis--mere-christianity"


@pytest.mark.asyncio
async def test_keyword_search_empty_results() -> None:
    """keyword_search returns empty list when no matches found."""
    pool = FakePostgresPool([])  # type: ignore[arg-type]
    results = await keyword_search(pool, "nonexistent query")  # type: ignore[arg-type]

    assert results == []


@pytest.mark.asyncio
async def test_phrase_search_empty_results() -> None:
    """phrase_search returns empty list when no matches found."""
    pool = FakePostgresPool([])  # type: ignore[arg-type]
    results = await phrase_search(pool, "nonexistent exact phrase")  # type: ignore[arg-type]

    assert results == []


@pytest.mark.asyncio
async def test_keyword_search_chunk_id_is_uuid() -> None:
    """Results have proper UUID chunk_id values."""
    pool = FakePostgresPool(SAMPLE_ROWS)  # type: ignore[arg-type]
    results = await keyword_search(pool, "Christianity")  # type: ignore[arg-type]

    for r in results:
        assert isinstance(r.chunk_id, UUID)


@pytest.mark.asyncio
async def test_phrase_search_deduplicates_by_chunk_id() -> None:
    """phrase_search deduplicates results with the same chunk_id."""
    duplicate_id = uuid4()
    rows_with_dupes = [
        {
            "chunk_id": duplicate_id,
            "work_id": "lewis--mere-christianity",
            "rank": 0.85,
            "snippet": "In the neighborhood of faith...",
            "granularity": "meso",
            "source_class": "primary",
            "pass_number": 1,
            "speaker": None,
        },
        {
            "chunk_id": duplicate_id,
            "work_id": "lewis--mere-christianity",
            "rank": 0.80,
            "snippet": "In the neighborhood of faith...",
            "granularity": "micro",
            "source_class": "primary",
            "pass_number": 1,
            "speaker": None,
        },
        {
            "chunk_id": uuid4(),
            "work_id": "lewis--weight-of-glory",
            "rank": 0.72,
            "snippet": "The neighborhood of glory...",
            "granularity": "meso",
            "source_class": "primary",
            "pass_number": 1,
            "speaker": None,
        },
    ]
    pool = FakePostgresPool(rows_with_dupes)  # type: ignore[arg-type]
    results = await phrase_search(pool, "neighborhood")  # type: ignore[arg-type]

    # Should have 2 results, not 3 (duplicate chunk_id removed)
    assert len(results) == 2
    # First result should be the higher-ranked one (order preserved from PG)
    assert results[0].score == 0.85
