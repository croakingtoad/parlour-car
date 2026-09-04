"""Full-text and BM25 search wrapper over the storage.search module.

Wraps search_fulltext() and search_phrase() to return uniform
RetrievalResult objects compatible with the vector search output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from author_library.errors import RetrievalError
from author_library.retrieval.models import RetrievalResult
from author_library.storage.search import search_fulltext, search_phrase

if TYPE_CHECKING:
    from author_library.storage.postgres import PostgresPool

log = structlog.get_logger(__name__)


async def keyword_search(
    pool: PostgresPool,
    query: str,
    *,
    source_class_filter: str | None = None,
    work_filter: str | None = None,
    subject_headings_filter: list[str] | None = None,
    genre_tags_filter: list[str] | None = None,
    limit: int = 20,
) -> list[RetrievalResult]:
    """Run full-text keyword search with ts_rank scoring.

    Args:
        pool: PostgresPool instance.
        query: Natural-language search query.
        source_class_filter: Optional filter on source_class.
        work_filter: Optional filter on work_id.
        subject_headings_filter: Match works with any requested subject heading.
        genre_tags_filter: Match works with any requested genre tag.
        limit: Maximum number of results.

    Returns:
        Ranked list of RetrievalResult objects.

    Raises:
        RetrievalError: If the search operation fails.
    """
    try:
        hits = await search_fulltext(
            pool,
            query,
            source_class_filter=source_class_filter,
            work_filter=work_filter,
            subject_headings_filter=subject_headings_filter,
            genre_tags_filter=genre_tags_filter,
            limit=limit,
        )
    except Exception as exc:
        raise RetrievalError(
            f"Full-text keyword search failed: {exc}",
            context={"query": query},
            cause=exc,
        ) from exc

    results = [
        RetrievalResult(
            chunk_id=hit.chunk_id,
            work_id=hit.work_id,
            text=hit.snippet,
            score=hit.rank,
            granularity=hit.granularity,
            source_class=hit.source_class,
            source="fulltext",
            metadata={
                k: v
                for k, v in [
                    ("pass_number", hit.pass_number),
                    ("speaker", hit.speaker),
                ]
                if v is not None
            },
        )
        for hit in hits
    ]

    log.info(
        "keyword_search_complete",
        query_length=len(query),
        results=len(results),
    )
    return results


async def phrase_search(
    pool: PostgresPool,
    exact_phrase: str,
    *,
    source_class_filter: str | None = None,
    work_filter: str | None = None,
    limit: int = 20,
) -> list[RetrievalResult]:
    """Run exact phrase search for quote lookups.

    Results are deduplicated by chunk_id, keeping the highest-ranked
    hit per chunk. This prevents duplicate results when the same chunk
    matches a phrase search through overlapping index entries.

    Args:
        pool: PostgresPool instance.
        exact_phrase: Exact phrase to search for.
        source_class_filter: Optional filter on source_class.
        work_filter: Optional filter on work_id.
        limit: Maximum number of results.

    Returns:
        Ranked, deduplicated list of RetrievalResult objects.

    Raises:
        RetrievalError: If the search operation fails.
    """
    try:
        hits = await search_phrase(
            pool,
            exact_phrase,
            source_class_filter=source_class_filter,
            work_filter=work_filter,
            limit=limit,
        )
    except Exception as exc:
        raise RetrievalError(
            f"Phrase search failed: {exc}",
            context={"phrase_length": len(exact_phrase)},
            cause=exc,
        ) from exc

    # Deduplicate by chunk_id, keeping the highest-ranked hit per chunk
    seen_chunk_ids: set[str] = set()
    results: list[RetrievalResult] = []
    for hit in hits:
        cid = str(hit.chunk_id)
        if cid in seen_chunk_ids:
            continue
        seen_chunk_ids.add(cid)
        results.append(
            RetrievalResult(
                chunk_id=hit.chunk_id,
                work_id=hit.work_id,
                text=hit.snippet,
                score=hit.rank,
                granularity=hit.granularity,
                source_class=hit.source_class,
                source="phrase",
                metadata={
                    k: v
                    for k, v in [
                        ("pass_number", hit.pass_number),
                        ("speaker", hit.speaker),
                    ]
                    if v is not None
                },
            )
        )

    log.info(
        "phrase_search_complete",
        phrase_length=len(exact_phrase),
        raw_hits=len(hits),
        deduplicated_results=len(results),
    )
    return results
