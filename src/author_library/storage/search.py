"""Full-text search over the chunks table using PostgreSQL tsvector/tsquery.

Provides ranked search with ts_rank scoring and ts_headline snippet generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

    from author_library.storage.postgres import PostgresPool


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single full-text search hit."""

    chunk_id: UUID
    work_id: str
    rank: float
    snippet: str
    granularity: str
    source_class: str


async def search_fulltext(
    pool: PostgresPool,
    query: str,
    *,
    source_class_filter: str | None = None,
    work_filter: str | None = None,
    limit: int = 20,
) -> list[SearchResult]:
    """Run a full-text search over chunks using websearch_to_tsquery.

    Args:
        pool: PostgresPool instance.
        query: Natural-language search query.
        source_class_filter: Optional filter on source_class.
        work_filter: Optional filter on work_id.
        limit: Maximum number of results.

    Returns:
        Ranked list of SearchResult objects.
    """
    conditions = ["c.search_vector @@ websearch_to_tsquery('english', $1)"]
    params: list[object] = [query]
    idx = 2

    if source_class_filter is not None:
        conditions.append(f"c.source_class = ${idx}")
        params.append(source_class_filter)
        idx += 1

    if work_filter is not None:
        conditions.append(f"c.work_id = ${idx}")
        params.append(work_filter)
        idx += 1

    where_clause = " AND ".join(conditions)
    params.append(limit)

    sql = f"""
        SELECT
            c.id AS chunk_id,
            c.work_id,
            ts_rank(c.search_vector, websearch_to_tsquery('english', $1)) AS rank,
            ts_headline('english', c.text, websearch_to_tsquery('english', $1),
                        'StartSel=**, StopSel=**, MaxWords=60, MinWords=20') AS snippet,
            c.granularity,
            c.source_class
        FROM chunks c
        WHERE {where_clause}
        ORDER BY rank DESC
        LIMIT ${idx}
    """

    rows = await pool.fetch_all(sql, *params)
    return [
        SearchResult(
            chunk_id=row["chunk_id"],
            work_id=row["work_id"],
            rank=float(row["rank"]),
            snippet=row["snippet"],
            granularity=row["granularity"],
            source_class=row["source_class"],
        )
        for row in rows
    ]


async def search_phrase(
    pool: PostgresPool,
    exact_phrase: str,
    *,
    source_class_filter: str | None = None,
    work_filter: str | None = None,
    limit: int = 20,
) -> list[SearchResult]:
    """Run a phrase search using phraseto_tsquery for exact quote matching.

    Args:
        pool: PostgresPool instance.
        exact_phrase: Exact phrase to search for.
        source_class_filter: Optional filter on source_class.
        work_filter: Optional filter on work_id.
        limit: Maximum number of results.

    Returns:
        Ranked list of SearchResult objects.
    """
    conditions = ["c.search_vector @@ phraseto_tsquery('english', $1)"]
    params: list[object] = [exact_phrase]
    idx = 2

    if source_class_filter is not None:
        conditions.append(f"c.source_class = ${idx}")
        params.append(source_class_filter)
        idx += 1

    if work_filter is not None:
        conditions.append(f"c.work_id = ${idx}")
        params.append(work_filter)
        idx += 1

    where_clause = " AND ".join(conditions)
    params.append(limit)

    sql = f"""
        SELECT
            c.id AS chunk_id,
            c.work_id,
            ts_rank(c.search_vector, phraseto_tsquery('english', $1)) AS rank,
            ts_headline('english', c.text, phraseto_tsquery('english', $1),
                        'StartSel=**, StopSel=**, MaxWords=60, MinWords=20') AS snippet,
            c.granularity,
            c.source_class
        FROM chunks c
        WHERE {where_clause}
        ORDER BY rank DESC
        LIMIT ${idx}
    """

    rows = await pool.fetch_all(sql, *params)
    return [
        SearchResult(
            chunk_id=row["chunk_id"],
            work_id=row["work_id"],
            rank=float(row["rank"]),
            snippet=row["snippet"],
            granularity=row["granularity"],
            source_class=row["source_class"],
        )
        for row in rows
    ]
