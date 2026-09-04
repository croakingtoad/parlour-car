"""Vector similarity search over chunk embeddings using pgvector.

Wraps the PgEmbeddingRepository.similarity_search() method with
query embedding via the configured EmbeddingProvider, returning
uniform RetrievalResult objects with cosine similarity scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from author_library.errors import RetrievalError
from author_library.retrieval.models import RetrievalResult

if TYPE_CHECKING:
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.repositories import EmbeddingRepository

log = structlog.get_logger(__name__)


async def vector_search(
    query: str,
    *,
    embedding_provider: EmbeddingProvider,
    embedding_repo: EmbeddingRepository,
    limit: int = 20,
    score_threshold: float = 0.0,
    source_class_filter: str | None = None,
    work_id_filter: str | None = None,
    granularity_filter: str | None = None,
    subject_headings_filter: list[str] | None = None,
    genre_tags_filter: list[str] | None = None,
) -> list[RetrievalResult]:
    """Search chunk embeddings by cosine similarity.

    Embeds the query text using the provider's query-optimised method,
    then runs pgvector HNSW search on the chunk_embeddings table.

    Args:
        query: Natural-language search query.
        embedding_provider: Provider for generating the query embedding.
        embedding_repo: Repository for pgvector similarity search.
        limit: Maximum results to return.
        score_threshold: Minimum similarity score (0-1). Results below
            this threshold are discarded.  pgvector returns *distance*
            (lower = closer), so we convert to similarity = 1 - distance.
        source_class_filter: Optional filter on source_class.
        work_id_filter: Optional filter on work_id (post-filter).
        granularity_filter: Optional filter on granularity (post-filter).
        subject_headings_filter: Match works with any requested subject heading.
        genre_tags_filter: Match works with any requested genre tag.

    Returns:
        Ranked list of RetrievalResult objects, highest score first.

    Raises:
        RetrievalError: If embedding or search fails.
    """
    try:
        query_result = await embedding_provider.embed_query(query)
    except Exception as exc:
        raise RetrievalError(
            f"Failed to embed query: {exc}",
            context={"query_length": len(query)},
            cause=exc,
        ) from exc

    log.debug(
        "vector_search_query_embedded",
        provider=query_result.provider,
        model=query_result.model,
        dimensions=query_result.dimensions,
    )

    try:
        # Request more results than limit to account for post-filtering
        fetch_limit = limit * 3 if (work_id_filter or granularity_filter) else limit
        rows = await embedding_repo.similarity_search(
            query_result.vector,
            provider=query_result.provider,
            model=query_result.model,
            limit=fetch_limit,
            source_class_filter=source_class_filter,
            subject_headings_filter=subject_headings_filter,
            genre_tags_filter=genre_tags_filter,
        )
    except Exception as exc:
        raise RetrievalError(
            f"Vector similarity search failed: {exc}",
            context={"provider": query_result.provider, "model": query_result.model},
            cause=exc,
        ) from exc

    results: list[RetrievalResult] = []
    for row in rows:
        # pgvector <=> returns cosine distance; convert to similarity
        distance = float(row["distance"])
        similarity = 1.0 - distance

        if similarity < score_threshold:
            continue

        # Post-filters not supported by the repository query
        if work_id_filter and row["work_id"] != work_id_filter:
            continue
        if granularity_filter and row["granularity"] != granularity_filter:
            continue

        result_metadata: dict[str, object] = {}
        if "pass_number" in row:
            result_metadata["pass_number"] = row["pass_number"]
        if row.get("speaker") is not None:
            result_metadata["speaker"] = row["speaker"]

        results.append(
            RetrievalResult(
                chunk_id=UUID(str(row["chunk_id"])),
                work_id=row["work_id"],
                text=row["text"],
                score=similarity,
                granularity=row["granularity"],
                source_class=row["source_class"],
                source="vector",
                metadata=result_metadata,
            )
        )

        if len(results) >= limit:
            break

    log.info(
        "vector_search_complete",
        query_length=len(query),
        results=len(results),
        limit=limit,
        threshold=score_threshold,
    )
    return results
