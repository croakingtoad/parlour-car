"""Reciprocal Rank Fusion (RRF) for hybrid retrieval.

Combines results from vector similarity search and full-text search
into a single fused ranked list, with configurable weights and
deduplication by chunk_id.
"""

from __future__ import annotations

import structlog

from author_library.retrieval.models import RetrievalResult

log = structlog.get_logger(__name__)

# Standard RRF constant (prevents division by zero for top-ranked items)
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    *result_lists: list[RetrievalResult],
    k: int = DEFAULT_RRF_K,
    weights: list[float] | None = None,
    limit: int = 20,
) -> list[RetrievalResult]:
    """Fuse multiple ranked retrieval result lists using RRF.

    For each result appearing in any list, its fused score is:
        score = sum(weight_i / (k + rank_i))
    where rank_i is the 1-based rank in list i (0 if absent).

    Args:
        *result_lists: One or more ranked lists of RetrievalResult.
        k: RRF constant (default 60). Higher values reduce the impact
            of top-ranked items relative to lower-ranked ones.
        weights: Optional per-list weights. Defaults to equal weighting.
            Must have same length as result_lists if provided.
        limit: Maximum number of fused results to return.

    Returns:
        Fused ranked list of RetrievalResult, highest score first.
        Each result carries its fused score and a metadata dict
        recording which sources contributed and their original ranks.
    """
    n_lists = len(result_lists)
    if n_lists == 0:
        return []

    if weights is None:
        weights = [1.0] * n_lists
    elif len(weights) != n_lists:
        msg = f"weights length ({len(weights)}) must match result_lists count ({n_lists})"
        raise ValueError(msg)

    # Accumulate RRF scores by chunk_id
    # Keep the best RetrievalResult instance for each chunk (highest original score)
    scores: dict[str, float] = {}  # chunk_id hex -> fused score
    best_result: dict[str, RetrievalResult] = {}  # chunk_id hex -> best result
    source_ranks: dict[str, dict[str, int]] = {}  # chunk_id hex -> {source: rank}

    for list_idx, results in enumerate(result_lists):
        weight = weights[list_idx]
        for rank_0based, result in enumerate(results):
            cid = str(result.chunk_id)
            rank_1based = rank_0based + 1
            rrf_contrib = weight / (k + rank_1based)

            scores[cid] = scores.get(cid, 0.0) + rrf_contrib

            # Track source contributions
            if cid not in source_ranks:
                source_ranks[cid] = {}
            source_ranks[cid][result.source] = rank_1based

            # Keep the result with the highest original score
            if cid not in best_result or result.score > best_result[cid].score:
                best_result[cid] = result

    # Sort by fused score descending
    ranked_cids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    fused: list[RetrievalResult] = []
    for cid in ranked_cids[:limit]:
        original = best_result[cid]
        fused.append(
            RetrievalResult(
                chunk_id=original.chunk_id,
                work_id=original.work_id,
                text=original.text,
                score=scores[cid],
                granularity=original.granularity,
                source_class=original.source_class,
                source="fusion",
                metadata={
                    "source_ranks": source_ranks[cid],
                    "original_score": original.score,
                    "original_source": original.source,
                },
            )
        )

    log.info(
        "rrf_fusion_complete",
        input_lists=n_lists,
        unique_chunks=len(scores),
        output_count=len(fused),
    )
    return fused
