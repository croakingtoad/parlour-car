"""Graph-augmented retrieval via Neo4j knowledge graph expansion.

After initial retrieval returns primary chunks, follows Neo4j edges
to expand context with related passages while preserving source-class
provenance. NEVER presents secondary material as primary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from author_library.errors import RetrievalError
from author_library.retrieval.models import GraphExpansionResult, RetrievalResult

if TYPE_CHECKING:
    from author_library.graph.queries import GraphQueryService

log = structlog.get_logger(__name__)


async def expand_via_engagement(
    query_service: GraphQueryService,
    seed_chunk_ids: list[str],
    *,
    max_depth: int = 3,
) -> list[GraphExpansionResult]:
    """Follow ENGAGES_WITH edges from seed chunks to contextual sources.

    Discovers passages from works the author engages with, preserving
    their original source_class so contextual material is never
    misrepresented as primary.

    Args:
        query_service: GraphQueryService for Neo4j traversal.
        seed_chunk_ids: Chunk IDs from initial retrieval pass.
        max_depth: Maximum engagement chain depth.

    Returns:
        List of GraphExpansionResult with relationship metadata.
    """
    results: list[GraphExpansionResult] = []
    seen: set[str] = set(seed_chunk_ids)

    for chunk_id in seed_chunk_ids:
        try:
            chain = await query_service.get_engagement_chain(
                chunk_id, max_depth=max_depth
            )
        except Exception as exc:
            log.warning(
                "engagement_chain_failed",
                chunk_id=chunk_id,
                error=str(exc),
            )
            continue

        if chain is None:
            continue

        for link in chain.links:
            target = link.target_chunk
            if target.chunk_id in seen:
                continue
            seen.add(target.chunk_id)

            results.append(
                GraphExpansionResult(
                    chunk_id=target.chunk_id,
                    work_id=target.work_id,
                    text_preview=target.text_preview,
                    granularity=target.granularity,
                    source_class=target.source_class,
                    relationship_type="ENGAGES_WITH",
                    confidence=link.confidence,
                    evidence=link.evidence,
                )
            )

    log.info(
        "engagement_expansion_complete",
        seed_count=len(seed_chunk_ids),
        expanded=len(results),
    )
    return results


async def expand_via_themes(
    query_service: GraphQueryService,
    theme_names: list[str],
    *,
    exclude_chunk_ids: set[str] | None = None,
) -> list[GraphExpansionResult]:
    """Follow EXPLORES_THEME edges to find all chunks related to themes.

    Args:
        query_service: GraphQueryService for Neo4j traversal.
        theme_names: Theme names identified from initial results.
        exclude_chunk_ids: Chunk IDs to exclude (already retrieved).

    Returns:
        List of GraphExpansionResult with theme relationship metadata.
    """
    excluded = exclude_chunk_ids or set()
    results: list[GraphExpansionResult] = []
    seen: set[str] = set(excluded)

    for theme_name in theme_names:
        try:
            subgraph = await query_service.get_theme_subgraph(theme_name)
        except Exception as exc:
            log.warning(
                "theme_subgraph_failed",
                theme_name=theme_name,
                error=str(exc),
            )
            continue

        if subgraph is None:
            continue

        for chunk in subgraph.chunks:
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)

            results.append(
                GraphExpansionResult(
                    chunk_id=chunk.chunk_id,
                    work_id=chunk.work_id,
                    text_preview=chunk.text_preview,
                    granularity=chunk.granularity,
                    source_class=chunk.source_class,
                    relationship_type="EXPLORES_THEME",
                    confidence="high",
                    evidence=f"Explores theme: {subgraph.theme_name}",
                )
            )

    log.info(
        "theme_expansion_complete",
        theme_count=len(theme_names),
        expanded=len(results),
    )
    return results


async def expand_via_argument_chains(
    query_service: GraphQueryService,
    theme_names: list[str],
    *,
    exclude_chunk_ids: set[str] | None = None,
) -> list[GraphExpansionResult]:
    """Follow DEVELOPS_FROM chains for argument progression across works.

    Args:
        query_service: GraphQueryService for Neo4j traversal.
        theme_names: Theme names to trace argument progression for.
        exclude_chunk_ids: Chunk IDs to exclude.

    Returns:
        List of GraphExpansionResult with development relationship metadata.
    """
    excluded = exclude_chunk_ids or set()
    results: list[GraphExpansionResult] = []
    seen: set[str] = set(excluded)

    for theme_name in theme_names:
        try:
            arg_data = await query_service.get_argument_evolution(theme_name)
        except Exception as exc:
            log.warning(
                "argument_chain_failed",
                theme_name=theme_name,
                error=str(exc),
            )
            continue

        for arg_node in arg_data.arguments:
            for chunk in arg_node.source_chunks:
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)

                results.append(
                    GraphExpansionResult(
                        chunk_id=chunk.chunk_id,
                        work_id=chunk.work_id,
                        text_preview=chunk.text_preview,
                        granularity=chunk.granularity,
                        source_class=chunk.source_class,
                        relationship_type="DEVELOPS_FROM",
                        confidence="medium",
                        evidence=(
                            f"Argument '{arg_node.canonical_name}': "
                            f"{arg_node.claim}"
                        ),
                    )
                )

    log.info(
        "argument_chain_expansion_complete",
        theme_count=len(theme_names),
        expanded=len(results),
    )
    return results


async def graph_augmented_retrieval(
    query_service: GraphQueryService,
    seed_results: list[RetrievalResult],
    *,
    theme_names: list[str] | None = None,
    max_engagement_depth: int = 3,
) -> list[GraphExpansionResult]:
    """Run full graph-augmented retrieval expansion.

    Combines engagement, theme, and argument chain expansion
    from seed results into a single deduplicated expansion set.

    Args:
        query_service: GraphQueryService for Neo4j traversal.
        seed_results: Initial retrieval results to expand from.
        theme_names: Optional theme names for theme/argument expansion.
            If None, only engagement expansion is performed.
        max_engagement_depth: Maximum depth for engagement chains.

    Returns:
        Combined, deduplicated list of GraphExpansionResult objects.

    Raises:
        RetrievalError: If all expansion methods fail.
    """
    seed_chunk_ids = [str(r.chunk_id) for r in seed_results]
    seed_set = set(seed_chunk_ids)
    all_results: list[GraphExpansionResult] = []
    errors: list[str] = []

    # 1. Engagement expansion
    try:
        engagement_results = await expand_via_engagement(
            query_service, seed_chunk_ids, max_depth=max_engagement_depth
        )
        all_results.extend(engagement_results)
    except Exception as exc:
        errors.append(f"engagement: {exc}")

    # Track all seen chunk IDs for dedup across expansion methods
    seen = seed_set | {r.chunk_id for r in all_results}

    # 2. Theme expansion (if themes provided)
    if theme_names:
        try:
            theme_results = await expand_via_themes(
                query_service, theme_names, exclude_chunk_ids=seen
            )
            all_results.extend(theme_results)
            seen.update(r.chunk_id for r in theme_results)
        except Exception as exc:
            errors.append(f"themes: {exc}")

        # 3. Argument chain expansion
        try:
            arg_results = await expand_via_argument_chains(
                query_service, theme_names, exclude_chunk_ids=seen
            )
            all_results.extend(arg_results)
        except Exception as exc:
            errors.append(f"argument_chains: {exc}")

    if not all_results and errors:
        raise RetrievalError(
            f"All graph expansion methods failed: {'; '.join(errors)}",
            context={"seed_count": len(seed_chunk_ids)},
        )

    log.info(
        "graph_augmented_retrieval_complete",
        seed_count=len(seed_chunk_ids),
        total_expanded=len(all_results),
        errors=errors or None,
    )
    return all_results
