"""Composable query MCP tool handlers (Epic C).

Provides:
  - search_chunks: Filtered vector + FTS search with provenance rules.
  - get_passage_links: Direct Neo4j traversal for passage links.
  - manage_vocabulary: Vocabulary term management (list/propose/promote/merge/deprecate).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from author_library.catalog.models import SourceClass
from author_library.errors import RetrievalError
from author_library.graph.queries import GraphQueryService
from author_library.retrieval.text_search import keyword_search
from author_library.retrieval.vector_search import vector_search

if TYPE_CHECKING:
    from author_library.cache import CacheManager
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# C1: search_chunks
# ---------------------------------------------------------------------------


async def handle_search_chunks(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
    cache_manager: CacheManager | None = None,
) -> str:
    """Handle the search_chunks MCP tool call.

    Searches chunks using combined vector + FTS retrieval with
    source-class filtering and provenance rules.

    Arguments:
        query (str): Search query.
        filters (dict, optional): source_class[], work_ids[], speaker, granularity[],
            themes[], pass_number, subject_headings[], genre_tags[].
        include_personal (bool, optional): Include Personal source class (default true).
        max_results (int, optional): Maximum results (default 10).
        include_passage_links (bool, optional): Include passage links (default true).

    Returns:
        JSON with results array including provenance_rules.
    """
    query = arguments.get("query")
    if not query:
        raise RetrievalError("query is required", context={"arguments": arguments})

    filters = arguments.get("filters") or {}
    include_personal = arguments.get("include_personal", True)
    max_results = arguments.get("max_results", 10)
    include_passage_links = arguments.get("include_passage_links", True)

    # Extract filter parameters
    source_class_filter = filters.get("source_class")  # list or None
    work_ids_filter = filters.get("work_ids")  # list or None
    granularity_filter = filters.get("granularity")  # list or None
    pass_number_filter = filters.get("pass_number")  # int or None
    speaker_filter = filters.get("speaker")  # str or None
    themes_filter = filters.get("themes")  # list[str] or None
    subject_headings_filter = filters.get("subject_headings")  # list[str] or None
    genre_tags_filter = filters.get("genre_tags")  # list[str] or None

    # Build source class filter string for search functions
    # If multiple source classes given, we'll filter post-search
    sc_filter_str = None
    if source_class_filter and len(source_class_filter) == 1:
        sc_filter_str = source_class_filter[0]

    # Work ID filter — only supported as single value for the search functions
    work_filter_str = None
    if work_ids_filter and len(work_ids_filter) == 1:
        work_filter_str = work_ids_filter[0]

    # Granularity — pick first if single
    gran_filter_str = None
    if granularity_filter and len(granularity_filter) == 1:
        gran_filter_str = granularity_filter[0]

    # Run vector search and keyword search in parallel
    import asyncio

    vector_task = asyncio.create_task(
        vector_search(
            query,
            embedding_provider=embedding_provider,
            embedding_repo=storage.embeddings,
            limit=max_results * 2,
            source_class_filter=sc_filter_str,
            work_id_filter=work_filter_str,
            granularity_filter=gran_filter_str,
            subject_headings_filter=subject_headings_filter,
            genre_tags_filter=genre_tags_filter,
        )
    )
    keyword_task = asyncio.create_task(
        keyword_search(
            storage.pg,
            query,
            source_class_filter=sc_filter_str,
            work_filter=work_filter_str,
            subject_headings_filter=subject_headings_filter,
            genre_tags_filter=genre_tags_filter,
            limit=max_results * 2,
        )
    )

    vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)

    # Merge and deduplicate
    seen_chunk_ids: set[str] = set()
    merged: list[Any] = []

    for r in vector_results:
        cid = str(r.chunk_id)
        if cid not in seen_chunk_ids:
            seen_chunk_ids.add(cid)
            merged.append(r)

    for r in keyword_results:
        cid = str(r.chunk_id)
        if cid not in seen_chunk_ids:
            seen_chunk_ids.add(cid)
            merged.append(r)

    # Apply post-search filters
    if source_class_filter and len(source_class_filter) > 1:
        sc_set = set(source_class_filter)
        merged = [r for r in merged if r.source_class in sc_set]

    if work_ids_filter and len(work_ids_filter) > 1:
        wid_set = set(work_ids_filter)
        merged = [r for r in merged if r.work_id in wid_set]

    if granularity_filter and len(granularity_filter) > 1:
        gran_set = set(granularity_filter)
        merged = [r for r in merged if r.granularity in gran_set]

    # Apply speaker filter (speaker is stored in chunk metadata JSONB)
    if speaker_filter:
        merged = [
            r for r in merged if r.metadata.get("speaker") == speaker_filter
        ]

    # Apply pass_number filter (pass_number is stored on the chunks table)
    if pass_number_filter is not None:
        merged = [
            r
            for r in merged
            if r.metadata.get("pass_number") == pass_number_filter
        ]

    # Apply themes filter via Neo4j (themes are stored as EXPLORES_THEME edges)
    if themes_filter:
        themes_set = {t.lower() for t in themes_filter}
        filtered_by_theme: list[Any] = []
        for r in merged:
            try:
                chunk_themes = await storage.graph.get_themes_for_chunk(
                    str(r.chunk_id)
                )
                chunk_theme_names = {
                    t.get("canonical_name", t.get("name", "")).lower()
                    for t in chunk_themes
                }
                if chunk_theme_names & themes_set:
                    filtered_by_theme.append(r)
            except Exception:
                # If Neo4j is unavailable, skip theme filtering for this chunk
                log.warning(
                    "theme_filter_failed",
                    chunk_id=str(r.chunk_id),
                    exc_info=True,
                )
        merged = filtered_by_theme

    # Exclude personal if not requested
    if not include_personal:
        merged = [r for r in merged if r.source_class != "personal"]

    # Sort by score descending
    merged.sort(key=lambda r: r.score, reverse=True)
    merged = merged[:max_results]

    # Build results with provenance rules
    graph_service = GraphQueryService(storage.neo4j, cache=cache_manager)
    results = []

    for r in merged:
        # Build metadata
        work_info = await storage.works.get(r.work_id)
        metadata = {
            "work_id": r.work_id,
            "work_title": work_info.get("title", "") if work_info else "",
            "author": work_info.get("author", "") if work_info else "",
            "source_class": r.source_class,
            "granularity": r.granularity,
            "chapter": getattr(r, "chapter", None),
            "voice_profile_eligible": r.source_class == "primary",
        }

        # Build provenance rules
        provenance = _build_provenance_rules(r.source_class)

        entry: dict[str, Any] = {
            "chunk_id": str(r.chunk_id),
            "text": r.text,
            "metadata": metadata,
            "relevance_score": round(r.score, 4),
            "provenance_rules": provenance,
        }

        # Optionally include passage links
        if include_passage_links:
            try:
                chain = await graph_service.get_engagement_chain(str(r.chunk_id))
                if chain and chain.links:
                    entry["links"] = [
                        {
                            "target_chunk_id": link.target_chunk.chunk_id,
                            "target_work": link.target_chunk.work_id,
                            "link_type": link.link_type,
                            "confidence": link.confidence,
                            "annotation": getattr(link, "evidence", ""),
                        }
                        for link in chain.links[:5]
                    ]
                else:
                    entry["links"] = []
            except Exception:
                entry["links"] = []

        results.append(entry)

    result = {
        "results": results,
        "total_available": len(results),
    }

    return json.dumps(result, indent=2)


def _build_provenance_rules(source_class: str) -> dict[str, Any]:
    """Build provenance rules based on source class."""
    rules: dict[str, dict[str, Any]] = {
        "primary": {
            "attribution": "Quote directly as the author's own words",
            "voice_eligible": True,
            "presentation_guidance": "Present as the author's own argument",
        },
        "secondary": {
            "attribution": "Attribute to the external author writing about the subject",
            "voice_eligible": False,
            "presentation_guidance": (
                "Present as a critic's or scholar's observation about the author"
            ),
        },
        "contextual": {
            "attribution": "Attribute to the original author; note the subject author engages with this",
            "voice_eligible": False,
            "presentation_guidance": (
                "Present as a source the author reads, references, or responds to"
            ),
        },
        "tertiary": {
            "attribution": "Cite as reference material",
            "voice_eligible": False,
            "presentation_guidance": "Present as bibliographic or reference information",
        },
        "personal": {
            "attribution": "Present as the user's own reflection",
            "voice_eligible": False,
            "presentation_guidance": (
                "Present as the user's personal notes or reflections — "
                "never attribute to the subject author"
            ),
        },
    }
    return rules.get(source_class, rules["secondary"])


# ---------------------------------------------------------------------------
# C2: get_passage_links
# ---------------------------------------------------------------------------


async def handle_get_passage_links(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
    cache_manager: CacheManager | None = None,
) -> str:
    """Handle the get_passage_links MCP tool call.

    Direct Neo4j traversal for passage links from a given chunk.

    Arguments:
        chunk_id (str): The chunk to get links for.
        link_types (list[str], optional): Filter by link type.
        depth (int, optional): How many hops to follow (default 1, max 3).

    Returns:
        JSON with source_chunk and links array.
    """
    chunk_id = arguments.get("chunk_id")
    if not chunk_id:
        raise RetrievalError("chunk_id is required", context={"arguments": arguments})

    link_types = arguments.get("link_types")
    depth = min(arguments.get("depth", 1), 3)

    graph_service = GraphQueryService(storage.neo4j, cache=cache_manager)

    # Get the source chunk info
    source_chunk_data = await _get_chunk_info(storage, chunk_id)
    if source_chunk_data is None:
        return json.dumps({
            "error": f"Chunk not found: {chunk_id}",
        }, indent=2)

    # Get engagement chain from Neo4j
    chain = await graph_service.get_engagement_chain(
        chunk_id,
        max_depth=depth * 5,
    )

    links: list[dict[str, Any]] = []
    if chain and chain.links:
        for link in chain.links:
            # Filter by link type if specified
            if link_types and link.link_type not in link_types:
                continue

            target_info = {
                "chunk_id": link.target_chunk.chunk_id,
                "text": link.target_chunk.text_preview,
                "metadata": {
                    "work_id": link.target_chunk.work_id,
                },
            }

            relationship = {
                "type": link.link_type,
                "direction": f"{source_chunk_data.get('work_id', '')} → {link.target_chunk.work_id}",
                "confidence": link.confidence,
                "engagement_type": getattr(link, "detection_method", link.link_type),
                "annotation": getattr(link, "evidence", ""),
            }

            # Count further links from target
            further_chain = await graph_service.get_engagement_chain(
                link.target_chunk.chunk_id, max_depth=1,
            )
            further_count = len(further_chain.links) if further_chain else 0

            links.append({
                "target_chunk": target_info,
                "relationship": relationship,
                "further_links": further_count,
            })

    # Multi-hop: follow links if depth > 1
    if depth > 1 and links:
        visited = {chunk_id}
        for current_depth in range(2, depth + 1):
            next_targets = [
                link["target_chunk"]["chunk_id"]
                for link in links
                if link["target_chunk"]["chunk_id"] not in visited
            ]
            for target_id in next_targets:
                visited.add(target_id)
                deeper_chain = await graph_service.get_engagement_chain(
                    target_id, max_depth=5,
                )
                if deeper_chain and deeper_chain.links:
                    for dlink in deeper_chain.links:
                        if dlink.target_chunk.chunk_id in visited:
                            continue
                        if link_types and dlink.link_type not in link_types:
                            continue

                        links.append({
                            "target_chunk": {
                                "chunk_id": dlink.target_chunk.chunk_id,
                                "text": dlink.target_chunk.text_preview,
                                "metadata": {
                                    "work_id": dlink.target_chunk.work_id,
                                },
                            },
                            "relationship": {
                                "type": dlink.link_type,
                                "direction": f"{target_id} → {dlink.target_chunk.work_id}",
                                "confidence": dlink.confidence,
                                "engagement_type": getattr(dlink, "detection_method", dlink.link_type),
                                "annotation": getattr(dlink, "evidence", ""),
                            },
                            "further_links": 0,
                            "hop": current_depth,
                        })

    result = {
        "source_chunk": {
            "chunk_id": chunk_id,
            "text": source_chunk_data.get("text", ""),
            "metadata": {
                "work_id": source_chunk_data.get("work_id", ""),
                "source_class": source_chunk_data.get("source_class", ""),
                "granularity": source_chunk_data.get("granularity", ""),
            },
        },
        "links": links,
    }

    return json.dumps(result, indent=2)


async def _get_chunk_info(
    storage: StorageManager,
    chunk_id: str,
) -> dict[str, Any] | None:
    """Fetch chunk info from PG by chunk_id."""
    row = await storage.pg.fetch_one(
        """
        SELECT id, work_id, text, source_class, granularity, chapter, section
        FROM chunks
        WHERE id::text = $1
        LIMIT 1
        """,
        chunk_id,
    )
    if row is None:
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# C3: manage_vocabulary
# ---------------------------------------------------------------------------


async def handle_manage_vocabulary(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
    cache_manager: CacheManager | None = None,
) -> str:
    """Handle the manage_vocabulary MCP tool call.

    Manages canonical vocabulary terms for the library's thematic tagging.

    Arguments:
        action (str): list | propose | promote | merge | deprecate.
        term (str, optional): The vocabulary term to act on.
        merge_into (str, optional): Target term for merge action.
        note (str, optional): Note/reason for the action.

    Returns:
        JSON with vocabulary state or confirmation + affected_chunks count.
    """
    action = arguments.get("action")
    if not action:
        raise RetrievalError(
            "action is required (list, propose, promote, merge, deprecate)",
            context={"arguments": arguments},
        )

    valid_actions = {"list", "propose", "promote", "merge", "deprecate"}
    if action not in valid_actions:
        raise RetrievalError(
            f"Invalid action: {action}. Must be one of: {', '.join(sorted(valid_actions))}",
            context={"arguments": arguments},
        )

    term = arguments.get("term")
    note = arguments.get("note")

    # Validate term is provided for non-list actions
    if action != "list" and not term:
        raise RetrievalError(
            f"term is required for action '{action}'",
            context={"arguments": arguments},
        )

    # Validate merge_into for merge action
    if action == "merge":
        merge_into = arguments.get("merge_into")
        if not merge_into:
            raise RetrievalError(
                "merge_into is required for merge action",
                context={"arguments": arguments},
            )

    from author_library.vocabulary import VocabularyManager

    vocab = VocabularyManager(storage.pg)

    if action == "list":
        terms = await vocab.list_terms()
        return json.dumps({
            "action": "list",
            "terms": terms,
            "total": len(terms),
        }, indent=2)

    if action == "propose":
        term_record = await vocab.propose(term, note=note)
        return json.dumps({
            "action": "propose",
            "term": term_record,
            "message": f"Term '{term}' proposed for vocabulary.",
        }, indent=2)

    elif action == "promote":
        affected = await vocab.promote(term)
        return json.dumps({
            "action": "promote",
            "term": term,
            "affected_chunks": affected,
            "message": f"Term '{term}' promoted to canonical.",
        }, indent=2)

    elif action == "merge":
        merge_into = arguments.get("merge_into", "")
        affected = await vocab.merge(term, merge_into, note=note)
        return json.dumps({
            "action": "merge",
            "source_term": term,
            "target_term": merge_into,
            "affected_chunks": affected,
            "message": f"Term '{term}' merged into '{merge_into}'.",
        }, indent=2)

    else:  # deprecate
        affected = await vocab.deprecate(term, note=note)
        return json.dumps({
            "action": "deprecate",
            "term": term,
            "affected_chunks": affected,
            "message": f"Term '{term}' deprecated.",
        }, indent=2)
