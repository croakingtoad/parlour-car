"""MCP tool handlers for query operations (E011).

Provides:
  - ask_author: Conversational Q&A with voice-calibrated responses.
  - trace_theme: Chronological theme tracing across works.
  - find_quotes: Full-text + vector search for specific passages.
  - compare_ideas: Cross-author thematic comparison.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from author_library.errors import RetrievalError
from author_library.graph.queries import GraphQueryService
from author_library.intelligence.voice_crud import VoiceProfileManager
from author_library.retrieval.context_assembly import assemble_context
from author_library.retrieval.orchestrator import RetrievalOrchestrator
from author_library.retrieval.text_search import phrase_search
from author_library.retrieval.vector_search import vector_search

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# ask_author
# ---------------------------------------------------------------------------


async def handle_ask_author(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
) -> str:
    """Handle the ask_author MCP tool call.

    Arguments:
        question (str): The user's question about/to the author.
        author_id (str): The subject author's slug identifier.
        works_filter (list[str], optional): Limit retrieval to specific work IDs.
        response_style (str, optional): conversational|academic|devotional|lecture.

    Returns:
        JSON with response text and citations.
    """
    question = arguments.get("question")
    if not question:
        raise RetrievalError("question is required", context={"arguments": arguments})

    author_id = arguments.get("author_id")
    if not author_id:
        raise RetrievalError("author_id is required", context={"arguments": arguments})

    response_style = arguments.get("response_style", "conversational")

    # Build graph query service
    graph_service = GraphQueryService(storage.neo4j)

    # Step 1: Multi-pass retrieval
    orchestrator = RetrievalOrchestrator(
        settings=settings,
        embedding_provider=embedding_provider,
        embedding_repo=storage.embeddings,
        pg_pool=storage.pg,
        graph_query_service=graph_service,
    )

    orchestrated = await orchestrator.retrieve(
        question,
        author_id=author_id,
    )

    # Step 2: Get voice profile for calibration
    voice_manager = VoiceProfileManager(settings)
    voice_profile = await voice_manager.get_current(
        author_id=author_id,
        voice_repo=storage.voice_profiles,
    )

    # Step 3: Get thematic entries for context
    thematic_entries_raw = await storage.thematic.list_entries(author_id)
    from author_library.intelligence.thematic_index import ThematicEntry

    thematic_entries = [
        ThematicEntry(
            theme=e.get("theme", ""),
            author_stance=e.get("author_stance", ""),
            related_themes=e.get("related_themes", []),
        )
        for e in thematic_entries_raw
    ]

    # Step 4: Assemble context window
    context_window = assemble_context(
        orchestrated,
        voice_profile=voice_profile,
        author_name=author_id,
        thematic_entries=thematic_entries,
    )

    # Step 5: Generate response via Anthropic API
    api_key = settings.api_keys.anthropic_api_key.get_secret_value()
    if not api_key:
        raise RetrievalError(
            "Anthropic API key is required for ask_author",
            context={"author_id": author_id},
        )

    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Build the user message with retrieved passages
    passage_texts = [p.text for p in context_window.passages]
    passages_block = "\n\n---\n\n".join(passage_texts)

    style_instruction = _style_instruction(response_style)

    user_message = (
        f"{style_instruction}\n\n"
        f"Question: {question}\n\n"
        f"Retrieved passages:\n\n{passages_block}"
    )

    if context_window.thematic_summaries:
        user_message += "\n\nThematic context:\n" + "\n".join(context_window.thematic_summaries)

    response = await client.messages.create(
        model=settings.llm.query_model,
        max_tokens=4096,
        system=context_window.system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text = block.text
            break

    # Build citations from passages
    citations = []
    for passage in context_window.passages:
        citations.append({
            "work_id": passage.work_id,
            "source_class": passage.source_class,
            "citation": passage.citation_label,
            "relevance": round(passage.relevance_score, 3),
        })

    result = {
        "response": response_text,
        "question_type": orchestrated.question_type.value,
        "citations": citations,
        "passages_used": len(context_window.passages),
        "token_estimate": context_window.total_tokens_estimate,
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# trace_theme
# ---------------------------------------------------------------------------


async def handle_trace_theme(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
) -> str:
    """Handle the trace_theme MCP tool call.

    Arguments:
        theme_name (str): Name of the theme to trace.
        author_id (str, optional): Limit to a specific author.

    Returns:
        JSON with chronological theme narrative.
    """
    theme_name = arguments.get("theme_name")
    if not theme_name:
        raise RetrievalError("theme_name is required", context={"arguments": arguments})

    author_id = arguments.get("author_id")

    graph_service = GraphQueryService(storage.neo4j)

    # Get theme subgraph from Neo4j
    subgraph = await graph_service.get_theme_subgraph(theme_name)
    if subgraph is None:
        return json.dumps({
            "theme": theme_name,
            "found": False,
            "message": f"Theme '{theme_name}' not found in the knowledge graph.",
        })

    # Get argument evolution for this theme
    evolution = await graph_service.get_argument_evolution(theme_name)

    # Get engagement chains for theme-related chunks
    engagement_passages: list[dict[str, Any]] = []
    for chunk in subgraph.chunks[:10]:  # Limit to top 10
        chain = await graph_service.get_engagement_chain(chunk.chunk_id)
        if chain and chain.links:
            for link in chain.links:
                engagement_passages.append({
                    "source_chunk_id": chunk.chunk_id,
                    "source_work_id": chunk.work_id,
                    "target_chunk_id": link.target_chunk.chunk_id,
                    "target_work_id": link.target_chunk.work_id,
                    "target_text": link.target_chunk.text_preview,
                    "link_type": link.link_type,
                    "confidence": link.confidence,
                })

    # Get thematic entries from the repository for narrative
    theme_narrative = ""
    if author_id:
        entries = await storage.thematic.list_entries(author_id)
        for entry in entries:
            if entry.get("theme", "").lower() == theme_name.lower():
                theme_narrative = entry.get("author_stance", "")
                break

    # Build chronological view from works metadata
    chronology: list[dict[str, Any]] = []
    work_ids = {c.work_id for c in subgraph.chunks}
    for work_id in sorted(work_ids):
        work = await storage.works.get(work_id)
        if work:
            work_chunks = [c for c in subgraph.chunks if c.work_id == work_id]
            chronology.append({
                "work_id": work_id,
                "title": work.get("title", ""),
                "publication_year": work.get("publication_year"),
                "source_class": work.get("source_class", ""),
                "chunk_count": len(work_chunks),
                "sample_passages": [c.text_preview for c in work_chunks[:3]],
            })

    chronology.sort(key=lambda x: x.get("publication_year") or 0)

    result = {
        "theme": subgraph.theme_name,
        "found": True,
        "author_stance": theme_narrative,
        "chronology": chronology,
        "arguments": [
            {
                "claim": arg.claim,
                "source_chunks": len(arg.source_chunks),
            }
            for arg in evolution.arguments
        ],
        "development_links": [
            {"from": f, "to": t} for f, t in evolution.development_links
        ],
        "engagement_passages": engagement_passages,
        "total_chunks": len(subgraph.chunks),
        "total_works": len(subgraph.works),
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# find_quotes
# ---------------------------------------------------------------------------


async def handle_find_quotes(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
) -> str:
    """Handle the find_quotes MCP tool call.

    Arguments:
        query (str): Search query (exact phrase or semantic).
        author_id (str, optional): Limit to a specific author's works.
        source_class_filter (str, optional): Limit to a specific source class.
        limit (int, optional): Maximum results (default 10).

    Returns:
        JSON with matching quotes and citations.
    """
    query = arguments.get("query")
    if not query:
        raise RetrievalError("query is required", context={"arguments": arguments})

    source_class_filter = arguments.get("source_class_filter")
    limit = arguments.get("limit", 10)

    # Run both phrase search and vector search in parallel
    import asyncio

    phrase_task = asyncio.create_task(
        phrase_search(
            storage.pg,
            query,
            source_class_filter=source_class_filter,
            limit=limit,
        )
    )
    vector_task = asyncio.create_task(
        vector_search(
            query,
            embedding_provider=embedding_provider,
            embedding_repo=storage.embeddings,
            limit=limit,
            source_class_filter=source_class_filter,
            granularity_filter="micro",
        )
    )

    phrase_results, vector_results = await asyncio.gather(phrase_task, vector_task)

    # Merge and deduplicate results
    seen_chunk_ids: set[str] = set()
    quotes: list[dict[str, Any]] = []

    # Phrase matches first (exact match is highest priority)
    for r in phrase_results:
        cid = str(r.chunk_id)
        if cid not in seen_chunk_ids:
            seen_chunk_ids.add(cid)
            quotes.append(_result_to_quote(r, match_type="phrase"))

    # Vector matches second (semantic similarity)
    for r in vector_results:
        cid = str(r.chunk_id)
        if cid not in seen_chunk_ids:
            seen_chunk_ids.add(cid)
            quotes.append(_result_to_quote(r, match_type="semantic"))

    # Trim to limit
    quotes = quotes[:limit]

    # Enrich with cross-resource links (graceful degradation if Neo4j unavailable)
    try:
        graph_service = GraphQueryService(storage.neo4j)
        for quote in quotes:
            chunk_id = quote["chunk_id"]
            chain = await graph_service.get_engagement_chain(chunk_id)
            if chain and chain.links:
                quote["cross_references"] = [
                    {
                        "target_work_id": link.target_chunk.work_id,
                        "link_type": link.link_type,
                        "confidence": link.confidence,
                        "text_preview": link.target_chunk.text_preview,
                    }
                    for link in chain.links[:3]
                ]
    except Exception as exc:
        log.warning("find_quotes_graph_enrichment_failed", error=str(exc), degraded=True)

    result = {
        "query": query,
        "total_results": len(quotes),
        "quotes": quotes,
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# compare_ideas
# ---------------------------------------------------------------------------


async def handle_compare_ideas(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
) -> str:
    """Handle the compare_ideas MCP tool call.

    Arguments:
        topic (str): The topic/theme to compare across authors.
        author_ids (list[str]): List of author slug identifiers to compare.

    Returns:
        JSON with side-by-side thematic treatment summaries.
    """
    topic = arguments.get("topic")
    if not topic:
        raise RetrievalError("topic is required", context={"arguments": arguments})

    author_ids = arguments.get("author_ids")
    if not author_ids or len(author_ids) < 2:
        raise RetrievalError(
            "author_ids requires at least 2 author identifiers",
            context={"arguments": arguments},
        )

    graph_service = GraphQueryService(storage.neo4j)
    comparisons: list[dict[str, Any]] = []

    for author_id in author_ids:
        author_data: dict[str, Any] = {"author_id": author_id}

        # Get thematic entries for this author
        entries = await storage.thematic.list_entries(author_id)
        matching_entries = [
            e for e in entries if topic.lower() in e.get("theme", "").lower()
        ]

        if matching_entries:
            entry = matching_entries[0]
            author_data["theme_found"] = True
            author_data["theme_name"] = entry.get("theme", "")
            author_data["author_stance"] = entry.get("author_stance", "")
            author_data["related_themes"] = entry.get("related_themes", [])
        else:
            author_data["theme_found"] = False

        # Get author's network for context
        network = await graph_service.get_author_network(author_id)
        relevant_themes = [
            t for t in network.themes_explored
            if topic.lower() in (t.get("name", "") or "").lower()
        ]
        author_data["graph_theme_chunks"] = sum(
            t.get("chunk_count", 0) for t in relevant_themes
        )
        author_data["total_works"] = len(network.works)

        # Get a sample passage via vector search
        try:
            sample_results = await vector_search(
                topic,
                embedding_provider=embedding_provider,
                embedding_repo=storage.embeddings,
                limit=3,
                source_class_filter="primary",
            )
            # Filter to this author's works
            author_works = await storage.works.list_by_author(author_id)
            author_work_ids = {w["work_id"] for w in author_works}
            author_samples = [r for r in sample_results if r.work_id in author_work_ids]
            author_data["sample_passages"] = [
                {
                    "work_id": r.work_id,
                    "text": r.text[:300],
                    "score": round(r.score, 3),
                }
                for r in author_samples[:2]
            ]
        except Exception:
            author_data["sample_passages"] = []

        comparisons.append(author_data)

    result = {
        "topic": topic,
        "authors_compared": len(comparisons),
        "comparisons": comparisons,
    }

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _style_instruction(style: str) -> str:
    """Return a style instruction for the LLM response."""
    instructions = {
        "conversational": (
            "Respond in a warm, conversational tone as if the author were "
            "speaking directly to the questioner."
        ),
        "academic": (
            "Respond in a scholarly, analytical tone with precise references "
            "and nuanced argumentation."
        ),
        "devotional": (
            "Respond in a contemplative, devotional tone drawing out the "
            "spiritual and theological dimensions."
        ),
        "lecture": (
            "Respond in an engaging lecture style, building the argument "
            "step by step with illustrative examples."
        ),
    }
    return instructions.get(style, instructions["conversational"])


def _result_to_quote(
    result: Any,
    *,
    match_type: str,
) -> dict[str, Any]:
    """Convert a RetrievalResult to a quote dict."""
    return {
        "chunk_id": str(result.chunk_id),
        "work_id": result.work_id,
        "text": result.text,
        "source_class": result.source_class,
        "granularity": result.granularity,
        "score": round(result.score, 4),
        "match_type": match_type,
    }
