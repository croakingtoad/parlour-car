"""HTTP surfacing endpoint for the Parlour Sidebar Obsidian plugin.

Provides POST /api/v1/surfacing with Bearer token authentication.
Wraps the existing surfacing logic (PersonalContentBlender +
format_surfacing_results) to return results in the flat format
expected by the sidebar plugin.

Also provides GET /api/v1/works — lists all works in the library.

Sidebar sends:
    { parlour_car_id?: str, note_type?: str, tags?: list[str] }

Sidebar expects:
    {
      results: [{ title, source, excerpt, confidence, vault_path,
                   parlour_car_id, score }],
      query_context: { parlour_car_id, note_type },
      synthesis_suggestions?: [...]
    }
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

import structlog
from starlette.requests import Request
from starlette.responses import JSONResponse

from author_library.surfacing.personal_inclusion import PersonalContentBlender
from author_library.surfacing.response_format import format_surfacing_results

if TYPE_CHECKING:
    from author_library.cache import CacheManager
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)

# Vault root used by Parlour Notes (matches vault.py VAULT_DIRS layout)
_VAULT_ROOT = "Parlour Notes"

# Map the works.media column value to the vault subfolder under sources/
_MEDIA_TO_SUBFOLDER: dict[str, str] = {
    "book": "sources/books",
    "video": "sources/videos",
    "audio": "sources/podcasts",
    "podcast": "sources/podcasts",
    "article": "sources/articles",
    "sermon": "sources/sermons",
}


def _vault_path_for_work(work_id: str, media: str | None) -> str | None:
    """Derive a vault-relative note path from a work_id and media type.

    The vault filename is the work_id itself with an .md extension,
    matching how parlour-notes writes notes via ``write_note(dir, work_id, ...)``.

    Args:
        work_id: Parlour Car work identifier (e.g. "guite-malcolm--faith-hope-and-poetry").
        media: The works.media column value ("book", "video", etc.), or None.

    Returns:
        Vault-relative path like "Parlour Notes/sources/books/guite-malcolm--faith-hope-and-poetry.md",
        or None if the media type is unknown and no default applies.
    """
    subfolder = _MEDIA_TO_SUBFOLDER.get(media or "", "sources/books")
    return f"{_VAULT_ROOT}/{subfolder}/{work_id}.md"


def _authenticate(request: Request, expected_key: str) -> bool:
    """Validate Bearer token from Authorization header."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return False
    token = auth_header[7:]
    return secrets.compare_digest(token, expected_key)


async def handle_surfacing(request: Request) -> JSONResponse:
    """Handle POST /api/v1/surfacing from the Obsidian sidebar plugin.

    Authenticates via Bearer token, extracts note context from the
    request body, runs the surfacing pipeline, and returns results
    in the flat format expected by the sidebar.
    """
    state: dict[str, Any] = request.app.state.surfacing_state

    # Authenticate
    api_key: str = state.get("api_key", "")
    if not api_key:
        log.error("surfacing_endpoint_no_api_key_configured")
        return JSONResponse(
            {"error": "Server misconfigured: no API key set"},
            status_code=500,
        )

    if not _authenticate(request, api_key):
        log.warning(
            "surfacing_endpoint_auth_failed",
            remote=request.client.host if request.client else "unknown",
        )
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # Parse body
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    parlour_car_id: str = body.get("parlour_car_id") or ""
    note_type: str = body.get("note_type") or ""
    tags: list[str] = body.get("tags") or []
    request_text_context: str = body.get("text_context") or ""

    if not parlour_car_id and not tags and not request_text_context:
        return JSONResponse(
            {"error": "At least one of parlour_car_id, tags, or text_context is required."},
            status_code=422,
        )

    # Extract themes from tags (strip #parlour/ prefix)
    themes: list[str] = []
    for tag in tags:
        cleaned = tag.lstrip("#")
        if cleaned.startswith("parlour/"):
            themes.append(cleaned.removeprefix("parlour/"))
        else:
            themes.append(cleaned)

    # Run surfacing pipeline
    settings: Settings = state["settings"]
    storage: StorageManager = state["storage"]
    embedding_provider: EmbeddingProvider = state["embedding_provider"]
    cache_manager: CacheManager | None = state.get("cache_manager")

    try:
        # Resolve context for vector search. The RelatedContentFinder needs
        # text_context or chunk_id to run search strategies.
        # Priority: explicit text_context from request > work_id lookup
        text_context: str | None = request_text_context or None
        if parlour_car_id:
            work_info = await storage.works.get(parlour_car_id)
            if work_info:
                title = work_info.get("title", "")
                author = work_info.get("author", "")
                # Only use work title as context if no explicit text_context provided
                if not text_context:
                    text_context = f"{title} by {author}" if author else title
                log.debug(
                    "surfacing_endpoint_resolved_work",
                    work_id=parlour_car_id,
                    text_context=text_context,
                )

                # Also pull themes from the graph for this work
                if not themes:
                    try:
                        graph_themes = await storage.neo4j.execute_read(
                            """MATCH (c:Chunk {work_id: $work_id})
                               -[:EXPLORES_THEME]->(t:Theme)
                               RETURN DISTINCT t.canonical_name AS cn
                               LIMIT 10""",
                            {"work_id": parlour_car_id},
                        )
                        themes = [r["cn"] for r in graph_themes if r.get("cn")]
                    except Exception:
                        log.debug("surfacing_endpoint_theme_lookup_failed")

        # For content-based requests (no parlour_car_id), match the note's
        # text against themes in the graph so we get cross-work results.
        if text_context and not themes and not parlour_car_id:
            try:
                # Find themes whose chunks are most similar to the note content
                from author_library.retrieval.vector_search import vector_search

                theme_hits = await vector_search(
                    text_context[:1000],
                    embedding_provider=embedding_provider,
                    embedding_repo=storage.embeddings,
                    limit=10,
                )
                # Collect unique work_ids from top vector hits
                hit_work_ids = list({r.work_id for r in theme_hits})

                # Pull themes from those works — gives us cross-work theme coverage
                if hit_work_ids:
                    graph_themes = await storage.neo4j.execute_read(
                        """MATCH (c:Chunk)-[:EXPLORES_THEME]->(t:Theme)
                            WHERE c.work_id IN $work_ids
                            RETURN DISTINCT t.canonical_name AS cn
                            LIMIT 15""",
                        {"work_ids": hit_work_ids},
                    )
                    themes = [r["cn"] for r in graph_themes if r.get("cn")]
                    log.debug(
                        "surfacing_endpoint_content_themes_resolved",
                        theme_count=len(themes),
                        works_matched=len(hit_work_ids),
                    )
            except Exception as exc:
                log.debug("surfacing_endpoint_content_theme_extraction_failed", error=str(exc))

        blender = PersonalContentBlender(
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
            cache_manager=cache_manager,
        )

        blended = await blender.find_blended(
            work_id=parlour_car_id or None,
            text_context=text_context,
            themes=themes or None,
            include_personal=True,
            max_results=20,
        )

        response = format_surfacing_results(
            blended.items,
            context_chunk_id=blended.context_chunk_id,
            context_work_id=blended.context_work_id,
            strategies_used=blended.strategies_used,
        )

    except Exception as exc:
        log.error("surfacing_endpoint_pipeline_error", error=str(exc))
        return JSONResponse(
            {"error": f"Surfacing pipeline error: {exc}"},
            status_code=500,
        )

    # Flatten grouped results into the flat array the sidebar expects
    flat_results: list[dict[str, Any]] = []

    # Collect all unique work_ids from results to batch-lookup media types
    all_items = response.high_confidence + response.medium_confidence + response.low_confidence
    unique_work_ids = {item.work_id for item in all_items if item.work_id}
    work_media_map: dict[str, str | None] = {}
    for wid in unique_work_ids:
        try:
            work_row = await storage.works.get(wid)
            work_media_map[wid] = work_row.get("media") if work_row else None
        except Exception:
            work_media_map[wid] = None

    for level, items in [
        ("high", response.high_confidence),
        ("medium", response.medium_confidence),
        ("low", response.low_confidence),
    ]:
        for item in items:
            vault_path = _vault_path_for_work(item.work_id, work_media_map.get(item.work_id))
            flat_results.append({
                "title": item.title,
                "source": item.source,
                "excerpt": item.excerpt,
                "confidence": level,
                "vault_path": vault_path,
                "parlour_car_id": item.work_id,
                "score": item.metadata.get("relevance_score", 0.0),
            })

    log.info(
        "surfacing_endpoint_complete",
        parlour_car_id=parlour_car_id,
        note_type=note_type,
        result_count=len(flat_results),
    )

    return JSONResponse({
        "results": flat_results,
        "query_context": {
            "parlour_car_id": parlour_car_id,
            "note_type": note_type,
        },
    })


async def handle_works(request: Request) -> JSONResponse:
    """Handle GET /api/v1/works — return all works in the library.

    Authenticates via the same Bearer token as the surfacing endpoint.
    Returns a JSON object with a ``works`` array, each entry containing
    work_id, title, author, and media.
    """
    state: dict[str, Any] = request.app.state.surfacing_state

    # Authenticate
    api_key: str = state.get("api_key", "")
    if not api_key:
        log.error("works_endpoint_no_api_key_configured")
        return JSONResponse(
            {"error": "Server misconfigured: no API key set"},
            status_code=500,
        )

    if not _authenticate(request, api_key):
        log.warning(
            "works_endpoint_auth_failed",
            remote=request.client.host if request.client else "unknown",
        )
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    storage: StorageManager = state["storage"]

    try:
        rows = await storage.pg.fetch_all(
            "SELECT work_id, title, author, media FROM works ORDER BY title"
        )
        works = [
            {
                "work_id": row["work_id"],
                "title": row["title"],
                "author": row["author"],
                "media": row["media"],
            }
            for row in rows
        ]
    except Exception as exc:
        log.error("works_endpoint_query_error", error=str(exc))
        return JSONResponse(
            {"error": f"Failed to list works: {exc}"},
            status_code=500,
        )

    log.info("works_endpoint_complete", work_count=len(works))
    return JSONResponse({"works": works})
