"""M5: MCP tool handler for passive surfacing.

Exposes surfacing via the `surface_related` MCP tool, which:
  - Finds forgotten connections for a given chunk, work, or text context
  - Blends personal reflections with author content
  - Returns results grouped by confidence level with presentation labels
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from author_library.surfacing.personal_inclusion import PersonalContentBlender
from author_library.surfacing.related_content import RelatedContentFinder
from author_library.surfacing.response_format import format_surfacing_results

if TYPE_CHECKING:
    from author_library.cache import CacheManager
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


async def handle_surface_related(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
    cache_manager: CacheManager | None = None,
) -> str:
    """Handle the surface_related MCP tool call.

    Args:
        arguments: Tool input parameters.
        settings: Application settings.
        storage: Storage manager.
        embedding_provider: Embedding provider.
        cache_manager: Optional cache manager.

    Returns:
        JSON string with surfacing results.
    """
    chunk_id = arguments.get("chunk_id")
    work_id = arguments.get("work_id")
    text_context = arguments.get("text_context")
    themes = arguments.get("themes")
    include_personal = arguments.get("include_personal", True)
    max_results = arguments.get("max_results", 20)
    max_per_level = arguments.get("max_per_level")

    if not any([chunk_id, work_id, text_context]):
        return json.dumps({
            "error": "At least one of chunk_id, work_id, or text_context is required.",
        })

    blender = PersonalContentBlender(
        settings=settings,
        storage=storage,
        embedding_provider=embedding_provider,
        cache_manager=cache_manager,
    )

    blended = await blender.find_blended(
        chunk_id=chunk_id,
        work_id=work_id,
        text_context=text_context,
        themes=themes,
        include_personal=include_personal,
        max_results=max_results,
    )

    # Format into presentation response
    response = format_surfacing_results(
        blended.items,
        context_chunk_id=blended.context_chunk_id,
        context_work_id=blended.context_work_id,
        strategies_used=blended.strategies_used,
        max_per_level=max_per_level,
    )

    result = response.to_dict()
    result["personal_count"] = blended.personal_count
    result["author_count"] = blended.author_count

    return json.dumps(result, indent=2)
