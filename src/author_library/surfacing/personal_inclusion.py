"""M4: Personal content inclusion — mix user reflections with author passages.

Ensures the user's own Personal reflections are surfaced alongside primary
and secondary author content. Implements blending logic that:
  - Guarantees a minimum number of personal reflections in results
  - Boosts personal content relevance when it directly relates to context
  - Respects the user's choice to include or exclude personal content
  - Never attributes personal data to any speaker (Inviolable Rule #2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from author_library.surfacing.related_content import (
    ConnectionType,
    RelatedContentFinder,
    RelatedContentResult,
    RelatedItem,
)

if TYPE_CHECKING:
    from author_library.cache import CacheManager
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)

# Default minimum personal results when personal content is included
DEFAULT_MIN_PERSONAL = 2


@dataclass(frozen=True, slots=True)
class BlendedSurfacingResult:
    """Surfacing result with guaranteed personal content blending."""

    context_chunk_id: str
    context_work_id: str
    items: list[RelatedItem]
    strategies_used: list[str]
    personal_count: int
    author_count: int


class PersonalContentBlender:
    """Blends personal reflections with author passages in surfacing results.

    Wraps RelatedContentFinder to ensure personal content is not drowned out
    by high-scoring author passages. When the user has reflections relevant
    to the context, those reflections appear alongside the primary content.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        storage: StorageManager,
        embedding_provider: EmbeddingProvider,
        cache_manager: CacheManager | None = None,
    ) -> None:
        self._finder = RelatedContentFinder(
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
            cache_manager=cache_manager,
        )
        self._storage = storage

    async def find_blended(
        self,
        *,
        chunk_id: str | None = None,
        work_id: str | None = None,
        text_context: str | None = None,
        themes: list[str] | None = None,
        include_personal: bool = True,
        max_results: int = 20,
        min_personal: int = DEFAULT_MIN_PERSONAL,
    ) -> BlendedSurfacingResult:
        """Find related content with guaranteed personal content blending.

        When include_personal is True, guarantees at least min_personal
        personal items in the result (if that many exist). Personal items
        that would otherwise be pushed out by higher-scoring author content
        are preserved by reserving slots.

        Args:
            chunk_id: Context chunk to find relations for.
            work_id: Context work ID.
            text_context: Free text context for search.
            themes: Theme names to focus search.
            include_personal: Whether to include personal content.
            max_results: Maximum total results.
            min_personal: Minimum personal items to guarantee.

        Returns:
            BlendedSurfacingResult with balanced personal/author content.
        """
        result = await self._finder.find_related(
            chunk_id=chunk_id,
            work_id=work_id,
            text_context=text_context,
            themes=themes,
            include_personal=include_personal,
            max_results=max_results * 2,  # Fetch extra to have room for blending
        )

        if not include_personal:
            # Filter out any personal items when explicitly excluded
            author_items = [i for i in result.items if i.source_class != "personal"]
            return BlendedSurfacingResult(
                context_chunk_id=result.context_chunk_id,
                context_work_id=result.context_work_id,
                items=author_items[:max_results],
                strategies_used=result.strategies_used,
                personal_count=0,
                author_count=len(author_items[:max_results]),
            )

        blended = blend_results(
            result.items,
            max_results=max_results,
            min_personal=min_personal,
        )

        personal_count = sum(1 for i in blended if i.source_class == "personal")
        author_count = len(blended) - personal_count

        return BlendedSurfacingResult(
            context_chunk_id=result.context_chunk_id,
            context_work_id=result.context_work_id,
            items=blended,
            strategies_used=result.strategies_used,
            personal_count=personal_count,
            author_count=author_count,
        )


def blend_results(
    items: list[RelatedItem],
    *,
    max_results: int = 20,
    min_personal: int = DEFAULT_MIN_PERSONAL,
) -> list[RelatedItem]:
    """Blend personal and author items ensuring minimum personal representation.

    Algorithm:
    1. Separate personal and author items (both already sorted by score)
    2. Reserve min_personal slots for personal items
    3. Fill remaining slots with highest-scoring items from either pool
    4. Re-sort the final list by relevance score

    Args:
        items: All candidate items, sorted by relevance descending.
        max_results: Maximum total items to return.
        min_personal: Minimum personal items to include.

    Returns:
        Blended list sorted by relevance score descending.
    """
    personal = [i for i in items if i.source_class == "personal"]
    author = [i for i in items if i.source_class != "personal"]

    # If no personal items, just return author items
    if not personal:
        return author[:max_results]

    # Reserve slots for personal items
    reserved_personal = personal[:min_personal]
    remaining_personal = personal[min_personal:]

    # Available slots for open competition
    open_slots = max_results - len(reserved_personal)
    if open_slots <= 0:
        return reserved_personal[:max_results]

    # Merge remaining personal and author items, take top by score
    open_pool = sorted(
        remaining_personal + author,
        key=lambda i: i.relevance_score,
        reverse=True,
    )
    open_items = open_pool[:open_slots]

    # Combine and sort by score
    result = reserved_personal + open_items
    result.sort(key=lambda i: i.relevance_score, reverse=True)

    return result


def boost_personal_for_context(
    items: list[RelatedItem],
    context_work_id: str,
) -> list[RelatedItem]:
    """Boost personal items that directly reference the context work.

    Personal reflections that are about the same work the user is currently
    viewing get a relevance boost since they are directly relevant.

    Args:
        items: Items to consider for boosting.
        context_work_id: The work the user is currently viewing.

    Returns:
        New list with boosted personal items (re-sorted by score).
    """
    if not context_work_id:
        return items

    boosted: list[RelatedItem] = []
    for item in items:
        if (
            item.source_class == "personal"
            and item.metadata.get("source_note", "")
            and context_work_id in item.metadata.get("source_note", "")
        ):
            # Create new item with boosted score
            boosted.append(RelatedItem(
                chunk_id=item.chunk_id,
                work_id=item.work_id,
                text=item.text,
                source_class=item.source_class,
                granularity=item.granularity,
                connection_type=item.connection_type,
                relevance_score=min(item.relevance_score * 1.3, 1.0),
                metadata={**item.metadata, "boosted": True},
            ))
        else:
            boosted.append(item)

    boosted.sort(key=lambda i: i.relevance_score, reverse=True)
    return boosted
