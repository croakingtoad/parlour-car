"""M1: Related content query — find forgotten connections across the knowledge base.

Given a chunk or note context, searches for related content the user may have
forgotten. Combines multiple search strategies:
  - Thematic parallels (chunks exploring the same themes)
  - Passage links (explicit/implicit engagement chains)
  - Personal reflections (user's own reflections on related content)
  - Temporal proximity (content captured around the same time)
  - Vector similarity (semantically related content)

Results are ranked by a composite score combining strategy-specific signals.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from author_library.cache import CacheManager
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


class ConnectionType(StrEnum):
    """Types of connections between content items."""

    PASSAGE_LINK = "passage_link"
    THEMATIC_PARALLEL = "thematic_parallel"
    PERSONAL_REFLECTION = "personal_reflection"
    VECTOR_SIMILARITY = "vector_similarity"
    TEMPORAL_PROXIMITY = "temporal_proximity"
    ENTITY_OVERLAP = "entity_overlap"


@dataclass(frozen=True, slots=True)
class RelatedItem:
    """A single related content item with connection metadata."""

    chunk_id: str
    work_id: str
    text: str
    source_class: str
    granularity: str
    connection_type: ConnectionType
    relevance_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def work_title(self) -> str:
        return self.metadata.get("work_title", "")

    @property
    def author(self) -> str:
        return self.metadata.get("author", "")


@dataclass(slots=True)
class RelatedContentResult:
    """Aggregated result from related content query."""

    context_chunk_id: str
    context_work_id: str
    items: list[RelatedItem]
    strategies_used: list[str]


class RelatedContentFinder:
    """Finds forgotten connections across the knowledge base.

    Uses existing search_chunks and get_passage_links as building blocks,
    augmented with graph-based thematic queries and personal reflection search.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        storage: StorageManager,
        embedding_provider: EmbeddingProvider,
        cache_manager: CacheManager | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._embedding_provider = embedding_provider
        self._cache = cache_manager

    async def find_related(
        self,
        *,
        chunk_id: str | None = None,
        work_id: str | None = None,
        text_context: str | None = None,
        themes: list[str] | None = None,
        include_personal: bool = True,
        max_results: int = 20,
    ) -> RelatedContentResult:
        """Find related content given a chunk, work, or text context.

        At least one of chunk_id, work_id, or text_context must be provided.

        Args:
            chunk_id: ID of the context chunk to find relations for.
            work_id: ID of the work to find relations for.
            text_context: Free text to use as search context.
            themes: Optional theme names to focus the search.
            include_personal: Whether to include Personal reflections.
            max_results: Maximum total results to return.

        Returns:
            RelatedContentResult with ranked items.
        """
        if not any([chunk_id, work_id, text_context]):
            return RelatedContentResult(
                context_chunk_id=chunk_id or "",
                context_work_id=work_id or "",
                items=[],
                strategies_used=[],
            )

        # Resolve chunk data if we have a chunk_id
        chunk_data: dict[str, Any] | None = None
        if chunk_id:
            chunk_data = await self._get_chunk(chunk_id)
            if chunk_data and not work_id:
                work_id = chunk_data.get("work_id", "")
            if chunk_data and not text_context:
                text_context = chunk_data.get("text", "")

        # Resolve themes from graph if not provided
        if not themes and chunk_id:
            themes = await self._get_chunk_themes(chunk_id)

        # Run search strategies in parallel
        strategies_used: list[str] = []
        tasks: list[asyncio.Task[list[RelatedItem]]] = []

        # Strategy 1: Passage links (if chunk_id available)
        if chunk_id:
            tasks.append(asyncio.create_task(
                self._find_via_passage_links(chunk_id)
            ))
            strategies_used.append("passage_links")

        # Strategy 2: Thematic parallels (if themes available)
        if themes:
            tasks.append(asyncio.create_task(
                self._find_via_themes(
                    themes,
                    exclude_work_id=work_id,
                    exclude_chunk_id=chunk_id,
                )
            ))
            strategies_used.append("thematic_parallels")

        # Strategy 3: Personal reflections (if enabled)
        if include_personal and chunk_id:
            tasks.append(asyncio.create_task(
                self._find_personal_reflections(
                    chunk_id=chunk_id,
                    themes=themes,
                )
            ))
            strategies_used.append("personal_reflections")

        # Strategy 4: Vector similarity (if text context available)
        if text_context:
            tasks.append(asyncio.create_task(
                self._find_via_vector_similarity(
                    text_context,
                    exclude_chunk_id=chunk_id,
                    include_personal=include_personal,
                )
            ))
            strategies_used.append("vector_similarity")

        # Gather all results
        all_results: list[RelatedItem] = []
        for completed in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(completed, BaseException):
                log.warning("surfacing_strategy_failed", error=str(completed))
                continue
            all_results.extend(completed)

        # Deduplicate by chunk_id, keeping highest-scoring entry
        deduped = self._deduplicate(all_results)

        # Sort by relevance score descending
        deduped.sort(key=lambda r: r.relevance_score, reverse=True)

        return RelatedContentResult(
            context_chunk_id=chunk_id or "",
            context_work_id=work_id or "",
            items=deduped[:max_results],
            strategies_used=strategies_used,
        )

    async def _get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Fetch chunk data from PostgreSQL.

        Normalizes chunk_id to UUID format (with dashes) since Neo4j stores
        hex strings without dashes while PostgreSQL uses standard UUIDs.
        """
        # Normalize: insert dashes if we got a 32-char hex string from Neo4j
        normalized = chunk_id
        if len(chunk_id) == 32 and "-" not in chunk_id:
            normalized = (
                f"{chunk_id[:8]}-{chunk_id[8:12]}-{chunk_id[12:16]}"
                f"-{chunk_id[16:20]}-{chunk_id[20:]}"
            )

        row = await self._storage.pg.fetch_one(
            """SELECT id, work_id, text, source_class, granularity,
                      chapter, section, metadata, pass_number
            FROM chunks WHERE id::text = $1 LIMIT 1""",
            normalized,
        )
        return dict(row) if row else None

    async def _get_chunk_themes(self, chunk_id: str) -> list[str]:
        """Get theme names for a chunk from the graph."""
        try:
            results = await self._storage.graph.get_themes_for_chunk(chunk_id)
            return [r["canonical_name"] for r in results if r.get("canonical_name")]
        except Exception:
            log.debug("get_chunk_themes_failed", chunk_id=chunk_id)
            return []

    async def _find_via_passage_links(
        self, chunk_id: str
    ) -> list[RelatedItem]:
        """Find related content via passage link chains."""
        from author_library.graph.queries import GraphQueryService

        graph_service = GraphQueryService(self._storage.neo4j, cache=self._cache)
        chain = await graph_service.get_engagement_chain(chunk_id, max_depth=3)
        if not chain or not chain.links:
            return []

        items: list[RelatedItem] = []
        for link in chain.links:
            # Convert link confidence to a numeric score
            confidence_scores = {"high": 0.95, "medium": 0.7, "low": 0.4}
            score = confidence_scores.get(link.confidence, 0.5)

            work_info = await self._storage.works.get(link.target_chunk.work_id)

            # Fetch full text from PG (Neo4j text_preview is truncated)
            full_chunk = await self._get_chunk(link.target_chunk.chunk_id)
            full_text = (
                full_chunk.get("text", link.target_chunk.text_preview)
                if full_chunk
                else link.target_chunk.text_preview
            )

            items.append(RelatedItem(
                chunk_id=link.target_chunk.chunk_id,
                work_id=link.target_chunk.work_id,
                text=full_text,
                source_class=link.target_chunk.source_class,
                granularity=link.target_chunk.granularity,
                connection_type=ConnectionType.PASSAGE_LINK,
                relevance_score=score,
                metadata={
                    "link_type": link.link_type,
                    "confidence": link.confidence,
                    "evidence": link.evidence,
                    "work_title": work_info.get("title", "") if work_info else "",
                    "author": work_info.get("author", "") if work_info else "",
                },
            ))

        return items

    async def _find_via_themes(
        self,
        themes: list[str],
        *,
        exclude_work_id: str | None = None,
        exclude_chunk_id: str | None = None,
        max_chunks_per_theme: int = 20,
    ) -> list[RelatedItem]:
        """Find content exploring the same themes."""
        from author_library.graph.queries import GraphQueryService

        graph_service = GraphQueryService(self._storage.neo4j, cache=self._cache)
        items: list[RelatedItem] = []

        for theme in themes[:5]:  # Cap at 5 themes to avoid excessive queries
            subgraph = await graph_service.get_theme_subgraph(theme)
            if not subgraph:
                continue

            chunks_processed = 0
            for chunk in subgraph.chunks:
                if chunks_processed >= max_chunks_per_theme:
                    break

                # Skip the context chunk and its siblings from same work
                if chunk.chunk_id == exclude_chunk_id:
                    continue
                if exclude_work_id and chunk.work_id == exclude_work_id:
                    continue

                work_info = next(
                    (w for w in subgraph.works if w.get("work_id") == chunk.work_id),
                    {},
                )

                items.append(RelatedItem(
                    chunk_id=chunk.chunk_id,
                    work_id=chunk.work_id,
                    text=chunk.text_preview,
                    source_class=chunk.source_class,
                    granularity=chunk.granularity,
                    connection_type=ConnectionType.THEMATIC_PARALLEL,
                    relevance_score=0.6,  # Base score for thematic parallels
                    metadata={
                        "theme": subgraph.theme_name,
                        "work_title": work_info.get("title", ""),
                        "author": work_info.get("author", ""),
                    },
                ))
                chunks_processed += 1

        return items

    async def _find_personal_reflections(
        self,
        *,
        chunk_id: str,
        themes: list[str] | None = None,
    ) -> list[RelatedItem]:
        """Find user's Personal reflections related to the context.

        Searches for USER_REFLECTS_ON edges from personal chunks to the
        context chunk, and also finds personal chunks on the same themes.
        """
        items: list[RelatedItem] = []

        # Direct reflections on this chunk
        reflections = await self._storage.graph.get_reflections_for_target(
            target_id=chunk_id,
            target_key="chunk_id",
            target_label="Chunk",
        )

        for ref in reflections:
            # Get full text from PG
            full_chunk = await self._get_chunk(ref["chunk_id"])
            text = full_chunk.get("text", ref.get("text_preview", "")) if full_chunk else ref.get("text_preview", "")

            items.append(RelatedItem(
                chunk_id=ref["chunk_id"],
                work_id=ref.get("work_id", ""),
                text=text,
                source_class="personal",
                granularity=ref.get("granularity", ""),
                connection_type=ConnectionType.PERSONAL_REFLECTION,
                relevance_score=0.85,  # High score for direct reflections
                metadata={
                    "date_created": ref.get("date_created", ""),
                    "target_type": ref.get("target_type", ""),
                },
            ))

        # Also search for personal chunks on the same themes
        if themes:
            from author_library.retrieval.vector_search import vector_search

            theme_query = " ".join(themes[:3])
            try:
                personal_results = await vector_search(
                    theme_query,
                    embedding_provider=self._embedding_provider,
                    embedding_repo=self._storage.embeddings,
                    limit=10,
                    source_class_filter="personal",
                )

                for r in personal_results:
                    cid = str(r.chunk_id)
                    # Skip if already found via direct reflection
                    if any(item.chunk_id == cid for item in items):
                        continue

                    items.append(RelatedItem(
                        chunk_id=cid,
                        work_id=r.work_id,
                        text=r.text,
                        source_class="personal",
                        granularity=r.granularity,
                        connection_type=ConnectionType.PERSONAL_REFLECTION,
                        relevance_score=r.score * 0.7,  # Scaled vector score
                        metadata={
                            "search_source": "theme_vector_search",
                        },
                    ))
            except Exception:
                log.debug("personal_theme_search_failed", themes=themes)

        return items

    async def _find_via_vector_similarity(
        self,
        text_context: str,
        *,
        exclude_chunk_id: str | None = None,
        include_personal: bool = True,
    ) -> list[RelatedItem]:
        """Find semantically similar content via vector search."""
        from author_library.retrieval.vector_search import vector_search

        try:
            results = await vector_search(
                text_context[:2000],  # Cap context length
                embedding_provider=self._embedding_provider,
                embedding_repo=self._storage.embeddings,
                limit=15,
            )
        except Exception:
            log.debug("vector_similarity_search_failed")
            return []

        items: list[RelatedItem] = []
        for r in results:
            cid = str(r.chunk_id)
            if cid == exclude_chunk_id:
                continue
            if not include_personal and r.source_class == "personal":
                continue

            work_info = await self._storage.works.get(r.work_id)

            items.append(RelatedItem(
                chunk_id=cid,
                work_id=r.work_id,
                text=r.text,
                source_class=r.source_class,
                granularity=r.granularity,
                connection_type=ConnectionType.VECTOR_SIMILARITY,
                relevance_score=r.score * 0.65,  # Scaled for ranking alongside other strategies
                metadata={
                    "work_title": work_info.get("title", "") if work_info else "",
                    "author": work_info.get("author", "") if work_info else "",
                },
            ))

        return items

    @staticmethod
    def _deduplicate(items: list[RelatedItem]) -> list[RelatedItem]:
        """Deduplicate items by chunk_id, keeping the highest-scoring entry."""
        best: dict[str, RelatedItem] = {}
        for item in items:
            existing = best.get(item.chunk_id)
            if existing is None or item.relevance_score > existing.relevance_score:
                best[item.chunk_id] = item
        return list(best.values())
