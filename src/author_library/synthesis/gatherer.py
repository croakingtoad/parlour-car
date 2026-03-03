"""O1: Personal reflection gatherer.

Queries all Personal source class chunks filtered by theme, speaker, and
date range. Aggregates reflections from "My Thoughts", "Session Reflections",
and "My Response" sections, returning them chronologically with source
attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from author_library.cache import CacheManager
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PersonalReflection:
    """A single personal reflection with source attribution."""

    chunk_id: str
    work_id: str
    text: str
    date_created: str
    granularity: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_note(self) -> str:
        """The note path or capture this reflection came from."""
        return self.metadata.get("source_note", "")

    @property
    def section_type(self) -> str:
        """Which section: 'my_thoughts', 'session_reflections', 'my_response', or 'freeform'."""
        return self.metadata.get("section_type", "freeform")

    @property
    def themes(self) -> list[str]:
        return self.metadata.get("themes", [])


@dataclass(slots=True)
class GatheredReflections:
    """Result of gathering personal reflections."""

    reflections: list[PersonalReflection]
    total_found: int
    filters_applied: dict[str, Any]
    date_range: tuple[str, str] | None = None

    @property
    def theme_counts(self) -> dict[str, int]:
        """Count reflections per theme."""
        counts: dict[str, int] = {}
        for r in self.reflections:
            for theme in r.themes:
                counts[theme] = counts.get(theme, 0) + 1
        return counts


class PersonalReflectionGatherer:
    """Gathers Personal source class chunks for synthesis.

    Queries PostgreSQL for chunks with source_class='personal',
    optionally filtered by theme (via graph), speaker (via work metadata),
    and date range. Combines results from different retrieval strategies
    to ensure comprehensive coverage.
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

    async def gather(
        self,
        *,
        theme: str | None = None,
        speaker: str | None = None,
        date_after: str | None = None,
        date_before: str | None = None,
        prompt: str | None = None,
        limit: int = 100,
    ) -> GatheredReflections:
        """Gather personal reflections matching the given criteria.

        At least one of theme, speaker, date range, or prompt should be
        provided to narrow results. Without filters, returns all personal
        reflections (up to limit).

        Args:
            theme: Focus on a specific theme (canonical name or display name).
            speaker: Focus on reflections about a specific speaker slug.
            date_after: ISO date — only reflections after this date.
            date_before: ISO date — only reflections before this date.
            prompt: User's framing question to guide semantic search.
            limit: Maximum reflections to return.

        Returns:
            GatheredReflections with chronologically ordered results.
        """
        filters_applied: dict[str, Any] = {}
        all_reflections: list[PersonalReflection] = []

        # Strategy 1: Direct SQL query for personal chunks
        sql_results = await self._query_personal_chunks(
            date_after=date_after,
            date_before=date_before,
            speaker=speaker,
            limit=limit,
        )
        all_reflections.extend(sql_results)
        if date_after:
            filters_applied["date_after"] = date_after
        if date_before:
            filters_applied["date_before"] = date_before
        if speaker:
            filters_applied["speaker"] = speaker

        # Strategy 2: Theme-based retrieval from graph
        if theme:
            filters_applied["theme"] = theme
            theme_results = await self._query_by_theme(theme, limit=limit)
            all_reflections.extend(theme_results)

        # Strategy 3: Semantic search using the prompt
        if prompt:
            filters_applied["prompt"] = prompt
            semantic_results = await self._query_by_semantic_similarity(
                prompt, limit=min(limit, 30),
            )
            all_reflections.extend(semantic_results)

        # Deduplicate by chunk_id
        seen: set[str] = set()
        unique: list[PersonalReflection] = []
        for r in all_reflections:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                unique.append(r)

        # Sort chronologically
        unique.sort(key=lambda r: r.date_created or "")

        # Compute date range
        date_range = None
        if unique:
            dates = [r.date_created for r in unique if r.date_created]
            if dates:
                date_range = (min(dates), max(dates))

        return GatheredReflections(
            reflections=unique[:limit],
            total_found=len(unique),
            filters_applied=filters_applied,
            date_range=date_range,
        )

    async def _query_personal_chunks(
        self,
        *,
        date_after: str | None = None,
        date_before: str | None = None,
        speaker: str | None = None,
        limit: int = 100,
    ) -> list[PersonalReflection]:
        """Query personal chunks directly from PostgreSQL."""
        conditions = ["c.source_class = 'personal'"]
        params: list[object] = []
        idx = 1

        if date_after:
            conditions.append(f"c.created_at >= ${idx}::timestamptz")
            params.append(date_after)
            idx += 1

        if date_before:
            conditions.append(f"c.created_at <= ${idx}::timestamptz")
            params.append(date_before)
            idx += 1

        if speaker:
            # Filter by speaker: join through works where author matches
            conditions.append(f"""c.work_id IN (
                SELECT work_id FROM works
                WHERE author ILIKE ${idx}
                   OR source_metadata->>'subject_author_id' = ${idx}
            )""")
            params.append(speaker)
            idx += 1

        params.append(limit)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT c.id::text AS chunk_id, c.work_id, c.text,
                   c.created_at::text AS date_created, c.granularity,
                   c.metadata, c.pass_number
            FROM chunks c
            WHERE {where}
            ORDER BY c.created_at ASC
            LIMIT ${idx}
        """

        rows = await self._storage.pg.fetch_all(sql, *params)
        return [self._row_to_reflection(row) for row in rows]

    async def _query_by_theme(
        self,
        theme: str,
        *,
        limit: int = 100,
    ) -> list[PersonalReflection]:
        """Find personal chunks connected to a theme via the graph."""
        canonical = theme.lower().replace(" ", "-")

        try:
            # Query graph for personal chunks exploring this theme
            records = await self._storage.neo4j.execute_read(
                """MATCH (c:Chunk)-[:EXPLORES_THEME]->(t:Theme)
                WHERE (t.canonical_name = $canonical OR t.name = $theme)
                  AND c.source_class = 'personal'
                RETURN c.chunk_id AS chunk_id, c.work_id AS work_id,
                       c.text_preview AS text_preview
                LIMIT $limit""",
                {"canonical": canonical, "theme": theme, "limit": limit},
            )
        except Exception:
            log.debug("theme_graph_query_failed", theme=theme)
            return []

        reflections: list[PersonalReflection] = []
        for rec in records:
            chunk_id = rec["chunk_id"]
            # Get full chunk data from PG
            full = await self._storage.pg.fetch_one(
                """SELECT id::text AS chunk_id, work_id, text,
                          created_at::text AS date_created, granularity, metadata
                FROM chunks WHERE id::text = $1""",
                chunk_id,
            )
            if full:
                reflection = self._row_to_reflection(dict(full))
                reflection.metadata["themes"] = [theme]
                reflections.append(reflection)

        return reflections

    async def _query_by_semantic_similarity(
        self,
        prompt: str,
        *,
        limit: int = 30,
    ) -> list[PersonalReflection]:
        """Find personal chunks semantically related to the prompt."""
        from author_library.retrieval.vector_search import vector_search

        try:
            results = await vector_search(
                prompt,
                embedding_provider=self._embedding_provider,
                embedding_repo=self._storage.embeddings,
                limit=limit,
                source_class_filter="personal",
            )
        except Exception:
            log.debug("semantic_search_personal_failed", prompt=prompt[:100])
            return []

        reflections: list[PersonalReflection] = []
        for r in results:
            chunk_id = str(r.chunk_id)
            # Get creation date from PG
            row = await self._storage.pg.fetch_one(
                "SELECT created_at::text AS date_created, metadata FROM chunks WHERE id = $1",
                r.chunk_id,
            )
            date_created = row["date_created"] if row else ""
            raw_metadata = row["metadata"] if row else {}

            metadata: dict[str, Any] = {}
            if isinstance(raw_metadata, dict):
                metadata = raw_metadata
            metadata["search_source"] = "semantic"
            metadata["relevance_score"] = round(r.score, 4)

            reflections.append(PersonalReflection(
                chunk_id=chunk_id,
                work_id=r.work_id,
                text=r.text,
                date_created=date_created,
                granularity=r.granularity,
                metadata=metadata,
            ))

        return reflections

    @staticmethod
    def _row_to_reflection(row: dict[str, Any]) -> PersonalReflection:
        """Convert a database row to a PersonalReflection."""
        raw_metadata = row.get("metadata") or {}
        metadata: dict[str, Any] = {}
        if isinstance(raw_metadata, dict):
            metadata = raw_metadata
        elif isinstance(raw_metadata, str):
            import json
            try:
                metadata = json.loads(raw_metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}

        return PersonalReflection(
            chunk_id=str(row.get("chunk_id", row.get("id", ""))),
            work_id=row.get("work_id", ""),
            text=row.get("text", ""),
            date_created=row.get("date_created", ""),
            granularity=row.get("granularity", ""),
            metadata=metadata,
        )
