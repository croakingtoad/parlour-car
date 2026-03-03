"""N1: Post-ingestion connection scanner.

After new material is ingested, scans existing content for new connections.
Extends retroactive linking from Epic B4 (detect_passage_links with
retroactive_scan). New connections are staged (not immediately applied) —
they go through the PR workflow for user approval.

Usage:
    scanner = ConnectionScanner(settings, storage, embedding_provider)
    result = await scanner.scan_new_connections(work_id)
    # result contains staged connections grouped by type/confidence
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from author_library.surfacing.confidence import ConfidenceLevel, classify_confidence
from author_library.surfacing.related_content import (
    ConnectionType,
    RelatedContentFinder,
    RelatedItem,
)

if TYPE_CHECKING:
    from author_library.cache import CacheManager
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class StagedConnection:
    """A discovered connection staged for PR review."""

    source_chunk_id: str
    target_chunk_id: str
    source_work_id: str
    target_work_id: str
    connection_type: str
    confidence_level: str
    confidence_label: str
    source_excerpt: str
    target_excerpt: str
    explanation: str


@dataclass
class ScanResult:
    """Result of a post-ingestion connection scan."""

    work_id: str
    connections: list[StagedConnection] = field(default_factory=list)
    by_confidence: dict[str, list[StagedConnection]] = field(default_factory=dict)
    by_target_work: dict[str, list[StagedConnection]] = field(default_factory=dict)
    total_found: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON output."""
        return {
            "work_id": self.work_id,
            "total_found": self.total_found,
            "by_confidence": {
                level: [
                    {
                        "source_chunk_id": c.source_chunk_id,
                        "target_chunk_id": c.target_chunk_id,
                        "source_work_id": c.source_work_id,
                        "target_work_id": c.target_work_id,
                        "connection_type": c.connection_type,
                        "confidence_level": c.confidence_level,
                        "confidence_label": c.confidence_label,
                        "source_excerpt": c.source_excerpt,
                        "target_excerpt": c.target_excerpt,
                        "explanation": c.explanation,
                    }
                    for c in conns
                ]
                for level, conns in self.by_confidence.items()
            },
            "by_target_work": {
                work: len(conns) for work, conns in self.by_target_work.items()
            },
            "errors": self.errors,
        }


# Confidence level → presentation label mapping
_CONFIDENCE_LABELS = {
    ConfidenceLevel.HIGH: "This directly engages with",
    ConfidenceLevel.MEDIUM: "This appears to connect to",
    ConfidenceLevel.LOW: "You might find this relevant",
}


class ConnectionScanner:
    """Scans for new connections after material ingestion.

    After a work is ingested, this scanner:
    1. Loads all chunks from the newly ingested work
    2. For each chunk, finds related content across existing works
    3. Filters out connections that already exist as passage links
    4. Classifies and stages new connections for PR review
    """

    def __init__(
        self,
        settings: Settings,
        storage: StorageManager,
        embedding_provider: EmbeddingProvider,
        cache_manager: CacheManager | None = None,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._embedding = embedding_provider
        self._cache = cache_manager
        self._finder = RelatedContentFinder(
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
            cache_manager=cache_manager,
        )

    async def scan_new_connections(
        self,
        work_id: str,
        *,
        confidence_threshold: float = 0.4,
        max_per_chunk: int = 10,
        batch_size: int = 5,
    ) -> ScanResult:
        """Scan for new connections after ingesting a work.

        Args:
            work_id: The newly ingested work to scan from.
            confidence_threshold: Minimum confidence to include.
            max_per_chunk: Max connections per source chunk.
            batch_size: How many chunks to scan in parallel.

        Returns:
            ScanResult with all staged connections.
        """
        result = ScanResult(work_id=work_id)

        log.info(
            "connection_scan_start",
            work_id=work_id,
            confidence_threshold=confidence_threshold,
        )

        # Load chunks for the new work
        try:
            chunks = await self._storage.chunks.list_by_work(work_id)
        except Exception as exc:
            error_msg = f"Failed to load chunks for {work_id}: {exc}"
            log.error("connection_scan_chunk_load_failed", error=error_msg)
            result.errors.append(error_msg)
            return result

        if not chunks:
            log.info("connection_scan_no_chunks", work_id=work_id)
            return result

        log.info("connection_scan_chunks_loaded", work_id=work_id, count=len(chunks))

        # Load existing passage links to avoid duplicates
        existing_links = await self._get_existing_links(work_id)

        # Per-chunk vector similarity scan (only if embedding provider available)
        if self._embedding is not None:
            for batch_start in range(0, len(chunks), batch_size):
                batch = chunks[batch_start : batch_start + batch_size]
                tasks = [
                    self._scan_single_chunk(
                        chunk,
                        work_id=work_id,
                        existing_links=existing_links,
                        confidence_threshold=confidence_threshold,
                        max_results=max_per_chunk,
                    )
                    for chunk in batch
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for batch_result in batch_results:
                    if isinstance(batch_result, Exception):
                        result.errors.append(str(batch_result))
                        continue
                    result.connections.extend(batch_result)
        else:
            log.info(
                "connection_scan_skip_vector",
                work_id=work_id,
                reason="no embedding provider — using entity overlap only",
            )

        # Discover cross-work connections via shared entity graph nodes
        try:
            entity_connections = await self._find_entity_overlap_connections(
                work_id=work_id,
                existing_links=existing_links,
                limit=100,
            )
            result.connections.extend(entity_connections)
            log.info(
                "connection_scan_entity_overlap",
                work_id=work_id,
                found=len(entity_connections),
            )
        except Exception as exc:
            error_msg = f"Entity overlap scan failed for {work_id}: {exc}"
            log.warning("connection_scan_entity_overlap_failed", error=error_msg)
            result.errors.append(error_msg)

        # Group by confidence and target work
        result.total_found = len(result.connections)
        result.by_confidence = self._group_by_confidence(result.connections)
        result.by_target_work = self._group_by_target_work(result.connections)

        log.info(
            "connection_scan_complete",
            work_id=work_id,
            total_found=result.total_found,
            high=len(result.by_confidence.get("high", [])),
            medium=len(result.by_confidence.get("medium", [])),
            low=len(result.by_confidence.get("low", [])),
        )

        return result

    async def _scan_single_chunk(
        self,
        chunk: dict[str, Any],
        *,
        work_id: str,
        existing_links: set[tuple[str, str]],
        confidence_threshold: float,
        max_results: int,
    ) -> list[StagedConnection]:
        """Scan a single chunk for connections to existing content."""
        chunk_id = str(chunk.get("id", ""))
        chunk_text = chunk.get("text", "")

        if not chunk_id or not chunk_text:
            return []

        # Use RelatedContentFinder to discover connections
        try:
            related = await self._finder.find_related(
                chunk_id=chunk_id,
                text_context=chunk_text[:500],
                include_personal=False,
                max_results=max_results * 2,
            )
        except Exception as exc:
            log.warning(
                "connection_scan_chunk_failed",
                chunk_id=chunk_id,
                error=str(exc),
            )
            return []

        staged: list[StagedConnection] = []
        for item in related.items:
            # Skip self-references
            if item.work_id == work_id:
                continue

            # Skip already-linked pairs
            pair = (chunk_id, item.chunk_id)
            reverse_pair = (item.chunk_id, chunk_id)
            if pair in existing_links or reverse_pair in existing_links:
                continue

            # Classify confidence
            scored = classify_confidence(item)
            level = scored.confidence_level
            if level == ConfidenceLevel.LOW and item.relevance_score < confidence_threshold:
                continue

            label = _CONFIDENCE_LABELS.get(level, "Related content")
            explanation = self._build_explanation(item, level)

            staged.append(
                StagedConnection(
                    source_chunk_id=chunk_id,
                    target_chunk_id=item.chunk_id,
                    source_work_id=work_id,
                    target_work_id=item.work_id or "",
                    connection_type=item.connection_type.value,
                    confidence_level=level.value,
                    confidence_label=label,
                    source_excerpt=chunk_text[:300],
                    target_excerpt=item.text[:300] if item.text else "",
                    explanation=explanation,
                )
            )

        return staged[:max_results]

    async def _get_existing_links(self, work_id: str) -> set[tuple[str, str]]:
        """Get existing passage links involving this work to avoid duplicates."""
        links: set[tuple[str, str]] = set()
        try:
            if hasattr(self._storage, "graph") and self._storage.graph:
                existing = await self._storage.graph.get_passage_links_for_work(work_id)
                for link in existing:
                    src = link.get("source_chunk_id", "")
                    tgt = link.get("target_chunk_id", "")
                    if src and tgt:
                        links.add((src, tgt))
        except Exception as exc:
            log.warning(
                "connection_scan_existing_links_failed",
                work_id=work_id,
                error=str(exc),
            )
        return links

    async def _find_entity_overlap_connections(
        self,
        *,
        work_id: str,
        existing_links: set[tuple[str, str]],
        limit: int = 100,
    ) -> list[StagedConnection]:
        """Discover cross-work connections via shared entity graph nodes.

        Queries Neo4j for chunk pairs where a chunk from the new work and a
        chunk from a different work both connect to the same entity nodes
        (themes, persons, or concepts). Pairs with 2+ shared entities are
        returned as entity-overlap connections.

        Args:
            work_id: The newly ingested work to find connections for.
            existing_links: Set of (source, target) chunk ID pairs to skip.
            limit: Maximum number of connections to return.

        Returns:
            List of StagedConnections from entity overlap discovery.
        """
        if not (hasattr(self._storage, "neo4j") and self._storage.neo4j):
            log.debug("entity_overlap_skip_no_neo4j")
            return []

        # Query for chunk pairs from different works sharing entity nodes
        query = """
        MATCH (c1:Chunk {work_id: $new_work_id})
              -[:EXPLORES_THEME|REFERENCES_PERSON|CONCEPT_USED_IN]->(entity)
              <-[:EXPLORES_THEME|REFERENCES_PERSON|CONCEPT_USED_IN]-(c2:Chunk)
        WHERE c2.work_id <> $new_work_id
        WITH c1, c2,
             collect(DISTINCT entity.name) AS shared_entities,
             collect(DISTINCT labels(entity)[0]) AS entity_types,
             count(DISTINCT entity) AS overlap_count
        WHERE overlap_count >= 2
        RETURN c1.chunk_id AS source_chunk_id,
               c2.chunk_id AS target_chunk_id,
               c1.work_id AS source_work,
               c2.work_id AS target_work,
               c1.text_preview AS source_preview,
               c2.text_preview AS target_preview,
               shared_entities,
               entity_types,
               overlap_count
        ORDER BY overlap_count DESC
        LIMIT $limit
        """

        try:
            records = await self._storage.neo4j.execute_read(
                query,
                {"new_work_id": work_id, "limit": limit},
            )
        except Exception as exc:
            log.warning(
                "entity_overlap_query_failed",
                work_id=work_id,
                error=str(exc),
            )
            return []

        if not records:
            return []

        staged: list[StagedConnection] = []
        for record in records:
            source_id = record["source_chunk_id"]
            target_id = record["target_chunk_id"]

            # Skip already-linked pairs
            pair = (source_id, target_id)
            reverse_pair = (target_id, source_id)
            if pair in existing_links or reverse_pair in existing_links:
                continue

            # Also skip duplicates within this batch (same pair found
            # via different entity combinations)
            if any(
                c.source_chunk_id == source_id and c.target_chunk_id == target_id
                for c in staged
            ):
                continue

            overlap_count = record["overlap_count"]
            shared_entities = record["shared_entities"]

            # Map overlap count to confidence level
            if overlap_count >= 5:
                level = ConfidenceLevel.HIGH
                # Map to a relevance_score for consistent scoring
                relevance_score = 0.95
            elif overlap_count >= 3:
                level = ConfidenceLevel.MEDIUM
                relevance_score = 0.75
            else:
                level = ConfidenceLevel.LOW
                relevance_score = 0.5

            label = _CONFIDENCE_LABELS.get(level, "Related content")

            # Build explanation listing the shared entities
            entity_list = ", ".join(shared_entities[:5])
            if len(shared_entities) > 5:
                entity_list += f" (and {len(shared_entities) - 5} more)"
            explanation = (
                f"Both passages share {overlap_count} entities: {entity_list}."
            )

            staged.append(
                StagedConnection(
                    source_chunk_id=source_id,
                    target_chunk_id=target_id,
                    source_work_id=work_id,
                    target_work_id=record["target_work"],
                    connection_type=ConnectionType.ENTITY_OVERLAP.value,
                    confidence_level=level.value,
                    confidence_label=label,
                    source_excerpt=record.get("source_preview", "") or "",
                    target_excerpt=record.get("target_preview", "") or "",
                    explanation=explanation,
                )
            )

        log.info(
            "entity_overlap_connections_found",
            work_id=work_id,
            total_candidates=len(records),
            after_dedup=len(staged),
        )

        return staged

    @staticmethod
    def _build_explanation(item: RelatedItem, level: ConfidenceLevel) -> str:
        """Build human-readable explanation of WHY a connection exists."""
        conn_type = item.connection_type

        if conn_type == ConnectionType.PASSAGE_LINK:
            return (
                f"This passage directly engages with the referenced content "
                f"through {'explicit citation' if level == ConfidenceLevel.HIGH else 'implicit engagement'}."
            )
        elif conn_type == ConnectionType.THEMATIC_PARALLEL:
            themes = item.metadata.get("shared_themes", []) if item.metadata else []
            if themes:
                theme_str = ", ".join(themes[:3])
                return f"Both passages explore shared themes: {theme_str}."
            return "These passages explore parallel thematic territory."
        elif conn_type == ConnectionType.VECTOR_SIMILARITY:
            return "Semantic similarity suggests these passages address related ideas."
        elif conn_type == ConnectionType.PERSONAL_REFLECTION:
            return "Your personal reflection engages with this content."
        elif conn_type == ConnectionType.ENTITY_OVERLAP:
            entities = item.metadata.get("shared_entities", []) if item.metadata else []
            if entities:
                entity_str = ", ".join(entities[:5])
                return f"Both passages engage with shared entities: {entity_str}."
            return "Both passages share multiple entities in the knowledge graph."
        else:
            return "A potential connection was detected between these passages."

    @staticmethod
    def _group_by_confidence(
        connections: list[StagedConnection],
    ) -> dict[str, list[StagedConnection]]:
        """Group connections by confidence level."""
        grouped: dict[str, list[StagedConnection]] = {
            "high": [],
            "medium": [],
            "low": [],
        }
        for conn in connections:
            grouped.setdefault(conn.confidence_level, []).append(conn)
        return {k: v for k, v in grouped.items() if v}

    @staticmethod
    def _group_by_target_work(
        connections: list[StagedConnection],
    ) -> dict[str, list[StagedConnection]]:
        """Group connections by target work for batch PR creation."""
        grouped: dict[str, list[StagedConnection]] = {}
        for conn in connections:
            grouped.setdefault(conn.target_work_id, []).append(conn)
        return grouped
