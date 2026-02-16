"""Thematic parallel passage linking — LOW confidence tier.

Detects semantic similarity between primary and contextual chunks that
share canonical themes. Creates THEMATIC_PARALLEL edges with confidence="low"
when cosine similarity exceeds 0.85 and no ENGAGES_WITH edge already exists.

This tier is exploratory — low confidence, presented as "possibly related"
in retrieval, never as direct engagement evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from author_library.chunking.models import Chunk
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.neo4j import Neo4jConnection

log = structlog.get_logger(__name__)

# Cosine similarity threshold for creating THEMATIC_PARALLEL edges
_SIMILARITY_THRESHOLD = 0.85


@dataclass(frozen=True)
class ThematicLink:
    """A confirmed thematic parallel between two chunks."""

    source_chunk_id: str
    target_chunk_id: str
    similarity_score: float
    shared_themes: list[str]


@dataclass
class ThematicLinkResult:
    """Aggregated result of thematic parallel detection."""

    links: list[ThematicLink] = field(default_factory=list)
    edges_created: int = 0
    pairs_evaluated: int = 0
    pairs_above_threshold: int = 0
    pairs_skipped_existing: int = 0


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot: float = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class ThematicParallelDetector:
    """Detects thematic parallels via embedding similarity + shared themes."""

    def __init__(
        self,
        neo4j: Neo4jConnection,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._neo4j = neo4j
        self._embedding = embedding_provider

    async def detect_and_link(
        self,
        primary_chunks: list[Chunk],
        contextual_chunks: list[Chunk],
        *,
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
    ) -> ThematicLinkResult:
        """Detect thematic parallels between primary and contextual chunks.

        Only considers pairs that share at least one canonical theme in Neo4j.
        Skips pairs that already have ENGAGES_WITH edges.
        """
        result = ThematicLinkResult()

        if not primary_chunks or not contextual_chunks:
            return result

        # Get theme assignments for all chunks from Neo4j
        primary_themes = await self._get_chunk_themes(primary_chunks)
        contextual_themes = await self._get_chunk_themes(contextual_chunks)

        # Find candidate pairs: chunks sharing at least one theme
        candidate_pairs = self._find_theme_overlapping_pairs(
            primary_chunks, contextual_chunks, primary_themes, contextual_themes
        )

        if not candidate_pairs:
            log.info("thematic_parallel_no_candidates", message="No theme-overlapping pairs found")
            return result

        # Get existing ENGAGES_WITH edges to filter out
        existing_engagement = await self._get_existing_engagements(primary_chunks)

        # Compute embeddings for candidate chunks
        all_chunk_ids = set()
        for p_id, c_id, _ in candidate_pairs:
            all_chunk_ids.add(p_id)
            all_chunk_ids.add(c_id)

        primary_by_id = {c.id: c for c in primary_chunks}
        contextual_by_id = {c.id: c for c in contextual_chunks}

        embeddings_cache: dict[str, list[float]] = {}
        # Batch embed primary chunks
        primary_texts = []
        primary_ids = []
        for cid in all_chunk_ids:
            chunk = primary_by_id.get(cid) or contextual_by_id.get(cid)
            if chunk and cid not in embeddings_cache:
                primary_texts.append(chunk.annotated_text)
                primary_ids.append(cid)

        if primary_texts:
            batch_result = await self._embedding.embed_batch(primary_texts)
            for cid, vec in zip(primary_ids, batch_result.vectors, strict=True):
                embeddings_cache[cid] = vec

        # Evaluate each candidate pair
        for p_id, c_id, shared in candidate_pairs:
            result.pairs_evaluated += 1

            # Skip if already has ENGAGES_WITH
            if (p_id, c_id) in existing_engagement:
                result.pairs_skipped_existing += 1
                continue

            vec_p = embeddings_cache.get(p_id)
            vec_c = embeddings_cache.get(c_id)
            if vec_p is None or vec_c is None:
                continue

            similarity = _cosine_similarity(vec_p, vec_c)
            if similarity >= similarity_threshold:
                result.pairs_above_threshold += 1

                link = ThematicLink(
                    source_chunk_id=p_id,
                    target_chunk_id=c_id,
                    similarity_score=similarity,
                    shared_themes=sorted(shared),
                )
                result.links.append(link)

                await self._create_thematic_edge(link)
                result.edges_created += 1

        log.info(
            "thematic_parallel_detection_complete",
            candidates=len(candidate_pairs),
            evaluated=result.pairs_evaluated,
            above_threshold=result.pairs_above_threshold,
            skipped_existing=result.pairs_skipped_existing,
            edges_created=result.edges_created,
        )
        return result

    async def _get_chunk_themes(
        self, chunks: list[Chunk]
    ) -> dict[str, set[str]]:
        """Get canonical theme names for each chunk from Neo4j."""
        themes_map: dict[str, set[str]] = {}
        for chunk in chunks:
            records = await self._neo4j.execute_read(
                """MATCH (c:Chunk {chunk_id: $chunk_id})-[:EXPLORES_THEME]->(t:Theme)
                RETURN t.canonical_name AS theme""",
                {"chunk_id": chunk.id},
            )
            themes_map[chunk.id] = {r["theme"] for r in records}
        return themes_map

    def _find_theme_overlapping_pairs(
        self,
        primary_chunks: list[Chunk],
        contextual_chunks: list[Chunk],
        primary_themes: dict[str, set[str]],
        contextual_themes: dict[str, set[str]],
    ) -> list[tuple[str, str, list[str]]]:
        """Find pairs of chunks that share at least one canonical theme.

        Returns list of (primary_chunk_id, contextual_chunk_id, shared_themes).
        """
        pairs: list[tuple[str, str, list[str]]] = []
        for p_chunk in primary_chunks:
            p_themes = primary_themes.get(p_chunk.id, set())
            if not p_themes:
                continue
            for c_chunk in contextual_chunks:
                c_themes = contextual_themes.get(c_chunk.id, set())
                shared = p_themes & c_themes
                if shared:
                    pairs.append((p_chunk.id, c_chunk.id, sorted(shared)))
        return pairs

    async def _get_existing_engagements(
        self, primary_chunks: list[Chunk]
    ) -> set[tuple[str, str]]:
        """Get all existing ENGAGES_WITH edges from primary chunks."""
        existing: set[tuple[str, str]] = set()
        for chunk in primary_chunks:
            records = await self._neo4j.execute_read(
                """MATCH (c:Chunk {chunk_id: $chunk_id})-[:ENGAGES_WITH]->(t:Chunk)
                RETURN c.chunk_id AS source, t.chunk_id AS target""",
                {"chunk_id": chunk.id},
            )
            for r in records:
                existing.add((r["source"], r["target"]))
        return existing

    async def _create_thematic_edge(self, link: ThematicLink) -> None:
        """Create a THEMATIC_PARALLEL edge in Neo4j."""
        await self._neo4j.execute_write(
            """MATCH (src:Chunk {chunk_id: $source_chunk_id}),
                   (tgt:Chunk {chunk_id: $target_chunk_id})
            MERGE (src)-[r:THEMATIC_PARALLEL]->(tgt)
            SET r.link_type = "thematic_parallel",
                r.confidence = "low",
                r.detection_method = "semantic_similarity",
                r.similarity_score = $similarity_score,
                r.shared_themes = $shared_themes""",
            {
                "source_chunk_id": link.source_chunk_id,
                "target_chunk_id": link.target_chunk_id,
                "similarity_score": link.similarity_score,
                "shared_themes": link.shared_themes,
            },
        )
