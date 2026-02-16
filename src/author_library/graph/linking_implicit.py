"""Implicit engagement passage linking — MEDIUM confidence tier.

Detects controlled vocabulary terms from contextual sources appearing in
primary chunks via terminology fingerprinting. Creates ENGAGES_WITH edges
with link_type="implicit_engagement" and confidence="medium".

Detection strategies:
  1. Terminology fingerprinting: unique/distinctive terms from contextual
     sources found in primary chunks
  2. Common terms downweighted: terms appearing across many works score lower
  3. Minimum threshold: at least 2 distinct triggering terms for a link
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from author_library.chunking.models import Chunk
    from author_library.storage.neo4j import Neo4jConnection

log = structlog.get_logger(__name__)

# Minimum distinct triggering terms for an implicit link
_MIN_TRIGGERING_TERMS = 2

# Terms appearing in more than this fraction of works are considered common
_COMMON_TERM_THRESHOLD = 0.5

# Minimum word length for candidate terms
_MIN_TERM_LENGTH = 4


@dataclass(frozen=True)
class TermFingerprint:
    """A distinctive term associated with a contextual source."""

    term: str
    canonical_form: str  # lowercased, normalized
    source_work_id: str
    distinctiveness: float  # 0.0 to 1.0 — how unique to this source


@dataclass(frozen=True)
class ImplicitLink:
    """A confirmed implicit engagement link between two chunks."""

    source_chunk_id: str
    target_chunk_id: str
    triggering_terms: list[str]
    engagement_type: str  # extends, responds-to, etc.
    score: float


@dataclass
class ImplicitLinkResult:
    """Aggregated result of implicit engagement detection."""

    links: list[ImplicitLink] = field(default_factory=list)
    edges_created: int = 0
    term_index_size: int = 0
    primary_chunks_scanned: int = 0


class ImplicitEngagementDetector:
    """Detects implicit engagement between primary and contextual chunks."""

    def __init__(self, neo4j: Neo4jConnection) -> None:
        self._neo4j = neo4j

    async def detect_and_link(
        self,
        primary_chunks: list[Chunk],
        contextual_chunks: list[Chunk],
        *,
        term_lists: dict[str, list[str]] | None = None,
        existing_links: set[tuple[str, str]] | None = None,
    ) -> ImplicitLinkResult:
        """Detect implicit engagement via terminology fingerprinting.

        Args:
            primary_chunks: Chunks from primary source works.
            contextual_chunks: Chunks from contextual source works.
            term_lists: Optional pre-computed term lists per work_id.
                        If not provided, terms are extracted from chunk text.
            existing_links: Set of (source_id, target_id) pairs already linked
                           explicitly — these will be skipped.
        """
        result = ImplicitLinkResult()
        already_linked = existing_links or set()

        # Build terminology fingerprint index
        fingerprints = self._build_term_index(contextual_chunks, term_lists)
        result.term_index_size = len(fingerprints)

        # Group contextual chunks by work_id for efficient lookup
        ctx_by_work: dict[str, list[Chunk]] = {}
        for chunk in contextual_chunks:
            ctx_by_work.setdefault(chunk.work_id, []).append(chunk)

        # Scan each primary chunk for fingerprinted terms
        result.primary_chunks_scanned = len(primary_chunks)
        for primary_chunk in primary_chunks:
            primary_terms = self._extract_terms(primary_chunk.text)

            # Group matching fingerprints by source work_id
            matches_by_work: dict[str, list[TermFingerprint]] = {}
            for term in primary_terms:
                for fp in fingerprints:
                    if fp.canonical_form == term:
                        matches_by_work.setdefault(fp.source_work_id, []).append(fp)

            for work_id, matching_fps in matches_by_work.items():
                # Deduplicate by canonical_form
                unique_terms = {fp.canonical_form: fp for fp in matching_fps}
                if len(unique_terms) < _MIN_TRIGGERING_TERMS:
                    continue

                # Find best contextual chunk from this work
                best_ctx = self._find_best_contextual_chunk(
                    list(unique_terms.keys()),
                    ctx_by_work.get(work_id, []),
                )
                if best_ctx is None:
                    continue

                # Skip if already explicitly linked
                pair = (primary_chunk.id, best_ctx.id)
                if pair in already_linked:
                    continue

                # Compute engagement score
                score = sum(fp.distinctiveness for fp in unique_terms.values())
                triggering = sorted(unique_terms.keys())

                link = ImplicitLink(
                    source_chunk_id=primary_chunk.id,
                    target_chunk_id=best_ctx.id,
                    triggering_terms=triggering,
                    engagement_type="extends",  # default; could be refined by LLM
                    score=score,
                )
                result.links.append(link)

                # Persist to Neo4j
                await self._create_engagement_edge(link)
                result.edges_created += 1

        log.info(
            "implicit_link_detection_complete",
            primary_scanned=result.primary_chunks_scanned,
            term_index_size=result.term_index_size,
            links_created=result.edges_created,
        )
        return result

    def _build_term_index(
        self,
        contextual_chunks: list[Chunk],
        term_lists: dict[str, list[str]] | None,
    ) -> list[TermFingerprint]:
        """Build a fingerprint index of distinctive terms from contextual sources."""
        # Count term frequency across works
        work_term_sets: dict[str, set[str]] = {}

        if term_lists:
            # Use provided term lists
            for work_id, tl_terms in term_lists.items():
                work_term_sets[work_id] = {t.lower() for t in tl_terms}
        else:
            # Extract terms from chunk text
            for chunk in contextual_chunks:
                extracted = self._extract_terms(chunk.text)
                existing = work_term_sets.get(chunk.work_id, set())
                existing.update(extracted)
                work_term_sets[chunk.work_id] = existing

        total_works = len(work_term_sets) if work_term_sets else 1

        # Count how many works each term appears in
        term_work_count: Counter[str] = Counter()
        for term_set in work_term_sets.values():
            for term in term_set:
                term_work_count[term] += 1

        # Build fingerprints with distinctiveness scores
        fingerprints: list[TermFingerprint] = []
        for work_id, work_terms in work_term_sets.items():
            for term in work_terms:
                frequency_ratio = term_work_count[term] / total_works
                if frequency_ratio > _COMMON_TERM_THRESHOLD:
                    continue  # Too common across works
                distinctiveness = 1.0 - frequency_ratio
                fingerprints.append(
                    TermFingerprint(
                        term=term,
                        canonical_form=term,
                        source_work_id=work_id,
                        distinctiveness=distinctiveness,
                    )
                )

        return fingerprints

    def _extract_terms(self, text: str) -> set[str]:
        """Extract candidate terms from text, filtering short/common words."""
        # Split into words, lowercase, strip punctuation
        words = re.findall(r"\b[a-z][a-z-]*[a-z]\b", text.lower())
        return {w for w in words if len(w) >= _MIN_TERM_LENGTH and not self._is_stop_word(w)}

    def _is_stop_word(self, word: str) -> bool:
        """Check if a word is a common stop word."""
        stop_words = {
            "that", "this", "with", "from", "have", "been", "were", "will",
            "would", "could", "should", "their", "there", "these", "those",
            "which", "what", "when", "where", "about", "into", "than",
            "them", "then", "some", "such", "also", "more", "most", "very",
            "just", "over", "only", "even", "each", "much", "many", "does",
            "being", "other", "after", "before", "between", "through",
            "under", "during", "without", "however", "another", "because",
        }
        return word in stop_words

    def _find_best_contextual_chunk(
        self,
        triggering_terms: list[str],
        ctx_chunks: list[Chunk],
    ) -> Chunk | None:
        """Find the contextual chunk with highest overlap with triggering terms."""
        if not ctx_chunks:
            return None

        best_chunk: Chunk | None = None
        best_overlap = 0

        term_set = set(triggering_terms)
        for chunk in ctx_chunks:
            chunk_terms = self._extract_terms(chunk.text)
            overlap = len(term_set & chunk_terms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk = chunk

        return best_chunk

    async def _create_engagement_edge(self, link: ImplicitLink) -> None:
        """Create an ENGAGES_WITH edge in Neo4j for an implicit engagement."""
        await self._neo4j.execute_write(
            """MATCH (src:Chunk {chunk_id: $source_chunk_id}),
                   (tgt:Chunk {chunk_id: $target_chunk_id})
            MERGE (src)-[r:ENGAGES_WITH]->(tgt)
            SET r.link_type = "implicit_engagement",
                r.confidence = "medium",
                r.detection_method = "terminology_fingerprint",
                r.triggering_terms = $triggering_terms,
                r.engagement_type = $engagement_type,
                r.score = $score""",
            {
                "source_chunk_id": link.source_chunk_id,
                "target_chunk_id": link.target_chunk_id,
                "triggering_terms": link.triggering_terms,
                "engagement_type": link.engagement_type,
                "score": link.score,
            },
        )
