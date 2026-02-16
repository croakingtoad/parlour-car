"""Explicit citation passage linking — HIGH confidence tier.

Detects direct citations between primary source chunks and contextual source
chunks by parsing footnotes, endnotes, inline citations, and block quotations.
Creates ENGAGES_WITH edges with link_type="explicit_citation" and confidence="high".

Detection strategies:
  1. Footnote/endnote references → match cited work against collection
  2. Author name + title keywords → locate specific contextual chunks
  3. Block quotations → text matching against contextual source text
  4. Inline citations → parenthetical or textual references
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from author_library.chunking.models import Chunk
    from author_library.storage.neo4j import Neo4jConnection

log = structlog.get_logger(__name__)

# Patterns for detecting explicit citations in text
_FOOTNOTE_REF_PATTERN = re.compile(
    r"(?:See|Cf\.|cf\.|see)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
    r"(?:,\s*(?:(?:pp?\.\s*\d+(?:-\d+)?)|(?:ch(?:apter)?\.?\s*\d+)))?",
)

_PARENTHETICAL_CITE_PATTERN = re.compile(
    r"\(([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:,?\s*(\d{4}))?"
    r"(?:,\s*(?:pp?\.\s*)?(\d+(?:-\d+)?))?\)",
)

_BLOCK_QUOTE_ATTRIBUTION_PATTERN = re.compile(
    r"(?:\u2014|--|\u2013)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
    r"(?:,\s*[\"']?([^\"'\n]+)[\"']?)?"
)

_INLINE_CITE_PATTERN = re.compile(
    r"(?:In\s+|in\s+|According to\s+|as\s+)"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\s+"
    r"(?:[\"']([^\"']+)[\"']|(\w+(?:\s+\w+)*))"
)

# Minimum text overlap ratio for block quotation matching
_QUOTE_MATCH_THRESHOLD = 0.7
# Minimum word overlap for author+title keyword matching
_TITLE_KEYWORD_MIN_MATCHES = 2


@dataclass(frozen=True)
class CitationSignal:
    """A detected citation signal in a primary chunk's text."""

    detection_method: str  # footnote_reference, block_quotation, inline_citation, parenthetical
    cited_author: str
    cited_title: str | None = None
    page_ref: str | None = None
    chapter_ref: str | None = None
    quote_text: str | None = None
    source_location: str | None = None


@dataclass(frozen=True)
class ExplicitLink:
    """A confirmed explicit citation link between two chunks."""

    source_chunk_id: str
    target_chunk_id: str
    detection_method: str
    evidence: str
    source_location: str | None = None
    target_location: str | None = None


@dataclass
class ExplicitLinkResult:
    """Aggregated result of explicit link detection."""

    links: list[ExplicitLink] = field(default_factory=list)
    edges_created: int = 0
    signals_found: int = 0
    signals_matched: int = 0


class ExplicitLinkDetector:
    """Detects explicit citations in primary chunks and links to contextual sources."""

    def __init__(self, neo4j: Neo4jConnection) -> None:
        self._neo4j = neo4j

    async def detect_and_link(
        self,
        primary_chunks: list[Chunk],
        contextual_chunks: list[Chunk],
        *,
        works_metadata: dict[str, dict[str, Any]] | None = None,
    ) -> ExplicitLinkResult:
        """Scan primary chunks for citations, match against contextual chunks.

        Args:
            primary_chunks: Chunks from primary source works.
            contextual_chunks: Chunks from contextual source works.
            works_metadata: Optional dict mapping work_id to metadata (title, author).
        """
        result = ExplicitLinkResult()
        works_meta = works_metadata or {}

        # Build index of contextual chunks by work_id for efficient lookup
        ctx_by_work: dict[str, list[Chunk]] = {}
        for chunk in contextual_chunks:
            ctx_by_work.setdefault(chunk.work_id, []).append(chunk)

        # Build author→work_ids index from works metadata
        author_works: dict[str, list[str]] = {}
        for work_id, meta in works_meta.items():
            author_name = meta.get("author", "").lower()
            if author_name:
                # Index by last name
                last_name = author_name.split()[-1] if author_name.split() else author_name
                author_works.setdefault(last_name, []).append(work_id)

        for chunk in primary_chunks:
            signals = self._detect_citations(chunk)
            result.signals_found += len(signals)

            for signal in signals:
                matched = await self._match_signal(
                    chunk,
                    signal,
                    ctx_by_work,
                    author_works,
                    works_meta,
                )
                if matched:
                    result.signals_matched += 1
                    result.links.append(matched)

                    # Persist edge to Neo4j
                    await self._create_engagement_edge(matched)
                    result.edges_created += 1

        log.info(
            "explicit_link_detection_complete",
            primary_chunks=len(primary_chunks),
            signals_found=result.signals_found,
            signals_matched=result.signals_matched,
            edges_created=result.edges_created,
        )
        return result

    def _detect_citations(self, chunk: Chunk) -> list[CitationSignal]:
        """Parse a primary chunk's text for citation signals."""
        signals: list[CitationSignal] = []
        text = chunk.text

        # Footnote/endnote references
        for match in _FOOTNOTE_REF_PATTERN.finditer(text):
            signals.append(
                CitationSignal(
                    detection_method="footnote_reference",
                    cited_author=match.group(1),
                    source_location=self._chunk_location(chunk),
                )
            )

        # Parenthetical citations: (Author, Year, pp. NN)
        for match in _PARENTHETICAL_CITE_PATTERN.finditer(text):
            signals.append(
                CitationSignal(
                    detection_method="parenthetical_citation",
                    cited_author=match.group(1),
                    page_ref=match.group(3),
                    source_location=self._chunk_location(chunk),
                )
            )

        # Block quotation with attribution
        for match in _BLOCK_QUOTE_ATTRIBUTION_PATTERN.finditer(text):
            signals.append(
                CitationSignal(
                    detection_method="block_quotation_with_attribution",
                    cited_author=match.group(1),
                    cited_title=match.group(2),
                    quote_text=self._extract_preceding_quote(text, match.start()),
                    source_location=self._chunk_location(chunk),
                )
            )

        # Inline citations: "In Author's Title" or "as Author's Work"
        for match in _INLINE_CITE_PATTERN.finditer(text):
            cited_title = match.group(2) or match.group(3)
            signals.append(
                CitationSignal(
                    detection_method="inline_citation",
                    cited_author=match.group(1),
                    cited_title=cited_title,
                    source_location=self._chunk_location(chunk),
                )
            )

        return signals

    @staticmethod
    def _chunk_location(chunk: Chunk) -> str:
        """Format a chunk's location for evidence strings."""
        ch = chunk.chapter or "?"
        return f"{chunk.work_id}, ch. {ch}, pos {chunk.position}"

    def _extract_preceding_quote(self, text: str, attribution_pos: int) -> str:
        """Extract quoted text preceding a block quote attribution marker."""
        # Look backwards from the attribution for quoted text
        preceding = text[:attribution_pos].rstrip()
        # Find the last paragraph break or quote marker
        for marker in ["\n\n", "\n>", "\n  "]:
            idx = preceding.rfind(marker)
            if idx != -1:
                return preceding[idx:].strip().strip(">").strip()
        # Fallback: last 200 chars
        return preceding[-200:].strip()

    async def _match_signal(
        self,
        source_chunk: Chunk,
        signal: CitationSignal,
        ctx_by_work: dict[str, list[Chunk]],
        author_works: dict[str, list[str]],
        works_meta: dict[str, dict[str, Any]],
    ) -> ExplicitLink | None:
        """Try to match a citation signal to a specific contextual chunk."""
        cited_author_lower = signal.cited_author.lower()
        parts = cited_author_lower.split()
        cited_last_name = parts[-1] if parts else cited_author_lower

        # Find candidate work_ids by author
        candidate_work_ids = author_works.get(cited_last_name, [])
        if not candidate_work_ids:
            return None

        # If title hint available, narrow by title keyword overlap
        if signal.cited_title:
            title_words = set(signal.cited_title.lower().split())
            scored_works: list[tuple[str, int]] = []
            for wid in candidate_work_ids:
                meta = works_meta.get(wid, {})
                work_title_words = set(meta.get("title", "").lower().split())
                overlap = len(title_words & work_title_words)
                if overlap >= _TITLE_KEYWORD_MIN_MATCHES:
                    scored_works.append((wid, overlap))
            scored_works.sort(key=lambda x: x[1], reverse=True)
            candidate_work_ids = [w[0] for w in scored_works]

        if not candidate_work_ids:
            return None

        # Find best matching contextual chunk
        best_chunk: Chunk | None = None
        best_score = 0.0

        for wid in candidate_work_ids:
            ctx_chunks = ctx_by_work.get(wid, [])
            for ctx_chunk in ctx_chunks:
                score = self._compute_match_score(signal, ctx_chunk)
                if score > best_score:
                    best_score = score
                    best_chunk = ctx_chunk

        if best_chunk is None:
            return None

        # Build evidence string
        evidence_parts: list[str] = [f"Detection: {signal.detection_method}"]
        if signal.cited_author:
            evidence_parts.append(f"Cited author: {signal.cited_author}")
        if signal.cited_title:
            evidence_parts.append(f"Cited title: {signal.cited_title}")
        if signal.quote_text:
            evidence_parts.append(f"Quote: {signal.quote_text[:100]}...")
        evidence = "; ".join(evidence_parts)

        return ExplicitLink(
            source_chunk_id=source_chunk.id,
            target_chunk_id=best_chunk.id,
            detection_method=signal.detection_method,
            evidence=evidence,
            source_location=signal.source_location,
            target_location=(
                f"{best_chunk.work_id}, ch. {best_chunk.chapter or '?'}, pos {best_chunk.position}"
            ),
        )

    def _compute_match_score(self, signal: CitationSignal, ctx_chunk: Chunk) -> float:
        """Score how well a contextual chunk matches a citation signal."""
        score = 0.0

        # Block quotation text overlap
        if signal.quote_text:
            quote_words = set(signal.quote_text.lower().split())
            chunk_words = set(ctx_chunk.text.lower().split())
            if quote_words:
                overlap = len(quote_words & chunk_words) / len(quote_words)
                if overlap >= _QUOTE_MATCH_THRESHOLD:
                    score += overlap * 2.0  # Strong signal

        # Page/chapter reference matching
        if signal.chapter_ref and ctx_chunk.chapter and signal.chapter_ref in ctx_chunk.chapter:
            score += 1.0

        # If no strong signals, give small baseline for author match
        if score == 0.0:
            score = 0.1

        return score

    async def _create_engagement_edge(self, link: ExplicitLink) -> None:
        """Create an ENGAGES_WITH edge in Neo4j for an explicit citation."""
        await self._neo4j.execute_write(
            """MATCH (src:Chunk {chunk_id: $source_chunk_id}),
                   (tgt:Chunk {chunk_id: $target_chunk_id})
            MERGE (src)-[r:ENGAGES_WITH]->(tgt)
            SET r.link_type = "explicit_citation",
                r.confidence = "high",
                r.detection_method = $detection_method,
                r.evidence = $evidence,
                r.source_location = $source_location,
                r.target_location = $target_location""",
            {
                "source_chunk_id": link.source_chunk_id,
                "target_chunk_id": link.target_chunk_id,
                "detection_method": link.detection_method,
                "evidence": link.evidence,
                "source_location": link.source_location or "",
                "target_location": link.target_location or "",
            },
        )
