"""O3: Source citation — link synthesis claims back to specific captures.

Provides detailed citation tracking for synthesis results, including:
  - Full provenance chain from synthesis → reflection → source note
  - Citation verification against actual chunk data
  - Citation formatting for different output contexts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from author_library.synthesis.gatherer import PersonalReflection
from author_library.synthesis.prompt_engine import SourceCitation, SynthesisResult

if TYPE_CHECKING:
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EnrichedCitation:
    """A source citation enriched with full provenance data."""

    capture_id: str
    note_path: str
    excerpt: str
    date: str
    work_id: str
    work_title: str
    author: str
    section_type: str
    themes: list[str]
    chapter: str = ""
    verified: bool = False


@dataclass(frozen=True, slots=True)
class CitationReport:
    """Complete citation report for a synthesis result."""

    citations: list[EnrichedCitation]
    total_citations: int
    verified_count: int
    unverified_count: int
    unique_works: int
    date_span: tuple[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "citations": [
                {
                    "capture_id": c.capture_id,
                    "note_path": c.note_path,
                    "excerpt": c.excerpt,
                    "date": c.date,
                    "work_id": c.work_id,
                    "work_title": c.work_title,
                    "author": c.author,
                    "section_type": c.section_type,
                    "themes": c.themes,
                    "chapter": c.chapter,
                    "verified": c.verified,
                }
                for c in self.citations
            ],
            "total_citations": self.total_citations,
            "verified_count": self.verified_count,
            "unverified_count": self.unverified_count,
            "unique_works": self.unique_works,
            "date_span": list(self.date_span) if self.date_span else None,
        }


class CitationEnricher:
    """Enriches synthesis citations with full provenance data.

    Takes the basic SourceCitation references from a synthesis result
    and resolves them against the database to add work titles, authors,
    themes, and verification status.
    """

    def __init__(self, *, storage: StorageManager) -> None:
        self._storage = storage

    async def enrich(
        self,
        synthesis_result: SynthesisResult,
        reflections: list[PersonalReflection],
    ) -> CitationReport:
        """Enrich citations from a synthesis result with full provenance.

        Args:
            synthesis_result: The synthesis result containing basic citations.
            reflections: The original gathered reflections.

        Returns:
            CitationReport with enriched and verified citations.
        """
        enriched: list[EnrichedCitation] = []

        # Build lookup from chunk_id to reflection
        reflection_map = {r.chunk_id: r for r in reflections}

        for cite in synthesis_result.sources_used:
            reflection = reflection_map.get(cite.capture_id)
            enriched_cite = await self._enrich_single(cite, reflection)
            enriched.append(enriched_cite)

        # Compute summary stats
        verified_count = sum(1 for c in enriched if c.verified)
        unique_works = len({c.work_id for c in enriched if c.work_id})

        dates = [c.date for c in enriched if c.date]
        date_span = (min(dates), max(dates)) if dates else None

        return CitationReport(
            citations=enriched,
            total_citations=len(enriched),
            verified_count=verified_count,
            unverified_count=len(enriched) - verified_count,
            unique_works=unique_works,
            date_span=date_span,
        )

    async def _enrich_single(
        self,
        citation: SourceCitation,
        reflection: PersonalReflection | None,
    ) -> EnrichedCitation:
        """Enrich a single citation with database data."""
        work_id = ""
        work_title = ""
        author = ""
        section_type = "freeform"
        themes: list[str] = []
        chapter = ""
        verified = False

        if reflection:
            work_id = reflection.work_id
            section_type = reflection.section_type
            themes = reflection.themes

        # Verify the chunk exists in the database
        chunk_data = await self._storage.pg.fetch_one(
            """SELECT id, work_id, source_class, granularity, chapter, metadata
            FROM chunks WHERE id::text = $1""",
            citation.capture_id,
        )

        if chunk_data:
            verified = True
            work_id = work_id or chunk_data.get("work_id", "")
            chapter = chunk_data.get("chapter", "") or ""

        # Get work info
        if work_id:
            work_info = await self._storage.works.get(work_id)
            if work_info:
                work_title = work_info.get("title", "")
                author = work_info.get("author", "")

        # Get themes from graph if not available from reflection
        if not themes and citation.capture_id:
            themes = await self._get_chunk_themes(citation.capture_id)

        return EnrichedCitation(
            capture_id=citation.capture_id,
            note_path=citation.note_path,
            excerpt=citation.excerpt,
            date=citation.date,
            work_id=work_id,
            work_title=work_title,
            author=author,
            section_type=section_type,
            themes=themes,
            chapter=chapter,
            verified=verified,
        )

    async def _get_chunk_themes(self, chunk_id: str) -> list[str]:
        """Get themes for a chunk from the graph."""
        try:
            records = await self._storage.neo4j.execute_read(
                """MATCH (c:Chunk)-[:EXPLORES_THEME]->(t:Theme)
                WHERE c.chunk_id = $chunk_id
                RETURN t.canonical_name AS theme""",
                {"chunk_id": chunk_id},
            )
            return [r["theme"] for r in records if r.get("theme")]
        except Exception:
            return []
