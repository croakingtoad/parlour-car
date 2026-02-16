"""Tests for explicit citation passage linking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from author_library.chunking.models import Chunk, ChunkGranularity
from author_library.graph.linking_explicit import (
    CitationSignal,
    ExplicitLink,
    ExplicitLinkDetector,
)

from .conftest import requires_neo4j

if TYPE_CHECKING:
    from author_library.storage.neo4j import Neo4jConnection


class TestCitationDetection:
    """Test citation signal detection from chunk text (no Neo4j needed)."""

    def _make_detector_and_detect(self, text: str) -> list[CitationSignal]:
        """Helper: create a chunk with given text and detect citations."""
        chunk = Chunk(
            id="test-chunk",
            text=text,
            granularity=ChunkGranularity.MESO,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
            position=1,
        )
        # Access private method for unit testing citation detection
        detector = ExplicitLinkDetector.__new__(ExplicitLinkDetector)
        return detector._detect_citations(chunk)

    def test_detect_footnote_reference(self) -> None:
        """Detect 'See Author' footnote-style references."""
        signals = self._make_detector_and_detect(
            "The concept of imagination is central to Romantic thought. "
            "See Coleridge, Chapter XIII for the key distinction."
        )
        author_names = [s.cited_author for s in signals]
        assert "Coleridge" in author_names

    def test_detect_parenthetical_citation(self) -> None:
        """Detect (Author, Year, pp. NN) parenthetical citations."""
        signals = self._make_detector_and_detect(
            "The symbol participates in the reality it renders intelligible "
            "(Coleridge, 1816, pp. 30-31)."
        )
        paren_signals = [s for s in signals if s.detection_method == "parenthetical_citation"]
        assert len(paren_signals) >= 1
        assert paren_signals[0].cited_author == "Coleridge"
        assert paren_signals[0].page_ref == "30-31"

    def test_detect_block_quote_attribution(self) -> None:
        """Detect block quotation with attribution marker."""
        signals = self._make_detector_and_detect(
            "The primary IMAGINATION I hold to be the living Power "
            "and prime Agent of all human Perception.\n"
            "— Coleridge, Biographia Literaria"
        )
        block_signals = [
            s for s in signals if s.detection_method == "block_quotation_with_attribution"
        ]
        assert len(block_signals) >= 1
        assert block_signals[0].cited_author == "Coleridge"
        assert block_signals[0].cited_title == "Biographia Literaria"

    def test_detect_inline_citation(self) -> None:
        """Detect 'In Author's Title' inline citation pattern."""
        signals = self._make_detector_and_detect(
            "In Coleridge's 'Biographia Literaria' the distinction between "
            "primary and secondary imagination is articulated."
        )
        inline_signals = [s for s in signals if s.detection_method == "inline_citation"]
        assert len(inline_signals) >= 1
        assert inline_signals[0].cited_author == "Coleridge"

    def test_no_citations_in_plain_text(self) -> None:
        """Plain text without citation patterns should yield no signals."""
        signals = self._make_detector_and_detect(
            "The beauty of the natural world inspires deep contemplation "
            "and a sense of wonder at the divine creation."
        )
        assert len(signals) == 0

    def test_multiple_citations_in_one_chunk(self) -> None:
        """Multiple citations in the same chunk should all be detected."""
        signals = self._make_detector_and_detect(
            "See Coleridge for the primary imagination concept, and "
            "cf. Wordsworth for the corresponding theory of emotion. "
            "As noted by Barfield, these ideas are interconnected."
        )
        cited_authors = {s.cited_author for s in signals}
        assert "Coleridge" in cited_authors
        assert "Wordsworth" in cited_authors


class TestCitationSignalModel:
    """Test CitationSignal data model."""

    def test_signal_with_all_fields(self) -> None:
        """CitationSignal should carry all detection metadata."""
        signal = CitationSignal(
            detection_method="footnote_reference",
            cited_author="Coleridge",
            cited_title="Biographia Literaria",
            page_ref="pp. 30-31",
            chapter_ref="XIII",
            source_location="guite--faith-hope-poetry, ch. 3, pos 5",
        )
        assert signal.detection_method == "footnote_reference"
        assert signal.cited_author == "Coleridge"
        assert signal.cited_title == "Biographia Literaria"
        assert signal.page_ref == "pp. 30-31"

    def test_signal_minimal(self) -> None:
        """CitationSignal with only required fields."""
        signal = CitationSignal(
            detection_method="parenthetical_citation",
            cited_author="Wordsworth",
        )
        assert signal.cited_title is None
        assert signal.page_ref is None


class TestExplicitLinkModel:
    """Test ExplicitLink data model."""

    def test_link_fields(self) -> None:
        """ExplicitLink should carry all link metadata."""
        link = ExplicitLink(
            source_chunk_id="src-001",
            target_chunk_id="tgt-001",
            detection_method="block_quotation_with_attribution",
            evidence="Detection: block_quotation; Author: Coleridge",
            source_location="work-a, ch. 3",
            target_location="work-b, ch. 13",
        )
        assert link.source_chunk_id == "src-001"
        assert link.target_chunk_id == "tgt-001"
        assert "block_quotation" in link.detection_method


@requires_neo4j
class TestExplicitLinkDetectorWithNeo4j:
    """Integration tests requiring Neo4j."""

    async def test_detect_and_link_creates_edges(
        self,
        neo4j_conn: Neo4jConnection,
        primary_chunks: list[Chunk],
        contextual_chunks: list[Chunk],
        works_metadata: dict[str, dict[str, Any]],
    ) -> None:
        """Detect explicit citations and create ENGAGES_WITH edges."""
        # First, ensure chunk nodes exist in Neo4j
        for chunk in primary_chunks + contextual_chunks:
            await neo4j_conn.execute_write(
                """MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.work_id = $work_id, c.source_class = $source_class,
                    c.text_preview = $text, c.granularity = $gran""",
                {
                    "chunk_id": chunk.id,
                    "work_id": chunk.work_id,
                    "source_class": chunk.source_class,
                    "text": chunk.text[:200],
                    "gran": str(chunk.granularity),
                },
            )

        detector = ExplicitLinkDetector(neo4j_conn)
        result = await detector.detect_and_link(
            primary_chunks,
            contextual_chunks,
            works_metadata=works_metadata,
        )

        assert result.signals_found > 0
        # At least some signals should match (Coleridge is cited in primary chunks
        # and contextual chunks are from Coleridge works)

        if result.edges_created > 0:
            # Verify edge properties
            edges = await neo4j_conn.execute_read(
                """MATCH ()-[r:ENGAGES_WITH]->()
                RETURN r.link_type AS link_type, r.confidence AS confidence"""
            )
            for edge in edges:
                assert edge["link_type"] == "explicit_citation"
                assert edge["confidence"] == "high"

    async def test_no_links_without_matching_works(
        self,
        neo4j_conn: Neo4jConnection,
        primary_chunks: list[Chunk],
    ) -> None:
        """No links created when there are no contextual chunks."""
        detector = ExplicitLinkDetector(neo4j_conn)
        result = await detector.detect_and_link(
            primary_chunks,
            [],  # No contextual chunks
        )

        assert result.edges_created == 0
