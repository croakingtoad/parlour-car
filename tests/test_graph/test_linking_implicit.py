"""Tests for implicit engagement passage linking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from author_library.chunking.models import Chunk, ChunkGranularity
from author_library.graph.linking_implicit import (
    ImplicitEngagementDetector,
    ImplicitLink,
)

from .conftest import requires_neo4j

if TYPE_CHECKING:
    from author_library.storage.neo4j import Neo4jConnection


class TestTermExtraction:
    """Test terminology extraction and fingerprinting."""

    def _get_detector(self) -> ImplicitEngagementDetector:
        """Create a detector for testing private methods."""
        return ImplicitEngagementDetector.__new__(ImplicitEngagementDetector)

    def test_extract_terms_filters_short_words(self) -> None:
        """Words shorter than 4 chars should be excluded."""
        detector = self._get_detector()
        terms = detector._extract_terms("The cat sat on a mat and was happy about it.")
        # "the", "cat", "sat", "mat", "and", "was" are all <= 3 chars or stop words
        assert "cat" not in terms  # 3 chars
        assert "sat" not in terms  # 3 chars
        assert "happy" in terms

    def test_extract_terms_filters_stop_words(self) -> None:
        """Common stop words should be excluded."""
        detector = self._get_detector()
        terms = detector._extract_terms(
            "However, through these various concepts, there were many other things."
        )
        assert "however" not in terms
        assert "through" not in terms
        assert "these" not in terms
        assert "various" in terms
        assert "concepts" in terms

    def test_extract_terms_with_scholarly_vocabulary(self) -> None:
        """Scholarly vocabulary should be preserved."""
        detector = self._get_detector()
        terms = detector._extract_terms(
            "The esemplastic power of imagination operates through symbols "
            "that participate in sacramental reality."
        )
        assert "esemplastic" in terms
        assert "imagination" in terms
        assert "symbols" in terms
        assert "sacramental" in terms


class TestTermFingerprinting:
    """Test the term fingerprint index building."""

    def test_build_term_index_from_chunks(self) -> None:
        """Build term index from contextual chunk text."""
        detector = ImplicitEngagementDetector.__new__(ImplicitEngagementDetector)

        # Need 2+ works so terms aren't filtered as common (>50% threshold)
        ctx_chunks = [
            Chunk(
                id="ctx-fp-001",
                text="The esemplastic power shapes imagination into living unity.",
                granularity=ChunkGranularity.MESO,
                work_id="coleridge--biographia-literaria",
                source_class="contextual",
                position=1,
            ),
            Chunk(
                id="ctx-fp-002",
                text="Nature provides solace through beauty and wonder.",
                granularity=ChunkGranularity.MESO,
                work_id="wordsworth--prelude",
                source_class="contextual",
                position=1,
            ),
        ]

        fingerprints = detector._build_term_index(ctx_chunks, term_lists=None)
        canonical_forms = {fp.canonical_form for fp in fingerprints}
        assert "esemplastic" in canonical_forms
        # "imagination" only in one work, should be preserved
        assert "imagination" in canonical_forms

    def test_build_term_index_from_term_lists(self) -> None:
        """Build term index from provided term lists."""
        detector = ImplicitEngagementDetector.__new__(ImplicitEngagementDetector)

        # Need 2+ works so terms aren't filtered as common (>50% threshold)
        term_lists = {
            "coleridge--biographia-literaria": [
                "Primary Imagination",
                "esemplastic",
                "secondary Imagination",
            ],
            "wordsworth--prelude": [
                "Nature",
                "solitude",
                "recollection",
            ],
        }

        fingerprints = detector._build_term_index([], term_lists=term_lists)
        canonical_forms = {fp.canonical_form for fp in fingerprints}
        assert "esemplastic" in canonical_forms
        # Term lists preserve full phrases as-is (lowercased)
        assert "primary imagination" in canonical_forms
        assert "secondary imagination" in canonical_forms

    def test_common_terms_filtered_out(self) -> None:
        """Terms appearing in > 50% of works should be excluded."""
        detector = ImplicitEngagementDetector.__new__(ImplicitEngagementDetector)

        # Create chunks from 3 different works, all containing "poetry"
        chunks = [
            Chunk(
                id=f"ctx-common-{i}",
                text="Poetry and imagination are central themes in literature.",
                granularity=ChunkGranularity.MESO,
                work_id=f"work-{i}",
                source_class="contextual",
                position=1,
            )
            for i in range(3)
        ]

        fingerprints = detector._build_term_index(chunks, term_lists=None)
        # "poetry" appears in all 3 works (100%) — should be filtered
        canonical_forms = {fp.canonical_form for fp in fingerprints}
        assert "poetry" not in canonical_forms
        assert "imagination" not in canonical_forms
        assert "central" not in canonical_forms


class TestImplicitLinkModel:
    """Test ImplicitLink data model."""

    def test_link_fields(self) -> None:
        """ImplicitLink should carry triggering terms and score."""
        link = ImplicitLink(
            source_chunk_id="p-001",
            target_chunk_id="c-001",
            triggering_terms=["esemplastic", "imagination"],
            engagement_type="extends",
            score=1.8,
        )
        assert len(link.triggering_terms) == 2
        assert link.engagement_type == "extends"
        assert link.score == 1.8


class TestFindBestContextualChunk:
    """Test contextual chunk matching by term overlap."""

    def test_best_chunk_by_overlap(self) -> None:
        """The chunk with most triggering term overlap should be selected."""
        detector = ImplicitEngagementDetector.__new__(ImplicitEngagementDetector)

        chunks = [
            Chunk(
                id="ctx-match-001",
                text="The weather today is warm and pleasant with sunshine.",
                granularity=ChunkGranularity.MESO,
                work_id="test-work",
                source_class="contextual",
                position=1,
            ),
            Chunk(
                id="ctx-match-002",
                text="The esemplastic imagination shapes reality into living symbol.",
                granularity=ChunkGranularity.MESO,
                work_id="test-work",
                source_class="contextual",
                position=2,
            ),
        ]

        best = detector._find_best_contextual_chunk(
            ["esemplastic", "imagination", "symbol"],
            chunks,
        )
        assert best is not None
        assert best.id == "ctx-match-002"

    def test_no_match_returns_none(self) -> None:
        """Empty chunk list returns None."""
        detector = ImplicitEngagementDetector.__new__(ImplicitEngagementDetector)
        result = detector._find_best_contextual_chunk(["term1", "term2"], [])
        assert result is None


@requires_neo4j
class TestImplicitEngagementWithNeo4j:
    """Integration tests requiring Neo4j."""

    async def test_detect_implicit_engagement(
        self,
        neo4j_conn: Neo4jConnection,
        primary_chunks: list[Chunk],
        contextual_chunks: list[Chunk],
    ) -> None:
        """Detect implicit engagement and create edges."""
        # Create chunk nodes
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

        detector = ImplicitEngagementDetector(neo4j_conn)
        result = await detector.detect_and_link(
            primary_chunks,
            contextual_chunks,
        )

        # The primary chunks contain Coleridgean terminology
        # (esemplastic, Primary Imagination, etc.) that should fingerprint
        # against the contextual Coleridge chunks
        assert result.term_index_size > 0
        assert result.primary_chunks_scanned == len(primary_chunks)

        if result.edges_created > 0:
            edges = await neo4j_conn.execute_read(
                """MATCH ()-[r:ENGAGES_WITH]->()
                WHERE r.link_type = 'implicit_engagement'
                RETURN r.confidence AS confidence, r.triggering_terms AS terms"""
            )
            for edge in edges:
                assert edge["confidence"] == "medium"
                assert len(edge["terms"]) >= 2

    async def test_skips_already_linked_pairs(
        self,
        neo4j_conn: Neo4jConnection,
        primary_chunks: list[Chunk],
        contextual_chunks: list[Chunk],
    ) -> None:
        """Pairs with existing explicit links should be skipped."""
        # Create chunk nodes
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

        # Pre-create an explicit link between first primary and first contextual
        existing = {(primary_chunks[0].id, contextual_chunks[0].id)}

        detector = ImplicitEngagementDetector(neo4j_conn)
        result = await detector.detect_and_link(
            primary_chunks,
            contextual_chunks,
            existing_links=existing,
        )

        # Any link that would be between primary-chunk-001 and ctx-chunk-001
        # should not appear
        for link in result.links:
            assert not (
                link.source_chunk_id == primary_chunks[0].id
                and link.target_chunk_id == contextual_chunks[0].id
            )
