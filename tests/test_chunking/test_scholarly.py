"""Tests for scholarly prose chunking strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from author_library.chunking.models import ChunkGranularity
from author_library.chunking.scholarly import ScholarlyProseStrategy

if TYPE_CHECKING:
    from author_library.parsing.models import ParsedDocument


class TestScholarlyProseStrategy:
    def setup_method(self) -> None:
        self.strategy = ScholarlyProseStrategy()

    def test_supported_genres(self) -> None:
        genres = self.strategy.supported_genres()
        assert "scholarly_prose" in genres
        assert "monograph" in genres
        assert "academic_paper" in genres

    def test_produces_all_granularities(self, scholarly_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            scholarly_document,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
        )
        granularities = {c.granularity for c in chunks}
        assert ChunkGranularity.MACRO in granularities
        assert ChunkGranularity.MESO in granularities
        assert ChunkGranularity.MICRO in granularities

    def test_macro_chunks_per_chapter(self, scholarly_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            scholarly_document,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
        )
        macro_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MACRO]
        # One chapter in fixture → one macro chunk
        assert len(macro_chunks) == 1
        assert macro_chunks[0].chapter == "The Poetic Imagination"

    def test_source_class_propagates(self, scholarly_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            scholarly_document,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
        )
        for chunk in chunks:
            assert chunk.source_class == "primary"

    def test_work_id_propagates(self, scholarly_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            scholarly_document,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
        )
        for chunk in chunks:
            assert chunk.work_id == "guite--faith-hope-poetry"

    def test_meso_chunks_from_sections(self, scholarly_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            scholarly_document,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
        )
        meso_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        # Two sections in fixture
        assert len(meso_chunks) >= 2

    def test_micro_chunks_from_paragraphs(self, scholarly_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            scholarly_document,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
        )
        micro_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MICRO]
        # At least some micro chunks from the paragraphs
        assert len(micro_chunks) >= 1

    def test_bibliography_not_chunked(self, scholarly_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            scholarly_document,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
        )
        for chunk in chunks:
            # Bibliography entries should not appear as standalone chunks
            assert (
                "Biographia Literaria. 1817." not in chunk.text
                or chunk.granularity == ChunkGranularity.MACRO
            )

    def test_block_quote_metadata(self, scholarly_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            scholarly_document,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
        )
        # At least one chunk should contain the block quote text
        block_quote_text = "The primary IMAGINATION I hold to be"
        has_quote = any(block_quote_text in c.text for c in chunks)
        assert has_quote

    def test_parent_child_relationships(self, scholarly_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            scholarly_document,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
        )
        macro_ids = {c.id for c in chunks if c.granularity == ChunkGranularity.MACRO}
        meso_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        micro_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MICRO]

        # Meso chunks should have macro parents
        for meso in meso_chunks:
            assert meso.parent_chunk_id in macro_ids

        # Micro chunks should have meso parents
        meso_ids = {c.id for c in meso_chunks}
        for micro in micro_chunks:
            assert micro.parent_chunk_id in meso_ids

    def test_genre_metadata(self, scholarly_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            scholarly_document,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
        )
        for chunk in chunks:
            assert chunk.metadata.get("genre") == "scholarly_prose"

    def test_positions_are_sequential(self, scholarly_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            scholarly_document,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
        )
        for granularity in ChunkGranularity:
            positions = [c.position for c in chunks if c.granularity == granularity]
            assert positions == sorted(positions)
            assert positions == list(range(len(positions)))
