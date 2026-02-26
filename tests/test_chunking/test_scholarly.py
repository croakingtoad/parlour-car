"""Tests for scholarly prose chunking strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from author_library.chunking.models import ChunkGranularity
from author_library.chunking.scholarly import ScholarlyProseStrategy, _split_text_at_sentences
from author_library.parsing.models import (
    DocumentMetadata,
    DocumentNode,
    NodeType,
    ParsedDocument,
)

if TYPE_CHECKING:
    pass


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


class TestChunkGranularityRatios:
    """Verify that multi-chapter documents produce correct granularity ratios.

    A 5000-word chapter should produce >3x more micro than macro chunks.
    This catches the bug where all three granularities had equal counts.
    """

    @staticmethod
    def _make_multi_chapter_doc(
        num_chapters: int = 5,
        paragraphs_per_chapter: int = 50,
        words_per_paragraph: int = 100,
    ) -> ParsedDocument:
        """Build a realistic multi-chapter scholarly document."""
        chapters = []
        for ch_idx in range(num_chapters):
            paras = []
            for p_idx in range(paragraphs_per_chapter):
                # Use varied sentence structure so sentence-splitting works
                sentences = []
                remaining = words_per_paragraph
                while remaining > 0:
                    slen = min(remaining, 15)
                    sentences.append(
                        " ".join(f"word{w}" for w in range(slen)) + "."
                    )
                    remaining -= slen
                text = " ".join(sentences)
                paras.append(DocumentNode(node_type=NodeType.PARAGRAPH, text=text))
            chapters.append(
                DocumentNode(
                    node_type=NodeType.CHAPTER,
                    children=paras,
                    metadata={"title": f"Chapter {ch_idx + 1}"},
                )
            )

        tree = DocumentNode(node_type=NodeType.BOOK, children=chapters)
        total = num_chapters * paragraphs_per_chapter * words_per_paragraph
        return ParsedDocument(
            source_path="/test/scholarly-multi.epub",
            format="epub",
            metadata=DocumentMetadata(
                title="Multi-Chapter Test",
                author="Test Author",
                word_count=total,
            ),
            tree=tree,
            raw_text="test",
        )

    def test_micro_exceeds_macro_by_3x(self) -> None:
        doc = self._make_multi_chapter_doc()
        strategy = ScholarlyProseStrategy()
        chunks = strategy.chunk(doc, work_id="test--ratios", source_class="primary")

        macro = sum(1 for c in chunks if c.granularity == ChunkGranularity.MACRO)
        meso = sum(1 for c in chunks if c.granularity == ChunkGranularity.MESO)
        micro = sum(1 for c in chunks if c.granularity == ChunkGranularity.MICRO)

        assert macro == 5, f"Expected 5 macro (one per chapter), got {macro}"
        assert micro > macro * 3, (
            f"Expected micro ({micro}) > 3 * macro ({macro * 3})"
        )
        assert meso > macro, f"Expected meso ({meso}) > macro ({macro})"

    def test_oversized_paragraph_gets_split(self) -> None:
        """A chapter with a single huge paragraph should still produce many chunks."""
        # Simulate a single 2000-word paragraph (e.g. from a collapsed div)
        sentences = [
            f"Sentence {i} with enough words to be realistic in scholarly prose."
            for i in range(150)
        ]
        huge_text = " ".join(sentences)

        chapter = DocumentNode(
            node_type=NodeType.CHAPTER,
            children=[DocumentNode(node_type=NodeType.PARAGRAPH, text=huge_text)],
            metadata={"title": "Dense Chapter"},
        )
        tree = DocumentNode(node_type=NodeType.BOOK, children=[chapter])
        doc = ParsedDocument(
            source_path="/test.epub",
            format="epub",
            metadata=DocumentMetadata(title="Test", word_count=len(huge_text.split())),
            tree=tree,
            raw_text=huge_text,
        )

        strategy = ScholarlyProseStrategy()
        chunks = strategy.chunk(doc, work_id="test--oversized", source_class="primary")

        macro = sum(1 for c in chunks if c.granularity == ChunkGranularity.MACRO)
        meso = sum(1 for c in chunks if c.granularity == ChunkGranularity.MESO)
        micro = sum(1 for c in chunks if c.granularity == ChunkGranularity.MICRO)

        assert macro == 1
        assert meso > 1, f"Oversized paragraph should produce >1 meso, got {meso}"
        assert micro > 1, f"Oversized paragraph should produce >1 micro, got {micro}"

    def test_parent_child_integrity_multi_chapter(self) -> None:
        doc = self._make_multi_chapter_doc(num_chapters=3, paragraphs_per_chapter=20)
        strategy = ScholarlyProseStrategy()
        chunks = strategy.chunk(doc, work_id="test--integrity", source_class="primary")

        macro_ids = {c.id for c in chunks if c.granularity == ChunkGranularity.MACRO}
        meso_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        micro_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MICRO]

        for meso in meso_chunks:
            assert meso.parent_chunk_id in macro_ids

        meso_ids = {c.id for c in meso_chunks}
        for micro in micro_chunks:
            assert micro.parent_chunk_id in meso_ids


class TestSplitTextAtSentences:
    def test_short_text_not_split(self) -> None:
        result = _split_text_at_sentences("Short text. Another sentence.", target_words=400)
        assert len(result) == 1

    def test_long_text_split(self) -> None:
        sentences = [
            f"Sentence number {i} with enough words to be a real sentence."
            for i in range(100)
        ]
        text = " ".join(sentences)
        result = _split_text_at_sentences(text, target_words=50)
        assert len(result) > 1
        # Rejoining should produce the same content
        reassembled = " ".join(result)
        assert reassembled.replace("  ", " ") == text.replace("  ", " ")

    def test_no_sentence_boundaries(self) -> None:
        text = "no periods or sentence breaks just a long run of words " * 50
        result = _split_text_at_sentences(text, target_words=20)
        assert len(result) == 1  # cannot split without sentence boundaries
