"""Tests for poetry chunking strategy.

Explicitly tests the NEVER-SPLIT-A-POEM rule.
"""

from __future__ import annotations

from author_library.chunking.models import ChunkGranularity
from author_library.chunking.poetry import PoetryStrategy
from author_library.parsing.models import (
    DocumentMetadata,
    DocumentNode,
    NodeType,
    ParsedDocument,
)


class TestPoetryStrategy:
    def setup_method(self) -> None:
        self.strategy = PoetryStrategy()

    def test_supported_genres(self) -> None:
        genres = self.strategy.supported_genres()
        assert "poetry" in genres
        assert "poems" in genres
        assert "verse" in genres

    def test_produces_meso_chunks(self, poetry_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            poetry_document,
            work_id="guite--sounding-the-seasons",
            source_class="primary",
        )
        meso_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        # Two poems in fixture
        assert len(meso_chunks) == 2

    def test_never_split_short_poem(self, poetry_document: ParsedDocument) -> None:
        """CRITICAL: Short poems must NEVER be split into micro chunks."""
        chunks = self.strategy.chunk(
            poetry_document,
            work_id="guite--sounding-the-seasons",
            source_class="primary",
        )
        # Find the short poem's meso chunk
        cathedral_chunks = [
            c for c in chunks
            if c.granularity == ChunkGranularity.MESO
            and c.metadata.get("poem_title") == "Cathedral"
        ]
        assert len(cathedral_chunks) == 1

        # No micro chunks should reference the short poem
        micro_for_cathedral = [
            c for c in chunks
            if c.granularity == ChunkGranularity.MICRO
            and c.parent_chunk_id == cathedral_chunks[0].id
        ]
        assert len(micro_for_cathedral) == 0

    def test_long_poem_gets_stanza_micros(self, poetry_document: ParsedDocument) -> None:
        """Poems > 40 lines should have stanza-level micro chunks."""
        chunks = self.strategy.chunk(
            poetry_document,
            work_id="guite--sounding-the-seasons",
            source_class="primary",
        )
        # Find the long poem's meso chunk
        long_poem_chunks = [
            c for c in chunks
            if c.granularity == ChunkGranularity.MESO
            and c.metadata.get("poem_title") == "The Long Pilgrimage"
        ]
        assert len(long_poem_chunks) == 1

        # Should have micro chunks (stanzas)
        micro_for_long = [
            c for c in chunks
            if c.granularity == ChunkGranularity.MICRO
            and c.parent_chunk_id == long_poem_chunks[0].id
        ]
        assert len(micro_for_long) >= 3  # 5 stanzas in fixture

    def test_poem_is_atomic_meso(self, poetry_document: ParsedDocument) -> None:
        """Each poem must be exactly one meso chunk, regardless of length."""
        chunks = self.strategy.chunk(
            poetry_document,
            work_id="guite--sounding-the-seasons",
            source_class="primary",
        )
        meso_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        # Exactly 2 meso chunks: one per poem
        assert len(meso_chunks) == 2

    def test_macro_chunks_from_sections(self, poetry_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            poetry_document,
            work_id="guite--sounding-the-seasons",
            source_class="primary",
        )
        macro_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MACRO]
        assert len(macro_chunks) >= 1

    def test_poem_metadata(self, poetry_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            poetry_document,
            work_id="guite--sounding-the-seasons",
            source_class="primary",
        )
        meso_chunks = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        for meso in meso_chunks:
            assert meso.metadata.get("genre") == "poetry"
            assert "first_line" in meso.metadata

    def test_epigraph_in_metadata(self, poetry_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            poetry_document,
            work_id="guite--sounding-the-seasons",
            source_class="primary",
        )
        cathedral_chunk = next(
            c for c in chunks
            if c.granularity == ChunkGranularity.MESO
            and c.metadata.get("poem_title") == "Cathedral"
        )
        assert cathedral_chunk.metadata.get("epigraph") == "For the builders"

    def test_stanza_metadata(self, poetry_document: ParsedDocument) -> None:
        """Stanza micro chunks should have stanza_number metadata."""
        chunks = self.strategy.chunk(
            poetry_document,
            work_id="guite--sounding-the-seasons",
            source_class="primary",
        )
        stanza_micros = [
            c for c in chunks
            if c.granularity == ChunkGranularity.MICRO
            and "stanza_number" in c.metadata
        ]
        assert len(stanza_micros) >= 3
        stanza_numbers = [c.metadata["stanza_number"] for c in stanza_micros]
        assert stanza_numbers == sorted(stanza_numbers)

    def test_never_split_poem_explicit(self) -> None:
        """Explicit test: a 20-line poem with stanzas must remain one meso chunk."""
        stanza1 = DocumentNode(
            node_type=NodeType.STANZA,
            text="Line 1\nLine 2\nLine 3\nLine 4\nLine 5",
        )
        stanza2 = DocumentNode(
            node_type=NodeType.STANZA,
            text="Line 6\nLine 7\nLine 8\nLine 9\nLine 10",
        )
        stanza3 = DocumentNode(
            node_type=NodeType.STANZA,
            text="Line 11\nLine 12\nLine 13\nLine 14\nLine 15",
        )
        stanza4 = DocumentNode(
            node_type=NodeType.STANZA,
            text="Line 16\nLine 17\nLine 18\nLine 19\nLine 20",
        )

        poem = DocumentNode(
            node_type=NodeType.POEM,
            children=[stanza1, stanza2, stanza3, stanza4],
            metadata={"title": "Short Poem"},
        )
        tree = DocumentNode(node_type=NodeType.BOOK, children=[poem])
        doc = ParsedDocument(
            source_path="/test.txt",
            format="txt",
            metadata=DocumentMetadata(title="Test", word_count=20),
            tree=tree,
        )

        chunks = self.strategy.chunk(doc, work_id="test", source_class="primary")
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        micro = [c for c in chunks if c.granularity == ChunkGranularity.MICRO]

        assert len(meso) == 1  # poem is ONE meso chunk
        assert len(micro) == 0  # no micro chunks (< 40 lines)

    def test_source_class_propagates(self, poetry_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            poetry_document,
            work_id="guite--sounding-the-seasons",
            source_class="primary",
        )
        for chunk in chunks:
            assert chunk.source_class == "primary"
