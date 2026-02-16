"""Tests for sermon chunking strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from author_library.chunking.models import ChunkGranularity
from author_library.chunking.sermon import SermonStrategy

if TYPE_CHECKING:
    from author_library.parsing.models import ParsedDocument


class TestSermonStrategy:
    def setup_method(self) -> None:
        self.strategy = SermonStrategy()

    def test_supported_genres(self) -> None:
        genres = self.strategy.supported_genres()
        assert "sermon" in genres
        assert "lecture" in genres
        assert "transcript" in genres

    def test_produces_all_granularities(self, sermon_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            sermon_document,
            work_id="guite--advent-word",
            source_class="primary",
        )
        granularities = {c.granularity for c in chunks}
        assert ChunkGranularity.MACRO in granularities
        assert ChunkGranularity.MESO in granularities
        assert ChunkGranularity.MICRO in granularities

    def test_macro_is_full_sermon(self, sermon_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            sermon_document,
            work_id="guite--advent-word",
            source_class="primary",
        )
        macro = [c for c in chunks if c.granularity == ChunkGranularity.MACRO]
        assert len(macro) == 1
        assert macro[0].chapter == "The Word Made Flesh"

    def test_meso_are_movements(self, sermon_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            sermon_document,
            work_id="guite--advent-word",
            source_class="primary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        # 4 sections (movements) in fixture
        assert len(meso) == 4

    def test_scripture_refs_detected(self, sermon_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            sermon_document,
            work_id="guite--advent-word",
            source_class="primary",
        )
        macro = next(c for c in chunks if c.granularity == ChunkGranularity.MACRO)
        refs = macro.metadata.get("scripture_refs", [])
        assert isinstance(refs, list)
        assert len(refs) >= 1
        # John 1:14 should be detected
        assert any("John 1:14" in str(ref) for ref in refs)

    def test_occasion_metadata(self, sermon_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            sermon_document,
            work_id="guite--advent-word",
            source_class="primary",
        )
        macro = next(c for c in chunks if c.granularity == ChunkGranularity.MACRO)
        assert macro.metadata.get("occasion") == "Advent Sunday"
        assert macro.metadata.get("venue") == "Girton College Chapel"
        assert macro.metadata.get("date") == "2018-12-02"

    def test_movement_number_metadata(self, sermon_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            sermon_document,
            work_id="guite--advent-word",
            source_class="primary",
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        numbers = [c.metadata.get("movement_number") for c in meso]
        assert numbers == [1, 2, 3, 4]

    def test_parent_child_relationships(self, sermon_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            sermon_document,
            work_id="guite--advent-word",
            source_class="primary",
        )
        macro_id = next(
            c.id for c in chunks if c.granularity == ChunkGranularity.MACRO
        )
        meso = [c for c in chunks if c.granularity == ChunkGranularity.MESO]
        for m in meso:
            assert m.parent_chunk_id == macro_id

    def test_micro_chunks_from_paragraphs(self, sermon_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            sermon_document,
            work_id="guite--advent-word",
            source_class="primary",
        )
        micro = [c for c in chunks if c.granularity == ChunkGranularity.MICRO]
        # Each movement has 1 paragraph → 4 micro chunks
        assert len(micro) == 4

    def test_genre_metadata(self, sermon_document: ParsedDocument) -> None:
        chunks = self.strategy.chunk(
            sermon_document,
            work_id="guite--advent-word",
            source_class="primary",
        )
        for chunk in chunks:
            assert chunk.metadata.get("genre") == "sermon"
