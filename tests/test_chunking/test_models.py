"""Tests for chunk models and granularity enum."""

from __future__ import annotations

from author_library.chunking.models import GRANULARITY_BOUNDS, Chunk, ChunkGranularity


class TestChunkGranularity:
    def test_enum_values(self) -> None:
        assert ChunkGranularity.MACRO == "macro"
        assert ChunkGranularity.MESO == "meso"
        assert ChunkGranularity.MICRO == "micro"

    def test_granularity_bounds_defined(self) -> None:
        for g in ChunkGranularity:
            assert g in GRANULARITY_BOUNDS
            lo, hi = GRANULARITY_BOUNDS[g]
            assert lo < hi

    def test_macro_bounds(self) -> None:
        lo, hi = GRANULARITY_BOUNDS[ChunkGranularity.MACRO]
        assert lo == 500
        assert hi == 1500

    def test_meso_bounds(self) -> None:
        lo, hi = GRANULARITY_BOUNDS[ChunkGranularity.MESO]
        assert lo == 150
        assert hi == 500

    def test_micro_bounds(self) -> None:
        lo, hi = GRANULARITY_BOUNDS[ChunkGranularity.MICRO]
        assert lo == 30
        assert hi == 200


class TestChunk:
    def test_create_chunk(self) -> None:
        chunk = Chunk(
            text="This is a test chunk with some words.",
            granularity=ChunkGranularity.MESO,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
            position=0,
        )
        assert chunk.text == "This is a test chunk with some words."
        assert chunk.granularity == ChunkGranularity.MESO
        assert chunk.work_id == "guite--faith-hope-poetry"
        assert chunk.source_class == "primary"
        assert chunk.position == 0
        assert chunk.id  # auto-generated
        assert chunk.annotation is None
        assert chunk.chapter is None
        assert chunk.section is None
        assert chunk.parent_chunk_id is None
        assert chunk.metadata == {}

    def test_word_count_property(self) -> None:
        chunk = Chunk(
            text="one two three four five",
            granularity=ChunkGranularity.MICRO,
            work_id="test",
            source_class="primary",
            position=0,
        )
        assert chunk.word_count == 5

    def test_annotated_text_without_annotation(self) -> None:
        chunk = Chunk(
            text="The body text.",
            granularity=ChunkGranularity.MICRO,
            work_id="test",
            source_class="primary",
            position=0,
        )
        assert chunk.annotated_text == "The body text."

    def test_annotated_text_with_annotation(self) -> None:
        chunk = Chunk(
            text="The body text.",
            granularity=ChunkGranularity.MICRO,
            work_id="test",
            source_class="primary",
            position=0,
            annotation="[PRIMARY] Context info.",
        )
        assert chunk.annotated_text == "[PRIMARY] Context info.\n\nThe body text."

    def test_unique_ids(self) -> None:
        c1 = Chunk(
            text="a", granularity=ChunkGranularity.MICRO,
            work_id="t", source_class="p", position=0,
        )
        c2 = Chunk(
            text="b", granularity=ChunkGranularity.MICRO,
            work_id="t", source_class="p", position=1,
        )
        assert c1.id != c2.id

    def test_metadata_with_genre(self) -> None:
        chunk = Chunk(
            text="text",
            granularity=ChunkGranularity.MESO,
            work_id="test",
            source_class="primary",
            position=0,
            metadata={"genre": "poetry", "poem_title": "Cathedral"},
        )
        assert chunk.metadata["genre"] == "poetry"
        assert chunk.metadata["poem_title"] == "Cathedral"

    def test_parent_chunk_relationship(self) -> None:
        parent = Chunk(
            text="parent",
            granularity=ChunkGranularity.MACRO,
            work_id="test",
            source_class="primary",
            position=0,
        )
        child = Chunk(
            text="child",
            granularity=ChunkGranularity.MESO,
            work_id="test",
            source_class="primary",
            position=0,
            parent_chunk_id=parent.id,
        )
        assert child.parent_chunk_id == parent.id
