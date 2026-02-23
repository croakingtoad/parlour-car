"""Tests for Nano granularity (A4) and pass_number tracking (A7).

Covers:
- ChunkGranularity.NANO enum validity
- Chunk model raw_content and raw_content_window fields
- GRANULARITY_BOUNDS for NANO tier
- pass_number defaults and increment behavior
"""

from __future__ import annotations

from author_library.chunking.models import Chunk, ChunkGranularity, GRANULARITY_BOUNDS


# ---------------------------------------------------------------------------
# ChunkGranularity.NANO
# ---------------------------------------------------------------------------


class TestNanoGranularity:
    def test_nano_is_valid_enum_value(self) -> None:
        assert ChunkGranularity.NANO == "nano"

    def test_nano_in_all_granularities(self) -> None:
        all_granularities = list(ChunkGranularity)
        assert ChunkGranularity.NANO in all_granularities

    def test_four_granularity_tiers(self) -> None:
        assert len(ChunkGranularity) == 4

    def test_nano_bounds(self) -> None:
        lower, upper = GRANULARITY_BOUNDS[ChunkGranularity.NANO]
        assert lower == 1
        assert upper == 50

    def test_all_granularities_have_bounds(self) -> None:
        for g in ChunkGranularity:
            assert g in GRANULARITY_BOUNDS
            lower, upper = GRANULARITY_BOUNDS[g]
            assert lower > 0
            assert upper > lower


# ---------------------------------------------------------------------------
# Chunk raw_content and raw_content_window
# ---------------------------------------------------------------------------


class TestChunkRawContentFields:
    def test_raw_content_default_none(self) -> None:
        chunk = Chunk(
            text="A brief capture moment.",
            granularity=ChunkGranularity.NANO,
            work_id="test--raw-content",
            source_class="personal",
            position=0,
        )
        assert chunk.raw_content is None

    def test_raw_content_set(self) -> None:
        chunk = Chunk(
            text="Processed text of the capture.",
            granularity=ChunkGranularity.NANO,
            work_id="test--raw-content",
            source_class="personal",
            position=0,
            raw_content="[00:15:30] Original raw capture text here",
        )
        assert chunk.raw_content == "[00:15:30] Original raw capture text here"

    def test_raw_content_window_default_none(self) -> None:
        chunk = Chunk(
            text="A capture.",
            granularity=ChunkGranularity.NANO,
            work_id="test--window",
            source_class="personal",
            position=0,
        )
        assert chunk.raw_content_window is None

    def test_raw_content_window_set(self) -> None:
        chunk = Chunk(
            text="A capture.",
            granularity=ChunkGranularity.NANO,
            work_id="test--window",
            source_class="personal",
            position=0,
            raw_content_window="meso-chunk-abc",
        )
        assert chunk.raw_content_window == "meso-chunk-abc"

    def test_nano_chunk_with_all_fields(self) -> None:
        chunk = Chunk(
            text="Brief moment captured.",
            granularity=ChunkGranularity.NANO,
            work_id="guite--imagination-conversation",
            source_class="personal",
            position=3,
            raw_content="[00:15:30] He said something about imagination being a living power",
            raw_content_window="meso-chunk-xyz",
            metadata={"timestamp": "00:15:30", "speaker": "Malcolm Guite"},
        )
        assert chunk.granularity == ChunkGranularity.NANO
        assert chunk.raw_content is not None
        assert chunk.raw_content_window is not None
        assert chunk.word_count <= 50  # within NANO bounds


# ---------------------------------------------------------------------------
# pass_number tracking (A7)
# ---------------------------------------------------------------------------


class TestPassNumber:
    def test_pass_number_defaults_to_1(self) -> None:
        chunk = Chunk(
            text="First engagement with this text.",
            granularity=ChunkGranularity.MESO,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
            position=0,
        )
        assert chunk.pass_number == 1

    def test_pass_number_can_be_set(self) -> None:
        chunk = Chunk(
            text="Second pass through this text.",
            granularity=ChunkGranularity.MESO,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
            position=0,
            pass_number=2,
        )
        assert chunk.pass_number == 2

    def test_pass_number_increment(self) -> None:
        """Simulates re-ingestion: pass_number should increment."""
        chunk_pass_1 = Chunk(
            text="The imagination is a living power.",
            granularity=ChunkGranularity.MESO,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
            position=0,
            pass_number=1,
        )
        chunk_pass_2 = Chunk(
            text="The imagination is a living power.",
            granularity=ChunkGranularity.MESO,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
            position=0,
            pass_number=chunk_pass_1.pass_number + 1,
        )
        assert chunk_pass_2.pass_number == 2

    def test_pass_number_on_nano_chunk(self) -> None:
        chunk = Chunk(
            text="Brief capture moment.",
            granularity=ChunkGranularity.NANO,
            work_id="test--pass",
            source_class="personal",
            position=0,
            pass_number=3,
        )
        assert chunk.pass_number == 3

    def test_source_class_personal_on_chunk(self) -> None:
        """Verify chunk model accepts 'personal' as source_class."""
        chunk = Chunk(
            text="My personal reflection on this passage.",
            granularity=ChunkGranularity.MESO,
            work_id="marty--reflection",
            source_class="personal",
            position=0,
        )
        assert chunk.source_class == "personal"
