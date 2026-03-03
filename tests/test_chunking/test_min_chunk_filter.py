"""Tests for the minimum chunk size filter.

Verifies that micro/nano chunks under the threshold are removed,
while macro and meso chunks are always preserved regardless of size.
"""

from __future__ import annotations

from author_library.chunking.models import Chunk, ChunkGranularity
from author_library.chunking.scholarly import MIN_CHUNK_CHARS, filter_min_chunk_size


class TestFilterMinChunkSize:
    """Tests for filter_min_chunk_size."""

    @staticmethod
    def _make_chunk(
        text: str,
        granularity: ChunkGranularity = ChunkGranularity.MICRO,
        **kwargs: object,
    ) -> Chunk:
        return Chunk(
            text=text,
            granularity=granularity,
            work_id="test--filter",
            source_class="primary",
            position=0,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_tiny_micro_removed(self) -> None:
        """Micro chunks under the threshold should be filtered out."""
        chunks = [
            self._make_chunk("I"),
            self._make_chunk("Dante"),
            self._make_chunk("Bible"),
            self._make_chunk("42-44, 94"),
            self._make_chunk(
                "This is a substantial paragraph about the nature of poetic imagination."
            ),
        ]
        result = filter_min_chunk_size(chunks)
        assert len(result) == 1
        assert "substantial paragraph" in result[0].text

    def test_macro_never_filtered(self) -> None:
        """Macro chunks are never filtered regardless of text length."""
        chunks = [
            self._make_chunk("Short macro", granularity=ChunkGranularity.MACRO),
        ]
        result = filter_min_chunk_size(chunks)
        assert len(result) == 1

    def test_meso_never_filtered(self) -> None:
        """Meso chunks are never filtered regardless of text length."""
        chunks = [
            self._make_chunk("Short meso", granularity=ChunkGranularity.MESO),
        ]
        result = filter_min_chunk_size(chunks)
        assert len(result) == 1

    def test_nano_filtered(self) -> None:
        """Nano chunks under the threshold should be filtered out."""
        chunks = [
            self._make_chunk("tiny", granularity=ChunkGranularity.NANO),
        ]
        result = filter_min_chunk_size(chunks)
        assert len(result) == 0

    def test_empty_input(self) -> None:
        result = filter_min_chunk_size([])
        assert result == []

    def test_all_above_threshold(self) -> None:
        """Chunks all above the threshold should pass through unchanged."""
        chunks = [
            self._make_chunk(
                "This is a sufficiently long micro chunk about poetic theology "
                "and the imagination."
            ),
            self._make_chunk(
                "Another reasonably sized chunk discussing Coleridge and the "
                "romantic tradition of creative perception."
            ),
        ]
        result = filter_min_chunk_size(chunks)
        assert len(result) == 2

    def test_custom_threshold(self) -> None:
        """Custom min_chars threshold should work."""
        chunks = [
            self._make_chunk("Medium length text here"),  # 23 chars
        ]
        result = filter_min_chunk_size(chunks, min_chars=10)
        assert len(result) == 1
        result = filter_min_chunk_size(chunks, min_chars=100)
        assert len(result) == 0

    def test_default_threshold_is_50(self) -> None:
        assert MIN_CHUNK_CHARS == 50

    def test_whitespace_only_filtered(self) -> None:
        """Chunks with only whitespace should be filtered."""
        chunks = [
            self._make_chunk("   \n\t   "),
        ]
        result = filter_min_chunk_size(chunks)
        assert len(result) == 0

    def test_mixed_granularities(self) -> None:
        """A mix of granularities: only micro/nano under threshold are removed."""
        chunks = [
            self._make_chunk("Short macro", granularity=ChunkGranularity.MACRO),
            self._make_chunk("Short meso", granularity=ChunkGranularity.MESO),
            self._make_chunk("I", granularity=ChunkGranularity.MICRO),
            self._make_chunk(
                "A proper micro chunk with enough content to pass the filter easily.",
                granularity=ChunkGranularity.MICRO,
            ),
            self._make_chunk("42", granularity=ChunkGranularity.NANO),
        ]
        result = filter_min_chunk_size(chunks)
        assert len(result) == 3  # macro + meso + the long micro
        grans = [c.granularity for c in result]
        assert ChunkGranularity.MACRO in grans
        assert ChunkGranularity.MESO in grans
        assert ChunkGranularity.MICRO in grans
