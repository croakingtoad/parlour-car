"""Tests for M3: Surfacing response format."""

from __future__ import annotations

import json

import pytest

from author_library.surfacing.confidence import ConfidenceLevel
from author_library.surfacing.related_content import ConnectionType, RelatedItem
from author_library.surfacing.response_format import (
    FormattedSurfacingItem,
    SurfacingResponse,
    format_surfacing_results,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    *,
    chunk_id: str = "chunk-1",
    work_id: str = "guite--faith-hope-poetry",
    text: str = "Imagination as a faculty of perception",
    source_class: str = "primary",
    connection_type: ConnectionType = ConnectionType.PASSAGE_LINK,
    relevance_score: float = 0.9,
    metadata: dict | None = None,
) -> RelatedItem:
    return RelatedItem(
        chunk_id=chunk_id,
        work_id=work_id,
        text=text,
        source_class=source_class,
        granularity="meso",
        connection_type=connection_type,
        relevance_score=relevance_score,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# FormattedSurfacingItem tests
# ---------------------------------------------------------------------------


class TestFormattedSurfacingItem:
    def test_creation(self):
        item = FormattedSurfacingItem(
            chunk_id="c-1",
            work_id="w-1",
            title="Faith, Hope and Poetry",
            source="Malcolm Guite, Faith, Hope and Poetry",
            excerpt="Imagination is...",
            confidence_level="high",
            confidence_label="This directly engages with",
            connection_type="passage_link",
            source_class="primary",
            metadata={"relevance_score": 0.9},
        )
        assert item.chunk_id == "c-1"
        assert item.confidence_level == "high"
        assert item.confidence_label == "This directly engages with"

    def test_frozen(self):
        item = FormattedSurfacingItem(
            chunk_id="c-1",
            work_id="w-1",
            title="T",
            source="S",
            excerpt="E",
            confidence_level="high",
            confidence_label="L",
            connection_type="passage_link",
            source_class="primary",
        )
        with pytest.raises(AttributeError):
            item.title = "New Title"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SurfacingResponse tests
# ---------------------------------------------------------------------------


class TestSurfacingResponse:
    def _make_response(self) -> SurfacingResponse:
        high_item = FormattedSurfacingItem(
            chunk_id="h-1", work_id="w-1", title="High",
            source="Source A", excerpt="High excerpt",
            confidence_level="high", confidence_label="This directly engages with",
            connection_type="passage_link", source_class="primary",
        )
        medium_item = FormattedSurfacingItem(
            chunk_id="m-1", work_id="w-2", title="Medium",
            source="Source B", excerpt="Medium excerpt",
            confidence_level="medium", confidence_label="This appears to connect to",
            connection_type="thematic_parallel", source_class="primary",
        )
        return SurfacingResponse(
            context_chunk_id="ctx-1",
            context_work_id="work-ctx",
            high_confidence=[high_item],
            medium_confidence=[medium_item],
            low_confidence=[],
            strategies_used=["passage_links", "thematic_parallels"],
            total_results=2,
        )

    def test_to_dict_structure(self):
        resp = self._make_response()
        d = resp.to_dict()

        assert d["context"]["chunk_id"] == "ctx-1"
        assert d["context"]["work_id"] == "work-ctx"
        assert len(d["results"]["high"]) == 1
        assert len(d["results"]["medium"]) == 1
        assert len(d["results"]["low"]) == 0
        assert d["total_results"] == 2
        assert d["strategies_used"] == ["passage_links", "thematic_parallels"]

    def test_to_dict_item_fields(self):
        resp = self._make_response()
        d = resp.to_dict()
        high = d["results"]["high"][0]

        assert high["chunk_id"] == "h-1"
        assert high["title"] == "High"
        assert high["confidence_level"] == "high"
        assert high["confidence_label"] == "This directly engages with"
        assert high["connection_type"] == "passage_link"

    def test_to_json(self):
        resp = self._make_response()
        j = resp.to_json()
        parsed = json.loads(j)
        assert parsed["total_results"] == 2

    def test_to_json_indent(self):
        resp = self._make_response()
        j = resp.to_json(indent=4)
        assert "    " in j  # 4-space indentation

    def test_empty_response(self):
        resp = SurfacingResponse(
            context_chunk_id="",
            context_work_id="",
            high_confidence=[],
            medium_confidence=[],
            low_confidence=[],
            strategies_used=[],
            total_results=0,
        )
        d = resp.to_dict()
        assert d["total_results"] == 0
        assert d["results"]["high"] == []


# ---------------------------------------------------------------------------
# format_surfacing_results tests
# ---------------------------------------------------------------------------


class TestFormatSurfacingResults:
    def test_empty_input(self):
        result = format_surfacing_results([])
        assert result.total_results == 0
        assert result.high_confidence == []
        assert result.medium_confidence == []
        assert result.low_confidence == []

    def test_groups_by_confidence(self):
        items = [
            _make_item(
                chunk_id="high-1",
                connection_type=ConnectionType.VECTOR_SIMILARITY,
                relevance_score=0.9,
            ),
            _make_item(
                chunk_id="med-1",
                connection_type=ConnectionType.VECTOR_SIMILARITY,
                relevance_score=0.6,
            ),
            _make_item(
                chunk_id="low-1",
                connection_type=ConnectionType.VECTOR_SIMILARITY,
                relevance_score=0.3,
            ),
        ]
        result = format_surfacing_results(items)

        assert len(result.high_confidence) == 1
        assert len(result.medium_confidence) == 1
        assert len(result.low_confidence) == 1
        assert result.total_results == 3

    def test_context_ids_passed_through(self):
        result = format_surfacing_results(
            [],
            context_chunk_id="ctx-chunk",
            context_work_id="ctx-work",
        )
        assert result.context_chunk_id == "ctx-chunk"
        assert result.context_work_id == "ctx-work"

    def test_strategies_passed_through(self):
        result = format_surfacing_results(
            [],
            strategies_used=["passage_links", "vector_similarity"],
        )
        assert result.strategies_used == ["passage_links", "vector_similarity"]

    def test_strategies_default_empty(self):
        result = format_surfacing_results([])
        assert result.strategies_used == []

    def test_max_per_level(self):
        items = [
            _make_item(
                chunk_id=f"high-{i}",
                connection_type=ConnectionType.VECTOR_SIMILARITY,
                relevance_score=0.9 - (i * 0.01),
            )
            for i in range(5)
        ]
        result = format_surfacing_results(items, max_per_level=2)

        assert len(result.high_confidence) == 2
        assert result.total_results == 2

    def test_max_per_level_applies_per_group(self):
        items = [
            _make_item(chunk_id="h-1", relevance_score=0.9,
                       connection_type=ConnectionType.VECTOR_SIMILARITY),
            _make_item(chunk_id="h-2", relevance_score=0.85,
                       connection_type=ConnectionType.VECTOR_SIMILARITY),
            _make_item(chunk_id="m-1", relevance_score=0.6,
                       connection_type=ConnectionType.VECTOR_SIMILARITY),
            _make_item(chunk_id="m-2", relevance_score=0.55,
                       connection_type=ConnectionType.VECTOR_SIMILARITY),
            _make_item(chunk_id="l-1", relevance_score=0.3,
                       connection_type=ConnectionType.VECTOR_SIMILARITY),
            _make_item(chunk_id="l-2", relevance_score=0.25,
                       connection_type=ConnectionType.VECTOR_SIMILARITY),
        ]
        result = format_surfacing_results(items, max_per_level=1)

        assert len(result.high_confidence) == 1
        assert len(result.medium_confidence) == 1
        assert len(result.low_confidence) == 1
        assert result.total_results == 3


# ---------------------------------------------------------------------------
# _format_item behavior tests (via format_surfacing_results)
# ---------------------------------------------------------------------------


class TestItemFormatting:
    def test_title_from_metadata(self):
        items = [
            _make_item(
                relevance_score=0.9,
                connection_type=ConnectionType.VECTOR_SIMILARITY,
                metadata={"work_title": "Faith, Hope and Poetry"},
            ),
        ]
        result = format_surfacing_results(items)
        assert result.high_confidence[0].title == "Faith, Hope and Poetry"

    def test_title_from_work_id(self):
        """When no work_title in metadata, derive from work_id."""
        items = [
            _make_item(
                work_id="guite--faith-hope-poetry",
                relevance_score=0.9,
                connection_type=ConnectionType.VECTOR_SIMILARITY,
            ),
        ]
        result = format_surfacing_results(items)
        title = result.high_confidence[0].title
        # work_id formatted: hyphens → spaces, -- → em dash, title case
        assert "Guite" in title

    def test_excerpt_preserves_full_text(self):
        """Full text is preserved for drag-and-drop; sidebar CSS handles display truncation."""
        long_text = "A" * 500
        items = [
            _make_item(
                text=long_text,
                relevance_score=0.9,
                connection_type=ConnectionType.VECTOR_SIMILARITY,
            ),
        ]
        result = format_surfacing_results(items)
        excerpt = result.high_confidence[0].excerpt
        assert len(excerpt) == 500

    def test_excerpt_no_truncation_for_short_text(self):
        items = [
            _make_item(
                text="Short passage",
                relevance_score=0.9,
                connection_type=ConnectionType.VECTOR_SIMILARITY,
            ),
        ]
        result = format_surfacing_results(items)
        excerpt = result.high_confidence[0].excerpt
        assert excerpt == "Short passage"
        assert not excerpt.endswith("...")

    def test_confidence_level_in_formatted_item(self):
        items = [
            _make_item(
                relevance_score=0.9,
                connection_type=ConnectionType.VECTOR_SIMILARITY,
            ),
        ]
        result = format_surfacing_results(items)
        item = result.high_confidence[0]
        assert item.confidence_level == ConfidenceLevel.HIGH.value

    def test_metadata_fields(self):
        items = [
            _make_item(
                relevance_score=0.9,
                connection_type=ConnectionType.VECTOR_SIMILARITY,
                metadata={
                    "link_type": "explicit_citation",
                    "theme": "imagination",
                    "evidence": "Guite argues...",
                    "date_created": "2025-06-15",
                },
            ),
        ]
        result = format_surfacing_results(items)
        meta = result.high_confidence[0].metadata
        assert meta["link_type"] == "explicit_citation"
        assert meta["theme"] == "imagination"
        assert meta["evidence"] == "Guite argues..."
        assert meta["date_created"] == "2025-06-15"


# ---------------------------------------------------------------------------
# Source attribution tests (via format_surfacing_results)
# ---------------------------------------------------------------------------


class TestSourceAttribution:
    def test_personal_with_date(self):
        items = [
            _make_item(
                source_class="personal",
                relevance_score=0.9,
                connection_type=ConnectionType.PERSONAL_REFLECTION,
                metadata={"date_created": "2026-01-15", "target_type": "work"},
            ),
        ]
        result = format_surfacing_results(items)
        source = result.high_confidence[0].source
        assert "Your reflection from 2026-01-15" == source

    def test_personal_without_date(self):
        items = [
            _make_item(
                source_class="personal",
                relevance_score=0.9,
                connection_type=ConnectionType.PERSONAL_REFLECTION,
                metadata={"target_type": "work"},
            ),
        ]
        result = format_surfacing_results(items)
        assert result.high_confidence[0].source == "Your personal reflection"

    def test_author_and_title(self):
        items = [
            _make_item(
                relevance_score=0.9,
                connection_type=ConnectionType.VECTOR_SIMILARITY,
                metadata={"author": "Malcolm Guite", "work_title": "Faith, Hope and Poetry"},
            ),
        ]
        result = format_surfacing_results(items)
        assert result.high_confidence[0].source == "Malcolm Guite, Faith, Hope and Poetry"

    def test_title_only(self):
        items = [
            _make_item(
                relevance_score=0.9,
                connection_type=ConnectionType.VECTOR_SIMILARITY,
                metadata={"work_title": "Faith, Hope and Poetry"},
            ),
        ]
        result = format_surfacing_results(items)
        assert result.high_confidence[0].source == "Faith, Hope and Poetry"

    def test_author_only(self):
        items = [
            _make_item(
                relevance_score=0.9,
                connection_type=ConnectionType.VECTOR_SIMILARITY,
                metadata={"author": "Malcolm Guite"},
            ),
        ]
        result = format_surfacing_results(items)
        assert result.high_confidence[0].source == "Malcolm Guite"

    def test_fallback_to_work_id(self):
        items = [
            _make_item(
                work_id="guite--faith-hope",
                relevance_score=0.9,
                connection_type=ConnectionType.VECTOR_SIMILARITY,
            ),
        ]
        result = format_surfacing_results(items)
        assert result.high_confidence[0].source == "guite--faith-hope"
