"""Tests for M2: Confidence scoring for surfaced connections."""

from __future__ import annotations

import pytest

from author_library.surfacing.confidence import (
    CONFIDENCE_LABELS,
    CONFIDENCE_LABEL_VARIANTS,
    ConfidenceLevel,
    ScoredConnection,
    classify_batch,
    classify_confidence,
)
from author_library.surfacing.related_content import ConnectionType, RelatedItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_item(
    *,
    connection_type: ConnectionType = ConnectionType.VECTOR_SIMILARITY,
    relevance_score: float = 0.5,
    metadata: dict | None = None,
) -> RelatedItem:
    return RelatedItem(
        chunk_id="chunk-1",
        work_id="work-1",
        text="Test passage",
        source_class="primary",
        granularity="meso",
        connection_type=connection_type,
        relevance_score=relevance_score,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# ConfidenceLevel enum tests
# ---------------------------------------------------------------------------


class TestConfidenceLevel:
    def test_enum_values(self):
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.LOW == "low"

    def test_all_levels_have_labels(self):
        for level in ConfidenceLevel:
            assert level in CONFIDENCE_LABELS
            assert isinstance(CONFIDENCE_LABELS[level], str)

    def test_all_levels_have_label_variants(self):
        for level in ConfidenceLevel:
            assert level in CONFIDENCE_LABEL_VARIANTS
            variants = CONFIDENCE_LABEL_VARIANTS[level]
            assert len(variants) >= 1
            # Default label should be first variant
            assert CONFIDENCE_LABELS[level] == variants[0]


# ---------------------------------------------------------------------------
# Passage link classification
# ---------------------------------------------------------------------------


class TestPassageLinkClassification:
    def test_high_confidence_metadata(self):
        item = _make_item(
            connection_type=ConnectionType.PASSAGE_LINK,
            relevance_score=0.6,
            metadata={"confidence": "high"},
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_high_score(self):
        item = _make_item(
            connection_type=ConnectionType.PASSAGE_LINK,
            relevance_score=0.85,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_medium_confidence_metadata(self):
        item = _make_item(
            connection_type=ConnectionType.PASSAGE_LINK,
            relevance_score=0.3,
            metadata={"confidence": "medium"},
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.MEDIUM

    def test_medium_score(self):
        item = _make_item(
            connection_type=ConnectionType.PASSAGE_LINK,
            relevance_score=0.65,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.MEDIUM

    def test_low(self):
        item = _make_item(
            connection_type=ConnectionType.PASSAGE_LINK,
            relevance_score=0.3,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Personal reflection classification
# ---------------------------------------------------------------------------


class TestPersonalReflectionClassification:
    def test_high_with_target_type(self):
        item = _make_item(
            connection_type=ConnectionType.PERSONAL_REFLECTION,
            relevance_score=0.3,
            metadata={"target_type": "work"},
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_high_score(self):
        item = _make_item(
            connection_type=ConnectionType.PERSONAL_REFLECTION,
            relevance_score=0.75,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_medium(self):
        item = _make_item(
            connection_type=ConnectionType.PERSONAL_REFLECTION,
            relevance_score=0.5,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.MEDIUM

    def test_low(self):
        item = _make_item(
            connection_type=ConnectionType.PERSONAL_REFLECTION,
            relevance_score=0.2,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Thematic parallel classification
# ---------------------------------------------------------------------------


class TestThematicParallelClassification:
    def test_medium_high_score(self):
        """Thematic parallels cap at MEDIUM — never HIGH."""
        item = _make_item(
            connection_type=ConnectionType.THEMATIC_PARALLEL,
            relevance_score=0.8,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.MEDIUM

    def test_low(self):
        item = _make_item(
            connection_type=ConnectionType.THEMATIC_PARALLEL,
            relevance_score=0.5,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Vector similarity classification
# ---------------------------------------------------------------------------


class TestVectorSimilarityClassification:
    def test_high(self):
        item = _make_item(
            connection_type=ConnectionType.VECTOR_SIMILARITY,
            relevance_score=0.85,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_medium(self):
        item = _make_item(
            connection_type=ConnectionType.VECTOR_SIMILARITY,
            relevance_score=0.65,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.MEDIUM

    def test_low(self):
        item = _make_item(
            connection_type=ConnectionType.VECTOR_SIMILARITY,
            relevance_score=0.3,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Temporal proximity / fallback classification
# ---------------------------------------------------------------------------


class TestTemporalProximityClassification:
    def test_high(self):
        item = _make_item(
            connection_type=ConnectionType.TEMPORAL_PROXIMITY,
            relevance_score=0.9,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_medium(self):
        item = _make_item(
            connection_type=ConnectionType.TEMPORAL_PROXIMITY,
            relevance_score=0.6,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.MEDIUM

    def test_low(self):
        item = _make_item(
            connection_type=ConnectionType.TEMPORAL_PROXIMITY,
            relevance_score=0.3,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# ScoredConnection structure
# ---------------------------------------------------------------------------


class TestScoredConnection:
    def test_fields(self):
        item = _make_item(relevance_score=0.85)
        sc = classify_confidence(item)
        assert sc.item is item
        assert sc.raw_score == 0.85
        assert isinstance(sc.label, str)
        assert len(sc.label) > 0

    def test_label_matches_level(self):
        item = _make_item(relevance_score=0.9)
        sc = classify_confidence(item)
        assert sc.label == CONFIDENCE_LABELS[sc.confidence_level]

    def test_frozen(self):
        item = _make_item(relevance_score=0.9)
        sc = classify_confidence(item)
        with pytest.raises(AttributeError):
            sc.confidence_level = ConfidenceLevel.LOW  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------


class TestClassifyBatch:
    def test_empty(self):
        assert classify_batch([]) == []

    def test_sorted_by_level_then_score(self):
        items = [
            _make_item(connection_type=ConnectionType.VECTOR_SIMILARITY, relevance_score=0.3),
            _make_item(connection_type=ConnectionType.VECTOR_SIMILARITY, relevance_score=0.9),
            _make_item(connection_type=ConnectionType.VECTOR_SIMILARITY, relevance_score=0.6),
        ]
        scored = classify_batch(items)

        # HIGH (0.9) first, then MEDIUM (0.6), then LOW (0.3)
        assert scored[0].confidence_level == ConfidenceLevel.HIGH
        assert scored[1].confidence_level == ConfidenceLevel.MEDIUM
        assert scored[2].confidence_level == ConfidenceLevel.LOW

    def test_within_level_sorted_by_score_descending(self):
        items = [
            _make_item(connection_type=ConnectionType.VECTOR_SIMILARITY, relevance_score=0.55),
            _make_item(connection_type=ConnectionType.VECTOR_SIMILARITY, relevance_score=0.75),
            _make_item(connection_type=ConnectionType.VECTOR_SIMILARITY, relevance_score=0.65),
        ]
        scored = classify_batch(items)

        # All MEDIUM, sorted by score descending
        assert all(s.confidence_level == ConfidenceLevel.MEDIUM for s in scored)
        scores = [s.raw_score for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_mixed_types(self):
        items = [
            _make_item(
                connection_type=ConnectionType.THEMATIC_PARALLEL,
                relevance_score=0.5,
            ),
            _make_item(
                connection_type=ConnectionType.PASSAGE_LINK,
                relevance_score=0.9,
                metadata={"confidence": "high"},
            ),
            _make_item(
                connection_type=ConnectionType.PERSONAL_REFLECTION,
                relevance_score=0.45,
            ),
        ]
        scored = classify_batch(items)

        # Passage link (HIGH) first, then personal (MEDIUM), then thematic (LOW)
        assert scored[0].confidence_level == ConfidenceLevel.HIGH
        assert scored[1].confidence_level == ConfidenceLevel.MEDIUM
        assert scored[2].confidence_level == ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Boundary value tests
# ---------------------------------------------------------------------------


class TestBoundaryValues:
    def test_vector_at_exactly_0_8(self):
        """Score exactly at threshold — not > 0.8, so MEDIUM."""
        item = _make_item(
            connection_type=ConnectionType.VECTOR_SIMILARITY,
            relevance_score=0.8,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.MEDIUM

    def test_vector_at_exactly_0_5(self):
        """Score exactly at threshold — not > 0.5, so LOW."""
        item = _make_item(
            connection_type=ConnectionType.VECTOR_SIMILARITY,
            relevance_score=0.5,
        )
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.LOW

    def test_zero_score(self):
        item = _make_item(relevance_score=0.0)
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.LOW

    def test_perfect_score(self):
        item = _make_item(relevance_score=1.0)
        result = classify_confidence(item)
        assert result.confidence_level == ConfidenceLevel.HIGH
