"""Tests for Reciprocal Rank Fusion (RRF).

RRF is deterministic — no LLM or database needed.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from author_library.retrieval.fusion import reciprocal_rank_fusion
from author_library.retrieval.models import RetrievalResult


def _make_result(
    chunk_id_hex: str | None = None,
    score: float = 0.9,
    source: str = "vector",
    work_id: str = "test--work",
    source_class: str = "primary",
) -> RetrievalResult:
    """Helper to create test RetrievalResult."""
    from uuid import UUID

    cid = UUID(chunk_id_hex) if chunk_id_hex else uuid4()
    return RetrievalResult(
        chunk_id=cid,
        work_id=work_id,
        text="test text",
        score=score,
        granularity="meso",
        source_class=source_class,
        source=source,
    )


class TestRRFBasics:
    """Test basic RRF behavior."""

    def test_empty_input(self) -> None:
        """RRF with no inputs returns empty list."""
        result = reciprocal_rank_fusion()
        assert result == []

    def test_single_list(self) -> None:
        """RRF with a single list preserves ranking."""
        items = [_make_result(score=0.9), _make_result(score=0.8)]
        fused = reciprocal_rank_fusion(items)
        assert len(fused) == 2
        assert fused[0].score > fused[1].score

    def test_deduplication_by_chunk_id(self) -> None:
        """Same chunk_id in multiple lists is deduplicated."""
        shared_id = uuid4().hex
        list_a = [_make_result(chunk_id_hex=shared_id, score=0.9, source="vector")]
        list_b = [_make_result(chunk_id_hex=shared_id, score=0.8, source="fulltext")]

        fused = reciprocal_rank_fusion(list_a, list_b)
        assert len(fused) == 1
        # Score should be sum of RRF contributions from both lists
        assert fused[0].score > 0

    def test_consensus_boosting(self) -> None:
        """Chunks appearing in multiple lists get higher fused scores."""
        shared_uuid = uuid4()
        unique_uuid = uuid4()

        list_a = [
            _make_result(chunk_id_hex=shared_uuid.hex, score=0.9, source="vector"),
            _make_result(chunk_id_hex=unique_uuid.hex, score=0.95, source="vector"),
        ]
        list_b = [
            _make_result(chunk_id_hex=shared_uuid.hex, score=0.8, source="fulltext"),
        ]

        fused = reciprocal_rank_fusion(list_a, list_b)
        shared_result = next(r for r in fused if r.chunk_id == shared_uuid)
        unique_result = next(r for r in fused if r.chunk_id == unique_uuid)

        # Shared chunk should have higher fused score (appears in both lists)
        assert shared_result.score > unique_result.score


class TestRRFWeights:
    """Test RRF weighting behavior."""

    def test_equal_weights(self) -> None:
        """Equal weights produce balanced fusion."""
        id_a = uuid4().hex
        id_b = uuid4().hex

        list_a = [_make_result(chunk_id_hex=id_a, source="vector")]
        list_b = [_make_result(chunk_id_hex=id_b, source="fulltext")]

        fused = reciprocal_rank_fusion(list_a, list_b, weights=[1.0, 1.0])
        # Both should have equal scores since they're rank 1 in their lists
        assert len(fused) == 2
        assert abs(fused[0].score - fused[1].score) < 0.001

    def test_unequal_weights(self) -> None:
        """Higher weight on a list boosts its items."""
        uuid_a = uuid4()
        uuid_b = uuid4()

        list_a = [_make_result(chunk_id_hex=uuid_a.hex, source="vector")]
        list_b = [_make_result(chunk_id_hex=uuid_b.hex, source="fulltext")]

        fused = reciprocal_rank_fusion(list_a, list_b, weights=[2.0, 1.0])
        # list_a item should have higher score due to 2x weight
        a_result = next(r for r in fused if r.chunk_id == uuid_a)
        b_result = next(r for r in fused if r.chunk_id == uuid_b)
        assert a_result.score > b_result.score

    def test_mismatched_weights_raises(self) -> None:
        """Mismatched weights count raises ValueError."""
        items = [_make_result()]
        with pytest.raises(ValueError, match="weights length"):
            reciprocal_rank_fusion(items, weights=[1.0, 2.0])


class TestRRFMetadata:
    """Test RRF metadata tracking."""

    def test_fused_source_is_fusion(self) -> None:
        """Fused results have source='fusion'."""
        items = [_make_result()]
        fused = reciprocal_rank_fusion(items)
        assert all(r.source == "fusion" for r in fused)

    def test_source_ranks_tracked(self) -> None:
        """Metadata tracks which sources contributed and their ranks."""
        shared_id = uuid4().hex
        list_a = [_make_result(chunk_id_hex=shared_id, source="vector")]
        list_b = [_make_result(chunk_id_hex=shared_id, source="fulltext")]

        fused = reciprocal_rank_fusion(list_a, list_b)
        assert len(fused) == 1
        ranks = fused[0].metadata["source_ranks"]
        assert ranks["vector"] == 1
        assert ranks["fulltext"] == 1

    def test_limit_respected(self) -> None:
        """Fusion respects the limit parameter."""
        items = [_make_result() for _ in range(10)]
        fused = reciprocal_rank_fusion(items, limit=3)
        assert len(fused) == 3


class TestRRFFormula:
    """Test the RRF formula correctness."""

    def test_rrf_formula_single_list(self) -> None:
        """Verify RRF score = 1/(k+rank) for single list."""
        k = 60
        items = [_make_result()]
        fused = reciprocal_rank_fusion(items, k=k)
        expected = 1.0 / (k + 1)  # rank 1
        assert abs(fused[0].score - expected) < 1e-10

    def test_rrf_formula_dual_list(self) -> None:
        """Verify RRF score = sum(1/(k+rank_i)) for dual lists."""
        k = 60
        shared_id = uuid4().hex
        list_a = [_make_result(chunk_id_hex=shared_id, source="vector")]
        list_b = [_make_result(chunk_id_hex=shared_id, source="fulltext")]

        fused = reciprocal_rank_fusion(list_a, list_b, k=k)
        expected = 1.0 / (k + 1) + 1.0 / (k + 1)  # rank 1 in both
        assert abs(fused[0].score - expected) < 1e-10

    def test_rrf_k_parameter_effect(self) -> None:
        """Smaller k gives more weight to top-ranked items."""
        items = [_make_result(), _make_result()]

        fused_small_k = reciprocal_rank_fusion(items, k=1)
        fused_large_k = reciprocal_rank_fusion(items, k=100)

        # With small k, gap between rank 1 and rank 2 is larger
        gap_small = fused_small_k[0].score - fused_small_k[1].score
        gap_large = fused_large_k[0].score - fused_large_k[1].score
        assert gap_small > gap_large
