"""Tests for multi-pass retrieval orchestration.

LLM-based question classification tests are skipped if no ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from author_library.retrieval.models import QuestionType, RetrievalResult
from author_library.retrieval.orchestrator import (
    OrchestratedResult,
    _apply_consensus_boost,
    _preferred_granularity,
    _strategy_for_question_type,
)


class TestStrategySelection:
    """Test question-type-to-strategy mapping."""

    def test_factual_strategy(self) -> None:
        """Factual questions favor text search and micro chunks."""
        strategy = _strategy_for_question_type(QuestionType.FACTUAL)
        assert strategy["text_weight"] > 0.3
        assert strategy["pass1_limit"] > 0

    def test_thematic_strategy(self) -> None:
        """Thematic questions favor vector search."""
        strategy = _strategy_for_question_type(QuestionType.THEMATIC)
        assert strategy["vector_weight"] > strategy["text_weight"]

    def test_generative_strategy(self) -> None:
        """Generative questions heavily favor vector search."""
        strategy = _strategy_for_question_type(QuestionType.GENERATIVE)
        assert strategy["vector_weight"] >= 0.8

    def test_quote_strategy(self) -> None:
        """Quote questions favor text search."""
        strategy = _strategy_for_question_type(QuestionType.QUOTE)
        assert strategy["text_weight"] > strategy["vector_weight"]


class TestGranularityPreference:
    """Test granularity selection for question types."""

    def test_factual_prefers_micro(self) -> None:
        assert _preferred_granularity(QuestionType.FACTUAL) == "micro"

    def test_thematic_prefers_macro(self) -> None:
        assert _preferred_granularity(QuestionType.THEMATIC) == "macro"

    def test_generative_prefers_meso(self) -> None:
        assert _preferred_granularity(QuestionType.GENERATIVE) == "meso"

    def test_quote_prefers_micro(self) -> None:
        assert _preferred_granularity(QuestionType.QUOTE) == "micro"


class TestConsensusBoost:
    """Test cross-pass consensus boosting."""

    def test_no_boost_when_no_overlap(self) -> None:
        """Chunks only in pass1 get no boost."""
        pass1 = [
            RetrievalResult(
                chunk_id=uuid4(),
                work_id="w1",
                text="text",
                score=0.8,
                granularity="meso",
                source_class="primary",
                source="fusion",
            )
        ]
        boosted = _apply_consensus_boost(pass1, [], [])
        assert boosted[0].score == 0.8

    def test_boost_from_pass3_overlap(self) -> None:
        """Chunks in both pass1 and pass3 get 1.5x boost."""
        cid = uuid4()
        pass1 = [
            RetrievalResult(
                chunk_id=cid,
                work_id="w1",
                text="text",
                score=0.8,
                granularity="meso",
                source_class="primary",
                source="fusion",
            )
        ]
        pass3 = [
            RetrievalResult(
                chunk_id=cid,
                work_id="w1",
                text="text",
                score=0.7,
                granularity="micro",
                source_class="primary",
                source="vector",
            )
        ]
        boosted = _apply_consensus_boost(pass1, [], pass3)
        assert abs(boosted[0].score - 0.8 * 1.5) < 0.001

    def test_boost_from_graph_overlap(self) -> None:
        """Chunks in pass1 with graph connections get 1.3x boost."""
        from author_library.retrieval.models import GraphExpansionResult

        cid = uuid4()
        pass1 = [
            RetrievalResult(
                chunk_id=cid,
                work_id="w1",
                text="text",
                score=0.8,
                granularity="meso",
                source_class="primary",
                source="fusion",
            )
        ]
        pass2 = [
            GraphExpansionResult(
                chunk_id=str(cid),
                work_id="w2",
                text_preview="related",
                granularity="meso",
                source_class="contextual",
                relationship_type="ENGAGES_WITH",
                confidence="high",
                evidence="test",
            )
        ]
        boosted = _apply_consensus_boost(pass1, pass2, [])
        assert abs(boosted[0].score - 0.8 * 1.3) < 0.001

    def test_combined_boost(self) -> None:
        """Chunks in all three passes get 1.5 * 1.3 = 1.95x boost."""
        from author_library.retrieval.models import GraphExpansionResult

        cid = uuid4()
        pass1 = [
            RetrievalResult(
                chunk_id=cid,
                work_id="w1",
                text="text",
                score=0.5,
                granularity="meso",
                source_class="primary",
                source="fusion",
            )
        ]
        pass2 = [
            GraphExpansionResult(
                chunk_id=str(cid),
                work_id="w2",
                text_preview="related",
                granularity="meso",
                source_class="contextual",
                relationship_type="ENGAGES_WITH",
                confidence="high",
                evidence="test",
            )
        ]
        pass3 = [
            RetrievalResult(
                chunk_id=cid,
                work_id="w1",
                text="text",
                score=0.4,
                granularity="micro",
                source_class="primary",
                source="vector",
            )
        ]
        boosted = _apply_consensus_boost(pass1, pass2, pass3)
        expected = 0.5 * 1.5 * 1.3
        assert abs(boosted[0].score - expected) < 0.001

    def test_boost_preserves_order(self) -> None:
        """After boosting, results are re-sorted by score descending."""
        cid_boosted = uuid4()
        cid_not_boosted = uuid4()

        pass1 = [
            RetrievalResult(
                chunk_id=cid_not_boosted,
                work_id="w1",
                text="text",
                score=0.9,
                granularity="meso",
                source_class="primary",
                source="fusion",
            ),
            RetrievalResult(
                chunk_id=cid_boosted,
                work_id="w2",
                text="text",
                score=0.7,
                granularity="meso",
                source_class="primary",
                source="fusion",
            ),
        ]
        pass3 = [
            RetrievalResult(
                chunk_id=cid_boosted,
                work_id="w2",
                text="text",
                score=0.6,
                granularity="micro",
                source_class="primary",
                source="vector",
            )
        ]
        boosted = _apply_consensus_boost(pass1, [], pass3)
        # cid_boosted: 0.7 * 1.5 = 1.05 > cid_not_boosted: 0.9
        assert str(boosted[0].chunk_id) == str(cid_boosted)


class TestOrchestratedResult:
    """Test the OrchestratedResult container."""

    def test_total_chunks(self) -> None:
        """total_chunks sums all three pass counts."""
        result = OrchestratedResult(
            question="test",
            question_type=QuestionType.THEMATIC,
            primary_results=[
                RetrievalResult(
                    chunk_id=uuid4(),
                    work_id="w1",
                    text="text",
                    score=0.9,
                    granularity="meso",
                    source_class="primary",
                    source="fusion",
                )
            ],
            graph_expansions=[],
            supporting_evidence=[],
        )
        assert result.total_chunks == 1


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping LLM classification test",
)
class TestQuestionClassificationLive:
    """Live tests for question classification (requires API key)."""

    @pytest.mark.asyncio
    async def test_factual_classification(self) -> None:
        from author_library.config import get_settings
        from author_library.retrieval.orchestrator import classify_question

        settings = get_settings()
        q_type = await classify_question(
            "When was Mere Christianity published?", settings
        )
        assert q_type == QuestionType.FACTUAL

    @pytest.mark.asyncio
    async def test_thematic_classification(self) -> None:
        from author_library.config import get_settings
        from author_library.retrieval.orchestrator import classify_question

        settings = get_settings()
        q_type = await classify_question(
            "How does Lewis treat the concept of joy?", settings
        )
        assert q_type == QuestionType.THEMATIC
