"""Tests for M4: Personal content inclusion."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from author_library.surfacing.personal_inclusion import (
    BlendedSurfacingResult,
    PersonalContentBlender,
    blend_results,
    boost_personal_for_context,
)
from author_library.surfacing.related_content import (
    ConnectionType,
    RelatedContentResult,
    RelatedItem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    *,
    chunk_id: str = "chunk-1",
    source_class: str = "primary",
    relevance_score: float = 0.7,
    connection_type: ConnectionType = ConnectionType.VECTOR_SIMILARITY,
    metadata: dict | None = None,
) -> RelatedItem:
    return RelatedItem(
        chunk_id=chunk_id,
        work_id="work-1",
        text="Test passage",
        source_class=source_class,
        granularity="meso",
        connection_type=connection_type,
        relevance_score=relevance_score,
        metadata=metadata or {},
    )


def _make_personal(
    *,
    chunk_id: str = "p-1",
    relevance_score: float = 0.5,
    metadata: dict | None = None,
) -> RelatedItem:
    return _make_item(
        chunk_id=chunk_id,
        source_class="personal",
        relevance_score=relevance_score,
        connection_type=ConnectionType.PERSONAL_REFLECTION,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# blend_results tests
# ---------------------------------------------------------------------------


class TestBlendResults:
    def test_empty(self):
        assert blend_results([]) == []

    def test_no_personal_items(self):
        items = [
            _make_item(chunk_id="a-1", relevance_score=0.9),
            _make_item(chunk_id="a-2", relevance_score=0.8),
        ]
        result = blend_results(items, max_results=5)
        assert len(result) == 2

    def test_only_personal_items(self):
        items = [
            _make_personal(chunk_id="p-1", relevance_score=0.7),
            _make_personal(chunk_id="p-2", relevance_score=0.5),
        ]
        result = blend_results(items, max_results=5)
        assert len(result) == 2
        assert all(i.source_class == "personal" for i in result)

    def test_guarantees_min_personal(self):
        """Personal items are guaranteed even if lower-scored than author items."""
        items = [
            _make_item(chunk_id="a-1", relevance_score=0.95),
            _make_item(chunk_id="a-2", relevance_score=0.9),
            _make_item(chunk_id="a-3", relevance_score=0.85),
            _make_item(chunk_id="a-4", relevance_score=0.8),
            _make_personal(chunk_id="p-1", relevance_score=0.3),
            _make_personal(chunk_id="p-2", relevance_score=0.2),
        ]
        result = blend_results(items, max_results=5, min_personal=2)

        personal = [i for i in result if i.source_class == "personal"]
        assert len(personal) >= 2

    def test_respects_max_results(self):
        items = [
            _make_item(chunk_id=f"a-{i}", relevance_score=0.9 - i * 0.05)
            for i in range(10)
        ] + [
            _make_personal(chunk_id=f"p-{i}", relevance_score=0.5 - i * 0.05)
            for i in range(5)
        ]
        result = blend_results(items, max_results=7, min_personal=2)
        assert len(result) == 7

    def test_personal_exceeds_max(self):
        """When min_personal > max_results, caps at max_results."""
        items = [
            _make_personal(chunk_id=f"p-{i}", relevance_score=0.5 - i * 0.05)
            for i in range(5)
        ]
        result = blend_results(items, max_results=3, min_personal=5)
        assert len(result) == 3

    def test_sorted_by_relevance(self):
        items = [
            _make_item(chunk_id="a-1", relevance_score=0.9),
            _make_personal(chunk_id="p-1", relevance_score=0.3),
            _make_item(chunk_id="a-2", relevance_score=0.7),
            _make_personal(chunk_id="p-2", relevance_score=0.2),
        ]
        result = blend_results(items, max_results=10, min_personal=2)
        scores = [i.relevance_score for i in result]
        assert scores == sorted(scores, reverse=True)

    def test_high_scoring_personal_competes_normally(self):
        """Personal items with high scores don't get double-counted."""
        items = [
            _make_item(chunk_id="a-1", relevance_score=0.9),
            _make_personal(chunk_id="p-1", relevance_score=0.95),
            _make_personal(chunk_id="p-2", relevance_score=0.85),
            _make_personal(chunk_id="p-3", relevance_score=0.1),
            _make_item(chunk_id="a-2", relevance_score=0.5),
        ]
        result = blend_results(items, max_results=4, min_personal=2)
        assert len(result) == 4
        # p-1 and p-2 should be in the result (they're high-scoring)
        chunk_ids = {i.chunk_id for i in result}
        assert "p-1" in chunk_ids
        assert "p-2" in chunk_ids


# ---------------------------------------------------------------------------
# boost_personal_for_context tests
# ---------------------------------------------------------------------------


class TestBoostPersonalForContext:
    def test_no_context_no_change(self):
        items = [_make_personal(chunk_id="p-1", relevance_score=0.5)]
        result = boost_personal_for_context(items, context_work_id="")
        assert result[0].relevance_score == 0.5

    def test_boosts_matching_source_note(self):
        items = [
            _make_personal(
                chunk_id="p-1",
                relevance_score=0.5,
                metadata={"source_note": "guite--faith-hope-poetry-ch3"},
            ),
        ]
        result = boost_personal_for_context(items, context_work_id="guite--faith-hope-poetry")
        assert result[0].relevance_score > 0.5
        assert result[0].metadata.get("boosted") is True

    def test_does_not_boost_unrelated(self):
        items = [
            _make_personal(
                chunk_id="p-1",
                relevance_score=0.5,
                metadata={"source_note": "ordway--tolkien-modern-middle-ages"},
            ),
        ]
        result = boost_personal_for_context(items, context_work_id="guite--faith-hope-poetry")
        assert result[0].relevance_score == 0.5

    def test_caps_at_1(self):
        items = [
            _make_personal(
                chunk_id="p-1",
                relevance_score=0.9,
                metadata={"source_note": "guite--faith-hope-poetry-ch3"},
            ),
        ]
        result = boost_personal_for_context(items, context_work_id="guite--faith-hope-poetry")
        assert result[0].relevance_score <= 1.0

    def test_author_items_unchanged(self):
        items = [
            _make_item(chunk_id="a-1", relevance_score=0.8),
        ]
        result = boost_personal_for_context(items, context_work_id="guite--faith-hope-poetry")
        assert result[0].relevance_score == 0.8

    def test_reorders_by_score(self):
        items = [
            _make_item(chunk_id="a-1", relevance_score=0.7),
            _make_personal(
                chunk_id="p-1",
                relevance_score=0.6,
                metadata={"source_note": "guite--faith-hope-poetry-ch3"},
            ),
        ]
        result = boost_personal_for_context(items, context_work_id="guite--faith-hope-poetry")
        # After 1.3x boost, personal should be 0.78 > 0.7
        assert result[0].chunk_id == "p-1"


# ---------------------------------------------------------------------------
# PersonalContentBlender tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_settings():
    settings = MagicMock()
    settings.llm.query_model = "claude-sonnet-4-5-20250929"
    settings.api_keys.anthropic_api_key.get_secret_value.return_value = "test-key"
    return settings


@pytest.fixture()
def mock_storage():
    storage = MagicMock()
    storage.pg = MagicMock()
    storage.neo4j = MagicMock()
    storage.works = MagicMock()
    storage.chunks = MagicMock()
    storage.embeddings = MagicMock()
    storage.graph = MagicMock()
    return storage


@pytest.fixture()
def mock_embedding_provider():
    return MagicMock()


@pytest.fixture()
def blender(mock_settings, mock_storage, mock_embedding_provider):
    return PersonalContentBlender(
        settings=mock_settings,
        storage=mock_storage,
        embedding_provider=mock_embedding_provider,
    )


class TestPersonalContentBlender:
    @pytest.mark.asyncio()
    async def test_exclude_personal(self, blender):
        mixed_items = [
            _make_item(chunk_id="a-1", relevance_score=0.9),
            _make_personal(chunk_id="p-1", relevance_score=0.8),
            _make_item(chunk_id="a-2", relevance_score=0.7),
        ]

        with patch.object(
            blender._finder, "find_related",
            new_callable=AsyncMock,
            return_value=RelatedContentResult(
                context_chunk_id="ctx",
                context_work_id="w-1",
                items=mixed_items,
                strategies_used=["vector_similarity"],
            ),
        ):
            result = await blender.find_blended(
                text_context="test",
                include_personal=False,
            )

        assert result.personal_count == 0
        assert result.author_count == 2
        assert all(i.source_class != "personal" for i in result.items)

    @pytest.mark.asyncio()
    async def test_include_personal_with_minimum(self, blender):
        items = [
            _make_item(chunk_id="a-1", relevance_score=0.95),
            _make_item(chunk_id="a-2", relevance_score=0.9),
            _make_item(chunk_id="a-3", relevance_score=0.85),
            _make_personal(chunk_id="p-1", relevance_score=0.3),
            _make_personal(chunk_id="p-2", relevance_score=0.2),
        ]

        with patch.object(
            blender._finder, "find_related",
            new_callable=AsyncMock,
            return_value=RelatedContentResult(
                context_chunk_id="ctx",
                context_work_id="w-1",
                items=items,
                strategies_used=["vector_similarity", "personal_reflections"],
            ),
        ):
            result = await blender.find_blended(
                text_context="test",
                max_results=4,
                min_personal=2,
            )

        assert result.personal_count >= 2

    @pytest.mark.asyncio()
    async def test_empty_result(self, blender):
        with patch.object(
            blender._finder, "find_related",
            new_callable=AsyncMock,
            return_value=RelatedContentResult(
                context_chunk_id="",
                context_work_id="",
                items=[],
                strategies_used=[],
            ),
        ):
            result = await blender.find_blended(text_context="test")

        assert result.personal_count == 0
        assert result.author_count == 0
        assert result.items == []
