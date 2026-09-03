"""Tests for M1: Related content query."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from author_library.surfacing.related_content import (
    ConnectionType,
    RelatedContentFinder,
    RelatedContentResult,
    RelatedItem,
)


# ---------------------------------------------------------------------------
# Fixtures
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
def finder(mock_settings, mock_storage, mock_embedding_provider):
    return RelatedContentFinder(
        settings=mock_settings,
        storage=mock_storage,
        embedding_provider=mock_embedding_provider,
    )


# ---------------------------------------------------------------------------
# RelatedItem tests
# ---------------------------------------------------------------------------


class TestRelatedItem:
    def test_creation(self):
        item = RelatedItem(
            chunk_id="chunk-1",
            work_id="work-1",
            text="Test passage about imagination",
            source_class="primary",
            granularity="meso",
            connection_type=ConnectionType.PASSAGE_LINK,
            relevance_score=0.85,
            metadata={"work_title": "Faith, Hope and Poetry", "author": "Test Guite"},
        )
        assert item.chunk_id == "chunk-1"
        assert item.work_title == "Faith, Hope and Poetry"
        assert item.author == "Test Guite"
        assert item.connection_type == ConnectionType.PASSAGE_LINK

    def test_metadata_defaults(self):
        item = RelatedItem(
            chunk_id="chunk-1",
            work_id="work-1",
            text="Test",
            source_class="primary",
            granularity="meso",
            connection_type=ConnectionType.VECTOR_SIMILARITY,
            relevance_score=0.5,
        )
        assert item.work_title == ""
        assert item.author == ""


class TestConnectionType:
    def test_enum_values(self):
        assert ConnectionType.PASSAGE_LINK == "passage_link"
        assert ConnectionType.THEMATIC_PARALLEL == "thematic_parallel"
        assert ConnectionType.PERSONAL_REFLECTION == "personal_reflection"
        assert ConnectionType.VECTOR_SIMILARITY == "vector_similarity"
        assert ConnectionType.TEMPORAL_PROXIMITY == "temporal_proximity"


# ---------------------------------------------------------------------------
# RelatedContentResult tests
# ---------------------------------------------------------------------------


class TestRelatedContentResult:
    def test_empty_result(self):
        result = RelatedContentResult(
            context_chunk_id="chunk-1",
            context_work_id="work-1",
            items=[],
            strategies_used=[],
        )
        assert result.items == []
        assert result.strategies_used == []

    def test_with_items(self):
        items = [
            RelatedItem(
                chunk_id="r-1",
                work_id="w-1",
                text="Related content",
                source_class="primary",
                granularity="meso",
                connection_type=ConnectionType.PASSAGE_LINK,
                relevance_score=0.9,
            ),
            RelatedItem(
                chunk_id="r-2",
                work_id="w-2",
                text="Another related item",
                source_class="personal",
                granularity="micro",
                connection_type=ConnectionType.PERSONAL_REFLECTION,
                relevance_score=0.7,
            ),
        ]
        result = RelatedContentResult(
            context_chunk_id="chunk-1",
            context_work_id="work-1",
            items=items,
            strategies_used=["passage_links", "personal_reflections"],
        )
        assert len(result.items) == 2
        assert len(result.strategies_used) == 2


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_dedup_keeps_highest_score(self):
        items = [
            RelatedItem(
                chunk_id="chunk-1",
                work_id="w-1",
                text="Text 1",
                source_class="primary",
                granularity="meso",
                connection_type=ConnectionType.PASSAGE_LINK,
                relevance_score=0.8,
            ),
            RelatedItem(
                chunk_id="chunk-1",  # Same chunk_id
                work_id="w-1",
                text="Text 1",
                source_class="primary",
                granularity="meso",
                connection_type=ConnectionType.VECTOR_SIMILARITY,
                relevance_score=0.9,  # Higher score
            ),
        ]
        deduped = RelatedContentFinder._deduplicate(items)
        assert len(deduped) == 1
        assert deduped[0].relevance_score == 0.9
        assert deduped[0].connection_type == ConnectionType.VECTOR_SIMILARITY

    def test_dedup_different_chunks(self):
        items = [
            RelatedItem(
                chunk_id="chunk-1",
                work_id="w-1",
                text="Text 1",
                source_class="primary",
                granularity="meso",
                connection_type=ConnectionType.PASSAGE_LINK,
                relevance_score=0.8,
            ),
            RelatedItem(
                chunk_id="chunk-2",
                work_id="w-2",
                text="Text 2",
                source_class="secondary",
                granularity="macro",
                connection_type=ConnectionType.THEMATIC_PARALLEL,
                relevance_score=0.6,
            ),
        ]
        deduped = RelatedContentFinder._deduplicate(items)
        assert len(deduped) == 2

    def test_dedup_empty(self):
        assert RelatedContentFinder._deduplicate([]) == []


# ---------------------------------------------------------------------------
# find_related tests
# ---------------------------------------------------------------------------


class TestFindRelated:
    @pytest.mark.asyncio()
    async def test_no_context_returns_empty(self, finder):
        result = await finder.find_related()
        assert result.items == []
        assert result.strategies_used == []

    @pytest.mark.asyncio()
    async def test_with_chunk_id_resolves_chunk(self, finder, mock_storage):
        chunk_id = str(uuid4())
        mock_storage.pg.fetch_one = AsyncMock(return_value={
            "id": chunk_id,
            "work_id": "guite--faith-hope-poetry",
            "text": "Imagination as a faculty of perception",
            "source_class": "primary",
            "granularity": "meso",
            "chapter": "Chapter 3",
            "section": None,
            "metadata": {},
            "pass_number": 1,
        })
        mock_storage.graph.get_themes_for_chunk = AsyncMock(return_value=[])

        # Mock all search strategies to return empty
        with patch.object(finder, "_find_via_passage_links", new_callable=AsyncMock, return_value=[]), \
             patch.object(finder, "_find_via_vector_similarity", new_callable=AsyncMock, return_value=[]):
            result = await finder.find_related(chunk_id=chunk_id)

        assert result.context_chunk_id == chunk_id
        assert result.context_work_id == "guite--faith-hope-poetry"

    @pytest.mark.asyncio()
    async def test_with_text_context_only(self, finder):
        with patch.object(finder, "_find_via_vector_similarity", new_callable=AsyncMock, return_value=[
            RelatedItem(
                chunk_id="sim-1",
                work_id="w-1",
                text="Similar text",
                source_class="primary",
                granularity="meso",
                connection_type=ConnectionType.VECTOR_SIMILARITY,
                relevance_score=0.65,
            ),
        ]):
            result = await finder.find_related(text_context="Imagination as prayer")

        assert len(result.items) == 1
        assert "vector_similarity" in result.strategies_used

    @pytest.mark.asyncio()
    async def test_strategies_run_in_parallel(self, finder, mock_storage):
        chunk_id = str(uuid4())
        mock_storage.pg.fetch_one = AsyncMock(return_value={
            "id": chunk_id,
            "work_id": "work-1",
            "text": "Context text",
            "source_class": "primary",
            "granularity": "meso",
            "chapter": None,
            "section": None,
            "metadata": {},
            "pass_number": 1,
        })
        mock_storage.graph.get_themes_for_chunk = AsyncMock(return_value=[
            {"canonical_name": "imagination"},
        ])

        passage_result = RelatedItem(
            chunk_id="pl-1", work_id="w-1", text="Passage link",
            source_class="primary", granularity="meso",
            connection_type=ConnectionType.PASSAGE_LINK, relevance_score=0.9,
        )
        theme_result = RelatedItem(
            chunk_id="th-1", work_id="w-2", text="Thematic",
            source_class="primary", granularity="macro",
            connection_type=ConnectionType.THEMATIC_PARALLEL, relevance_score=0.6,
        )
        personal_result = RelatedItem(
            chunk_id="pr-1", work_id="w-3", text="My reflection",
            source_class="personal", granularity="micro",
            connection_type=ConnectionType.PERSONAL_REFLECTION, relevance_score=0.85,
        )
        vector_result = RelatedItem(
            chunk_id="vs-1", work_id="w-4", text="Vector similar",
            source_class="contextual", granularity="meso",
            connection_type=ConnectionType.VECTOR_SIMILARITY, relevance_score=0.5,
        )

        with patch.object(finder, "_find_via_passage_links", new_callable=AsyncMock, return_value=[passage_result]), \
             patch.object(finder, "_find_via_themes", new_callable=AsyncMock, return_value=[theme_result]), \
             patch.object(finder, "_find_personal_reflections", new_callable=AsyncMock, return_value=[personal_result]), \
             patch.object(finder, "_find_via_vector_similarity", new_callable=AsyncMock, return_value=[vector_result]):
            result = await finder.find_related(chunk_id=chunk_id, include_personal=True)

        assert len(result.items) == 4
        assert set(result.strategies_used) == {
            "passage_links", "thematic_parallels",
            "personal_reflections", "vector_similarity",
        }
        # Should be sorted by relevance descending
        scores = [item.relevance_score for item in result.items]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio()
    async def test_max_results_limit(self, finder, mock_storage):
        mock_storage.pg.fetch_one = AsyncMock(return_value=None)

        many_items = [
            RelatedItem(
                chunk_id=f"chunk-{i}",
                work_id=f"w-{i}",
                text=f"Item {i}",
                source_class="primary",
                granularity="meso",
                connection_type=ConnectionType.VECTOR_SIMILARITY,
                relevance_score=1.0 - (i * 0.01),
            )
            for i in range(30)
        ]

        with patch.object(finder, "_find_via_vector_similarity", new_callable=AsyncMock, return_value=many_items):
            result = await finder.find_related(text_context="test", max_results=5)

        assert len(result.items) == 5

    @pytest.mark.asyncio()
    async def test_strategy_failure_graceful(self, finder, mock_storage):
        """If one strategy fails, others still return results."""
        mock_storage.pg.fetch_one = AsyncMock(return_value={
            "id": "chunk-1",
            "work_id": "w-1",
            "text": "Context",
            "source_class": "primary",
            "granularity": "meso",
            "chapter": None,
            "section": None,
            "metadata": {},
            "pass_number": 1,
        })
        mock_storage.graph.get_themes_for_chunk = AsyncMock(return_value=[])

        good_result = RelatedItem(
            chunk_id="vs-1", work_id="w-2", text="Good result",
            source_class="primary", granularity="meso",
            connection_type=ConnectionType.VECTOR_SIMILARITY, relevance_score=0.7,
        )

        with patch.object(finder, "_find_via_passage_links", new_callable=AsyncMock, side_effect=RuntimeError("Neo4j down")), \
             patch.object(finder, "_find_via_vector_similarity", new_callable=AsyncMock, return_value=[good_result]):
            result = await finder.find_related(chunk_id="chunk-1")

        assert len(result.items) == 1  # Vector results survived
        assert result.items[0].chunk_id == "vs-1"

    @pytest.mark.asyncio()
    async def test_exclude_personal_when_disabled(self, finder, mock_storage):
        mock_storage.pg.fetch_one = AsyncMock(return_value={
            "id": "chunk-1",
            "work_id": "w-1",
            "text": "Context",
            "source_class": "primary",
            "granularity": "meso",
            "chapter": None,
            "section": None,
            "metadata": {},
            "pass_number": 1,
        })
        mock_storage.graph.get_themes_for_chunk = AsyncMock(return_value=[])

        with patch.object(finder, "_find_via_passage_links", new_callable=AsyncMock, return_value=[]), \
             patch.object(finder, "_find_via_vector_similarity", new_callable=AsyncMock, return_value=[]):
            result = await finder.find_related(
                chunk_id="chunk-1",
                include_personal=False,
            )

        # personal_reflections should NOT be in strategies_used
        assert "personal_reflections" not in result.strategies_used
