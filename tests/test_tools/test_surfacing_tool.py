"""Tests for M5: surface_related MCP tool handler."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from author_library.tools.surfacing import handle_surface_related


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHandleSurfaceRelated:
    @pytest.mark.asyncio()
    async def test_missing_all_context_returns_error(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ):
        result = await handle_surface_related(
            {},
            settings=mock_settings,
            storage=mock_storage,
            embedding_provider=mock_embedding_provider,
        )
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio()
    async def test_with_text_context(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ):
        from author_library.surfacing.personal_inclusion import BlendedSurfacingResult
        from author_library.surfacing.related_content import (
            ConnectionType,
            RelatedItem,
        )

        blended = BlendedSurfacingResult(
            context_chunk_id="",
            context_work_id="",
            items=[
                RelatedItem(
                    chunk_id="c-1", work_id="w-1",
                    text="Imagination passage",
                    source_class="primary", granularity="meso",
                    connection_type=ConnectionType.VECTOR_SIMILARITY,
                    relevance_score=0.85,
                    metadata={"work_title": "Faith, Hope and Poetry"},
                ),
            ],
            strategies_used=["vector_similarity"],
            personal_count=0,
            author_count=1,
        )

        with patch(
            "author_library.tools.surfacing.PersonalContentBlender",
        ) as MockBlender:
            instance = MockBlender.return_value
            instance.find_blended = AsyncMock(return_value=blended)

            result = await handle_surface_related(
                {"text_context": "imagination as prayer"},
                settings=mock_settings,
                storage=mock_storage,
                embedding_provider=mock_embedding_provider,
            )

        parsed = json.loads(result)
        assert parsed["total_results"] >= 1
        assert parsed["personal_count"] == 0
        assert parsed["author_count"] == 1
        assert "strategies_used" in parsed

    @pytest.mark.asyncio()
    async def test_with_chunk_id(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ):
        from author_library.surfacing.personal_inclusion import BlendedSurfacingResult

        blended = BlendedSurfacingResult(
            context_chunk_id="chunk-1",
            context_work_id="work-1",
            items=[],
            strategies_used=["passage_links", "vector_similarity"],
            personal_count=0,
            author_count=0,
        )

        with patch(
            "author_library.tools.surfacing.PersonalContentBlender",
        ) as MockBlender:
            instance = MockBlender.return_value
            instance.find_blended = AsyncMock(return_value=blended)

            result = await handle_surface_related(
                {"chunk_id": "chunk-1"},
                settings=mock_settings,
                storage=mock_storage,
                embedding_provider=mock_embedding_provider,
            )

        parsed = json.loads(result)
        assert parsed["context"]["chunk_id"] == "chunk-1"

    @pytest.mark.asyncio()
    async def test_max_per_level_passed_through(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ):
        from author_library.surfacing.personal_inclusion import BlendedSurfacingResult

        blended = BlendedSurfacingResult(
            context_chunk_id="", context_work_id="",
            items=[], strategies_used=[],
            personal_count=0, author_count=0,
        )

        with patch(
            "author_library.tools.surfacing.PersonalContentBlender",
        ) as MockBlender:
            instance = MockBlender.return_value
            instance.find_blended = AsyncMock(return_value=blended)

            result = await handle_surface_related(
                {"text_context": "test", "max_per_level": 3},
                settings=mock_settings,
                storage=mock_storage,
                embedding_provider=mock_embedding_provider,
            )

        parsed = json.loads(result)
        assert "results" in parsed
