"""Tests for O1: Personal reflection gatherer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from author_library.synthesis.gatherer import (
    GatheredReflections,
    PersonalReflection,
    PersonalReflectionGatherer,
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
    return storage


@pytest.fixture()
def mock_embedding_provider():
    return MagicMock()


@pytest.fixture()
def gatherer(mock_settings, mock_storage, mock_embedding_provider):
    return PersonalReflectionGatherer(
        settings=mock_settings,
        storage=mock_storage,
        embedding_provider=mock_embedding_provider,
    )


# ---------------------------------------------------------------------------
# PersonalReflection model tests
# ---------------------------------------------------------------------------


class TestPersonalReflection:
    def test_creation(self):
        ref = PersonalReflection(
            chunk_id="ref-1",
            work_id="personal--my-notes",
            text="I think Guite is right about imagination as prayer.",
            date_created="2026-01-15T10:30:00",
            granularity="micro",
            metadata={
                "section_type": "my_thoughts",
                "source_note": "guite-faith-hope-poetry-ch3",
                "themes": ["imagination", "prayer"],
            },
        )
        assert ref.section_type == "my_thoughts"
        assert ref.source_note == "guite-faith-hope-poetry-ch3"
        assert ref.themes == ["imagination", "prayer"]

    def test_defaults(self):
        ref = PersonalReflection(
            chunk_id="ref-1",
            work_id="w-1",
            text="Reflection text",
            date_created="2026-02-01",
            granularity="micro",
        )
        assert ref.section_type == "freeform"
        assert ref.source_note == ""
        assert ref.themes == []


class TestGatheredReflections:
    def test_empty(self):
        result = GatheredReflections(
            reflections=[],
            total_found=0,
            filters_applied={},
        )
        assert result.theme_counts == {}
        assert result.date_range is None

    def test_theme_counts(self):
        reflections = [
            PersonalReflection(
                chunk_id="r-1", work_id="w-1", text="T1",
                date_created="2026-01-01", granularity="micro",
                metadata={"themes": ["imagination", "prayer"]},
            ),
            PersonalReflection(
                chunk_id="r-2", work_id="w-1", text="T2",
                date_created="2026-01-15", granularity="micro",
                metadata={"themes": ["imagination"]},
            ),
            PersonalReflection(
                chunk_id="r-3", work_id="w-1", text="T3",
                date_created="2026-02-01", granularity="micro",
                metadata={"themes": ["liturgy"]},
            ),
        ]
        result = GatheredReflections(
            reflections=reflections,
            total_found=3,
            filters_applied={"theme": "imagination"},
            date_range=("2026-01-01", "2026-02-01"),
        )
        assert result.theme_counts == {
            "imagination": 2,
            "prayer": 1,
            "liturgy": 1,
        }

    def test_date_range(self):
        result = GatheredReflections(
            reflections=[],
            total_found=0,
            filters_applied={},
            date_range=("2025-06-01", "2026-02-01"),
        )
        assert result.date_range == ("2025-06-01", "2026-02-01")


# ---------------------------------------------------------------------------
# Gatherer tests
# ---------------------------------------------------------------------------


class TestGatherer:
    @pytest.mark.asyncio()
    async def test_gather_no_filters(self, gatherer, mock_storage):
        """With no filters, queries all personal chunks."""
        mock_storage.pg.fetch_all = AsyncMock(return_value=[
            {
                "chunk_id": "ref-1",
                "work_id": "personal--notes",
                "text": "My reflection on imagination",
                "date_created": "2026-01-15T10:30:00",
                "granularity": "micro",
                "metadata": {},
                "pass_number": 1,
            },
        ])

        result = await gatherer.gather()

        assert result.total_found == 1
        assert result.reflections[0].chunk_id == "ref-1"
        assert result.reflections[0].text == "My reflection on imagination"

    @pytest.mark.asyncio()
    async def test_gather_with_date_range(self, gatherer, mock_storage):
        """Date range filters are passed to SQL query."""
        mock_storage.pg.fetch_all = AsyncMock(return_value=[])

        result = await gatherer.gather(
            date_after="2025-06-01",
            date_before="2026-02-01",
        )

        assert result.filters_applied["date_after"] == "2025-06-01"
        assert result.filters_applied["date_before"] == "2026-02-01"
        # Verify the SQL was called with date params
        call_args = mock_storage.pg.fetch_all.call_args
        sql = call_args[0][0]
        assert "created_at >=" in sql
        assert "created_at <=" in sql

    @pytest.mark.asyncio()
    async def test_gather_with_speaker(self, gatherer, mock_storage):
        """Speaker filter narrows by work author."""
        mock_storage.pg.fetch_all = AsyncMock(return_value=[])

        result = await gatherer.gather(speaker="malcolm-guite")

        assert result.filters_applied["speaker"] == "malcolm-guite"
        call_args = mock_storage.pg.fetch_all.call_args
        sql = call_args[0][0]
        assert "author" in sql.lower()

    @pytest.mark.asyncio()
    async def test_gather_with_theme(self, gatherer, mock_storage):
        """Theme filter triggers graph query."""
        # SQL query returns nothing
        mock_storage.pg.fetch_all = AsyncMock(return_value=[])
        # Graph query returns a personal chunk on the theme
        mock_storage.neo4j.execute_read = AsyncMock(return_value=[
            {
                "chunk_id": "graph-ref-1",
                "work_id": "personal--notes",
                "text_preview": "My thoughts on imagination",
            },
        ])
        # Full chunk fetch from PG
        mock_storage.pg.fetch_one = AsyncMock(return_value={
            "chunk_id": "graph-ref-1",
            "work_id": "personal--notes",
            "text": "My detailed thoughts on imagination and prayer",
            "date_created": "2026-01-20T14:00:00",
            "granularity": "micro",
            "metadata": {},
        })

        result = await gatherer.gather(theme="imagination")

        assert result.filters_applied["theme"] == "imagination"
        assert result.total_found >= 1

    @pytest.mark.asyncio()
    async def test_gather_deduplicates(self, gatherer, mock_storage):
        """Results from multiple strategies are deduplicated."""
        # SQL returns a chunk
        mock_storage.pg.fetch_all = AsyncMock(return_value=[
            {
                "chunk_id": "ref-1",
                "work_id": "personal--notes",
                "text": "Reflection text",
                "date_created": "2026-01-15",
                "granularity": "micro",
                "metadata": {},
                "pass_number": 1,
            },
        ])
        # Graph also returns the same chunk
        mock_storage.neo4j.execute_read = AsyncMock(return_value=[
            {
                "chunk_id": "ref-1",
                "work_id": "personal--notes",
                "text_preview": "Reflection text",
            },
        ])
        mock_storage.pg.fetch_one = AsyncMock(return_value={
            "chunk_id": "ref-1",
            "work_id": "personal--notes",
            "text": "Reflection text",
            "date_created": "2026-01-15",
            "granularity": "micro",
            "metadata": {},
        })

        result = await gatherer.gather(theme="imagination")

        # Should be deduplicated to 1
        assert result.total_found == 1

    @pytest.mark.asyncio()
    async def test_gather_chronological_order(self, gatherer, mock_storage):
        """Results are sorted chronologically."""
        mock_storage.pg.fetch_all = AsyncMock(return_value=[
            {
                "chunk_id": "ref-2",
                "work_id": "personal--notes",
                "text": "Later reflection",
                "date_created": "2026-02-01",
                "granularity": "micro",
                "metadata": {},
                "pass_number": 1,
            },
            {
                "chunk_id": "ref-1",
                "work_id": "personal--notes",
                "text": "Earlier reflection",
                "date_created": "2025-06-15",
                "granularity": "micro",
                "metadata": {},
                "pass_number": 1,
            },
        ])

        result = await gatherer.gather()

        dates = [r.date_created for r in result.reflections]
        assert dates == sorted(dates)

    @pytest.mark.asyncio()
    async def test_gather_with_prompt_uses_semantic_search(self, gatherer, mock_storage):
        """Prompt parameter triggers semantic search strategy."""
        mock_storage.pg.fetch_all = AsyncMock(return_value=[])

        with patch(
            "author_library.retrieval.vector_search.vector_search",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await gatherer.gather(prompt="What do I think about imagination?")

        assert result.filters_applied["prompt"] == "What do I think about imagination?"

    @pytest.mark.asyncio()
    async def test_row_to_reflection_with_json_metadata(self):
        """Metadata can be a JSON string or dict."""
        import json

        row_dict = {
            "chunk_id": "ref-1",
            "work_id": "w-1",
            "text": "Text",
            "date_created": "2026-01-01",
            "granularity": "micro",
            "metadata": json.dumps({"section_type": "my_thoughts", "themes": ["prayer"]}),
        }
        ref = PersonalReflectionGatherer._row_to_reflection(row_dict)
        assert ref.section_type == "my_thoughts"
        assert ref.themes == ["prayer"]

    @pytest.mark.asyncio()
    async def test_row_to_reflection_with_dict_metadata(self):
        row_dict = {
            "chunk_id": "ref-1",
            "work_id": "w-1",
            "text": "Text",
            "date_created": "2026-01-01",
            "granularity": "micro",
            "metadata": {"section_type": "session_reflections"},
        }
        ref = PersonalReflectionGatherer._row_to_reflection(row_dict)
        assert ref.section_type == "session_reflections"

    @pytest.mark.asyncio()
    async def test_gather_limit(self, gatherer, mock_storage):
        """Limit parameter caps results."""
        rows = [
            {
                "chunk_id": f"ref-{i}",
                "work_id": "personal--notes",
                "text": f"Reflection {i}",
                "date_created": f"2026-01-{i + 1:02d}",
                "granularity": "micro",
                "metadata": {},
                "pass_number": 1,
            }
            for i in range(20)
        ]
        mock_storage.pg.fetch_all = AsyncMock(return_value=rows)

        result = await gatherer.gather(limit=5)

        assert len(result.reflections) == 5
