"""Tests for composable query tool handlers — input validation and error paths."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from author_library.errors import RetrievalError
from author_library.retrieval.models import RetrievalResult
from author_library.tools.composable_query import (
    _build_provenance_rules,
    handle_get_passage_links,
    handle_manage_vocabulary,
    handle_search_chunks,
)


# ---------------------------------------------------------------------------
# Helpers for search_chunks filter tests
# ---------------------------------------------------------------------------

_CHUNK_IDS = [uuid4() for _ in range(6)]


def _make_results() -> list[RetrievalResult]:
    """Create a known set of RetrievalResult objects for filter testing."""
    return [
        RetrievalResult(
            chunk_id=_CHUNK_IDS[0],
            work_id="lewis--mere-christianity",
            text="In Mere Christianity Lewis argues about moral law.",
            score=0.95,
            granularity="meso",
            source_class="primary",
            source="vector",
            metadata={"pass_number": 1, "speaker": "C.S. Lewis"},
        ),
        RetrievalResult(
            chunk_id=_CHUNK_IDS[1],
            work_id="lewis--weight-of-glory",
            text="The Weight of Glory is Lewis's most eloquent sermon.",
            score=0.90,
            granularity="meso",
            source_class="primary",
            source="vector",
            metadata={"pass_number": 2, "speaker": "C.S. Lewis"},
        ),
        RetrievalResult(
            chunk_id=_CHUNK_IDS[2],
            work_id="lewis--podcast-interview",
            text="Tolkien discusses the Inklings and Lewis.",
            score=0.85,
            granularity="micro",
            source_class="primary",
            source="vector",
            metadata={"pass_number": 1, "speaker": "J.R.R. Tolkien"},
        ),
        RetrievalResult(
            chunk_id=_CHUNK_IDS[3],
            work_id="mcgrath--cs-lewis-biography",
            text="Lewis's conversion is well documented.",
            score=0.80,
            granularity="meso",
            source_class="secondary",
            source="vector",
            metadata={"pass_number": 1},
        ),
        RetrievalResult(
            chunk_id=_CHUNK_IDS[4],
            work_id="lewis--surprised-by-joy",
            text="Joy is the serious business of Heaven.",
            score=0.75,
            granularity="macro",
            source_class="primary",
            source="vector",
            metadata={"pass_number": 3},
        ),
    ]


def _make_storage_mock() -> MagicMock:
    """Create a minimal storage mock for handle_search_chunks."""
    storage = MagicMock()
    # Works repo — get() returns a minimal work dict
    storage.works.get = AsyncMock(
        return_value={"title": "Test Work", "author": "Test Author"}
    )
    # Neo4j connection (needed for GraphQueryService instantiation)
    storage.neo4j = MagicMock()
    # Graph repo — get_themes_for_chunk returns empty by default
    storage.graph.get_themes_for_chunk = AsyncMock(return_value=[])
    return storage


def _make_graph_service_mock() -> MagicMock:
    """Create a mock GraphQueryService that returns no engagement chain."""
    mock = MagicMock()
    mock.get_engagement_chain = AsyncMock(return_value=None)
    return mock


class TestHandleSearchChunksValidation:
    """Validate required argument checks for search_chunks."""

    async def test_missing_query_raises(self) -> None:
        with pytest.raises(RetrievalError, match="query is required"):
            await handle_search_chunks(
                {},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_empty_query_raises(self) -> None:
        with pytest.raises(RetrievalError, match="query is required"):
            await handle_search_chunks(
                {"query": ""},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )


class TestHandleGetPassageLinksValidation:
    """Validate required argument checks for get_passage_links."""

    async def test_missing_chunk_id_raises(self) -> None:
        with pytest.raises(RetrievalError, match="chunk_id is required"):
            await handle_get_passage_links(
                {},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(RetrievalError, match="chunk_id is required"):
            await handle_get_passage_links(
                {"chunk_id": ""},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )


class TestHandleManageVocabularyValidation:
    """Validate required argument checks for manage_vocabulary."""

    async def test_missing_action_raises(self) -> None:
        with pytest.raises(RetrievalError, match="action is required"):
            await handle_manage_vocabulary(
                {},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_invalid_action_raises(self) -> None:
        with pytest.raises(RetrievalError, match="Invalid action"):
            await handle_manage_vocabulary(
                {"action": "invalid_action"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_propose_missing_term_raises(self) -> None:
        with pytest.raises(RetrievalError, match="term is required"):
            await handle_manage_vocabulary(
                {"action": "propose"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_promote_missing_term_raises(self) -> None:
        with pytest.raises(RetrievalError, match="term is required"):
            await handle_manage_vocabulary(
                {"action": "promote"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_merge_missing_term_raises(self) -> None:
        with pytest.raises(RetrievalError, match="term is required"):
            await handle_manage_vocabulary(
                {"action": "merge"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_merge_missing_merge_into_raises(self) -> None:
        with pytest.raises(RetrievalError, match="merge_into is required"):
            await handle_manage_vocabulary(
                {"action": "merge", "term": "my-term"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_deprecate_missing_term_raises(self) -> None:
        with pytest.raises(RetrievalError, match="term is required"):
            await handle_manage_vocabulary(
                {"action": "deprecate"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_valid_actions_accepted(self) -> None:
        """All valid actions should be accepted (may fail later due to None deps)."""
        for action in ("list", "propose", "promote", "merge", "deprecate"):
            # These will either pass validation and fail later,
            # or raise a term-related error (which is past action validation).
            if action == "list":
                # list doesn't require term, so it will fail on None storage
                with pytest.raises(Exception):  # noqa: B017
                    await handle_manage_vocabulary(
                        {"action": action},
                        settings=None,  # type: ignore[arg-type]
                        storage=None,  # type: ignore[arg-type]
                        embedding_provider=None,  # type: ignore[arg-type]
                    )
            else:
                # These need a term
                with pytest.raises(RetrievalError, match="term is required"):
                    await handle_manage_vocabulary(
                        {"action": action},
                        settings=None,  # type: ignore[arg-type]
                        storage=None,  # type: ignore[arg-type]
                        embedding_provider=None,  # type: ignore[arg-type]
                    )


class TestBuildProvenanceRules:
    """Test the provenance rule builder."""

    def test_primary_rules(self) -> None:
        rules = _build_provenance_rules("primary")
        assert rules["voice_eligible"] is True
        assert "author's own" in rules["presentation_guidance"]

    def test_secondary_rules(self) -> None:
        rules = _build_provenance_rules("secondary")
        assert rules["voice_eligible"] is False
        assert "critic" in rules["presentation_guidance"]

    def test_contextual_rules(self) -> None:
        rules = _build_provenance_rules("contextual")
        assert rules["voice_eligible"] is False
        assert "references" in rules["presentation_guidance"]

    def test_tertiary_rules(self) -> None:
        rules = _build_provenance_rules("tertiary")
        assert rules["voice_eligible"] is False
        assert "reference" in rules["presentation_guidance"]

    def test_personal_rules(self) -> None:
        rules = _build_provenance_rules("personal")
        assert rules["voice_eligible"] is False
        assert "user" in rules["presentation_guidance"].lower()

    def test_unknown_defaults_to_secondary(self) -> None:
        rules = _build_provenance_rules("unknown_class")
        assert rules["voice_eligible"] is False


# ---------------------------------------------------------------------------
# search_chunks filter tests — speaker, pass_number, themes
# ---------------------------------------------------------------------------


class TestSearchChunksSpeakerFilter:
    """Verify that the speaker filter is actually applied to search results."""

    @pytest.mark.asyncio
    async def test_speaker_filter_returns_only_matching_speaker(self) -> None:
        """When speaker filter is set, only results with that speaker are returned."""
        results = _make_results()
        storage = _make_storage_mock()

        with (
            patch(
                "author_library.tools.composable_query.vector_search",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch(
                "author_library.tools.composable_query.keyword_search",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "author_library.tools.composable_query.GraphQueryService",
                return_value=_make_graph_service_mock(),
            ),
        ):
            raw = await handle_search_chunks(
                {
                    "query": "Lewis moral law",
                    "filters": {"speaker": "J.R.R. Tolkien"},
                    "include_passage_links": False,
                },
                settings=None,  # type: ignore[arg-type]
                storage=storage,
                embedding_provider=MagicMock(),
            )

        data = json.loads(raw)
        assert len(data["results"]) == 1
        assert data["results"][0]["chunk_id"] == str(_CHUNK_IDS[2])

    @pytest.mark.asyncio
    async def test_speaker_filter_excludes_chunks_without_speaker(self) -> None:
        """Chunks without a speaker in metadata are excluded by speaker filter."""
        results = _make_results()
        storage = _make_storage_mock()

        with (
            patch(
                "author_library.tools.composable_query.vector_search",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch(
                "author_library.tools.composable_query.keyword_search",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "author_library.tools.composable_query.GraphQueryService",
                return_value=_make_graph_service_mock(),
            ),
        ):
            raw = await handle_search_chunks(
                {
                    "query": "Lewis moral law",
                    "filters": {"speaker": "Nonexistent Speaker"},
                    "include_passage_links": False,
                },
                settings=None,  # type: ignore[arg-type]
                storage=storage,
                embedding_provider=MagicMock(),
            )

        data = json.loads(raw)
        assert len(data["results"]) == 0

    @pytest.mark.asyncio
    async def test_no_speaker_filter_returns_all(self) -> None:
        """Without a speaker filter, all results are returned."""
        results = _make_results()
        storage = _make_storage_mock()

        with (
            patch(
                "author_library.tools.composable_query.vector_search",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch(
                "author_library.tools.composable_query.keyword_search",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "author_library.tools.composable_query.GraphQueryService",
                return_value=_make_graph_service_mock(),
            ),
        ):
            raw = await handle_search_chunks(
                {
                    "query": "Lewis moral law",
                    "include_passage_links": False,
                },
                settings=None,  # type: ignore[arg-type]
                storage=storage,
                embedding_provider=MagicMock(),
            )

        data = json.loads(raw)
        assert len(data["results"]) == 5


class TestSearchChunksPassNumberFilter:
    """Verify that the pass_number filter is actually applied to search results."""

    @pytest.mark.asyncio
    async def test_pass_number_filter_returns_only_matching_pass(self) -> None:
        """When pass_number filter is set, only matching chunks are returned."""
        results = _make_results()
        storage = _make_storage_mock()

        with (
            patch(
                "author_library.tools.composable_query.vector_search",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch(
                "author_library.tools.composable_query.keyword_search",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "author_library.tools.composable_query.GraphQueryService",
                return_value=_make_graph_service_mock(),
            ),
        ):
            raw = await handle_search_chunks(
                {
                    "query": "Lewis moral law",
                    "filters": {"pass_number": 2},
                    "include_passage_links": False,
                },
                settings=None,  # type: ignore[arg-type]
                storage=storage,
                embedding_provider=MagicMock(),
            )

        data = json.loads(raw)
        assert len(data["results"]) == 1
        assert data["results"][0]["chunk_id"] == str(_CHUNK_IDS[1])

    @pytest.mark.asyncio
    async def test_pass_number_filter_zero_matches(self) -> None:
        """A pass_number with no matching chunks returns empty results."""
        results = _make_results()
        storage = _make_storage_mock()

        with (
            patch(
                "author_library.tools.composable_query.vector_search",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch(
                "author_library.tools.composable_query.keyword_search",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "author_library.tools.composable_query.GraphQueryService",
                return_value=_make_graph_service_mock(),
            ),
        ):
            raw = await handle_search_chunks(
                {
                    "query": "Lewis moral law",
                    "filters": {"pass_number": 99},
                    "include_passage_links": False,
                },
                settings=None,  # type: ignore[arg-type]
                storage=storage,
                embedding_provider=MagicMock(),
            )

        data = json.loads(raw)
        assert len(data["results"]) == 0

    @pytest.mark.asyncio
    async def test_pass_number_filter_multiple_matches(self) -> None:
        """pass_number=1 matches the three chunks with that pass number."""
        results = _make_results()
        storage = _make_storage_mock()

        with (
            patch(
                "author_library.tools.composable_query.vector_search",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch(
                "author_library.tools.composable_query.keyword_search",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "author_library.tools.composable_query.GraphQueryService",
                return_value=_make_graph_service_mock(),
            ),
        ):
            raw = await handle_search_chunks(
                {
                    "query": "Lewis moral law",
                    "filters": {"pass_number": 1},
                    "include_passage_links": False,
                },
                settings=None,  # type: ignore[arg-type]
                storage=storage,
                embedding_provider=MagicMock(),
            )

        data = json.loads(raw)
        # _CHUNK_IDS[0], [2], [3] all have pass_number=1
        assert len(data["results"]) == 3
        returned_ids = {r["chunk_id"] for r in data["results"]}
        assert str(_CHUNK_IDS[0]) in returned_ids
        assert str(_CHUNK_IDS[2]) in returned_ids
        assert str(_CHUNK_IDS[3]) in returned_ids


class TestSearchChunksThemesFilter:
    """Verify that the themes filter is actually applied to search results."""

    @pytest.mark.asyncio
    async def test_themes_filter_includes_matching_chunks(self) -> None:
        """When themes filter is set, only chunks with matching themes are returned."""
        results = _make_results()
        storage = _make_storage_mock()

        # Configure get_themes_for_chunk to return themes for specific chunks
        async def _get_themes(chunk_id: str) -> list[dict[str, Any]]:
            themes_map: dict[str, list[dict[str, Any]]] = {
                str(_CHUNK_IDS[0]): [
                    {"name": "Moral Law", "canonical_name": "moral-law"},
                ],
                str(_CHUNK_IDS[1]): [
                    {"name": "Glory", "canonical_name": "glory"},
                    {"name": "Heaven", "canonical_name": "heaven"},
                ],
                str(_CHUNK_IDS[2]): [
                    {"name": "Friendship", "canonical_name": "friendship"},
                ],
            }
            return themes_map.get(chunk_id, [])

        storage.graph.get_themes_for_chunk = AsyncMock(side_effect=_get_themes)

        with (
            patch(
                "author_library.tools.composable_query.vector_search",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch(
                "author_library.tools.composable_query.keyword_search",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "author_library.tools.composable_query.GraphQueryService",
                return_value=_make_graph_service_mock(),
            ),
        ):
            raw = await handle_search_chunks(
                {
                    "query": "Lewis moral law",
                    "filters": {"themes": ["moral-law"]},
                    "include_passage_links": False,
                },
                settings=None,  # type: ignore[arg-type]
                storage=storage,
                embedding_provider=MagicMock(),
            )

        data = json.loads(raw)
        assert len(data["results"]) == 1
        assert data["results"][0]["chunk_id"] == str(_CHUNK_IDS[0])

    @pytest.mark.asyncio
    async def test_themes_filter_case_insensitive(self) -> None:
        """Theme filtering is case-insensitive."""
        results = _make_results()
        storage = _make_storage_mock()

        async def _get_themes(chunk_id: str) -> list[dict[str, Any]]:
            if chunk_id == str(_CHUNK_IDS[1]):
                return [{"name": "Glory", "canonical_name": "glory"}]
            return []

        storage.graph.get_themes_for_chunk = AsyncMock(side_effect=_get_themes)

        with (
            patch(
                "author_library.tools.composable_query.vector_search",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch(
                "author_library.tools.composable_query.keyword_search",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "author_library.tools.composable_query.GraphQueryService",
                return_value=_make_graph_service_mock(),
            ),
        ):
            raw = await handle_search_chunks(
                {
                    "query": "Lewis sermon",
                    "filters": {"themes": ["Glory"]},
                    "include_passage_links": False,
                },
                settings=None,  # type: ignore[arg-type]
                storage=storage,
                embedding_provider=MagicMock(),
            )

        data = json.loads(raw)
        assert len(data["results"]) == 1
        assert data["results"][0]["chunk_id"] == str(_CHUNK_IDS[1])

    @pytest.mark.asyncio
    async def test_themes_filter_multiple_themes_or_match(self) -> None:
        """Multiple themes filter uses OR — chunks matching any theme are included."""
        results = _make_results()
        storage = _make_storage_mock()

        async def _get_themes(chunk_id: str) -> list[dict[str, Any]]:
            themes_map: dict[str, list[dict[str, Any]]] = {
                str(_CHUNK_IDS[0]): [
                    {"name": "Moral Law", "canonical_name": "moral-law"},
                ],
                str(_CHUNK_IDS[2]): [
                    {"name": "Friendship", "canonical_name": "friendship"},
                ],
            }
            return themes_map.get(chunk_id, [])

        storage.graph.get_themes_for_chunk = AsyncMock(side_effect=_get_themes)

        with (
            patch(
                "author_library.tools.composable_query.vector_search",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch(
                "author_library.tools.composable_query.keyword_search",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "author_library.tools.composable_query.GraphQueryService",
                return_value=_make_graph_service_mock(),
            ),
        ):
            raw = await handle_search_chunks(
                {
                    "query": "Lewis",
                    "filters": {"themes": ["moral-law", "friendship"]},
                    "include_passage_links": False,
                },
                settings=None,  # type: ignore[arg-type]
                storage=storage,
                embedding_provider=MagicMock(),
            )

        data = json.loads(raw)
        assert len(data["results"]) == 2
        returned_ids = {r["chunk_id"] for r in data["results"]}
        assert str(_CHUNK_IDS[0]) in returned_ids
        assert str(_CHUNK_IDS[2]) in returned_ids

    @pytest.mark.asyncio
    async def test_no_themes_filter_returns_all(self) -> None:
        """Without a themes filter, all results are returned (no theme filtering)."""
        results = _make_results()
        storage = _make_storage_mock()

        with (
            patch(
                "author_library.tools.composable_query.vector_search",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch(
                "author_library.tools.composable_query.keyword_search",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "author_library.tools.composable_query.GraphQueryService",
                return_value=_make_graph_service_mock(),
            ),
        ):
            raw = await handle_search_chunks(
                {
                    "query": "Lewis moral law",
                    "include_passage_links": False,
                },
                settings=None,  # type: ignore[arg-type]
                storage=storage,
                embedding_provider=MagicMock(),
            )

        data = json.loads(raw)
        # All 5 results should come through when no themes filter is active
        assert len(data["results"]) == 5


class TestSearchChunksCombinedFilters:
    """Verify that multiple filters combine correctly (AND semantics)."""

    @pytest.mark.asyncio
    async def test_speaker_and_pass_number_combined(self) -> None:
        """Speaker + pass_number filters combine with AND semantics."""
        results = _make_results()
        storage = _make_storage_mock()

        with (
            patch(
                "author_library.tools.composable_query.vector_search",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch(
                "author_library.tools.composable_query.keyword_search",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "author_library.tools.composable_query.GraphQueryService",
                return_value=_make_graph_service_mock(),
            ),
        ):
            raw = await handle_search_chunks(
                {
                    "query": "Lewis",
                    "filters": {"speaker": "C.S. Lewis", "pass_number": 2},
                    "include_passage_links": False,
                },
                settings=None,  # type: ignore[arg-type]
                storage=storage,
                embedding_provider=MagicMock(),
            )

        data = json.loads(raw)
        # Only chunk[1] has speaker="C.S. Lewis" AND pass_number=2
        assert len(data["results"]) == 1
        assert data["results"][0]["chunk_id"] == str(_CHUNK_IDS[1])
