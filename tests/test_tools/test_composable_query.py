"""Tests for composable query tool handlers — input validation, error paths, and filters."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from author_library.errors import RetrievalError
from author_library.retrieval.models import RetrievalResult
from author_library.tools.composable_query import (
    _apply_chunk_metadata_filters,
    _batch_fetch_chunk_metadata,
    _batch_fetch_chunk_themes,
    _build_provenance_rules,
    handle_get_passage_links,
    handle_manage_vocabulary,
    handle_search_chunks,
)


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
# Helpers for filter tests
# ---------------------------------------------------------------------------

_UUID_A = UUID("00000000-0000-0000-0000-000000000001")
_UUID_B = UUID("00000000-0000-0000-0000-000000000002")
_UUID_C = UUID("00000000-0000-0000-0000-000000000003")


def _make_result(
    chunk_id: UUID,
    work_id: str = "work-1",
    text: str = "sample text",
    score: float = 0.9,
    granularity: str = "meso",
    source_class: str = "primary",
    source: str = "vector",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        work_id=work_id,
        text=text,
        score=score,
        granularity=granularity,
        source_class=source_class,
        source=source,
    )


def _mock_storage(
    pg_rows: list[dict] | None = None,
    neo4j_records: list[dict] | None = None,
) -> MagicMock:
    """Build a mock StorageManager with stubbed PG and Neo4j."""
    storage = MagicMock()
    storage.pg = MagicMock()
    storage.pg.fetch_all = AsyncMock(return_value=pg_rows or [])
    storage.neo4j = MagicMock()
    storage.neo4j.execute_read = AsyncMock(return_value=neo4j_records or [])
    storage.works = MagicMock()
    storage.works.get = AsyncMock(return_value={"title": "Test Work", "author": "Author"})
    storage.embeddings = MagicMock()
    storage.graph = MagicMock()
    return storage


# ---------------------------------------------------------------------------
# Tests for _apply_chunk_metadata_filters
# ---------------------------------------------------------------------------


class TestApplyChunkMetadataFilters:
    """Test the post-search metadata filter logic."""

    async def test_speaker_filter_keeps_matching(self) -> None:
        """Chunks with matching speaker are kept."""
        results = [_make_result(_UUID_A), _make_result(_UUID_B)]
        pg_rows = [
            {"chunk_id": str(_UUID_A), "metadata": {"speaker": "Malcolm Guite"}, "pass_number": 1},
            {"chunk_id": str(_UUID_B), "metadata": {"speaker": "N.T. Wright"}, "pass_number": 1},
        ]
        storage = _mock_storage(pg_rows=pg_rows)

        filtered = await _apply_chunk_metadata_filters(
            results,
            storage=storage,
            speaker_filter="Malcolm Guite",
        )

        assert len(filtered) == 1
        assert str(filtered[0].chunk_id) == str(_UUID_A)

    async def test_speaker_filter_case_insensitive(self) -> None:
        """Speaker matching is case-insensitive."""
        results = [_make_result(_UUID_A)]
        pg_rows = [
            {"chunk_id": str(_UUID_A), "metadata": {"speaker": "Malcolm Guite"}, "pass_number": 1},
        ]
        storage = _mock_storage(pg_rows=pg_rows)

        filtered = await _apply_chunk_metadata_filters(
            results,
            storage=storage,
            speaker_filter="malcolm guite",
        )

        assert len(filtered) == 1

    async def test_speaker_filter_removes_no_speaker(self) -> None:
        """Chunks without a speaker in metadata are filtered out."""
        results = [_make_result(_UUID_A)]
        pg_rows = [
            {"chunk_id": str(_UUID_A), "metadata": {}, "pass_number": 1},
        ]
        storage = _mock_storage(pg_rows=pg_rows)

        filtered = await _apply_chunk_metadata_filters(
            results,
            storage=storage,
            speaker_filter="Malcolm Guite",
        )

        assert len(filtered) == 0

    async def test_pass_number_filter_keeps_matching(self) -> None:
        """Only chunks with the requested pass_number are kept."""
        results = [_make_result(_UUID_A), _make_result(_UUID_B), _make_result(_UUID_C)]
        pg_rows = [
            {"chunk_id": str(_UUID_A), "metadata": {}, "pass_number": 1},
            {"chunk_id": str(_UUID_B), "metadata": {}, "pass_number": 2},
            {"chunk_id": str(_UUID_C), "metadata": {}, "pass_number": 2},
        ]
        storage = _mock_storage(pg_rows=pg_rows)

        filtered = await _apply_chunk_metadata_filters(
            results,
            storage=storage,
            pass_number_filter=2,
        )

        assert len(filtered) == 2
        ids = {str(r.chunk_id) for r in filtered}
        assert str(_UUID_B) in ids
        assert str(_UUID_C) in ids

    async def test_pass_number_filter_zero(self) -> None:
        """pass_number=0 should not match chunks with pass_number=1."""
        results = [_make_result(_UUID_A)]
        pg_rows = [
            {"chunk_id": str(_UUID_A), "metadata": {}, "pass_number": 1},
        ]
        storage = _mock_storage(pg_rows=pg_rows)

        filtered = await _apply_chunk_metadata_filters(
            results,
            storage=storage,
            pass_number_filter=0,
        )

        assert len(filtered) == 0

    async def test_themes_filter_keeps_matching(self) -> None:
        """Chunks exploring at least one requested theme are kept."""
        results = [_make_result(_UUID_A), _make_result(_UUID_B)]
        neo4j_records = [
            {"chunk_id": str(_UUID_A), "canonical_name": "imagination", "name": "Imagination"},
            {"chunk_id": str(_UUID_A), "canonical_name": "prayer", "name": "Prayer"},
            {"chunk_id": str(_UUID_B), "canonical_name": "poetry", "name": "Poetry"},
        ]
        storage = _mock_storage(neo4j_records=neo4j_records)

        filtered = await _apply_chunk_metadata_filters(
            results,
            storage=storage,
            themes_filter=["imagination"],
        )

        assert len(filtered) == 1
        assert str(filtered[0].chunk_id) == str(_UUID_A)

    async def test_themes_filter_case_insensitive(self) -> None:
        """Theme matching is case-insensitive."""
        results = [_make_result(_UUID_A)]
        neo4j_records = [
            {"chunk_id": str(_UUID_A), "canonical_name": "imagination", "name": "Imagination"},
        ]
        storage = _mock_storage(neo4j_records=neo4j_records)

        filtered = await _apply_chunk_metadata_filters(
            results,
            storage=storage,
            themes_filter=["Imagination"],
        )

        assert len(filtered) == 1

    async def test_themes_filter_any_match(self) -> None:
        """If chunk explores ANY of the requested themes, it is kept."""
        results = [_make_result(_UUID_A)]
        neo4j_records = [
            {"chunk_id": str(_UUID_A), "canonical_name": "prayer", "name": "Prayer"},
        ]
        storage = _mock_storage(neo4j_records=neo4j_records)

        filtered = await _apply_chunk_metadata_filters(
            results,
            storage=storage,
            themes_filter=["imagination", "prayer"],
        )

        assert len(filtered) == 1

    async def test_themes_filter_no_themes_removes(self) -> None:
        """Chunks with no theme associations are filtered out."""
        results = [_make_result(_UUID_A)]
        # No neo4j records for this chunk
        storage = _mock_storage(neo4j_records=[])

        filtered = await _apply_chunk_metadata_filters(
            results,
            storage=storage,
            themes_filter=["imagination"],
        )

        assert len(filtered) == 0

    async def test_combined_filters_all_must_pass(self) -> None:
        """When multiple filters are set, ALL must pass for a result to be kept."""
        results = [_make_result(_UUID_A), _make_result(_UUID_B)]
        pg_rows = [
            {"chunk_id": str(_UUID_A), "metadata": {"speaker": "Malcolm Guite"}, "pass_number": 2},
            {"chunk_id": str(_UUID_B), "metadata": {"speaker": "Malcolm Guite"}, "pass_number": 1},
        ]
        neo4j_records = [
            {"chunk_id": str(_UUID_A), "canonical_name": "imagination", "name": "Imagination"},
            {"chunk_id": str(_UUID_B), "canonical_name": "imagination", "name": "Imagination"},
        ]
        storage = _mock_storage(pg_rows=pg_rows, neo4j_records=neo4j_records)

        filtered = await _apply_chunk_metadata_filters(
            results,
            storage=storage,
            speaker_filter="Malcolm Guite",
            themes_filter=["imagination"],
            pass_number_filter=2,
        )

        # Only UUID_A matches all three filters
        assert len(filtered) == 1
        assert str(filtered[0].chunk_id) == str(_UUID_A)

    async def test_no_filters_returns_all(self) -> None:
        """When no filters are provided, all results are returned unchanged."""
        results = [_make_result(_UUID_A), _make_result(_UUID_B)]
        storage = _mock_storage()

        filtered = await _apply_chunk_metadata_filters(
            results,
            storage=storage,
        )

        assert len(filtered) == 2

    async def test_empty_results_returns_empty(self) -> None:
        """Empty input returns empty output."""
        storage = _mock_storage()

        filtered = await _apply_chunk_metadata_filters(
            [],
            storage=storage,
            speaker_filter="test",
        )

        assert filtered == []

    async def test_neo4j_failure_graceful_degradation(self) -> None:
        """If Neo4j theme query fails, themes filter removes all results."""
        results = [_make_result(_UUID_A)]
        storage = _mock_storage()
        storage.neo4j.execute_read = AsyncMock(side_effect=Exception("Neo4j down"))

        filtered = await _apply_chunk_metadata_filters(
            results,
            storage=storage,
            themes_filter=["imagination"],
        )

        # When Neo4j is down, no themes are found, so nothing matches
        assert len(filtered) == 0

    async def test_metadata_as_json_string(self) -> None:
        """metadata stored as JSON string is parsed correctly."""
        results = [_make_result(_UUID_A)]
        pg_rows = [
            {"chunk_id": str(_UUID_A), "metadata": '{"speaker": "Malcolm Guite"}', "pass_number": 1},
        ]
        storage = _mock_storage(pg_rows=pg_rows)

        filtered = await _apply_chunk_metadata_filters(
            results,
            storage=storage,
            speaker_filter="Malcolm Guite",
        )

        assert len(filtered) == 1


# ---------------------------------------------------------------------------
# Tests for _batch_fetch_chunk_metadata
# ---------------------------------------------------------------------------


class TestBatchFetchChunkMetadata:
    """Test batch PG metadata fetching."""

    async def test_empty_input_returns_empty(self) -> None:
        storage = _mock_storage()
        result = await _batch_fetch_chunk_metadata(storage, [])
        assert result == {}

    async def test_returns_metadata_with_pass_number(self) -> None:
        pg_rows = [
            {"chunk_id": str(_UUID_A), "metadata": {"speaker": "Guite"}, "pass_number": 3},
        ]
        storage = _mock_storage(pg_rows=pg_rows)

        result = await _batch_fetch_chunk_metadata(storage, [str(_UUID_A)])

        assert str(_UUID_A) in result
        assert result[str(_UUID_A)]["speaker"] == "Guite"
        assert result[str(_UUID_A)]["pass_number"] == 3


# ---------------------------------------------------------------------------
# Tests for _batch_fetch_chunk_themes
# ---------------------------------------------------------------------------


class TestBatchFetchChunkThemes:
    """Test batch Neo4j theme fetching."""

    async def test_empty_input_returns_empty(self) -> None:
        storage = _mock_storage()
        result = await _batch_fetch_chunk_themes(storage, [])
        assert result == {}

    async def test_returns_lowercase_themes(self) -> None:
        neo4j_records = [
            {"chunk_id": str(_UUID_A), "canonical_name": "Imagination", "name": "Imagination"},
        ]
        storage = _mock_storage(neo4j_records=neo4j_records)

        result = await _batch_fetch_chunk_themes(storage, [str(_UUID_A)])

        assert str(_UUID_A) in result
        assert "imagination" in result[str(_UUID_A)]

    async def test_neo4j_failure_returns_empty(self) -> None:
        storage = _mock_storage()
        storage.neo4j.execute_read = AsyncMock(side_effect=Exception("down"))

        result = await _batch_fetch_chunk_themes(storage, [str(_UUID_A)])

        assert result == {}


# ---------------------------------------------------------------------------
# Integration: handle_search_chunks with filters
# ---------------------------------------------------------------------------


class TestSearchChunksFilterIntegration:
    """Test that handle_search_chunks applies speaker, themes, and pass_number filters."""

    async def _run_search(
        self,
        arguments: dict,
        vector_results: list[RetrievalResult] | None = None,
        keyword_results: list[RetrievalResult] | None = None,
        pg_metadata_rows: list[dict] | None = None,
        neo4j_theme_records: list[dict] | None = None,
    ) -> dict:
        """Run handle_search_chunks with mocked dependencies."""
        storage = _mock_storage(
            pg_rows=pg_metadata_rows or [],
            neo4j_records=neo4j_theme_records or [],
        )
        embedding_provider = MagicMock()
        settings = MagicMock()

        # Mock the engagement chain (passage links)
        mock_chain = MagicMock()
        mock_chain.links = []

        with (
            patch(
                "author_library.tools.composable_query.vector_search",
                new_callable=AsyncMock,
                return_value=vector_results or [],
            ),
            patch(
                "author_library.tools.composable_query.keyword_search",
                new_callable=AsyncMock,
                return_value=keyword_results or [],
            ),
            patch(
                "author_library.tools.composable_query.GraphQueryService",
            ) as MockGraphService,
        ):
            MockGraphService.return_value.get_engagement_chain = AsyncMock(
                return_value=mock_chain,
            )

            result_json = await handle_search_chunks(
                arguments,
                settings=settings,
                storage=storage,
                embedding_provider=embedding_provider,
            )

        return json.loads(result_json)

    async def test_speaker_filter_applied(self) -> None:
        """search_chunks with speaker filter only returns matching chunks."""
        results = [
            _make_result(_UUID_A, text="Guite speaks"),
            _make_result(_UUID_B, text="Wright speaks"),
        ]
        pg_rows = [
            {"chunk_id": str(_UUID_A), "metadata": {"speaker": "Malcolm Guite"}, "pass_number": 1},
            {"chunk_id": str(_UUID_B), "metadata": {"speaker": "N.T. Wright"}, "pass_number": 1},
        ]

        parsed = await self._run_search(
            {"query": "test", "filters": {"speaker": "Malcolm Guite"}},
            vector_results=results,
            pg_metadata_rows=pg_rows,
        )

        assert parsed["total_available"] == 1
        assert parsed["results"][0]["text"] == "Guite speaks"

    async def test_pass_number_filter_applied(self) -> None:
        """search_chunks with pass_number filter only returns matching chunks."""
        results = [
            _make_result(_UUID_A, text="pass 1"),
            _make_result(_UUID_B, text="pass 2"),
        ]
        pg_rows = [
            {"chunk_id": str(_UUID_A), "metadata": {}, "pass_number": 1},
            {"chunk_id": str(_UUID_B), "metadata": {}, "pass_number": 2},
        ]

        parsed = await self._run_search(
            {"query": "test", "filters": {"pass_number": 2}},
            vector_results=results,
            pg_metadata_rows=pg_rows,
        )

        assert parsed["total_available"] == 1
        assert parsed["results"][0]["text"] == "pass 2"

    async def test_themes_filter_applied(self) -> None:
        """search_chunks with themes filter only returns chunks exploring those themes."""
        results = [
            _make_result(_UUID_A, text="about imagination"),
            _make_result(_UUID_B, text="about poetry"),
        ]
        neo4j_records = [
            {"chunk_id": str(_UUID_A), "canonical_name": "imagination", "name": "Imagination"},
            {"chunk_id": str(_UUID_B), "canonical_name": "poetry", "name": "Poetry"},
        ]

        parsed = await self._run_search(
            {"query": "test", "filters": {"themes": ["imagination"]}},
            vector_results=results,
            neo4j_theme_records=neo4j_records,
        )

        assert parsed["total_available"] == 1
        assert parsed["results"][0]["text"] == "about imagination"

    async def test_no_filters_returns_all(self) -> None:
        """search_chunks without metadata filters returns all results."""
        results = [
            _make_result(_UUID_A, text="chunk A"),
            _make_result(_UUID_B, text="chunk B"),
        ]

        parsed = await self._run_search(
            {"query": "test"},
            vector_results=results,
        )

        assert parsed["total_available"] == 2

    async def test_combined_filters(self) -> None:
        """All filters can be applied simultaneously; all must pass."""
        results = [
            _make_result(_UUID_A, text="match all"),
            _make_result(_UUID_B, text="wrong pass"),
            _make_result(_UUID_C, text="wrong speaker"),
        ]
        pg_rows = [
            {"chunk_id": str(_UUID_A), "metadata": {"speaker": "Guite"}, "pass_number": 2},
            {"chunk_id": str(_UUID_B), "metadata": {"speaker": "Guite"}, "pass_number": 1},
            {"chunk_id": str(_UUID_C), "metadata": {"speaker": "Wright"}, "pass_number": 2},
        ]
        neo4j_records = [
            {"chunk_id": str(_UUID_A), "canonical_name": "imagination", "name": "Imagination"},
            {"chunk_id": str(_UUID_B), "canonical_name": "imagination", "name": "Imagination"},
            {"chunk_id": str(_UUID_C), "canonical_name": "imagination", "name": "Imagination"},
        ]

        parsed = await self._run_search(
            {
                "query": "test",
                "filters": {
                    "speaker": "Guite",
                    "themes": ["imagination"],
                    "pass_number": 2,
                },
            },
            vector_results=results,
            pg_metadata_rows=pg_rows,
            neo4j_theme_records=neo4j_records,
        )

        assert parsed["total_available"] == 1
        assert parsed["results"][0]["text"] == "match all"
