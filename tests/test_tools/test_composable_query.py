"""Tests for composable query tool handlers — input validation and error paths."""

from __future__ import annotations

import pytest

from author_library.errors import RetrievalError
from author_library.tools.composable_query import (
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
