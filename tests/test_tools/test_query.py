"""Tests for query tool handlers — validation, style helpers, result formatting."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from author_library.errors import RetrievalError
from author_library.tools.query import (
    _result_to_quote,
    _style_instruction,
    handle_ask_author,
    handle_compare_ideas,
    handle_find_quotes,
    handle_trace_theme,
)

# ---------------------------------------------------------------------------
# _style_instruction helper
# ---------------------------------------------------------------------------


class TestStyleInstruction:
    """Test response style instruction lookup."""

    def test_conversational(self) -> None:
        inst = _style_instruction("conversational")
        assert "conversational" in inst.lower() or "warm" in inst.lower()

    def test_academic(self) -> None:
        inst = _style_instruction("academic")
        assert "scholarly" in inst.lower() or "analytical" in inst.lower()

    def test_devotional(self) -> None:
        inst = _style_instruction("devotional")
        assert "devotional" in inst.lower() or "contemplative" in inst.lower()

    def test_lecture(self) -> None:
        inst = _style_instruction("lecture")
        assert "lecture" in inst.lower()

    def test_unknown_falls_back_to_conversational(self) -> None:
        inst = _style_instruction("unknown_style")
        conversational = _style_instruction("conversational")
        assert inst == conversational

    def test_all_styles_return_non_empty_string(self) -> None:
        for style in ["conversational", "academic", "devotional", "lecture"]:
            assert len(_style_instruction(style)) > 20


# ---------------------------------------------------------------------------
# _result_to_quote helper
# ---------------------------------------------------------------------------


class TestResultToQuote:
    """Test conversion of a RetrievalResult-like object to quote dict."""

    def test_converts_all_fields(self) -> None:
        cid = uuid4()
        result = SimpleNamespace(
            chunk_id=cid,
            work_id="lewis--mere-christianity",
            text="I believe in Christianity as I believe that the Sun has risen.",
            source_class="primary",
            granularity="micro",
            score=0.92345,
        )
        quote = _result_to_quote(result, match_type="phrase")
        assert quote["chunk_id"] == str(cid)
        assert quote["work_id"] == "lewis--mere-christianity"
        assert quote["text"].startswith("I believe in Christianity")
        assert quote["source_class"] == "primary"
        assert quote["granularity"] == "micro"
        assert quote["score"] == 0.9234  # rounded to 4 decimal places
        assert quote["match_type"] == "phrase"

    def test_semantic_match_type(self) -> None:
        result = SimpleNamespace(
            chunk_id=uuid4(),
            work_id="lewis--the-weight-of-glory",
            text="If we find ourselves with a desire...",
            source_class="primary",
            granularity="meso",
            score=0.8871,
        )
        quote = _result_to_quote(result, match_type="semantic")
        assert quote["match_type"] == "semantic"
        assert quote["score"] == 0.8871

    def test_score_rounding(self) -> None:
        result = SimpleNamespace(
            chunk_id=uuid4(),
            work_id="w1",
            text="text",
            source_class="primary",
            granularity="micro",
            score=0.123456789,
        )
        quote = _result_to_quote(result, match_type="phrase")
        assert quote["score"] == 0.1235


# ---------------------------------------------------------------------------
# Handler input validation
# ---------------------------------------------------------------------------


class TestHandleAskAuthorValidation:
    """Validate required argument checks for ask_author."""

    async def test_missing_question_raises(self) -> None:
        with pytest.raises(RetrievalError, match="question is required"):
            await handle_ask_author(
                {"author_id": "cs-lewis"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_missing_author_id_raises(self) -> None:
        with pytest.raises(RetrievalError, match="author_id is required"):
            await handle_ask_author(
                {"question": "What is joy?"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_empty_question_raises(self) -> None:
        with pytest.raises(RetrievalError, match="question is required"):
            await handle_ask_author(
                {"question": "", "author_id": "cs-lewis"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )


class TestHandleTraceThemeValidation:
    """Validate required argument checks for trace_theme."""

    async def test_missing_theme_name_raises(self) -> None:
        with pytest.raises(RetrievalError, match="theme_name is required"):
            await handle_trace_theme(
                {},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )


class TestHandleFindQuotesValidation:
    """Validate required argument checks for find_quotes."""

    async def test_missing_query_raises(self) -> None:
        with pytest.raises(RetrievalError, match="query is required"):
            await handle_find_quotes(
                {},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )


class TestHandleCompareIdeasValidation:
    """Validate required argument checks for compare_ideas."""

    async def test_missing_topic_raises(self) -> None:
        with pytest.raises(RetrievalError, match="topic is required"):
            await handle_compare_ideas(
                {"author_ids": ["cs-lewis", "tolkien"]},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_missing_author_ids_raises(self) -> None:
        with pytest.raises(RetrievalError, match="author_ids requires at least 2"):
            await handle_compare_ideas(
                {"topic": "Joy"},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )

    async def test_single_author_id_raises(self) -> None:
        with pytest.raises(RetrievalError, match="author_ids requires at least 2"):
            await handle_compare_ideas(
                {"topic": "Joy", "author_ids": ["cs-lewis"]},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
                embedding_provider=None,  # type: ignore[arg-type]
            )
