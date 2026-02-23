"""Tests for the chunking strategy registry."""

from __future__ import annotations

import pytest

from author_library.chunking import (
    get_chunking_strategy,
    list_strategies,
)
from author_library.chunking.correspondence import (
    BlogStrategy,
    InterviewStrategy,
    LetterStrategy,
)
from author_library.chunking.poetry import PoetryStrategy
from author_library.chunking.scholarly import ScholarlyProseStrategy
from author_library.chunking.sermon import SermonStrategy
from author_library.chunking.transcript import TranscriptChunkingStrategy
from author_library.errors import IngestionError


class TestGetChunkingStrategy:
    def test_scholarly_prose(self) -> None:
        strategy = get_chunking_strategy(["scholarly_prose"])
        assert isinstance(strategy, ScholarlyProseStrategy)

    def test_monograph(self) -> None:
        strategy = get_chunking_strategy(["monograph"])
        assert isinstance(strategy, ScholarlyProseStrategy)

    def test_poetry(self) -> None:
        strategy = get_chunking_strategy(["poetry"])
        assert isinstance(strategy, PoetryStrategy)

    def test_sermon(self) -> None:
        strategy = get_chunking_strategy(["sermon"])
        assert isinstance(strategy, SermonStrategy)

    def test_lecture(self) -> None:
        strategy = get_chunking_strategy(["lecture"])
        assert isinstance(strategy, SermonStrategy)

    def test_letters(self) -> None:
        strategy = get_chunking_strategy(["letters"])
        assert isinstance(strategy, LetterStrategy)

    def test_correspondence(self) -> None:
        strategy = get_chunking_strategy(["correspondence"])
        assert isinstance(strategy, LetterStrategy)

    def test_blog(self) -> None:
        strategy = get_chunking_strategy(["blog"])
        assert isinstance(strategy, BlogStrategy)

    def test_interview(self) -> None:
        strategy = get_chunking_strategy(["interview"])
        assert isinstance(strategy, InterviewStrategy)

    def test_multiple_tags_first_match_wins(self) -> None:
        # Poetry should take priority over scholarly_prose when both present
        strategy = get_chunking_strategy(["poetry", "scholarly_prose"])
        assert isinstance(strategy, PoetryStrategy)

    def test_unknown_genre_falls_back_to_scholarly(self) -> None:
        strategy = get_chunking_strategy(["unknown_genre_xyz"])
        assert isinstance(strategy, ScholarlyProseStrategy)

    def test_empty_tags_raises(self) -> None:
        with pytest.raises(IngestionError, match="no genre tags"):
            get_chunking_strategy([])

    def test_theology_uses_scholarly(self) -> None:
        strategy = get_chunking_strategy(["theology"])
        assert isinstance(strategy, ScholarlyProseStrategy)


class TestListStrategies:
    def test_returns_all_strategies(self) -> None:
        strategies = list_strategies()
        types = {type(s) for s in strategies}
        assert PoetryStrategy in types
        assert TranscriptChunkingStrategy in types
        assert InterviewStrategy in types
        assert LetterStrategy in types
        assert BlogStrategy in types
        assert SermonStrategy in types
        assert ScholarlyProseStrategy in types

    def test_strategy_count(self) -> None:
        assert len(list_strategies()) == 7
