"""Tests for the error hierarchy."""

from __future__ import annotations

import pytest

from author_library.errors import (
    AuthorLibraryError,
    ClassificationError,
    ConfigurationError,
    EmbeddingError,
    IngestionError,
    ParsingError,
    RetrievalError,
    StorageError,
)


class TestErrorHierarchy:
    def test_all_errors_inherit_from_base(self) -> None:
        subclasses = [
            IngestionError,
            ClassificationError,
            RetrievalError,
            EmbeddingError,
            StorageError,
            ConfigurationError,
            ParsingError,
        ]
        for cls in subclasses:
            assert issubclass(cls, AuthorLibraryError)
            assert issubclass(cls, Exception)

    def test_message(self) -> None:
        err = AuthorLibraryError("something went wrong")
        assert str(err) == "something went wrong"

    def test_context_default(self) -> None:
        err = AuthorLibraryError("fail")
        assert err.context == {}

    def test_context_provided(self) -> None:
        ctx = {"file": "test.md", "line": 42}
        err = IngestionError("parse failed", context=ctx)
        assert err.context == ctx

    def test_cause_chaining(self) -> None:
        original = ValueError("bad value")
        err = StorageError("db write failed", cause=original)
        assert err.__cause__ is original

    def test_can_be_caught_as_base(self) -> None:
        with pytest.raises(AuthorLibraryError):
            raise ClassificationError("unknown genre")
