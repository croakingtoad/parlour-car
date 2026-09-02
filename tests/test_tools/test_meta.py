"""Tests for meta tool handlers — input validation."""

from __future__ import annotations

import pytest

from author_library.errors import RetrievalError
from author_library.tools.meta import (
    _format_year_range,
    handle_author_bio,
    handle_list_works,
)


def test_undated_year_range_has_explicit_label() -> None:
    assert _format_year_range(None, None) == "undated"
    assert _format_year_range(1990, 2020) == "1990-2020"


class TestHandleAuthorBioValidation:
    """Validate required argument checks for author_bio."""

    async def test_missing_author_id_raises(self) -> None:
        with pytest.raises(RetrievalError, match="author_id is required"):
            await handle_author_bio(
                {},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
            )

    async def test_empty_author_id_raises(self) -> None:
        with pytest.raises(RetrievalError, match="author_id is required"):
            await handle_author_bio(
                {"author_id": ""},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
            )


class TestHandleListWorksValidation:
    """Validate required argument checks for list_works."""

    async def test_missing_author_id_raises(self) -> None:
        with pytest.raises(RetrievalError, match="author_id is required"):
            await handle_list_works(
                {},
                storage=None,  # type: ignore[arg-type]
            )
