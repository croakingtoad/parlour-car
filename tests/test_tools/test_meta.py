"""Tests for meta tool handlers — input validation."""

from __future__ import annotations

import pytest

from author_library.errors import RetrievalError
from author_library.tools.meta import (
    handle_author_bio,
    handle_list_works,
)


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
