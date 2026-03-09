"""Live integration tests for author/library MCP tools.

Tests: list_authors, author_bio, list_works, library_stats.

These tools are read-only DB operations — no LLM calls needed.
Data is inserted directly via storage repositories to keep tests
fast (no full ingestion pipeline).

Runs against the test database (author_library_test).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio

from author_library.config import Settings
from author_library.storage.manager import StorageManager
from author_library.tools.meta import (
    handle_author_bio,
    handle_library_stats,
    handle_list_authors,
    handle_list_works,
)

from .conftest import SKIP_NO_DB

if TYPE_CHECKING:
    from author_library.storage.manager import StorageManager as SM

# ---------------------------------------------------------------------------
# Test fixtures: insert minimal author data
# ---------------------------------------------------------------------------

_WORK_DEFAULTS: dict[str, Any] = {
    "source_class_note": "Ingested for testing",
    "publication_year": 2000,
    "publisher": "Test Publisher",
    "format_ingested": "txt",
    "word_count": 5000,
    "genre_tags": ["poetry"],
    "subject_headings": ["English poetry"],
    "source_metadata": {},
}


def _make_work(
    work_id: str,
    title: str,
    author: str,
    source_class: str = "primary",
    subject_author_id: str = "test-author",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a minimal work dict for insertion."""
    return {
        **_WORK_DEFAULTS,
        "work_id": work_id,
        "title": title,
        "author": author,
        "source_class": source_class,
        "source_metadata": {"subject_author_id": subject_author_id},
        **overrides,
    }


@pytest_asyncio.fixture
async def author_storage(clean_storage: SM) -> SM:
    """Storage seeded with two authors and three works."""
    # Author 1: "test-author" with 2 primary works
    await clean_storage.works.create(_make_work(
        "test--author-work-one",
        title="First Work",
        author="Test Author",
        publication_year=1990,
        word_count=10000,
    ))
    await clean_storage.works.create(_make_work(
        "test--author-work-two",
        title="Second Work",
        author="Test Author",
        source_class="secondary",
        subject_author_id="test-author",
        source_metadata={
            "about_author_id": "test-author",
            "external_author": "Other Critic",
            "relationship": "critical study",
        },
        publication_year=2005,
        word_count=3000,
    ))

    # Author 2: "other-author" with 1 primary work
    await clean_storage.works.create(_make_work(
        "test--other-author-work",
        title="Other Author Book",
        author="Other Author",
        subject_author_id="other-author",
        publication_year=2010,
        word_count=8000,
    ))

    yield clean_storage


# ---------------------------------------------------------------------------
# TestListAuthors
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestListAuthors:
    """list_authors returns all authors grouped from the works table."""

    async def test_list_authors_returns_all(self, author_storage: SM) -> None:
        """list_authors includes all authors from the works table."""
        result_str = await handle_list_authors({}, storage=author_storage)
        result = json.loads(result_str)

        assert "authors" in result
        assert "total_authors" in result
        assert result["total_authors"] == 2

        names = {a["author"] for a in result["authors"]}
        assert "Test Author" in names
        assert "Other Author" in names

    async def test_list_authors_work_count(self, author_storage: SM) -> None:
        """list_authors shows correct work count per author."""
        result_str = await handle_list_authors({}, storage=author_storage)
        result = json.loads(result_str)

        by_name = {a["author"]: a for a in result["authors"]}
        test_author = by_name["Test Author"]

        assert test_author["work_count"] == 2
        assert test_author["primary_works"] == 1
        assert test_author["secondary_works"] == 1

    async def test_list_authors_empty_library(self, clean_storage: SM) -> None:
        """list_authors returns empty list when no works are ingested."""
        result_str = await handle_list_authors({}, storage=clean_storage)
        result = json.loads(result_str)

        assert result["total_authors"] == 0
        assert result["authors"] == []

    async def test_list_authors_has_word_count(self, author_storage: SM) -> None:
        """list_authors includes total word counts per author."""
        result_str = await handle_list_authors({}, storage=author_storage)
        result = json.loads(result_str)

        by_name = {a["author"]: a for a in result["authors"]}
        # 10000 (primary) + 3000 (secondary)
        assert by_name["Test Author"]["total_words"] == 13000
        assert by_name["Other Author"]["total_words"] == 8000


# ---------------------------------------------------------------------------
# TestListWorks
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestListWorks:
    """list_works returns catalog of works for an author."""

    async def test_list_works_by_author_id(self, author_storage: SM) -> None:
        """list_works returns all works for the given author_id."""
        result_str = await handle_list_works(
            {"author_id": "test-author"}, storage=author_storage
        )
        result = json.loads(result_str)

        assert result["author_id"] == "test-author"
        assert result["total_works"] == 2
        assert len(result["works"]) == 2

    async def test_list_works_filter_by_source_class(
        self, author_storage: SM
    ) -> None:
        """list_works filter by source_class returns only matching works."""
        result_str = await handle_list_works(
            {"author_id": "test-author", "source_class": "primary"},
            storage=author_storage,
        )
        result = json.loads(result_str)

        assert result["total_works"] == 1
        assert result["filter"] == "primary"
        assert result["works"][0]["source_class"] == "primary"
        assert result["works"][0]["work_id"] == "test--author-work-one"

    async def test_list_works_no_match_returns_empty(
        self, author_storage: SM
    ) -> None:
        """list_works with unknown author_id returns empty list."""
        result_str = await handle_list_works(
            {"author_id": "nobody-here"}, storage=author_storage
        )
        result = json.loads(result_str)

        assert result["total_works"] == 0
        assert result["works"] == []

    async def test_list_works_missing_author_id_raises(
        self, author_storage: SM
    ) -> None:
        """list_works raises RetrievalError if author_id is missing."""
        from author_library.errors import RetrievalError

        with pytest.raises(RetrievalError, match="author_id is required"):
            await handle_list_works({}, storage=author_storage)

    async def test_list_works_includes_work_metadata(
        self, author_storage: SM
    ) -> None:
        """list_works returns title, source_class, word_count etc. per work."""
        result_str = await handle_list_works(
            {"author_id": "test-author", "source_class": "primary"},
            storage=author_storage,
        )
        result = json.loads(result_str)

        work = result["works"][0]
        assert "work_id" in work
        assert "title" in work
        assert "source_class" in work
        assert "word_count" in work
        assert work["title"] == "First Work"
        assert work["word_count"] == 10000


# ---------------------------------------------------------------------------
# TestAuthorBio
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestAuthorBio:
    """author_bio returns biographical summary for an author."""

    async def test_author_bio_returns_structure(
        self, author_storage: SM, integration_settings: Settings
    ) -> None:
        """author_bio returns valid bio dict for a known author."""
        result_str = await handle_author_bio(
            {"author_id": "test-author"},
            settings=integration_settings,
            storage=author_storage,
        )
        result = json.loads(result_str)

        assert result["author_id"] == "test-author"
        assert "works_in_library" in result
        assert "primary_works" in result
        assert "voice_profile" in result
        assert "major_themes" in result
        assert result["works_in_library"] == 2
        assert result["primary_works"] == 1

    async def test_author_bio_no_voice_profile(
        self, author_storage: SM, integration_settings: Settings
    ) -> None:
        """author_bio returns voice_profile=None when no profile is stored."""
        result_str = await handle_author_bio(
            {"author_id": "test-author"},
            settings=integration_settings,
            storage=author_storage,
        )
        result = json.loads(result_str)

        # No profile ingested → voice_profile is None
        assert result["voice_profile"] is None

    async def test_author_bio_primary_titles_included(
        self, author_storage: SM, integration_settings: Settings
    ) -> None:
        """author_bio includes list of primary work titles."""
        result_str = await handle_author_bio(
            {"author_id": "test-author"},
            settings=integration_settings,
            storage=author_storage,
        )
        result = json.loads(result_str)

        assert "primary_titles" in result
        assert "First Work" in result["primary_titles"]

    async def test_author_bio_missing_author_id_raises(
        self, author_storage: SM, integration_settings: Settings
    ) -> None:
        """author_bio raises RetrievalError if author_id is missing."""
        from author_library.errors import RetrievalError

        with pytest.raises(RetrievalError, match="author_id is required"):
            await handle_author_bio(
                {},
                settings=integration_settings,
                storage=author_storage,
            )


# ---------------------------------------------------------------------------
# TestLibraryStats
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestLibraryStats:
    """library_stats returns aggregate collection statistics."""

    async def test_library_stats_returns_structure(
        self, author_storage: SM
    ) -> None:
        """library_stats returns expected top-level section keys."""
        result_str = await handle_library_stats({}, storage=author_storage)
        result = json.loads(result_str)

        assert "works" in result
        assert "chunks" in result
        assert "embeddings" in result
        assert "graph" in result
        assert "thematic_index" in result
        assert "voice_profiles" in result

    async def test_library_stats_work_counts(self, author_storage: SM) -> None:
        """library_stats reflects the test data (3 works inserted)."""
        result_str = await handle_library_stats({}, storage=author_storage)
        result = json.loads(result_str)

        assert result["works"]["total_works"] == 3  # 2 test-author + 1 other-author
        assert result["chunks"]["total_chunks"] == 0  # No chunks inserted here

    async def test_library_stats_source_breakdown(
        self, author_storage: SM
    ) -> None:
        """library_stats works section shows primary/secondary breakdown."""
        result_str = await handle_library_stats({}, storage=author_storage)
        result = json.loads(result_str)

        works = result["works"]
        assert works.get("primary_works", 0) == 2
        assert works.get("secondary_works", 0) == 1

    async def test_library_stats_empty_library(self, clean_storage: SM) -> None:
        """library_stats works on an empty library (zero counts)."""
        result_str = await handle_library_stats({}, storage=clean_storage)
        result = json.loads(result_str)

        assert result["works"]["total_works"] == 0
        assert result["chunks"]["total_chunks"] == 0
