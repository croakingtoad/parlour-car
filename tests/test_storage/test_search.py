"""Tests for full-text search operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from author_library.storage.migrations.runner import run_migrations
from author_library.storage.repositories import PgChunkRepository, PgWorkRepository
from author_library.storage.search import search_fulltext, search_phrase

if TYPE_CHECKING:
    from author_library.storage.postgres import PostgresPool


# -- Helpers -----------------------------------------------------------------

SAMPLE_AUTHOR: dict[str, str] = {
    "id": "malcolm-guite",
    "canonical_name": "Malcolm Guite",
}

SAMPLE_WORK: dict[str, Any] = {
    "work_id": "malcolm-guite--faith-hope-and-poetry",
    "title": "Faith, Hope and Poetry",
    "author": "malcolm-guite",
    "source_class": "primary",
    "source_class_note": (
        "Authored by Malcolm Guite, the subject author of this collection"
    ),
    "publication_year": 2010,
    "publisher": "Ashgate Publishing",
    "format_ingested": "epub",
    "word_count": 85000,
    "genre_tags": ["literary criticism", "theology"],
    "subject_headings": ["poetry", "imagination", "theology"],
}

SAMPLE_CHUNKS: list[dict[str, Any]] = [
    {
        "work_id": "malcolm-guite--faith-hope-and-poetry",
        "text": (
            "The imagination is not merely a faculty for producing images; "
            "it is the living power and prime agent of all human perception, "
            "a repetition in the finite mind of the eternal act of creation."
        ),
        "granularity": "meso",
        "source_class": "primary",
        "position": 1,
    },
    {
        "work_id": "malcolm-guite--faith-hope-and-poetry",
        "text": (
            "Coleridge's understanding of the imagination owed much to the "
            "German idealist tradition, particularly to Schelling, but he gave "
            "it a distinctly theological turn, grounding imagination in the "
            "creative act of God."
        ),
        "granularity": "meso",
        "source_class": "primary",
        "position": 2,
    },
    {
        "work_id": "malcolm-guite--faith-hope-and-poetry",
        "text": (
            "The sonnet form itself became for Guite a means of prayer and "
            "meditation, each fourteen lines a compressed act of attention "
            "to the divine word speaking through the liturgical calendar."
        ),
        "granularity": "meso",
        "source_class": "primary",
        "position": 3,
    },
]

SECOND_WORK: dict[str, Any] = {
    **SAMPLE_WORK,
    "work_id": "test--critic--poetic-imagination",
    "title": "Poetic Imagination",
    "author": "test--critic",
    "source_class": "tertiary",
}

SECOND_CHUNK: dict[str, Any] = {
    "work_id": SECOND_WORK["work_id"],
    "text": "This reference traces imagination through several schools of poetry.",
    "granularity": "meso",
    "source_class": "tertiary",
    "position": 1,
}


async def _seed_data(pool: PostgresPool) -> None:
    """Insert author, work, and chunks for search tests."""
    await pool.execute(
        "INSERT INTO authors (id, canonical_name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        SAMPLE_AUTHOR["id"],
        SAMPLE_AUTHOR["canonical_name"],
    )
    work_repo = PgWorkRepository(pool)
    await work_repo.create(SAMPLE_WORK)
    await work_repo.create(SECOND_WORK)

    chunk_repo = PgChunkRepository(pool)
    for chunk in SAMPLE_CHUNKS:
        await chunk_repo.create(chunk)
    await chunk_repo.create(SECOND_CHUNK)


# -- Tests -------------------------------------------------------------------


async def test_fulltext_search(pg_pool: PostgresPool) -> None:
    """Full-text search returns ranked results for 'imagination'."""
    await run_migrations(pg_pool)
    await _seed_data(pg_pool)

    results = await search_fulltext(pg_pool, "imagination perception")
    assert len(results) >= 1
    # First result should be about imagination and perception
    snippet_lower = results[0].snippet.lower()
    assert "imagination" in snippet_lower or "perception" in snippet_lower
    assert results[0].rank > 0


async def test_fulltext_search_with_filters(pg_pool: PostgresPool) -> None:
    """Full-text search respects source_class and work filters."""
    await run_migrations(pg_pool)
    await _seed_data(pg_pool)

    # Should find results with matching source_class
    results = await search_fulltext(
        pg_pool, "imagination", source_class_filter="primary"
    )
    assert len(results) >= 1

    # Should find no results with non-matching source_class
    results = await search_fulltext(
        pg_pool, "imagination", source_class_filter="secondary"
    )
    assert len(results) == 0


async def test_fulltext_search_with_work_metadata_filters(
    pg_pool: PostgresPool,
) -> None:
    """Work metadata filters match any requested array value."""
    await run_migrations(pg_pool)
    await _seed_data(pg_pool)

    results = await search_fulltext(
        pg_pool,
        "imagination",
        subject_headings_filter=["missing", "poetry"],
        genre_tags_filter=["missing", "theology"],
    )
    assert len(results) == 3
    assert {result.source_class for result in results} == {"primary", "tertiary"}
    assert {result.work_id for result in results} == {
        SAMPLE_WORK["work_id"],
        SECOND_WORK["work_id"],
    }

    results = await search_fulltext(
        pg_pool,
        "imagination",
        subject_headings_filter=["quantum mechanics"],
    )
    assert results == []


async def test_phrase_search(pg_pool: PostgresPool) -> None:
    """Phrase search finds exact phrase matches."""
    await run_migrations(pg_pool)
    await _seed_data(pg_pool)

    results = await search_phrase(pg_pool, "living power and prime agent")
    assert len(results) >= 1


async def test_search_no_results(pg_pool: PostgresPool) -> None:
    """Search returns empty list when no matches."""
    await run_migrations(pg_pool)
    await _seed_data(pg_pool)

    results = await search_fulltext(pg_pool, "quantum mechanics")
    assert len(results) == 0
