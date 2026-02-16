"""Shared fixtures for intelligence extraction tests.

Tests that require LLM calls check for ANTHROPIC_API_KEY and skip
gracefully if not available. Database-backed tests connect to real
PostgreSQL and Neo4j containers.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING, Any

import pytest

from author_library.config import DatabaseSettings, Settings
from author_library.storage.manager import StorageManager
from author_library.storage.migrations.runner import run_migrations
from author_library.storage.neo4j import Neo4jConnection
from author_library.storage.postgres import PostgresPool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Skip marker for LLM-dependent tests
# ---------------------------------------------------------------------------

requires_anthropic_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping LLM-dependent test",
)


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def db_settings() -> DatabaseSettings:
    """Real database settings pointing at Docker containers."""
    return DatabaseSettings(
        postgres_url="postgresql://author_library:author_library@localhost:5432/author_library",
        neo4j_url="bolt://localhost:7687",
        neo4j_user="neo4j",
    )


@pytest.fixture(scope="session")
def app_settings() -> Settings:
    """Full application settings."""
    return Settings()


@pytest.fixture
async def pg_pool(db_settings: DatabaseSettings) -> AsyncIterator[PostgresPool]:
    """Provide a connected PostgreSQL pool."""
    pool = PostgresPool(db_settings, min_size=1, max_size=3)
    await pool.connect()
    await run_migrations(pool)
    yield pool
    await pool.close()


@pytest.fixture
async def neo4j_conn(db_settings: DatabaseSettings) -> AsyncIterator[Neo4jConnection]:
    """Provide a connected Neo4j driver."""
    conn = Neo4jConnection(db_settings)
    await conn.connect()
    await conn.init_schema()
    yield conn
    await conn.close()


@pytest.fixture
async def storage(db_settings: DatabaseSettings) -> AsyncIterator[StorageManager]:
    """Provide a fully connected StorageManager."""
    mgr = StorageManager(db_settings, pg_min_pool=1, pg_max_pool=3)
    await mgr.connect(run_pg_migrations=True, init_neo4j_schema=True)
    yield mgr
    await mgr.close()


@pytest.fixture(autouse=True)
async def _cleanup_pg(pg_pool: PostgresPool) -> AsyncIterator[None]:
    """Clean up test data after each test."""
    yield
    for table in [
        "thematic_appearances",
        "thematic_entries",
        "voice_profiles",
        "chunk_embeddings",
        "chunks",
        "works",
        "authors",
    ]:
        with contextlib.suppress(Exception):
            await pg_pool.execute(f"DELETE FROM {table}")


@pytest.fixture(autouse=True)
async def _cleanup_neo4j(neo4j_conn: Neo4jConnection) -> AsyncIterator[None]:
    """Clean up Neo4j test data after each test."""
    yield
    with contextlib.suppress(Exception):
        await neo4j_conn.execute_write("MATCH (n) DETACH DELETE n", {})


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

SAMPLE_AUTHOR: dict[str, str] = {
    "id": "malcolm-guite",
    "canonical_name": "Malcolm Guite",
}


SAMPLE_PRIMARY_WORK: dict[str, Any] = {
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
    "source_metadata": {},
}


SAMPLE_PRIMARY_WORK_2: dict[str, Any] = {
    "work_id": "malcolm-guite--mariner",
    "title": "Mariner: A Voyage with Samuel Taylor Coleridge",
    "author": "malcolm-guite",
    "source_class": "primary",
    "source_class_note": (
        "Authored by Malcolm Guite, the subject author of this collection"
    ),
    "publication_year": 2017,
    "publisher": "Hodder & Stoughton",
    "format_ingested": "epub",
    "word_count": 90000,
    "genre_tags": ["biography", "literary criticism"],
    "subject_headings": ["coleridge", "poetry", "romanticism"],
    "source_metadata": {},
}


# Representative literary text chunks for testing
PRIMARY_CHUNKS: list[dict[str, Any]] = [
    {
        "work_id": "malcolm-guite--faith-hope-and-poetry",
        "text": (
            "Coleridge makes a vital distinction between what he calls the Primary "
            "Imagination, which is 'the living power and prime agent of all human "
            "perception, and as a repetition in the finite mind of the eternal act "
            "of creation in the infinite I AM', and the Secondary Imagination, which "
            "he describes as 'an echo of the former, co-existing with the conscious "
            "will'. This distinction is crucial for understanding how poetry mediates "
            "between the divine creative act and human creative response."
        ),
        "granularity": "meso",
        "source_class": "primary",
        "chapter": "Chapter 2: The Romantics",
        "position": 1,
    },
    {
        "work_id": "malcolm-guite--faith-hope-and-poetry",
        "text": (
            "The sacramental vision at the heart of Coleridge's poetics sees every "
            "created thing as potentially revelatory, a window onto the divine. This "
            "is not pantheism, but rather a deeply incarnational understanding in "
            "which the material world is charged with spiritual significance. The "
            "poet's task is to help us see this significance, to restore our vision "
            "of the numinous that lies within and behind the ordinary."
        ),
        "granularity": "meso",
        "source_class": "primary",
        "chapter": "Chapter 3: Sacramental Vision",
        "position": 2,
    },
    {
        "work_id": "malcolm-guite--faith-hope-and-poetry",
        "text": (
            "I want to argue that the best poetry does not simply describe religious "
            "experience from the outside, but actually becomes a means of mediating "
            "it. The poem is not merely about encounter with the divine; it becomes "
            "itself a place of encounter. In this sense, the poem functions "
            "sacramentally — it is an outward and visible sign of an inward and "
            "spiritual grace."
        ),
        "granularity": "meso",
        "source_class": "primary",
        "chapter": "Chapter 1: Introduction",
        "position": 3,
    },
    {
        "work_id": "malcolm-guite--faith-hope-and-poetry",
        "text": (
            "Herbert's poetry, like all great devotional verse, achieves its power "
            "not through abstraction but through a startling particularity. When he "
            "writes of 'the board' spread for communion, or the flowers that 'bid us "
            "draw nigh', he grounds the transcendent in the tactile. This is poetry "
            "that you can taste and touch, and it is precisely this embodied quality "
            "that makes it such an effective vehicle for spiritual truth."
        ),
        "granularity": "meso",
        "source_class": "primary",
        "chapter": "Chapter 5: George Herbert",
        "position": 4,
    },
    {
        "work_id": "malcolm-guite--mariner",
        "text": (
            "What draws me to Coleridge is not simply his genius as a poet but his "
            "extraordinary capacity for self-reflection. He was always examining his "
            "own processes of thought and imagination, always asking how the mind "
            "works and what happens when we truly perceive something. This restless "
            "self-examination, which could sometimes tip into paralysing self-doubt, "
            "is also what makes his philosophical writings so penetrating."
        ),
        "granularity": "meso",
        "source_class": "primary",
        "chapter": "Chapter 1: The Beginning",
        "position": 5,
    },
    {
        "work_id": "malcolm-guite--mariner",
        "text": (
            "In an earlier book I argued that Coleridge's distinction between Primary "
            "and Secondary Imagination remains one of the most useful frameworks we "
            "have for understanding the poetic act. I still believe this, but having "
            "lived with Coleridge's thought more intimately over these past years, "
            "I now see that the distinction is more fluid and less schematic than I "
            "once presented it. The boundary between perception and creation is "
            "itself a creative act."
        ),
        "granularity": "meso",
        "source_class": "primary",
        "chapter": "Chapter 4: The Imagination",
        "position": 6,
    },
]


async def insert_sample_data(pool: PostgresPool) -> None:
    """Insert sample author, works, and chunks for testing."""
    # Author
    await pool.execute(
        "INSERT INTO authors (id, canonical_name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        SAMPLE_AUTHOR["id"],
        SAMPLE_AUTHOR["canonical_name"],
    )

    # Works
    from author_library.storage.repositories import PgWorkRepository

    work_repo = PgWorkRepository(pool)
    for work in [SAMPLE_PRIMARY_WORK, SAMPLE_PRIMARY_WORK_2]:
        with contextlib.suppress(Exception):
            await work_repo.create(work)

    # Chunks
    from author_library.storage.repositories import PgChunkRepository

    chunk_repo = PgChunkRepository(pool)
    for chunk in PRIMARY_CHUNKS:
        with contextlib.suppress(Exception):
            await chunk_repo.create(chunk)
