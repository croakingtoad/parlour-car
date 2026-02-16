"""Tests for the migration runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from author_library.storage.migrations.runner import run_migrations

if TYPE_CHECKING:
    from author_library.storage.postgres import PostgresPool


async def test_migrations_apply(pg_pool: PostgresPool) -> None:
    """Running migrations ensures all migration files are tracked."""
    applied = await run_migrations(pg_pool)
    # May be 0 (already applied) or 3 (first run). Either is valid.
    assert len(applied) in (0, 3)
    # Verify all 3 are recorded in the _migrations table
    rows = await pg_pool.fetch_all("SELECT filename FROM _migrations ORDER BY id")
    filenames = [r["filename"] for r in rows]
    assert "001_initial.sql" in filenames
    assert "002_indexes.sql" in filenames
    assert "003_fulltext.sql" in filenames


async def test_migrations_idempotent(pg_pool: PostgresPool) -> None:
    """Running migrations twice does not re-apply."""
    # First run may apply 0 or 3 depending on test ordering
    await run_migrations(pg_pool)
    # Second run should always apply 0 — the idempotency guarantee
    second = await run_migrations(pg_pool)
    assert len(second) == 0


async def test_tables_exist_after_migration(pg_pool: PostgresPool) -> None:
    """Core tables exist after migration."""
    await run_migrations(pg_pool)
    tables = await pg_pool.fetch_all(
        """SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name NOT LIKE '\\_%'
        ORDER BY table_name"""
    )
    table_names = {r["table_name"] for r in tables}
    expected = {
        "authors",
        "works",
        "chunks",
        "chunk_embeddings",
        "thematic_entries",
        "thematic_appearances",
        "voice_profiles",
    }
    assert expected.issubset(table_names), f"Missing tables: {expected - table_names}"


async def test_search_vector_column(pg_pool: PostgresPool) -> None:
    """Chunks table has the generated search_vector column after migration 003."""
    await run_migrations(pg_pool)
    row = await pg_pool.fetch_one(
        """SELECT column_name, data_type FROM information_schema.columns
        WHERE table_name = 'chunks' AND column_name = 'search_vector'"""
    )
    assert row is not None
    assert row["data_type"] == "tsvector"
