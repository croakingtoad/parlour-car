"""Tests for the migration runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from author_library.storage.migrations.runner import (
    _discover_migrations,
    run_migrations,
)

if TYPE_CHECKING:
    from author_library.storage.postgres import PostgresPool


async def test_migrations_apply(pg_pool: PostgresPool) -> None:
    """Running migrations ensures all migration files are tracked."""
    await run_migrations(pg_pool)
    # Verify all discovered migrations are recorded in the _migrations table
    rows = await pg_pool.fetch_all("SELECT filename FROM _migrations ORDER BY id")
    filenames = [r["filename"] for r in rows]
    discovered = {migration.name for migration in _discover_migrations()}
    assert discovered.issubset(filenames), (
        f"Not recorded: {discovered - set(filenames)}"
    )
    assert "001_initial.sql" in filenames
    assert "002_indexes.sql" in filenames
    assert "003_fulltext.sql" in filenames
    assert "004_personal_source_class.sql" in filenames
    assert "005_nano_granularity.sql" in filenames
    assert "006_media_formats.sql" in filenames
    assert "007_pass_number.sql" in filenames
    assert "008_sessions.sql" in filenames
    assert "009_transcript_cache.sql" in filenames
    assert "010_search_vector_annotation.sql" in filenames
    assert "011_backfill_section_type.sql" in filenames
    assert "012_delete_noise_chunks.sql" in filenames
    assert "013_ingestion_lessons.sql" in filenames
    assert "014_reference_source_class.sql" in filenames
    assert "015_nullable_publication_metadata.sql" in filenames
    assert "016_deferrable_work_id_foreign_keys.sql" in filenames


async def test_publication_metadata_is_nullable(pg_pool: PostgresPool) -> None:
    """Unknown years and rejected PDF Producer values can be stored as NULL."""
    await run_migrations(pg_pool)
    rows = await pg_pool.fetch_all(
        """SELECT column_name, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'works'
          AND column_name IN ('publication_year', 'publisher')"""
    )
    nullability = {row["column_name"]: row["is_nullable"] for row in rows}
    assert nullability == {"publication_year": "YES", "publisher": "YES"}


async def test_migrations_idempotent(pg_pool: PostgresPool) -> None:
    """Running migrations twice does not re-apply."""
    # First-run results depend on test ordering.
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


def test_no_duplicate_migration_prefixes() -> None:
    """Each migration file must have a unique numeric prefix."""
    migrations = _discover_migrations()
    prefixes: dict[str, str] = {}
    for m in migrations:
        prefix = m.name.split("_", 1)[0]
        assert prefix not in prefixes, (
            f"Duplicate migration prefix {prefix!r}: {prefixes[prefix]} and {m.name}"
        )
        prefixes[prefix] = m.name


async def test_rename_from_004_is_not_reapplied(pg_pool: PostgresPool) -> None:
    """If a DB already applied old 004_transcript_cache.sql, rename handling
    updates the tracking row and the migration is NOT re-executed."""
    # Apply all migrations first
    await run_migrations(pg_pool)

    # Simulate a pre-rename database: remove the new name and insert the old.
    # This handles the case where the DB already has both rows from a prior
    # test run (the _migrations table is not cleaned between tests).
    await pg_pool.execute("DELETE FROM _migrations WHERE filename = '009_transcript_cache.sql'")
    await pg_pool.execute("DELETE FROM _migrations WHERE filename = '004_transcript_cache.sql'")
    await pg_pool.execute("INSERT INTO _migrations (filename) VALUES ('004_transcript_cache.sql')")

    # Running migrations again should apply the rename, not re-run the SQL
    newly_applied = await run_migrations(pg_pool)
    assert "009_transcript_cache.sql" not in newly_applied

    # Verify the tracker now has the new name
    rows = await pg_pool.fetch_all("SELECT filename FROM _migrations ORDER BY id")
    filenames = [r["filename"] for r in rows]
    assert "009_transcript_cache.sql" in filenames
    assert "004_transcript_cache.sql" not in filenames


async def test_renames_from_014_and_015_are_not_reapplied(
    pg_pool: PostgresPool,
) -> None:
    """Renumbered migrations are tracked without replaying their SQL."""
    await run_migrations(pg_pool)

    # Simulate the pre-main database, where the two branch migrations were
    # applied as 014 and 015 before main introduced its own 014.
    for filename in (
        "014_nullable_publication_metadata.sql",
        "015_nullable_publication_metadata.sql",
        "015_deferrable_work_id_foreign_keys.sql",
        "016_deferrable_work_id_foreign_keys.sql",
    ):
        await pg_pool.execute("DELETE FROM _migrations WHERE filename = $1", filename)
    await pg_pool.execute(
        "INSERT INTO _migrations (filename) VALUES ($1)",
        "014_nullable_publication_metadata.sql",
    )
    await pg_pool.execute(
        "INSERT INTO _migrations (filename) VALUES ($1)",
        "015_deferrable_work_id_foreign_keys.sql",
    )

    newly_applied = await run_migrations(pg_pool)
    assert "015_nullable_publication_metadata.sql" not in newly_applied
    assert "016_deferrable_work_id_foreign_keys.sql" not in newly_applied

    rows = await pg_pool.fetch_all("SELECT filename FROM _migrations ORDER BY id")
    filenames = [row["filename"] for row in rows]
    assert "015_nullable_publication_metadata.sql" in filenames
    assert "016_deferrable_work_id_foreign_keys.sql" in filenames
    assert "014_nullable_publication_metadata.sql" not in filenames
    assert "015_deferrable_work_id_foreign_keys.sql" not in filenames
