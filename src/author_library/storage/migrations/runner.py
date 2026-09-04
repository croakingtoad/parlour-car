"""Simple forward-only SQL migration runner.

Migrations are numbered SQL files (001_initial.sql, 002_indexes.sql, etc.)
stored alongside this module. Applied migrations are tracked in a
`_migrations` table to ensure idempotency.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from author_library.errors import StorageError

if TYPE_CHECKING:
    from author_library.storage.postgres import PostgresPool

log = structlog.get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).parent

# Historical renames: old filename -> new filename.
# When upgrading a database that already applied the old name, the tracker
# is updated in-place so the renamed file is not re-applied.
_RENAMES: dict[str, str] = {
    "004_transcript_cache.sql": "009_transcript_cache.sql",
    "014_nullable_publication_metadata.sql": "015_nullable_publication_metadata.sql",
    "015_deferrable_work_id_foreign_keys.sql": "016_deferrable_work_id_foreign_keys.sql",
}


async def _apply_renames(pool: PostgresPool) -> None:
    """Update _migrations rows for any historically renamed migration files.

    Three possible states per rename entry:
    1. Only old name present  -> UPDATE old row to new name.
    2. Both names present     -> DELETE old row (new row is canonical).
    3. Only new name / neither -> no-op.
    """
    # Process in reverse so a migration that takes another migration's old
    # number does not collide with its still-unrenamed tracking row.
    for old_name, new_name in reversed(tuple(_RENAMES.items())):
        # If both rows exist, just drop the stale old-name row.
        await pool.execute(
            "DELETE FROM _migrations WHERE filename = $1 "
            "AND EXISTS (SELECT 1 FROM _migrations WHERE filename = $2)",
            old_name,
            new_name,
        )
        # If only the old name remains, rename it in place.
        await pool.execute(
            "UPDATE _migrations SET filename = $1 WHERE filename = $2",
            new_name,
            old_name,
        )


async def _ensure_migrations_table(pool: PostgresPool) -> None:
    """Create the migrations tracking table if it does not exist."""
    await pool.execute("""
        CREATE TABLE IF NOT EXISTS _migrations (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL UNIQUE,
            applied_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)


async def _applied_migrations(pool: PostgresPool) -> set[str]:
    """Return the set of already-applied migration filenames."""
    rows = await pool.fetch_all("SELECT filename FROM _migrations ORDER BY id")
    return {row["filename"] for row in rows}


def _discover_migrations() -> list[Path]:
    """Find all .sql files in the migrations directory, sorted by name."""
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


async def run_migrations(pool: PostgresPool) -> list[str]:
    """Apply any pending SQL migrations.

    Returns the list of newly-applied migration filenames.
    """
    await _ensure_migrations_table(pool)
    await _apply_renames(pool)
    applied = await _applied_migrations(pool)
    pending = [m for m in _discover_migrations() if m.name not in applied]

    newly_applied: list[str] = []
    for migration_file in pending:
        sql = migration_file.read_text(encoding="utf-8")
        try:
            async with pool.transaction() as conn:
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO _migrations (filename) VALUES ($1)",
                    migration_file.name,
                )
            newly_applied.append(migration_file.name)
            log.info("migration_applied", filename=migration_file.name)
        except Exception as exc:
            raise StorageError(
                f"Migration failed: {migration_file.name}",
                context={"filename": migration_file.name},
                cause=exc,
            ) from exc

    if newly_applied:
        log.info(
            "migrations_complete",
            applied=len(newly_applied),
            total=len(applied) + len(newly_applied),
        )
    else:
        log.info("migrations_up_to_date", total=len(applied))

    return newly_applied
