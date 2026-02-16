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
