"""Acquisition candidate storage for The Author Library.

Tracks unresolved citations as acquisition candidates — works referenced
in the corpus but not yet in the library. Stored in PostgreSQL so they
persist across sessions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from author_library.storage.postgres import PostgresPool

log = structlog.get_logger(__name__)

# Migration SQL — called by the manager on first use.
_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS acquisition_candidates (
    id              SERIAL PRIMARY KEY,
    citation_text   TEXT NOT NULL,
    probable_work   TEXT,
    priority        TEXT NOT NULL DEFAULT 'medium',
    note            TEXT,
    flagged_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (citation_text)
);
"""


class AcquisitionManager:
    """Manages the acquisition candidate list in PostgreSQL."""

    def __init__(self, pg_pool: PostgresPool) -> None:
        self._pg = pg_pool
        self._table_ensured = False

    async def _ensure_table(self) -> None:
        """Create the acquisition_candidates table if it doesn't exist."""
        if self._table_ensured:
            return
        await self._pg.execute(_CREATE_TABLE_SQL)
        self._table_ensured = True

    async def flag(
        self,
        *,
        citation_text: str,
        probable_work: str | None = None,
        priority: str = "medium",
        note: str | None = None,
    ) -> bool:
        """Flag a citation as an acquisition candidate.

        Returns True if newly added, False if already flagged.
        """
        await self._ensure_table()

        # Check if already exists
        existing = await self._pg.fetch_one(
            "SELECT id FROM acquisition_candidates WHERE citation_text = $1",
            citation_text,
        )
        if existing is not None:
            return False

        await self._pg.execute(
            """
            INSERT INTO acquisition_candidates (citation_text, probable_work, priority, note)
            VALUES ($1, $2, $3, $4)
            """,
            citation_text,
            probable_work,
            priority,
            note,
        )
        log.info(
            "acquisition_flagged",
            citation_text=citation_text[:80],
            priority=priority,
        )
        return True

    async def count_total(self) -> int:
        """Return the total number of acquisition candidates."""
        await self._ensure_table()
        row = await self._pg.fetch_one(
            "SELECT COUNT(*) AS cnt FROM acquisition_candidates"
        )
        if row is None:
            return 0
        return int(row["cnt"])

    async def list_all(
        self,
        *,
        priority: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List acquisition candidates, optionally filtered by priority."""
        await self._ensure_table()

        if priority:
            rows = await self._pg.fetch_all(
                """
                SELECT id, citation_text, probable_work, priority, note, flagged_at
                FROM acquisition_candidates
                WHERE priority = $1
                ORDER BY flagged_at DESC
                LIMIT $2
                """,
                priority,
                limit,
            )
        else:
            rows = await self._pg.fetch_all(
                """
                SELECT id, citation_text, probable_work, priority, note, flagged_at
                FROM acquisition_candidates
                ORDER BY flagged_at DESC
                LIMIT $1
                """,
                limit,
            )

        return [dict(row) for row in rows]
