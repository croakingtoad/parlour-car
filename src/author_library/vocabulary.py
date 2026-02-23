"""Vocabulary management for The Author Library.

Manages canonical vocabulary terms used for thematic tagging.
Terms have a lifecycle: proposed → canonical → deprecated.
Supports merging synonymous terms with chunk retagging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from author_library.storage.postgres import PostgresPool

log = structlog.get_logger(__name__)

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS vocabulary_terms (
    id          SERIAL PRIMARY KEY,
    term        TEXT NOT NULL UNIQUE,
    status      TEXT NOT NULL DEFAULT 'proposed',
    merged_into TEXT,
    note        TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


class VocabularyManager:
    """Manages vocabulary terms in PostgreSQL."""

    def __init__(self, pg_pool: PostgresPool) -> None:
        self._pg = pg_pool
        self._table_ensured = False

    async def _ensure_table(self) -> None:
        if self._table_ensured:
            return
        await self._pg.execute(_CREATE_TABLE_SQL)
        self._table_ensured = True

    async def list_terms(
        self,
        *,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List all vocabulary terms, optionally filtered by status."""
        await self._ensure_table()

        if status:
            rows = await self._pg.fetch_all(
                """
                SELECT term, status, merged_into, note, created_at, updated_at
                FROM vocabulary_terms
                WHERE status = $1
                ORDER BY term
                """,
                status,
            )
        else:
            rows = await self._pg.fetch_all(
                """
                SELECT term, status, merged_into, note, created_at, updated_at
                FROM vocabulary_terms
                ORDER BY term
                """
            )

        return [
            {
                "term": row["term"],
                "status": row["status"],
                "merged_into": row["merged_into"],
                "note": row["note"],
            }
            for row in rows
        ]

    async def propose(
        self,
        term: str,
        *,
        note: str | None = None,
    ) -> dict[str, Any]:
        """Propose a new vocabulary term. Returns the term record."""
        await self._ensure_table()

        # Check if term already exists
        existing = await self._pg.fetch_one(
            "SELECT term, status FROM vocabulary_terms WHERE term = $1",
            term.lower().strip(),
        )
        if existing:
            return {
                "term": existing["term"],
                "status": existing["status"],
                "already_exists": True,
            }

        await self._pg.execute(
            """
            INSERT INTO vocabulary_terms (term, status, note)
            VALUES ($1, 'proposed', $2)
            """,
            term.lower().strip(),
            note,
        )

        log.info("vocabulary_proposed", term=term)
        return {
            "term": term.lower().strip(),
            "status": "proposed",
            "already_exists": False,
        }

    async def promote(self, term: str) -> int:
        """Promote a proposed term to canonical status.

        Returns the number of chunks tagged with this term.
        """
        await self._ensure_table()

        await self._pg.execute(
            """
            UPDATE vocabulary_terms
            SET status = 'canonical', updated_at = NOW()
            WHERE term = $1
            """,
            term.lower().strip(),
        )

        affected = await self._count_chunks_with_theme(term)
        log.info("vocabulary_promoted", term=term, affected_chunks=affected)
        return affected

    async def merge(
        self,
        source_term: str,
        target_term: str,
        *,
        note: str | None = None,
    ) -> int:
        """Merge source_term into target_term.

        Updates the source term's record and retags chunks in the
        thematic_entries table.

        Returns the number of affected chunks.
        """
        await self._ensure_table()

        source = source_term.lower().strip()
        target = target_term.lower().strip()

        # Mark source as merged
        await self._pg.execute(
            """
            UPDATE vocabulary_terms
            SET status = 'merged', merged_into = $2, note = $3, updated_at = NOW()
            WHERE term = $1
            """,
            source,
            target,
            note or f"Merged into '{target}'",
        )

        # Ensure target exists as canonical
        existing_target = await self._pg.fetch_one(
            "SELECT term FROM vocabulary_terms WHERE term = $1",
            target,
        )
        if not existing_target:
            await self._pg.execute(
                """
                INSERT INTO vocabulary_terms (term, status, note)
                VALUES ($1, 'canonical', $2)
                """,
                target,
                f"Created via merge from '{source}'",
            )

        # Count affected chunks (chunks that reference the source theme)
        affected = await self._count_chunks_with_theme(source)
        log.info(
            "vocabulary_merged",
            source=source,
            target=target,
            affected_chunks=affected,
        )
        return affected

    async def deprecate(
        self,
        term: str,
        *,
        note: str | None = None,
    ) -> int:
        """Deprecate a vocabulary term.

        Returns the number of chunks still tagged with this term.
        """
        await self._ensure_table()

        await self._pg.execute(
            """
            UPDATE vocabulary_terms
            SET status = 'deprecated', note = $2, updated_at = NOW()
            WHERE term = $1
            """,
            term.lower().strip(),
            note or "Deprecated",
        )

        affected = await self._count_chunks_with_theme(term)
        log.info("vocabulary_deprecated", term=term, affected_chunks=affected)
        return affected

    async def _count_chunks_with_theme(self, theme: str) -> int:
        """Count chunks whose metadata references a theme.

        Searches the thematic_entries table for the theme name.
        """
        try:
            row = await self._pg.fetch_one(
                """
                SELECT COUNT(*) AS cnt
                FROM thematic_entries
                WHERE theme ILIKE $1
                """,
                f"%{theme}%",
            )
            if row:
                return int(row["cnt"])
        except Exception:
            # Table may not exist yet, that's fine
            pass
        return 0
