"""Tests for vocabulary management module."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock

import asyncpg
import pytest

from author_library.storage.postgres import PostgresPool
from author_library.vocabulary import _CREATE_TABLE_SQL, VocabularyManager


class FakeVocabularyPool(PostgresPool):
    """Minimal transactional pool for vocabulary-manager unit tests."""

    def __init__(self) -> None:
        self.connection = AsyncMock()
        self.connection.fetchrow.return_value = None
        self._fetch_one = AsyncMock(return_value={"cnt": 0})

    @asynccontextmanager
    async def transaction(self):
        yield self.connection

    async def fetch_one(self, query: str, *args: Any) -> Any:
        return await self._fetch_one(query, *args)


class TestVocabularyManagerCreation:
    """Verify VocabularyManager can be instantiated."""

    def test_create_with_pool(self) -> None:
        manager = VocabularyManager(None)  # type: ignore[arg-type]
        assert manager is not None
        assert manager._table_ensured is False


class TestVocabularyTableSQL:
    """Verify the table creation SQL is well-formed."""

    def test_create_table_sql_contains_required_columns(self) -> None:
        assert "term" in _CREATE_TABLE_SQL
        assert "status" in _CREATE_TABLE_SQL
        assert "merged_into" in _CREATE_TABLE_SQL
        assert "note" in _CREATE_TABLE_SQL
        assert "created_at" in _CREATE_TABLE_SQL
        assert "updated_at" in _CREATE_TABLE_SQL

    def test_create_table_sql_has_unique_constraint(self) -> None:
        assert "UNIQUE" in _CREATE_TABLE_SQL

    def test_create_table_sql_is_idempotent(self) -> None:
        assert "IF NOT EXISTS" in _CREATE_TABLE_SQL

    def test_default_status_is_proposed(self) -> None:
        assert "'proposed'" in _CREATE_TABLE_SQL

    def test_statuses_are_constrained(self) -> None:
        assert "'canonical'" in _CREATE_TABLE_SQL
        assert "'deprecated'" in _CREATE_TABLE_SQL
        assert "'merged'" in _CREATE_TABLE_SQL


class TestVocabularyManagerTransitions:
    async def test_merge_persists_a_new_alias_and_canonical_target(self) -> None:
        pool = FakeVocabularyPool()
        manager = VocabularyManager(pool)

        affected = await manager.merge("Poetic Form", "Prosody")

        assert affected == 0
        statements = [call.args[0] for call in pool.connection.execute.await_args_list]
        assert any("VALUES ($1, 'canonical', $2)" in statement for statement in statements)
        assert any("VALUES ($1, 'merged', $2, $3)" in statement for statement in statements)

    async def test_merge_rejects_self_merge(self) -> None:
        pool = FakeVocabularyPool()
        manager = VocabularyManager(pool)

        with pytest.raises(ValueError, match="cannot be merged into itself"):
            await manager.merge("Prosody", "Prosody")

    async def test_promote_clears_a_stale_merge_target(self) -> None:
        pool = FakeVocabularyPool()
        manager = VocabularyManager(pool)

        await manager.promote("Prosody")

        statements = [call.args[0] for call in pool.connection.execute.await_args_list]
        assert any("merged_into = NULL" in statement for statement in statements)


class TestVocabularyThemeCount:
    async def test_missing_thematic_entries_table_returns_zero(self) -> None:
        pool = FakeVocabularyPool()
        pool._fetch_one.side_effect = asyncpg.UndefinedTableError("thematic_entries")
        manager = VocabularyManager(pool)

        assert await manager._count_chunks_with_theme("Prosody") == 0

    async def test_unexpected_database_error_propagates(self) -> None:
        pool = FakeVocabularyPool()
        pool._fetch_one.side_effect = RuntimeError("database connection lost")
        manager = VocabularyManager(pool)

        with pytest.raises(RuntimeError, match="database connection lost"):
            await manager._count_chunks_with_theme("Prosody")
