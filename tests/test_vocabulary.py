"""Tests for vocabulary management module."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from author_library.vocabulary import _CREATE_TABLE_SQL, VocabularyManager


class FakeVocabularyPool:
    """Minimal transactional pool for vocabulary-manager unit tests."""

    def __init__(self) -> None:
        self.connection = AsyncMock()
        self.connection.fetchrow.return_value = None
        self.fetch_one = AsyncMock(return_value={"cnt": 0})

    @asynccontextmanager
    async def transaction(self):
        yield self.connection


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
        manager = VocabularyManager(pool)  # type: ignore[arg-type]

        affected = await manager.merge("Poetic Form", "Prosody")

        assert affected == 0
        statements = [call.args[0] for call in pool.connection.execute.await_args_list]
        assert any("VALUES ($1, 'canonical', $2)" in statement for statement in statements)
        assert any("VALUES ($1, 'merged', $2, $3)" in statement for statement in statements)

    async def test_merge_rejects_self_merge(self) -> None:
        pool = FakeVocabularyPool()
        manager = VocabularyManager(pool)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="cannot be merged into itself"):
            await manager.merge("Prosody", "Prosody")

    async def test_promote_clears_a_stale_merge_target(self) -> None:
        pool = FakeVocabularyPool()
        manager = VocabularyManager(pool)  # type: ignore[arg-type]

        await manager.promote("Prosody")

        statements = [call.args[0] for call in pool.connection.execute.await_args_list]
        assert any("merged_into = NULL" in statement for statement in statements)
