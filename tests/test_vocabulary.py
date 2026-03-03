"""Tests for vocabulary management module."""

from __future__ import annotations

from author_library.vocabulary import _CREATE_TABLE_SQL, VocabularyManager


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
