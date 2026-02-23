"""Tests for acquisition candidate storage module."""

from __future__ import annotations

from author_library.catalog.acquisition import _CREATE_TABLE_SQL, AcquisitionManager


class TestAcquisitionManagerCreation:
    """Verify AcquisitionManager can be instantiated."""

    def test_create_with_pool(self) -> None:
        # Should not raise even with a None pool (lazy initialization)
        manager = AcquisitionManager(None)  # type: ignore[arg-type]
        assert manager is not None
        assert manager._table_ensured is False


class TestAcquisitionTableSQL:
    """Verify the table creation SQL is well-formed."""

    def test_create_table_sql_contains_required_columns(self) -> None:
        assert "citation_text" in _CREATE_TABLE_SQL
        assert "probable_work" in _CREATE_TABLE_SQL
        assert "priority" in _CREATE_TABLE_SQL
        assert "note" in _CREATE_TABLE_SQL
        assert "flagged_at" in _CREATE_TABLE_SQL

    def test_create_table_sql_has_unique_constraint(self) -> None:
        assert "UNIQUE (citation_text)" in _CREATE_TABLE_SQL

    def test_create_table_sql_is_idempotent(self) -> None:
        assert "IF NOT EXISTS" in _CREATE_TABLE_SQL
