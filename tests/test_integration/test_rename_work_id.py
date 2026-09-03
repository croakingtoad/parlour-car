"""Integration tests for the work_id rename script on disposable stores."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.test_integration.conftest import SKIP_NO_DB

if TYPE_CHECKING:
    from author_library.storage.manager import StorageManager

_SCRIPT = Path(__file__).parents[2] / "scripts" / "rename_work_id.py"
_SPEC = importlib.util.spec_from_file_location("rename_work_id_integration", _SCRIPT)
assert _SPEC and _SPEC.loader
rename = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rename
_SPEC.loader.exec_module(rename)


@SKIP_NO_DB
@pytest.mark.asyncio
async def test_execution_refuses_immediate_fks_before_cross_store_writes(
    clean_storage: StorageManager,
    assert_graph_is_disposable: None,
) -> None:
    old_id = "test--rename-source"
    new_id = "test--rename-target"
    await _insert_rename_source(clean_storage, old_id)

    # The migration under test has already made this constraint deferrable.
    # Force it back to the real immediate form so refusal remains fail-closed.
    await clean_storage.pg.execute(
        "ALTER TABLE chunks ALTER CONSTRAINT chunks_work_id_fkey NOT DEFERRABLE"
    )
    assert (
        await clean_storage.pg.fetch_val(
            """SELECT NOT condeferrable FROM pg_constraint
        WHERE conname = 'chunks_work_id_fkey'"""
        )
        is True
    )
    try:
        with pytest.raises(rename.RenameError, match="immediate foreign key"):
            await rename.run(clean_storage, old_id, new_id, execute=True)

        assert (
            await clean_storage.pg.fetch_val(
                "SELECT count(*) FROM works WHERE work_id = $1", old_id
            )
            == 1
        )
        records = await clean_storage.neo4j.execute_read(
            "MATCH (w:Work {work_id: $work_id}) RETURN count(w) AS n",
            {"work_id": old_id},
        )
        assert records[0]["n"] == 1
    finally:
        await clean_storage.pg.execute(
            "ALTER TABLE chunks ALTER CONSTRAINT chunks_work_id_fkey DEFERRABLE INITIALLY IMMEDIATE"
        )


@SKIP_NO_DB
@pytest.mark.asyncio
async def test_execution_renames_all_postgresql_and_neo4j_records(
    clean_storage: StorageManager,
    assert_graph_is_disposable: None,
) -> None:
    old_id = "test--rename-source"
    new_id = "test--rename-target"
    await _insert_rename_source(clean_storage, old_id, with_children=True)

    source = await rename.execute_rename(clean_storage, old_id, new_id)

    assert source.pg_work == 1
    assert source.pg_children == {
        "chunks": 1,
        "session_sources": 0,
        "thematic_appearances": 1,
        "transcript_cache": 0,
    }
    assert (source.neo_work, source.neo_chunk) == (1, 1)
    expected_target_counts = {
        "works": 1,
        "chunks": 1,
        "session_sources": 0,
        "thematic_appearances": 1,
        "transcript_cache": 0,
    }
    for table, expected_count in expected_target_counts.items():
        assert (
            await clean_storage.pg.fetch_val(
                f"SELECT count(*) FROM {table} WHERE work_id = $1", old_id
            )
            == 0
        )
        assert (
            await clean_storage.pg.fetch_val(
                f"SELECT count(*) FROM {table} WHERE work_id = $1", new_id
            )
            == expected_count
        )
    records = await clean_storage.neo4j.execute_read(
        """MATCH (n) WHERE n:Work OR n:Chunk
        RETURN labels(n) AS labels, n.work_id AS work_id, count(n) AS count
        ORDER BY labels(n)""",
        {},
    )
    assert records == [
        {"labels": ["Chunk"], "work_id": new_id, "count": 1},
        {"labels": ["Work"], "work_id": new_id, "count": 1},
    ]
    from author_library.graph import check_pg_neo4j_consistency

    assert (await check_pg_neo4j_consistency(clean_storage))["is_consistent"] is True


@SKIP_NO_DB
@pytest.mark.asyncio
async def test_neo4j_count_change_after_preflight_compensates_both_stores(
    clean_storage: StorageManager,
    assert_graph_is_disposable: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_id = "test--rename-source"
    new_id = "test--rename-target"
    await _insert_rename_source(clean_storage, old_id, with_children=True)
    apply_postgres = rename._apply_postgres

    async def apply_postgres_then_change_neo4j_count(*args: object) -> None:
        await apply_postgres(*args)
        await clean_storage.neo4j.execute_write(
            "CREATE (:Chunk {chunk_id: 'test--rename-racing-chunk', work_id: $work_id})",
            {"work_id": old_id},
        )

    monkeypatch.setattr(rename, "_apply_postgres", apply_postgres_then_change_neo4j_count)

    with pytest.raises(
        rename.RenameError,
        match="Neo4j rename count changed during operation: Work=1, Chunk=2",
    ):
        await rename.execute_rename(clean_storage, old_id, new_id)

    for table in ("works", "chunks", "thematic_appearances"):
        assert (
            await clean_storage.pg.fetch_val(
                f"SELECT count(*) FROM {table} WHERE work_id = $1", old_id
            )
            == 1
        )
        assert (
            await clean_storage.pg.fetch_val(
                f"SELECT count(*) FROM {table} WHERE work_id = $1", new_id
            )
            == 0
        )
    records = await clean_storage.neo4j.execute_read(
        """MATCH (n) WHERE (n:Work OR n:Chunk) AND n.work_id IN [$old_id, $new_id]
        RETURN labels(n) AS labels, n.work_id AS work_id, count(n) AS count
        ORDER BY labels(n), work_id""",
        {"old_id": old_id, "new_id": new_id},
    )
    assert records == [
        {"labels": ["Chunk"], "work_id": old_id, "count": 2},
        {"labels": ["Work"], "work_id": old_id, "count": 1},
    ]


async def _insert_rename_source(
    storage: StorageManager, work_id: str, *, with_children: bool = False
) -> None:
    await storage.pg.execute(
        """
        INSERT INTO works (
            work_id, title, author, source_class, source_class_note,
            publication_year, publisher, format_ingested, word_count,
            genre_tags, subject_headings
        ) VALUES ($1, 'Source', 'Test Author', 'secondary',
                  'safe disposable test record', 2026, 'Test', 'txt', 1,
                  ARRAY['test'], ARRAY['test'])
        """,
        work_id,
    )
    await storage.neo4j.execute_write("CREATE (:Work {work_id: $work_id})", {"work_id": work_id})
    if not with_children:
        return

    await storage.pg.execute(
        """INSERT INTO chunks (work_id, text, granularity, source_class, position)
        VALUES ($1, 'Source text', 'macro', 'secondary', 1)""",
        work_id,
    )
    await storage.pg.execute(
        "INSERT INTO authors (id, canonical_name) VALUES "
        "('test--rename-author', 'Rename Test Author')"
    )
    await storage.pg.execute(
        """INSERT INTO thematic_entries (author_id, theme)
        VALUES ('test--rename-author', 'rename test theme')"""
    )
    await storage.pg.execute(
        """INSERT INTO thematic_appearances (entry_id, work_id)
        SELECT id, $1 FROM thematic_entries WHERE theme = 'rename test theme'""",
        work_id,
    )
    await storage.neo4j.execute_write(
        "CREATE (:Chunk {chunk_id: 'test--rename-chunk', work_id: $work_id})",
        {"work_id": work_id},
    )
