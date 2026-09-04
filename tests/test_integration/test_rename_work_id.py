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
    reset_disposable_graph,
) -> None:
    await reset_disposable_graph(clean_storage.neo4j)
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
async def test_unrelated_preexisting_drift_does_not_block_scoped_rename(
    clean_storage: StorageManager,
    assert_graph_is_disposable: None,
) -> None:
    old_id = "test--rename-source"
    new_id = "test--rename-target"
    unrelated_id = "test--rename-unrelated-drift"
    await _insert_rename_source(clean_storage, old_id, with_children=True)
    await clean_storage.neo4j.execute_write(
        "CREATE (:Work {work_id: $work_id})-[:HAS_CHUNK]->"
        "(:Chunk {chunk_id: 'test--unrelated-drift-chunk', work_id: $work_id})",
        {"work_id": unrelated_id},
    )

    await rename.execute_rename(clean_storage, old_id, new_id)

    assert await clean_storage.pg.fetch_val(
        "SELECT count(*) FROM works WHERE work_id = $1", new_id
    ) == 1
    records = await clean_storage.neo4j.execute_read(
        "MATCH (c:Chunk {work_id: $work_id}) RETURN c.chunk_id AS chunk_id",
        {"work_id": new_id},
    )
    pg_chunk_ids = await clean_storage.pg.fetch_all(
        "SELECT id FROM chunks WHERE work_id = $1", new_id
    )
    assert {record["chunk_id"] for record in records} == {str(row["id"]) for row in pg_chunk_ids}


@SKIP_NO_DB
@pytest.mark.asyncio
async def test_mid_rename_chunk_identity_drift_blocks_and_compensates(
    clean_storage: StorageManager,
    assert_graph_is_disposable: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_id = "test--rename-source"
    new_id = "test--rename-target"
    drifted_chunk_id = "test--rename-chunk-identity-drift"
    await _insert_rename_source(clean_storage, old_id, with_children=True)
    apply_neo4j = rename._apply_neo4j

    async def apply_neo4j_then_change_chunk_identity(*args: object) -> object:
        records = await apply_neo4j(*args)
        await clean_storage.neo4j.execute_write(
            "MATCH (c:Chunk {work_id: $work_id}) SET c.chunk_id = $chunk_id",
            {"work_id": new_id, "chunk_id": drifted_chunk_id},
        )
        return records

    monkeypatch.setattr(rename, "_apply_neo4j", apply_neo4j_then_change_chunk_identity)

    with pytest.raises(rename.RenameError, match="scoped PG/Neo4j chunk_id mismatch"):
        await rename.execute_rename(clean_storage, old_id, new_id)

    assert await clean_storage.pg.fetch_val(
        "SELECT count(*) FROM works WHERE work_id = $1", old_id
    ) == 1
    assert await clean_storage.pg.fetch_val(
        "SELECT count(*) FROM works WHERE work_id = $1", new_id
    ) == 0
    records = await clean_storage.neo4j.execute_read(
        "MATCH (n) WHERE (n:Work OR n:Chunk) AND n.work_id IN [$old_id, $new_id] "
        "RETURN labels(n) AS labels, n.chunk_id AS chunk_id, n.work_id AS work_id "
        "ORDER BY labels(n), chunk_id",
        {"old_id": old_id, "new_id": new_id},
    )
    assert records == [
        {"labels": ["Chunk"], "chunk_id": drifted_chunk_id, "work_id": old_id},
        {"labels": ["Work"], "chunk_id": None, "work_id": old_id},
    ]


@SKIP_NO_DB
@pytest.mark.asyncio
async def test_post_commit_chunk_identity_drift_fails_without_compensation(
    clean_storage: StorageManager,
    assert_graph_is_disposable: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_id = "test--rename-source"
    new_id = "test--rename-target"
    drifted_chunk_id = "test--rename-post-commit-identity-drift"
    await _insert_rename_source(clean_storage, old_id, with_children=True)
    scoped_identity = rename._assert_scoped_chunk_identity
    assertion_calls = 0

    async def mutate_after_precommit_check(*args: object, **kwargs: object) -> None:
        nonlocal assertion_calls
        assertion_calls += 1
        if assertion_calls == 3:
            assert kwargs.get("pg") is None
            await clean_storage.neo4j.execute_write(
                "MATCH (c:Chunk {work_id: $work_id}) SET c.chunk_id = $chunk_id",
                {"work_id": new_id, "chunk_id": drifted_chunk_id},
            )
        await scoped_identity(*args, **kwargs)

    monkeypatch.setattr(rename, "_assert_scoped_chunk_identity", mutate_after_precommit_check)

    with pytest.raises(
        rename.RenameError,
        match="PostgreSQL committed while Neo4j did not match",
    ):
        await rename.execute_rename(clean_storage, old_id, new_id)

    assert assertion_calls == 3
    assert await clean_storage.pg.fetch_val(
        "SELECT count(*) FROM works WHERE work_id = $1", old_id
    ) == 0
    assert await clean_storage.pg.fetch_val(
        "SELECT count(*) FROM works WHERE work_id = $1", new_id
    ) == 1
    records = await clean_storage.neo4j.execute_read(
        "MATCH (n) WHERE (n:Work OR n:Chunk) AND n.work_id IN [$old_id, $new_id] "
        "RETURN labels(n) AS labels, n.chunk_id AS chunk_id, n.work_id AS work_id "
        "ORDER BY labels(n), chunk_id",
        {"old_id": old_id, "new_id": new_id},
    )
    assert records == [
        {"labels": ["Chunk"], "chunk_id": drifted_chunk_id, "work_id": new_id},
        {"labels": ["Work"], "chunk_id": None, "work_id": new_id},
    ]


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


@SKIP_NO_DB
@pytest.mark.asyncio
async def test_target_side_neo4j_drift_preserves_independent_target_nodes(
    clean_storage: StorageManager,
    assert_graph_is_disposable: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_id = "test--rename-source"
    new_id = "test--rename-target"
    await _insert_rename_source(clean_storage, old_id, with_children=True)
    apply_postgres = rename._apply_postgres

    async def apply_postgres_then_insert_target_drift(*args: object) -> None:
        await apply_postgres(*args)
        # Work.work_id is unique in the disposable graph. Chunk.work_id is
        # intentionally non-unique, which is the reverse-scope hole here.
        await clean_storage.neo4j.execute_write(
            "CREATE (:Chunk {chunk_id: 'test--rename-target-racing-chunk', work_id: $work_id})",
            {"work_id": new_id},
        )

    monkeypatch.setattr(rename, "_apply_postgres", apply_postgres_then_insert_target_drift)

    with pytest.raises(rename.RenameError, match="new work_id post-condition failed"):
        await rename.execute_rename(clean_storage, old_id, new_id)

    for table in ("works", "chunks", "thematic_appearances"):
        assert await clean_storage.pg.fetch_val(
            f"SELECT count(*) FROM {table} WHERE work_id = $1", old_id
        ) == 1
        assert await clean_storage.pg.fetch_val(
            f"SELECT count(*) FROM {table} WHERE work_id = $1", new_id
        ) == 0
    records = await clean_storage.neo4j.execute_read(
        """MATCH (n) WHERE (n:Work OR n:Chunk) AND n.work_id IN [$old_id, $new_id]
        RETURN labels(n) AS labels, n.work_id AS work_id, count(n) AS count
        ORDER BY labels(n), work_id""",
        {"old_id": old_id, "new_id": new_id},
    )
    assert records == [
        {"labels": ["Chunk"], "work_id": old_id, "count": 1},
        {"labels": ["Chunk"], "work_id": new_id, "count": 1},
        {"labels": ["Work"], "work_id": old_id, "count": 1},
    ]


@SKIP_NO_DB
@pytest.mark.asyncio
async def test_reverse_scope_mismatch_leaves_captured_neo4j_nodes_untouched(
    clean_storage: StorageManager,
    assert_graph_is_disposable: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_id = "test--rename-source"
    new_id = "test--rename-target"
    drifted_id = "test--rename-third-id"
    await _insert_rename_source(clean_storage, old_id, with_children=True)
    first_graph_chunk_id = await clean_storage.pg.fetch_val(
        "SELECT id FROM chunks WHERE work_id = $1 AND position = 1", old_id
    )
    graph_chunk_id = await clean_storage.pg.fetch_val(
        """INSERT INTO chunks (work_id, text, granularity, source_class, position)
        VALUES ($1, 'Second source text', 'macro', 'secondary', 2)
        RETURNING id""",
        old_id,
    )
    await clean_storage.neo4j.execute_write(
        "CREATE (:Chunk {chunk_id: $chunk_id, work_id: $work_id})",
        {"chunk_id": str(graph_chunk_id), "work_id": old_id},
    )
    apply_neo4j = rename._apply_neo4j

    async def apply_neo4j_then_move_captured_chunk(*args: object) -> object:
        records = await apply_neo4j(*args)
        await clean_storage.neo4j.execute_write(
            """MATCH (c:Chunk {chunk_id: $chunk_id})
            SET c.work_id = $work_id""",
            {"chunk_id": str(graph_chunk_id), "work_id": drifted_id},
        )
        return records

    monkeypatch.setattr(rename, "_apply_neo4j", apply_neo4j_then_move_captured_chunk)

    with pytest.raises(
        rename.RenameError,
        match="reverse: captured identity scope did not match before mutation",
    ):
        await rename.execute_rename(clean_storage, old_id, new_id)

    assert await clean_storage.pg.fetch_val(
        "SELECT count(*) FROM works WHERE work_id = $1", old_id
    ) == 1
    assert await clean_storage.pg.fetch_val(
        "SELECT count(*) FROM chunks WHERE work_id = $1", old_id
    ) == 2
    assert await clean_storage.pg.fetch_val(
        "SELECT count(*) FROM works WHERE work_id = $1", new_id
    ) == 0
    records = await clean_storage.neo4j.execute_read(
        """MATCH (n) WHERE (n:Work OR n:Chunk) AND n.work_id IN $work_ids
        RETURN labels(n) AS labels, n.chunk_id AS chunk_id, n.work_id AS work_id
        ORDER BY labels(n), chunk_id""",
        {"work_ids": [old_id, new_id, drifted_id]},
    )
    expected_records = [
        {"labels": ["Chunk"], "chunk_id": str(first_graph_chunk_id), "work_id": new_id},
        {
            "labels": ["Chunk"],
            "chunk_id": str(graph_chunk_id),
            "work_id": drifted_id,
        },
        {"labels": ["Work"], "chunk_id": None, "work_id": new_id},
    ]
    assert sorted(
        records, key=lambda record: (record["labels"], record["chunk_id"] or "")
    ) == sorted(
        expected_records,
        key=lambda record: (record["labels"], record["chunk_id"] or ""),
    )


@SKIP_NO_DB
@pytest.mark.asyncio
async def test_zero_chunk_source_compensates_its_captured_work_node(
    clean_storage: StorageManager,
    assert_graph_is_disposable: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_id = "test--rename-source"
    new_id = "test--rename-target"
    await _insert_rename_source(clean_storage, old_id)
    apply_neo4j = rename._apply_neo4j

    async def apply_neo4j_then_insert_target_drift(*args: object) -> object:
        records = await apply_neo4j(*args)
        await clean_storage.neo4j.execute_write(
            "CREATE (:Chunk {chunk_id: 'test--rename-zero-chunk-drift', work_id: $work_id})",
            {"work_id": new_id},
        )
        return records

    monkeypatch.setattr(rename, "_apply_neo4j", apply_neo4j_then_insert_target_drift)

    with pytest.raises(rename.RenameError, match="new work_id post-condition failed"):
        await rename.execute_rename(clean_storage, old_id, new_id)

    assert await clean_storage.pg.fetch_val(
        "SELECT count(*) FROM works WHERE work_id = $1", old_id
    ) == 1
    assert await clean_storage.pg.fetch_val(
        "SELECT count(*) FROM chunks WHERE work_id = $1", old_id
    ) == 0
    records = await clean_storage.neo4j.execute_read(
        """MATCH (n) WHERE (n:Work OR n:Chunk) AND n.work_id IN [$old_id, $new_id]
        RETURN labels(n) AS labels, n.work_id AS work_id, count(n) AS count
        ORDER BY labels(n), work_id""",
        {"old_id": old_id, "new_id": new_id},
    )
    assert records == [
        {"labels": ["Chunk"], "work_id": new_id, "count": 1},
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

    graph_chunk_id = await storage.pg.fetch_val(
        """INSERT INTO chunks (work_id, text, granularity, source_class, position)
        VALUES ($1, 'Source text', 'macro', 'secondary', 1)
        RETURNING id""",
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
        "CREATE (:Chunk {chunk_id: $chunk_id, work_id: $work_id})",
        {"chunk_id": str(graph_chunk_id), "work_id": work_id},
    )
