"""Unit tests for the transactional work_id rename script."""

from __future__ import annotations

import importlib.util
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


_SCRIPT = Path(__file__).parents[2] / "scripts" / "rename_work_id.py"
_SPEC = importlib.util.spec_from_file_location("rename_work_id", _SCRIPT)
assert _SPEC and _SPEC.loader
rename = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rename
_SPEC.loader.exec_module(rename)


class FakeConnection:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on

    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append(query)
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("injected PostgreSQL failure")
        if query == "SET CONSTRAINTS ALL DEFERRED":
            return "SET CONSTRAINTS"
        return "UPDATE 1"

    async def fetchval(self, query: str, *args: Any) -> int:
        return 0

    async def fetch(self, query: str, *args: Any) -> list[dict[str, str]]:
        return []


class FakeStorage:
    pg = object()


class StatefulPg:
    """In-memory PostgreSQL double that rolls back applied row mutations."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.rows = {
            "works": {"old"},
            "chunks": {"old"},
            "thematic_appearances": {"old"},
            "session_sources": {"old"},
        }
        self.applied_updates: list[str] = []
        self.rolled_back = False

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[StatefulPg]:
        before = {table: values.copy() for table, values in self.rows.items()}
        try:
            yield self
        except Exception:
            self.rows = before
            self.rolled_back = True
            raise

    async def execute(self, query: str, *args: Any) -> str:
        if query == "SET CONSTRAINTS ALL DEFERRED":
            return "SET CONSTRAINTS"
        table = next(table for table in self.rows if f'"{table}"' in query)
        if self.fail_on == table:
            raise RuntimeError("injected PostgreSQL failure")
        new_id, old_id = args
        changed = int(old_id in self.rows[table])
        if changed:
            self.rows[table].remove(old_id)
            self.rows[table].add(new_id)
            self.applied_updates.append(table)
        return f"UPDATE {changed}"

    async def fetch_val(self, query: str, work_id: str) -> int:
        table = re.search(r'FROM "([a-z_]+)"', query)
        assert table is not None
        return int(work_id in self.rows[table.group(1)])

    async def fetch_all(self, *args: Any) -> list[dict[str, Any]]:
        return [
            {
                "schema_name": child.schema,
                "table_name": child.table,
                "column_name": child.column,
                "is_deferrable": True,
                "on_update": "a",
            }
            for child in _children()
        ]


class StatefulNeo4j:
    def __init__(self) -> None:
        self.rows = {"Work": {"old"}, "Chunk": {"old"}}

    async def execute_read(self, _query: str, params: dict[str, str]) -> list[dict[str, int]]:
        work_id = params["work_id"]
        return [
            {
                "work_count": int(work_id in self.rows["Work"]),
                "chunk_count": int(work_id in self.rows["Chunk"]),
            }
        ]


class StatefulStorage:
    def __init__(self, *, fail_on: str) -> None:
        self.pg = StatefulPg(fail_on=fail_on)
        self.neo4j = StatefulNeo4j()


def _children() -> list[Any]:
    return [
        rename.PgChildTable("public", "chunks", "work_id"),
        rename.PgChildTable("public", "thematic_appearances", "work_id"),
        rename.PgChildTable("public", "session_sources", "work_id"),
    ]


def _counts() -> Any:
    return rename.StoreCounts(1, {child.display_name: 1 for child in _children()}, 1, 1)


@pytest.mark.asyncio
async def test_children_update_before_parent() -> None:
    conn = FakeConnection()
    await rename._apply_postgres(conn, "old", "new", _children(), _counts())
    expected_order = ["chunks", "thematic_appearances", "session_sources", "works"]
    actual_order = [
        next(name for name in expected_order if name in query)
        for query in conn.calls[1:]
    ]
    assert actual_order == expected_order


@pytest.mark.asyncio
async def test_mid_transaction_failure_rolls_back_prior_child_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = StatefulStorage(fail_on="thematic_appearances")

    async def clean(*args: Any) -> None:
        return None

    monkeypatch.setattr(rename, "_assert_global_consistency", clean)
    with pytest.raises(RuntimeError, match="injected"):
        await rename.execute_rename(storage, "old", "new")

    assert storage.pg.applied_updates == ["chunks"]
    assert storage.pg.rows == {
        "works": {"old"},
        "chunks": {"old"},
        "thematic_appearances": {"old"},
        "session_sources": {"old"},
    }
    assert storage.pg.rolled_back is True


def test_immediate_fk_constraints_are_refused() -> None:
    with pytest.raises(rename.RenameError, match="immediate foreign key"):
        rename._assert_constraints_are_deferrable(_children())


def test_target_work_id_is_refused() -> None:
    source = _counts()
    target = rename.StoreCounts(1, {child.display_name: 0 for child in _children()}, 1, 0)
    with pytest.raises(rename.RenameError, match="target work_id already exists"):
        rename._assert_source_is_renameable(source, target)


@pytest.mark.asyncio
async def test_local_postcondition_detects_old_id_remaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old = _counts()
    new = _counts()

    async def fake_collect(*args: Any, **kwargs: Any) -> Any:
        return old if args[1] == "old" else new

    monkeypatch.setattr(rename, "collect_counts", fake_collect)
    with pytest.raises(rename.RenameError, match="old work_id remains"):
        await rename._assert_local_postconditions(
            object(), FakeConnection(), "old", "new", _children(), _counts()
        )


@pytest.mark.asyncio
async def test_already_renamed_is_noop_without_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    children = _children()
    empty = rename.StoreCounts(0, {child.display_name: 0 for child in children}, 0, 0)
    completed = rename.StoreCounts(1, {child.display_name: 1 for child in children}, 1, 1)

    async def fake_discover(*args: Any) -> list[Any]:
        return children

    async def fake_collect(_storage: Any, work_id: str, *args: Any, **kwargs: Any) -> Any:
        return empty if work_id == "old" else completed

    async def clean(*args: Any) -> None:
        return None

    monkeypatch.setattr(rename, "discover_child_tables", fake_discover)
    monkeypatch.setattr(rename, "collect_counts", fake_collect)
    monkeypatch.setattr(rename, "_assert_global_consistency", clean)
    assert await rename.run(FakeStorage(), "old", "new", execute=True) == 0
