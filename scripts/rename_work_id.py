#!/usr/bin/env python3
"""Safely rename a work_id across PostgreSQL and Neo4j.

The default is a read-only dry run.  Applying a rename requires ``--execute``.
The PostgreSQL changes are made in one transaction, with every FK child updated
before ``works``.  Neo4j nodes are updated in place; no nodes are deleted or
recreated.

Usage:
    uv run python scripts/rename_work_id.py --from OLD_ID --to NEW_ID [--execute]
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class RenameError(RuntimeError):
    """The requested rename cannot be performed safely."""


@dataclass(frozen=True)
class PgChildTable:
    schema: str
    table: str
    column: str
    is_deferrable: bool = False
    on_update: str = "a"

    @property
    def display_name(self) -> str:
        return self.table if self.schema == "public" else f"{self.schema}.{self.table}"


@dataclass(frozen=True)
class StoreCounts:
    pg_work: int
    pg_children: dict[str, int]
    neo_work: int
    neo_chunk: int


@dataclass(frozen=True)
class Neo4jRename:
    """The exact nodes committed by one Neo4j rename transaction."""

    work_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]

    @property
    def counts(self) -> tuple[int, int]:
        return len(self.work_ids), len(self.chunk_ids)


class PgConnection(Protocol):
    async def execute(self, query: str, *args: Any) -> str: ...

    async def fetchval(self, query: str, *args: Any) -> Any: ...

    async def fetch(self, query: str, *args: Any) -> Sequence[Any]: ...


class Storage(Protocol):
    pg: Any
    neo4j: Any


# Catalog discovery means a future FK table is included automatically instead
# of making a parent update fail halfway through an otherwise valid repair.
CHILD_TABLES_SQL = """
SELECT child_ns.nspname AS schema_name,
       child.relname AS table_name,
       child_attr.attname AS column_name,
       con.condeferrable AS is_deferrable,
       con.confupdtype AS on_update
FROM pg_constraint AS con
JOIN pg_class AS parent ON parent.oid = con.confrelid
JOIN pg_namespace AS parent_ns ON parent_ns.oid = parent.relnamespace
JOIN pg_class AS child ON child.oid = con.conrelid
JOIN pg_namespace AS child_ns ON child_ns.oid = child.relnamespace
JOIN LATERAL unnest(con.conkey) AS key(attnum) ON TRUE
JOIN pg_attribute AS child_attr
  ON child_attr.attrelid = child.oid AND child_attr.attnum = key.attnum
JOIN LATERAL unnest(con.confkey) AS ref(attnum) ON TRUE
JOIN pg_attribute AS parent_attr
  ON parent_attr.attrelid = parent.oid AND parent_attr.attnum = ref.attnum
WHERE con.contype = 'f'
  AND parent_ns.nspname = 'public'
  AND parent.relname = 'works'
  AND parent_attr.attname = 'work_id'
ORDER BY child_ns.nspname, child.relname, child_attr.attname
"""

NEO_COUNTS_QUERY = """
CALL () {
  MATCH (w:Work {work_id: $work_id})
  RETURN count(w) AS work_count
}
CALL () {
  MATCH (c:Chunk {work_id: $work_id})
  RETURN count(c) AS chunk_count
}
RETURN work_count, chunk_count
"""

NEO_RENAME_QUERY = """
CALL () {
  MATCH (w:Work {work_id: $from_id})
  SET w.work_id = $to_id
  RETURN count(w) AS work_count, collect(elementId(w)) AS work_ids
}
CALL () {
  MATCH (c:Chunk {work_id: $from_id})
  SET c.work_id = $to_id
  RETURN count(c) AS chunk_count, collect(elementId(c)) AS chunk_ids
}
RETURN work_count, work_ids, chunk_count, chunk_ids
"""

# Both SET clauses are conditional on every captured identity still carrying
# the replacement id. A changed scope therefore commits a no-op, never a
# partial or broad reverse rename.
NEO_COMPENSATE_QUERY = """
CALL () {
  UNWIND $work_ids AS work_element_id
  OPTIONAL MATCH (w:Work)
  WHERE elementId(w) = work_element_id AND w.work_id = $from_id
  RETURN collect(w) AS works
}
CALL () {
  UNWIND $chunk_ids AS chunk_element_id
  OPTIONAL MATCH (c:Chunk)
  WHERE elementId(c) = chunk_element_id AND c.work_id = $from_id
  RETURN collect(c) AS chunks
}
WITH works, chunks,
     size($work_ids) AS expected_work_count,
     size($chunk_ids) AS expected_chunk_count
WITH works, chunks, expected_work_count, expected_chunk_count,
     size(works) = expected_work_count AND size(chunks) = expected_chunk_count
       AS exact_scope
FOREACH (w IN CASE WHEN exact_scope THEN works ELSE [] END |
  SET w.work_id = $to_id
)
FOREACH (c IN CASE WHEN exact_scope THEN chunks ELSE [] END |
  SET c.work_id = $to_id
)
RETURN size(works) AS work_count,
       expected_work_count,
       size(chunks) AS chunk_count,
       expected_chunk_count
"""


def _quote_identifier(identifier: str) -> str:
    """Quote a catalog-derived PostgreSQL identifier defensively."""
    return '"' + identifier.replace('"', '""') + '"'


def _updated_row_count(status: str) -> int:
    match = re.fullmatch(r"UPDATE (\d+)", status.strip())
    if not match:
        raise RenameError(f"unexpected PostgreSQL update status: {status!r}")
    return int(match.group(1))


def _row_value(row: Any, key: str) -> Any:
    """Read a value from either asyncpg records or test dictionaries."""
    return row[key]


async def discover_child_tables(pg: Any) -> list[PgChildTable]:
    rows = await pg.fetch_all(CHILD_TABLES_SQL)
    return [
        PgChildTable(
            schema=str(_row_value(row, "schema_name")),
            table=str(_row_value(row, "table_name")),
            column=str(_row_value(row, "column_name")),
            is_deferrable=bool(_row_value(row, "is_deferrable")),
            on_update=str(_row_value(row, "on_update")),
        )
        for row in rows
    ]


async def _pg_count(pg: Any, table: str, column: str, work_id: str) -> int:
    query = (
        f"SELECT count(*) FROM {_quote_identifier(table)} WHERE {_quote_identifier(column)} = $1"
    )
    # PostgresPool calls the helper fetch_val; an acquired asyncpg connection
    # calls it fetchval.  Supporting both keeps post-condition reads on the
    # same transaction connection as the updates.
    fetch_value = getattr(pg, "fetch_val", None) or pg.fetchval
    return int(await fetch_value(query, work_id))


async def collect_counts(
    storage: Storage, work_id: str, children: Sequence[PgChildTable], *, pg: Any | None = None
) -> StoreCounts:
    """Collect all rename-relevant counts without mutating either store."""
    pg_reader = pg or storage.pg
    pg_work = await _pg_count(pg_reader, "works", "work_id", work_id)
    child_counts = {
        child.display_name: await _pg_count(pg_reader, child.table, child.column, work_id)
        for child in children
    }
    records = await storage.neo4j.execute_read(NEO_COUNTS_QUERY, {"work_id": work_id})
    if len(records) != 1:
        raise RenameError("Neo4j count query returned an unexpected result")
    return StoreCounts(
        pg_work=pg_work,
        pg_children=child_counts,
        neo_work=int(records[0]["work_count"]),
        neo_chunk=int(records[0]["chunk_count"]),
    )


def _format_counts(label: str, counts: StoreCounts) -> list[str]:
    lines = [f"{label} PostgreSQL:", f"  works: {counts.pg_work}"]
    for table, count in counts.pg_children.items():
        lines.append(f"  {table}: {count}")
    lines.extend(
        [
            f"{label} Neo4j:",
            f"  Work: {counts.neo_work}",
            f"  Chunk: {counts.neo_chunk}",
        ]
    )
    return lines


def _assert_source_is_renameable(source: StoreCounts, target: StoreCounts) -> None:
    if target != StoreCounts(0, {name: 0 for name in target.pg_children}, 0, 0):
        raise RenameError("target work_id already exists in PostgreSQL or Neo4j")
    if source.pg_work != 1:
        raise RenameError(
            f"expected exactly one PostgreSQL works row for source, found {source.pg_work}"
        )
    if source.neo_work != 1:
        raise RenameError(
            f"expected exactly one Neo4j Work node for source, found {source.neo_work}"
        )


async def _assert_global_consistency(storage: Storage) -> None:
    from author_library.graph import check_pg_neo4j_consistency

    report = await check_pg_neo4j_consistency(storage)
    if not report["is_consistent"]:
        raise RenameError(f"PG/Neo4j consistency check failed: {report}")


async def _apply_postgres(
    conn: PgConnection,
    old_id: str,
    new_id: str,
    children: Sequence[PgChildTable],
    expected: StoreCounts,
) -> None:
    """Update every FK child first, then the parent, on one connection."""
    # `NO ACTION` FKs are normally immediate.  With deferred checks, child
    # rows may point at the replacement id until the parent update completes.
    await conn.execute("SET CONSTRAINTS ALL DEFERRED")
    for child in children:
        table = f"{_quote_identifier(child.schema)}.{_quote_identifier(child.table)}"
        status = await conn.execute(
            f"UPDATE {table} SET {_quote_identifier(child.column)} = $1 "
            f"WHERE {_quote_identifier(child.column)} = $2",
            new_id,
            old_id,
        )
        actual = _updated_row_count(status)
        expected_rows = expected.pg_children[child.display_name]
        if actual != expected_rows:
            raise RenameError(
                f"{child.display_name}: expected to update {expected_rows} rows, updated {actual}"
            )

    status = await conn.execute("UPDATE works SET work_id = $1 WHERE work_id = $2", new_id, old_id)
    if _updated_row_count(status) != 1:
        raise RenameError("works: expected to update exactly one row")


async def _apply_neo4j(storage: Storage, old_id: str, new_id: str) -> list[Any]:
    return await storage.neo4j.execute_write(NEO_RENAME_QUERY, {"from_id": old_id, "to_id": new_id})


def _neo4j_rename_result(records: list[Any]) -> Neo4jRename:
    if len(records) != 1:
        raise RenameError("Neo4j rename returned an unexpected result")
    actual = records[0]
    work_ids = tuple(str(element_id) for element_id in actual["work_ids"])
    chunk_ids = tuple(str(element_id) for element_id in actual["chunk_ids"])
    result = Neo4jRename(work_ids=work_ids, chunk_ids=chunk_ids)
    reported_counts = int(actual["work_count"]), int(actual["chunk_count"])
    if result.counts != reported_counts:
        raise RenameError(
            "Neo4j rename returned counts inconsistent with captured identities: "
            f"counts Work={reported_counts[0]}, Chunk={reported_counts[1]}; "
            f"identities Work={result.counts[0]}, Chunk={result.counts[1]}"
        )
    if len(set(work_ids)) != len(work_ids) or len(set(chunk_ids)) != len(chunk_ids):
        raise RenameError("Neo4j rename returned duplicate captured identities")
    return result


def _assert_neo4j_affected_counts(actual: tuple[int, int], expected: tuple[int, int]) -> None:
    if actual != expected:
        raise RenameError(
            f"Neo4j rename count changed during operation: Work={actual[0]}, Chunk={actual[1]}"
        )


async def _compensate_neo4j(
    storage: Storage,
    old_id: str,
    new_id: str,
    forward: Neo4jRename | None,
    preflight_counts: tuple[int, int],
    cause: Exception,
) -> None:
    reverse_counts: tuple[int, int] | None = None
    try:
        if forward is None:
            raise RenameError(
                "forward: unavailable; reverse: not attempted"
            )
        records = await storage.neo4j.execute_write(
            NEO_COMPENSATE_QUERY,
            {
                "from_id": new_id,
                "to_id": old_id,
                "work_ids": list(forward.work_ids),
                "chunk_ids": list(forward.chunk_ids),
            },
        )
        if len(records) != 1:
            raise RenameError("reverse: unexpected Neo4j result")
        actual = records[0]
        reverse_counts = int(actual["work_count"]), int(actual["chunk_count"])
        expected_counts = int(actual["expected_work_count"]), int(actual["expected_chunk_count"])
        if reverse_counts != expected_counts or expected_counts != forward.counts:
            raise RenameError(
                "reverse: captured identity scope did not match before mutation: "
                f"matched Work={reverse_counts[0]}/{expected_counts[0]}, "
                f"Chunk={reverse_counts[1]}/{expected_counts[1]}"
            )
    except Exception as exc:
        forward_details = (
            f"Work={forward.counts[0]}, Chunk={forward.counts[1]}"
            if forward is not None
            else "unavailable"
        )
        reverse_details = (
            f"Work={reverse_counts[0]}, Chunk={reverse_counts[1]}"
            if reverse_counts is not None
            else "not attempted"
        )
        raise RenameError(
            "Neo4j compensation failed; manual repair required: "
            f"rename Work/Chunk work_id from {new_id!r} back to {old_id!r}. "
            f"Preflight Work={preflight_counts[0]}, Chunk={preflight_counts[1]}; "
            f"forward: {forward_details}; reverse: {reverse_details}. "
            f"Cause: {cause}. Reverse error: {exc}"
        ) from exc


async def _assert_local_postconditions(
    storage: Storage,
    conn: PgConnection,
    old_id: str,
    new_id: str,
    children: Sequence[PgChildTable],
    expected: StoreCounts,
) -> None:
    old_counts = await collect_counts(storage, old_id, children, pg=conn)
    new_counts = await collect_counts(storage, new_id, children, pg=conn)
    empty = StoreCounts(0, {child.display_name: 0 for child in children}, 0, 0)
    if old_counts != empty:
        raise RenameError(f"old work_id remains after rename: {old_counts}")
    if new_counts != expected:
        raise RenameError(
            f"new work_id post-condition failed: expected {expected}, got {new_counts}"
        )


def _assert_constraints_are_deferrable(children: Sequence[PgChildTable]) -> None:
    """Reject an execution that cannot satisfy immediate NO ACTION FKs."""
    unsafe = [child.display_name for child in children if not child.is_deferrable]
    if unsafe:
        joined = ", ".join(unsafe)
        raise RenameError(
            "cannot execute child-before-parent rename: immediate foreign key "
            f"constraint(s) on {joined}. Dry-run is safe; make the constraints "
            "DEFERRABLE in an approved, rehearsed migration before executing."
        )


async def execute_rename(storage: Storage, old_id: str, new_id: str) -> StoreCounts:
    """Apply the coordinated rename and assert all required post-conditions."""
    children = await discover_child_tables(storage.pg)
    source = await collect_counts(storage, old_id, children)
    target = await collect_counts(storage, new_id, children)
    _assert_source_is_renameable(source, target)
    _assert_constraints_are_deferrable(children)
    await _assert_global_consistency(storage)

    neo_changed = False
    neo_affected: Neo4jRename | None = None
    pg_committed = False
    try:
        async with storage.pg.transaction() as conn:
            await _apply_postgres(conn, old_id, new_id, children, source)
            records = await _apply_neo4j(storage, old_id, new_id)
            # execute_write returns only after its transaction commits. Record
            # that fact before inspecting its result, because every validation
            # below can raise and must trigger the reverse rename.
            neo_changed = True
            neo_affected = _neo4j_rename_result(records)
            _assert_neo4j_affected_counts(neo_affected.counts, (source.neo_work, source.neo_chunk))
            await _assert_local_postconditions(storage, conn, old_id, new_id, children, source)
        pg_committed = True
        await _assert_global_consistency(storage)
        return source
    except Exception as exc:
        # PostgreSQL rolls back automatically while inside its transaction. If
        # Neo4j committed before the PostgreSQL context exited; put it back if
        # the enclosing PG transaction is about to roll back.  If a later
        # global read detects concurrent external damage after both commits,
        # the tool fails loudly rather than attempting an unsafe second rename.
        if neo_changed and not pg_committed:
            await _compensate_neo4j(
                storage,
                old_id,
                new_id,
                neo_affected,
                (source.neo_work, source.neo_chunk),
                exc,
            )
        raise


def _is_completed_noop(source: StoreCounts, target: StoreCounts) -> bool:
    source_empty = StoreCounts(0, {name: 0 for name in source.pg_children}, 0, 0)
    return source == source_empty and target.pg_work == 1 and target.neo_work == 1


async def run(storage: Storage, old_id: str, new_id: str, *, execute: bool) -> int:
    if old_id == new_id:
        raise RenameError("--from and --to must differ")
    children = await discover_child_tables(storage.pg)
    source = await collect_counts(storage, old_id, children)
    target = await collect_counts(storage, new_id, children)

    print("=== work_id rename ===")
    print(f"from: {old_id}")
    print(f"to:   {new_id}")
    print(*_format_counts("source", source), sep="\n")
    print(*_format_counts("target", target), sep="\n")

    if _is_completed_noop(source, target):
        await _assert_global_consistency(storage)
        print("NO-OP — source is already absent and target is present.")
        return 0
    _assert_source_is_renameable(source, target)
    if not execute:
        print("DRY RUN — no PostgreSQL rows or Neo4j nodes were updated.")
        return 0

    await execute_rename(storage, old_id, new_id)
    print("Rename complete — post-conditions and PG/Neo4j consistency passed.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="from_id", required=True, help="existing work_id")
    parser.add_argument("--to", dest="to_id", required=True, help="replacement work_id")
    parser.add_argument(
        "--execute", action="store_true", help="apply the rename (default: dry run only)"
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    from author_library.config import get_settings
    from author_library.storage.manager import StorageManager

    storage = StorageManager(get_settings().database)
    await storage.connect(run_pg_migrations=False, init_neo4j_schema=False)
    try:
        return await run(storage, args.from_id, args.to_id, execute=args.execute)
    finally:
        await storage.close()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except RenameError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
