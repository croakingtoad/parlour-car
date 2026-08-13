#!/usr/bin/env python3
"""Restore a work's semantic graph edges from a backup.sh Neo4j export.

Generalises restore_themes_from_backup.py (EXPLORES_THEME only) to every
chunk-to-entity relationship: EXPLORES_THEME, CONCEPT_USED_IN,
REFERENCES_PERSON, MAKES_ARGUMENT.

Use this when Chunk nodes have been rebuilt structurally from PostgreSQL
(graph.backfill.backfill_work_graph) but their entity edges — which cost real
LLM spend to produce — are gone. Written for the 2026-08-13 incident, where a
test teardown fixture deleted 5 production works, and the follow-on orphan
sweep also removed entity nodes referenced only by those works. Missing target
nodes are therefore re-created from the backup's own properties.

Structural edges (PART_OF, AUTHORED) are NOT restored here; rebuild those from
PostgreSQL instead, since PG is authoritative for them.

The export is `src_label, src_props, rel_type, rel_props, tgt_label, tgt_props`
with one record per line, EXCEPT that Chunk text_preview values contain raw
newlines, so records are re-assembled by detecting record starts.

Usage:
    uv run python scripts/restore_work_graph_from_backup.py \
        --rels  /path/to/PREFIX.dump.rels.gz \
        --work-prefix malcolm-guite--            # dry run
    uv run python scripts/restore_work_graph_from_backup.py ... --execute
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

BATCH_SIZE = 2000

#: Only these are restored; PART_OF/AUTHORED come from PostgreSQL.
SEMANTIC_EDGES: dict[str, str] = {
    "EXPLORES_THEME": "Theme",
    "CONCEPT_USED_IN": "Concept",
    "REFERENCES_PERSON": "Person",
    "MAKES_ARGUMENT": "Argument",
}

_RECORD_START = re.compile(r'^"[A-Za-z]+", \{')
_SRC_CHUNK_ID = re.compile(r'chunk_id: "(?P<cid>[0-9a-fA-F-]{32,36})"')
_REL_TYPE = re.compile(r'\}, "(?P<rel>[A-Z_]+)", \{')
_CANON = re.compile(r'canonical_name: "(?P<canon>[^"]*)"')
_NAME = re.compile(r'name: "(?P<name>[^"]*)"')
_CLAIM = re.compile(r'claim: "(?P<claim>[^"]*)"')


def _target_blob(record: str, rel: str, label: str) -> str | None:
    """Return the target property blob for a record, or None."""
    m = re.search(
        r'\}, "' + rel + r'", \{[^}]*\}, "' + label + r'", (\{.*)$',
        record,
        re.S,
    )
    return m.group(1) if m else None


def parse_rels(
    rels_path: Path, work_prefix: str
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Extract semantic edges for chunks whose work_id starts with work_prefix."""
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    needle = f'work_id: "{work_prefix}'

    def handle(record_lines: list[str]) -> None:
        if not record_lines:
            return
        record = "".join(record_lines)
        if not record.startswith('"Chunk", {') or needle not in record:
            return
        rel_match = _REL_TYPE.search(record)
        cid_match = _SRC_CHUNK_ID.search(record)
        if not rel_match or not cid_match:
            return
        rel = rel_match.group("rel")
        label = SEMANTIC_EDGES.get(rel)
        if label is None:
            stats[f"skipped:{rel}"] += 1
            return
        blob = _target_blob(record, rel, label)
        if blob is None:
            stats[f"unparsed_target:{rel}"] += 1
            return
        canon = _CANON.search(blob)
        if not canon or not canon.group("canon"):
            stats[f"no_canonical_name:{rel}"] += 1
            return
        name = _NAME.search(blob)
        claim = _CLAIM.search(blob)
        rows.append(
            {
                "rel": rel,
                "label": label,
                "cid": cid_match.group("cid"),
                "canon": canon.group("canon"),
                "name": name.group("name") if name else None,
                "claim": claim.group("claim") if claim else None,
            }
        )
        stats[rel] += 1

    with gzip.open(rels_path, "rt", encoding="utf-8", errors="replace") as fh:
        next(fh, None)  # header
        current: list[str] = []
        for line in fh:
            if _RECORD_START.match(line) and current:
                handle(current)
                current = [line]
            else:
                current.append(line)
        handle(current)

    return rows, stats


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rels", required=True, type=Path, help="PREFIX.dump.rels.gz")
    parser.add_argument(
        "--work-prefix",
        required=True,
        help="Restore chunks whose work_id starts with this (e.g. malcolm-guite--)",
    )
    parser.add_argument("--execute", action="store_true", help="Apply (default: dry run)")
    args = parser.parse_args()

    if not args.rels.exists():
        parser.error(f"no such file: {args.rels}")

    print(f"parsing {args.rels.name} for work_id prefix {args.work_prefix!r} ...")
    rows, stats = parse_rels(args.rels, args.work_prefix)
    print(f"\nparsed {len(rows)} restorable edges")
    for key, count in sorted(stats.items()):
        print(f"  {key:26s} {count}")

    if not rows:
        print("\nnothing to restore")
        return

    distinct_chunks = len({r['cid'] for r in rows})
    distinct_targets = len({(r['label'], r['canon']) for r in rows})
    print(f"\ndistinct chunks: {distinct_chunks}, distinct targets: {distinct_targets}")

    if not args.execute:
        print("\nDRY RUN — re-run with --execute to apply")
        return

    from author_library.config import get_settings
    from author_library.storage.manager import StorageManager

    settings = get_settings()
    storage = StorageManager(settings.database)
    await storage.connect(run_pg_migrations=False, init_neo4j_schema=False)

    try:
        # Group by (rel, label): neither can be parameterised in Cypher.
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault((row["rel"], row["label"]), []).append(row)

        total_edges = 0
        for (rel, label), group in sorted(groups.items()):
            created = 0
            for start in range(0, len(group), BATCH_SIZE):
                batch = group[start : start + BATCH_SIZE]
                # MERGE the target so a node the orphan sweep removed comes
                # back with its original properties, and an existing shared
                # node is reused rather than duplicated.
                result = await storage.neo4j.execute_write(
                    f"""
                    UNWIND $rows AS row
                    MATCH (c:Chunk {{chunk_id: row.cid}})
                    MERGE (t:{label} {{canonical_name: row.canon}})
                      ON CREATE SET t.name  = coalesce(row.name, row.canon),
                                    t.claim = row.claim
                    MERGE (c)-[:{rel}]->(t)
                    RETURN count(*) AS n
                    """,
                    {"rows": batch},
                )
                created += result[0]["n"] if result else 0
            total_edges += created
            print(f"  {rel:20s} -> {label:10s} {created} edges merged")

        print(f"\ntotal edges merged: {total_edges}")
    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
