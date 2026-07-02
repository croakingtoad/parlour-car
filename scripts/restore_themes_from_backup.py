#!/usr/bin/env python3
"""Selectively restore Theme nodes + EXPLORES_THEME edges from a cypher-shell backup.

Repairs the 2026-07-02 incident where test_theme_dedup.py integration tests ran
deduplicate_themes() against the shared production graph with a degenerate mock
embedder, merging all Theme nodes into 3 (see td-aef7c5).

Reads the plain-format node/rel exports produced by backup.sh:
    <prefix>.dump.nodes.gz  — lines: "Label", {props}
    <prefix>.dump.rels.gz   — lines: "SrcLabel", {props}, "REL", {props}, "TgtLabel", {props}

Restore is selective and additive except for one scoped destructive step:
ALL existing Theme nodes are deleted (with their edges) before recreation,
because every surviving Theme node holds merged/corrupted edge sets.
Chunks, Works, Concepts, Persons, Arguments are never touched.

Edges are replayed by chunk_id; chunk_ids absent from the live graph are
skipped (counts reported). Run with --execute to apply; default is dry-run.

Usage:
    uv run python scripts/restore_themes_from_backup.py \
        /home/marty/parlour-backups/neo4j/20260428-030001_daily.dump [--execute]
"""

from __future__ import annotations

import asyncio
import gzip
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import os

from neo4j import AsyncGraphDatabase

BATCH_SIZE = 5000

THEME_LINE = re.compile(r'^"Theme", \{name: "(?P<name>.*)", canonical_name: "(?P<canon>[^"]+)"\}\s*$')
# chunk_id appears as dashed UUID (newer) or dashless 32-hex (older ingests)
EDGE_CHUNK_ID = re.compile(
    r'chunk_id: "(?P<cid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{32})"'
)
# A record is complete when the physical line ends with the Theme tail.
# Records span multiple lines when text_preview contains literal newlines.
EDGE_TAIL = re.compile(r'"EXPLORES_THEME", \{\}, "Theme", \{name: ".*", canonical_name: "(?P<canon>[^"]+)"\}\s*$')


def parse_backup(prefix: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Return ({canonical_name: display_name}, [(chunk_id, canonical_name), ...])."""
    themes: dict[str, str] = {}
    with gzip.open(f"{prefix}.nodes.gz", "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = THEME_LINE.match(line)
            if m:
                themes[m.group("canon")] = m.group("name")

    edges: list[tuple[str, str]] = []
    skipped = 0
    buffer: list[str] = []
    with gzip.open(f"{prefix}.rels.gz", "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            buffer.append(line.rstrip("\n"))
            mt = EDGE_TAIL.search(line)
            if mt is None:
                # Not the end of an EXPLORES_THEME record. Cap the buffer so
                # non-theme rel records don't accumulate unboundedly.
                if len(buffer) > 200:
                    buffer = buffer[-200:]
                continue
            # Trim to this record's head: the most recent line that starts a
            # Chunk record, so stray chunk_ids from earlier buffered records
            # of other rel types cannot leak in.
            head = 0
            for i in range(len(buffer) - 1, -1, -1):
                if buffer[i].startswith('"Chunk", {'):
                    head = i
                    break
            record = "\n".join(buffer[head:])
            buffer = []
            mc = EDGE_CHUNK_ID.search(record)
            if mc:
                edges.append((mc.group("cid"), mt.group("canon")))
            else:
                skipped += 1
    if skipped:
        print(f"WARNING: {skipped} EXPLORES_THEME records had no parseable chunk_id")
    return themes, edges


async def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    prefix = Path(sys.argv[1])
    execute = "--execute" in sys.argv

    themes, edges = parse_backup(prefix)
    edge_themes = {canon for _, canon in edges}
    print(f"backup: {len(themes)} Theme nodes, {len(edges)} EXPLORES_THEME edges")
    unknown = edge_themes - set(themes)
    if unknown:
        print(f"WARNING: {len(unknown)} edge target themes missing from node export: {sorted(unknown)[:5]}")

    driver = AsyncGraphDatabase.driver(
        os.environ.get("DB_NEO4J_URL", "bolt://localhost:7687"),
        auth=(
            os.environ.get("DB_NEO4J_USER", "neo4j"),
            os.environ.get("DB_NEO4J_PASSWORD", "neo4j_dev"),
        ),
    )
    async with driver.session() as s:
        r = await s.run("MATCH (t:Theme) OPTIONAL MATCH (t)-[e]-() RETURN count(DISTINCT t) AS n, count(e) AS edges")
        rec = await r.single()
        print(f"live before: {rec['n']} Theme nodes, {rec['edges']} attached edges")

        chunk_ids = sorted({cid for cid, _ in edges})
        found = 0
        for i in range(0, len(chunk_ids), BATCH_SIZE):
            r = await s.run(
                "UNWIND $ids AS cid MATCH (c:Chunk {chunk_id: cid}) RETURN count(c) AS f",
                ids=chunk_ids[i : i + BATCH_SIZE],
            )
            found += (await r.single())["f"]
        print(f"chunk survival: {found}/{len(chunk_ids)} backup chunk_ids exist live")

        if not execute:
            print("DRY RUN — no changes made. Re-run with --execute to apply.")
            await driver.close()
            return

        print("PHASE A: deleting all existing (corrupted) Theme nodes...")
        r = await s.run("MATCH (t:Theme) DETACH DELETE t RETURN count(t) AS deleted")
        print(f"  deleted {(await r.single())['deleted']} Theme nodes")

        print(f"PHASE B: recreating {len(themes)} Theme nodes...")
        rows = [{"canon": c, "name": n} for c, n in themes.items()]
        await s.run(
            "UNWIND $rows AS row MERGE (t:Theme {canonical_name: row.canon}) SET t.name = row.name",
            rows=rows,
        )

        print(f"PHASE C: replaying {len(edges)} edges in batches of {BATCH_SIZE}...")
        created = 0
        pairs = [{"cid": cid, "canon": canon} for cid, canon in edges]
        for i in range(0, len(pairs), BATCH_SIZE):
            result = await s.run(
                """UNWIND $pairs AS p
                MATCH (c:Chunk {chunk_id: p.cid})
                MATCH (t:Theme {canonical_name: p.canon})
                MERGE (c)-[:EXPLORES_THEME]->(t)""",
                pairs=pairs[i : i + BATCH_SIZE],
            )
            summary = await result.consume()
            created += summary.counters.relationships_created
            print(f"  batch {i // BATCH_SIZE + 1}: +{summary.counters.relationships_created} (total {created})")

        print("VERIFY:")
        r = await s.run("MATCH (t:Theme) RETURN count(t) AS n")
        print(f"  Theme nodes: {(await r.single())['n']}")
        r = await s.run("MATCH ()-[e:EXPLORES_THEME]->() RETURN count(e) AS n")
        print(f"  EXPLORES_THEME edges: {(await r.single())['n']}")
        r = await s.run(
            "MATCH (t:Theme) WHERE NOT (t)--() RETURN count(t) AS orphans"
        )
        print(f"  orphan themes (no edges — gap works): {(await r.single())['orphans']}")
    await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
