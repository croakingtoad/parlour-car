"""One-time cleanup: remove stale Neo4j Chunk nodes and orphan Work/TestNodes.

Background: Three works were re-ingested while delete_chunks_for_work timed
out (single unbatched DETACH DELETE). PG got new UUIDs; old Neo4j Chunk nodes
were never removed, leaving ~29k stale nodes alongside the current set.

This script:
  1. For each affected work: deletes Neo4j Chunk nodes whose chunk_id does not
     match any current PG chunk id (batched, 500 at a time).
  2. Deletes orphan Work nodes that have no corresponding PG work.
  3. Deletes leftover TestNodes.

Dry-run by default. Pass --execute to apply changes.
"""

import asyncio
import argparse
import os
import sys

import asyncpg
from neo4j import AsyncGraphDatabase

AFFECTED_WORKS = [
    "richard-holmes--coleridge-darker-reflections-1804-1834",
    "samuel-taylor-coleridge--untitled",
    "richard-holmes--coleridge-early-visions-1772-1804",
]

ORPHAN_WORK_NODES = [
    "coleridge--biographia",
    "work-a",
    "work-b",
    "work-1",
    "work-2",
    "work-primary",
    "work-no-links",
    "work-contextual",
    "william-shakespeare--untitled",
    "unknown--untitled",
]

BATCH = 500


async def get_pg_chunk_ids(conn: asyncpg.Connection, work_id: str) -> set[str]:
    rows = await conn.fetch("SELECT id FROM chunks WHERE work_id = $1", work_id)
    return {str(r["id"]).replace("-", "") for r in rows}


async def delete_stale_chunks(session, work_id: str, stale_ids: list[str], dry_run: bool) -> int:
    if dry_run:
        print(f"  [dry-run] would delete {len(stale_ids)} stale Chunk nodes")
        return len(stale_ids)

    deleted = 0
    for i in range(0, len(stale_ids), BATCH):
        batch = stale_ids[i : i + BATCH]
        result = await session.run(
            "MATCH (c:Chunk) WHERE c.chunk_id IN $ids OR replace(c.chunk_id, '-', '') IN $ids "
            "DETACH DELETE c RETURN count(c) AS n",
            ids=batch,
        )
        rec = await result.single()
        deleted += rec["n"] if rec else 0
        print(f"  deleted batch {i // BATCH + 1}: {rec['n'] if rec else 0} nodes")
    return deleted


async def main(dry_run: bool) -> None:
    pg_url = os.environ["DB_POSTGRES_URL"]
    neo4j_url = os.environ["DB_NEO4J_URL"]
    neo4j_user = os.environ["DB_NEO4J_USER"]
    neo4j_password = os.environ["DB_NEO4J_PASSWORD"]

    pg = await asyncpg.connect(pg_url)
    driver = AsyncGraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))

    mode = "DRY RUN" if dry_run else "EXECUTE"
    print(f"\n=== Neo4j orphan cleanup [{mode}] ===\n")

    total_stale = 0
    async with driver.session() as session:
        # Step 1: stale Chunk nodes for affected works
        for work_id in AFFECTED_WORKS:
            print(f"Work: {work_id}")
            pg_ids = await get_pg_chunk_ids(pg, work_id)

            result = await session.run(
                "MATCH (c:Chunk {work_id: $wid}) RETURN c.chunk_id AS cid",
                wid=work_id,
            )
            records = await result.data()
            neo4j_ids = {r["cid"].replace("-", "") for r in records}

            stale = neo4j_ids - pg_ids
            current_overlap = neo4j_ids & pg_ids
            print(f"  PG: {len(pg_ids)} | Neo4j: {len(neo4j_ids)} | Overlap: {len(current_overlap)} | Stale: {len(stale)}")

            if stale:
                stale_list = list(stale)
                n = await delete_stale_chunks(session, work_id, stale_list, dry_run)
                total_stale += n
            else:
                print("  clean — nothing to do")
            print()

        # Step 2: orphan Work nodes
        print("Orphan Work nodes:")
        for work_id in ORPHAN_WORK_NODES:
            result = await session.run(
                "MATCH (w:Work {work_id: $wid}) RETURN count(w) AS n",
                wid=work_id,
            )
            rec = await result.single()
            exists = rec and rec["n"] > 0
            if exists:
                print(f"  {work_id} — found")
                if not dry_run:
                    await session.run(
                        "MATCH (w:Work {work_id: $wid}) DETACH DELETE w",
                        wid=work_id,
                    )
                    print(f"    deleted")
                else:
                    print(f"    [dry-run] would delete")
            else:
                print(f"  {work_id} — not found, skip")

        # Step 3: TestNodes
        print("\nTestNodes:")
        result = await session.run("MATCH (n:TestNode) RETURN count(n) AS n")
        rec = await result.single()
        test_count = rec["n"] if rec else 0
        print(f"  found {test_count} TestNode(s)")
        if test_count > 0:
            if not dry_run:
                await session.run("MATCH (n:TestNode) DETACH DELETE n")
                print("  deleted")
            else:
                print("  [dry-run] would delete")

    await pg.close()
    await driver.close()

    print(f"\n=== Summary ===")
    print(f"Stale Chunk nodes {'would be ' if dry_run else ''}deleted: {total_stale}")
    if dry_run:
        print("\nRe-run with --execute to apply changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="Apply changes (default is dry-run)")
    args = parser.parse_args()

    asyncio.run(main(dry_run=not args.execute))
