"""Diagnose PG vs Neo4j chunk sync issues."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main() -> None:
    from author_library.config import get_settings
    from author_library.storage import StorageManager

    settings = get_settings()
    storage = StorageManager(settings.database)
    await storage.connect(run_pg_migrations=False, init_neo4j_schema=False)

    # Per-work: PG chunks vs Neo4j chunks
    works = await storage.pg.fetch_all(
        "SELECT w.work_id, w.title, w.source_class "
        "FROM works w ORDER BY w.work_id"
    )

    print(f"{'Work ID':60s} {'PG':>6s} {'Neo4j':>6s} {'Diff':>6s} {'Entities':>8s}")
    print("-" * 95)

    total_pg = 0
    total_neo4j = 0
    mismatch_works = []

    for w in works:
        wid = w["work_id"]

        pg = await storage.pg.fetch_val(
            "SELECT count(*) FROM chunks WHERE work_id = $1", wid
        )
        total_pg += pg

        result = await storage.neo4j.execute_read(
            "MATCH (c:Chunk {work_id: $wid}) RETURN count(c) as cnt",
            {"wid": wid},
        )
        neo4j = result[0]["cnt"] if result else 0
        total_neo4j += neo4j

        result = await storage.neo4j.execute_read(
            "MATCH (c:Chunk {work_id: $wid})-[r]->(e) "
            "WHERE NOT e:Work AND NOT e:Chunk "
            "RETURN count(r) as cnt",
            {"wid": wid},
        )
        entities = result[0]["cnt"] if result else 0

        diff = neo4j - pg
        flag = "  !!!" if diff != 0 else ""
        print(f"{wid:60s} {pg:6d} {neo4j:6d} {diff:+6d} {entities:8d}{flag}")

        if diff != 0:
            mismatch_works.append((wid, pg, neo4j, diff))

    print("-" * 95)
    print(f"{'TOTAL':60s} {total_pg:6d} {total_neo4j:6d} {total_neo4j - total_pg:+6d}")

    # Check for Neo4j work_ids not in PG
    result = await storage.neo4j.execute_read(
        "MATCH (c:Chunk) WITH c.work_id as wid, count(c) as cnt "
        "RETURN wid, cnt ORDER BY cnt DESC"
    )
    pg_work_ids = {w["work_id"] for w in works}

    print("\nNeo4j chunk work_ids not in PG works table:")
    orphan_found = False
    for r in result:
        wid = r["wid"]
        if wid not in pg_work_ids:
            orphan_found = True
            print(f"  ORPHAN: {wid:55s} {r['cnt']:6d} chunks")
    if not orphan_found:
        print("  (none)")

    # For mismatched works, check if it's UUID remapping
    if mismatch_works:
        print("\nMismatch details:")
        for wid, pg, neo4j, diff in mismatch_works:
            if diff > 0:
                # More in Neo4j than PG — check if Neo4j has old UUIDs
                result = await storage.neo4j.execute_read(
                    "MATCH (c:Chunk {work_id: $wid}) RETURN c.chunk_id as cid LIMIT 5",
                    {"wid": wid},
                )
                neo4j_sample = [r["cid"] for r in result[:5]]

                pg_sample = await storage.pg.fetch_all(
                    "SELECT id::text FROM chunks WHERE work_id = $1 LIMIT 5", wid
                )
                pg_ids = [r["id"] for r in pg_sample]

                overlap = set(neo4j_sample) & set(pg_ids)
                print(f"  {wid}: Neo4j has {diff:+d} extra chunks")
                print(f"    Neo4j sample: {neo4j_sample[:3]}")
                print(f"    PG sample:    {pg_ids[:3]}")
                print(f"    Overlap in sample: {len(overlap)}/5")
            elif diff < 0:
                print(f"  {wid}: PG has {-diff} chunks missing from Neo4j")

    await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
