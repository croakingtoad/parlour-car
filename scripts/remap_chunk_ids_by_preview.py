#!/usr/bin/env python3
"""Remap Neo4j chunk_ids to PG chunk ids by (granularity, text_preview) match.

Some works (post-Apr-2026 ingests: McGilchrist ×3, Kingsnorth) have Neo4j
chunk nodes whose chunk_ids don't correspond to PG ids at all (dashless hex
from a different ingest pass), while chunk COUNTS match exactly. Their nodes
carry paid entity edges (CONCEPT_USED_IN, MAKES_ARGUMENT, REFERENCES_PERSON),
so deletion is not acceptable — remap the chunk_id property in place instead.

Match key: (granularity, whitespace-normalized text_preview prefix). Only
pairs unique on BOTH sides are remapped; ambiguous/unmatched nodes are
reported and left untouched.

Usage:
    uv run python scripts/remap_chunk_ids_by_preview.py <work-id> [--execute]
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

KEY_LEN = 150


def norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:KEY_LEN]


async def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    work_id = sys.argv[1]
    execute = "--execute" in sys.argv

    from author_library.config import get_settings
    from author_library.storage.manager import StorageManager

    settings = get_settings()
    storage = StorageManager(settings.database)
    await storage.connect(run_pg_migrations=False, init_neo4j_schema=False)
    try:
        pg_chunks = await storage.chunks.list_by_work(work_id)
        pg_map: dict[tuple[str, str], list[str]] = {}
        for c in pg_chunks:
            key = (c.get("granularity", ""), norm(c.get("text", "")))
            pg_map.setdefault(key, []).append(str(c["id"]))

        records = await storage.neo4j.execute_read(
            """MATCH (c:Chunk {work_id: $wid})
            RETURN c.chunk_id AS cid, c.granularity AS g, c.text_preview AS prev""",
            {"wid": work_id},
        )
        pg_ids = {str(c["id"]) for c in pg_chunks}

        pairs: list[dict[str, str]] = []
        already, ambiguous, unmatched = 0, 0, 0
        for r in records:
            if r["cid"] in pg_ids:
                already += 1
                continue
            key = (r["g"] or "", norm(r["prev"]))
            candidates = pg_map.get(key, [])
            if len(candidates) == 1:
                pairs.append({"old": r["cid"], "new": candidates[0]})
            elif len(candidates) > 1:
                ambiguous += 1
            else:
                unmatched += 1

        new_ids = [p["new"] for p in pairs]
        if len(set(new_ids)) != len(new_ids):
            # Two neo nodes mapping to one PG id — drop those pairs
            from collections import Counter

            dupes = {k for k, v in Counter(new_ids).items() if v > 1}
            pairs = [p for p in pairs if p["new"] not in dupes]
            ambiguous += len(dupes)

        print(f"[{work_id}] neo4j chunks: {len(records)}, pg chunks: {len(pg_chunks)}")
        print(f"  already aligned: {already}, remappable: {len(pairs)}, "
              f"ambiguous: {ambiguous}, unmatched: {unmatched}")

        if not execute:
            print("DRY RUN — re-run with --execute to apply.")
            return

        BATCH = 2000
        done = 0
        for i in range(0, len(pairs), BATCH):
            await storage.neo4j.execute_write(
                """UNWIND $pairs AS p
                MATCH (c:Chunk {work_id: $wid, chunk_id: p.old})
                SET c.chunk_id = p.new""",
                {"pairs": pairs[i : i + BATCH], "wid": work_id},
            )
            done += len(pairs[i : i + BATCH])
            print(f"  remapped {done}/{len(pairs)}")
        print("Done.")
    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
