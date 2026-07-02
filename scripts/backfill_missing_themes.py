#!/usr/bin/env python3
"""Targeted entity extraction for chunks lacking EXPLORES_THEME edges.

Completes the 2026-07-02 theme repair (td-aef7c5): after the selective
restore from the Apr 28 backup, runs entity extraction only for chunks
that still have no theme edges — re-ingested works, post-backup ingests,
and partially-covered works. Works whose chunks are missing from Neo4j
entirely (PG↔Neo4j sync gap) get a structural backfill first.

Extraction uses the production EntityExtractor, which feeds existing
Theme canonical_names to the LLM so new extractions reuse the restored
vocabulary. Run deduplicate_themes (real embeddings) once afterwards.

Usage:
    uv run python scripts/backfill_missing_themes.py               # dry run
    uv run python scripts/backfill_missing_themes.py --execute
    uv run python scripts/backfill_missing_themes.py --execute --work <work-id>
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MIN_CHUNK_CHARS = 50


async def main() -> None:
    execute = "--execute" in sys.argv
    only_work = None
    if "--work" in sys.argv:
        only_work = sys.argv[sys.argv.index("--work") + 1]

    from author_library.chunking.models import Chunk, ChunkGranularity
    from author_library.config import get_settings
    from author_library.graph.backfill import backfill_work_graph
    from author_library.graph.entity_extraction import EntityExtractor
    from author_library.storage.manager import StorageManager

    settings = get_settings()
    storage = StorageManager(settings.database)
    await storage.connect(run_pg_migrations=False, init_neo4j_schema=False)
    allowed_grans = {
        g.strip() for g in settings.llm.entity_extraction_granularities.split(",")
    }

    totals = {"works": 0, "eligible": 0, "extracted": 0, "errors": 0}
    try:
        works = await storage.pg.fetch_all(
            "SELECT work_id, title, author, source_class, publication_year "
            "FROM works ORDER BY work_id"
        )
        for work_row in works:
            work = dict(work_row)
            work_id = work["work_id"]
            if only_work and work_id != only_work:
                continue

            r = await storage.neo4j.execute_read(
                "MATCH (c:Chunk {work_id: $wid}) RETURN count(c) AS n",
                {"wid": work_id},
            )
            neo4j_chunks = r[0]["n"]
            pg_chunks = await storage.chunks.list_by_work(work_id)

            if neo4j_chunks == 0 and pg_chunks:
                print(f"[{work_id}] SYNC GAP: 0 Neo4j chunks, {len(pg_chunks)} in PG")
                if execute:
                    created, errors = await backfill_work_graph(storage, work)
                    print(f"[{work_id}]   structural backfill: {created} chunk nodes"
                          + (f", {len(errors)} errors" if errors else ""))
                else:
                    print(f"[{work_id}]   (dry run: would backfill {len(pg_chunks)} chunk nodes)")

            r = await storage.neo4j.execute_read(
                """MATCH (c:Chunk {work_id: $wid})
                WHERE NOT (c)-[:EXPLORES_THEME]->()
                RETURN c.chunk_id AS cid""",
                {"wid": work_id},
            )
            unthemed = {rec["cid"] for rec in r}
            if execute and neo4j_chunks == 0:
                # after structural backfill everything is unthemed
                unthemed = {str(c["id"]) for c in pg_chunks}
            elif not execute and neo4j_chunks == 0:
                unthemed = {str(c["id"]) for c in pg_chunks}

            chunks: list[Chunk] = []
            for pg_chunk in pg_chunks:
                cid = str(pg_chunk["id"])
                if cid not in unthemed:
                    continue
                granularity = pg_chunk.get("granularity", "meso")
                if granularity not in allowed_grans:
                    continue
                text = pg_chunk.get("text", "") or ""
                if len(text.strip()) < MIN_CHUNK_CHARS:
                    continue
                try:
                    chunks.append(Chunk(
                        id=cid,
                        text=text,
                        annotation=pg_chunk.get("annotation"),
                        granularity=ChunkGranularity(granularity),
                        work_id=pg_chunk.get("work_id", work_id),
                        source_class=pg_chunk.get("source_class", work["source_class"]),
                        chapter=pg_chunk.get("chapter"),
                        section=pg_chunk.get("section"),
                        position=pg_chunk.get("position", 0),
                    ))
                except Exception as exc:
                    print(f"[{work_id}]   chunk conversion failed {cid}: {exc}")

            if not chunks:
                continue
            totals["works"] += 1
            totals["eligible"] += len(chunks)
            batches = (len(chunks) + 9) // 10
            print(f"[{work_id}] {len(chunks)} eligible unthemed chunks (~{batches} LLM calls)")

            if not execute:
                continue
            extractor = EntityExtractor(
                storage.neo4j, settings.api_keys, settings.llm, storage=storage
            )
            result = await extractor.extract_and_persist(
                chunks, work_title=work["title"] or "", author=work["author"] or ""
            )
            totals["extracted"] += result.chunks_processed
            totals["errors"] += len(result.errors)
            print(f"[{work_id}]   done: {result.chunks_processed} chunks processed, "
                  f"{result.nodes_created} nodes, {result.edges_created} edges, "
                  f"{len(result.errors)} errors")

        print(f"\nTOTAL: {totals['works']} works, {totals['eligible']} eligible chunks"
              + ("" if execute else f", ~{(totals['eligible'] + 9) // 10} LLM calls (DRY RUN)")
              + (f", {totals['extracted']} extracted, {totals['errors']} errors" if execute else ""))
    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
