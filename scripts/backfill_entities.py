"""Backfill entity extraction for works that have Neo4j chunk nodes but no entities.

Usage:
    uv run python scripts/backfill_entities.py
    uv run python scripts/backfill_entities.py --work-id malcolm-guite--faith-hope-and-poetry
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main(work_id_filter: str | None = None) -> None:
    import structlog

    from author_library.config import DatabaseSettings, get_settings
    from author_library.embeddings import ProviderRegistry
    from author_library.graph.backfill import _run_entity_extraction_for_work
    from author_library.storage.manager import StorageManager

    log = structlog.get_logger(__name__)
    settings = get_settings()
    db_settings = DatabaseSettings()

    storage = StorageManager(db_settings)
    await storage.connect(run_pg_migrations=False, init_neo4j_schema=False)
    try:
        embedding_provider = ProviderRegistry.create(settings)

        # Get all works from PG
        rows = await storage.pg.fetch_all(
            "SELECT work_id, title, author, source_class, publication_year FROM works ORDER BY work_id"
        )

        # Check which have entity relationships in Neo4j
        for row in rows:
            wid = row["work_id"]
            if work_id_filter and wid != work_id_filter:
                continue

            result = await storage.neo4j.execute_read(
                "MATCH (c:Chunk {work_id: $wid})-[r]->(e) WHERE NOT e:Work RETURN count(r) as entity_edges",
                {"wid": wid},
            )
            entity_edges = result[0]["entity_edges"] if result else 0

            if entity_edges > 0:
                log.info("skip_has_entities", work_id=wid, entity_edges=entity_edges)
                continue

            log.info("extracting_entities", work_id=wid, title=row["title"])
            nodes_created = await _run_entity_extraction_for_work(
                storage,
                dict(row),
                settings,
            )
            log.info("done", work_id=wid, nodes_created=nodes_created)
    finally:
        await storage.close()


if __name__ == "__main__":
    filter_id = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(filter_id))
