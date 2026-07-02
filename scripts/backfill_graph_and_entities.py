"""Backfill Neo4j chunk nodes + entity extraction for a work.

Handles the case where PG has chunks but Neo4j has no chunk nodes
(e.g. ingestion was interrupted after embedding but before graph sync).

Usage:
    uv run python scripts/backfill_graph_and_entities.py <work-id>
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main(work_id: str) -> None:
    import structlog

    from author_library.config import get_settings
    from author_library.graph.backfill import (
        backfill_work_graph,
        _run_entity_extraction_for_work,
    )
    from author_library.storage.manager import StorageManager

    log = structlog.get_logger(__name__)
    settings = get_settings()
    storage = StorageManager(settings.database)
    await storage.connect(run_pg_migrations=False, init_neo4j_schema=False)

    try:
        # Get work metadata
        work_row = await storage.pg.fetch_one(
            "SELECT work_id, title, author, source_class, publication_year "
            "FROM works WHERE work_id = $1",
            work_id,
        )
        if not work_row:
            print(f"Work not found: {work_id}")
            return

        work = dict(work_row)
        print(f"Work: {work['title']} by {work['author']} ({work['source_class']})")

        # Step 1: Sync chunk nodes to Neo4j
        print("\n--- Step 1: Syncing chunk nodes to Neo4j ---")
        chunks_created, errors = await backfill_work_graph(storage, work)
        print(f"Chunk nodes created: {chunks_created}")
        if errors:
            for e in errors:
                print(f"  ERROR: {e}")

        # Step 2: Entity extraction
        print("\n--- Step 2: Running entity extraction ---")
        nodes_created = await _run_entity_extraction_for_work(
            storage, work, settings
        )
        print(f"Entity nodes created: {nodes_created}")

        # Step 3: Theme deduplication
        print("\n--- Step 3: Theme deduplication ---")
        try:
            from author_library.embeddings import ProviderRegistry
            from author_library.graph.theme_dedup import deduplicate_themes

            embedding_provider = ProviderRegistry.create(settings)
            dedup_result = await deduplicate_themes(
                storage.neo4j, embedding_provider
            )
            print(
                f"Themes: {dedup_result.original_count} → "
                f"{dedup_result.canonical_count} "
                f"(merged {dedup_result.merged_count})"
            )
        except Exception as exc:
            print(f"Theme dedup error: {exc}")

        print("\nDone!")

    finally:
        await storage.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/backfill_graph_and_entities.py <work-id>")
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
