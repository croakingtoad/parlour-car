"""Backfill Neo4j chunk nodes + entity extraction for a work.

Handles the case where PG has chunks but Neo4j has no chunk nodes
(e.g. ingestion was interrupted after embedding but before graph sync).

Usage:
    uv run python scripts/backfill_graph_and_entities.py <work-id>

Theme deduplication can merge nodes and delete relationships. It is excluded
from the default execution plan and must be requested explicitly with
``--deduplicate-themes``.
"""

import argparse
import asyncio
import sys
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class BackfillStep(Enum):
    """Operations available to the graph/entity backfill command."""

    SYNC_CHUNKS = "sync_chunks"
    EXTRACT_ENTITIES = "extract_entities"
    DEDUPLICATE_THEMES = "deduplicate_themes"


DESTRUCTIVE_GRAPH_STEPS = frozenset({BackfillStep.DEDUPLICATE_THEMES})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for a graph/entity backfill."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_id", help="Work identifier to backfill")
    parser.add_argument(
        "--deduplicate-themes",
        action="store_true",
        help=(
            "also merge duplicate themes; this can delete graph nodes and relationships "
            "and is never enabled by audit recommendations"
        ),
    )
    return parser.parse_args(argv)


def build_execution_plan(args: argparse.Namespace) -> tuple[BackfillStep, ...]:
    """Build the operations that the parsed invocation will execute."""
    steps = [BackfillStep.SYNC_CHUNKS, BackfillStep.EXTRACT_ENTITIES]
    if args.deduplicate_themes:
        steps.append(BackfillStep.DEDUPLICATE_THEMES)
    return tuple(steps)


async def main(args: argparse.Namespace) -> None:
    from author_library.config import get_settings
    from author_library.graph.backfill import (
        _run_entity_extraction_for_work,
        backfill_work_graph,
    )
    from author_library.storage.manager import StorageManager

    settings = get_settings()
    storage = StorageManager(settings.database)
    await storage.connect(run_pg_migrations=False, init_neo4j_schema=False)
    execution_plan = build_execution_plan(args)

    try:
        # Get work metadata
        work_row = await storage.pg.fetch_one(
            "SELECT work_id, title, author, source_class, publication_year "
            "FROM works WHERE work_id = $1",
            args.work_id,
        )
        if not work_row:
            print(f"Work not found: {args.work_id}")
            return

        work = dict(work_row)
        print(f"Work: {work['title']} by {work['author']} ({work['source_class']})")

        for step in execution_plan:
            if step is BackfillStep.SYNC_CHUNKS:
                print("\n--- Step 1: Syncing chunk nodes to Neo4j ---")
                chunks_created, errors = await backfill_work_graph(storage, work)
                print(f"Chunk nodes created: {chunks_created}")
                if errors:
                    for error in errors:
                        print(f"  ERROR: {error}")
            elif step is BackfillStep.EXTRACT_ENTITIES:
                print("\n--- Step 2: Running entity extraction ---")
                nodes_created = await _run_entity_extraction_for_work(storage, work, settings)
                print(f"Entity nodes created: {nodes_created}")
            elif step is BackfillStep.DEDUPLICATE_THEMES:
                print("\n--- Optional step: Theme deduplication ---")
                from author_library.embeddings import ProviderRegistry
                from author_library.graph.theme_dedup import deduplicate_themes

                embedding_provider = ProviderRegistry.create(settings)
                dedup_result = await deduplicate_themes(storage.neo4j, embedding_provider)
                print(
                    f"Themes: {dedup_result.original_count} → "
                    f"{dedup_result.canonical_count} "
                    f"(merged {dedup_result.merged_count})"
                )

        print("\nDone!")

    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
