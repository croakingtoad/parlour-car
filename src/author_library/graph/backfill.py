"""Backfill missing graph data from PostgreSQL into Neo4j.

When works are ingested with Neo4j connectivity issues, or when graph nodes
are removed during re-ingestion, PG can have works/chunks that are completely
absent from Neo4j.  This module detects the discrepancy and reconstructs the
graph layer from the PG source of truth.

The backfill is fully idempotent — all Neo4j writes use MERGE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from author_library.chunking.models import Chunk, ChunkGranularity

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


@dataclass
class BackfillResult:
    """Statistics from a backfill run."""

    works_checked: int = 0
    works_missing: int = 0
    works_backfilled: int = 0
    chunks_created: int = 0
    entities_extracted: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "works_checked": self.works_checked,
            "works_missing": self.works_missing,
            "works_backfilled": self.works_backfilled,
            "chunks_created": self.chunks_created,
            "entities_extracted": self.entities_extracted,
            "errors": self.errors,
        }


async def get_pg_work_ids(storage: StorageManager) -> list[dict[str, Any]]:
    """Get all works from PostgreSQL with their metadata.

    Returns a list of dicts with work_id, title, author, source_class,
    and publication_year for each work.
    """
    rows = await storage.pg.fetch_all(
        "SELECT work_id, title, author, source_class, publication_year "
        "FROM works ORDER BY work_id"
    )
    return [dict(r) for r in rows]


async def get_neo4j_work_ids(storage: StorageManager) -> set[str]:
    """Get all work_ids present in Neo4j."""
    records = await storage.neo4j.execute_read(
        "MATCH (w:Work) RETURN w.work_id AS work_id"
    )
    return {r["work_id"] for r in records}


async def get_neo4j_chunk_ids_for_work(
    storage: StorageManager, work_id: str
) -> set[str]:
    """Get all chunk_ids in Neo4j for a specific work."""
    records = await storage.neo4j.execute_read(
        "MATCH (c:Chunk {work_id: $work_id}) RETURN c.chunk_id AS chunk_id",
        {"work_id": work_id},
    )
    return {r["chunk_id"] for r in records}


async def backfill_work_graph(
    storage: StorageManager,
    work: dict[str, Any],
) -> tuple[int, list[str]]:
    """Backfill a single work's graph data from PG into Neo4j.

    Creates the Work node, Author->Work edge, Chunk nodes, and
    Chunk-[:PART_OF]->Work edges.

    Args:
        storage: The StorageManager with active PG and Neo4j connections.
        work: Dict with work_id, title, author, source_class, publication_year.

    Returns:
        Tuple of (chunks_created, errors).
    """
    work_id = work["work_id"]
    errors: list[str] = []

    # 1. Upsert Work node
    await storage.graph.upsert_work_node({
        "work_id": work_id,
        "title": work["title"],
        "author": work["author"],
        "source_class": work["source_class"],
        "publication_year": work.get("publication_year"),
    })

    log.info("backfill_work_node_created", work_id=work_id, title=work["title"])

    # 2. Upsert Author node and AUTHORED->Work edge
    # Derive author_id from work_id (everything before the --)
    author_id = work_id.split("--")[0] if "--" in work_id else work["author"]
    await storage.neo4j.execute_write(
        """MERGE (a:Author {author_id: $author_id})
        SET a.canonical_name = $name
        WITH a
        MATCH (w:Work {work_id: $work_id})
        MERGE (a)-[:AUTHORED]->(w)""",
        {
            "author_id": author_id,
            "name": work["author"],
            "work_id": work_id,
        },
    )

    # 3. Get all chunks for this work from PG
    pg_chunks = await storage.chunks.list_by_work(work_id)
    if not pg_chunks:
        log.warning("backfill_no_chunks_in_pg", work_id=work_id)
        return 0, errors

    # 4. Get existing chunk IDs from Neo4j for this work
    existing_chunk_ids = await get_neo4j_chunk_ids_for_work(storage, work_id)

    # 5. Create chunk nodes for each PG chunk missing from Neo4j
    chunks_created = 0
    for pg_chunk in pg_chunks:
        chunk_id = str(pg_chunk["id"])
        if chunk_id in existing_chunk_ids:
            continue

        chunk_node: dict[str, Any] = {
            "chunk_id": chunk_id,
            "work_id": work_id,
            "text_preview": (pg_chunk.get("text") or "")[:200],
            "granularity": pg_chunk.get("granularity", "meso"),
            "source_class": pg_chunk.get("source_class", work["source_class"]),
        }

        try:
            await storage.graph.upsert_chunk_node(chunk_node)
            chunks_created += 1
        except Exception as exc:
            error_msg = f"Failed to create chunk node {chunk_id}: {exc}"
            log.error("backfill_chunk_failed", chunk_id=chunk_id, error=str(exc))
            errors.append(error_msg)

    log.info(
        "backfill_chunks_created",
        work_id=work_id,
        chunks_created=chunks_created,
        total_pg_chunks=len(pg_chunks),
        already_in_neo4j=len(existing_chunk_ids),
    )

    return chunks_created, errors


async def backfill_missing_graph_data(
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
    settings: Settings,
    *,
    run_entity_extraction: bool = True,
) -> BackfillResult:
    """Find works in PG missing from Neo4j and backfill their graph data.

    This is the main entry point for the backfill operation. It:
    1. Gets all work_ids from PG
    2. Gets all work_ids from Neo4j
    3. Finds the diff (works in PG but not Neo4j)
    4. For each missing work: creates Work + Chunk nodes with relationships
    5. Optionally runs entity extraction for the missing works

    Args:
        storage: StorageManager with active connections.
        embedding_provider: Embedding provider (needed for entity extraction context).
        settings: Application settings (API keys, LLM config for extraction).
        run_entity_extraction: Whether to run LLM entity extraction on
            backfilled chunks. Default True. Set False for structural-only
            backfill (faster, no API cost).

    Returns:
        BackfillResult with statistics.
    """
    result = BackfillResult()

    # Step 1: Get all works from PG
    pg_works = await get_pg_work_ids(storage)
    result.works_checked = len(pg_works)

    if not pg_works:
        log.info("backfill_no_works_in_pg")
        return result

    # Step 2: Get all work_ids from Neo4j
    neo4j_work_ids = await get_neo4j_work_ids(storage)

    # Step 3: Find missing works
    missing_works = [w for w in pg_works if w["work_id"] not in neo4j_work_ids]
    result.works_missing = len(missing_works)

    if not missing_works:
        log.info(
            "backfill_all_works_present",
            pg_works=len(pg_works),
            neo4j_works=len(neo4j_work_ids),
        )
        return result

    log.info(
        "backfill_missing_works_found",
        missing_count=len(missing_works),
        missing_work_ids=[w["work_id"] for w in missing_works],
    )

    # Step 4: Backfill each missing work
    for work in missing_works:
        work_id = work["work_id"]
        try:
            chunks_created, errors = await backfill_work_graph(storage, work)
            result.chunks_created += chunks_created
            result.works_backfilled += 1
            result.errors.extend(errors)
        except Exception as exc:
            error_msg = f"Failed to backfill work {work_id}: {exc}"
            log.error("backfill_work_failed", work_id=work_id, error=str(exc))
            result.errors.append(error_msg)
            continue

        # Step 5: Entity extraction (optional, requires LLM API)
        if run_entity_extraction and chunks_created > 0:
            try:
                entities = await _run_entity_extraction_for_work(
                    storage, work, settings
                )
                result.entities_extracted += entities
            except Exception as exc:
                error_msg = f"Entity extraction failed for {work_id}: {exc}"
                log.error(
                    "backfill_entity_extraction_failed",
                    work_id=work_id,
                    error=str(exc),
                )
                result.errors.append(error_msg)

    log.info(
        "backfill_complete",
        works_backfilled=result.works_backfilled,
        chunks_created=result.chunks_created,
        entities_extracted=result.entities_extracted,
        errors=len(result.errors),
    )

    return result


async def _run_entity_extraction_for_work(
    storage: StorageManager,
    work: dict[str, Any],
    settings: Settings,
) -> int:
    """Run entity extraction for a backfilled work's chunks.

    Loads chunks from PG, converts them to Chunk model objects, then
    runs the EntityExtractor pipeline.

    Returns the total entity nodes created.
    """
    from author_library.graph.entity_extraction import EntityExtractor

    work_id = work["work_id"]

    # Load chunks from PG
    pg_chunks = await storage.chunks.list_by_work(work_id)
    if not pg_chunks:
        return 0

    # Filter to entity-extraction-eligible granularities
    allowed_grans = {
        g.strip()
        for g in settings.llm.entity_extraction_granularities.split(",")
    }

    # Convert PG rows to Chunk model objects for the extractor
    chunks: list[Chunk] = []
    for pg_chunk in pg_chunks:
        granularity = pg_chunk.get("granularity", "meso")
        if granularity not in allowed_grans:
            continue

        try:
            chunk = Chunk(
                id=str(pg_chunk["id"]),
                text=pg_chunk.get("text", ""),
                annotation=pg_chunk.get("annotation"),
                granularity=ChunkGranularity(granularity),
                work_id=pg_chunk.get("work_id", work_id),
                source_class=pg_chunk.get("source_class", work["source_class"]),
                chapter=pg_chunk.get("chapter"),
                section=pg_chunk.get("section"),
                position=pg_chunk.get("position", 0),
            )
            chunks.append(chunk)
        except Exception as exc:
            log.warning(
                "backfill_chunk_conversion_failed",
                chunk_id=str(pg_chunk.get("id")),
                error=str(exc),
            )

    if not chunks:
        log.info("backfill_no_extraction_chunks", work_id=work_id)
        return 0

    log.info(
        "backfill_entity_extraction_starting",
        work_id=work_id,
        chunks=len(chunks),
    )

    extractor = EntityExtractor(
        storage.neo4j,
        settings.api_keys,
        settings.llm,
    )

    extraction_result = await extractor.extract_and_persist(
        chunks,
        work_title=work["title"],
        author=work["author"],
    )

    log.info(
        "backfill_entity_extraction_complete",
        work_id=work_id,
        nodes_created=extraction_result.nodes_created,
        edges_created=extraction_result.edges_created,
        errors=len(extraction_result.errors),
    )

    return extraction_result.nodes_created


async def check_pg_neo4j_consistency(
    storage: StorageManager,
) -> dict[str, Any]:
    """Check consistency between PG and Neo4j for works and chunks.

    Returns a report dict with:
    - pg_work_count: total works in PG
    - neo4j_work_count: total Work nodes in Neo4j
    - missing_from_neo4j: list of work_ids in PG but not Neo4j
    - extra_in_neo4j: list of work_ids in Neo4j but not PG
    - chunk_counts: per-work chunk count comparison
    """
    # Get PG works
    pg_works = await get_pg_work_ids(storage)
    pg_work_ids = {w["work_id"] for w in pg_works}

    # Get Neo4j works
    neo4j_work_ids = await get_neo4j_work_ids(storage)

    # Get chunk counts per work from PG
    pg_chunk_rows = await storage.pg.fetch_all(
        "SELECT work_id, COUNT(*) AS chunk_count FROM chunks GROUP BY work_id"
    )
    pg_chunk_counts = {r["work_id"]: r["chunk_count"] for r in pg_chunk_rows}

    # Get chunk counts per work from Neo4j
    neo4j_chunk_records = await storage.neo4j.execute_read(
        "MATCH (c:Chunk) RETURN c.work_id AS work_id, COUNT(c) AS chunk_count"
    )
    neo4j_chunk_counts = {r["work_id"]: r["chunk_count"] for r in neo4j_chunk_records}

    # Build per-work comparison
    all_work_ids = pg_work_ids | neo4j_work_ids
    chunk_comparison: list[dict[str, Any]] = []
    for wid in sorted(all_work_ids):
        pg_count = pg_chunk_counts.get(wid, 0)
        neo4j_count = neo4j_chunk_counts.get(wid, 0)
        chunk_comparison.append({
            "work_id": wid,
            "pg_chunks": pg_count,
            "neo4j_chunks": neo4j_count,
            "in_sync": pg_count == neo4j_count,
        })

    return {
        "pg_work_count": len(pg_work_ids),
        "neo4j_work_count": len(neo4j_work_ids),
        "missing_from_neo4j": sorted(pg_work_ids - neo4j_work_ids),
        "extra_in_neo4j": sorted(neo4j_work_ids - pg_work_ids),
        "chunk_counts": chunk_comparison,
        "is_consistent": pg_work_ids == neo4j_work_ids
        and all(c["in_sync"] for c in chunk_comparison),
    }
