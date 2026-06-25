"""Post-ingestion report generator with self-healing.

Queries live database state after ingestion to produce a verified report
of what was actually stored — not just what the pipeline thinks it stored.
When issues are detected, automatically attempts remediation:
  - Embedding gaps → backfill missing embeddings
  - PG/Neo4j mismatch → sync chunk nodes to Neo4j
  - Missing entity extraction → run extraction for the work

Writes a timestamped report file and logs a summary.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)

_REPORTS_DIR = Path("/home/marty/parlour-backups/ingestion-reports")


async def generate_ingestion_report(
    work_id: str,
    *,
    storage: StorageManager,
    pipeline_result: dict[str, Any] | None = None,
    settings: Settings | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    auto_heal: bool = True,
) -> str:
    """Generate a post-ingestion report by querying live database state.

    Args:
        work_id: The work_id of the just-ingested work.
        storage: Active StorageManager with PG and Neo4j connections.
        pipeline_result: Optional pipeline result dict (from IngestionResult.to_dict())
            to include pipeline-reported values alongside verified counts.
        settings: App settings (needed for auto-heal entity extraction).
        embedding_provider: Embedding provider (needed for auto-heal embedding backfill).
        auto_heal: If True, attempt to fix detected issues automatically.

    Returns:
        The report as a string (also written to disk).
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    pr = pipeline_result or {}
    pipeline_stats = pr.get("post_ingestion_stats", {})

    # --- Query PostgreSQL ---
    work_row = await storage.pg.fetch_one(
        "SELECT work_id, title, author, source_class, publication_year, "
        "created_at FROM works WHERE work_id = $1",
        work_id,
    )
    if not work_row:
        msg = f"Work not found in PG: {work_id}"
        log.warning("ingestion_report_no_work", work_id=work_id)
        return msg

    work = dict(work_row)

    # Chunk counts by granularity
    gran_rows = await storage.pg.fetch_all(
        "SELECT granularity, COUNT(*) as cnt "
        "FROM chunks WHERE work_id = $1 GROUP BY granularity ORDER BY granularity",
        work_id,
    )
    chunks_by_gran = {r["granularity"]: r["cnt"] for r in gran_rows}
    total_chunks = sum(chunks_by_gran.values())

    # Embedding count
    embedded_count = await storage.pg.fetch_val(
        "SELECT COUNT(DISTINCT c.id) FROM chunks c "
        "JOIN chunk_embeddings ce ON ce.chunk_id = c.id "
        "WHERE c.work_id = $1",
        work_id,
    )

    # Word count estimate
    word_count = await storage.pg.fetch_val(
        "SELECT SUM(array_length(regexp_split_to_array(text, '\\s+'), 1)) "
        "FROM chunks WHERE work_id = $1",
        work_id,
    )

    # Section type breakdown
    section_rows = await storage.pg.fetch_all(
        "SELECT COALESCE(metadata->>'section_type', 'body') as stype, COUNT(*) as cnt "
        "FROM chunks WHERE work_id = $1 GROUP BY stype ORDER BY cnt DESC",
        work_id,
    )
    section_breakdown = {r["stype"]: r["cnt"] for r in section_rows}

    # --- Query Neo4j ---
    neo4j_chunks = 0
    entity_summary: dict[str, int] = {}
    total_edges = 0
    cross_work_connections = 0
    entity_nodes: dict[str, int] = {}

    try:
        result = await storage.neo4j.execute_read(
            "MATCH (c:Chunk {work_id: $wid}) RETURN count(c) as cnt",
            {"wid": work_id},
        )
        neo4j_chunks = result[0]["cnt"] if result else 0

        result = await storage.neo4j.execute_read(
            "MATCH (c:Chunk {work_id: $wid})-[r]->(e) "
            "WHERE NOT e:Work AND NOT e:Chunk "
            "RETURN type(r) as rtype, count(r) as cnt "
            "ORDER BY cnt DESC",
            {"wid": work_id},
        )
        for r in result:
            entity_summary[r["rtype"]] = r["cnt"]
        total_edges = sum(entity_summary.values())

        result = await storage.neo4j.execute_read(
            "MATCH (c:Chunk {work_id: $wid})-[]->(e) "
            "WHERE NOT e:Work AND NOT e:Chunk "
            "RETURN labels(e)[0] as label, count(DISTINCT e) as cnt",
            {"wid": work_id},
        )
        entity_nodes = {r["label"]: r["cnt"] for r in result}

        result = await storage.neo4j.execute_read(
            "MATCH (c1:Chunk {work_id: $wid})-[]->(e)<-[]-(c2:Chunk) "
            "WHERE c2.work_id <> $wid "
            "RETURN count(DISTINCT e) as shared",
            {"wid": work_id},
        )
        cross_work_connections = result[0]["shared"] if result else 0

    except Exception as exc:
        log.warning("ingestion_report_neo4j_error", error=str(exc))

    # --- Detect issues ---
    embedding_coverage = round(embedded_count / total_chunks * 100, 1) if total_chunks else 0
    pg_neo4j_sync = "ok" if neo4j_chunks == total_chunks else f"MISMATCH (PG={total_chunks}, Neo4j={neo4j_chunks})"

    issues: list[str] = []
    if embedding_coverage < 100:
        issues.append(f"Embedding gap: {total_chunks - embedded_count} chunks missing embeddings")
    if neo4j_chunks != total_chunks:
        issues.append(f"PG/Neo4j sync: {total_chunks - neo4j_chunks} chunks not in graph")
    if total_edges == 0 and work.get("source_class") in ("primary", "secondary"):
        issues.append("No entity edges — entity extraction may have failed")
    if pr.get("errors"):
        issues.append(f"{len(pr['errors'])} pipeline errors reported")

    # --- Auto-heal ---
    remediation_log: list[str] = []
    if auto_heal and issues:
        remediation_log = await _auto_heal(
            work_id=work_id,
            work=work,
            storage=storage,
            settings=settings,
            embedding_provider=embedding_provider,
            total_chunks=total_chunks,
            embedded_count=embedded_count,
            neo4j_chunks=neo4j_chunks,
            total_edges=total_edges,
        )

        # Re-query after healing to get updated counts
        if remediation_log:
            embedded_count = await storage.pg.fetch_val(
                "SELECT COUNT(DISTINCT c.id) FROM chunks c "
                "JOIN chunk_embeddings ce ON ce.chunk_id = c.id "
                "WHERE c.work_id = $1",
                work_id,
            )
            embedding_coverage = round(embedded_count / total_chunks * 100, 1) if total_chunks else 0

            try:
                result = await storage.neo4j.execute_read(
                    "MATCH (c:Chunk {work_id: $wid}) RETURN count(c) as cnt",
                    {"wid": work_id},
                )
                neo4j_chunks = result[0]["cnt"] if result else 0
                pg_neo4j_sync = "ok" if neo4j_chunks == total_chunks else f"MISMATCH (PG={total_chunks}, Neo4j={neo4j_chunks})"

                result = await storage.neo4j.execute_read(
                    "MATCH (c:Chunk {work_id: $wid})-[r]->(e) "
                    "WHERE NOT e:Work AND NOT e:Chunk "
                    "RETURN type(r) as rtype, count(r) as cnt "
                    "ORDER BY cnt DESC",
                    {"wid": work_id},
                )
                entity_summary = {}
                for r in result:
                    entity_summary[r["rtype"]] = r["cnt"]
                total_edges = sum(entity_summary.values())

                result = await storage.neo4j.execute_read(
                    "MATCH (c:Chunk {work_id: $wid})-[]->(e) "
                    "WHERE NOT e:Work AND NOT e:Chunk "
                    "RETURN labels(e)[0] as label, count(DISTINCT e) as cnt",
                    {"wid": work_id},
                )
                entity_nodes = {r["label"]: r["cnt"] for r in result}

                result = await storage.neo4j.execute_read(
                    "MATCH (c1:Chunk {work_id: $wid})-[]->(e)<-[]-(c2:Chunk) "
                    "WHERE c2.work_id <> $wid "
                    "RETURN count(DISTINCT e) as shared",
                    {"wid": work_id},
                )
                cross_work_connections = result[0]["shared"] if result else 0
            except Exception:
                pass

            # Re-evaluate issues after healing
            issues = []
            if embedding_coverage < 100:
                issues.append(f"Embedding gap: {total_chunks - embedded_count} chunks still missing")
            if neo4j_chunks != total_chunks:
                issues.append(f"PG/Neo4j sync: still {total_chunks - neo4j_chunks} chunks not in graph")
            if total_edges == 0 and work.get("source_class") in ("primary", "secondary"):
                issues.append("No entity edges after remediation")
            if pr.get("errors"):
                issues.append(f"{len(pr['errors'])} pipeline errors reported")

    status = "CLEAN" if not issues else f"{len(issues)} ISSUE(S)"

    # --- Build report ---
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("POST-INGESTION REPORT")
    lines.append("=" * 70)
    lines.append(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Status:    {status}")
    lines.append("")

    lines.append("WORK DETAILS")
    lines.append("-" * 40)
    lines.append(f"  Work ID:      {work_id}")
    lines.append(f"  Title:        {work.get('title', 'Unknown')}")
    lines.append(f"  Author:       {work.get('author', 'Unknown')}")
    lines.append(f"  Source Class:  {work.get('source_class', 'Unknown')}")
    lines.append(f"  Year:         {work.get('publication_year') or 'Unknown'}")
    lines.append(f"  Route:        {pr.get('processing_route', 'Unknown')}")
    lines.append("")

    lines.append("CHUNKS (PostgreSQL)")
    lines.append("-" * 40)
    lines.append(f"  Total:        {total_chunks:,}")
    lines.append(f"  Words:        {word_count:,}" if word_count else "  Words:        Unknown")
    for gran, cnt in sorted(chunks_by_gran.items()):
        lines.append(f"    {gran:12s}  {cnt:,}")
    if section_breakdown:
        lines.append("  By section type:")
        for stype, cnt in section_breakdown.items():
            lines.append(f"    {stype:20s}  {cnt:,}")
    lines.append("")

    lines.append("EMBEDDINGS")
    lines.append("-" * 40)
    lines.append(f"  Stored:       {embedded_count:,} / {total_chunks:,}")
    lines.append(f"  Coverage:     {embedding_coverage}%")
    if pipeline_stats.get("embeddings_stored") is not None:
        lines.append(f"  (pipeline reported: {pipeline_stats['embeddings_stored']:,})")
    lines.append("")

    lines.append("KNOWLEDGE GRAPH (Neo4j)")
    lines.append("-" * 40)
    lines.append(f"  Chunk nodes:  {neo4j_chunks:,}")
    lines.append(f"  PG/Neo4j:     {pg_neo4j_sync}")
    lines.append(f"  Entity edges: {total_edges:,}")
    if entity_summary:
        for rtype, cnt in sorted(entity_summary.items(), key=lambda x: -x[1]):
            lines.append(f"    {rtype:30s}  {cnt:,}")
    if entity_nodes:
        lines.append(f"  Unique entities: {sum(entity_nodes.values()):,}")
        for label, cnt in sorted(entity_nodes.items(), key=lambda x: -x[1]):
            lines.append(f"    {label:20s}  {cnt:,}")
    lines.append(f"  Cross-work:   {cross_work_connections:,} shared entities")
    if pipeline_stats.get("entity_count") is not None:
        lines.append(f"  (pipeline reported: {pipeline_stats['entity_count']:,} nodes, {pipeline_stats.get('edge_count', 0):,} edges)")
    lines.append("")

    if remediation_log:
        lines.append("AUTO-HEAL ACTIONS")
        lines.append("-" * 40)
        for action in remediation_log:
            lines.append(f"  > {action}")
        lines.append("")

    if issues:
        lines.append("REMAINING ISSUES")
        lines.append("-" * 40)
        for issue in issues:
            lines.append(f"  ! {issue}")
        lines.append("")

    if pr.get("errors"):
        lines.append("PIPELINE ERRORS")
        lines.append("-" * 40)
        for err in pr["errors"][:20]:
            lines.append(f"  - {err}")
        if len(pr["errors"]) > 20:
            lines.append(f"  ... and {len(pr['errors']) - 20} more")
        lines.append("")

    if pr.get("quality_checks"):
        qc = pr["quality_checks"]
        lines.append("QUALITY CHECKS")
        lines.append("-" * 40)
        lines.append(f"  Status:             {qc.get('status', 'unknown')}")
        lines.append(f"  Orphans cleaned:    {qc.get('orphans_cleaned', 0)}")
        lines.append(f"  Noise chunks:       {qc.get('noise_chunks', 0)}")
        lines.append(f"  Embedding coverage: {qc.get('embedding_coverage_pct', 0)}%")
        lines.append(f"  Entity coverage:    {qc.get('entity_coverage_pct', 0)}%")
        if qc.get("classification_warning"):
            lines.append(f"  Classification:     {qc['classification_warning']}")
        lines.append("")

    lines.append("=" * 70)

    report = "\n".join(lines)

    # --- Write to disk ---
    try:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        report_file = _REPORTS_DIR / f"{timestamp}--{work_id}.txt"
        report_file.write_text(report)
        log.info(
            "ingestion_report_written",
            path=str(report_file),
            work_id=work_id,
            status=status,
        )
    except Exception as exc:
        log.warning("ingestion_report_write_failed", error=str(exc))

    # --- Log summary ---
    log.info(
        "ingestion_report_summary",
        work_id=work_id,
        title=work.get("title"),
        source_class=work.get("source_class"),
        chunks=total_chunks,
        embeddings=embedded_count,
        embedding_coverage_pct=embedding_coverage,
        neo4j_chunks=neo4j_chunks,
        entity_edges=total_edges,
        cross_work=cross_work_connections,
        issues=len(issues),
        remediations=len(remediation_log),
        status=status,
    )

    return report


async def _auto_heal(
    *,
    work_id: str,
    work: dict[str, Any],
    storage: StorageManager,
    settings: Settings | None,
    embedding_provider: EmbeddingProvider | None,
    total_chunks: int,
    embedded_count: int,
    neo4j_chunks: int,
    total_edges: int,
) -> list[str]:
    """Attempt automatic remediation of detected issues.

    Returns a list of human-readable action descriptions.
    """
    actions: list[str] = []

    # --- Heal 1: Embedding gaps ---
    if embedded_count < total_chunks and embedding_provider is not None:
        missing = total_chunks - embedded_count
        log.info("auto_heal_embeddings_starting", work_id=work_id, missing=missing)
        try:
            filled = await _backfill_embeddings(
                work_id, storage=storage, provider=embedding_provider, settings=settings
            )
            actions.append(f"Backfilled {filled} missing embeddings")
            log.info("auto_heal_embeddings_complete", work_id=work_id, filled=filled)
        except Exception as exc:
            actions.append(f"Embedding backfill FAILED: {exc}")
            log.error("auto_heal_embeddings_failed", work_id=work_id, error=str(exc))

    # --- Heal 2: PG/Neo4j chunk sync ---
    if neo4j_chunks != total_chunks:
        log.info("auto_heal_graph_sync_starting", work_id=work_id,
                 pg=total_chunks, neo4j=neo4j_chunks)
        try:
            from author_library.graph.backfill import backfill_work_graph
            synced, sync_errors = await backfill_work_graph(storage, work)
            actions.append(f"Synced {synced} chunk nodes to Neo4j")
            if sync_errors:
                actions.append(f"  ({len(sync_errors)} sync errors)")
            log.info("auto_heal_graph_sync_complete", work_id=work_id, synced=synced)
        except Exception as exc:
            actions.append(f"Graph sync FAILED: {exc}")
            log.error("auto_heal_graph_sync_failed", work_id=work_id, error=str(exc))

    # --- Heal 3: Missing entity extraction ---
    if (
        total_edges == 0
        and work.get("source_class") in ("primary", "secondary")
        and settings is not None
    ):
        log.info("auto_heal_entity_extraction_starting", work_id=work_id)
        try:
            from author_library.graph.backfill import _run_entity_extraction_for_work
            nodes_created = await _run_entity_extraction_for_work(storage, work, settings)
            actions.append(f"Ran entity extraction: {nodes_created} nodes created")
            log.info("auto_heal_entity_extraction_complete",
                     work_id=work_id, nodes_created=nodes_created)
        except Exception as exc:
            actions.append(f"Entity extraction FAILED: {exc}")
            log.error("auto_heal_entity_extraction_failed",
                       work_id=work_id, error=str(exc))

    return actions


async def _backfill_embeddings(
    work_id: str,
    *,
    storage: StorageManager,
    provider: EmbeddingProvider,
    settings: Settings | None,
) -> int:
    """Backfill missing embeddings for a work.

    Queries for chunks without embeddings, embeds them in batches,
    and stores the results. Returns the count of embeddings created.
    """
    from author_library.embeddings.base import build_token_aware_batches

    rows = await storage.pg.fetch_all(
        """
        SELECT c.id,
               CASE WHEN c.annotation IS NOT NULL AND c.annotation != ''
                    THEN c.annotation || E'\\n\\n' || c.text
                    ELSE c.text
               END AS embed_text
        FROM chunks c
        LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id
        WHERE c.work_id = $1 AND ce.id IS NULL
        ORDER BY c.position
        """,
        work_id,
    )

    if not rows:
        return 0

    texts = [r["embed_text"] for r in rows]
    chunk_ids = [r["id"] for r in rows]
    batches = build_token_aware_batches(texts)

    total_embedded = 0
    offset = 0
    for batch_texts in batches:
        batch_ids = chunk_ids[offset : offset + len(batch_texts)]
        offset += len(batch_texts)

        result = await provider.embed_batch(batch_texts)

        for cid, vec in zip(batch_ids, result.vectors):
            await storage.embeddings.store(
                cid,
                vec,
                provider.provider_name,
                provider.model_name,
                provider.dimensions,
            )
            total_embedded += 1

    return total_embedded
