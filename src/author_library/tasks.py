"""arq background task definitions for the ingestion pipeline.

Wraps IngestionPipeline stages as arq-compatible async functions that
can be enqueued and processed by the background worker. Each task
receives its dependencies from the arq worker context (injected at
startup via worker.py).

Tasks:
  - task_ingest_book: Full single-work ingestion (all pipeline stages)
  - task_ingest_corpus: Bulk ingestion of multiple works with cross-work analysis
  - task_process_capture: Chrome extension capture event processing
  - task_surface_connections: Post-ingestion connection scanning + PR content generation
  - task_quality_gate: Post-ingestion async quality checks (theme dedup, consistency, linking, coverage)

The existing synchronous pipeline (IngestionPipeline.ingest) remains
available for direct invocation when immediate results are needed
(e.g., ingest_book with auto_confirm: true).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from author_library.errors import IngestionError
from author_library.tools.ingestion_pipeline import IngestionPipeline

log = structlog.get_logger(__name__)


async def task_ingest_book(
    ctx: dict[str, Any],
    *,
    file_path: str,
    subject_author_id: str,
    metadata_hints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """arq task: ingest a single work through the full pipeline.

    This is the background equivalent of handle_ingest_book. The arq
    worker context provides settings, storage, and embedding_provider
    (initialized in worker.startup).

    Args:
        ctx: arq worker context with 'settings', 'storage', 'embedding_provider'.
        file_path: Path to the document file.
        subject_author_id: The subject author's slug identifier.
        metadata_hints: Optional classification overrides.

    Returns:
        dict with ingestion result data (serializable for arq result storage).
    """
    settings = ctx["settings"]
    storage = ctx["storage"]
    embedding_provider = ctx["embedding_provider"]

    path = Path(file_path)
    if not path.exists():
        raise IngestionError(
            f"File not found: {file_path}",
            context={"file_path": file_path},
        )

    log.info(
        "task_ingest_book_starting",
        file_path=file_path,
        subject_author=subject_author_id,
    )

    pipeline = IngestionPipeline(
        settings=settings,
        storage=storage,
        embedding_provider=embedding_provider,
    )

    result = await pipeline.ingest(
        path,
        subject_author_id=subject_author_id,
        metadata_hints=metadata_hints or {},
    )

    result_dict = result.to_dict()
    log.info(
        "task_ingest_book_complete",
        work_id=result.work_id,
        chunks=sum(result.chunks_by_granularity.values()),
        embeddings=result.embeddings_stored,
        errors=len(result.errors),
    )

    return result_dict


async def task_ingest_corpus(
    ctx: dict[str, Any],
    *,
    file_paths: list[str],
    subject_author_id: str,
    metadata_hints: dict[str, Any] | None = None,
    run_cross_work_analysis: bool = True,
) -> dict[str, Any]:
    """arq task: bulk ingest multiple works with optional cross-work analysis.

    Processes each file through the full pipeline, then optionally runs
    thematic index generation, voice profile extraction, and thematic
    evolution analysis across the corpus.

    Args:
        ctx: arq worker context.
        file_paths: List of file path strings to ingest.
        subject_author_id: The subject author's slug identifier.
        metadata_hints: Optional shared metadata hints.
        run_cross_work_analysis: Whether to run post-ingestion analysis.

    Returns:
        dict with corpus-level ingestion summary.
    """
    settings = ctx["settings"]
    storage = ctx["storage"]
    embedding_provider = ctx["embedding_provider"]

    log.info(
        "task_ingest_corpus_starting",
        file_count=len(file_paths),
        subject_author=subject_author_id,
    )

    pipeline = IngestionPipeline(
        settings=settings,
        storage=storage,
        embedding_provider=embedding_provider,
    )

    results = []
    per_work_errors: dict[str, list[str]] = {}

    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            per_work_errors[fp] = [f"File not found: {fp}"]
            continue

        try:
            result = await pipeline.ingest(
                path,
                subject_author_id=subject_author_id,
                metadata_hints=metadata_hints or {},
            )
            results.append(result)
            if result.errors:
                per_work_errors[result.work_id] = result.errors
        except Exception as exc:
            error_msg = f"Failed to ingest {path.name}: {exc}"
            log.error("task_corpus_work_failed", file=str(path), error=str(exc))
            per_work_errors[str(path)] = [error_msg]

    # Cross-work analysis
    cross_work_summary: dict[str, Any] = {}
    primary_work_ids = [r.work_id for r in results if r.source_class == "primary"]

    if run_cross_work_analysis and primary_work_ids:
        from author_library.tools.ingest import _run_cross_work_analysis

        cross_work_summary = await _run_cross_work_analysis(
            subject_author_id=subject_author_id,
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
        )

    # Build summary
    total_chunks = sum(sum(r.chunks_by_granularity.values()) for r in results)
    total_embeddings = sum(r.embeddings_stored for r in results)
    total_entities = sum(r.entity_count for r in results)
    total_edges = sum(r.edge_count for r in results)

    source_class_breakdown: dict[str, int] = {}
    for r in results:
        source_class_breakdown[r.source_class] = (
            source_class_breakdown.get(r.source_class, 0) + 1
        )

    corpus_summary = {
        "works_processed": len(results),
        "works_failed": len(per_work_errors),
        "total_chunks": total_chunks,
        "total_embeddings": total_embeddings,
        "total_entities": total_entities,
        "total_edges": total_edges,
        "source_class_breakdown": source_class_breakdown,
        "per_work_results": [r.to_dict() for r in results],
        "cross_work_analysis": cross_work_summary,
        "errors": per_work_errors,
    }

    log.info(
        "task_ingest_corpus_complete",
        works_processed=len(results),
        works_failed=len(per_work_errors),
        total_chunks=total_chunks,
    )

    return corpus_summary


async def task_process_capture(
    ctx: dict[str, Any],
    *,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """arq task: process a capture event from the Chrome extension.

    Deserializes the capture payload, delegates to the capture processor,
    and returns the result as a serializable dict.

    Args:
        ctx: arq worker context with 'settings', 'storage', 'embedding_provider'.
        payload: Serialized CapturePayload dict.

    Returns:
        dict with capture result data.
    """
    settings = ctx["settings"]
    storage = ctx["storage"]
    embedding_provider = ctx["embedding_provider"]

    from author_library.captures.models import CapturePayload
    from author_library.captures.processor import process_capture

    capture_payload = CapturePayload(**payload)

    log.info(
        "task_process_capture_starting",
        source_url=capture_payload.source_url,
        mode=capture_payload.mode.value,
        timestamp=capture_payload.timestamp_seconds,
    )

    result = await process_capture(
        capture_payload,
        settings=settings,
        storage=storage,
        embedding_provider=embedding_provider,
    )

    result_dict = result.to_dict()
    log.info(
        "task_process_capture_complete",
        capture_id=result.capture_id,
        chunk_id=result.chunk_id,
        errors=len(result.errors),
    )

    return result_dict


async def task_surface_connections(
    ctx: dict[str, Any],
    *,
    work_id: str,
    work_title: str = "",
    work_author: str = "",
    confidence_threshold: float = 0.4,
    min_connections_for_pr: int = 1,
) -> dict[str, Any]:
    """arq task: scan for new connections after ingestion and generate PR content.

    Runs BatchSurfacer.surface_after_ingestion() which:
    1. Scans the newly ingested work's chunks for cross-work connections
    2. Filters out already-linked pairs
    3. Groups connections by confidence and target work
    4. Generates PR content if enough connections are found

    This task is enqueued automatically after passage link detection
    completes during ingestion. PR content is generated but actual PR
    creation (via GitHub API) is delegated to the vault sync layer.

    Args:
        ctx: arq worker context with 'settings', 'storage', 'embedding_provider'.
        work_id: The newly ingested work to scan connections for.
        work_title: Title for PR readability.
        work_author: Author for PR readability.
        confidence_threshold: Minimum confidence to include (default 0.4).
        min_connections_for_pr: Skip PR if fewer connections found (default 1).

    Returns:
        dict with scan results and PR content (serializable for arq result storage).
    """
    settings = ctx["settings"]
    storage = ctx["storage"]
    embedding_provider = ctx["embedding_provider"]

    log.info(
        "task_surface_connections_starting",
        work_id=work_id,
        work_title=work_title,
    )

    from author_library.surfacing.batch_surfacing import BatchSurfacer

    surfacer = BatchSurfacer(
        settings=settings,
        storage=storage,
        embedding_provider=embedding_provider,
    )

    result = await surfacer.surface_after_ingestion(
        work_id,
        work_title=work_title,
        work_author=work_author,
        confidence_threshold=confidence_threshold,
        min_connections_for_pr=min_connections_for_pr,
    )

    result_dict = result.to_dict()
    log.info(
        "task_surface_connections_complete",
        work_id=work_id,
        total_connections=result.scan_result.total_found if result.scan_result else 0,
        pr_content_generated=result.pr_content is not None,
        errors=len(result.errors),
    )

    return result_dict


async def task_quality_gate(
    ctx: dict[str, Any],
    *,
    work_id: str,
    author_id: str,
) -> dict[str, Any]:
    """arq task: run expensive post-ingestion quality checks asynchronously.

    Runs after ingestion completes — enqueued automatically by handle_ingest_book.
    Performs checks too slow for inline execution:
    1. Theme deduplication (embedding-based cosine similarity clustering)
    2. PG-Neo4j consistency verification + structural backfill
    3. Cross-work passage re-linking for the newly ingested work
    4. Entity extraction coverage audit across all works

    Args:
        ctx: arq worker context with 'settings', 'storage', 'embedding_provider'.
        work_id: The newly ingested work that triggered this gate.
        author_id: The subject author for cross-work analysis.

    Returns:
        dict with per-check results and overall status.
    """
    settings = ctx["settings"]
    storage = ctx["storage"]
    embedding_provider = ctx["embedding_provider"]

    log.info("task_quality_gate_starting", work_id=work_id, author_id=author_id)

    result: dict[str, Any] = {}

    # Check 1: Theme deduplication
    try:
        from author_library.graph.theme_dedup import deduplicate_themes

        dedup = await deduplicate_themes(storage.neo4j, embedding_provider)
        result["theme_dedup"] = {
            "original": dedup.original_count,
            "canonical": dedup.canonical_count,
            "merged": dedup.merged_count,
        }
        if dedup.merged_count > 0:
            try:
                from author_library.intelligence.lesson_writer import record_lesson
                await record_lesson(
                    storage,
                    problem_type="theme_explosion",
                    detection_method="qg2_async",
                    trigger_context={"work_id": work_id, "author_id": author_id},
                    problem_description=(
                        f"Theme deduplication merged {dedup.merged_count} near-duplicate "
                        f"themes (from {dedup.original_count} to {dedup.canonical_count}) "
                        f"after ingesting work '{work_id}'."
                    ),
                    fix_applied=(
                        f"Merged {dedup.merged_count} themes via cosine similarity clustering."
                    ),
                    prevention_rule=(
                        "LLM theme extraction tends to produce near-duplicate themes "
                        "(e.g. 'faith' vs 'faithful trust'). Consider using canonical "
                        "theme matching before insertion."
                    ),
                    prevention_step="entity_extraction",
                )
            except Exception as _lesson_exc:
                log.warning("qg2_lesson_write_failed", problem_type="theme_explosion", error=str(_lesson_exc))
    except Exception as exc:
        log.error("quality_gate_theme_dedup_failed", error=str(exc))
        result["theme_dedup"] = {"error": str(exc)}

    # Check 2: PG-Neo4j consistency
    try:
        from author_library.graph.backfill import (
            backfill_missing_graph_data,
            check_pg_neo4j_consistency,
        )

        report = await check_pg_neo4j_consistency(storage)
        backfilled = 0
        missing = report.get("missing_from_neo4j", [])
        if missing:
            bf_result = await backfill_missing_graph_data(
                storage, embedding_provider, settings, run_entity_extraction=False
            )
            backfilled = bf_result.works_backfilled
            try:
                from author_library.intelligence.lesson_writer import record_lesson
                await record_lesson(
                    storage,
                    problem_type="pg_neo4j_desync",
                    detection_method="qg2_async",
                    trigger_context={"work_id": work_id, "author_id": author_id},
                    problem_description=(
                        f"{len(missing)} works in PG are missing from Neo4j "
                        f"after ingesting work '{work_id}': {missing[:3]}"
                        + (" ..." if len(missing) > 3 else "")
                    ),
                    fix_applied=(
                        f"Structural backfill ran and created Neo4j nodes for "
                        f"{backfilled} missing works."
                    ),
                    prevention_rule=(
                        "PG and Neo4j should stay in sync; missing Neo4j nodes "
                        "indicate a failed graph write during prior ingestion."
                    ),
                    prevention_step="graph_storage",
                )
            except Exception as _lesson_exc:
                log.warning("qg2_lesson_write_failed", problem_type="pg_neo4j_desync", error=str(_lesson_exc))
        result["pg_neo4j_consistency"] = {
            "is_consistent": report["is_consistent"],
            "backfilled": backfilled,
        }
    except Exception as exc:
        log.error("quality_gate_consistency_failed", error=str(exc))
        result["pg_neo4j_consistency"] = {"error": str(exc)}

    # Check 3: Cross-work passage re-linking
    try:
        from author_library.graph.linking_explicit import ExplicitLinkDetector
        from author_library.graph.linking_implicit import ImplicitEngagementDetector
        from author_library.graph.linking_thematic import ThematicParallelDetector

        new_edges = 0
        for DetectorClass in (ExplicitLinkDetector, ImplicitEngagementDetector, ThematicParallelDetector):
            try:
                detector = DetectorClass(storage.neo4j, embedding_provider)
                link_result = await detector.detect_and_link(work_id)
                new_edges += link_result.edges_created
            except Exception as det_exc:
                log.warning(
                    "quality_gate_linking_detector_failed",
                    detector=DetectorClass.__name__,
                    error=str(det_exc),
                )
        result["cross_work_links"] = {"new_edges": new_edges}
    except Exception as exc:
        log.error("quality_gate_linking_failed", error=str(exc))
        result["cross_work_links"] = {"error": str(exc)}

    # Check 4: Entity extraction coverage audit
    try:
        works = await storage.pg.fetch_all("SELECT work_id FROM works")
        below_threshold: list[str] = []
        for w in works:
            wid = w["work_id"]
            total = await storage.neo4j.driver.execute_query(
                "MATCH (c:Chunk {work_id: $wid}) RETURN count(c) as c", wid=wid
            )
            with_entities = await storage.neo4j.driver.execute_query(
                "MATCH (c:Chunk {work_id: $wid})-[]->() RETURN count(DISTINCT c) as c", wid=wid
            )
            total_count = total.records[0]["c"]
            entity_count = with_entities.records[0]["c"]
            if total_count > 0:
                coverage = entity_count / total_count
                if coverage < 0.9:
                    below_threshold.append(f"{wid} ({coverage:.0%})")
        result["entity_coverage"] = {
            "works_audited": len(works),
            "below_threshold": below_threshold,
        }
    except Exception as exc:
        log.error("quality_gate_coverage_failed", error=str(exc))
        result["entity_coverage"] = {"error": str(exc)}

    # Overall status
    has_errors = any("error" in v for v in result.values() if isinstance(v, dict))
    result["status"] = "errors" if has_errors else "pass"

    log.info("task_quality_gate_complete", work_id=work_id, status=result["status"])
    return result
