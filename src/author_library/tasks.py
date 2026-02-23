"""arq background task definitions for the ingestion pipeline.

Wraps IngestionPipeline stages as arq-compatible async functions that
can be enqueued and processed by the background worker. Each task
receives its dependencies from the arq worker context (injected at
startup via worker.py).

Tasks:
  - task_ingest_book: Full single-work ingestion (all pipeline stages)
  - task_ingest_corpus: Bulk ingestion of multiple works with cross-work analysis

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
