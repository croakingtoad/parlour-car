"""MCP tool handlers for ingestion (E010).

Provides:
  - ingest_book: Ingest a single work through the full pipeline.
  - ingest_corpus: Bulk ingest multiple works and run cross-work analysis.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from author_library.errors import IngestionError
from author_library.queue import TaskQueue
from author_library.tools.ingestion_pipeline import IngestionPipeline, IngestionResult

if TYPE_CHECKING:
    from author_library.cache import CacheManager
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


async def handle_ingest_book(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
    cache_manager: CacheManager | None = None,
    task_queue: TaskQueue | None = None,
) -> str:
    """Handle the ingest_book MCP tool call.

    Arguments:
        file_path (str): Path to the document file.
        subject_author_id (str): The subject author's slug identifier.
        metadata_hints (dict, optional): Overrides for classification and catalog fields.
        auto_confirm (bool, optional): When True (default), runs the full pipeline.
            When False, pauses after classification and returns the suggested
            source class for human review.

    Returns:
        JSON string with ingestion summary (or classification preview if
        auto_confirm is False).
    """
    file_path = arguments.get("file_path")
    if not file_path:
        raise IngestionError(
            "file_path is required",
            context={"arguments": arguments},
        )

    subject_author_id = arguments.get("subject_author_id")
    if not subject_author_id:
        raise IngestionError(
            "subject_author_id is required",
            context={"arguments": arguments},
        )

    path = Path(file_path)
    if not path.exists():
        raise IngestionError(
            f"File not found: {file_path}",
            context={"file_path": file_path},
        )

    metadata_hints = arguments.get("metadata_hints") or {}
    auto_confirm = arguments.get("auto_confirm", True)

    # When auto_confirm is False, only run classification and return for review
    if not auto_confirm:
        return await _classify_only(
            path,
            subject_author=subject_author_id,
            metadata_hints=metadata_hints,
            settings=settings,
            storage=storage,
        )

    pipeline = IngestionPipeline(
        settings=settings,
        storage=storage,
        embedding_provider=embedding_provider,
    )

    result = await pipeline.ingest(
        path,
        subject_author_id=subject_author_id,
        metadata_hints=metadata_hints,
    )

    # After successful ingestion, run cross-work analysis for primary sources.
    # This triggers voice profile extraction, thematic index generation, and
    # thematic evolution analysis — previously only called by ingest_corpus.
    cross_work_summary: dict[str, Any] = {}
    if result.source_class == "primary":
        cross_work_summary = await _run_cross_work_analysis(
            subject_author_id=subject_author_id,
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
        )

    # Invalidate caches — new content may affect query/graph/voice/thematic results
    if cache_manager is not None:
        await cache_manager.invalidate_on_ingestion(author_id=subject_author_id)

    # Enqueue async quality gate (theme dedup, consistency, cross-work linking)
    if task_queue is not None:
        try:
            qg_job = await task_queue.enqueue_quality_gate(
                work_id=result.work_id,
                author_id=subject_author_id,
            )
            if qg_job:
                log.info("quality_gate_enqueued", job_id=qg_job, work_id=result.work_id)
        except Exception as exc:
            log.warning("quality_gate_enqueue_failed", error=str(exc))

    # Fire-and-forget production backup after successful ingestion
    await _run_post_ingest_backup(result.work_id)

    response = result.to_dict()
    if cross_work_summary:
        response["cross_work_analysis"] = cross_work_summary

    return json.dumps(response, indent=2)


async def _classify_only(
    path: Path,
    *,
    subject_author: str,
    metadata_hints: dict[str, Any],
    settings: Settings,
    storage: StorageManager,
) -> str:
    """Run only the classification step and return a preview for human review.

    The user can then proceed with the composable tools (catalog_source,
    chunk_source, etc.) using the suggested classification or an override.
    """
    from author_library.catalog.classifier import SourceClassifier
    from author_library.catalog.mixed_authorship import MixedAuthorshipAnalyzer
    from author_library.parsing import parse_document

    document = await parse_document(path, metadata_hints=metadata_hints)

    classifier = SourceClassifier(settings)
    classification = await classifier.classify(
        document,
        subject_author=subject_author,
        metadata_hints=metadata_hints,
    )

    requires_human_judgment = classification.confidence < 0.8

    result: dict[str, Any] = {
        "status": "awaiting_confirmation",
        "file_path": str(path),
        "subject_author": subject_author,
        "suggested_class": classification.source_class.value,
        "confidence": classification.confidence,
        "signals": classification.signals_detected,
        "requires_human_judgment": requires_human_judgment,
        "next_steps": [
            f"Review the suggested source class: {classification.source_class.value}",
            "Use catalog_source to confirm (or override) the classification and catalog the work",
            "Then use chunk_source, detect_passage_links, and flag_acquisition to complete ingestion",
        ],
    }

    # Check for mixed authorship
    if classification.source_class.value in ("primary", "contextual"):
        try:
            analyzer = MixedAuthorshipAnalyzer(subject_author)
            mixed_result = analyzer.analyze(
                document,
                document_source_class=classification.source_class.value,
            )
            if mixed_result.is_mixed:
                requires_human_judgment = True
                result["requires_human_judgment"] = True
                result["mixed_authorship"] = {
                    "is_mixed": True,
                    "segments": len(mixed_result.segments),
                    "analysis_notes": mixed_result.analysis_notes,
                }
        except Exception:
            log.warning("classify_only_mixed_authorship_failed", path=str(path), exc_info=True)

    return json.dumps(result, indent=2)


async def handle_ingest_corpus(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
    cache_manager: CacheManager | None = None,
) -> str:
    """Handle the ingest_corpus MCP tool call.

    Arguments:
        directory (str, optional): Directory containing documents to ingest.
        file_list (list[str], optional): Explicit list of file paths to ingest.
        subject_author_id (str): The subject author's slug identifier.
        metadata_hints (dict, optional): Shared metadata hints applied to all works.

    Returns:
        JSON string with corpus-level summary.
    """
    subject_author_id = arguments.get("subject_author_id")
    if not subject_author_id:
        raise IngestionError(
            "subject_author_id is required",
            context={"arguments": arguments},
        )

    # Gather file paths from directory or explicit list
    file_paths: list[Path] = []
    directory = arguments.get("directory")
    file_list = arguments.get("file_list")

    if directory:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise IngestionError(
                f"Directory not found: {directory}",
                context={"directory": directory},
            )
        supported_extensions = {".epub", ".pdf", ".txt", ".html", ".docx"}
        for item in sorted(dir_path.iterdir()):
            if item.is_file() and item.suffix.lower() in supported_extensions:
                file_paths.append(item)
    elif file_list:
        for fp in file_list:
            p = Path(fp)
            if p.exists():
                file_paths.append(p)
            else:
                log.warning("corpus_file_not_found", path=str(fp))
    else:
        raise IngestionError(
            "Either directory or file_list is required",
            context={"arguments": arguments},
        )

    if not file_paths:
        raise IngestionError(
            "No valid files found for ingestion",
            context={"directory": directory, "file_list": file_list},
        )

    metadata_hints = arguments.get("metadata_hints") or {}

    pipeline = IngestionPipeline(
        settings=settings,
        storage=storage,
        embedding_provider=embedding_provider,
    )

    # Ingest each work
    results: list[IngestionResult] = []
    per_work_errors: dict[str, list[str]] = {}

    for path in file_paths:
        log.info("corpus_ingesting_work", file=str(path))
        try:
            result = await pipeline.ingest(
                path,
                subject_author_id=subject_author_id,
                metadata_hints=metadata_hints,
            )
            results.append(result)
            if result.errors:
                per_work_errors[result.work_id] = result.errors
        except Exception as exc:
            error_msg = f"Failed to ingest {path.name}: {exc}"
            log.error("corpus_work_failed", file=str(path), error=str(exc))
            per_work_errors[str(path)] = [error_msg]

    # Cross-work analysis (only if we have primary works)
    primary_work_ids = [r.work_id for r in results if r.source_class == "primary"]
    cross_work_summary: dict[str, Any] = {}

    if primary_work_ids:
        cross_work_summary = await _run_cross_work_analysis(
            subject_author_id=subject_author_id,
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
        )

    # Build corpus summary
    total_chunks = sum(
        sum(r.chunks_by_granularity.values()) for r in results
    )
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

    # Invalidate caches — new content affects query/graph/voice/thematic results
    if cache_manager is not None:
        await cache_manager.invalidate_on_ingestion(author_id=subject_author_id)

    # Backup after corpus ingestion
    await _run_post_ingest_backup(f"corpus-{subject_author_id}")

    return json.dumps(corpus_summary, indent=2)


async def _run_cross_work_analysis(
    *,
    subject_author_id: str,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
) -> dict[str, Any]:
    """Run cross-work analysis after corpus ingestion.

    Includes: thematic index generation, voice profile extraction,
    and thematic evolution analysis.
    """
    from author_library.intelligence.evolution import ThematicEvolutionAnalyzer
    from author_library.intelligence.thematic_index import ThematicIndexGenerator
    from author_library.intelligence.voice_crud import VoiceProfileManager
    from author_library.intelligence.voice_profile import VoiceProfileExtractor

    summary: dict[str, Any] = {}

    # 1. Thematic index generation
    try:
        generator = ThematicIndexGenerator(settings, storage=storage)
        themes = await generator.generate(
            author_id=subject_author_id,
            author_name=subject_author_id,
            work_repo=storage.works,
            chunk_repo=storage.chunks,
            thematic_repo=storage.thematic,
        )
        summary["thematic_index"] = {
            "themes_identified": len(themes),
            "theme_names": [t.theme for t in themes],
        }
    except Exception as exc:
        log.error("cross_work_thematic_failed", error=str(exc))
        summary["thematic_index"] = {"error": str(exc)}

    # 2. Voice profile extraction (with staleness check to avoid redundant LLM calls)
    try:
        voice_skipped = False
        extractor = VoiceProfileExtractor(settings, storage=storage)

        # Check if re-extraction is actually needed.  We store a
        # ``_primary_chunk_count`` marker inside the profile JSONB so we
        # can compare the corpus size at last extraction vs now.
        current_eligible_chunks = await extractor._gather_eligible_chunks(
            author_id=subject_author_id,
            work_repo=storage.works,
            chunk_repo=storage.chunks,
        )
        current_chunk_count = len(current_eligible_chunks)

        existing_profile_row = await storage.voice_profiles.get_current(subject_author_id)
        if existing_profile_row is not None:
            profile_data = existing_profile_row.get("profile", {})
            if isinstance(profile_data, str):
                import json as _json
                profile_data = _json.loads(profile_data)
            prev_chunk_count = profile_data.get("_primary_chunk_count", 0)

            if prev_chunk_count > 0 and current_chunk_count > 0:
                increase_ratio = (current_chunk_count - prev_chunk_count) / prev_chunk_count
                if increase_ratio < 0.10:
                    voice_skipped = True
                    log.info(
                        "voice_profile_skip_not_stale",
                        author_id=subject_author_id,
                        prev_chunks=prev_chunk_count,
                        current_chunks=current_chunk_count,
                        increase_pct=round(increase_ratio * 100, 1),
                    )
                    summary["voice_profile"] = {
                        "skipped": True,
                        "reason": "profile not stale",
                        "prev_chunks": prev_chunk_count,
                        "current_chunks": current_chunk_count,
                    }

        if not voice_skipped:
            profile = await extractor.extract(
                author_id=subject_author_id,
                author_name=subject_author_id,
                work_repo=storage.works,
                chunk_repo=storage.chunks,
            )
            manager = VoiceProfileManager(settings)
            # Persist the chunk count used for this extraction so future
            # runs can compare and decide whether re-extraction is needed.
            await manager.store_profile(
                profile=profile,
                voice_repo=storage.voice_profiles,
                extra_metadata={"_primary_chunk_count": current_chunk_count},
            )
            summary["voice_profile"] = {
                "confidence": profile.confidence,
                "register": profile.register,
                "characteristic_phrases": len(profile.characteristic_phrases),
            }
    except Exception as exc:
        log.error("cross_work_voice_failed", error=str(exc))
        summary["voice_profile"] = {"error": str(exc)}

    # 3. Thematic evolution analysis
    try:
        themes_data = summary.get("thematic_index", {})
        if "themes_identified" in themes_data and themes_data["themes_identified"] > 0:
            # Re-fetch themes from the repository for the analyzer
            entries = await storage.thematic.list_entries(subject_author_id)
            from author_library.intelligence.thematic_index import ThematicEntry

            theme_entries = [
                ThematicEntry(
                    theme=e.get("theme", ""),
                    author_stance=e.get("author_stance", ""),
                    related_themes=e.get("related_themes", []),
                )
                for e in entries
            ]

            analyzer = ThematicEvolutionAnalyzer(settings)
            evolutions = await analyzer.analyze(
                author_id=subject_author_id,
                themes=theme_entries,
                work_repo=storage.works,
                chunk_repo=storage.chunks,
                graph_repo=storage.graph,
            )
            summary["thematic_evolution"] = {
                "themes_analyzed": len(evolutions),
                "total_develops_from_edges": sum(
                    len(e.develops_from_edges) for e in evolutions
                ),
            }
    except Exception as exc:
        log.error("cross_work_evolution_failed", error=str(exc))
        summary["thematic_evolution"] = {"error": str(exc)}

    return summary


_BACKUP_SCRIPT = Path("/home/marty/parlour-backups/backup.sh")


async def _run_post_ingest_backup(work_id: str) -> None:
    """Run production database backup after successful ingestion.

    Fire-and-forget: logs errors but never blocks or fails the ingestion.
    The backup script path is hardcoded (not user input) and the work_id
    label is passed as a single argument to the script (no shell expansion).
    """
    # Never fire the production backup from test runs: test-triggered
    # backups pollute the backup directory and the keep-10 retention for
    # post-ingest backups can prune REAL backups (observed 2026-07-02).
    if "author_library_test" in os.environ.get("DB_POSTGRES_URL", ""):
        log.debug("post_ingest_backup_skipped", reason="test database in use")
        return
    if not _BACKUP_SCRIPT.exists():
        log.debug("post_ingest_backup_skipped", reason="backup script not found")
        return
    try:
        label = f"post-ingest-{work_id}"
        proc = await asyncio.create_subprocess_exec(
            str(_BACKUP_SCRIPT), label,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode == 0:
            log.info("post_ingest_backup_complete", work_id=work_id)
        else:
            log.warning(
                "post_ingest_backup_failed",
                returncode=proc.returncode,
                stderr=stderr.decode()[:500],
            )
    except asyncio.TimeoutError:
        log.warning("post_ingest_backup_timeout", work_id=work_id)
    except Exception as exc:
        log.warning("post_ingest_backup_error", error=str(exc))


async def run_post_ingestion_hooks(
    *,
    result: IngestionResult,
    subject_author_id: str,
    settings: "Settings",
    storage: "StorageManager",
    embedding_provider: "EmbeddingProvider",
) -> dict[str, Any]:
    """Run post-ingestion hooks shared by all ingestion entry points.

    Performs: cross-work analysis (voice profiles, thematic index),
    production backup, and returns the result dict.  Cache invalidation
    and QG2 enqueueing are skipped when no cache_manager/task_queue is
    available (e.g. standalone scripts).
    """
    cross_work_summary: dict[str, Any] = {}
    if result.source_class == "primary":
        cross_work_summary = await _run_cross_work_analysis(
            subject_author_id=subject_author_id,
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
        )

    await _run_post_ingest_backup(result.work_id)

    response = result.to_dict()
    if cross_work_summary:
        response["cross_work_analysis"] = cross_work_summary

    return response
