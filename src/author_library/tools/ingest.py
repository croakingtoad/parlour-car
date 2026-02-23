"""MCP tool handlers for ingestion (E010).

Provides:
  - ingest_book: Ingest a single work through the full pipeline.
  - ingest_corpus: Bulk ingest multiple works and run cross-work analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from author_library.errors import IngestionError
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
) -> str:
    """Handle the ingest_book MCP tool call.

    Arguments:
        file_path (str): Path to the document file.
        subject_author_id (str): The subject author's slug identifier.
        metadata_hints (dict, optional): Overrides for classification and catalog fields.

    Returns:
        JSON string with ingestion summary.
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

    # Invalidate caches — new content may affect query/graph/voice/thematic results
    if cache_manager is not None:
        await cache_manager.invalidate_on_ingestion(author_id=subject_author_id)

    return json.dumps(result.to_dict(), indent=2)


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
        generator = ThematicIndexGenerator(settings)
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

    # 2. Voice profile extraction
    try:
        extractor = VoiceProfileExtractor(settings)
        profile = await extractor.extract(
            author_id=subject_author_id,
            author_name=subject_author_id,
            work_repo=storage.works,
            chunk_repo=storage.chunks,
        )
        manager = VoiceProfileManager(settings)
        await manager.store_profile(
            profile=profile,
            voice_repo=storage.voice_profiles,
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
