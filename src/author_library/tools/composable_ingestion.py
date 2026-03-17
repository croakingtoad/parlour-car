"""Composable ingestion MCP tool handlers (Epic B).

Exposes the existing ingestion pipeline's individual stages as separate
MCP tools so a human can confirm at each step:

  classify_source  →  catalog_source  →  chunk_source  →  detect_passage_links  →  flag_acquisition

Each tool wraps real pipeline code — no mocks, no placeholders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from author_library.catalog.classifier import SourceClassifier
from author_library.catalog.mixed_authorship import MixedAuthorshipAnalyzer
from author_library.catalog.models import (
    ClassificationResult,
    ProcessingRoute,
    SourceClass,
    route_for_source_class,
)
from author_library.catalog.pipeline import ClassificationPipeline
from author_library.chunking import get_chunking_strategy
from author_library.chunking.annotator import AnnotationContext, ChunkAnnotator
from author_library.errors import IngestionError
from author_library.graph.entity_extraction import EntityExtractor
from author_library.graph.linking_explicit import ExplicitLinkDetector
from author_library.graph.linking_implicit import ImplicitEngagementDetector
from author_library.graph.linking_thematic import ThematicParallelDetector
from author_library.parsing import get_parser
from author_library.parsing.models import SectionType

if TYPE_CHECKING:
    from author_library.cache import CacheManager
    from author_library.chunking.models import Chunk
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# B1: classify_source
# ---------------------------------------------------------------------------


async def handle_classify_source(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
    cache_manager: CacheManager | None = None,
) -> str:
    """Handle the classify_source MCP tool call.

    Parses a document and runs classification without storing anything.
    Returns suggested classification for human review.

    Arguments:
        file_path (str): Path to the document file.
        subject_author (str): The subject author's slug identifier.
        hints (dict, optional): User-provided hints for classification.

    Returns:
        JSON with suggested_class, confidence, signals, work_type, etc.
    """
    file_path = arguments.get("file_path")
    if not file_path:
        raise IngestionError(
            "file_path is required",
            context={"arguments": arguments},
        )

    subject_author = arguments.get("subject_author")
    if not subject_author:
        raise IngestionError(
            "subject_author is required",
            context={"arguments": arguments},
        )

    path = Path(file_path)
    if not path.exists():
        raise IngestionError(
            f"File not found: {file_path}",
            context={"file_path": file_path},
        )

    hints = arguments.get("hints") or {}

    # Step 1: Parse the document
    parser = get_parser(path)
    document = await parser.parse(str(path))

    log.info(
        "classify_source_parsed",
        title=document.metadata.title,
        word_count=document.metadata.word_count,
    )

    # Step 2: Run the classifier (does NOT store anything)
    classifier = SourceClassifier(settings)
    classification = await classifier.classify(
        document,
        subject_author=subject_author,
        metadata_hints=hints,
    )

    # Step 3: Detect mixed authorship
    analyzer = MixedAuthorshipAnalyzer(subject_author)
    mixed_result = analyzer.analyze(
        document,
        document_source_class=classification.source_class,
    )

    # Step 4: Infer a suggested work type from genre tags or hints
    suggested_work_type = hints.get("work_type", "other")
    if suggested_work_type == "other" and document.metadata.title:
        # Use genre tags if available from classification signals
        suggested_work_type = _infer_work_type(classification)

    # Step 5: Determine if human judgment is required
    requires_human = (
        classification.confidence < 0.8
        or mixed_result.is_mixed
        or mixed_result.requires_extraction
    )
    judgment_reason = None
    if classification.confidence < 0.8:
        judgment_reason = (
            f"Classification confidence ({classification.confidence:.2f}) is below 0.80 threshold."
        )
    elif mixed_result.is_mixed:
        judgment_reason = (
            f"Mixed authorship detected: {len(mixed_result.segments)} segment(s). "
            f"{mixed_result.analysis_notes}"
        )

    # Build signal details
    signals = []
    for sig in classification.signals_detected:
        signals.append({
            "signal": sig,
            "weight": "moderate",
            "interpretation": sig,
        })

    result = {
        "suggested_class": classification.source_class.value,
        "confidence": round(classification.confidence, 3),
        "signals": signals,
        "reasoning": classification.reasoning,
        "suggested_work_type": suggested_work_type,
        "mixed_authorship_detected": mixed_result.is_mixed,
        "mixed_authorship_note": mixed_result.analysis_notes if mixed_result.is_mixed else None,
        "requires_human_judgment": requires_human,
        "judgment_reason": judgment_reason,
        "document_metadata": {
            "title": document.metadata.title,
            "author": document.metadata.author,
            "word_count": document.metadata.word_count,
            "format": document.format,
        },
    }

    return json.dumps(result, indent=2)


def _infer_work_type(classification: ClassificationResult) -> str:
    """Infer a work type from classification signals."""
    signals_text = " ".join(classification.signals_detected).lower()
    reasoning = classification.reasoning.lower()
    combined = f"{signals_text} {reasoning}"

    if "poetry" in combined or "poem" in combined:
        return "poetry-collection"
    if "sermon" in combined:
        return "sermon"
    if "letter" in combined:
        return "letter"
    if "lecture" in combined:
        return "lecture-transcript"
    if "essay" in combined:
        return "essay-collection"
    if "interview" in combined:
        return "interview-responses"
    if "academic" in combined or "paper" in combined:
        return "academic-paper"
    return "monograph"


# ---------------------------------------------------------------------------
# B2: catalog_source
# ---------------------------------------------------------------------------


async def handle_catalog_source(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
    cache_manager: CacheManager | None = None,
) -> str:
    """Handle the catalog_source MCP tool call.

    Runs the classification pipeline with user-confirmed source_class
    and stores the catalog entry. Returns the work_id and full record.

    Arguments:
        file_path (str): Path to the document file.
        source_class (str): Confirmed source class (primary/secondary/contextual/tertiary/personal).
        work_type (str): Confirmed work type.
        metadata_overrides (dict, optional): User corrections to auto-detected metadata.

    Returns:
        JSON with work_id, catalog_record, chapters_detected, table_of_contents.
    """
    file_path = arguments.get("file_path")
    if not file_path:
        raise IngestionError(
            "file_path is required",
            context={"arguments": arguments},
        )

    source_class_str = arguments.get("source_class")
    if not source_class_str:
        raise IngestionError(
            "source_class is required",
            context={"arguments": arguments},
        )

    # Validate source class
    try:
        source_class = SourceClass(source_class_str)
    except ValueError as exc:
        raise IngestionError(
            f"Invalid source_class: {source_class_str}. "
            f"Must be one of: primary, secondary, contextual, tertiary, personal",
            context={"source_class": source_class_str},
        ) from exc

    path = Path(file_path)
    if not path.exists():
        raise IngestionError(
            f"File not found: {file_path}",
            context={"file_path": file_path},
        )

    work_type = arguments.get("work_type", "other")
    metadata_overrides = arguments.get("metadata_overrides") or {}

    # Merge user-confirmed fields into overrides
    overrides = {
        **metadata_overrides,
        "source_class": source_class_str,
        "work_type": work_type,
    }

    # Infer subject_author from overrides or metadata
    subject_author = metadata_overrides.get(
        "subject_author_id",
        metadata_overrides.get("subject_author", "unknown"),
    )

    # Step 1: Parse document
    parser = get_parser(path)
    document = await parser.parse(str(path))

    # Step 2: Run classification pipeline (stores catalog entry in works table)
    pipeline = ClassificationPipeline(
        settings=settings,
        work_repository=storage.works,
        subject_author=subject_author,
        pg_pool=storage.pg,
    )

    pipeline_result = await pipeline.process(
        document,
        metadata_hints=overrides,
        user_overrides=overrides,
    )

    catalog_entry = pipeline_result.catalog_entry
    work_id = catalog_entry.work_id

    # Step 3: Upsert work node in Neo4j graph
    await storage.graph.upsert_work_node({
        "work_id": work_id,
        "title": catalog_entry.title,
        "author": catalog_entry.author,
        "source_class": source_class.value,
        "publication_year": catalog_entry.publication_year,
    })

    # Extract chapters from document tree
    chapters_detected = _extract_chapters(document)

    # Build catalog record for response
    catalog_record = {
        "title": catalog_entry.title,
        "author": catalog_entry.author,
        "publication_year": catalog_entry.publication_year,
        "publisher": catalog_entry.publisher,
        "isbn": catalog_entry.isbn,
        "word_count": catalog_entry.word_count,
        "language": catalog_entry.language,
        "genre_tags": catalog_entry.genre_tags,
        "subject_headings": catalog_entry.subject_headings,
        "format_ingested": catalog_entry.format_ingested.value
        if hasattr(catalog_entry.format_ingested, "value")
        else str(catalog_entry.format_ingested),
        "source_class": source_class.value,
        "source_class_note": catalog_entry.source_class_note,
    }

    # Table of contents from parsed document
    toc = document.metadata.table_of_contents or []

    result = {
        "work_id": work_id,
        "catalog_record": catalog_record,
        "chapters_detected": chapters_detected,
        "table_of_contents": toc,
    }

    log.info(
        "catalog_source_complete",
        work_id=work_id,
        source_class=source_class.value,
        chapters=len(chapters_detected),
    )

    return json.dumps(result, indent=2, default=str)


def _extract_chapters(document: Any) -> list[dict[str, Any]]:
    """Extract chapter information from the parsed document tree."""
    from author_library.parsing.models import NodeType

    chapters: list[dict[str, Any]] = []
    chapter_num = 0

    def walk(node: Any) -> None:
        nonlocal chapter_num
        if node.node_type == NodeType.CHAPTER:
            chapter_num += 1
            title = node.metadata.get("title", "")
            if not title and node.children:
                for child in node.children:
                    if child.node_type == NodeType.HEADING and child.text:
                        title = child.text
                        break
            if not title:
                title = f"Chapter {chapter_num}"

            word_count = len((node.text or "").split())
            # Also count words in children
            for child in node.children:
                if child.text:
                    word_count += len(child.text.split())

            chapters.append({
                "number": chapter_num,
                "title": title,
                "word_count": word_count,
            })
        for child in node.children:
            walk(child)

    walk(document.tree)
    return chapters


# ---------------------------------------------------------------------------
# B3: chunk_source
# ---------------------------------------------------------------------------


async def handle_chunk_source(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
    cache_manager: CacheManager | None = None,
) -> str:
    """Handle the chunk_source MCP tool call.

    Chunks a previously cataloged work, annotates chunks, stores in PG,
    generates embeddings, and upserts chunk nodes in Neo4j.

    Arguments:
        work_id (str): The work ID from catalog_source.
        chunking_strategy_override (str, optional): Override auto-detected genre strategy.

    Returns:
        JSON with chunks_created breakdown, genre_detected, embeddings_generated, status.
    """
    work_id = arguments.get("work_id")
    if not work_id:
        raise IngestionError(
            "work_id is required",
            context={"arguments": arguments},
        )

    chunking_strategy_override = arguments.get("chunking_strategy_override")

    # Step 1: Fetch the work record from storage
    work = await storage.works.get(work_id)
    if not work:
        raise IngestionError(
            f"Work not found: {work_id}",
            context={"work_id": work_id},
        )

    source_class_str = work.get("source_class", "primary")
    source_class = SourceClass(source_class_str)
    route = route_for_source_class(source_class)

    # Tertiary sources get metadata only — no chunking
    if route == ProcessingRoute.METADATA_ONLY:
        return json.dumps({
            "work_id": work_id,
            "chunks_created": {"macro": 0, "meso": 0, "micro": 0},
            "genre_detected": "n/a",
            "chunking_strategy_used": "none (metadata_only route)",
            "embeddings_generated": 0,
            "embedding_provider": embedding_provider.provider_name,
            "status": "complete",
            "note": "Tertiary sources receive metadata only — no content processing.",
        }, indent=2)

    # Step 2: Re-parse document from file path stored in work metadata
    # The work record stores the original file path in source_metadata
    # asyncpg may return JSONB columns as a JSON string rather than a parsed dict
    source_meta_raw = work.get("source_metadata") or {}
    source_meta = (
        json.loads(source_meta_raw)
        if isinstance(source_meta_raw, str)
        else dict(source_meta_raw)
    )
    file_path = work.get("file_path") or source_meta.get("file_path")

    # If file_path not in work record, we need to re-parse from the work data
    # The document text lives in chunks already or we need the original file
    # We'll look for the file_path in metadata_overrides stored during cataloging
    if not file_path:
        raise IngestionError(
            "Cannot chunk: no file_path found in work record. "
            "Re-run catalog_source with the file_path.",
            context={"work_id": work_id},
        )

    path = Path(file_path)
    if not path.exists():
        raise IngestionError(
            f"Original file no longer accessible: {file_path}",
            context={"work_id": work_id, "file_path": file_path},
        )

    parser = get_parser(path)
    document = await parser.parse(str(path))

    # Step 3: Determine pass number (idempotent re-ingestion)
    current_max_pass = await storage.chunks.get_max_pass_number(work_id)
    pass_number = current_max_pass + 1 if current_max_pass > 0 else 1

    # Delete existing chunks for re-chunking
    deleted = await storage.chunks.delete_by_work(work_id)
    if deleted > 0:
        log.info("chunk_source_cleared_old", work_id=work_id, deleted=deleted)

    # Also clear Neo4j chunk nodes so stale UUIDs don't persist
    try:
        deleted_graph = await storage.graph.delete_chunks_for_work(work_id)
        if deleted_graph > 0:
            log.info("chunk_source_cleared_graph", work_id=work_id, deleted_graph=deleted_graph)
    except Exception as exc:
        log.warning("neo4j_chunk_cleanup_failed", work_id=work_id, error=str(exc))

    # Step 4: Select chunking strategy
    genre_tags = work.get("genre_tags") or ["unclassified"]
    if chunking_strategy_override:
        genre_tags = [chunking_strategy_override]

    strategy = get_chunking_strategy(genre_tags)
    genre_detected = type(strategy).__name__

    # Step 5: Chunk
    chunks = strategy.chunk(document, work_id, source_class_str)

    chunks_by_gran: dict[str, int] = {}
    for chunk in chunks:
        gran = str(chunk.granularity)
        chunks_by_gran[gran] = chunks_by_gran.get(gran, 0) + 1

    log.info(
        "chunk_source_chunked",
        work_id=work_id,
        total=len(chunks),
        by_granularity=chunks_by_gran,
    )

    # Step 5b: Section-type routing — filter out non-content sections
    # (matches IngestionPipeline._filter_by_section_type logic)
    _CONTENT_SECTION_TYPES = {"chapter", "preface", "back_matter"}
    pre_filter_count = len(chunks)
    skipped_sections: dict[str, int] = {}
    content_chunks: list[Chunk] = []
    for chunk in chunks:
        if chunk.section_type in _CONTENT_SECTION_TYPES:
            content_chunks.append(chunk)
        else:
            skipped_sections[chunk.section_type] = (
                skipped_sections.get(chunk.section_type, 0) + 1
            )

    if skipped_sections:
        log.info(
            "chunk_source_section_type_filter",
            work_id=work_id,
            original_chunks=pre_filter_count,
            kept_chunks=len(content_chunks),
            skipped_by_type=skipped_sections,
        )
    chunks = content_chunks

    # Step 6: Annotate
    annotation_ctx = _build_annotation_context_from_work(work, source_class)
    annotator = ChunkAnnotator(settings)
    chunks = await annotator.annotate_chunks(chunks, annotation_ctx)

    # Step 7: Store chunks in PG (sorted by granularity for FK resolution)
    _gran_order = {"macro": 0, "meso": 1, "micro": 2, "nano": 3}
    sorted_chunks = sorted(
        chunks,
        key=lambda c: (_gran_order.get(str(c.granularity), 9), c.position),
    )

    errors: list[str] = []
    chunk_id_map: dict[str, UUID] = {}
    pg_chunk_errors = 0
    for chunk in sorted_chunks:
        resolved_parent: UUID | None = None
        if chunk.parent_chunk_id is not None:
            resolved_parent = chunk_id_map.get(chunk.parent_chunk_id)

        metadata = dict(chunk.metadata)
        metadata["section_type"] = chunk.section_type

        chunk_data: dict[str, Any] = {
            "work_id": chunk.work_id,
            "text": chunk.text,
            "annotation": chunk.annotation,
            "granularity": str(chunk.granularity),
            "source_class": chunk.source_class,
            "chapter": chunk.chapter,
            "section": chunk.section,
            "position": chunk.position,
            "parent_chunk_id": resolved_parent,
            "metadata": metadata,
            "raw_content": chunk.raw_content,
            "raw_content_window": chunk.raw_content_window,
            "pass_number": pass_number,
        }
        try:
            pg_id = await storage.chunks.create(chunk_data)
            chunk_id_map[chunk.id] = pg_id
        except Exception as exc:
            pg_chunk_errors += 1
            if pg_chunk_errors <= 3:
                log.error(
                    "pg_chunk_store_failed",
                    work_id=work_id,
                    chunk_id=chunk.id,
                    error=str(exc),
                    position=chunk.position,
                )
            elif pg_chunk_errors == 4:
                log.error(
                    "pg_chunk_store_failed_suppressed",
                    work_id=work_id,
                    message="Further PG chunk store errors will be suppressed",
                )

    if pg_chunk_errors > 0:
        log.error(
            "pg_chunk_store_summary",
            work_id=work_id,
            stored=len(chunk_id_map),
            failed=pg_chunk_errors,
            total=len(sorted_chunks),
        )
        errors.append(
            f"PG chunk storage: {pg_chunk_errors}/{len(sorted_chunks)} chunks failed"
        )

    # Update engagement_passes on the work record
    await storage.works.update(work_id, {"engagement_passes": pass_number})

    # Step 8: Embed chunks
    import asyncio as _asyncio

    embeddings_stored = 0
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.annotated_text for c in batch]
        try:
            batch_result = await embedding_provider.embed_batch(texts)
            pg_store_errors = 0
            for chunk, vector in zip(batch, batch_result.vectors, strict=True):
                maybe_id = chunk_id_map.get(chunk.id)
                if maybe_id is None:
                    continue
                pg_id = maybe_id
                try:
                    await storage.embeddings.store(
                        pg_id,
                        vector,
                        embedding_provider.provider_name,
                        embedding_provider.model_name,
                        embedding_provider.dimensions,
                    )
                    embeddings_stored += 1
                except Exception:
                    try:
                        await _asyncio.sleep(0.5)
                        await storage.embeddings.store(
                            pg_id,
                            vector,
                            embedding_provider.provider_name,
                            embedding_provider.model_name,
                            embedding_provider.dimensions,
                        )
                        embeddings_stored += 1
                    except Exception as retry_exc:
                        pg_store_errors += 1
                        if pg_store_errors <= 3:
                            log.error(
                                "pg_embedding_store_failed",
                                work_id=work_id,
                                chunk_id=chunk.id,
                                error=str(retry_exc),
                            )
                        elif pg_store_errors == 4:
                            log.error(
                                "pg_embedding_store_failed_suppressed",
                                work_id=work_id,
                                message="Further PG embedding store errors will be suppressed",
                            )
            if pg_store_errors > 0:
                errors.append(
                    f"PG embedding store: {pg_store_errors} chunks failed in batch {i}"
                )
        except Exception as exc:
            error_msg = f"Embedding batch {i}-{i + len(batch)} failed: {exc}"
            log.error("chunk_source_embedding_failed", error=error_msg)
            errors.append(error_msg)

    # Step 9: Upsert chunk nodes in Neo4j
    neo4j_chunk_errors = 0
    for chunk in chunks:
        chunk_node: dict[str, Any] = {
            "chunk_id": chunk.id,
            "work_id": chunk.work_id,
            "text_preview": chunk.text[:200],
            "granularity": str(chunk.granularity),
            "source_class": chunk.source_class,
        }
        if source_class == SourceClass.PERSONAL:
            chunk_node["user_id"] = "marty"
        try:
            await storage.graph.upsert_chunk_node(chunk_node)
        except Exception as exc:
            neo4j_chunk_errors += 1
            if neo4j_chunk_errors <= 3:
                log.error(
                    "neo4j_chunk_upsert_failed",
                    work_id=work_id,
                    chunk_id=chunk.id,
                    error=str(exc),
                )
            elif neo4j_chunk_errors == 4:
                log.error(
                    "neo4j_chunk_upsert_failed_suppressed",
                    work_id=work_id,
                    message="Further Neo4j chunk upsert errors will be suppressed",
                )

    # Entity extraction (PRIMARY and SECONDARY only)
    # Structural sections are excluded even though they are already filtered out
    # by the section_type filter above; this is defense-in-depth.
    _ENTITY_EXTRACT_EXCLUDED = {
        SectionType.BIBLIOGRAPHY.value,
        SectionType.INDEX.value,
        SectionType.TABLE_OF_CONTENTS.value,
        SectionType.FRONT_MATTER.value,
    }
    entity_count = 0
    if route in (ProcessingRoute.FULL_ENRICHMENT, ProcessingRoute.EMBEDDINGS_AND_GRAPH):
        extraction_chunks = [
            c for c in chunks if c.section_type not in _ENTITY_EXTRACT_EXCLUDED
        ]
        try:
            extractor = EntityExtractor(
                storage.neo4j,
                settings.api_keys,
                settings.llm,
            )
            extraction_result = await extractor.extract_and_persist(
                extraction_chunks,
                work_title=work.get("title", ""),
                author=work.get("author", ""),
            )
            entity_count = extraction_result.nodes_created
            if extraction_result.errors:
                errors.extend(extraction_result.errors)
        except Exception as exc:
            errors.append(f"Entity extraction failed: {exc}")

        # Theme deduplication — merge near-duplicate Theme nodes
        try:
            from author_library.graph.theme_dedup import deduplicate_themes

            dedup_result = await deduplicate_themes(
                storage.neo4j,
                embedding_provider,
            )
            if dedup_result.merged_count > 0:
                log.info(
                    "chunk_source_theme_dedup",
                    work_id=work_id,
                    original_themes=dedup_result.original_count,
                    canonical_themes=dedup_result.canonical_count,
                    merged=dedup_result.merged_count,
                )
            if dedup_result.errors:
                errors.extend(dedup_result.errors)
        except Exception as exc:
            errors.append(f"Theme deduplication failed: {exc}")

    # Post-ingestion quality checks
    quality_checks = None
    try:
        from author_library.tools.ingestion_pipeline import IngestionPipeline

        _qc_pipeline = IngestionPipeline(
            settings=settings,
            storage=storage,
            embedding_provider=embedding_provider,
        )
        _source_meta_raw = work.get("source_metadata") or {}
        _source_meta = json.loads(_source_meta_raw) if isinstance(_source_meta_raw, str) else dict(_source_meta_raw)
        subject_author_id = _source_meta.get("subject_author_id", "")
        quality_checks = await _qc_pipeline._run_quality_checks(
            work_id, source_class_str, subject_author_id,
        )
        if quality_checks.get("warnings"):
            errors.extend(f"[quality] {w}" for w in quality_checks["warnings"])
    except Exception as exc:
        log.error("chunk_source_quality_checks_failed", work_id=work_id, error=str(exc))
        errors.append(f"Quality checks failed: {exc}")

    # First macro chunk as sample for user review
    sample_macro = ""
    macro_chunks = [c for c in chunks if str(c.granularity) == "macro"]
    if macro_chunks:
        sample_macro = macro_chunks[0].text[:500]

    status = "complete" if not errors else "partial"

    result: dict[str, Any] = {
        "work_id": work_id,
        "chunks_created": {
            "macro": chunks_by_gran.get("macro", 0),
            "meso": chunks_by_gran.get("meso", 0),
            "micro": chunks_by_gran.get("micro", 0),
        },
        "sample_macro_summary": sample_macro,
        "genre_detected": genre_detected,
        "chunking_strategy_used": type(strategy).__name__,
        "embeddings_generated": embeddings_stored,
        "embedding_provider": embedding_provider.provider_name,
        "entity_count": entity_count,
        "pass_number": pass_number,
        "status": status,
        "errors": errors if errors else None,
    }
    if quality_checks is not None:
        result["quality_checks"] = quality_checks

    log.info(
        "chunk_source_complete",
        work_id=work_id,
        chunks=len(chunks),
        embeddings=embeddings_stored,
        status=status,
    )

    return json.dumps(result, indent=2)


def _build_annotation_context_from_work(
    work: dict[str, Any], source_class: SourceClass
) -> AnnotationContext:
    """Build AnnotationContext from a work record dict."""
    # asyncpg may return JSONB columns as a JSON string rather than a parsed dict
    source_meta_raw = work.get("source_metadata") or {}
    source_meta = (
        json.loads(source_meta_raw)
        if isinstance(source_meta_raw, str)
        else dict(source_meta_raw)
    )

    subject_author = source_meta.get("subject_author_id", "")
    if not subject_author:
        subject_author = source_meta.get("about_author_id", "")
    if not subject_author:
        subject_author = source_meta.get("referenced_by", "")
    if not subject_author:
        subject_author = work.get("author", "")

    return AnnotationContext(
        work_title=work.get("title", ""),
        publication_year=work.get("publication_year"),
        author=work.get("author", ""),
        subject_author=subject_author,
        relationship_type=source_meta.get("relationship"),
        perspective_note=source_meta.get("perspective_note"),
        engagement_note=source_meta.get("engagement_note"),
        engagement_works=", ".join(source_meta.get("engagement_works") or []),
    )


# ---------------------------------------------------------------------------
# B4: detect_passage_links
# ---------------------------------------------------------------------------


async def handle_detect_passage_links(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
    cache_manager: CacheManager | None = None,
) -> str:
    """Handle the detect_passage_links MCP tool call.

    Detects cross-resource passage links for a work's chunks using
    the 3-tier linking system. Optionally runs retroactive scan.

    Arguments:
        work_id (str): The work to detect links for.
        scan_types (list[str]): Types to scan — explicit_citation, implicit_engagement, thematic_parallel.
        confidence_threshold (float, optional): Minimum confidence (default 0.5).
        retroactive_scan (bool, optional): Also scan existing chunks against new work.

    Returns:
        JSON with links_created breakdown, contextual_sources_referenced, unresolved_references.
    """
    work_id = arguments.get("work_id")
    if not work_id:
        raise IngestionError(
            "work_id is required",
            context={"arguments": arguments},
        )

    scan_types = arguments.get("scan_types") or [
        "explicit_citation",
        "implicit_engagement",
        "thematic_parallel",
    ]
    confidence_threshold = arguments.get("confidence_threshold", 0.5)
    retroactive_scan = arguments.get("retroactive_scan", False)

    # Fetch work record
    work = await storage.works.get(work_id)
    if not work:
        raise IngestionError(
            f"Work not found: {work_id}",
            context={"work_id": work_id},
        )

    source_class = SourceClass(work.get("source_class", "primary"))

    # Load this work's meso chunks from the database
    db_chunks_raw = await storage.chunks.list_by_work(work_id, granularity="meso")
    if not db_chunks_raw:
        return json.dumps({
            "work_id": work_id,
            "links_created": {
                "explicit_citation": 0,
                "implicit_engagement": 0,
                "thematic_parallel": 0,
            },
            "contextual_sources_referenced": [],
            "unresolved_references": [],
            "note": "No meso chunks found for this work. Run chunk_source first.",
        }, indent=2)

    from author_library.chunking.models import Chunk as ChunkModel

    work_chunks = [
        ChunkModel(
            id=str(c.get("id", "")),
            text=c.get("text", ""),
            granularity=c.get("granularity", "meso"),
            work_id=c.get("work_id", ""),
            source_class=c.get("source_class", source_class.value),
            position=c.get("position", 0),
        )
        for c in db_chunks_raw
    ]

    # Determine which chunks to link against
    # Extract author slug from work_id (everything before --)
    author_slug = work_id.split("--")[0] if "--" in work_id else work_id
    all_works = await storage.works.list_by_author(author_slug)

    # Load counterpart chunks based on source class
    counterpart_chunks = await _load_counterpart_chunks(
        storage, source_class, all_works, work_id
    )

    links_created = {
        "explicit_citation": 0,
        "implicit_engagement": 0,
        "thematic_parallel": 0,
    }
    errors: list[str] = []

    # Determine primary and contextual sides for linkers
    if source_class == SourceClass.PRIMARY:
        primary_side = work_chunks
        contextual_side = counterpart_chunks
    elif source_class == SourceClass.CONTEXTUAL:
        primary_side = counterpart_chunks
        contextual_side = work_chunks
    else:
        # Secondary/personal/tertiary — no passage linking
        return json.dumps({
            "work_id": work_id,
            "links_created": links_created,
            "contextual_sources_referenced": [],
            "unresolved_references": [],
            "note": f"Passage linking not applicable for {source_class.value} sources.",
        }, indent=2)

    if not contextual_side:
        return json.dumps({
            "work_id": work_id,
            "links_created": links_created,
            "contextual_sources_referenced": [],
            "unresolved_references": [],
            "note": "No counterpart chunks found to link against.",
        }, indent=2)

    existing_links: set[tuple[str, str]] = set()

    # Tier 1: Explicit citations
    if "explicit_citation" in scan_types:
        try:
            explicit = ExplicitLinkDetector(storage.neo4j)
            explicit_result = await explicit.detect_and_link(
                primary_side, contextual_side
            )
            links_created["explicit_citation"] = explicit_result.edges_created
            existing_links = {
                (link.source_chunk_id, link.target_chunk_id)
                for link in explicit_result.links
            }
        except Exception as exc:
            errors.append(f"Explicit citation detection failed: {exc}")

    # Tier 2: Implicit engagement
    if "implicit_engagement" in scan_types:
        try:
            implicit = ImplicitEngagementDetector(storage.neo4j)
            implicit_result = await implicit.detect_and_link(
                primary_side,
                contextual_side,
                existing_links=existing_links,
            )
            links_created["implicit_engagement"] = implicit_result.edges_created
        except Exception as exc:
            errors.append(f"Implicit engagement detection failed: {exc}")

    # Tier 3: Thematic parallels
    if "thematic_parallel" in scan_types:
        try:
            thematic = ThematicParallelDetector(storage.neo4j, embedding_provider)
            thematic_result = await thematic.detect_and_link(
                primary_side,
                contextual_side,
                similarity_threshold=confidence_threshold,
            )
            links_created["thematic_parallel"] = thematic_result.edges_created
        except Exception as exc:
            errors.append(f"Thematic parallel detection failed: {exc}")

    # Retroactive scan: also scan existing primary works against this new work
    if retroactive_scan and source_class == SourceClass.CONTEXTUAL:
        retro_links = await _retroactive_link_scan(
            storage, embedding_provider, work_chunks, all_works, scan_types,
            confidence_threshold,
        )
        for key in links_created:
            links_created[key] += retro_links.get(key, 0)

    # Build contextual sources referenced list
    ctx_sources = _build_contextual_sources_referenced(
        counterpart_chunks, all_works, source_class
    )

    # Post-detection: run connection surfacing (N1/N3)
    # Scan for new cross-work connections and generate PR content.
    surfacing_summary: dict[str, Any] = {}
    total_links = sum(links_created.values())
    if total_links > 0 or retroactive_scan:
        try:
            from author_library.surfacing.batch_surfacing import BatchSurfacer

            surfacer = BatchSurfacer(
                settings=settings,
                storage=storage,
                embedding_provider=embedding_provider,
                cache_manager=cache_manager,
            )
            surfacing_result = await surfacer.surface_after_ingestion(
                work_id,
                work_title=work.get("title", ""),
                work_author=work.get("author", ""),
            )
            surfacing_summary = surfacing_result.to_dict()
        except Exception as exc:
            surfacing_error = f"Post-detection surfacing failed: {exc}"
            log.error("detect_passage_links_surfacing_failed", error=surfacing_error)
            errors.append(surfacing_error)

    result: dict[str, Any] = {
        "work_id": work_id,
        "links_created": links_created,
        "contextual_sources_referenced": ctx_sources,
        "unresolved_references": [],
    }
    if surfacing_summary:
        result["surfacing"] = surfacing_summary
    if errors:
        result["errors"] = errors

    log.info(
        "detect_passage_links_complete",
        work_id=work_id,
        total_links=total_links,
        surfacing_connections=surfacing_summary.get("total_connections", 0),
    )

    return json.dumps(result, indent=2)


async def _load_counterpart_chunks(
    storage: StorageManager,
    source_class: SourceClass,
    all_works: list[dict[str, Any]],
    exclude_work_id: str,
) -> list[Any]:
    """Load counterpart chunks for passage linking."""
    from author_library.chunking.models import Chunk as ChunkModel

    target_class = "contextual" if source_class == SourceClass.PRIMARY else "primary"
    target_work_ids = [
        w["work_id"]
        for w in all_works
        if w.get("source_class") == target_class and w["work_id"] != exclude_work_id
    ]

    all_chunks: list[Any] = []
    for wid in target_work_ids:
        db_chunks = await storage.chunks.list_by_work(wid, granularity="meso")
        for c in db_chunks:
            all_chunks.append(
                ChunkModel(
                    id=str(c.get("id", "")),
                    text=c.get("text", ""),
                    granularity=c.get("granularity", "meso"),
                    work_id=c.get("work_id", ""),
                    source_class=c.get("source_class", target_class),
                    position=c.get("position", 0),
                )
            )
    return all_chunks


async def _retroactive_link_scan(
    storage: StorageManager,
    embedding_provider: Any,
    new_chunks: list[Any],
    all_works: list[dict[str, Any]],
    scan_types: list[str],
    confidence_threshold: float,
) -> dict[str, int]:
    """Scan existing primary works against newly ingested contextual chunks."""
    from author_library.chunking.models import Chunk as ChunkModel

    retro_links: dict[str, int] = {
        "explicit_citation": 0,
        "implicit_engagement": 0,
        "thematic_parallel": 0,
    }

    primary_work_ids = [
        w["work_id"] for w in all_works if w.get("source_class") == "primary"
    ]

    for prim_wid in primary_work_ids:
        db_chunks = await storage.chunks.list_by_work(prim_wid, granularity="meso")
        if not db_chunks:
            continue

        primary_chunks = [
            ChunkModel(
                id=str(c.get("id", "")),
                text=c.get("text", ""),
                granularity=c.get("granularity", "meso"),
                work_id=c.get("work_id", ""),
                source_class=c.get("source_class", "primary"),
                position=c.get("position", 0),
            )
            for c in db_chunks
        ]

        if "explicit_citation" in scan_types:
            try:
                explicit = ExplicitLinkDetector(storage.neo4j)
                result = await explicit.detect_and_link(primary_chunks, new_chunks)
                retro_links["explicit_citation"] += result.edges_created
            except Exception as exc:
                log.error(
                    "retroactive_explicit_citation_failed",
                    primary_work_id=prim_wid,
                    error=str(exc),
                    exc_info=exc,
                )

        if "implicit_engagement" in scan_types:
            try:
                implicit = ImplicitEngagementDetector(storage.neo4j)
                result = await implicit.detect_and_link(primary_chunks, new_chunks)
                retro_links["implicit_engagement"] += result.edges_created
            except Exception as exc:
                log.error(
                    "retroactive_implicit_engagement_failed",
                    primary_work_id=prim_wid,
                    error=str(exc),
                    exc_info=exc,
                )

        if "thematic_parallel" in scan_types:
            try:
                thematic = ThematicParallelDetector(storage.neo4j, embedding_provider)
                result = await thematic.detect_and_link(
                    primary_chunks, new_chunks,
                    similarity_threshold=confidence_threshold,
                )
                retro_links["thematic_parallel"] += result.edges_created
            except Exception as exc:
                log.error(
                    "retroactive_thematic_parallel_failed",
                    primary_work_id=prim_wid,
                    error=str(exc),
                    exc_info=exc,
                )

    return retro_links


def _build_contextual_sources_referenced(
    counterpart_chunks: list[Any],
    all_works: list[dict[str, Any]],
    source_class: SourceClass,
) -> list[dict[str, Any]]:
    """Build the contextual_sources_referenced output list."""
    work_lookup = {w["work_id"]: w for w in all_works}
    work_ref_counts: dict[str, int] = {}

    for chunk in counterpart_chunks:
        wid = chunk.work_id
        work_ref_counts[wid] = work_ref_counts.get(wid, 0) + 1

    sources = []
    for wid, count in sorted(work_ref_counts.items(), key=lambda x: -x[1]):
        work_info = work_lookup.get(wid, {})
        sources.append({
            "work_id": wid,
            "title": work_info.get("title", ""),
            "reference_count": count,
            "in_library": True,
        })

    return sources


# ---------------------------------------------------------------------------
# B5: flag_acquisition
# ---------------------------------------------------------------------------


async def handle_flag_acquisition(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
    embedding_provider: EmbeddingProvider,
    cache_manager: CacheManager | None = None,
) -> str:
    """Handle the flag_acquisition MCP tool call.

    Flags unresolved citations as acquisition candidates for the library.

    Arguments:
        citations (list): Each with citation_text, probable_work, priority, note.

    Returns:
        JSON with added count, already_flagged count, acquisition_list_total.
    """
    citations = arguments.get("citations")
    if not citations or not isinstance(citations, list):
        raise IngestionError(
            "citations list is required",
            context={"arguments": arguments},
        )

    from author_library.catalog.acquisition import AcquisitionManager

    manager = AcquisitionManager(storage.pg)

    added = 0
    already_flagged = 0

    for citation in citations:
        citation_text = citation.get("citation_text", "")
        if not citation_text:
            continue

        was_added = await manager.flag(
            citation_text=citation_text,
            probable_work=citation.get("probable_work"),
            priority=citation.get("priority", "medium"),
            note=citation.get("note"),
        )

        if was_added:
            added += 1
        else:
            already_flagged += 1

    total = await manager.count_total()

    result = {
        "added": added,
        "already_flagged": already_flagged,
        "acquisition_list_total": total,
    }

    log.info(
        "flag_acquisition_complete",
        added=added,
        already_flagged=already_flagged,
        total=total,
    )

    return json.dumps(result, indent=2)
