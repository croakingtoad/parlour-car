"""Ingestion pipeline orchestrator for The Author Library.

Coordinates the full pipeline: parse → classify → chunk → annotate →
embed → store → extract entities → create passage links.

Routes processing by source class:
  - PRIMARY: full enrichment (all steps)
  - SECONDARY: embeddings + attributed graph edges only
  - CONTEXTUAL: embeddings + cross-resource link targets
  - TERTIARY: metadata only, no content processing
  - PERSONAL: embeddings + USER_REFLECTS_ON graph edges, NO voice profile
  - REFERENCE: entities + passage links + connection surfacing, NO voice profile

Supports idempotent re-ingestion: deletes old chunks/embeddings for a
work before re-processing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from author_library.catalog.models import ProcessingRoute, SourceClass
from author_library.catalog.pipeline import ClassificationPipeline
from author_library.chunking import get_chunking_strategy
from author_library.chunking.annotator import AnnotationContext, ChunkAnnotator
from author_library.graph.entity_extraction import EntityExtractor
from author_library.graph.linking_explicit import ExplicitLinkDetector
from author_library.graph.linking_implicit import ImplicitEngagementDetector
from author_library.graph.linking_thematic import ThematicParallelDetector
from author_library.parsing import get_parser
from author_library.parsing.models import SectionType
from author_library.tools.chunk_persistence import canonicalize_stored_chunks

if TYPE_CHECKING:
    from uuid import UUID

    from author_library.catalog.models import CatalogEntry
    from author_library.chunking.models import Chunk
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
    from author_library.parsing.models import ParsedDocument
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


class IngestionResult:
    """Result of a single work ingestion."""

    __slots__ = (
        "chunks_by_granularity",
        "edge_count",
        "embeddings_stored",
        "entity_count",
        "errors",
        "processing_route",
        "quality_checks",
        "source_class",
        "total_chunks",
        "unembedded_chunk_ids",
        "work_id",
    )

    def __init__(
        self,
        *,
        work_id: str,
        source_class: str,
        processing_route: str,
        chunks_by_granularity: dict[str, int],
        embeddings_stored: int,
        entity_count: int,
        edge_count: int,
        errors: list[str],
        total_chunks: int = 0,
        unembedded_chunk_ids: list[str] | None = None,
        quality_checks: dict[str, Any] | None = None,
    ) -> None:
        self.work_id = work_id
        self.source_class = source_class
        self.processing_route = processing_route
        self.chunks_by_granularity = chunks_by_granularity
        self.embeddings_stored = embeddings_stored
        self.entity_count = entity_count
        self.edge_count = edge_count
        self.errors = errors
        self.total_chunks = total_chunks or sum(chunks_by_granularity.values())
        self.unembedded_chunk_ids = unembedded_chunk_ids or []
        self.quality_checks = quality_checks

    def to_dict(self) -> dict[str, Any]:
        total = self.total_chunks
        coverage = round(self.embeddings_stored / total * 100, 1) if total > 0 else 0.0
        result: dict[str, Any] = {
            "work_id": self.work_id,
            "source_class": self.source_class,
            "processing_route": self.processing_route,
            "chunks_by_granularity": self.chunks_by_granularity,
            "post_ingestion_stats": {
                "total_chunks": total,
                "embeddings_stored": self.embeddings_stored,
                "embedding_coverage_percent": coverage,
                "unembedded_chunks": len(self.unembedded_chunk_ids),
                "entity_count": self.entity_count,
                "edge_count": self.edge_count,
            },
            "errors": self.errors,
        }
        if self.unembedded_chunk_ids:
            result["post_ingestion_stats"]["unembedded_chunk_ids"] = self.unembedded_chunk_ids
            result["post_ingestion_stats"]["status"] = "incomplete — some chunks missing embeddings"
        else:
            result["post_ingestion_stats"]["status"] = "complete — all chunks embedded"
        if self.quality_checks is not None:
            result["quality_checks"] = self.quality_checks
        return result


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline for a single work.

    Coordinates parsing, classification, chunking, annotation, embedding,
    entity extraction, and passage linking according to source class routing.
    """

    _PASSAGE_LINK_BATCH_SIZE = 500

    def __init__(
        self,
        *,
        settings: Settings,
        storage: StorageManager,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._settings = settings
        self._storage = storage
        self._embedding = embedding_provider

    async def ingest(
        self,
        file_path: str | Path,
        *,
        subject_author_id: str,
        metadata_hints: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest a single work through the full pipeline.

        Args:
            file_path: Path to the document file.
            subject_author_id: The subject author's slug identifier.
            metadata_hints: Optional metadata overrides (source_class, genre_tags, etc.).

        Returns:
            IngestionResult with counts and any non-fatal errors.

        Raises:
            IngestionError: If a critical pipeline stage fails.
        """
        from author_library.parsing.models import ParsedDocument

        path = Path(file_path)

        log.info(
            "ingestion_starting",
            file_path=str(path),
            subject_author=subject_author_id,
        )

        # Step 1: Parse
        stage_start = time.monotonic()
        parser = get_parser(path)
        document = await parser.parse(str(path))
        log.info(
            "ingestion_parsed",
            title=document.metadata.title,
            word_count=document.metadata.word_count,
            format=document.format,
            elapsed_seconds=round(time.monotonic() - stage_start, 1),
        )

        # Delegate to ingest_document for steps 2+
        return await self.ingest_document(
            document,
            subject_author_id=subject_author_id,
            metadata_hints=metadata_hints,
        )

    async def ingest_document(
        self,
        document: ParsedDocument,
        *,
        subject_author_id: str,
        metadata_hints: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest a pre-built ParsedDocument through the pipeline.

        Skips the parsing step — runs classify → chunk → annotate → embed →
        extract entities → create passage links.

        Use this for custom document structures (e.g., epistolary collections,
        poetry collections with prose intros).

        Args:
            document: A pre-built ParsedDocument with the desired tree structure.
            subject_author_id: The subject author's slug identifier.
            metadata_hints: Optional metadata overrides (source_class, genre_tags, etc.).

        Returns:
            IngestionResult with counts and any non-fatal errors.

        Raises:
            IngestionError: If a critical pipeline stage fails.
        """
        from author_library.parsing.models import ParsedDocument

        hints = metadata_hints or {}
        errors: list[str] = []
        pipeline_start = time.monotonic()

        log.info(
            "ingest_document_starting",
            title=document.metadata.title,
            word_count=document.metadata.word_count,
            format=document.format,
            subject_author=subject_author_id,
        )

        # Step 2: Classify via pipeline (creates catalog entry + stores in works table)
        stage_start = time.monotonic()
        classification_pipeline = ClassificationPipeline(
            settings=self._settings,
            work_repository=self._storage.works,
            subject_author=subject_author_id,
            pg_pool=self._storage.pg,
            storage=self._storage,
        )

        pipeline_result = await classification_pipeline.process(
            document,
            metadata_hints=hints,
            user_overrides=hints,
        )

        catalog_entry = pipeline_result.catalog_entry
        work_id = catalog_entry.work_id
        source_class = pipeline_result.classification.source_class
        route = pipeline_result.processing_route

        log.info(
            "ingestion_classified",
            work_id=work_id,
            source_class=source_class.value,
            route=route.value,
            elapsed_seconds=round(time.monotonic() - stage_start, 1),
        )

        # Determine pass number: detect if source already has chunks
        current_max_pass = await self._storage.chunks.get_max_pass_number(work_id)
        pass_number = current_max_pass + 1 if current_max_pass > 0 else 1

        # Idempotent: delete existing chunks/embeddings for re-ingestion
        deleted_chunks = await self._storage.chunks.delete_by_work(work_id)
        if deleted_chunks > 0:
            log.info(
                "ingestion_cleared_old_data",
                work_id=work_id,
                deleted_chunks=deleted_chunks,
                new_pass_number=pass_number,
            )

        # Also clear Neo4j chunk nodes so stale UUIDs don't persist.
        # On re-ingestion (deleted_chunks > 0), this is mandatory — PG just
        # assigned new UUIDs, so any failure here would leave two disjoint
        # UUID sets in Neo4j permanently. Abort rather than silently corrupt.
        try:
            deleted_graph = await self._storage.graph.delete_chunks_for_work(work_id)
            if deleted_graph > 0:
                log.info(
                    "ingestion_cleared_graph_chunks",
                    work_id=work_id,
                    deleted_graph_chunks=deleted_graph,
                )
        except Exception as exc:
            if deleted_chunks > 0:
                raise RuntimeError(
                    f"Neo4j chunk cleanup failed for re-ingestion of {work_id!r} — "
                    f"PG chunks were already deleted and new UUIDs assigned. "
                    f"Aborting to prevent stale Neo4j nodes. Fix Neo4j and retry. "
                    f"Original error: {exc}"
                ) from exc
            log.warning(
                "neo4j_chunk_cleanup_failed",
                work_id=work_id,
                error=str(exc),
                message="First ingestion — no stale nodes possible, pipeline continues",
            )

        # Upsert work node in Neo4j
        try:
            await self._storage.graph.upsert_work_node({
                "work_id": work_id,
                "title": catalog_entry.title,
                "author": catalog_entry.author,
                "source_class": source_class.value,
                "publication_year": catalog_entry.publication_year,
            })
        except Exception as exc:
            log.warning(
                "neo4j_work_upsert_failed",
                work_id=work_id,
                error=str(exc),
                message="Pipeline continues without Neo4j work node",
            )

        # Upsert author record in PG (authors table)
        await self._storage.pg.execute(
            "INSERT INTO authors (id, canonical_name) VALUES ($1, $2) "
            "ON CONFLICT (id) DO NOTHING",
            subject_author_id,
            catalog_entry.author,
        )

        # Upsert Author node in Neo4j and create AUTHORED->Work edge
        try:
            await self._storage.neo4j.execute_write(
                """MERGE (a:Author {author_id: $author_id})
                SET a.canonical_name = $name
                WITH a
                MATCH (w:Work {work_id: $work_id})
                MERGE (a)-[:AUTHORED]->(w)""",
                {
                    "author_id": subject_author_id,
                    "name": catalog_entry.author,
                    "work_id": work_id,
                },
            )
        except Exception as exc:
            log.warning(
                "neo4j_author_upsert_failed",
                author_id=subject_author_id,
                work_id=work_id,
                error=str(exc),
                message="Pipeline continues without Neo4j author node",
            )

        log.info(
            "ingestion_author_upserted",
            author_id=subject_author_id,
            canonical_name=catalog_entry.author,
            work_id=work_id,
        )

        # Step 3: Route by source class
        if route == ProcessingRoute.METADATA_ONLY:
            log.info("ingestion_tertiary_metadata_only", work_id=work_id)
            return IngestionResult(
                work_id=work_id,
                source_class=source_class.value,
                processing_route=route.value,
                chunks_by_granularity={},
                embeddings_stored=0,
                entity_count=0,
                edge_count=0,
                errors=errors,
            )

        # Step 4: Chunk
        stage_start = time.monotonic()
        genre_tags = catalog_entry.genre_tags
        strategy = get_chunking_strategy(genre_tags)
        chunks = strategy.chunk(document, work_id, source_class.value)

        chunks_by_gran: dict[str, int] = {}
        for chunk in chunks:
            gran = str(chunk.granularity)
            chunks_by_gran[gran] = chunks_by_gran.get(gran, 0) + 1

        log.info(
            "ingestion_chunked",
            work_id=work_id,
            total_chunks=len(chunks),
            by_granularity=chunks_by_gran,
            elapsed_seconds=round(time.monotonic() - stage_start, 1),
        )

        # Step 4b: Section-type routing — filter out non-content sections,
        # then route structural sections to vocabulary/acquisition managers.
        chunks, skipped_sections, structural_chunks = self._filter_by_section_type(chunks, work_id)

        # Recount after filtering
        if skipped_sections:
            chunks_by_gran = {}
            for chunk in chunks:
                gran = str(chunk.granularity)
                chunks_by_gran[gran] = chunks_by_gran.get(gran, 0) + 1

        # Route structural sections: index → vocabulary proposals,
        # bibliography → acquisition candidates.
        if structural_chunks:
            await self._route_structural_sections(structural_chunks, work_id)

        # Step 5: Annotate
        stage_start = time.monotonic()
        annotation_ctx = self._build_annotation_context(catalog_entry, source_class)
        annotator = ChunkAnnotator(self._settings)
        chunks = await annotator.annotate_chunks(chunks, annotation_ctx)
        log.info(
            "ingestion_annotated",
            work_id=work_id,
            chunks=len(chunks),
            elapsed_seconds=round(time.monotonic() - stage_start, 1),
        )

        # Step 6: Store chunks in PG
        stage_start = time.monotonic()
        # Sort so parents (macro) are inserted before children (meso) before
        # grandchildren (micro), satisfying the parent_chunk_id foreign key.
        _gran_order = {"macro": 0, "meso": 1, "micro": 2, "nano": 3}
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (_gran_order.get(str(c.granularity), 9), c.position),
        )

        provisional_to_pg_id: dict[str, UUID] = {}
        pg_chunk_errors = 0
        for chunk in sorted_chunks:
            # Translate in-memory parent_chunk_id to DB-generated UUID
            resolved_parent: UUID | None = None
            if chunk.parent_chunk_id is not None:
                resolved_parent = provisional_to_pg_id.get(chunk.parent_chunk_id)

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
                pg_id = await self._storage.chunks.create(chunk_data)
                provisional_to_pg_id[chunk.id] = pg_id
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
                stored=len(provisional_to_pg_id),
                failed=pg_chunk_errors,
                total=len(sorted_chunks),
            )
            errors.append(
                f"PG chunk storage: {pg_chunk_errors}/{len(sorted_chunks)} chunks failed"
            )

        chunks, chunk_id_map = canonicalize_stored_chunks(
            chunks,
            provisional_to_pg_id,
        )

        log.info(
            "ingestion_chunks_stored",
            work_id=work_id,
            count=len(chunks),
            elapsed_seconds=round(time.monotonic() - stage_start, 1),
        )

        # Update engagement_passes on the work record
        await self._storage.works.update(work_id, {"engagement_passes": pass_number})

        # Step 7: Embed chunks (use annotated_text for embedding)
        # Build token-aware batches so no single API call exceeds provider limits
        embeddings_stored = 0
        from author_library.embeddings.base import (
            build_token_aware_batches,
            estimate_tokens,
        )

        all_texts = [c.annotated_text for c in chunks]
        token_batches = build_token_aware_batches(all_texts)

        # Map batch indices back to chunk objects
        chunk_offset = 0
        embed_start = time.monotonic()
        total_batches = len(token_batches)

        for batch_idx, batch_texts in enumerate(token_batches):
            batch_chunks = chunks[chunk_offset : chunk_offset + len(batch_texts)]
            chunk_offset += len(batch_texts)

            est_tokens = sum(estimate_tokens(t) for t in batch_texts)
            log.info(
                "ingestion_embedding_batch",
                work_id=work_id,
                batch=batch_idx + 1,
                total_batches=total_batches,
                texts=len(batch_texts),
                estimated_tokens=est_tokens,
            )

            stored = await self._embed_batch_with_retry(
                batch_texts, batch_chunks, chunk_id_map,
                batch_idx + 1, total_batches, work_id, errors,
            )
            embeddings_stored += stored

            # Warn if embedding stage is running long
            elapsed = time.monotonic() - embed_start
            if elapsed > 60:
                log.warning(
                    "ingestion_embedding_slow",
                    work_id=work_id,
                    elapsed_seconds=round(elapsed, 1),
                    batches_complete=batch_idx + 1,
                    total_batches=total_batches,
                )

        # Identify unembedded chunks
        embedded_chunk_ids = set()
        for chunk in chunks:
            pg_id = chunk_id_map.get(chunk.id)
            if pg_id is not None:
                embedded_chunk_ids.add(chunk.id)
        # Chunks that were stored but didn't get embedded
        all_chunk_ids = {c.id for c in chunks if chunk_id_map.get(c.id) is not None}
        # We track by count: if embeddings_stored < len(all_chunk_ids), some are missing
        unembedded_chunk_ids: list[str] = []
        if embeddings_stored < len(all_chunk_ids):
            # Query DB for chunks missing embeddings
            try:
                pg_ids = list(chunk_id_map.values())
                embedded_pg_ids = set()
                for pg_id in pg_ids:
                    has_embed = await self._storage.embeddings.exists(pg_id)
                    if has_embed:
                        embedded_pg_ids.add(pg_id)
                for chunk in chunks:
                    pg_id = chunk_id_map.get(chunk.id)
                    if pg_id and pg_id not in embedded_pg_ids:
                        unembedded_chunk_ids.append(chunk.id)
            except Exception:
                # If we can't check, estimate from counts
                unembedded_chunk_ids = [f"~{len(all_chunk_ids) - embeddings_stored}_chunks_unverified"]

        embed_elapsed = round(time.monotonic() - embed_start, 1)
        log.info(
            "ingestion_embeddings_stored",
            work_id=work_id,
            count=embeddings_stored,
            total_chunks=len(all_chunk_ids),
            coverage_percent=round(embeddings_stored / max(len(all_chunk_ids), 1) * 100, 1),
            unembedded=len(unembedded_chunk_ids),
            elapsed_seconds=embed_elapsed,
        )

        # Step 8: Upsert chunk nodes in Neo4j
        for chunk in chunks:
            chunk_node: dict[str, Any] = {
                "chunk_id": chunk.id,
                "work_id": chunk.work_id,
                "text_preview": chunk.text[:200],
                "granularity": str(chunk.granularity),
                "source_class": chunk.source_class,
            }
            # Personal chunks carry user_id for USER_REFLECTS_ON edges
            if source_class == SourceClass.PERSONAL:
                chunk_node["user_id"] = getattr(catalog_entry, "user_id", "marty")
            await self._storage.graph.upsert_chunk_node(chunk_node)

        # Step 9: Entity extraction (PRIMARY, SECONDARY, and REFERENCE)
        # Filter to configured granularities (default: macro+meso) to skip
        # redundant extraction on micro/nano chunks whose parents already
        # capture the same entities.
        entity_count = 0
        edge_count = 0
        if route in (
            ProcessingRoute.FULL_ENRICHMENT,
            ProcessingRoute.EMBEDDINGS_AND_GRAPH,
            ProcessingRoute.REFERENCE_ENRICHMENT,
        ):
            allowed_grans = {
                g.strip()
                for g in self._settings.llm.entity_extraction_granularities.split(",")
            }
            # Structural sections are excluded from entity extraction.
            # These are already filtered by _filter_by_section_type, but we
            # apply the check here as well for defense-in-depth.
            _ENTITY_EXTRACT_EXCLUDED = {
                SectionType.BIBLIOGRAPHY.value,
                SectionType.INDEX.value,
                SectionType.TABLE_OF_CONTENTS.value,
                SectionType.FRONT_MATTER.value,
            }
            extraction_chunks = [
                c
                for c in chunks
                if str(c.granularity) in allowed_grans
                and c.section_type not in _ENTITY_EXTRACT_EXCLUDED
            ]
            skipped = len(chunks) - len(extraction_chunks)
            if skipped:
                log.info(
                    "entity_extraction_granularity_filter",
                    total_chunks=len(chunks),
                    extraction_chunks=len(extraction_chunks),
                    skipped_chunks=skipped,
                    allowed_granularities=sorted(allowed_grans),
                )

            try:
                extractor = EntityExtractor(
                    self._storage.neo4j,
                    self._settings.api_keys,
                    self._settings.llm,
                    storage=self._storage,
                )
                extraction_result = await extractor.extract_and_persist(
                    extraction_chunks,
                    work_title=catalog_entry.title,
                    author=catalog_entry.author,
                )
                entity_count = extraction_result.nodes_created
                edge_count = extraction_result.edges_created
                if extraction_result.errors:
                    errors.extend(extraction_result.errors)
            except Exception as exc:
                error_msg = f"Entity extraction failed: {exc}"
                log.error("ingestion_entity_extraction_failed", error=error_msg)
                errors.append(error_msg)

            # Step 9b: Theme deduplication — merge near-duplicate Theme nodes
            # created by independent LLM calls across chunks.
            try:
                from author_library.graph.theme_dedup import deduplicate_themes

                dedup_result = await deduplicate_themes(
                    self._storage.neo4j,
                    self._embedding,
                    work_id=work_id,
                )
                if dedup_result.merged_count > 0:
                    log.info(
                        "ingestion_theme_dedup",
                        work_id=work_id,
                        original_themes=dedup_result.original_count,
                        canonical_themes=dedup_result.canonical_count,
                        merged=dedup_result.merged_count,
                    )
                if dedup_result.errors:
                    errors.extend(dedup_result.errors)
            except Exception as exc:
                error_msg = f"Theme deduplication failed: {exc}"
                log.error("ingestion_theme_dedup_failed", error=error_msg)
                errors.append(error_msg)

        # Step 10: Passage linking (PRIMARY, CONTEXTUAL, and REFERENCE)
        if route in (
            ProcessingRoute.FULL_ENRICHMENT,
            ProcessingRoute.EMBEDDINGS_AND_LINKS,
            ProcessingRoute.REFERENCE_ENRICHMENT,
        ):
            try:
                link_edges = await self._create_passage_links(chunks, source_class)
                edge_count += link_edges
            except Exception as exc:
                error_msg = f"Passage linking failed: {exc}"
                log.error("ingestion_passage_linking_failed", error=error_msg)
                errors.append(error_msg)

        # Step 11: Personal source — create USER_REFLECTS_ON edges
        # Personal chunks are stored with embeddings but do NOT get entity
        # extraction or passage linking. Instead they create USER_REFLECTS_ON
        # edges to targets (captures/themes) they reference.
        if route == ProcessingRoute.PERSONAL_ENRICHMENT:
            log.info(
                "ingestion_personal_route",
                work_id=work_id,
                chunks=len(chunks),
                note="Personal source: embeddings + graph nodes created, "
                "voice profile skipped, entity extraction skipped",
            )

        # Step 12: Post-ingestion connection surfacing (N1/N3)
        # After passage links are created, scan for new cross-work
        # connections and generate PR content for user review.
        surfacing_result = None
        if route in (
            ProcessingRoute.FULL_ENRICHMENT,
            ProcessingRoute.EMBEDDINGS_AND_LINKS,
            ProcessingRoute.REFERENCE_ENRICHMENT,
        ):
            try:
                surfacing_result = await self._surface_connections(
                    work_id=work_id,
                    work_title=catalog_entry.title,
                    work_author=catalog_entry.author,
                )
            except Exception as exc:
                error_msg = f"Post-ingestion surfacing failed: {exc}"
                log.error("ingestion_surfacing_failed", error=error_msg)
                errors.append(error_msg)

        # Step 13: Post-ingestion quality checks
        quality_checks = None
        try:
            quality_checks = await self._run_quality_checks(
                work_id, source_class.value, subject_author_id,
            )
            if quality_checks.get("warnings"):
                errors.extend(
                    f"[quality] {w}" for w in quality_checks["warnings"]
                )
        except Exception as exc:
            error_msg = f"Quality checks failed: {exc}"
            log.error("ingestion_quality_checks_failed", error=error_msg)
            errors.append(error_msg)

        total_elapsed = round(time.monotonic() - pipeline_start, 1)
        log.info(
            "ingestion_complete",
            work_id=work_id,
            source_class=source_class.value,
            route=route.value,
            chunks=len(chunks),
            embeddings=embeddings_stored,
            entities=entity_count,
            edges=edge_count,
            surfacing_connections=surfacing_result.scan_result.total_found
            if surfacing_result and surfacing_result.scan_result
            else 0,
            quality_status=quality_checks["status"] if quality_checks else "skipped",
            errors=len(errors),
            total_elapsed_seconds=total_elapsed,
        )
        if total_elapsed > 120:
            log.warning(
                "ingestion_slow_pipeline",
                work_id=work_id,
                total_elapsed_seconds=total_elapsed,
                hint="Consider using ingest_book_async for large works",
            )

        return IngestionResult(
            work_id=work_id,
            source_class=source_class.value,
            processing_route=route.value,
            chunks_by_granularity=chunks_by_gran,
            embeddings_stored=embeddings_stored,
            entity_count=entity_count,
            edge_count=edge_count,
            errors=errors,
            total_chunks=len(chunks),
            unembedded_chunk_ids=unembedded_chunk_ids,
            quality_checks=quality_checks,
        )

    async def _surface_connections(
        self,
        *,
        work_id: str,
        work_title: str = "",
        work_author: str = "",
    ) -> Any:
        """Run post-ingestion connection surfacing (N1/N3).

        Scans for new cross-work connections and generates PR content.
        Returns the BatchSurfacingResult for logging purposes.
        """
        from author_library.surfacing.batch_surfacing import BatchSurfacer

        surfacer = BatchSurfacer(
            settings=self._settings,
            storage=self._storage,
            embedding_provider=self._embedding,
        )

        result = await surfacer.surface_after_ingestion(
            work_id,
            work_title=work_title,
            work_author=work_author,
        )

        if result.scan_result and result.scan_result.total_found > 0:
            log.info(
                "ingestion_surfacing_complete",
                work_id=work_id,
                connections_found=result.scan_result.total_found,
                pr_content_generated=result.pr_content is not None,
            )

        return result

    async def _run_quality_checks(
        self,
        work_id: str,
        source_class: str,
        subject_author_id: str,
    ) -> dict[str, Any]:
        """Run post-ingestion quality checks for a work.

        Checks:
            1. Orphaned entity nodes (Argument/Theme/Concept/Person with zero rels)
            2. Classification sanity (author's own work classified as contextual/tertiary)
            3. Chunk noise (micro/nano chunks < 50 chars)
            4. Embedding coverage (chunks vs embeddings)
            5. Entity coverage (chunks with zero entity edges)

        Returns:
            Dict with orphans_cleaned, classification_warning, noise_chunks,
            embedding_coverage_pct, entity_coverage_pct, and status.
        """
        checks: dict[str, Any] = {
            "orphans_cleaned": 0,
            "classification_warning": None,
            "noise_chunks": 0,
            "embedding_coverage_pct": 100.0,
            "entity_coverage_pct": 100.0,
            "status": "pass",
        }
        warnings: list[str] = []

        # --- Check 1: Orphaned entity nodes ---
        # Find Argument/Theme/Concept/Person nodes created from this work's
        # chunks that have no relationships other than the extraction edge
        # from the chunk.  "Orphaned" means the entity node has degree <= 1
        # (only the single MENTIONS/EXPLORES_THEME edge from the chunk).
        try:
            orphan_query = """
                MATCH (c:Chunk {work_id: $work_id})-[r]->(e)
                WHERE e:Argument OR e:Theme OR e:Concept OR e:Person
                WITH e, count{ (e)--() } AS degree
                WHERE degree <= 1
                DETACH DELETE e
                RETURN count(e) AS deleted
            """
            result = await self._storage.neo4j.execute_write(
                orphan_query, {"work_id": work_id}
            )
            orphans = result[0]["deleted"] if result else 0
            checks["orphans_cleaned"] = orphans
            if orphans > 0:
                log.info(
                    "quality_check_orphans_cleaned",
                    work_id=work_id,
                    orphans_deleted=orphans,
                )
                try:
                    from author_library.intelligence.lesson_writer import record_lesson
                    await record_lesson(
                        self._storage,
                        problem_type="orphan_nodes",
                        detection_method="qg1_inline",
                        trigger_context={"work_id": work_id, "source_class": source_class},
                        problem_description=(
                            f"{orphans} orphaned entity nodes found for work '{work_id}' "
                            f"(nodes with degree <= 1 after entity extraction)."
                        ),
                        fix_applied=f"DETACH DELETE {orphans} orphaned entity nodes.",
                        prevention_rule=(
                            "Entity extraction should yield connected nodes; "
                            "lone entities with no cross-chunk relationships are noise."
                        ),
                        prevention_step="entity_extraction",
                    )
                except Exception as _lesson_exc:
                    log.warning("qg1_lesson_write_failed", problem_type="orphan_nodes", error=str(_lesson_exc))
        except Exception as exc:
            log.error("quality_check_orphans_failed", work_id=work_id, error=str(exc))
            warnings.append(f"Orphan check failed: {exc}")

        # --- Check 2: Classification sanity ---
        # If the work's author matches subject_author_id and source_class
        # is contextual or tertiary, that's suspicious — primary works by
        # the subject author should not be classified as contextual/tertiary.
        # Reference is deliberately excluded: it is filed under its own author.
        try:
            work_record = await self._storage.works.get(work_id)
            if work_record:
                import re as _re

                def _normalize_name(name: str) -> str:
                    """Strip punctuation and collapse whitespace for fuzzy comparison."""
                    return _re.sub(r"[^a-z]+", " ", name.lower()).strip()

                work_author_norm = _normalize_name(work_record.get("author") or "")
                author_slug_norm = _normalize_name(subject_author_id)
                author_matches = (
                    author_slug_norm in work_author_norm
                    or work_author_norm in author_slug_norm
                )
                if author_matches and source_class in ("contextual", "tertiary"):
                    warning = (
                        f"Work author '{work_record.get('author')}' appears to match "
                        f"subject author '{subject_author_id}' but is classified as "
                        f"'{source_class}'. Consider reclassifying as 'primary'."
                    )
                    checks["classification_warning"] = warning
                    warnings.append(warning)
                    log.warning(
                        "quality_check_classification_suspect",
                        work_id=work_id,
                        work_author=work_record.get("author"),
                        subject_author_id=subject_author_id,
                        source_class=source_class,
                    )
                    try:
                        from author_library.intelligence.lesson_writer import record_lesson
                        await record_lesson(
                            self._storage,
                            problem_type="misclassification",
                            detection_method="qg1_inline",
                            trigger_context={
                                "work_id": work_id,
                                "source_class": source_class,
                                "subject_author_id": subject_author_id,
                            },
                            problem_description=(
                                f"Work '{work_id}' by '{work_record.get('author')}' "
                                f"classified as '{source_class}' but author matches "
                                f"subject_author_id '{subject_author_id}'."
                            ),
                            fix_applied="Flagged for manual reclassification review.",
                            prevention_rule=(
                                "When author name matches subject_author_id slug, "
                                "default classification should be 'primary' not "
                                "'contextual' or 'tertiary'."
                            ),
                            prevention_step="classification",
                        )
                    except Exception as _lesson_exc:
                        log.warning("qg1_lesson_write_failed", problem_type="misclassification", error=str(_lesson_exc))
        except Exception as exc:
            log.error("quality_check_classification_failed", work_id=work_id, error=str(exc))
            warnings.append(f"Classification check failed: {exc}")

        # --- Check 3: Chunk noise ---
        # Query PG for chunks belonging to this work with text < 50 chars.
        try:
            noise_count = await self._storage.pg.fetch_val(
                "SELECT COUNT(*) FROM chunks WHERE work_id = $1 AND LENGTH(text) < 50",
                work_id,
            )
            checks["noise_chunks"] = int(noise_count)
            if noise_count > 0:
                log.warning(
                    "quality_check_noise_chunks",
                    work_id=work_id,
                    noise_chunks=int(noise_count),
                )
                warnings.append(f"{noise_count} chunks with text < 50 chars")
                try:
                    from author_library.intelligence.lesson_writer import record_lesson
                    # Get genre_tags for richer context
                    work_rec = await self._storage.works.get(work_id)
                    genre_tags = (work_rec or {}).get("genre_tags", []) if work_rec else []
                    await record_lesson(
                        self._storage,
                        problem_type="chunk_noise",
                        detection_method="qg1_inline",
                        trigger_context={
                            "work_id": work_id,
                            "source_class": source_class,
                            "genre_tags": genre_tags,
                        },
                        problem_description=(
                            f"{noise_count} chunks with text < 50 chars in work '{work_id}'."
                        ),
                        fix_applied="Noise chunks detected and flagged; not deleted.",
                        prevention_rule=(
                            "Consider raising the minimum chunk size filter from 50 "
                            "to eliminate noise during chunking rather than post-hoc."
                        ),
                        prevention_step="chunking",
                    )
                except Exception as _lesson_exc:
                    log.warning("qg1_lesson_write_failed", problem_type="chunk_noise", error=str(_lesson_exc))
        except Exception as exc:
            log.error("quality_check_noise_failed", work_id=work_id, error=str(exc))
            warnings.append(f"Noise check failed: {exc}")

        # --- Check 4: Embedding coverage ---
        # Compare chunk count to embedding count for this work.
        try:
            total_chunks = await self._storage.pg.fetch_val(
                "SELECT COUNT(*) FROM chunks WHERE work_id = $1",
                work_id,
            )
            total_embeddings = await self._storage.pg.fetch_val(
                """SELECT COUNT(*) FROM chunk_embeddings ce
                   JOIN chunks c ON c.id = ce.chunk_id
                   WHERE c.work_id = $1""",
                work_id,
            )
            total_chunks = int(total_chunks)
            total_embeddings = int(total_embeddings)
            if total_chunks > 0:
                coverage = round(total_embeddings / total_chunks * 100, 1)
            else:
                coverage = 100.0
            checks["embedding_coverage_pct"] = coverage
            if coverage < 100.0:
                log.warning(
                    "quality_check_embedding_gap",
                    work_id=work_id,
                    total_chunks=total_chunks,
                    total_embeddings=total_embeddings,
                    coverage_pct=coverage,
                )
                warnings.append(
                    f"Embedding coverage {coverage}% "
                    f"({total_embeddings}/{total_chunks} chunks)"
                )
        except Exception as exc:
            log.error("quality_check_embeddings_failed", work_id=work_id, error=str(exc))
            warnings.append(f"Embedding coverage check failed: {exc}")

        # --- Check 5: Entity coverage ---
        # Query Neo4j for chunks from this work that have zero entity edges.
        # Warn if more than 10% of chunks lack entities.
        try:
            entity_coverage_result = await self._storage.neo4j.execute_read(
                """MATCH (c:Chunk {work_id: $work_id})
                   OPTIONAL MATCH (c)-[:EXPLORES_THEME|MAKES_ARGUMENT|ATTRIBUTED_BY_CRITIC|CONCEPT_USED_IN|REFERENCES_PERSON]->(e)
                   WITH c, count(e) AS entity_count
                   RETURN
                     count(c) AS total_chunks,
                     sum(CASE WHEN entity_count = 0 THEN 1 ELSE 0 END) AS no_entity_chunks
                """,
                {"work_id": work_id},
            )
            if entity_coverage_result:
                row = entity_coverage_result[0]
                neo4j_total = row.get("total_chunks", 0)
                no_entity = row.get("no_entity_chunks", 0)
                if neo4j_total > 0:
                    entity_pct = round((neo4j_total - no_entity) / neo4j_total * 100, 1)
                else:
                    entity_pct = 100.0
                checks["entity_coverage_pct"] = entity_pct
                if entity_pct < 90.0:
                    log.warning(
                        "quality_check_entity_gap",
                        work_id=work_id,
                        total_chunks=neo4j_total,
                        no_entity_chunks=no_entity,
                        coverage_pct=entity_pct,
                    )
                    warnings.append(
                        f"Entity coverage {entity_pct}% "
                        f"({no_entity}/{neo4j_total} chunks without entities)"
                    )
        except Exception as exc:
            log.error("quality_check_entity_coverage_failed", work_id=work_id, error=str(exc))
            warnings.append(f"Entity coverage check failed: {exc}")

        # Determine overall status
        if warnings:
            checks["status"] = "warnings"
            checks["warnings"] = warnings
        else:
            checks["status"] = "pass"

        log.info(
            "quality_checks_complete",
            work_id=work_id,
            status=checks["status"],
            orphans_cleaned=checks["orphans_cleaned"],
            noise_chunks=checks["noise_chunks"],
            embedding_coverage_pct=checks["embedding_coverage_pct"],
            entity_coverage_pct=checks["entity_coverage_pct"],
            has_classification_warning=checks["classification_warning"] is not None,
        )

        return checks

    async def _embed_batch_with_retry(
        self,
        texts: list[str],
        batch_chunks: list,
        chunk_id_map: dict,
        batch_num: int,
        total_batches: int,
        work_id: str,
        errors: list[str],
        *,
        _depth: int = 0,
    ) -> int:
        """Embed a batch of texts with retry and split-on-failure logic.

        If a batch fails, splits it in half and retries each half recursively.
        This ensures that a single oversized chunk doesn't cause the entire
        batch to be permanently skipped.

        Returns the number of embeddings successfully stored.
        """
        import asyncio as _asyncio

        max_retries = 2
        stored = 0

        for attempt in range(max_retries + 1):
            try:
                batch_result = await self._embedding.embed_batch(texts)
                pg_store_errors = 0
                for chunk, vector in zip(
                    batch_chunks, batch_result.vectors, strict=True
                ):
                    maybe_id = chunk_id_map.get(chunk.id)
                    if maybe_id is None:
                        continue
                    pg_id = maybe_id
                    try:
                        await self._storage.embeddings.store(
                            pg_id,
                            vector,
                            self._embedding.provider_name,
                            self._embedding.model_name,
                            self._embedding.dimensions,
                        )
                        stored += 1
                    except Exception as store_exc:
                        try:
                            await _asyncio.sleep(0.5)
                            await self._storage.embeddings.store(
                                pg_id,
                                vector,
                                self._embedding.provider_name,
                                self._embedding.model_name,
                                self._embedding.dimensions,
                            )
                            stored += 1
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
                        f"PG embedding store: {pg_store_errors} chunks failed in batch {batch_num}"
                    )
                return stored
            except Exception as exc:
                if attempt < max_retries:
                    wait = 2 ** (attempt + 1)
                    log.warning(
                        "ingestion_embedding_retry",
                        work_id=work_id,
                        batch=batch_num,
                        attempt=attempt + 1,
                        wait_seconds=wait,
                        error=str(exc),
                    )
                    await _asyncio.sleep(wait)
                    continue

                # All retries exhausted — split in half and try each half
                if len(texts) > 1 and _depth < 3:
                    mid = len(texts) // 2
                    log.warning(
                        "ingestion_embedding_split",
                        work_id=work_id,
                        batch=batch_num,
                        original_size=len(texts),
                        split_sizes=[mid, len(texts) - mid],
                        depth=_depth + 1,
                    )
                    left = await self._embed_batch_with_retry(
                        texts[:mid], batch_chunks[:mid], chunk_id_map,
                        batch_num, total_batches, work_id, errors,
                        _depth=_depth + 1,
                    )
                    right = await self._embed_batch_with_retry(
                        texts[mid:], batch_chunks[mid:], chunk_id_map,
                        batch_num, total_batches, work_id, errors,
                        _depth=_depth + 1,
                    )
                    return left + right

                # Single chunk or max depth — truly failed
                failed_ids = [c.id for c in batch_chunks if chunk_id_map.get(c.id)]
                error_msg = (
                    f"Embedding batch {batch_num}/{total_batches} "
                    f"({len(texts)} texts) permanently failed after "
                    f"{max_retries} retries + split: {exc}. "
                    f"Unembedded chunk IDs: {failed_ids}"
                )
                log.error("ingestion_embedding_permanent_failure", error=error_msg,
                          chunk_ids=failed_ids, work_id=work_id)
                errors.append(error_msg)
                return 0

        return stored

    async def _load_chunks_paginated(
        self,
        work_ids: list[str],
        granularity: str = "meso",
        default_source_class: str = "primary",
    ) -> list[Any]:
        """Load chunks from multiple works using LIMIT/OFFSET pagination.

        Returns Chunk model objects suitable for passage linking.
        """
        from author_library.chunking.models import Chunk as ChunkModel

        result_chunks: list[Any] = []
        batch_size = self._PASSAGE_LINK_BATCH_SIZE

        for wid in work_ids:
            offset = 0
            while True:
                batch = await self._storage.chunks.list_by_work_paginated(
                    wid, limit=batch_size, offset=offset, granularity=granularity,
                )
                if not batch:
                    break
                for c in batch:
                    result_chunks.append(
                        ChunkModel(
                            id=str(c.get("id", "")),
                            text=c.get("text", ""),
                            annotation=c.get("annotation"),
                            granularity=c.get("granularity", granularity),
                            work_id=c.get("work_id", ""),
                            source_class=c.get("source_class", default_source_class),
                            position=c.get("position", 0),
                        )
                    )
                if len(batch) < batch_size:
                    break
                offset += batch_size

        return result_chunks

    async def _create_passage_links(
        self,
        chunks: list[Chunk],
        source_class: SourceClass,
    ) -> int:
        """Create cross-resource passage links for the work's chunks.

        Loads contextual chunks from the database using paginated queries
        to avoid loading all cross-work chunks at once.
        Returns the total number of edges created.
        """
        edges_created = 0
        primary_chunks = [c for c in chunks if c.source_class == "primary"]
        contextual_chunks = [c for c in chunks if c.source_class == "contextual"]
        reference_chunks = [c for c in chunks if c.source_class == "reference"]

        # For primary sources, we need to load existing contextual chunks to link against
        if source_class == SourceClass.PRIMARY and primary_chunks:
            all_works = await self._storage.works.list_by_author(
                primary_chunks[0].work_id.split("--")[0]
            )
            ctx_work_ids = [
                w["work_id"] for w in all_works if w.get("source_class") == "contextual"
            ]

            contextual_for_linking = await self._load_chunks_paginated(
                ctx_work_ids, granularity="meso", default_source_class="contextual",
            )

            if contextual_for_linking:
                # Tier 1: Explicit citations
                explicit = ExplicitLinkDetector(self._storage.neo4j)
                explicit_result = await explicit.detect_and_link(
                    primary_chunks, contextual_for_linking
                )
                edges_created += explicit_result.edges_created

                # Tier 2: Implicit engagement
                existing_links = {
                    (link.source_chunk_id, link.target_chunk_id)
                    for link in explicit_result.links
                }
                implicit = ImplicitEngagementDetector(self._storage.neo4j)
                implicit_result = await implicit.detect_and_link(
                    primary_chunks,
                    contextual_for_linking,
                    existing_links=existing_links,
                )
                edges_created += implicit_result.edges_created

                # Tier 3: Thematic parallels
                thematic = ThematicParallelDetector(self._storage.neo4j, self._embedding)
                thematic_result = await thematic.detect_and_link(
                    primary_chunks, contextual_for_linking
                )
                edges_created += thematic_result.edges_created

        # Contextual sources link within their subject-author corpus. References
        # have no author relationship, so they link against all primary works.
        elif source_class in (SourceClass.CONTEXTUAL, SourceClass.REFERENCE):
            link_chunks = (
                contextual_chunks
                if source_class == SourceClass.CONTEXTUAL
                else reference_chunks
            )
            if not link_chunks:
                return edges_created
            if source_class == SourceClass.REFERENCE:
                primary_rows = await self._storage.pg.fetch_all(
                    "SELECT work_id FROM works WHERE source_class = 'primary'"
                )
                prim_work_ids = [row["work_id"] for row in primary_rows]
            else:
                all_works = await self._storage.works.list_by_author(
                    link_chunks[0].work_id.split("--")[0]
                )
                prim_work_ids = [
                    w["work_id"] for w in all_works if w.get("source_class") == "primary"
                ]

            primary_for_linking = await self._load_chunks_paginated(
                prim_work_ids, granularity="meso", default_source_class="primary",
            )

            if primary_for_linking:
                explicit = ExplicitLinkDetector(self._storage.neo4j)
                explicit_result = await explicit.detect_and_link(
                    primary_for_linking, link_chunks
                )
                edges_created += explicit_result.edges_created

        log.info("passage_links_created", edges=edges_created)
        return edges_created

    def _filter_by_section_type(
        self,
        chunks: list[Chunk],
        work_id: str,
    ) -> tuple[list[Chunk], dict[str, int], dict[str, list[Chunk]]]:
        """Filter chunks by section type, removing non-content sections.

        Section types that get FULL pipeline processing:
        - chapter, back_matter: full pipeline
        Section types that are EXCLUDED from main pipeline but routed elsewhere:
        - index: route to vocabulary proposals (propose terms via VocabularyManager)
        - bibliography: route to acquisition candidates (flag via AcquisitionManager)
        - toc: structural metadata only (no additional routing)
        - front_matter: structural metadata only (no additional routing)
        - preface: kept in full pipeline (author voice, personal reflection content)

        Returns:
            Tuple of (content_chunks, skipped_counts_by_type, structural_chunks_by_type).
            structural_chunks_by_type only contains section types with routing logic
            (currently: index, bibliography).
        """
        # Section types that receive full pipeline processing
        _CONTENT_SECTION_TYPES = {
            SectionType.CHAPTER.value,
            SectionType.PREFACE.value,
            SectionType.BACK_MATTER.value,
        }

        # Section types that have downstream routing (index → vocab, bibliography → acquisition)
        _ROUTABLE_SECTION_TYPES = {
            SectionType.INDEX.value,
            SectionType.BIBLIOGRAPHY.value,
        }

        content_chunks: list[Chunk] = []
        skipped: dict[str, int] = {}
        structural_chunks: dict[str, list[Chunk]] = {}

        for chunk in chunks:
            if chunk.section_type in _CONTENT_SECTION_TYPES:
                content_chunks.append(chunk)
            else:
                skipped[chunk.section_type] = skipped.get(chunk.section_type, 0) + 1
                if chunk.section_type in _ROUTABLE_SECTION_TYPES:
                    structural_chunks.setdefault(chunk.section_type, []).append(chunk)

        if skipped:
            total_skipped = sum(skipped.values())
            log.info(
                "ingestion_section_type_filter",
                work_id=work_id,
                original_chunks=len(chunks),
                kept_chunks=len(content_chunks),
                skipped_chunks=total_skipped,
                skipped_by_type=skipped,
                routed_chunks=sum(len(v) for v in structural_chunks.values()),
            )

        return content_chunks, skipped, structural_chunks

    async def _route_structural_sections(
        self,
        structural_chunks: dict[str, list[Chunk]],
        work_id: str,
    ) -> None:
        """Route structural section chunks to appropriate downstream managers.

        - Index sections → VocabularyManager.propose() for each term extracted
          from chunk text (one term per line, stripped).
        - Bibliography sections → AcquisitionManager.flag() for each entry
          extracted from chunk text (one citation per line, stripped).

        Both managers are lazy — they create their tables on first use if absent.
        Errors here are non-fatal: logged and swallowed so the main pipeline
        can complete successfully.
        """
        from author_library.catalog.acquisition import AcquisitionManager
        from author_library.vocabulary import VocabularyManager

        index_chunks = structural_chunks.get(SectionType.INDEX.value, [])
        bibliography_chunks = structural_chunks.get(SectionType.BIBLIOGRAPHY.value, [])

        if index_chunks:
            vocab = VocabularyManager(self._storage.pg)
            proposed = 0
            already_known = 0
            vocab_errors = 0
            for chunk in index_chunks:
                for line in chunk.text.splitlines():
                    term = line.strip()
                    # Skip blank lines, section headings (all-caps short), page refs
                    if not term or len(term) < 3 or len(term) > 120:
                        continue
                    # Skip pure numeric strings (page numbers)
                    if term.replace(",", "").replace(" ", "").isdigit():
                        continue
                    try:
                        record = await vocab.propose(
                            term,
                            note=f"Auto-proposed from index section of {work_id}",
                        )
                        if record.get("already_exists"):
                            already_known += 1
                        else:
                            proposed += 1
                    except Exception as exc:
                        vocab_errors += 1
                        if vocab_errors <= 3:
                            log.error(
                                "vocab_propose_failed",
                                work_id=work_id,
                                term=term,
                                error=str(exc),
                            )
                        elif vocab_errors == 4:
                            log.error(
                                "vocab_propose_failed_suppressed",
                                work_id=work_id,
                                message="Further vocab propose errors will be suppressed",
                            )

            log.info(
                "ingestion_index_vocab_routing",
                work_id=work_id,
                index_chunks=len(index_chunks),
                terms_proposed=proposed,
                terms_already_known=already_known,
                errors=vocab_errors,
            )

        if bibliography_chunks:
            acquisition = AcquisitionManager(self._storage.pg)
            flagged = 0
            already_flagged = 0
            acq_errors = 0
            for chunk in bibliography_chunks:
                for line in chunk.text.splitlines():
                    citation = line.strip()
                    # Skip blank lines and very short fragments
                    if not citation or len(citation) < 10:
                        continue
                    try:
                        was_added = await acquisition.flag(
                            citation_text=citation,
                            note=f"Auto-flagged from bibliography section of {work_id}",
                            priority="low",
                        )
                        if was_added:
                            flagged += 1
                        else:
                            already_flagged += 1
                    except Exception as exc:
                        acq_errors += 1
                        if acq_errors <= 3:
                            log.error(
                                "acq_flag_failed",
                                work_id=work_id,
                                citation=citation[:100],
                                error=str(exc),
                            )
                        elif acq_errors == 4:
                            log.error(
                                "acq_flag_failed_suppressed",
                                work_id=work_id,
                                message="Further acquisition flag errors will be suppressed",
                            )

            log.info(
                "ingestion_bibliography_acquisition_routing",
                work_id=work_id,
                bibliography_chunks=len(bibliography_chunks),
                citations_flagged=flagged,
                citations_already_flagged=already_flagged,
                errors=acq_errors,
            )

    def _build_annotation_context(
        self,
        catalog_entry: CatalogEntry,
        source_class: SourceClass,
    ) -> AnnotationContext:
        """Build AnnotationContext from the catalog entry."""
        # Extract source-class-specific fields from the catalog entry
        entry_data = catalog_entry.model_dump()

        subject_author = entry_data.get("subject_author_id", "")
        if not subject_author:
            subject_author = entry_data.get("about_author_id", "")
        if not subject_author:
            subject_author = entry_data.get("referenced_by", "")
        if not subject_author:
            subject_author = catalog_entry.author

        return AnnotationContext(
            work_title=catalog_entry.title,
            publication_year=catalog_entry.publication_year,
            author=catalog_entry.author,
            subject_author=subject_author,
            relationship_type=entry_data.get("relationship"),
            perspective_note=entry_data.get("perspective_note"),
            engagement_note=entry_data.get("engagement_note"),
            engagement_works=", ".join(entry_data.get("engagement_works") or []),
        )
