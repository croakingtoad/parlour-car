"""Ingestion pipeline orchestrator for The Author Library.

Coordinates the full pipeline: parse → classify → chunk → annotate →
embed → store → extract entities → create passage links.

Routes processing by source class:
  - PRIMARY: full enrichment (all steps)
  - SECONDARY: embeddings + attributed graph edges only
  - CONTEXTUAL: embeddings + cross-resource link targets
  - TERTIARY: metadata only, no content processing
  - PERSONAL: embeddings + USER_REFLECTS_ON graph edges, NO voice profile

Supports idempotent re-ingestion: deletes old chunks/embeddings for a
work before re-processing.
"""

from __future__ import annotations

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

if TYPE_CHECKING:
    from uuid import UUID

    from author_library.catalog.models import CatalogEntry
    from author_library.chunking.models import Chunk
    from author_library.config import Settings
    from author_library.embeddings.base import EmbeddingProvider
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
        "source_class",
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
    ) -> None:
        self.work_id = work_id
        self.source_class = source_class
        self.processing_route = processing_route
        self.chunks_by_granularity = chunks_by_granularity
        self.embeddings_stored = embeddings_stored
        self.entity_count = entity_count
        self.edge_count = edge_count
        self.errors = errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_id": self.work_id,
            "source_class": self.source_class,
            "processing_route": self.processing_route,
            "chunks_by_granularity": self.chunks_by_granularity,
            "embeddings_stored": self.embeddings_stored,
            "entity_count": self.entity_count,
            "edge_count": self.edge_count,
            "errors": self.errors,
        }


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline for a single work.

    Coordinates parsing, classification, chunking, annotation, embedding,
    entity extraction, and passage linking according to source class routing.
    """

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
        path = Path(file_path)
        hints = metadata_hints or {}
        errors: list[str] = []

        log.info(
            "ingestion_starting",
            file_path=str(path),
            subject_author=subject_author_id,
        )

        # Step 1: Parse
        parser = get_parser(path)
        document = await parser.parse(str(path))
        log.info(
            "ingestion_parsed",
            title=document.metadata.title,
            word_count=document.metadata.word_count,
            format=document.format,
        )

        # Step 2: Classify via pipeline (creates catalog entry + stores in works table)
        classification_pipeline = ClassificationPipeline(
            settings=self._settings,
            work_repository=self._storage.works,
            subject_author=subject_author_id,
            pg_pool=self._storage.pg,
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

        # Upsert work node in Neo4j
        await self._storage.graph.upsert_work_node({
            "work_id": work_id,
            "title": catalog_entry.title,
            "author": catalog_entry.author,
            "source_class": source_class.value,
            "publication_year": catalog_entry.publication_year,
        })

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
        )

        # Step 5: Annotate
        annotation_ctx = self._build_annotation_context(catalog_entry, source_class)
        annotator = ChunkAnnotator(self._settings)
        chunks = await annotator.annotate_chunks(chunks, annotation_ctx)

        # Step 6: Store chunks in PG
        # Sort so parents (macro) are inserted before children (meso) before
        # grandchildren (micro), satisfying the parent_chunk_id foreign key.
        _gran_order = {"macro": 0, "meso": 1, "micro": 2, "nano": 3}
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (_gran_order.get(str(c.granularity), 9), c.position),
        )

        chunk_id_map: dict[str, UUID] = {}
        for chunk in sorted_chunks:
            # Translate in-memory parent_chunk_id to DB-generated UUID
            resolved_parent: UUID | None = None
            if chunk.parent_chunk_id is not None:
                resolved_parent = chunk_id_map.get(chunk.parent_chunk_id)

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
                "metadata": chunk.metadata,
                "raw_content": chunk.raw_content,
                "raw_content_window": chunk.raw_content_window,
                "pass_number": pass_number,
            }

            pg_id = await self._storage.chunks.create(chunk_data)
            chunk_id_map[chunk.id] = pg_id

        log.info("ingestion_chunks_stored", work_id=work_id, count=len(chunk_id_map))

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
        import time as _time

        embed_start = _time.monotonic()
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

            try:
                batch_result = await self._embedding.embed_batch(batch_texts)
                for chunk, vector in zip(
                    batch_chunks, batch_result.vectors, strict=True
                ):
                    maybe_id = chunk_id_map.get(chunk.id)
                    if maybe_id is None:
                        continue
                    pg_id = maybe_id
                    await self._storage.embeddings.store(
                        pg_id,
                        vector,
                        self._embedding.provider_name,
                        self._embedding.model_name,
                        self._embedding.dimensions,
                    )
                    embeddings_stored += 1
            except Exception as exc:
                error_msg = (
                    f"Embedding batch {batch_idx + 1}/{total_batches} "
                    f"({len(batch_texts)} texts, ~{est_tokens} tokens) failed: {exc}"
                )
                log.error("ingestion_embedding_failed", error=error_msg)
                errors.append(error_msg)

            # Warn if embedding stage is running long
            elapsed = _time.monotonic() - embed_start
            if elapsed > 60:
                log.warning(
                    "ingestion_embedding_slow",
                    work_id=work_id,
                    elapsed_seconds=round(elapsed, 1),
                    batches_complete=batch_idx + 1,
                    total_batches=total_batches,
                )

        embed_elapsed = round(_time.monotonic() - embed_start, 1)
        log.info(
            "ingestion_embeddings_stored",
            work_id=work_id,
            count=embeddings_stored,
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

        # Step 9: Entity extraction (PRIMARY and SECONDARY only — NOT personal)
        entity_count = 0
        edge_count = 0
        if route in (ProcessingRoute.FULL_ENRICHMENT, ProcessingRoute.EMBEDDINGS_AND_GRAPH):
            try:
                extractor = EntityExtractor(
                    self._storage.neo4j,
                    self._settings.api_keys,
                    self._settings.llm,
                )
                extraction_result = await extractor.extract_and_persist(
                    chunks,
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

        # Step 10: Passage linking (PRIMARY and CONTEXTUAL)
        if route in (ProcessingRoute.FULL_ENRICHMENT, ProcessingRoute.EMBEDDINGS_AND_LINKS):
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
            errors=len(errors),
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

    async def _create_passage_links(
        self,
        chunks: list[Chunk],
        source_class: SourceClass,
    ) -> int:
        """Create cross-resource passage links for the work's chunks.

        Loads contextual chunks from the database to link against.
        Returns the total number of edges created.
        """
        edges_created = 0
        primary_chunks = [c for c in chunks if c.source_class == "primary"]
        contextual_chunks = [c for c in chunks if c.source_class == "contextual"]

        # For primary sources, we need to load existing contextual chunks to link against
        if source_class == SourceClass.PRIMARY and primary_chunks:
            # Load all contextual chunks from the database
            all_works = await self._storage.works.list_by_author(
                primary_chunks[0].work_id.split("--")[0]
            )
            ctx_work_ids = [
                w["work_id"] for w in all_works if w.get("source_class") == "contextual"
            ]
            db_contextual: list[Any] = []
            for ctx_wid in ctx_work_ids:
                db_chunks = await self._storage.chunks.list_by_work(ctx_wid, granularity="meso")
                db_contextual.extend(db_chunks)

            # Convert to Chunk objects for the linkers
            from author_library.chunking.models import Chunk as ChunkModel

            contextual_for_linking = [
                ChunkModel(
                    id=str(c.get("id", "")),
                    text=c.get("text", ""),
                    granularity=c.get("granularity", "meso"),
                    work_id=c.get("work_id", ""),
                    source_class=c.get("source_class", "contextual"),
                    position=c.get("position", 0),
                )
                for c in db_contextual
            ]

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

        # For contextual sources, link against existing primary chunks
        elif source_class == SourceClass.CONTEXTUAL and contextual_chunks:
            all_works = await self._storage.works.list_by_author(
                contextual_chunks[0].work_id.split("--")[0]
            )
            prim_work_ids = [
                w["work_id"] for w in all_works if w.get("source_class") == "primary"
            ]
            db_primary: list[Any] = []
            for prim_wid in prim_work_ids:
                db_chunks = await self._storage.chunks.list_by_work(prim_wid, granularity="meso")
                db_primary.extend(db_chunks)

            from author_library.chunking.models import Chunk as ChunkModel

            primary_for_linking = [
                ChunkModel(
                    id=str(c.get("id", "")),
                    text=c.get("text", ""),
                    granularity=c.get("granularity", "meso"),
                    work_id=c.get("work_id", ""),
                    source_class=c.get("source_class", "primary"),
                    position=c.get("position", 0),
                )
                for c in db_primary
            ]

            if primary_for_linking:
                explicit = ExplicitLinkDetector(self._storage.neo4j)
                explicit_result = await explicit.detect_and_link(
                    primary_for_linking, contextual_chunks
                )
                edges_created += explicit_result.edges_created

        log.info("passage_links_created", edges=edges_created)
        return edges_created

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
