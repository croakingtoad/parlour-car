"""Classification pipeline — the gate through which every document enters.

No document passes into the system without classification. The pipeline:
1. Accepts a ParsedDocument + optional user metadata hints
2. Runs the classification engine
3. Creates the appropriate CatalogEntry subclass
4. Stores in the works table via WorkRepository
5. Returns the catalog entry with classification result
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import structlog

from author_library.catalog.classifier import SourceClassifier
from author_library.catalog.models import (
    CatalogEntry,
    ClassificationResult,
    ContextualCatalogEntry,
    FormatIngested,
    PersonalCatalogEntry,
    PrimaryCatalogEntry,
    ProcessingRoute,
    ReferenceCatalogEntry,
    SecondaryCatalogEntry,
    SourceClass,
    TertiaryCatalogEntry,
    route_for_source_class,
)
from author_library.errors import ClassificationError

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.parsing.models import ParsedDocument
    from author_library.storage.manager import StorageManager
    from author_library.storage.postgres import PostgresPool
    from author_library.storage.repositories import WorkRepository

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


class PipelineResult:
    """Result of the classification pipeline for a document.

    Contains the catalog entry, classification result, and the
    downstream processing route determined by source class.
    """

    def __init__(
        self,
        *,
        catalog_entry: CatalogEntry,
        classification: ClassificationResult,
        processing_route: ProcessingRoute,
    ) -> None:
        self.catalog_entry = catalog_entry
        self.classification = classification
        self.processing_route = processing_route

    def __repr__(self) -> str:
        return (
            f"PipelineResult(work_id={self.catalog_entry.work_id!r}, "
            f"source_class={self.classification.source_class}, "
            f"route={self.processing_route})"
        )


# ---------------------------------------------------------------------------
# Classification pipeline
# ---------------------------------------------------------------------------


class ClassificationPipeline:
    """Gates every document entering the system through classification.

    The pipeline coordinates between the classification engine, catalog
    entry creation, and storage persistence.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        work_repository: WorkRepository,
        subject_author: str,
        pg_pool: PostgresPool | None = None,
        storage: StorageManager | None = None,
    ) -> None:
        self._classifier = SourceClassifier(settings, storage=storage)
        self._work_repo = work_repository
        self._subject_author = subject_author
        self._pg_pool = pg_pool

    async def process(
        self,
        document: ParsedDocument,
        *,
        metadata_hints: dict[str, Any] | None = None,
        user_overrides: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Classify a document and create its catalog entry.

        Args:
            document: Parsed document to process.
            metadata_hints: Optional hints for the classifier (e.g., known bibliography entries).
            user_overrides: Optional field overrides for the catalog entry. Allows users to
                supply class-specific fields that cannot be inferred from the document alone
                (e.g., subject_author_id, work_type, engagement_type).

        Returns:
            PipelineResult with catalog entry, classification, and processing route.

        Raises:
            ClassificationError: If classification or catalog entry creation fails.
        """
        overrides = user_overrides or {}

        # A caller may provide a corrected document author without duplicating it
        # as external_author. Capture that explicit attribution before the generic
        # subject-author fallback below can populate ``author``.
        if (
            overrides.get("source_class") == SourceClass.REFERENCE.value
            and "external_author" not in overrides
            and "author" in overrides
        ):
            overrides = {**overrides, "external_author": overrides["author"]}

        # Step 1: Run the classification engine. Reference is a new user-confirmed
        # class whose filing author may match the document author; honoring that
        # confirmation prevents it from being reinterpreted as primary.
        if overrides.get("source_class") == SourceClass.REFERENCE.value:
            classification = ClassificationResult(
                source_class=SourceClass.REFERENCE,
                confidence=1.0,
                reasoning="User-confirmed standalone reference work.",
                signals_detected=["user_confirmed_reference"],
            )
        else:
            classification = await self._classifier.classify(
                document,
                subject_author=self._subject_author,
                metadata_hints=metadata_hints,
            )

        log.info(
            "pipeline_classification_complete",
            title=document.metadata.title,
            source_class=classification.source_class,
            confidence=classification.confidence,
        )

        # Step 1.5: Resolve missing author from subject_author slug
        if not document.metadata.author and "author" not in overrides:
            resolved = await self._resolve_author_name()
            if resolved:
                overrides = {**overrides, "author": resolved}
                log.info(
                    "pipeline_author_resolved",
                    resolved_author=resolved,
                    source="subject_author_lookup",
                )

        # Step 2: Build the appropriate catalog entry
        catalog_entry = self._build_catalog_entry(
            document=document,
            classification=classification,
            overrides=overrides,
        )
        await self._warn_for_uncontrolled_subject_headings(catalog_entry.subject_headings)

        # Step 3: Store in the works table (upsert for idempotent re-ingestion)
        try:
            work_data = self._catalog_entry_to_work_dict(catalog_entry, classification)
            existing = await self._work_repo.get(catalog_entry.work_id)
            if existing:
                # Only update fields safe for re-ingestion (skip PK and date types
                # that asyncpg requires as date objects, not strings)
                safe_fields = {
                    k: v for k, v in work_data.items()
                    if k not in ("work_id", "ingestion_date", "created_at", "updated_at",
                                 "date_published", "date_consumed")
                }
                await self._work_repo.update(catalog_entry.work_id, safe_fields)
                log.info("pipeline_entry_updated", work_id=catalog_entry.work_id)
            else:
                await self._work_repo.create(work_data)
        except Exception as exc:
            raise ClassificationError(
                f"Failed to store catalog entry: {exc}",
                context={"work_id": catalog_entry.work_id},
                cause=exc,
            ) from exc

        log.info(
            "pipeline_entry_stored",
            work_id=catalog_entry.work_id,
            source_class=classification.source_class,
        )

        # Step 4: Determine processing route
        route = route_for_source_class(classification.source_class)

        return PipelineResult(
            catalog_entry=catalog_entry,
            classification=classification,
            processing_route=route,
        )

    async def _warn_for_uncontrolled_subject_headings(
        self, subject_headings: list[str]
    ) -> None:
        """Log, but do not reject, headings absent from canonical vocabulary.

        Cataloging must continue to accept legitimate new subjects. The warning
        gives curators a review signal without creating proposals or imposing a
        hard vocabulary constraint on ingestion.
        """
        if self._pg_pool is None:
            return

        normalized = {heading.strip().casefold() for heading in subject_headings if heading.strip()}
        if not normalized:
            return

        try:
            table_exists = await self._pg_pool.fetch_val(
                "SELECT to_regclass('public.vocabulary_terms') IS NOT NULL"
            )
            if not table_exists:
                return

            rows = await self._pg_pool.fetch_all(
                """
                SELECT term
                FROM vocabulary_terms
                WHERE status = 'canonical' AND lower(term) = ANY($1::text[])
                """,
                list(normalized),
            )
            canonical = {str(row["term"]).casefold() for row in rows}
            unknown = sorted(normalized - canonical)
            if unknown:
                log.warning(
                    "catalog_subject_headings_not_canonical",
                    subject_headings=unknown,
                    message="Cataloging continues; review these headings in controlled vocabulary.",
                )
        except Exception as exc:
            # Vocabulary review must never make catalog ingestion unavailable.
            log.warning("catalog_subject_heading_validation_unavailable", error=str(exc))

    def _build_catalog_entry(
        self,
        *,
        document: ParsedDocument,
        classification: ClassificationResult,
        overrides: dict[str, Any],
    ) -> CatalogEntry:
        """Build the appropriate CatalogEntry subclass based on classification."""
        # Core fields shared across all classes
        core = self._extract_core_fields(document, classification, overrides)

        source_class = classification.source_class

        try:
            if source_class == SourceClass.PRIMARY:
                return PrimaryCatalogEntry(
                    **core,
                    subject_author_id=overrides.get(
                        "subject_author_id", self._subject_author_slug()
                    ),
                    work_type=overrides.get("work_type", "other"),
                    chronological_position=overrides.get("chronological_position"),
                    voice_profile_eligible=overrides.get("voice_profile_eligible", True),
                    dedication=overrides.get("dedication"),
                    table_of_contents=(
                        document.metadata.table_of_contents
                        if document.metadata.table_of_contents
                        else overrides.get("table_of_contents")
                    ),
                )
            elif source_class == SourceClass.SECONDARY:
                return SecondaryCatalogEntry(
                    **core,
                    about_author_id=overrides.get(
                        "about_author_id", self._subject_author_slug()
                    ),
                    external_author=overrides.get(
                        "external_author", document.metadata.author or "Unknown"
                    ),
                    external_author_affiliation=overrides.get("external_author_affiliation"),
                    relationship=overrides.get("relationship", "other"),
                    perspective_note=overrides.get(
                        "perspective_note",
                        f"Classified as secondary source about {self._subject_author}. "
                        f"Confidence: {classification.confidence:.2f}.",
                    ),
                    contains_primary_quotes=overrides.get("contains_primary_quotes", False),
                    quote_extraction_note=overrides.get("quote_extraction_note"),
                )
            elif source_class == SourceClass.CONTEXTUAL:
                return ContextualCatalogEntry(
                    **core,
                    referenced_by=overrides.get(
                        "referenced_by", self._subject_author_slug()
                    ),
                    engagement_type=overrides.get("engagement_type", "frequently-cites"),
                    engagement_note=overrides.get(
                        "engagement_note",
                        f"Classified as contextual source for {self._subject_author}. "
                        f"Confidence: {classification.confidence:.2f}.",
                    ),
                    engagement_works=overrides.get("engagement_works"),
                    engagement_frequency=overrides.get("engagement_frequency"),
                )
            elif source_class == SourceClass.TERTIARY:
                return TertiaryCatalogEntry(
                    **core,
                    reference_type=overrides.get("reference_type", "bibliography"),
                    coverage_note=overrides.get("coverage_note"),
                )
            elif source_class == SourceClass.REFERENCE:
                return ReferenceCatalogEntry(
                    **core,
                    external_author=self._required_reference_metadata(
                        "external_author",
                        overrides.get("external_author", document.metadata.author),
                    ),
                    reference_type=self._required_reference_metadata(
                        "reference_type",
                        overrides.get("reference_type"),
                    ),
                    subject_domain=self._required_reference_metadata(
                        "subject_domain",
                        overrides.get("subject_domain"),
                    ),
                )
            else:
                # Personal
                return PersonalCatalogEntry(
                    **core,
                    user_id=overrides.get("user_id", "marty"),
                )
        except Exception as exc:
            raise ClassificationError(
                f"Failed to build catalog entry: {exc}",
                context={
                    "source_class": source_class,
                    "title": document.metadata.title,
                },
                cause=exc,
            ) from exc

    @staticmethod
    def _required_reference_metadata(field: str, value: Any) -> str:
        """Return required reference metadata or fail before catalog persistence."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f'source_class="reference" requires metadata field "{field}"'
            )
        return value

    def _extract_core_fields(
        self,
        document: ParsedDocument,
        classification: ClassificationResult,
        overrides: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract core catalog fields from the document and overrides."""
        title = overrides.get("title", document.metadata.title or "Untitled")
        author = overrides.get("author", document.metadata.author or "Unknown")

        # Build work_id from author and title slugs
        work_id = overrides.get("work_id", self._generate_work_id(author, title))

        # Determine format from document
        format_ingested = overrides.get("format_ingested", document.format)

        return {
            "work_id": work_id,
            "title": title,
            "author": author,
            "source_class": classification.source_class,
            "source_class_note": overrides.get(
                "source_class_note", classification.reasoning
            ),
            "publication_year": overrides.get(
                "publication_year",
                self._extract_year(document.metadata.publication_date),
            ),
            "original_publication_year": overrides.get("original_publication_year"),
            "edition": overrides.get("edition"),
            "publisher": overrides.get("publisher", document.metadata.publisher or "Unknown"),
            "isbn": overrides.get("isbn", document.metadata.isbn),
            "format_ingested": format_ingested,
            "language": overrides.get("language", document.metadata.language),
            "word_count": overrides.get("word_count", document.metadata.word_count),
            "genre_tags": overrides.get("genre_tags", ["unclassified"]),
            "subject_headings": overrides.get("subject_headings", ["General"]),
            "ocr_quality": overrides.get("ocr_quality"),
            "ingestion_date": overrides.get("ingestion_date", date.today()),
            "notes": overrides.get("notes"),
            # Media/source fields (A5)
            "url": overrides.get("url"),
            "duration": overrides.get("duration"),
            "speakers": overrides.get("speakers", []),
            "date_published": overrides.get("date_published"),
            "date_consumed": overrides.get("date_consumed"),
            "transcript_cached": overrides.get("transcript_cached", False),
            "media": overrides.get("media"),
        }

    @staticmethod
    def _normalize_author_name(author: str) -> str:
        """Normalize author name from 'Last, First' to 'First Last' format.

        EPUB metadata may store author names in bibliographic order
        (e.g. "Guite, Malcolm") or natural order ("Malcolm Guite").
        This normalizes to natural order so that work_id slugs are
        consistent regardless of the metadata format.

        MARC-style records append life dates as a further comma-separated
        segment ("Coleridge, Samuel Taylor, 1772-1834"). Those are dropped:
        left in, they slugified into a *second* work_id for an author who
        already had one, which split 14,704 PART_OF edges onto a duplicate
        Work node (found 2026-08-13).
        """
        import re

        author = author.strip()
        if "," not in author:
            return author

        # Drop empty segments immediately. A MARC 100 field carries a trailing
        # comma when $d follows ("Coleridge, Samuel Taylor,"), and an empty tail
        # segment used to abort the reorder entirely, slugifying the raw string
        # into a SECOND work_id for an author that already had one.
        parts = [p.strip() for p in author.split(",") if p.strip()]
        if not parts:
            return author

        # A segment that is only a life date / bibliographic qualifier:
        # "1772-1834", "1637?-1674", "1950-", "b. 1900", "ca. 1200",
        # "fl. 1550-1600", "d. 1674?", "[1900-1980]" (RDA), "approximately
        # 1200", "19th cent.".
        date_qualifier = re.compile(
            r"^\[?\s*(?:b|d|ca|fl|active|approximately|circa)?\.?\s*"
            r"\d{1,4}(?:st|nd|rd|th)?\??"
            r"(?:\s*[-\u2013\u2014]\s*\d{0,4}\??)?"
            r"\s*(?:cent\.?|century)?\s*\]?\.?$",
            re.IGNORECASE,
        )

        # Strip date segments anywhere in the tail, not only the last one:
        # "Smith, John, 1900-1980, Jr." carries the date interior to a suffix.
        head = parts[0]
        tail = [p for p in parts[1:] if not date_qualifier.match(p)]

        # "Coleridge, 1772-1834" — surname plus dates, no given name.
        if not tail:
            return head

        # Reorder to natural order. Remaining segments stay joined so non-date
        # suffixes keep their prior behaviour ("Smith, John, Jr." -> "John, Jr.
        # Smith").
        return f"{', '.join(tail)} {head}"


    @staticmethod
    def _generate_work_id(author: str, title: str) -> str:
        """Generate a work_id from author and title per catalog-schema.md §6."""
        import hashlib
        import re
        import unicodedata

        def slugify(text: str) -> str:
            # Fold diacritics to ASCII rather than deleting them: stripping the
            # umlaut turned "Böll" into "bll", forking it from the same author
            # recorded as "Boll" — realistic MARC-vs-EPUB metadata variance.
            slug = unicodedata.normalize("NFKD", text)
            slug = "".join(c for c in slug if not unicodedata.combining(c))
            slug = slug.lower().strip()
            slug = re.sub(r"[^a-z0-9\s-]", "", slug)
            slug = re.sub(r"[\s]+", "-", slug)
            slug = re.sub(r"-+", "-", slug)
            return slug.strip("-")

        # Normalize "Last, First" → "First Last" before slugifying
        author = ClassificationPipeline._normalize_author_name(author)

        author_slug = slugify(author)
        title_slug = slugify(title)

        if not author_slug:
            # A fully non-Latin name slugifies to nothing, which produced a
            # malformed "--title" work_id and collapsed EVERY such author into
            # one id per title. Fall back to a deterministic digest of the
            # original name so distinct authors stay distinct.
            digest = hashlib.sha256(author.encode("utf-8")).hexdigest()[:12]
            author_slug = f"author-{digest}"

        work_id = f"{author_slug}--{title_slug}"

        # Truncate to max length
        if len(work_id) > 128:
            work_id = work_id[:128].rstrip("-")

        return work_id

    @staticmethod
    def _extract_year(publication_date: str | None) -> int:
        """Extract year from a date string, defaulting to current year."""
        if publication_date and len(publication_date) >= 4:
            try:
                return int(publication_date[:4])
            except ValueError:
                pass
        return date.today().year

    def _subject_author_slug(self) -> str:
        """Generate a slug from the subject author name."""
        import re

        slug = self._subject_author.lower().strip()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"[\s]+", "-", slug)
        return slug.strip("-")

    async def _resolve_author_name(self) -> str | None:
        """Resolve the subject author's canonical name from the authors table.

        Falls back to title-casing the slug if no database entry exists or
        if no pg_pool is available.
        """
        slug = self._subject_author_slug()

        if self._pg_pool is not None:
            try:
                row = await self._pg_pool.fetch_one(
                    "SELECT canonical_name FROM authors WHERE id = $1", slug
                )
                if row is not None:
                    name: str = row["canonical_name"]
                    log.debug(
                        "author_name_resolved_from_db",
                        slug=slug,
                        canonical_name=name,
                    )
                    return name
            except Exception as exc:
                log.warning(
                    "author_name_lookup_failed",
                    slug=slug,
                    error=str(exc),
                )

        # Fallback: title-case the slug (e.g. "fred-rogers" → "Fred Rogers")
        return slug.replace("-", " ").title()

    @staticmethod
    def _catalog_entry_to_work_dict(
        entry: CatalogEntry,
        classification: ClassificationResult,
    ) -> dict[str, Any]:
        """Convert a CatalogEntry to the dict format expected by WorkRepository."""
        # Get all model fields as a dict
        data = entry.model_dump()

        # The works table stores class-specific fields in source_metadata JSONB
        core_fields = {
            "work_id",
            "title",
            "author",
            "source_class",
            "source_class_note",
            "publication_year",
            "original_publication_year",
            "edition",
            "publisher",
            "isbn",
            "format_ingested",
            "language",
            "word_count",
            "genre_tags",
            "subject_headings",
            "ocr_quality",
            "ingestion_date",
            "notes",
            # Media/source fields (A5)
            "url",
            "duration",
            "speakers",
            "date_published",
            "date_consumed",
            "transcript_cached",
            "media",
        }

        # Separate core fields from source-class-specific fields
        work: dict[str, Any] = {}
        source_metadata: dict[str, Any] = {}

        for key, value in data.items():
            if key in core_fields:
                work[key] = value
            else:
                source_metadata[key] = value

        # Add classification result to source_metadata
        source_metadata["classification_confidence"] = classification.confidence
        source_metadata["classification_signals"] = classification.signals_detected

        work["source_metadata"] = source_metadata

        # Convert enums to their string values for storage
        if isinstance(work.get("source_class"), SourceClass):
            work["source_class"] = work["source_class"].value
        if isinstance(work.get("format_ingested"), FormatIngested):
            work["format_ingested"] = work["format_ingested"].value

        # Convert date to string for JSON serialization
        if isinstance(work.get("ingestion_date"), date):
            work["ingestion_date"] = work["ingestion_date"].isoformat()

        return work
