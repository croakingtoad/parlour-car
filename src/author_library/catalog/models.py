"""Catalog metadata schema for The Author Library.

Implements the full catalog-schema.md specification as Pydantic models.
Every work entering the library receives a catalog entry with class-specific
fields validated at construction time.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

WORK_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*--[a-z0-9]+(?:-[a-z0-9]+)*$")
WORK_ID_MAX_LENGTH = 128


class SourceClass(StrEnum):
    """Source classification hierarchy."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    CONTEXTUAL = "contextual"
    TERTIARY = "tertiary"
    PERSONAL = "personal"
    REFERENCE = "reference"


class FormatIngested(StrEnum):
    """Document formats accepted for ingestion."""

    EPUB = "epub"
    PDF = "pdf"
    TXT = "txt"
    HTML = "html"
    DOCX = "docx"
    VIDEO = "video"
    AUDIO = "audio"
    TRANSCRIPT = "transcript"
    YOUTUBE_CAPTIONS = "youtube_captions"


class OcrQuality(StrEnum):
    """OCR quality assessment levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOT_APPLICABLE = "not-applicable"


class WorkType(StrEnum):
    """Primary source work types."""

    MONOGRAPH = "monograph"
    ESSAY_COLLECTION = "essay-collection"
    POETRY_COLLECTION = "poetry-collection"
    ACADEMIC_PAPER = "academic-paper"
    LECTURE_TRANSCRIPT = "lecture-transcript"
    BLOG_POST = "blog-post"
    LETTER = "letter"
    SERMON = "sermon"
    FOREWORD = "foreword"
    INTERVIEW_RESPONSES = "interview-responses"
    OTHER = "other"


class SecondaryRelationship(StrEnum):
    """Relationship types for secondary sources."""

    BIOGRAPHY = "biography"
    CRITICAL_STUDY = "critical-study"
    REVIEW = "review"
    INTERVIEW = "interview"
    COMPANION = "companion"
    DISSERTATION = "dissertation"
    EDITED_COLLECTION = "edited-collection"
    OBITUARY = "obituary"
    PROFILE = "profile"
    OTHER = "other"


class EngagementType(StrEnum):
    """How the subject author engages with a contextual source."""

    INFLUENCES = "influences"
    RESPONDS_TO = "responds-to"
    CRITIQUES = "critiques"
    EXTENDS = "extends"
    INTERPRETS = "interprets"
    FREQUENTLY_CITES = "frequently-cites"


class EngagementFrequency(StrEnum):
    """Frequency of engagement with contextual source across corpus."""

    FOUNDATIONAL = "foundational"
    MAJOR = "major"
    MINOR = "minor"
    SINGLE_REFERENCE = "single-reference"


class ReferenceType(StrEnum):
    """Types for tertiary reference sources."""

    BIBLIOGRAPHY = "bibliography"
    ENCYCLOPEDIA_ENTRY = "encyclopedia-entry"
    CATALOG_RECORD = "catalog-record"
    DATABASE = "database"
    INDEX = "index"


class MediaType(StrEnum):
    """Media format classification for catalog records."""

    BOOK = "book"
    VIDEO = "video"
    AUDIO = "audio"
    PODCAST = "podcast"
    ARTICLE = "article"


# ---------------------------------------------------------------------------
# Core catalog entry (shared across all source classes)
# ---------------------------------------------------------------------------

# Custom annotated types for constrained fields
NonEmptyStr = Annotated[str, Field(min_length=1)]
ClassificationNote = Annotated[str, Field(min_length=10)]

# ISO 639-2 (3-char) → ISO 639-1 (2-char) mapping for common languages.
# EPUBs often embed 3-char codes in their OPF <dc:language> metadata.
_ISO_639_2_TO_1: dict[str, str] = {
    "eng": "en",
    "fra": "fr",
    "fre": "fr",  # bibliographic variant
    "deu": "de",
    "ger": "de",  # bibliographic variant
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "zho": "zh",
    "chi": "zh",  # bibliographic variant
    "jpn": "ja",
    "kor": "ko",
    "ara": "ar",
    "hin": "hi",
    "nld": "nl",
    "dut": "nl",  # bibliographic variant
    "swe": "sv",
    "nor": "no",
    "dan": "da",
    "fin": "fi",
    "pol": "pl",
    "ces": "cs",
    "cze": "cs",  # bibliographic variant
    "ell": "el",
    "gre": "el",  # bibliographic variant
    "heb": "he",
    "tur": "tr",
    "ukr": "uk",
    "ron": "ro",
    "rum": "ro",  # bibliographic variant
    "hun": "hu",
    "cat": "ca",
    "lat": "la",
    "gla": "gd",
    "cym": "cy",
    "wel": "cy",  # bibliographic variant
    "gle": "ga",
}


class CatalogEntry(BaseModel):
    """Base catalog entry with core fields required for all source classes.

    Implements catalog-schema.md §1 (Core Fields).
    """

    work_id: str
    title: NonEmptyStr
    author: NonEmptyStr
    source_class: SourceClass
    source_class_note: ClassificationNote
    publication_year: int
    original_publication_year: int | None = None
    edition: str | None = None
    publisher: NonEmptyStr
    isbn: str | None = None
    format_ingested: FormatIngested
    language: str = "en"
    word_count: int = Field(ge=0)
    genre_tags: list[str] = Field(min_length=1)
    subject_headings: list[str] = Field(min_length=1)
    ocr_quality: OcrQuality | None = None
    ingestion_date: date = Field(default_factory=date.today)
    notes: str | None = None
    # Media/source fields (A5)
    url: str | None = None
    duration: int | None = None  # duration in seconds
    speakers: list[str] = Field(default_factory=list)
    date_published: date | None = None
    date_consumed: date | None = None
    transcript_cached: bool = False
    media: MediaType | None = None

    @field_validator("work_id")
    @classmethod
    def validate_work_id(cls, v: str) -> str:
        if len(v) > WORK_ID_MAX_LENGTH:
            msg = f"work_id must be at most {WORK_ID_MAX_LENGTH} characters, got {len(v)}"
            raise ValueError(msg)
        if not WORK_ID_PATTERN.match(v):
            msg = (
                "work_id must be lowercase alphanumeric with hyphens, "
                "author-slug and title-slug separated by double-hyphen (--). "
                f"Got: {v!r}"
            )
            raise ValueError(msg)
        return v

    @field_validator("genre_tags")
    @classmethod
    def validate_genre_tags(cls, v: list[str]) -> list[str]:
        return [tag.lower().strip() for tag in v]

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        # Normalize BCP-47 tags like "en-US" to ISO 639-1 "en"
        code = v.split("-")[0].strip().lower()
        # Normalize ISO 639-2 (3-char) codes to ISO 639-1 (2-char).
        # EPUBs commonly provide 3-char codes from their OPF metadata.
        if len(code) == 3:
            code = _ISO_639_2_TO_1.get(code, code)
        if len(code) != 2:
            msg = f"language must be a 2-character ISO 639-1 code (or recognized ISO 639-2), got {v!r}"
            raise ValueError(msg)
        return code


# ---------------------------------------------------------------------------
# Source-class-specific catalog entries
# ---------------------------------------------------------------------------


class PrimaryCatalogEntry(CatalogEntry):
    """Catalog entry for primary sources (works BY the subject author).

    Implements catalog-schema.md §2 (Primary Source Additional Fields).
    """

    source_class: SourceClass = SourceClass.PRIMARY
    subject_author_id: NonEmptyStr
    work_type: WorkType
    chronological_position: int | None = None
    voice_profile_eligible: bool = True
    dedication: str | None = None
    table_of_contents: list[str] | None = None

    @model_validator(mode="after")
    def enforce_source_class(self) -> PrimaryCatalogEntry:
        if self.source_class != SourceClass.PRIMARY:
            msg = f"PrimaryCatalogEntry must have source_class='primary', got {self.source_class!r}"
            raise ValueError(msg)
        return self


class SecondaryCatalogEntry(CatalogEntry):
    """Catalog entry for secondary sources (works ABOUT the subject author).

    Implements catalog-schema.md §3 (Secondary Source Additional Fields).
    """

    source_class: SourceClass = SourceClass.SECONDARY
    about_author_id: NonEmptyStr
    external_author: NonEmptyStr
    external_author_affiliation: str | None = None
    relationship: SecondaryRelationship
    perspective_note: NonEmptyStr
    contains_primary_quotes: bool
    quote_extraction_note: str | None = None

    @model_validator(mode="after")
    def enforce_source_class(self) -> SecondaryCatalogEntry:
        if self.source_class != SourceClass.SECONDARY:
            msg = (
                f"SecondaryCatalogEntry must have source_class='secondary', "
                f"got {self.source_class!r}"
            )
            raise ValueError(msg)
        return self


class ContextualCatalogEntry(CatalogEntry):
    """Catalog entry for contextual sources (works the subject author ENGAGES WITH).

    Implements catalog-schema.md §4 (Contextual Source Additional Fields).
    """

    source_class: SourceClass = SourceClass.CONTEXTUAL
    referenced_by: NonEmptyStr
    engagement_type: EngagementType
    engagement_note: NonEmptyStr
    engagement_works: list[str] | None = None
    engagement_frequency: EngagementFrequency | None = None

    @model_validator(mode="after")
    def enforce_source_class(self) -> ContextualCatalogEntry:
        if self.source_class != SourceClass.CONTEXTUAL:
            msg = (
                f"ContextualCatalogEntry must have source_class='contextual', "
                f"got {self.source_class!r}"
            )
            raise ValueError(msg)
        return self


class TertiaryCatalogEntry(CatalogEntry):
    """Catalog entry for tertiary sources (reference works).

    Implements catalog-schema.md §5 (Tertiary Source Additional Fields).
    """

    source_class: SourceClass = SourceClass.TERTIARY
    reference_type: ReferenceType
    coverage_note: str | None = None

    @model_validator(mode="after")
    def enforce_source_class(self) -> TertiaryCatalogEntry:
        if self.source_class != SourceClass.TERTIARY:
            msg = (
                f"TertiaryCatalogEntry must have source_class='tertiary', "
                f"got {self.source_class!r}"
            )
            raise ValueError(msg)
        return self


class ReferenceCatalogEntry(CatalogEntry):
    """Catalog entry for standalone third-party reference works.

    Reference works are ingested for their content without implying any
    relationship to a subject author. They are never voice-profile eligible.
    """

    source_class: SourceClass = SourceClass.REFERENCE
    external_author: NonEmptyStr
    reference_type: NonEmptyStr
    subject_domain: NonEmptyStr

    @model_validator(mode="after")
    def enforce_source_class(self) -> ReferenceCatalogEntry:
        if self.source_class != SourceClass.REFERENCE:
            msg = (
                f"ReferenceCatalogEntry must have source_class='reference', "
                f"got {self.source_class!r}"
            )
            raise ValueError(msg)
        return self


class PersonalCatalogEntry(CatalogEntry):
    """Catalog entry for personal sources (user reflections/notes).

    Personal sources are the user's own writing — reflections, journal entries,
    annotations, responses. They are NEVER attributed to the subject author and
    NEVER contribute to voice profiles. They receive embeddings and graph edges
    (USER_REFLECTS_ON) but skip voice profile extraction entirely.
    """

    source_class: SourceClass = SourceClass.PERSONAL
    user_id: str = "marty"  # Single-user V1, schema-ready for multi-user

    @model_validator(mode="after")
    def enforce_source_class(self) -> PersonalCatalogEntry:
        if self.source_class != SourceClass.PERSONAL:
            msg = (
                f"PersonalCatalogEntry must have source_class='personal', "
                f"got {self.source_class!r}"
            )
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Classification result (produced by the classification engine)
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    """Result from the source classification engine.

    Captures the classification decision along with confidence,
    reasoning, and the signals that contributed.
    """

    source_class: SourceClass
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    signals_detected: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def apply_default_to_secondary_rule(self) -> ClassificationResult:
        """If confidence < 0.7, default to SECONDARY for safety.

        This prevents voice contamination from uncertain classifications.
        Per collection-librarian SKILL.md: "safer to exclude from voice
        than to contaminate."
        """
        if self.confidence < 0.7 and self.source_class != SourceClass.SECONDARY:
            self.source_class = SourceClass.SECONDARY
            self.reasoning = (
                f"[AUTO-DOWNGRADED TO SECONDARY: confidence {self.confidence:.2f} < 0.70] "
                f"{self.reasoning}"
            )
        return self


# ---------------------------------------------------------------------------
# Processing route (determined by source class)
# ---------------------------------------------------------------------------


class ProcessingRoute(StrEnum):
    """Downstream processing routes gated by source classification."""

    FULL_ENRICHMENT = "full_enrichment"
    EMBEDDINGS_AND_GRAPH = "embeddings_and_graph"
    EMBEDDINGS_AND_LINKS = "embeddings_and_links"
    METADATA_ONLY = "metadata_only"
    PERSONAL_ENRICHMENT = "personal_enrichment"
    REFERENCE_ENRICHMENT = "reference_enrichment"


def route_for_source_class(source_class: SourceClass) -> ProcessingRoute:
    """Determine the processing route for a given source class.

    - Primary → full enrichment (chunking, embedding, voice profile, graph)
    - Secondary → embeddings + attributed graph edges only
    - Contextual → embeddings + cross-resource link targets
    - Tertiary → metadata only, no content ingestion
    - Personal → embeddings + USER_REFLECTS_ON graph edges, NO voice profile
    - Reference → entities + passage links + connection surfacing, NO voice profile
    """
    return {
        SourceClass.PRIMARY: ProcessingRoute.FULL_ENRICHMENT,
        SourceClass.SECONDARY: ProcessingRoute.EMBEDDINGS_AND_GRAPH,
        SourceClass.CONTEXTUAL: ProcessingRoute.EMBEDDINGS_AND_LINKS,
        SourceClass.TERTIARY: ProcessingRoute.METADATA_ONLY,
        SourceClass.PERSONAL: ProcessingRoute.PERSONAL_ENRICHMENT,
        SourceClass.REFERENCE: ProcessingRoute.REFERENCE_ENRICHMENT,
    }[source_class]
