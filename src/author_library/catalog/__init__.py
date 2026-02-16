"""Catalog and source classification for The Author Library.

This package implements the catalog metadata schema, source classification
engine, classification pipeline, and mixed-authorship handling. It is the
gateway through which every document enters the system — nothing passes
through without classification.
"""

from author_library.catalog.classifier import SourceClassifier
from author_library.catalog.mixed_authorship import (
    AuthorshipSegment,
    MixedAuthorshipAnalyzer,
    MixedAuthorshipResult,
)
from author_library.catalog.models import (
    CatalogEntry,
    ClassificationResult,
    ContextualCatalogEntry,
    PrimaryCatalogEntry,
    ProcessingRoute,
    SecondaryCatalogEntry,
    SourceClass,
    TertiaryCatalogEntry,
    route_for_source_class,
)
from author_library.catalog.pipeline import ClassificationPipeline, PipelineResult

__all__ = [
    "AuthorshipSegment",
    "CatalogEntry",
    "ClassificationPipeline",
    "ClassificationResult",
    "ContextualCatalogEntry",
    "MixedAuthorshipAnalyzer",
    "MixedAuthorshipResult",
    "PipelineResult",
    "PrimaryCatalogEntry",
    "ProcessingRoute",
    "SecondaryCatalogEntry",
    "SourceClass",
    "SourceClassifier",
    "TertiaryCatalogEntry",
    "route_for_source_class",
]
