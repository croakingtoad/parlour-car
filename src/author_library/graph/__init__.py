"""Knowledge graph engine for The Author Library.

Provides entity extraction from chunks, cross-resource passage linking
at three confidence tiers, and graph query helpers for retrieval.
"""

from .entity_extraction import EntityExtractor, ExtractionResult
from .linking_explicit import ExplicitLinkDetector
from .linking_implicit import ImplicitEngagementDetector
from .linking_thematic import ThematicParallelDetector
from .queries import GraphQueryService

__all__ = [
    "EntityExtractor",
    "ExplicitLinkDetector",
    "ExtractionResult",
    "GraphQueryService",
    "ImplicitEngagementDetector",
    "ThematicParallelDetector",
]
