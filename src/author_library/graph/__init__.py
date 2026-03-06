"""Knowledge graph engine for The Author Library.

Provides entity extraction from chunks, cross-resource passage linking
at three confidence tiers, graph query helpers for retrieval, and
backfill utilities for PG→Neo4j consistency recovery.
"""

from .backfill import BackfillResult, backfill_missing_graph_data, check_pg_neo4j_consistency
from .entity_extraction import EntityExtractor, ExtractionResult
from .linking_explicit import ExplicitLinkDetector
from .linking_implicit import ImplicitEngagementDetector
from .linking_thematic import ThematicParallelDetector
from .queries import GraphQueryService
from .theme_dedup import ThemeDedupResult, deduplicate_themes

__all__ = [
    "BackfillResult",
    "EntityExtractor",
    "ExplicitLinkDetector",
    "ExtractionResult",
    "GraphQueryService",
    "ImplicitEngagementDetector",
    "ThemeDedupResult",
    "ThematicParallelDetector",
    "backfill_missing_graph_data",
    "check_pg_neo4j_consistency",
    "deduplicate_themes",
]
