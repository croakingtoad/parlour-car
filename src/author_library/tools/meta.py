"""MCP tool handlers for library metadata operations (E012).

Provides:
  - list_authors: All authors in the library with work counts.
  - author_bio: Biographical summary from voice profile.
  - list_works: Works catalog for an author with filtering.
  - library_stats: Collection statistics and coverage metrics.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from author_library.errors import RetrievalError
from author_library.intelligence.voice_crud import VoiceProfileManager

if TYPE_CHECKING:
    from collections.abc import Sequence

    from author_library.config import Settings
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)

# Flag either a sizable absolute gap or a systemic share of a smaller work.
ENTITY_EDGE_GAP_WARNING_MIN_CHUNKS = 10
ENTITY_EDGE_GAP_WARNING_MIN_SHARE = 0.1


def _entity_edge_gap_is_warning(total_chunks: int, uncovered_chunks: int) -> bool:
    """Return whether missing entity edges are material for one work."""
    if total_chunks <= 0 or uncovered_chunks <= 0:
        return False
    return (
        uncovered_chunks >= ENTITY_EDGE_GAP_WARNING_MIN_CHUNKS
        or uncovered_chunks / total_chunks >= ENTITY_EDGE_GAP_WARNING_MIN_SHARE
    )


def _graph_entity_backfill_commands(work_ids: list[str]) -> str:
    """Format bounded, non-destructive-by-default backfill invocations."""
    return ", ".join(
        f"`uv run python scripts/backfill_graph_and_entities.py {work_id}`"
        for work_id in work_ids[:3]
    ) + (" ..." if len(work_ids) > 3 else "")


_SHORT_LINE_GENRES = frozenset(
    {
        "address",
        "audio-transcript",
        "blessings",
        "homily",
        "interview-transcript",
        "lecture",
        "mixed_poetry_prose",
        "podcast-transcript",
        "poems",
        "poetry",
        "poetry_collection",
        "sermon",
        "sonnet_sequence",
        "transcript",
        "verse",
        "video-transcript",
        "youtube-captions",
    }
)


def _classify_chunk_noise(
    noise: int,
    total: int,
    genre_tags: Sequence[str] | None,
) -> tuple[str, str] | None:
    """Classify sub-50-character chunks as informational or a warning."""
    if noise == 0 or total == 0:
        return None

    percentage = noise / total
    message = f"noise_chunks ({noise} chunks < 50 chars, {percentage:.0%})"
    if any(tag.casefold() in _SHORT_LINE_GENRES for tag in genre_tags or ()):
        return (
            "info",
            f"{message} — informational: short lines are the literary form by design "
            "for this genre; already excluded from retrieval by the 50-char minimum",
        )
    if percentage > 0.1:
        return "warning", message
    return None


def _format_year_range(earliest_year: int | None, latest_year: int | None) -> str:
    """Format an author's dated range, explicitly labeling an undated catalog."""
    if earliest_year is None or latest_year is None:
        return "undated"
    return f"{earliest_year}-{latest_year}"


async def handle_list_authors(
    arguments: dict[str, Any],
    *,
    storage: StorageManager,
) -> str:
    """Handle the list_authors MCP tool call.

    Returns:
        JSON with all authors, work counts, and ingestion stats.
    """
    # Query all works grouped by author
    rows = await storage.pg.fetch_all(
        """SELECT author,
                  count(*) AS work_count,
                  count(*) FILTER (WHERE source_class = 'primary') AS primary_count,
                  count(*) FILTER (WHERE source_class = 'secondary') AS secondary_count,
                  count(*) FILTER (WHERE source_class = 'contextual') AS contextual_count,
                  count(*) FILTER (WHERE source_class = 'tertiary') AS tertiary_count,
                  count(*) FILTER (WHERE source_class = 'reference') AS reference_count,
                  sum(word_count) AS total_words,
                  min(publication_year) AS earliest_year,
                  max(publication_year) AS latest_year
           FROM works
           GROUP BY author
           ORDER BY work_count DESC"""
    )

    authors = []
    for row in rows:
        r = dict(row)
        authors.append(
            {
                "author": r["author"],
                "work_count": r["work_count"],
                "primary_works": r["primary_count"],
                "secondary_works": r["secondary_count"],
                "contextual_works": r["contextual_count"],
                "tertiary_works": r["tertiary_count"],
                "reference_works": r["reference_count"],
                "total_words": r["total_words"] or 0,
                "year_range": _format_year_range(r["earliest_year"], r["latest_year"]),
            }
        )

    return json.dumps({"authors": authors, "total_authors": len(authors)}, indent=2)


async def handle_author_bio(
    arguments: dict[str, Any],
    *,
    settings: Settings,
    storage: StorageManager,
) -> str:
    """Handle the author_bio MCP tool call.

    Arguments:
        author_id (str): The author's slug identifier.

    Returns:
        JSON with biographical summary from voice profile and corpus stats.
    """
    author_id = arguments.get("author_id")
    if not author_id:
        raise RetrievalError("author_id is required", context={"arguments": arguments})

    # Get voice profile
    voice_manager = VoiceProfileManager(settings)
    profile = await voice_manager.get_current(
        author_id=author_id,
        voice_repo=storage.voice_profiles,
    )

    # Get works for this author
    works = await storage.works.list_by_author(author_id)
    primary_works = [w for w in works if w.get("source_class") == "primary"]

    # Get thematic entries
    themes_raw = await storage.thematic.list_entries(author_id)

    bio: dict[str, Any] = {
        "author_id": author_id,
        "works_in_library": len(works),
        "primary_works": len(primary_works),
    }

    if primary_works:
        titles = [w.get("title", "") for w in primary_works]
        years = [w.get("publication_year", 0) for w in primary_works if w.get("publication_year")]
        bio["primary_titles"] = titles
        if years:
            bio["publication_range"] = f"{min(years)}-{max(years)}"
        bio["total_primary_words"] = sum(w.get("word_count", 0) for w in primary_works)

    if profile:
        bio["voice_profile"] = {
            "register": profile.register,
            "sentence_patterns": profile.sentence_patterns[:5],
            "vocabulary_tendencies": profile.vocabulary_tendencies[:5],
            "rhetorical_moves": profile.rhetorical_moves[:5],
            "characteristic_phrases": profile.characteristic_phrases[:5],
            "humor_style": profile.humor_style,
            "confidence": profile.confidence,
        }
    else:
        bio["voice_profile"] = None

    if themes_raw:
        bio["major_themes"] = [
            {
                "theme": t.get("theme", ""),
                "stance": t.get("author_stance", ""),
            }
            for t in themes_raw[:10]
        ]
    else:
        bio["major_themes"] = []

    return json.dumps(bio, indent=2)


async def handle_list_works(
    arguments: dict[str, Any],
    *,
    storage: StorageManager,
) -> str:
    """Handle the list_works MCP tool call.

    Arguments:
        author_id (str): The author's slug identifier.
        source_class (str, optional): Filter by source class.

    Returns:
        JSON with works catalog for the author.
    """
    author_id = arguments.get("author_id")
    if not author_id:
        raise RetrievalError("author_id is required", context={"arguments": arguments})

    source_class_filter = arguments.get("source_class")

    works = await storage.works.list_by_author(author_id)

    if source_class_filter:
        works = [w for w in works if w.get("source_class") == source_class_filter]

    catalog: list[dict[str, Any]] = []
    for w in works:
        entry: dict[str, Any] = {
            "work_id": w["work_id"],
            "title": w.get("title", ""),
            "author": w.get("author", ""),
            "source_class": w.get("source_class", ""),
            "publication_year": w.get("publication_year"),
            "format_ingested": w.get("format_ingested", ""),
            "word_count": w.get("word_count", 0),
            "genre_tags": w.get("genre_tags", []),
        }

        # Add source-class-specific metadata
        source_meta = w.get("source_metadata", {})
        if isinstance(source_meta, str):
            source_meta = json.loads(source_meta)

        if w.get("source_class") == "primary":
            entry["work_type"] = source_meta.get("work_type", "")
            entry["voice_profile_eligible"] = source_meta.get("voice_profile_eligible", True)
        elif w.get("source_class") == "secondary":
            entry["external_author"] = source_meta.get("external_author", "")
            entry["relationship"] = source_meta.get("relationship", "")
        elif w.get("source_class") == "contextual":
            entry["engagement_type"] = source_meta.get("engagement_type", "")

        catalog.append(entry)

    return json.dumps(
        {
            "author_id": author_id,
            "total_works": len(catalog),
            "filter": source_class_filter,
            "works": catalog,
        },
        indent=2,
    )


async def handle_library_stats(
    arguments: dict[str, Any],
    *,
    storage: StorageManager,
) -> str:
    """Handle the library_stats MCP tool call.

    Returns:
        JSON with collection statistics: works, chunks, graph, embeddings.
    """
    # Works statistics
    work_stats_rows = await storage.pg.fetch_all(
        """SELECT
             count(*) AS total_works,
             count(*) FILTER (WHERE source_class = 'primary') AS primary_works,
             count(*) FILTER (WHERE source_class = 'secondary') AS secondary_works,
             count(*) FILTER (WHERE source_class = 'contextual') AS contextual_works,
             count(*) FILTER (WHERE source_class = 'tertiary') AS tertiary_works,
             count(*) FILTER (WHERE source_class = 'reference') AS reference_works,
             coalesce(sum(word_count), 0) AS total_words,
             count(DISTINCT author) AS unique_authors
           FROM works"""
    )
    work_stats = dict(work_stats_rows[0]) if work_stats_rows else {}

    # Chunk statistics
    chunk_stats_rows = await storage.pg.fetch_all(
        """SELECT
             count(*) AS total_chunks,
             count(*) FILTER (WHERE granularity = 'macro') AS macro_chunks,
             count(*) FILTER (WHERE granularity = 'meso') AS meso_chunks,
             count(*) FILTER (WHERE granularity = 'micro') AS micro_chunks
           FROM chunks"""
    )
    chunk_stats = dict(chunk_stats_rows[0]) if chunk_stats_rows else {}

    # Embedding statistics
    embedding_stats_rows = await storage.pg.fetch_all(
        """SELECT
             count(*) AS total_embeddings,
             count(DISTINCT chunk_id) AS chunks_with_embeddings,
             count(DISTINCT provider) AS providers,
             count(DISTINCT model) AS models
           FROM chunk_embeddings"""
    )
    embedding_stats = dict(embedding_stats_rows[0]) if embedding_stats_rows else {}

    # Graph statistics from Neo4j
    graph_stats: dict[str, Any] = {}
    try:
        node_counts = await storage.neo4j.execute_read(
            """MATCH (n)
            RETURN labels(n) AS labels, count(n) AS count"""
        )
        label_counts: dict[str, int] = {}
        for record in node_counts:
            for label in record.get("labels", []):
                label_counts[label] = label_counts.get(label, 0) + record.get("count", 0)

        edge_counts = await storage.neo4j.execute_read(
            """MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count"""
        )
        rel_counts: dict[str, int] = {record["rel_type"]: record["count"] for record in edge_counts}

        graph_stats = {
            "node_counts": label_counts,
            "edge_counts": rel_counts,
            "total_nodes": sum(label_counts.values()),
            "total_edges": sum(rel_counts.values()),
        }
    except Exception as exc:
        log.warning("graph_stats_failed", error=str(exc))
        graph_stats = {"error": str(exc)}

    # Thematic index stats
    thematic_rows = await storage.pg.fetch_all(
        "SELECT count(*) AS total_themes FROM thematic_entries"
    )
    thematic_count = dict(thematic_rows[0]).get("total_themes", 0) if thematic_rows else 0

    # Voice profile stats
    voice_rows = await storage.pg.fetch_all(
        """SELECT count(DISTINCT author_id) AS authors_with_profiles,
                  count(*) AS total_profile_versions
           FROM voice_profiles"""
    )
    voice_stats = dict(voice_rows[0]) if voice_rows else {}

    # Embedding coverage
    chunks_with_embeds = embedding_stats.get("chunks_with_embeddings", 0)
    total_chunks = chunk_stats.get("total_chunks", 0)
    coverage = round(chunks_with_embeds / total_chunks * 100, 1) if total_chunks > 0 else 0.0

    stats = {
        "works": work_stats,
        "chunks": chunk_stats,
        "embeddings": {
            **embedding_stats,
            "coverage_percent": coverage,
        },
        "graph": graph_stats,
        "thematic_index": {
            "total_themes": thematic_count,
        },
        "voice_profiles": voice_stats,
    }

    return json.dumps(stats, indent=2, default=str)


async def handle_health_check(
    arguments: dict[str, Any],
    *,
    storage: StorageManager,
    embedding_provider: Any | None = None,
) -> str:
    """Handle the health_check MCP tool call.

    Tests connectivity to PostgreSQL, Neo4j, and the embedding provider.

    Returns:
        JSON with health status for each backend.
    """
    from author_library.embeddings.base import EmbeddingProvider

    health: dict[str, Any] = {}

    # PostgreSQL health
    pg_ok = await storage.pg.health_check()
    health["postgres"] = {"status": "healthy" if pg_ok else "unhealthy"}

    # Neo4j health
    neo4j_ok = await storage.neo4j.health_check()
    health["neo4j"] = {"status": "healthy" if neo4j_ok else "unhealthy"}

    # Embedding provider health
    if embedding_provider is not None and isinstance(embedding_provider, EmbeddingProvider):
        try:
            result = await embedding_provider.embed_text("health check")
            health["embedding"] = {
                "status": "healthy",
                "provider": embedding_provider.provider_name,
                "model": embedding_provider.model_name,
                "dimensions": result.dimensions,
            }
        except Exception as exc:
            health["embedding"] = {
                "status": "unhealthy",
                "provider": embedding_provider.provider_name,
                "error": str(exc),
            }
    else:
        health["embedding"] = {"status": "not_configured"}

    all_healthy = all(
        v.get("status") == "healthy" for v in health.values() if v.get("status") != "not_configured"
    )
    health["overall"] = "healthy" if all_healthy else "degraded"

    return json.dumps(health, indent=2)


async def handle_audit_library(
    arguments: dict[str, Any],
    *,
    storage: StorageManager,
) -> str:
    """Handle the audit_library MCP tool call.

    Runs a full library health check covering:
    - Per-work stats (chunks, embeddings, entities, orphaned chunks)
    - PG/Neo4j consistency
    - Theme graph quality
    - Classification anomalies
    - Chunk noise (micro/nano below threshold)

    Returns:
        JSON report with overall_status, per-work breakdown, and recommendations.
    """
    from author_library.graph.backfill import check_pg_neo4j_consistency

    recommendations: list[str] = []
    work_audit: list[dict[str, Any]] = []
    has_errors = False
    has_warnings = False

    # ------------------------------------------------------------------
    # 1. Per-work stats: chunks, embeddings, entity edges, orphaned chunks
    # ------------------------------------------------------------------
    works_rows = await storage.pg.fetch_all(
        "SELECT work_id, title, source_class, author, genre_tags FROM works ORDER BY work_id"
    )

    if not works_rows:
        return json.dumps(
            {
                "overall_status": "healthy",
                "works": [],
                "graph": {},
                "pg_neo4j": {
                    "is_consistent": True,
                    "missing_works": [],
                    "chunk_delta": [],
                    "work_property_delta": [],
                },
                "chunk_noise": {
                    "sub_50_chunks": 0,
                    "total_chunks": 0,
                    "warning_work_ids": [],
                    "informational_work_ids": [],
                },
                "recommendations": ["Library is empty — no works have been ingested."],
            },
            indent=2,
        )

    # Chunk counts per work
    chunk_rows = await storage.pg.fetch_all(
        "SELECT work_id, COUNT(*) AS chunk_count FROM chunks GROUP BY work_id"
    )
    chunk_counts = {r["work_id"]: int(r["chunk_count"]) for r in chunk_rows}

    # Embedding coverage per work (distinct chunk ids with embeddings)
    embed_rows = await storage.pg.fetch_all(
        """SELECT c.work_id, COUNT(DISTINCT ce.chunk_id) AS embedded_chunks
           FROM chunks c
           LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id
           GROUP BY c.work_id"""
    )
    embed_counts = {r["work_id"]: int(r["embedded_chunks"]) for r in embed_rows}

    # Micro/nano chunk noise per work (< 50 chars)
    noise_rows = await storage.pg.fetch_all(
        """SELECT work_id, COUNT(*) AS noise_count
           FROM chunks
           WHERE length(text) < 50
           GROUP BY work_id"""
    )
    noise_counts = {r["work_id"]: int(r["noise_count"]) for r in noise_rows}
    chunk_noise_warning_work_ids: list[str] = []
    chunk_noise_informational_work_ids: list[str] = []

    # Entity edge counts per work in Neo4j
    entity_counts: dict[str, int] = {}
    orphan_counts: dict[str, int] = {}
    try:
        entity_rows = await storage.neo4j.execute_read(
            "MATCH (c:Chunk)-[r:EXPLORES_THEME|MAKES_ARGUMENT|"
            "ATTRIBUTED_BY_CRITIC|CONCEPT_USED_IN|REFERENCES_PERSON]->() "
            "RETURN c.work_id AS work_id, COUNT(r) AS entity_edges"
        )
        for r in entity_rows:
            entity_counts[r["work_id"]] = int(r["entity_edges"])

        # Orphaned chunks: in Neo4j but no entity relationships
        orphan_rows = await storage.neo4j.execute_read(
            "MATCH (c:Chunk) "
            "WHERE NOT (c)-[:EXPLORES_THEME|MAKES_ARGUMENT|ATTRIBUTED_BY_CRITIC|"
            "CONCEPT_USED_IN|REFERENCES_PERSON]->() "
            "RETURN c.work_id AS work_id, COUNT(c) AS orphan_count"
        )
        for r in orphan_rows:
            orphan_counts[r["work_id"]] = int(r["orphan_count"])
    except Exception as exc:
        log.warning("audit_entity_counts_failed", error=str(exc))

    for w in works_rows:
        work_id = w["work_id"]
        total = chunk_counts.get(work_id, 0)
        embedded = embed_counts.get(work_id, 0)
        entities = entity_counts.get(work_id, 0)
        orphans = orphan_counts.get(work_id, 0)
        noise = noise_counts.get(work_id, 0)

        work_warnings: list[str] = []
        work_info: list[str] = []

        if total == 0:
            work_warnings.append("no_chunks")
            has_errors = True
        elif embedded == 0:
            work_warnings.append("no_embeddings")
            has_errors = True
        elif embedded < total:
            work_warnings.append(f"partial_embeddings ({embedded}/{total})")
            has_warnings = True

        if total > 0 and entities == 0:
            work_warnings.append("no_entity_extraction")
            has_warnings = True

        noise_classification = _classify_chunk_noise(
            noise,
            total,
            w.get("genre_tags"),
        )
        if noise_classification is not None:
            severity, message = noise_classification
            if severity == "warning":
                work_warnings.append(message)
                chunk_noise_warning_work_ids.append(work_id)
                has_warnings = True
            else:
                work_info.append(message)
                chunk_noise_informational_work_ids.append(work_id)

        if _entity_edge_gap_is_warning(total, orphans):
            work_warnings.append(
                f"entity_edge_coverage_gap ({orphans}/{total} Neo4j chunks without entity edges)"
            )
            has_warnings = True

        work_audit.append(
            {
                "work_id": work_id,
                "title": w.get("title", ""),
                "source_class": w.get("source_class", ""),
                "chunks": total,
                "embeddings": embedded,
                "entities": entities,
                "neo4j_chunks_without_entity_edges": orphans,
                "orphaned_neo4j_chunks": orphans,
                "noise_chunks": noise,
                "warnings": work_warnings,
                "info": work_info,
            }
        )

    # ------------------------------------------------------------------
    # 2. PG / Neo4j consistency
    # ------------------------------------------------------------------
    pg_neo4j: dict[str, Any] = {}
    try:
        consistency = await check_pg_neo4j_consistency(storage)
        missing = consistency.get("missing_from_neo4j", [])
        extra = consistency.get("extra_in_neo4j", [])
        chunk_delta = [c for c in consistency.get("chunk_counts", []) if not c.get("in_sync")]
        work_property_delta = consistency.get("work_property_delta", [])
        pg_only_chunk_drift = [c for c in chunk_delta if int(c.get("pg_only_chunk_count") or 0) > 0]
        neo4j_only_chunk_drift = [
            c for c in chunk_delta if int(c.get("neo4j_only_chunk_count") or 0) > 0
        ]
        pg_neo4j = {
            "is_consistent": consistency.get("is_consistent", False),
            "pg_works": consistency.get("pg_work_count", 0),
            "neo4j_works": consistency.get("neo4j_work_count", 0),
            "missing_from_neo4j": missing,
            "extra_in_neo4j": extra,
            "chunk_delta": chunk_delta,
            "work_property_delta": work_property_delta,
        }
        if missing or extra or chunk_delta or work_property_delta:
            has_warnings = True
            if missing:
                recommendations.append(
                    f"{len(missing)} works in PG are missing from Neo4j — run backfill."
                )
            if extra:
                recommendations.append(
                    f"{len(extra)} Neo4j works have no PG record — check for orphaned graph data."
                )
            if pg_only_chunk_drift:
                pg_only_count = sum(
                    int(work.get("pg_only_chunk_count") or 0) for work in pg_only_chunk_drift
                )
                pg_only_work_ids = [work["work_id"] for work in pg_only_chunk_drift]
                recommendations.append(
                    f"{pg_only_count} PostgreSQL chunks across "
                    f"{len(pg_only_work_ids)} works are missing from Neo4j — run the "
                    "non-destructive default graph/entity backfill for each affected work: "
                    f"{_graph_entity_backfill_commands(pg_only_work_ids)}. "
                    "Do not pass --deduplicate-themes."
                )
            if neo4j_only_chunk_drift:
                neo4j_only_count = sum(
                    int(work.get("neo4j_only_chunk_count") or 0) for work in neo4j_only_chunk_drift
                )
                neo4j_only_work_ids = [work["work_id"] for work in neo4j_only_chunk_drift]
                recommendations.append(
                    f"{neo4j_only_count} Neo4j chunks across "
                    f"{len(neo4j_only_work_ids)} works have no PostgreSQL row — correct the "
                    "scope of scripts/cleanup_neo4j_orphans.py to: "
                    f"{', '.join(neo4j_only_work_ids[:3])}"
                    + (" ..." if len(neo4j_only_work_ids) > 3 else "")
                    + "; then inspect a dry-run. Deletion requires explicit approval."
                )
            if work_property_delta:
                work_ids = [work["work_id"] for work in work_property_delta]
                recommendations.append(
                    f"Mirrored Work metadata differs for {len(work_ids)} works "
                    f"({', '.join(work_ids)}) — run a reviewed, targeted, idempotent "
                    "Work-node re-sync from PostgreSQL using upsert_work_node(). "
                    "Do not run chunk/entity backfill for metadata-only drift."
                )
    except Exception as exc:
        log.warning("audit_consistency_failed", error=str(exc))
        pg_neo4j = {"error": str(exc)}
        has_warnings = True
        recommendations.append(
            "PG/Neo4j consistency could not be checked — resolve the reported error "
            "and rerun the audit."
        )

    # ------------------------------------------------------------------
    # 3. Theme graph quality
    # ------------------------------------------------------------------
    graph_audit: dict[str, Any] = {}
    try:
        theme_rows = await storage.neo4j.execute_read(
            """MATCH (t:Theme)
               OPTIONAL MATCH (c:Chunk)-[:EXPLORES_THEME]->(t)
               RETURN t.canonical_name AS theme,
                      COUNT(c) AS chunk_count
               ORDER BY chunk_count DESC"""
        )
        total_themes = len(theme_rows)
        singletons = [r["theme"] for r in theme_rows if int(r["chunk_count"]) <= 1]
        avg_connectivity = (
            sum(int(r["chunk_count"]) for r in theme_rows) / total_themes
            if total_themes > 0
            else 0.0
        )

        # Person and Concept nodes
        entity_type_rows = await storage.neo4j.execute_read(
            """MATCH (n)
               WHERE n:Person OR n:Concept OR n:Argument
               RETURN labels(n)[0] AS type, COUNT(n) AS count"""
        )
        entity_type_counts: dict[str, int] = {r["type"]: int(r["count"]) for r in entity_type_rows}

        graph_audit = {
            "total_themes": total_themes,
            "singleton_themes": len(singletons),
            "avg_chunk_connectivity": round(avg_connectivity, 2),
            "entity_counts": entity_type_counts,
        }
        if singletons and len(singletons) > total_themes * 0.3:
            has_warnings = True
            recommendations.append(
                f"{len(singletons)} singleton themes detected — consider deduplication."
            )
    except Exception as exc:
        log.warning("audit_graph_quality_failed", error=str(exc))
        graph_audit = {"error": str(exc)}

    # ------------------------------------------------------------------
    # 4. Classification anomalies: author == subject but not primary
    # ------------------------------------------------------------------
    try:
        anomaly_rows = await storage.pg.fetch_all(
            """SELECT work_id, title, author, source_class,
                      source_metadata->>'subject_author_id' AS subject_author_id
               FROM works
               WHERE source_class != 'primary'
                 AND source_metadata->>'subject_author_id' IS NOT NULL
                 AND lower(author) = lower(source_metadata->>'subject_author_id')"""
        )
        if anomaly_rows:
            has_warnings = True
            for row in anomaly_rows:
                recommendations.append(
                    f"Classification anomaly: '{row['work_id']}' — author matches "
                    f"subject_author_id but source_class='{row['source_class']}'. "
                    "Review the catalog record and reclassify it if needed."
                )
    except Exception as exc:
        log.warning("audit_classification_anomaly_check_failed", error=str(exc))

    # ------------------------------------------------------------------
    # 5. Global recommendations
    # ------------------------------------------------------------------
    works_without_chunks = [w["work_id"] for w in work_audit if "no_chunks" in w["warnings"]]
    if works_without_chunks:
        recommendations.append(
            f"{len(works_without_chunks)} works have no chunks — re-run ingestion: "
            f"{', '.join(works_without_chunks[:3])}"
            + (" ..." if len(works_without_chunks) > 3 else "")
        )

    works_missing_embeddings = [
        w["work_id"] for w in work_audit if "no_embeddings" in w["warnings"]
    ]
    if works_missing_embeddings:
        recommendations.append(
            f"{len(works_missing_embeddings)} works have no embeddings — "
            f"re-run embed step: {', '.join(works_missing_embeddings[:3])}"
            + (" ..." if len(works_missing_embeddings) > 3 else "")
        )

    works_with_partial_embeddings = [
        w["work_id"]
        for w in work_audit
        if any(warning.startswith("partial_embeddings") for warning in w["warnings"])
    ]
    if works_with_partial_embeddings:
        recommendations.append(
            f"{len(works_with_partial_embeddings)} works have partial embedding coverage — "
            f"re-run embed step: {', '.join(works_with_partial_embeddings[:3])}"
            + (" ..." if len(works_with_partial_embeddings) > 3 else "")
        )

    works_no_entities = [
        w["work_id"] for w in work_audit if "no_entity_extraction" in w["warnings"]
    ]
    if works_no_entities:
        recommendations.append(
            f"{len(works_no_entities)} works have no entity extraction — "
            f"run backfill_entities.py: {', '.join(works_no_entities[:3])}"
            + (" ..." if len(works_no_entities) > 3 else "")
        )

    works_with_excessive_noise = chunk_noise_warning_work_ids
    if works_with_excessive_noise:
        recommendations.append(
            f"{len(works_with_excessive_noise)} works have excessive noise chunks "
            "(< 50 chars) — review chunk boundaries and re-chunk genuine noise: "
            f"{', '.join(works_with_excessive_noise[:3])}"
            + (" ..." if len(works_with_excessive_noise) > 3 else "")
        )

    works_with_entity_edge_gaps = [
        w["work_id"]
        for w in work_audit
        if _entity_edge_gap_is_warning(
            int(w["chunks"]),
            int(w["neo4j_chunks_without_entity_edges"]),
        )
    ]
    if works_with_entity_edge_gaps:
        recommendations.append(
            f"{len(works_with_entity_edge_gaps)} works have material Neo4j entity-edge "
            "coverage gaps — review extraction coverage; only if extraction is confirmed "
            "incomplete, run the non-destructive default graph/entity backfill: "
            f"{_graph_entity_backfill_commands(works_with_entity_edge_gaps)}. "
            "Do not pass --deduplicate-themes."
        )

    # ------------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------------
    if has_errors:
        overall_status = "errors"
    elif has_warnings:
        overall_status = "warnings"
    else:
        overall_status = "healthy"

    if overall_status == "healthy" and not recommendations:
        recommendations.append("Library is healthy — no issues detected.")
    elif overall_status != "healthy" and not recommendations:
        recommendations.append(
            "The audit found unresolved issues — inspect the warning and error fields, "
            "correct them, and rerun the audit."
        )

    report = {
        "overall_status": overall_status,
        "works": work_audit,
        "graph": graph_audit,
        "pg_neo4j": pg_neo4j,
        "chunk_noise": {
            "sub_50_chunks": sum(noise_counts.values()),
            "total_chunks": sum(chunk_counts.values()),
            "warning_work_ids": chunk_noise_warning_work_ids,
            "informational_work_ids": chunk_noise_informational_work_ids,
        },
        "recommendations": recommendations,
    }

    return json.dumps(report, indent=2, default=str)
