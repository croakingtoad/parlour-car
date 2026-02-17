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
    from author_library.config import Settings
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)


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
        authors.append({
            "author": r["author"],
            "work_count": r["work_count"],
            "primary_works": r["primary_count"],
            "secondary_works": r["secondary_count"],
            "contextual_works": r["contextual_count"],
            "tertiary_works": r["tertiary_count"],
            "total_words": r["total_words"] or 0,
            "year_range": f"{r['earliest_year']}-{r['latest_year']}",
        })

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

    return json.dumps({
        "author_id": author_id,
        "total_works": len(catalog),
        "filter": source_class_filter,
        "works": catalog,
    }, indent=2)


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
        rel_counts: dict[str, int] = {
            record["rel_type"]: record["count"]
            for record in edge_counts
        }

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
    coverage = (
        round(chunks_with_embeds / total_chunks * 100, 1) if total_chunks > 0 else 0.0
    )

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
        v.get("status") == "healthy"
        for v in health.values()
        if v.get("status") != "not_configured"
    )
    health["overall"] = "healthy" if all_healthy else "degraded"

    return json.dumps(health, indent=2)
