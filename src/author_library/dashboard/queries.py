"""Dashboard stat queries — pure async reads against PG and Neo4j.

All functions accept the storage sub-objects directly so they can be
tested independently of the full StorageManager lifecycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from author_library.storage.neo4j import Neo4jConnection
    from author_library.storage.postgres import PostgresPool


async def get_library_overview(pg: "PostgresPool") -> dict[str, Any]:
    """Aggregate counts: works by class, chunks, embeddings, voice profiles."""
    row = await pg.fetch_one(
        """
        SELECT
            count(*)                                                           AS total_works,
            count(*) FILTER (WHERE source_class = 'primary')                  AS primary_works,
            count(*) FILTER (WHERE source_class = 'secondary')                AS secondary_works,
            count(*) FILTER (WHERE source_class = 'contextual')               AS contextual_works,
            count(*) FILTER (WHERE source_class = 'tertiary')                 AS tertiary_works,
            count(*) FILTER (WHERE source_class = 'personal')                 AS personal_works,
            coalesce(sum(word_count), 0)                                       AS total_words,
            count(DISTINCT author)                                             AS unique_authors,
            max(ingestion_date)                                                AS last_ingestion_date
        FROM works
        """
    )
    overview: dict[str, Any] = dict(row) if row else {}

    chunk_row = await pg.fetch_one("SELECT count(*) AS total_chunks FROM chunks")
    overview["total_chunks"] = dict(chunk_row)["total_chunks"] if chunk_row else 0

    emb_row = await pg.fetch_one(
        "SELECT count(DISTINCT chunk_id) AS embedded_chunks FROM chunk_embeddings"
    )
    embedded = dict(emb_row)["embedded_chunks"] if emb_row else 0
    total = overview.get("total_chunks", 0) or 1
    overview["embedding_coverage_pct"] = round(100.0 * embedded / total, 1)

    vp_row = await pg.fetch_one(
        "SELECT count(*) AS voice_profile_count FROM voice_profiles WHERE is_current = TRUE"
    )
    overview["voice_profile_count"] = dict(vp_row)["voice_profile_count"] if vp_row else 0

    if overview.get("last_ingestion_date"):
        overview["last_ingestion_date"] = str(overview["last_ingestion_date"])

    return overview


async def get_per_work_details(pg: "PostgresPool") -> list[dict[str, Any]]:
    """Return one row per work with chunk count, embedding %, and confidence."""
    rows = await pg.fetch_all(
        """
        SELECT
            w.work_id,
            w.title,
            w.author,
            w.source_class,
            w.ingestion_date::text                                           AS ingestion_date,
            (w.source_metadata->>'classification_confidence')::float        AS classification_confidence,
            count(c.id)                                                      AS chunk_count,
            count(ce.chunk_id)                                               AS embedded_count
        FROM works w
        LEFT JOIN chunks c            ON c.work_id = w.work_id
        LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.chunk_id
        GROUP BY w.work_id, w.title, w.author, w.source_class,
                 w.ingestion_date, w.source_metadata
        ORDER BY w.ingestion_date DESC, w.title
        """
    )
    result: list[dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        chunk_count = d["chunk_count"] or 0
        embedded = d["embedded_count"] or 0
        d["embedding_pct"] = round(100.0 * embedded / chunk_count, 1) if chunk_count else 0.0
        result.append(d)
    return result


async def get_graph_stats(neo4j: "Neo4jConnection") -> dict[str, Any]:
    """Node/edge counts and top shared themes from Neo4j."""
    stats: dict[str, Any] = {"error": None}

    try:
        node_rows = await neo4j.execute_read(
            "MATCH (n) RETURN labels(n) AS lbl, count(n) AS cnt"
        )
        node_counts: dict[str, int] = {}
        for r in node_rows:
            for label in r.get("lbl", []):
                node_counts[label] = node_counts.get(label, 0) + r.get("cnt", 0)
        stats["node_counts"] = node_counts
        stats["total_nodes"] = sum(node_counts.values())

        edge_rows = await neo4j.execute_read(
            "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt"
        )
        edge_counts: dict[str, int] = {r["rel_type"]: r["cnt"] for r in edge_rows}
        stats["edge_counts"] = edge_counts
        stats["total_edges"] = sum(edge_counts.values())

        theme_rows = await neo4j.execute_read(
            """
            MATCH (c:Chunk)-[:EXPLORES_THEME]->(t:Theme)
            WITH t.canonical_name AS name, count(DISTINCT c.work_id) AS work_count,
                 count(c) AS chunk_count
            ORDER BY work_count DESC, chunk_count DESC
            LIMIT 15
            RETURN name, work_count, chunk_count
            """
        )
        stats["top_themes"] = [dict(r) for r in theme_rows]

    except Exception as exc:
        stats["error"] = str(exc)

    return stats
