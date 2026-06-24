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
    overview: dict[str, Any] = dict(row)

    chunk_row = await pg.fetch_one("SELECT count(*) AS total_chunks FROM chunks")
    overview["total_chunks"] = dict(chunk_row)["total_chunks"] if chunk_row else 0

    emb_row = await pg.fetch_one(
        "SELECT count(DISTINCT chunk_id) AS embedded_chunks FROM chunk_embeddings"
    )
    embedded = dict(emb_row)["embedded_chunks"] if emb_row else 0
    total = overview.get("total_chunks", 0)
    overview["embedding_coverage_pct"] = round(100.0 * embedded / total, 1) if total else 0.0

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
        LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id
        GROUP BY w.work_id, w.title, w.author, w.source_class, w.ingestion_date, w.source_metadata
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
            d = dict(r)
            for label in d.get("lbl", []):
                node_counts[label] = node_counts.get(label, 0) + d.get("cnt", 0)
        stats["node_counts"] = node_counts

        total_row = await neo4j.execute_read("MATCH (n) RETURN count(n) AS cnt")
        stats["total_nodes"] = total_row[0]["cnt"] if total_row else 0

        edge_rows = await neo4j.execute_read(
            "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt"
        )
        edge_counts: dict[str, int] = {dict(r)["rel_type"]: dict(r)["cnt"] for r in edge_rows}
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


async def get_voice_profiles(pg: "PostgresPool") -> list[dict[str, Any]]:
    """Return all current voice profiles with author name and work count."""
    rows = await pg.fetch_all(
        """
        SELECT
            vp.author_id,
            a.canonical_name,
            vp.version,
            vp.profile,
            vp.created_at::text                             AS created_at,
            count(w.work_id)                                AS work_count
        FROM voice_profiles vp
        JOIN authors a ON a.id = vp.author_id
        LEFT JOIN works w ON w.work_id LIKE vp.author_id || '--%'
                          AND w.source_class = 'primary'
        WHERE vp.is_current = TRUE
        GROUP BY vp.author_id, a.canonical_name, vp.version, vp.profile, vp.created_at
        ORDER BY a.canonical_name
        """
    )
    result = []
    for row in rows:
        d = dict(row)
        if isinstance(d["profile"], str):
            import json as _json
            d["profile"] = _json.loads(d["profile"])
        result.append(d)
    return result


async def get_work_detail(
    pg: "PostgresPool", neo4j: "Neo4jConnection", work_id: str
) -> dict[str, Any] | None:
    """Full work metadata, chunk breakdown, top Neo4j themes, sample macro chunks."""
    row = await pg.fetch_one(
        """
        SELECT work_id, title, author, source_class, publication_year,
               original_publication_year, publisher, format_ingested,
               word_count, genre_tags, subject_headings, ingestion_date::text,
               source_metadata, notes
        FROM works WHERE work_id = $1
        """,
        work_id,
    )
    if not row:
        return None
    detail: dict[str, Any] = dict(row)
    if isinstance(detail["source_metadata"], str):
        import json as _json
        detail["source_metadata"] = _json.loads(detail["source_metadata"])

    breakdown_rows = await pg.fetch_all(
        "SELECT granularity, count(*) AS cnt FROM chunks WHERE work_id = $1 GROUP BY granularity",
        work_id,
    )
    detail["chunk_breakdown"] = {dict(r)["granularity"]: dict(r)["cnt"] for r in breakdown_rows}

    themes: list[dict[str, Any]] = []
    try:
        theme_rows = await neo4j.execute_read(
            """
            MATCH (c:Chunk {work_id: $work_id})-[:EXPLORES_THEME]->(t:Theme)
            RETURN t.canonical_name AS name, count(c) AS chunk_count
            ORDER BY chunk_count DESC LIMIT 12
            """,
            {"work_id": work_id},
        )
        themes = [{"name": r["name"], "chunk_count": r["chunk_count"]} for r in theme_rows]
    except Exception:
        pass
    detail["themes"] = themes

    chunk_rows = await pg.fetch_all(
        """
        SELECT text, annotation, chapter
        FROM chunks WHERE work_id = $1 AND granularity = 'macro'
        ORDER BY position LIMIT 3
        """,
        work_id,
    )
    detail["sample_chunks"] = [dict(r) for r in chunk_rows]
    return detail


async def get_all_themes(pg: "PostgresPool") -> list[dict[str, Any]]:
    """All thematic entries with cross-work appearance counts."""
    rows = await pg.fetch_all(
        """
        SELECT
            te.id::text                                     AS id,
            te.author_id,
            a.canonical_name                                AS author_name,
            te.theme,
            te.author_stance,
            te.related_themes,
            count(DISTINCT ta.work_id)                      AS work_count,
            count(ta.id)                                    AS appearance_count
        FROM thematic_entries te
        JOIN authors a ON a.id = te.author_id
        LEFT JOIN thematic_appearances ta ON ta.entry_id = te.id
        GROUP BY te.id, te.author_id, a.canonical_name, te.theme,
                 te.author_stance, te.related_themes
        ORDER BY work_count DESC, te.theme
        """
    )
    return [dict(r) for r in rows]


async def get_theme_detail(
    pg: "PostgresPool", neo4j: "Neo4jConnection", entry_id: str
) -> dict[str, Any] | None:
    """Full theme detail: metadata + per-work appearances + chunk quotes from Neo4j."""
    row = await pg.fetch_one(
        """
        SELECT te.id::text AS id, te.author_id, a.canonical_name AS author_name,
               te.theme, te.author_stance, te.related_themes, te.key_passages
        FROM thematic_entries te
        JOIN authors a ON a.id = te.author_id
        WHERE te.id = $1::uuid
        """,
        entry_id,
    )
    if not row:
        return None
    detail: dict[str, Any] = dict(row)

    appearance_rows = await pg.fetch_all(
        """
        SELECT ta.work_id, w.title, w.author, ta.chapters, ta.treatment_summary
        FROM thematic_appearances ta
        JOIN works w ON w.work_id = ta.work_id
        WHERE ta.entry_id = $1::uuid
        ORDER BY w.title
        """,
        entry_id,
    )
    appearances = [dict(r) for r in appearance_rows]

    theme_name = detail["theme"]
    for appearance in appearances:
        quotes: list[str] = []
        try:
            quote_rows = await neo4j.execute_read(
                """
                MATCH (c:Chunk {work_id: $work_id})-[:EXPLORES_THEME]->(t:Theme)
                WHERE t.canonical_name = $theme
                RETURN c.text AS text
                ORDER BY c.position LIMIT 2
                """,
                {"work_id": appearance["work_id"], "theme": theme_name},
            )
            quotes = [r["text"][:400] for r in quote_rows if r.get("text")]
        except Exception:
            pass
        appearance["quotes"] = quotes

    detail["appearances"] = appearances
    return detail
