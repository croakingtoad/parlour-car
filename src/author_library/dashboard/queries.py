"""Dashboard stat queries — pure async reads against PG and Neo4j.

All functions accept the storage sub-objects directly so they can be
tested independently of the full StorageManager lifecycle.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

import structlog

log = structlog.get_logger(__name__)

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


async def get_author_health(pg: "PostgresPool") -> list[dict[str, Any]]:
    """Return a health row per author: work counts, slug consistency, voice profile status."""
    rows = await pg.fetch_all(
        """
        SELECT
            a.id                                                                AS author_id,
            a.canonical_name,
            -- Works whose work_id prefix matches the author slug
            count(DISTINCT w.work_id) FILTER (
                WHERE SPLIT_PART(w.work_id, '--', 1) = a.id
            )                                                                   AS works_matching_slug,
            -- Works claiming this author via subject_author_id metadata
            count(DISTINCT w2.work_id) FILTER (
                WHERE w2.source_class = 'primary'
            )                                                                   AS primary_works_by_said,
            -- Current voice profile confidence
            (vp.profile->>'confidence')::float                                  AS vp_confidence,
            vp.version                                                          AS vp_version,
            -- Stale subject_author_id refs (works with old slugs in metadata)
            count(DISTINCT w3.work_id) FILTER (
                WHERE w3.source_metadata->>'subject_author_id' != a.id
                  AND w3.source_metadata->>'subject_author_id' IS NOT NULL
                  AND SPLIT_PART(w3.work_id, '--', 1) = a.id
            )                                                                   AS said_mismatches
        FROM authors a
        LEFT JOIN works w   ON SPLIT_PART(w.work_id, '--', 1) = a.id
        LEFT JOIN works w2  ON w2.source_metadata->>'subject_author_id' = a.id
        LEFT JOIN works w3  ON SPLIT_PART(w3.work_id, '--', 1) = a.id
        LEFT JOIN voice_profiles vp ON vp.author_id = a.id AND vp.is_current = TRUE
        GROUP BY a.id, a.canonical_name, vp.profile, vp.version
        ORDER BY a.canonical_name
        """
    )
    result = []
    for row in rows:
        d = dict(row)
        issues = []
        if d["works_matching_slug"] == 0:
            issues.append("no works match slug")
        if d["said_mismatches"] and d["said_mismatches"] > 0:
            issues.append(f"{d['said_mismatches']} subject_author_id mismatch(es)")
        if d["vp_confidence"] is None:
            issues.append("no voice profile")
        elif d["vp_confidence"] < 0.75:
            issues.append(f"low confidence ({d['vp_confidence']:.0%})")
        d["issues"] = issues
        d["status"] = "error" if any("no works" in i or "no voice" in i for i in issues) \
                      else "warn" if issues else "ok"
        result.append(d)
    return result

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
        LEFT JOIN works w ON SPLIT_PART(w.work_id, '--', 1) = vp.author_id
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
            d["profile"] = json.loads(d["profile"])
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
        detail["source_metadata"] = json.loads(detail["source_metadata"])

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
    except Exception as exc:
        log.warning("get_work_detail_neo4j_failed", work_id=work_id, error=str(exc))
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
    try:
        uuid.UUID(entry_id)
    except ValueError:
        return None

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
    work_ids = [ap["work_id"] for ap in appearances]

    quotes_by_work: dict[str, list[str]] = {ap["work_id"]: [] for ap in appearances}
    if work_ids:
        try:
            quote_rows = await neo4j.execute_read(
                """
                MATCH (c:Chunk)-[:EXPLORES_THEME]->(t:Theme)
                WHERE c.work_id IN $work_ids AND t.canonical_name = $theme
                RETURN c.work_id AS work_id, c.text AS text
                ORDER BY c.work_id, c.position LIMIT 40
                """,
                {"work_ids": work_ids, "theme": theme_name},
            )
            for r in quote_rows:
                wid = r["work_id"]
                if wid in quotes_by_work and len(quotes_by_work[wid]) < 2:
                    text = r["text"]
                    if text:
                        quotes_by_work[wid].append(text[:400])
        except Exception as exc:
            log.warning("get_theme_detail_neo4j_failed", error=str(exc))

    for appearance in appearances:
        appearance["quotes"] = quotes_by_work.get(appearance["work_id"], [])

    detail["appearances"] = appearances
    return detail


async def get_pipeline_status(
    pg: "PostgresPool", neo4j: "Neo4jConnection"
) -> dict[str, Any]:
    """Return per-work pipeline completion and summary counts."""

    # All works with chunk + embedding counts
    work_rows = await pg.fetch_all(
        """
        SELECT
            w.work_id, w.title, w.author, w.source_class,
            w.ingestion_date::text                              AS ingestion_date,
            count(c.id)                                         AS chunk_count,
            count(ce.chunk_id)                                  AS embedded_count
        FROM works w
        LEFT JOIN chunks c            ON c.work_id = w.work_id
        LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.chunk_id
        GROUP BY w.work_id, w.title, w.author, w.source_class, w.ingestion_date
        ORDER BY w.ingestion_date DESC, w.title
        """
    )

    all_work_ids = [dict(r)["work_id"] for r in work_rows]

    # Works that have entity edges in Neo4j
    works_with_entities: set[str] = set()
    works_with_themes: set[str] = set()
    try:
        entity_rows = await neo4j.execute_read(
            """
            MATCH (c:Chunk)-[:MENTIONS|EXPLORES_THEME|MAKES_ARGUMENT]->()
            WHERE c.work_id IN $work_ids
            RETURN DISTINCT c.work_id AS work_id
            """,
            {"work_ids": all_work_ids},
        )
        works_with_entities = {r["work_id"] for r in entity_rows}

        theme_rows = await neo4j.execute_read(
            """
            MATCH (c:Chunk)-[:EXPLORES_THEME]->()
            WHERE c.work_id IN $work_ids
            RETURN DISTINCT c.work_id AS work_id
            """,
            {"work_ids": all_work_ids},
        )
        works_with_themes = {r["work_id"] for r in theme_rows}
    except Exception as exc:
        log.warning("pipeline_status_neo4j_failed", error=str(exc))

    # Current voice profiles
    vp_rows = await pg.fetch_all(
        "SELECT author_id FROM voice_profiles WHERE is_current = TRUE"
    )
    authors_with_profiles = {dict(r)["author_id"] for r in vp_rows}

    works: list[dict[str, Any]] = []
    pending_chunks = 0
    pending_embed = 0
    pending_entities = 0
    pending_themes = 0

    for row in work_rows:
        d = dict(row)
        wid = d["work_id"]
        chunk_count = d["chunk_count"] or 0
        embedded = d["embedded_count"] or 0

        author_slug = wid.split("--")[0]
        is_primary = d["source_class"] == "primary"

        stages = {
            "chunked":   chunk_count > 0,
            "embedded":  chunk_count > 0 and embedded == chunk_count,
            "entities":  wid in works_with_entities,
            "themes":    wid in works_with_themes,
            "voice":     (author_slug in authors_with_profiles) if is_primary else None,
        }

        embedding_pct = round(100.0 * embedded / chunk_count, 0) if chunk_count else 0.0

        if not stages["chunked"]:       pending_chunks += 1
        if not stages["embedded"]:      pending_embed += 1
        if not stages["entities"]:      pending_entities += 1
        if not stages["themes"]:        pending_themes += 1

        fully_done = (
            stages["chunked"] and stages["embedded"]
            and stages["entities"] and stages["themes"]
            and (stages["voice"] is not False)
        )

        works.append({
            "work_id":       wid,
            "title":         d["title"],
            "author":        d["author"],
            "source_class":  d["source_class"],
            "ingestion_date":d["ingestion_date"],
            "chunk_count":   chunk_count,
            "embedding_pct": embedding_pct,
            "stages":        stages,
            "fully_done":    fully_done,
        })

    total = len(works)
    fully_complete = sum(1 for w in works if w["fully_done"])

    return {
        "works": works,
        "summary": {
            "total":            total,
            "fully_complete":   fully_complete,
            "pending_chunks":   pending_chunks,
            "pending_embed":    pending_embed,
            "pending_entities": pending_entities,
            "pending_themes":   pending_themes,
        },
    }
