"""Dashboard health checks — one async function per check.

Each returns a CheckResult. run_all_checks() gathers them concurrently.
Checks are based on ingestion problems that have occurred in this project.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from author_library.storage.neo4j import Neo4jConnection
    from author_library.storage.postgres import PostgresPool

Status = Literal["ok", "warn", "error"]


@dataclass
class CheckResult:
    name: str
    status: Status
    label: str
    detail: str
    count: int = 0


async def check_pg_neo4j_sync(pg: "PostgresPool", neo4j: "Neo4jConnection") -> CheckResult:
    """Chunks in PG should have a corresponding Neo4j Chunk node."""
    try:
        pg_row = await pg.fetch_one("SELECT count(*) AS cnt FROM chunks")
        pg_count = dict(pg_row)["cnt"] if pg_row else 0

        neo4j_rows = await neo4j.execute_read("MATCH (c:Chunk) RETURN count(c) AS cnt")
        neo4j_count = neo4j_rows[0]["cnt"] if neo4j_rows else 0

        diff = abs(pg_count - neo4j_count)
        if diff == 0:
            return CheckResult(
                name="pg_neo4j_sync", status="ok",
                label="PG ↔ Neo4j sync",
                detail=f"{pg_count:,} chunks matched", count=0,
            )
        status: Status = "warn" if diff < 10 else "error"
        return CheckResult(
            name="pg_neo4j_sync", status=status,
            label="PG ↔ Neo4j sync",
            detail=f"PG={pg_count:,} vs Neo4j={neo4j_count:,} ({diff} orphans)",
            count=diff,
        )
    except Exception as exc:
        return CheckResult(
            name="pg_neo4j_sync", status="error",
            label="PG ↔ Neo4j sync", detail=f"Query failed: {exc}", count=-1,
        )


async def check_missing_embeddings(pg: "PostgresPool") -> CheckResult:
    """Every chunk should have an embedding vector."""
    try:
        row = await pg.fetch_one(
            """
            SELECT count(*) AS cnt
            FROM chunks c
            WHERE NOT EXISTS (
                SELECT 1 FROM chunk_embeddings ce WHERE ce.chunk_id = c.id
            )
            """
        )
        missing = dict(row)["cnt"] if row else 0
        if missing == 0:
            return CheckResult(
                name="missing_embeddings", status="ok",
                label="Embedding coverage", detail="All chunks have embeddings", count=0,
            )
        status: Status = "warn" if missing < 50 else "error"
        return CheckResult(
            name="missing_embeddings", status=status,
            label="Embedding coverage",
            detail=f"{missing:,} chunks missing embeddings", count=missing,
        )
    except Exception as exc:
        return CheckResult(
            name="missing_embeddings", status="error",
            label="Embedding coverage", detail=f"Query failed: {exc}", count=-1,
        )


async def check_unvoiced_primary_sources(pg: "PostgresPool") -> CheckResult:
    """Each distinct author with primary works should have a voice profile."""
    try:
        works_row = await pg.fetch_one(
            "SELECT count(DISTINCT author) AS cnt FROM works WHERE source_class = 'primary'"
        )
        primary_authors = dict(works_row)["cnt"] if works_row else 0

        vp_row = await pg.fetch_one(
            "SELECT count(*) AS cnt FROM voice_profiles WHERE is_current = TRUE"
        )
        profiled = dict(vp_row)["cnt"] if vp_row else 0

        missing = max(0, primary_authors - profiled)
        if missing == 0:
            return CheckResult(
                name="unvoiced_primary_sources", status="ok",
                label="Voice profiles",
                detail=f"{profiled} profile(s) for {primary_authors} primary author(s)",
                count=0,
            )
        return CheckResult(
            name="unvoiced_primary_sources", status="warn",
            label="Voice profiles",
            detail=f"{missing} primary author(s) missing voice profile",
            count=missing,
        )
    except Exception as exc:
        return CheckResult(
            name="unvoiced_primary_sources", status="error",
            label="Voice profiles", detail=f"Query failed: {exc}", count=-1,
        )


async def check_low_confidence_classifications(pg: "PostgresPool") -> CheckResult:
    """Works classified with < 0.90 confidence may be misclassified."""
    try:
        row = await pg.fetch_one(
            """
            SELECT count(*) AS cnt
            FROM works
            WHERE (source_metadata->>'classification_confidence')::float < 0.90
              AND source_metadata ? 'classification_confidence'
            """
        )
        flagged = dict(row)["cnt"] if row else 0
        if flagged == 0:
            return CheckResult(
                name="low_confidence_classifications", status="ok",
                label="Classification confidence",
                detail="All works classified at >= 0.90 confidence", count=0,
            )
        return CheckResult(
            name="low_confidence_classifications", status="warn",
            label="Classification confidence",
            detail=f"{flagged} work(s) classified below 0.90 — review source_class",
            count=flagged,
        )
    except Exception as exc:
        return CheckResult(
            name="low_confidence_classifications", status="error",
            label="Classification confidence", detail=f"Query failed: {exc}", count=-1,
        )


async def check_entity_extraction_gaps(
    pg: "PostgresPool", neo4j: "Neo4jConnection"
) -> CheckResult:
    """Primary works with chunks should have entity edges in Neo4j."""
    try:
        works_rows = await pg.fetch_all(
            "SELECT work_id FROM works WHERE source_class = 'primary'"
        )
        if not works_rows:
            return CheckResult(
                name="entity_extraction_gaps", status="ok",
                label="Entity extraction", detail="No primary works ingested", count=0,
            )

        work_ids = [dict(r)["work_id"] for r in works_rows]
        neo4j_rows = await neo4j.execute_read(
            """
            MATCH (c:Chunk)-[:MENTIONS|EXPLORES_THEME|MAKES_ARGUMENT]->()
            WHERE c.work_id IN $work_ids
            RETURN DISTINCT c.work_id AS work_id
            """,
            {"work_ids": work_ids},
        )
        works_with_entities = {r["work_id"] for r in neo4j_rows}
        gaps = [w for w in work_ids if w not in works_with_entities]

        if not gaps:
            return CheckResult(
                name="entity_extraction_gaps", status="ok",
                label="Entity extraction",
                detail=f"All {len(work_ids)} primary work(s) have entity edges",
                count=0,
            )
        preview = ", ".join(gaps[:3]) + ("..." if len(gaps) > 3 else "")
        return CheckResult(
            name="entity_extraction_gaps", status="warn",
            label="Entity extraction",
            detail=f"{len(gaps)} primary work(s) have no entity edges: {preview}",
            count=len(gaps),
        )
    except Exception as exc:
        return CheckResult(
            name="entity_extraction_gaps", status="error",
            label="Entity extraction", detail=f"Query failed: {exc}", count=-1,
        )


async def check_orphaned_theme_nodes(neo4j: "Neo4jConnection") -> CheckResult:
    """Theme nodes with no EXPLORES_THEME edges pointing to them are orphaned."""
    try:
        rows = await neo4j.execute_read(
            "MATCH (t:Theme) WHERE NOT ()-[:EXPLORES_THEME]->(t) RETURN count(t) AS cnt"
        )
        orphans = rows[0]["cnt"] if rows else 0
        if orphans == 0:
            return CheckResult(
                name="orphaned_theme_nodes", status="ok",
                label="Theme graph integrity",
                detail="No orphaned Theme nodes", count=0,
            )
        status: Status = "warn" if orphans < 20 else "error"
        return CheckResult(
            name="orphaned_theme_nodes", status=status,
            label="Theme graph integrity",
            detail=f"{orphans} Theme node(s) have no chunk connections",
            count=orphans,
        )
    except Exception as exc:
        return CheckResult(
            name="orphaned_theme_nodes", status="error",
            label="Theme graph integrity", detail=f"Query failed: {exc}", count=-1,
        )


async def check_theme_coverage(
    pg: "PostgresPool", neo4j: "Neo4jConnection"
) -> CheckResult:
    """Primary works should have thematic connections in Neo4j."""
    try:
        works_rows = await pg.fetch_all(
            "SELECT work_id FROM works WHERE source_class = 'primary'"
        )
        if not works_rows:
            return CheckResult(
                name="theme_coverage", status="ok",
                label="Theme coverage", detail="No primary works ingested", count=0,
            )

        work_ids = [dict(r)["work_id"] for r in works_rows]
        neo4j_rows = await neo4j.execute_read(
            """
            MATCH (c:Chunk)-[:EXPLORES_THEME]->()
            WHERE c.work_id IN $work_ids
            RETURN DISTINCT c.work_id AS work_id
            """,
            {"work_ids": work_ids},
        )
        works_with_themes = {r["work_id"] for r in neo4j_rows}
        gaps = [w for w in work_ids if w not in works_with_themes]

        if not gaps:
            return CheckResult(
                name="theme_coverage", status="ok",
                label="Theme coverage",
                detail=f"All {len(work_ids)} primary work(s) have theme connections",
                count=0,
            )
        preview = ", ".join(gaps[:3]) + ("..." if len(gaps) > 3 else "")
        return CheckResult(
            name="theme_coverage", status="warn",
            label="Theme coverage",
            detail=f"{len(gaps)} primary work(s) missing theme connections: {preview}",
            count=len(gaps),
        )
    except Exception as exc:
        return CheckResult(
            name="theme_coverage", status="error",
            label="Theme coverage", detail=f"Query failed: {exc}", count=-1,
        )


async def run_all_checks(
    pg: "PostgresPool", neo4j: "Neo4jConnection"
) -> list[CheckResult]:
    """Run all health checks concurrently and return results."""
    results = await asyncio.gather(
        check_pg_neo4j_sync(pg, neo4j),
        check_missing_embeddings(pg),
        check_unvoiced_primary_sources(pg),
        check_low_confidence_classifications(pg),
        check_entity_extraction_gaps(pg, neo4j),
        check_orphaned_theme_nodes(neo4j),
        check_theme_coverage(pg, neo4j),
    )
    return list(results)
