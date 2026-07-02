# Parlour Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only web dashboard at `http://localhost:8080/dashboard` that shows library stats, per-work details, knowledge graph metrics, and health checks for common ingestion problems.

**Architecture:** New `dashboard/` module inside parlour-car mounts three Starlette routes (`GET /dashboard`, `GET /dashboard/stats`, `GET /dashboard/health`) on the existing HTTP server. `queries.py` owns all PG/Neo4j reads; `health.py` owns check logic; `endpoint.py` wires routes. Template is a single HTML file served as static content with Chart.js from CDN.

**Tech Stack:** Python 3.13, asyncpg (via `storage.pg`), neo4j-driver (via `storage.neo4j`), Starlette, Chart.js 4 (CDN), vanilla JS

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/author_library/dashboard/__init__.py` | Empty package marker |
| Create | `src/author_library/dashboard/queries.py` | All PG + Neo4j stat queries |
| Create | `src/author_library/dashboard/health.py` | Health check logic, `CheckResult` dataclass |
| Create | `src/author_library/dashboard/endpoint.py` | Starlette route handlers |
| Create | `src/author_library/dashboard/template.html` | Single-page dashboard UI |
| Modify | `src/author_library/server.py` | Add routes + inject `dashboard_state` |
| Create | `tests/test_dashboard/__init__.py` | Empty package marker |
| Create | `tests/test_dashboard/conftest.py` | SKIP_NO_DB + re-export `storage` fixture |
| Create | `tests/test_dashboard/test_queries.py` | Shape tests for query functions |
| Create | `tests/test_dashboard/test_health.py` | Shape tests for health checks |

---

## Task 1: queries.py — Library and per-work stat queries

**Files:**
- Create: `src/author_library/dashboard/__init__.py`
- Create: `src/author_library/dashboard/queries.py`

- [ ] **Step 1: Create `src/author_library/dashboard/__init__.py`** (empty file)

- [ ] **Step 2: Write `queries.py`**

```python
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
```

- [ ] **Step 3: Commit**

```bash
cd /home/marty/repos/parlour/parlour-car
git add src/author_library/dashboard/__init__.py src/author_library/dashboard/queries.py
git commit -m "feat: dashboard queries module (library overview, per-work details, graph stats)"
```

---

## Task 2: health.py — Health check logic

**Files:**
- Create: `src/author_library/dashboard/health.py`

- [ ] **Step 1: Write `health.py`**

```python
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
                SELECT 1 FROM chunk_embeddings ce WHERE ce.chunk_id = c.chunk_id
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
```

- [ ] **Step 2: Commit**

```bash
git add src/author_library/dashboard/health.py
git commit -m "feat: dashboard health checks (7 checks: sync, embeddings, voice, confidence, entities, themes)"
```

---

## Task 3: template.html — Dashboard UI

**Files:**
- Create: `src/author_library/dashboard/template.html`

Note on XSS safety: all DB-sourced strings (titles, authors, theme names, check details) are
escaped with a `esc()` helper before being set via `innerHTML`. Number/percentage values are
formatted with `fmt()`/`pct()` which only produce numeric strings and are safe.

- [ ] **Step 1: Write `template.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Parlour Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #0f172a; color: #e2e8f0; min-height: 100vh; padding: 1.5rem; }
    h1  { font-size: 1.5rem; font-weight: 700; color: #f8fafc; }
    h2  { font-size: .8rem; font-weight: 600; color: #94a3b8; text-transform: uppercase;
          letter-spacing: .06em; margin-bottom: .75rem; }
    header { display: flex; align-items: center; justify-content: space-between;
             margin-bottom: 1.5rem; }
    #refresh-info { font-size: .8rem; color: #64748b; }
    .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
             gap: .75rem; margin-bottom: 1.5rem; }
    .card { background: #1e293b; border-radius: .5rem; padding: 1rem; }
    .card-value { font-size: 2rem; font-weight: 700; color: #38bdf8; }
    .card-label { font-size: .78rem; color: #94a3b8; margin-top: .25rem; }
    .charts { display: grid; grid-template-columns: 240px 1fr; gap: 1rem;
              margin-bottom: 1.5rem; }
    .chart-box { background: #1e293b; border-radius: .5rem; padding: 1rem; }
    h2.sec { margin-bottom: .5rem; }
    .section { background: #1e293b; border-radius: .5rem; padding: 1rem;
               margin-bottom: 1rem; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: .83rem; }
    th { text-align: left; padding: .45rem .75rem; color: #64748b;
         border-bottom: 1px solid #334155; white-space: nowrap; }
    td { padding: .45rem .75rem; border-bottom: 1px solid #263044; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #263044; }
    .badge { display: inline-block; padding: .12rem .5rem; border-radius: 999px;
             font-size: .72rem; font-weight: 600; }
    .badge-primary    { background: #0369a1; color: #bae6fd; }
    .badge-secondary  { background: #7c3aed; color: #ddd6fe; }
    .badge-contextual { background: #064e3b; color: #6ee7b7; }
    .badge-tertiary   { background: #374151; color: #9ca3af; }
    .badge-personal   { background: #92400e; color: #fde68a; }
    .checks { display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
              gap: .75rem; margin-bottom: 1.5rem; }
    .check { background: #1e293b; border-radius: .5rem; padding: .75rem 1rem;
             border-left: 4px solid; }
    .check-ok    { border-color: #22c55e; }
    .check-warn  { border-color: #f59e0b; }
    .check-error { border-color: #ef4444; }
    .check-name   { font-size: .83rem; font-weight: 600; margin-bottom: .2rem; }
    .check-detail { font-size: .78rem; color: #94a3b8; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
           margin-right: .4rem; vertical-align: middle; }
    .dot-ok    { background: #22c55e; }
    .dot-warn  { background: #f59e0b; }
    .dot-error { background: #ef4444; }
    .pbar-bg { background: #334155; border-radius: 3px; height: 5px;
               width: 72px; display: inline-block; vertical-align: middle; }
    .pbar    { background: #38bdf8; border-radius: 3px; height: 5px; }
    .conf-warn { color: #f59e0b; }
    .muted { color: #64748b; }
  </style>
</head>
<body>
<header>
  <h1>Parlour Library</h1>
  <span id="refresh-info">Loading...</span>
</header>

<div class="cards" id="cards"></div>

<div class="charts">
  <div class="chart-box">
    <h2>Source classes</h2>
    <canvas id="classChart" width="200" height="200"></canvas>
  </div>
  <div class="chart-box">
    <h2>Chunks per work</h2>
    <canvas id="chunksChart"></canvas>
  </div>
</div>

<h2 class="sec">Health checks</h2>
<div class="checks" id="checks"></div>

<div class="section">
  <h2>Works</h2>
  <table>
    <thead>
      <tr>
        <th>Title</th><th>Author</th><th>Class</th>
        <th>Chunks</th><th>Embedded</th><th>Confidence</th><th>Ingested</th>
      </tr>
    </thead>
    <tbody id="works-tbody"></tbody>
  </table>
</div>

<div class="section">
  <h2>Knowledge graph</h2>
  <div id="graph-stats"></div>
</div>

<script>
// Escape HTML special chars in any string from the server before
// inserting it via innerHTML.
function esc(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function fmt(n) { return n == null ? '&mdash;' : Number(n).toLocaleString(); }
function pct(v) { return v == null ? '&mdash;' : Number(v).toFixed(1) + '%'; }

let classChart = null, chunksChart = null;

async function load() {
  try {
    const [stats, health] = await Promise.all([
      fetch('/dashboard/stats').then(r => r.json()),
      fetch('/dashboard/health').then(r => r.json()),
    ]);
    renderCards(stats.library || {});
    renderCharts(stats.library || {}, stats.works || []);
    renderWorks(stats.works || []);
    renderGraph(stats.graph || {});
    renderHealth(health.checks || []);
    document.getElementById('refresh-info').textContent =
      'Updated ' + new Date().toLocaleTimeString() + ' · auto-refresh 60s';
  } catch (err) {
    document.getElementById('refresh-info').textContent = 'Load error: ' + err.message;
  }
}

function renderCards(lib) {
  const items = [
    { value: fmt(lib.total_works),            label: 'Works' },
    { value: fmt(lib.total_chunks),           label: 'Chunks' },
    { value: pct(lib.embedding_coverage_pct), label: 'Embedded' },
    { value: fmt(lib.voice_profile_count),    label: 'Voice profiles' },
    { value: fmt(lib.unique_authors),         label: 'Authors' },
    { value: esc(lib.last_ingestion_date) || '&mdash;', label: 'Last ingested' },
  ];
  const el = document.getElementById('cards');
  el.textContent = '';
  items.forEach(({ value, label }) => {
    const card = document.createElement('div');
    card.className = 'card';
    // value may contain safe HTML (fmt/pct produce numbers; esc sanitises strings)
    card.innerHTML =
      '<div class="card-value">' + value + '</div>' +
      '<div class="card-label">' + esc(label) + '</div>';
    el.appendChild(card);
  });
}

function renderCharts(lib, works) {
  const cls = lib.by_source_class || {};
  const clsLabels = Object.keys(cls);
  const clsData   = Object.values(cls);
  const clsColors = ['#0369a1','#7c3aed','#064e3b','#374151','#92400e'];

  if (classChart) classChart.destroy();
  classChart = new Chart(document.getElementById('classChart'), {
    type: 'doughnut',
    data: { labels: clsLabels,
            datasets: [{ data: clsData, backgroundColor: clsColors, borderWidth: 0 }] },
    options: {
      plugins: { legend: {
        position: 'bottom',
        labels: { color: '#94a3b8', font: { size: 11 }, boxWidth: 12 },
      }},
    },
  });

  const sorted  = [...works].sort((a, b) => b.chunk_count - a.chunk_count);
  const wLabels = sorted.map(w => {
    const t = String(w.title || '');
    return t.length > 32 ? t.slice(0, 30) + '…' : t;
  });
  const wData = sorted.map(w => w.chunk_count || 0);

  if (chunksChart) chunksChart.destroy();
  chunksChart = new Chart(document.getElementById('chunksChart'), {
    type: 'bar',
    data: {
      labels: wLabels,
      datasets: [{ label: 'Chunks', data: wData,
                   backgroundColor: '#38bdf8', borderRadius: 3 }],
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#64748b' }, grid: { color: '#263044' } },
        y: { ticks: { color: '#94a3b8', font: { size: 11 } } },
      },
    },
  });
}

function renderWorks(works) {
  const tbody = document.getElementById('works-tbody');
  tbody.textContent = '';
  works.forEach(w => {
    const conf = w.classification_confidence;
    const confStr  = conf != null ? (conf * 100).toFixed(0) + '%' : '&mdash;';
    const confCls  = conf != null && conf < 0.90 ? ' conf-warn' : '';
    const embPct   = w.embedding_pct ?? 0;
    const cls      = esc(w.source_class || '');
    const tr = document.createElement('tr');
    tr.innerHTML =
      '<td>' + esc(w.title)  + '</td>' +
      '<td class="muted">' + esc(w.author) + '</td>' +
      '<td><span class="badge badge-' + cls + '">' + cls + '</span></td>' +
      '<td>' + fmt(w.chunk_count) + '</td>' +
      '<td>' +
        '<span class="pbar-bg"><span class="pbar" style="width:' + embPct + '%"></span></span> ' +
        pct(embPct) +
      '</td>' +
      '<td class="' + confCls + '">' + confStr + '</td>' +
      '<td class="muted">' + esc(w.ingestion_date) + '</td>';
    tbody.appendChild(tr);
  });
}

function renderGraph(g) {
  const el = document.getElementById('graph-stats');
  el.textContent = '';
  if (g.error) {
    el.textContent = 'Neo4j unavailable: ' + g.error;
    el.style.color = '#94a3b8';
    return;
  }
  const nc = g.node_counts || {};
  const ec = g.edge_counts || {};
  const nodeStr  = Object.entries(nc).map(([k, v]) => esc(k) + ': ' + fmt(v)).join(' &middot; ');
  const edgeStr  = Object.entries(ec).map(([k, v]) => esc(k) + ': ' + fmt(v)).join(' &middot; ');
  const themes   = (g.top_themes || []).slice(0, 8)
    .map(t => esc(t.name) + ' (' + t.work_count + 'w)').join(', ');
  el.innerHTML =
    '<p style="font-size:.83rem"><strong style="color:#e2e8f0">Nodes:</strong> ' + (nodeStr || '&mdash;') + '</p>' +
    '<p style="font-size:.83rem;margin-top:.4rem"><strong style="color:#e2e8f0">Edges:</strong> ' + (edgeStr || '&mdash;') + '</p>' +
    '<p style="font-size:.83rem;margin-top:.4rem"><strong style="color:#e2e8f0">Top themes:</strong> ' + (themes || '&mdash;') + '</p>';
}

function renderHealth(checks) {
  const el = document.getElementById('checks');
  el.textContent = '';
  checks.forEach(c => {
    const div = document.createElement('div');
    div.className = 'check check-' + c.status;
    div.innerHTML =
      '<div class="check-name"><span class="dot dot-' + c.status + '"></span>' +
        esc(c.label) + '</div>' +
      '<div class="check-detail">' + esc(c.detail) + '</div>';
    el.appendChild(div);
  });
}

load();
setInterval(load, 60_000);
</script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add src/author_library/dashboard/template.html
git commit -m "feat: dashboard HTML template (stats cards, charts, works table, health checks)"
```

---

## Task 4: endpoint.py — Starlette route handlers

**Files:**
- Create: `src/author_library/dashboard/endpoint.py`

- [ ] **Step 1: Write `endpoint.py`**

```python
"""Dashboard HTTP endpoints.

GET /dashboard         — serve the single-page HTML template
GET /dashboard/stats   — JSON: library overview + per-work details + graph stats
GET /dashboard/health  — JSON: health check results
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

from author_library.dashboard.health import run_all_checks
from author_library.dashboard.queries import (
    get_graph_stats,
    get_library_overview,
    get_per_work_details,
)

if TYPE_CHECKING:
    from author_library.storage.manager import StorageManager

log = structlog.get_logger(__name__)

_TEMPLATE = Path(__file__).parent / "template.html"


def _storage(request: Request) -> "StorageManager":
    return request.app.state.dashboard_state["storage"]


async def handle_dashboard(request: Request) -> FileResponse:
    """Serve the dashboard HTML."""
    return FileResponse(_TEMPLATE, media_type="text/html")


async def handle_stats(request: Request) -> JSONResponse:
    """Return all library stats as JSON."""
    storage = _storage(request)
    try:
        library, works, graph = await asyncio.gather(
            get_library_overview(storage.pg),
            get_per_work_details(storage.pg),
            get_graph_stats(storage.neo4j),
        )
        library["by_source_class"] = {
            "primary":    library.get("primary_works", 0),
            "secondary":  library.get("secondary_works", 0),
            "contextual": library.get("contextual_works", 0),
            "tertiary":   library.get("tertiary_works", 0),
            "personal":   library.get("personal_works", 0),
        }
        return JSONResponse({"library": library, "works": works, "graph": graph})
    except Exception as exc:
        log.error("dashboard_stats_error", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


async def handle_health(request: Request) -> JSONResponse:
    """Return health check results as JSON."""
    storage = _storage(request)
    try:
        checks = await run_all_checks(storage.pg, storage.neo4j)
        statuses = [c.status for c in checks]
        overall = (
            "error" if "error" in statuses
            else "warn" if "warn" in statuses
            else "ok"
        )
        return JSONResponse({
            "checks": [dataclasses.asdict(c) for c in checks],
            "overall": overall,
        })
    except Exception as exc:
        log.error("dashboard_health_error", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)
```

- [ ] **Step 2: Commit**

```bash
git add src/author_library/dashboard/endpoint.py
git commit -m "feat: dashboard endpoint handlers (/dashboard, /dashboard/stats, /dashboard/health)"
```

---

## Task 5: Mount routes in server.py

**Files:**
- Modify: `src/author_library/server.py`

- [ ] **Step 1: Add imports inside `_run_http`**

Inside `_run_http`, after the existing local import block (which ends with `from mcp.server.sse import SseServerTransport`), add:

```python
from author_library.dashboard.endpoint import (
    handle_dashboard,
    handle_health,
    handle_stats,
)
```

- [ ] **Step 2: Add three routes to the Starlette routes list**

In the `Starlette(routes=[...])` call, append after the last existing `Route(...)`:

```python
Route("/dashboard", endpoint=handle_dashboard, methods=["GET"]),
Route("/dashboard/stats", endpoint=handle_stats, methods=["GET"]),
Route("/dashboard/health", endpoint=handle_health, methods=["GET"]),
```

- [ ] **Step 3: Inject `dashboard_state` after app creation**

After the line `app.state.surfacing_state = { ... }`, add:

```python
app.state.dashboard_state = {
    "storage": server._tool_state.get("storage"),  # type: ignore[attr-defined]
}
```

- [ ] **Step 4: Verify the diff**

```bash
git diff src/author_library/server.py
```

Expected: 3 new import lines, 3 new Route entries, 1 new state assignment (4 lines).

- [ ] **Step 5: Commit**

```bash
git add src/author_library/server.py
git commit -m "feat: mount dashboard routes on Starlette app, inject dashboard_state"
```

---

## Task 6: Tests for queries.py

**Files:**
- Create: `tests/test_dashboard/__init__.py`
- Create: `tests/test_dashboard/conftest.py`
- Create: `tests/test_dashboard/test_queries.py`

- [ ] **Step 1: Create `tests/test_dashboard/__init__.py`** (empty file)

- [ ] **Step 2: Write `tests/test_dashboard/conftest.py`**

```python
"""Re-export test fixtures from test_integration for dashboard tests."""
from tests.test_integration.conftest import (  # noqa: F401
    SKIP_NO_DB,
    clean_storage,
    integration_settings,
    storage,
)
```

- [ ] **Step 3: Write `tests/test_dashboard/test_queries.py`**

```python
"""Shape tests for dashboard queries.

These run against author_library_test (the live test DB).
They verify the return structure, not specific counts,
since the test DB may be empty.
"""

from author_library.dashboard.queries import (
    get_graph_stats,
    get_library_overview,
    get_per_work_details,
)
from tests.test_dashboard.conftest import SKIP_NO_DB


@SKIP_NO_DB
class TestGetLibraryOverview:
    async def test_returns_required_keys(self, storage):
        result = await get_library_overview(storage.pg)
        assert isinstance(result, dict)
        for key in (
            "total_works", "primary_works", "secondary_works",
            "contextual_works", "tertiary_works", "personal_works",
            "total_chunks", "embedding_coverage_pct", "voice_profile_count",
        ):
            assert key in result, f"Missing key: {key}"

    async def test_counts_non_negative(self, storage):
        result = await get_library_overview(storage.pg)
        assert result["total_works"] >= 0
        assert result["total_chunks"] >= 0
        assert 0.0 <= result["embedding_coverage_pct"] <= 100.0

    async def test_coverage_does_not_crash_on_empty_db(self, clean_storage):
        result = await get_library_overview(clean_storage.pg)
        assert isinstance(result["embedding_coverage_pct"], float)


@SKIP_NO_DB
class TestGetPerWorkDetails:
    async def test_returns_list(self, storage):
        result = await get_per_work_details(storage.pg)
        assert isinstance(result, list)

    async def test_each_row_has_required_fields(self, storage):
        for row in await get_per_work_details(storage.pg):
            for key in ("work_id", "title", "author", "source_class",
                        "chunk_count", "embedded_count", "embedding_pct"):
                assert key in row, f"Missing key: {key}"

    async def test_embedding_pct_in_range(self, storage):
        for row in await get_per_work_details(storage.pg):
            assert isinstance(row["embedding_pct"], float)
            assert 0.0 <= row["embedding_pct"] <= 100.0


@SKIP_NO_DB
class TestGetGraphStats:
    async def test_returns_dict(self, storage):
        result = await get_graph_stats(storage.neo4j)
        assert isinstance(result, dict)

    async def test_has_expected_keys_when_neo4j_up(self, storage):
        result = await get_graph_stats(storage.neo4j)
        if result.get("error") is None:
            for key in ("node_counts", "edge_counts", "total_nodes",
                        "total_edges", "top_themes"):
                assert key in result

    async def test_top_themes_is_list(self, storage):
        result = await get_graph_stats(storage.neo4j)
        if result.get("error") is None:
            assert isinstance(result["top_themes"], list)
```

- [ ] **Step 4: Run tests**

```bash
cd /home/marty/repos/parlour/parlour-car
uv run python -m pytest tests/test_dashboard/test_queries.py -v
```

Expected: all tests pass (or SKIP if DB is not running).

- [ ] **Step 5: Commit**

```bash
git add tests/test_dashboard/
git commit -m "test: dashboard query shape tests"
```

---

## Task 7: Tests for health.py

**Files:**
- Create: `tests/test_dashboard/test_health.py`

- [ ] **Step 1: Write `tests/test_dashboard/test_health.py`**

```python
"""Contract tests for dashboard health checks."""

from author_library.dashboard.health import (
    CheckResult,
    check_entity_extraction_gaps,
    check_low_confidence_classifications,
    check_missing_embeddings,
    check_orphaned_theme_nodes,
    check_pg_neo4j_sync,
    check_theme_coverage,
    check_unvoiced_primary_sources,
    run_all_checks,
)
from tests.test_dashboard.conftest import SKIP_NO_DB

VALID_STATUSES = {"ok", "warn", "error"}
ALL_CHECK_NAMES = {
    "pg_neo4j_sync",
    "missing_embeddings",
    "unvoiced_primary_sources",
    "low_confidence_classifications",
    "entity_extraction_gaps",
    "orphaned_theme_nodes",
    "theme_coverage",
}


@SKIP_NO_DB
class TestRunAllChecks:
    async def test_returns_seven_results(self, storage):
        results = await run_all_checks(storage.pg, storage.neo4j)
        assert len(results) == 7

    async def test_all_are_check_results(self, storage):
        for r in await run_all_checks(storage.pg, storage.neo4j):
            assert isinstance(r, CheckResult)

    async def test_all_statuses_valid(self, storage):
        for r in await run_all_checks(storage.pg, storage.neo4j):
            assert r.status in VALID_STATUSES, f"{r.name}: invalid status {r.status!r}"

    async def test_all_names_present(self, storage):
        names = {r.name for r in await run_all_checks(storage.pg, storage.neo4j)}
        assert names == ALL_CHECK_NAMES

    async def test_empty_db_no_crashes(self, clean_storage):
        results = await run_all_checks(clean_storage.pg, clean_storage.neo4j)
        for r in results:
            # count == -1 means the check itself errored; that should not happen on empty DB
            assert r.count != -1, f"{r.name} raised an exception on empty DB: {r.detail}"


@SKIP_NO_DB
class TestIndividualChecks:
    async def test_missing_embeddings_ok_on_empty_db(self, clean_storage):
        r = await check_missing_embeddings(clean_storage.pg)
        assert r.status == "ok"
        assert r.count == 0

    async def test_pg_neo4j_sync_ok_on_empty_db(self, clean_storage):
        r = await check_pg_neo4j_sync(clean_storage.pg, clean_storage.neo4j)
        assert r.status == "ok"

    async def test_low_confidence_ok_on_empty_db(self, clean_storage):
        r = await check_low_confidence_classifications(clean_storage.pg)
        assert r.status == "ok"
        assert r.count == 0

    async def test_orphaned_themes_ok_on_empty_neo4j(self, clean_storage):
        r = await check_orphaned_theme_nodes(clean_storage.neo4j)
        assert r.status == "ok"

    async def test_theme_coverage_ok_when_no_primary_works(self, clean_storage):
        r = await check_theme_coverage(clean_storage.pg, clean_storage.neo4j)
        assert r.status == "ok"
        assert "No primary works" in r.detail

    async def test_entity_gaps_ok_when_no_primary_works(self, clean_storage):
        r = await check_entity_extraction_gaps(clean_storage.pg, clean_storage.neo4j)
        assert r.status == "ok"

    async def test_unvoiced_ok_when_no_primary_works(self, clean_storage):
        r = await check_unvoiced_primary_sources(clean_storage.pg)
        assert r.status == "ok"
```

- [ ] **Step 2: Run full test suite**

```bash
uv run python -m pytest tests/test_dashboard/ -v
```

Expected: all tests pass (or SKIP if DB not running).

- [ ] **Step 3: Commit**

```bash
git add tests/test_dashboard/test_health.py
git commit -m "test: dashboard health check contract tests"
```

---

## Task 8: Verify end-to-end

- [ ] **Step 1: Restart the MCP server**

```bash
pkill -9 -f "python.*author_library" 2>/dev/null || true
sleep 1
setsid /home/marty/repos/parlour/parlour-car/start-mcp.sh > /tmp/parlour-mcp.log 2>&1 &
sleep 3
```

- [ ] **Step 2: Check server started cleanly**

```bash
curl -s http://localhost:8080/api/v1/health
```

Expected: `{"status":"ok","server":"parlour-car"}`

If this fails, check logs: `tail -20 /tmp/parlour-mcp.log`

- [ ] **Step 3: Verify stats endpoint returns correct shape**

```bash
curl -s http://localhost:8080/dashboard/stats | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('library keys:', sorted(d.get('library', {}).keys()))
print('works count:', len(d.get('works', [])))
print('graph error:', d.get('graph', {}).get('error'))
"
```

Expected output (approximately):
```
library keys: ['by_source_class', 'embedding_coverage_pct', 'last_ingestion_date', 'personal_works', ...]
works count: 8
graph error: None
```

- [ ] **Step 4: Verify health endpoint**

```bash
curl -s http://localhost:8080/dashboard/health | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('overall:', d['overall'])
for c in d['checks']:
    print(f\"  {c['status']:5}  {c['label']}: {c['detail']}\")
"
```

Expected: 7 check lines, all `ok` on a healthy system.

- [ ] **Step 5: Open dashboard in browser**

```bash
xdg-open http://localhost:8080/dashboard
```

Expected: dark-themed dashboard renders with stat cards, doughnut + bar charts, health check grid, and works table.
