# Dashboard Features: Voice Profiles, Work Detail, Themes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Parlour dashboard with three new tabs: Voice Profiles (per-author profile cards), Work Detail (click-to-expand row), and Themes (cross-work exploration with quotes).

**Architecture:** New async query functions in `queries.py` → new Starlette handlers in `endpoint.py` → registered in `server.py`. The frontend extends `template.html` with tab navigation and new render functions. All dynamic JS uses DOM methods (textContent/createElement/appendChild) exclusively — no innerHTML with server-sourced strings. Theme cross-work data uses `thematic_entries` + `thematic_appearances` from PG, supplemented by Neo4j chunk quotes once structural backfill is complete.

**Tech Stack:** Python 3.13, asyncpg, neo4j-driver, Starlette, vanilla JS, Chart.js 4 (CDN already loaded)

---

## Key Data Shapes

**Voice profile JSONB keys:** `register`, `confidence`, `humor_style`, `example_passages`, `rhetorical_moves`, `sentence_patterns`, `vocabulary_tendencies`, `characteristic_phrases`

**Thematic entry columns:** `id` (UUID), `author_id`, `theme`, `author_stance`, `related_themes` (text[]), `key_passages` (text)

**Thematic appearance columns:** `id`, `entry_id` (FK), `work_id`, `chapters` (text[]), `treatment_summary`

**Works extra columns:** `genre_tags` (text[]), `subject_headings` (text[]), `publication_year`, `publisher`, `format_ingested`, `word_count`, `source_metadata` (JSONB with `classification_confidence`, `classification_signals`)

**Neo4j (after backfill):** `MATCH (c:Chunk {work_id:$w})-[:EXPLORES_THEME]->(t:Theme) WHERE t.canonical_name=$n RETURN c.text, c.position`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/author_library/dashboard/queries.py` | 4 new query functions |
| Modify | `src/author_library/dashboard/endpoint.py` | 4 new route handlers |
| Modify | `src/author_library/server.py` | Register 4 new routes |
| Modify | `src/author_library/dashboard/template.html` | Tabs + 3 new panels + JS |

---

## Task 1: New queries — voice profiles, work detail, themes

**Files:**
- Modify: `src/author_library/dashboard/queries.py` (currently 126 lines — append below)

- [ ] **Step 1: Read the file**

```bash
cat src/author_library/dashboard/queries.py
```

- [ ] **Step 2: Append the four new functions**

At the end of `queries.py`, add:

```python
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
```

- [ ] **Step 3: Syntax check**

```bash
cd /home/marty/repos/parlour/parlour-car
uv run python -c "
from author_library.dashboard.queries import (
    get_voice_profiles, get_work_detail, get_all_themes, get_theme_detail
)
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/author_library/dashboard/queries.py
git commit -m "feat: dashboard queries for voice profiles, work detail, and themes"
```

---

## Task 2: New endpoint handlers

**Files:**
- Modify: `src/author_library/dashboard/endpoint.py`

- [ ] **Step 1: Read the file**

```bash
cat src/author_library/dashboard/endpoint.py
```

- [ ] **Step 2: Update the imports from queries**

Find:
```python
from author_library.dashboard.queries import (
    get_graph_stats,
    get_library_overview,
    get_per_work_details,
)
```

Replace with:
```python
from author_library.dashboard.queries import (
    get_all_themes,
    get_graph_stats,
    get_library_overview,
    get_per_work_details,
    get_theme_detail,
    get_voice_profiles,
    get_work_detail,
)
```

- [ ] **Step 3: Append the four new handlers after `handle_health`**

```python
async def handle_voice_profiles(request: Request) -> JSONResponse:
    """Return all current voice profiles."""
    storage = _storage(request)
    try:
        profiles = await get_voice_profiles(storage.pg)
        return JSONResponse({"profiles": profiles})
    except Exception as exc:
        log.error("dashboard_voice_profiles_error", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


async def handle_work_detail(request: Request) -> JSONResponse:
    """Return full detail for a single work."""
    storage = _storage(request)
    work_id = request.path_params["work_id"]
    try:
        detail = await get_work_detail(storage.pg, storage.neo4j, work_id)
        if detail is None:
            return JSONResponse({"error": "Not found"}, status_code=404)
        return JSONResponse(detail)
    except Exception as exc:
        log.error("dashboard_work_detail_error", work_id=work_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


async def handle_themes(request: Request) -> JSONResponse:
    """Return all themes with appearance counts."""
    storage = _storage(request)
    try:
        themes = await get_all_themes(storage.pg)
        return JSONResponse({"themes": themes})
    except Exception as exc:
        log.error("dashboard_themes_error", error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)


async def handle_theme_detail(request: Request) -> JSONResponse:
    """Return full theme detail including per-work appearances and quotes."""
    storage = _storage(request)
    entry_id = request.path_params["entry_id"]
    try:
        detail = await get_theme_detail(storage.pg, storage.neo4j, entry_id)
        if detail is None:
            return JSONResponse({"error": "Not found"}, status_code=404)
        return JSONResponse(detail)
    except Exception as exc:
        log.error("dashboard_theme_detail_error", entry_id=entry_id, error=str(exc))
        return JSONResponse({"error": str(exc)}, status_code=500)
```

- [ ] **Step 4: Syntax check**

```bash
uv run python -c "
from author_library.dashboard.endpoint import (
    handle_voice_profiles, handle_work_detail, handle_themes, handle_theme_detail
)
print('OK')
"
```

- [ ] **Step 5: Commit**

```bash
git add src/author_library/dashboard/endpoint.py
git commit -m "feat: dashboard handlers for voice profiles, work detail, and themes"
```

---

## Task 3: Register new routes in server.py

**Files:**
- Modify: `src/author_library/server.py`

- [ ] **Step 1: Expand the dashboard import block (around line 1292)**

Find:
```python
    from author_library.dashboard.endpoint import (
        handle_dashboard,
        handle_health,
        handle_stats,
    )
```

Replace with:
```python
    from author_library.dashboard.endpoint import (
        handle_dashboard,
        handle_health,
        handle_stats,
        handle_voice_profiles,
        handle_work_detail,
        handle_themes,
        handle_theme_detail,
    )
```

- [ ] **Step 2: Add 4 routes after `/dashboard/health`**

Find:
```python
            Route("/dashboard/health", endpoint=handle_health, methods=["GET"]),
```

Add after it:
```python
            Route("/dashboard/voice-profiles", endpoint=handle_voice_profiles, methods=["GET"]),
            Route("/dashboard/work/{work_id:str}", endpoint=handle_work_detail, methods=["GET"]),
            Route("/dashboard/themes", endpoint=handle_themes, methods=["GET"]),
            Route("/dashboard/themes/{entry_id:str}", endpoint=handle_theme_detail, methods=["GET"]),
```

- [ ] **Step 3: Verify**

```bash
grep -n "handle_voice\|handle_work\|handle_themes\|handle_theme" src/author_library/server.py
```

Expected: 8 lines (4 imports + 4 routes).

- [ ] **Step 4: Commit**

```bash
git add src/author_library/server.py
git commit -m "feat: register voice profile, work detail, and theme routes"
```

---

## Task 4: Template — CSS, tab bar, Voice Profiles tab

**Files:**
- Modify: `src/author_library/dashboard/template.html`

- [ ] **Step 1: Read the file**

```bash
cat src/author_library/dashboard/template.html
```

- [ ] **Step 2: Add CSS for tabs, voice profiles, work detail, and themes**

Before the closing `</style>` tag, insert:

```css
    /* Tabs */
    .tabs{display:flex;gap:.25rem;margin-bottom:1.5rem;border-bottom:1px solid #334155}
    .tab-btn{background:none;border:none;color:#64748b;font-size:.85rem;font-weight:600;
             padding:.5rem 1rem;cursor:pointer;border-bottom:2px solid transparent;
             margin-bottom:-1px;transition:color .15s,border-color .15s}
    .tab-btn:hover{color:#e2e8f0}
    .tab-btn.active{color:#38bdf8;border-bottom-color:#38bdf8}
    .tab-panel{display:none}
    .tab-panel.active{display:block}
    /* Voice profiles */
    .vp-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:1rem}
    .vp-card{background:#1e293b;border-radius:.5rem;padding:1rem}
    .vp-author{font-size:1rem;font-weight:700;color:#f8fafc}
    .vp-section-title{font-size:.72rem;font-weight:600;color:#64748b;text-transform:uppercase;
                      letter-spacing:.06em;margin-bottom:.3rem;margin-top:.75rem}
    .vp-text{font-size:.83rem;color:#cbd5e1;line-height:1.5}
    .vp-list{list-style:none;padding:0}
    .vp-list li{font-size:.83rem;color:#cbd5e1;padding:.15rem 0;border-bottom:1px solid #263044}
    .vp-list li:last-child{border-bottom:none}
    .expand-btn{background:none;border:1px solid #334155;color:#64748b;font-size:.75rem;
                padding:.25rem .6rem;border-radius:.25rem;cursor:pointer;margin-top:.6rem}
    .expand-btn:hover{color:#e2e8f0;border-color:#64748b}
    .vp-detail{display:none;margin-top:.75rem;border-top:1px solid #334155;padding-top:.75rem}
    .vp-detail.open{display:block}
    /* Work detail accordion */
    .work-row{cursor:pointer}
    .work-row:hover td{background:#263044}
    .work-detail-row td{padding:0}
    .work-detail-inner{padding:.75rem 1rem 1rem;background:#162032;border-bottom:1px solid #334155}
    .wd-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
    .wd-section{margin-bottom:.75rem}
    .wd-title{font-size:.72rem;font-weight:600;color:#64748b;text-transform:uppercase;
              letter-spacing:.06em;margin-bottom:.4rem}
    .tag-list{display:flex;flex-wrap:wrap;gap:.3rem}
    .tag{background:#1e293b;color:#94a3b8;font-size:.72rem;padding:.15rem .45rem;border-radius:.25rem}
    .theme-chip{background:#1e3a5f;color:#93c5fd;font-size:.72rem;padding:.15rem .45rem;
                border-radius:.25rem;cursor:pointer}
    .theme-chip:hover{background:#1e4a7f}
    .chunk-quote{font-size:.78rem;color:#94a3b8;font-style:italic;border-left:2px solid #334155;
                 padding-left:.6rem;margin-top:.4rem;line-height:1.5}
    /* Themes */
    .theme-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(255px,1fr));gap:.75rem;margin-bottom:1.5rem}
    .theme-card{background:#1e293b;border-radius:.5rem;padding:.85rem 1rem;cursor:pointer;
                transition:background .15s;border:1px solid transparent}
    .theme-card:hover{background:#263044;border-color:#334155}
    .theme-card.selected{border-color:#38bdf8;background:#162032}
    .theme-card-name{font-size:.88rem;font-weight:600;color:#e2e8f0;margin-bottom:.2rem}
    .theme-card-author{font-size:.73rem;color:#64748b;margin-bottom:.35rem}
    .theme-card-stance{font-size:.77rem;color:#94a3b8;line-height:1.4;
                       display:-webkit-box;-webkit-line-clamp:2;
                       -webkit-box-orient:vertical;overflow:hidden}
    .theme-panel{background:#1e293b;border-radius:.5rem;padding:1.25rem;margin-bottom:1rem}
    .theme-panel-title{font-size:1.1rem;font-weight:700;color:#f8fafc;margin-bottom:.3rem}
    .cw-block{background:#162032;border-radius:.4rem;padding:.75rem;margin-bottom:.75rem}
    .cw-title{font-size:.85rem;font-weight:600;color:#e2e8f0;margin-bottom:.3rem}
    .cw-summary{font-size:.8rem;color:#94a3b8;margin-bottom:.4rem;line-height:1.5}
    .quote-block{font-size:.78rem;color:#94a3b8;font-style:italic;border-left:2px solid #38bdf8;
                 padding-left:.6rem;margin-top:.4rem;line-height:1.5}
```

- [ ] **Step 3: Add tab bar after `<header>` block**

Find the closing `</header>` tag. Immediately after it, insert:

```html
<div class="tabs">
  <button class="tab-btn active" data-tab="library">Library</button>
  <button class="tab-btn" data-tab="voices">Voice Profiles</button>
  <button class="tab-btn" data-tab="themes">Themes</button>
</div>
```

- [ ] **Step 4: Wrap all current content in `<div id="tab-library" class="tab-panel active">`**

Find the first content div after the header area:
```html
<div class="cards" id="cards"></div>
```

Add `<div id="tab-library" class="tab-panel active">` immediately before it.

Find the last section div (knowledge graph section ends with `</div>`). After its closing `</div>`, add:
```html
</div><!-- /tab-library -->
```

- [ ] **Step 5: Add Voice Profiles and Themes tab panels**

After `</div><!-- /tab-library -->`, add:

```html
<div id="tab-voices" class="tab-panel">
  <div class="vp-grid" id="vp-grid"></div>
</div><!-- /tab-voices -->

<div id="tab-themes" class="tab-panel">
  <div class="theme-grid" id="theme-grid"></div>
  <div id="theme-panel"></div>
</div><!-- /tab-themes -->
```

- [ ] **Step 6: Add tab + voice-profile JS before `load();`**

In the `<script>` block, find the line `load();` and insert the following JS immediately before it:

```javascript
// ── Tab navigation ──────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const name = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    if (name === 'voices' && !document.getElementById('vp-grid').children.length) loadVoiceProfiles();
    if (name === 'themes' && !document.getElementById('theme-grid').children.length) loadThemes();
  });
});

// ── Voice Profiles ───────────────────────────────────────────────────────
async function loadVoiceProfiles() {
  const grid = document.getElementById('vp-grid');
  grid.textContent = 'Loading…';
  try {
    const data = await fetch('/dashboard/voice-profiles').then(r => r.json());
    grid.textContent = '';
    (data.profiles || []).forEach(p => grid.appendChild(buildVpCard(p)));
  } catch (e) {
    grid.textContent = 'Error: ' + e.message;
  }
}

function mkEl(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text != null) el.textContent = text;
  return el;
}

function buildVpCard(p) {
  const prof = p.profile || {};
  const card = mkEl('div', 'vp-card');

  // Header row
  const hdr = mkEl('div', null);
  hdr.style.cssText = 'display:flex;align-items:baseline;justify-content:space-between;margin-bottom:.5rem';
  hdr.appendChild(mkEl('span', 'vp-author', p.canonical_name));
  hdr.appendChild(mkEl('span', 'muted', 'v' + p.version + ' · ' + p.work_count + ' works'));
  card.appendChild(hdr);

  // Register
  if (prof.register) {
    card.appendChild(mkEl('div', 'vp-section-title', 'Register'));
    card.appendChild(mkEl('div', 'vp-text', prof.register));
  }

  // Confidence
  if (prof.confidence != null) {
    const pct = (prof.confidence * 100).toFixed(0);
    const col = prof.confidence >= 0.85 ? '#22c55e' : prof.confidence >= 0.70 ? '#f59e0b' : '#ef4444';
    const conf = mkEl('div', null, 'Confidence: ');
    conf.style.cssText = 'font-size:.75rem;color:#64748b;margin-top:.4rem';
    const strong = mkEl('strong', null, pct + '%');
    strong.style.color = col;
    conf.appendChild(strong);
    card.appendChild(conf);
  }

  // Expand button + detail pane
  const btn = mkEl('button', 'expand-btn', 'Show full profile');
  card.appendChild(btn);
  const detail = mkEl('div', 'vp-detail');
  detail.appendChild(buildVpDetail(prof));
  card.appendChild(detail);
  btn.addEventListener('click', () => {
    const open = detail.classList.toggle('open');
    btn.textContent = open ? 'Hide profile' : 'Show full profile';
  });

  return card;
}

function buildVpDetail(prof) {
  const frag = document.createDocumentFragment();

  [
    ['Characteristic Phrases', prof.characteristic_phrases],
    ['Rhetorical Moves',       prof.rhetorical_moves],
    ['Sentence Patterns',      prof.sentence_patterns],
  ].forEach(([title, val]) => {
    const items = Array.isArray(val) ? val : (val ? [val] : []);
    if (!items.length) return;
    frag.appendChild(mkEl('div', 'vp-section-title', title));
    const ul = mkEl('ul', 'vp-list');
    items.slice(0, 8).forEach(v => ul.appendChild(mkEl('li', null, String(v))));
    frag.appendChild(ul);
  });

  if (prof.humor_style) {
    frag.appendChild(mkEl('div', 'vp-section-title', 'Humor Style'));
    frag.appendChild(mkEl('div', 'vp-text', prof.humor_style));
  }

  const passages = Array.isArray(prof.example_passages) ? prof.example_passages : [];
  if (passages.length) {
    frag.appendChild(mkEl('div', 'vp-section-title', 'Example Passages'));
    passages.slice(0, 2).forEach(ep => {
      const q = mkEl('div', 'chunk-quote');
      q.textContent = typeof ep === 'string' ? ep : (ep.text || JSON.stringify(ep));
      frag.appendChild(q);
    });
  }

  return frag;
}
```

- [ ] **Step 7: Restart and verify**

```bash
pkill -f "python.*author_library"; sleep 1
cd /home/marty/repos/parlour/parlour-car && nohup bash start-mcp.sh > /tmp/parlour-mcp.log 2>&1 &
sleep 4
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/dashboard
echo ""
curl -s http://localhost:8080/dashboard/voice-profiles | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('profiles:', len(d.get('profiles', [])))
for p in d.get('profiles', []): print(' ', p['canonical_name'], 'v' + str(p['version']))
"
```

Expected: `200` and a list of authors.

- [ ] **Step 8: Commit**

```bash
git add src/author_library/dashboard/template.html
git commit -m "feat: tab navigation and Voice Profiles tab"
```

---

## Task 5: Template — work detail accordion

**Files:**
- Modify: `src/author_library/dashboard/template.html`

- [ ] **Step 1: Read the file (needed for exact edit targets)**

```bash
grep -n "const tr = document\|tbody.appendChild\|renderWorks\|function render" src/author_library/dashboard/template.html
```

- [ ] **Step 2: Add `work-row` class and click handler to work rows**

Inside `renderWorks`, find where the `tr` element is created:
```javascript
    const tr = document.createElement('tr');
```

Change it to:
```javascript
    const tr = document.createElement('tr');
    tr.className = 'work-row';
    tr.addEventListener('click', () => toggleWorkDetail(tr, w.work_id));
```

- [ ] **Step 3: Add work detail JS after `renderWorks`**

After the closing `}` of `renderWorks`, insert:

```javascript
// ── Work detail accordion ────────────────────────────────────────────────
const _wdCache = {};

async function toggleWorkDetail(tr, workId) {
  const existing = tr.parentNode.querySelector('.work-detail-row');
  if (existing) {
    const isSame = existing.previousElementSibling === tr;
    existing.remove();
    if (isSame) return;
  }

  const detailRow = document.createElement('tr');
  detailRow.className = 'work-detail-row';
  const cell = document.createElement('td');
  cell.colSpan = 7;
  detailRow.appendChild(cell);
  tr.insertAdjacentElement('afterend', detailRow);

  const inner = mkEl('div', 'work-detail-inner', 'Loading…');
  cell.appendChild(inner);

  if (!_wdCache[workId]) {
    try {
      _wdCache[workId] = await fetch('/dashboard/work/' + encodeURIComponent(workId)).then(r => r.json());
    } catch (e) {
      inner.textContent = 'Error: ' + e.message;
      return;
    }
  }

  inner.textContent = '';
  inner.appendChild(buildWorkDetail(_wdCache[workId]));
}

function buildWorkDetail(d) {
  const frag = document.createDocumentFragment();
  const grid = mkEl('div', 'wd-grid');

  // Left: metadata
  const left = mkEl('div');

  function addSection(title, items, chipClass) {
    if (!items?.length) return;
    const sec = mkEl('div', 'wd-section');
    sec.appendChild(mkEl('div', 'wd-title', title));
    const tl = mkEl('div', 'tag-list');
    items.forEach(t => tl.appendChild(mkEl('span', chipClass || 'tag', String(t))));
    sec.appendChild(tl);
    return sec;
  }

  const genreSec = addSection('Genre Tags', d.genre_tags);
  if (genreSec) left.appendChild(genreSec);

  const subjSec = addSection('Subject Headings', d.subject_headings);
  if (subjSec) left.appendChild(subjSec);

  const sm = d.source_metadata || {};
  const sigSec = addSection('Classification Signals', sm.classification_signals);
  if (sigSec) left.appendChild(sigSec);

  const pubParts = [d.publisher, d.publication_year, d.format_ingested].filter(Boolean);
  if (pubParts.length) {
    const sec = mkEl('div', 'wd-section');
    sec.appendChild(mkEl('div', 'wd-title', 'Publication'));
    sec.appendChild(mkEl('div', 'muted', pubParts.join(' · ')));
    left.appendChild(sec);
  }

  const breakdown = d.chunk_breakdown || {};
  if (Object.keys(breakdown).length) {
    const sec = mkEl('div', 'wd-section');
    sec.appendChild(mkEl('div', 'wd-title', 'Chunks'));
    const tl = mkEl('div', 'tag-list');
    Object.entries(breakdown).forEach(([g, n]) =>
      tl.appendChild(mkEl('span', 'tag', g + ': ' + Number(n).toLocaleString())));
    sec.appendChild(tl);
    left.appendChild(sec);
  }

  grid.appendChild(left);

  // Right: themes + sample chunk
  const right = mkEl('div');

  if (d.themes?.length) {
    const sec = mkEl('div', 'wd-section');
    sec.appendChild(mkEl('div', 'wd-title', 'Top Themes'));
    const tl = mkEl('div', 'tag-list');
    d.themes.forEach(t => {
      const chip = mkEl('span', 'theme-chip', t.name + ' (' + t.chunk_count + ')');
      chip.addEventListener('click', e => { e.stopPropagation(); document.querySelector('[data-tab="themes"]').click(); });
      tl.appendChild(chip);
    });
    sec.appendChild(tl);
    right.appendChild(sec);
  } else {
    const sec = mkEl('div', 'wd-section');
    sec.appendChild(mkEl('div', 'wd-title', 'Themes'));
    sec.appendChild(mkEl('div', 'muted', 'Neo4j sync pending'));
    right.appendChild(sec);
  }

  if (d.sample_chunks?.[0]) {
    const sec = mkEl('div', 'wd-section');
    sec.appendChild(mkEl('div', 'wd-title', 'Opening passage'));
    const q = mkEl('div', 'chunk-quote');
    const text = d.sample_chunks[0].text || '';
    q.textContent = text.slice(0, 350) + (text.length > 350 ? '…' : '');
    sec.appendChild(q);
    right.appendChild(sec);
  }

  grid.appendChild(right);
  frag.appendChild(grid);
  return frag;
}
```

- [ ] **Step 4: Verify work detail endpoint**

```bash
WORK_ID=$(curl -s http://localhost:8080/dashboard/stats | python3 -c "import json,sys; print(json.load(sys.stdin)['works'][0]['work_id'])")
curl -s "http://localhost:8080/dashboard/work/$WORK_ID" | python3 -c "
import json,sys; d=json.load(sys.stdin)
print('title:', d.get('title'))
print('chunks:', d.get('chunk_breakdown'))
print('sample_chunks:', len(d.get('sample_chunks',[])))
"
```

Expected: title, chunk breakdown, at least 1 sample chunk.

- [ ] **Step 5: Commit**

```bash
git add src/author_library/dashboard/template.html
git commit -m "feat: work detail accordion — click row to expand metadata, themes, sample chunk"
```

---

## Task 6: Template — Themes tab + cross-work tracing

**Files:**
- Modify: `src/author_library/dashboard/template.html`

- [ ] **Step 1: Add themes JS before `load();`**

Insert the following after `buildWorkDetail` and before `load();`:

```javascript
// ── Themes ───────────────────────────────────────────────────────────────
async function loadThemes() {
  const grid = document.getElementById('theme-grid');
  grid.textContent = 'Loading…';
  try {
    const data = await fetch('/dashboard/themes').then(r => r.json());
    grid.textContent = '';
    (data.themes || []).forEach(t => grid.appendChild(buildThemeCard(t)));
  } catch(e) {
    grid.textContent = 'Error: ' + e.message;
  }
}

function buildThemeCard(t) {
  const card = mkEl('div', 'theme-card');
  card.dataset.entryId = t.id;

  const nameLine = mkEl('div', 'theme-card-name');
  nameLine.appendChild(document.createTextNode(t.theme + ' '));
  const badge = mkEl('span', null,
    t.work_count > 1 ? t.work_count + ' works' : '1 work');
  badge.style.cssText = 'font-size:.7rem;color:' + (t.work_count > 1 ? '#38bdf8' : '#64748b');
  nameLine.appendChild(badge);
  card.appendChild(nameLine);

  card.appendChild(mkEl('div', 'theme-card-author', t.author_name || ''));
  card.appendChild(mkEl('div', 'theme-card-stance', t.author_stance || ''));

  card.addEventListener('click', () => openThemeDetail(t.id, card));
  return card;
}

async function openThemeDetail(entryId, cardEl) {
  document.querySelectorAll('.theme-card').forEach(c => c.classList.remove('selected'));
  cardEl.classList.add('selected');

  const panel = document.getElementById('theme-panel');
  panel.textContent = '';
  const loading = mkEl('div', null, 'Loading…');
  loading.style.cssText = 'color:#64748b;padding:1rem';
  panel.appendChild(loading);

  try {
    const detail = await fetch('/dashboard/themes/' + encodeURIComponent(entryId)).then(r => r.json());
    panel.textContent = '';
    panel.appendChild(buildThemeDetail(detail));
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch(e) {
    panel.textContent = 'Error: ' + e.message;
  }
}

function buildThemeDetail(d) {
  const wrap = mkEl('div', 'theme-panel');

  wrap.appendChild(mkEl('div', 'theme-panel-title', d.theme));

  const authorLine = mkEl('div', null, d.author_name || '');
  authorLine.style.cssText = 'font-size:.78rem;color:#64748b;margin-bottom:.5rem';
  wrap.appendChild(authorLine);

  if (d.author_stance) {
    const stance = mkEl('div', null, d.author_stance);
    stance.style.cssText = 'font-size:.85rem;color:#cbd5e1;line-height:1.6;margin-bottom:1rem';
    wrap.appendChild(stance);
  }

  if (d.key_passages) {
    const kpSec = mkEl('div', null);
    kpSec.style.marginBottom = '.75rem';
    kpSec.appendChild(mkEl('div', 'wd-title', 'Key Passages'));
    kpSec.appendChild(mkEl('div', 'quote-block', d.key_passages));
    wrap.appendChild(kpSec);
  }

  const related = Array.isArray(d.related_themes) ? d.related_themes : [];
  if (related.length) {
    const rtSec = mkEl('div', null);
    rtSec.style.marginBottom = '.75rem';
    rtSec.appendChild(mkEl('div', 'wd-title', 'Related Themes'));
    const tl = mkEl('div', 'tag-list');
    related.forEach(t => tl.appendChild(mkEl('span', 'theme-chip', String(t))));
    rtSec.appendChild(tl);
    wrap.appendChild(rtSec);
  }

  const appearances = Array.isArray(d.appearances) ? d.appearances : [];
  if (appearances.length) {
    const heading = mkEl('div', 'wd-title');
    heading.style.marginBottom = '.75rem';
    heading.textContent = 'This theme across ' + appearances.length +
      ' work' + (appearances.length !== 1 ? 's' : '');
    wrap.appendChild(heading);

    appearances.forEach(ap => {
      const block = mkEl('div', 'cw-block');
      block.appendChild(mkEl('div', 'cw-title', (ap.title || '') + ' — ' + (ap.author || '')));

      if (ap.treatment_summary) {
        block.appendChild(mkEl('div', 'cw-summary', ap.treatment_summary));
      }

      const chapters = Array.isArray(ap.chapters) ? ap.chapters : [];
      if (chapters.length) {
        const chap = mkEl('div', null, 'Chapters: ' + chapters.join(', '));
        chap.style.cssText = 'font-size:.72rem;color:#64748b;margin-bottom:.4rem';
        block.appendChild(chap);
      }

      const quotes = Array.isArray(ap.quotes) ? ap.quotes : [];
      if (quotes.length) {
        quotes.forEach(q => {
          block.appendChild(mkEl('div', 'quote-block',
            q + (q.length >= 400 ? '…' : '')));
        });
      } else {
        const noQ = mkEl('div', null, 'Quotes available after Neo4j sync completes');
        noQ.style.cssText = 'font-size:.75rem;color:#64748b;font-style:italic';
        block.appendChild(noQ);
      }

      wrap.appendChild(block);
    });
  } else {
    const noApp = mkEl('div', null, 'No cross-work appearances recorded yet.');
    noApp.style.cssText = 'font-size:.83rem;color:#64748b;margin-top:.5rem';
    wrap.appendChild(noApp);
  }

  return wrap;
}
```

- [ ] **Step 2: Test themes endpoints**

```bash
# Total themes
curl -s http://localhost:8080/dashboard/themes | python3 -c "
import json,sys; d=json.load(sys.stdin)
themes=d.get('themes',[])
print('themes:', len(themes))
multi=[t for t in themes if t['work_count']>1]
print('cross-work:', len(multi))
"

# Theme detail
ENTRY_ID=$(curl -s http://localhost:8080/dashboard/themes | python3 -c "
import json,sys; d=json.load(sys.stdin)
# Pick first cross-work theme
t=[t for t in d['themes'] if t['work_count']>1]
print(t[0]['id'] if t else d['themes'][0]['id'])
")
curl -s "http://localhost:8080/dashboard/themes/$ENTRY_ID" | python3 -c "
import json,sys; d=json.load(sys.stdin)
print('theme:', d.get('theme'))
print('appearances:', len(d.get('appearances',[])))
"
```

Expected: ~160 themes, some cross-work. Appearance data present.

- [ ] **Step 3: Restart server and do final check**

```bash
pkill -f "python.*author_library"; sleep 1
cd /home/marty/repos/parlour/parlour-car && nohup bash start-mcp.sh > /tmp/parlour-mcp.log 2>&1 &
sleep 4
for path in "/" "/dashboard" "/dashboard/voice-profiles" "/dashboard/themes"; do
  echo "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8080$path)  $path"
done
```

Expected: all `200`.

- [ ] **Step 4: Commit**

```bash
git add src/author_library/dashboard/template.html
git commit -m "feat: Themes tab with cross-work tracing panel and Neo4j quotes"
```

---

## Task 7: Check Neo4j sync + end-to-end

- [ ] **Step 1: Check if Neo4j backfill finished**

```bash
cat /tmp/neo4j-backfill.log | grep -E "DONE|error|work_id"
```

If still running: `pgrep -a -f backfill_work_graph`. Wait for completion before proceeding — theme quotes and work-detail themes won't populate until Neo4j has chunk nodes.

- [ ] **Step 2: Verify Neo4j has data**

```bash
cd /home/marty/repos/parlour/parlour-car
uv run python -c "
import asyncio
from author_library.config import get_settings
from author_library.storage.manager import StorageManager

async def check():
    mgr = StorageManager(get_settings().database)
    await mgr.connect(run_pg_migrations=False, init_neo4j_schema=False)
    rows = await mgr.neo4j.execute_read('MATCH (n) RETURN labels(n)[0] AS lbl, count(n) AS cnt ORDER BY cnt DESC LIMIT 5')
    for r in rows: print(r['lbl'], r['cnt'])
    await mgr.close()

asyncio.run(check())
" 2>/dev/null
```

Expected: Chunk node count matches PG chunks (~76,958).

- [ ] **Step 3: Open dashboard and smoke-test all tabs**

Visit `https://cc-claudesp.tail03afd8.ts.net:8080`

- Library tab: stat cards show 18 works, 76,958 chunks, 100% embedded
- Library tab: click a work row → detail expands with metadata, themes, opening passage
- Voice Profiles tab: cards for each author, expandable detail
- Themes tab: ~160 theme cards; click a cross-work theme → panel shows per-work treatment + quotes
- Knowledge Graph section: node/edge counts and top themes visible
