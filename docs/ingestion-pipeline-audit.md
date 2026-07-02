# Ingestion Pipeline Audit — "Choose Your Own Adventure"

**Auditor**: Claude (senior LLM/graph DB developer perspective)
**Date**: 2026-03-14
**Scope**: Full code trace of `ingest_book` → `IngestionPipeline.ingest()` → return
**Method**: 3-pass audit, consolidated into final report. Each step traced from actual code, not assumptions.

---

## Entry Point: `handle_ingest_book()` — `ingest.py:30`

```
User calls MCP tool "ingest_book" with:
  - file_path (required)
  - subject_author_id (required)
  - metadata_hints (optional)
  - auto_confirm (optional, default True)
```

### Gate 0: Input Validation (ingest.py:53-75)

- **Missing file_path** → raises `IngestionError` ❌ DEAD END
- **Missing subject_author_id** → raises `IngestionError` ❌ DEAD END
- **File doesn't exist** → raises `IngestionError` ❌ DEAD END
- All present → ✅ CONTINUE

### Branch: auto_confirm=False (ingest.py:78-85)

If `auto_confirm=False`:
- Calls `_classify_only()` → runs just classification + mixed authorship check
- Returns JSON preview for human review
- ⏹️ PIPELINE STOPS HERE — user must use composable tools to continue
- **⚠️ NOTE**: This path does NOT run pre-ingestion backup

### Branch: auto_confirm=True (default) (ingest.py:87-152)

```
Pre-ingestion backup → Pipeline.ingest() → Cross-work analysis →
Cache invalidation → Quality gate enqueue → Post-ingestion backup →
Ingestion report (auto-heal)
```

---

## Pre-Ingestion Backup (ingest.py:88)

- Calls `_run_pre_ingest_backup()` (ingest.py:512-542)
- Checks if `/home/marty/parlour-backups/backup.sh` exists
  - **Script missing** → `log.debug`, skip silently ✅ CONTINUE
  - **Script exists** → runs as subprocess with 120s timeout
    - **Success** → `log.info` ✅ CONTINUE
    - **Non-zero exit** → `log.warning` ✅ CONTINUE (non-blocking)
    - **Timeout** → `log.warning` ✅ CONTINUE (non-blocking)
    - **Any exception** → `log.warning` ✅ CONTINUE (non-blocking)
- **⚠️ NOTE**: Backup failure never blocks ingestion. This is intentional (fire-and-forget).

---

## Pipeline Construction (ingest.py:90-94)

```python
pipeline = IngestionPipeline(settings, storage, embedding_provider)
```

No validation here. Just stores references. If any dependency is None or misconfigured, it will fail later at usage point.

---

## Step 0: Booklore Metadata Resolution (ingestion_pipeline.py:168-179)

- **Only runs if `settings.booklore.enabled`**
  - Calls `resolve_metadata(path, db_url=...)` from `catalog/booklore.py`
  - Returns metadata dict (title, author, year, ISBN, publisher)
  - Booklore values are DEFAULTS — explicit `metadata_hints` from user take precedence
  - **Connection failure** → `log.warning` gracefully, returns empty dict ✅ CONTINUE
- **Booklore disabled** → skip entirely ✅ CONTINUE

---

## Step 1: Parse (ingestion_pipeline.py:181-199)

```python
parser = get_parser(path)        # Dispatches by file extension
document = await parser.parse(str(path))
```

**`get_parser(path)`** — `chunking/__init__.py`:
- Looks up parser from `_EXTENSION_MAP` by file extension
- Registered: `.epub`, `.pdf`, `.docx`, `.txt`, `.html`
- **Unknown extension** → raises `ParsingError` ❌ DEAD END (uncaught — propagates to caller)

**`parser.parse()`**:
- Each parser (EpubParser, PdfParser, etc.) returns a `ParsedDocument`
- Contains: `metadata` (title, word_count, etc.), `format`, document tree
- **Parse failure** → raises `ParsingError` ❌ DEAD END (uncaught)

**⚠️ OBSERVATION**: Parse errors are NOT caught in the pipeline. They propagate up through `ingest()` → `handle_ingest_book()` → MCP error response. This is correct behavior — if we can't parse, we can't ingest.

After parse succeeds, calls `_ingest_from_document()` for Steps 2-13.

---

## Step 1b: Alternative Entry — `ingest_document()` (ingestion_pipeline.py:201-238)

For pre-built ParsedDocument objects (e.g. epistolary pipeline):
- Skips Step 0 (no Booklore) and Step 1 (no file parsing)
- Goes directly to `_ingest_from_document()` with the provided document
- Same Steps 2-13 from here on

---

## Step 2: Classify (ingestion_pipeline.py:256-283)

```python
classification_pipeline = ClassificationPipeline(settings, work_repository, subject_author, pg_pool, storage)
pipeline_result = await classification_pipeline.process(document, metadata_hints, user_overrides)
```

**ClassificationPipeline.process()** does 4 things:
1. **SourceClassifier.classify()** — LLM call to determine source class (PRIMARY/SECONDARY/CONTEXTUAL/TERTIARY/PERSONAL)
   - If confidence < 0.7, auto-downgrades to SECONDARY (voice contamination safety)
   - **LLM failure** → raises exception ❌ DEAD END (uncaught in pipeline)
2. **Resolve missing author** — for PRIMARY, looks up canonical name from `authors` table
3. **Build CatalogEntry** — appropriate subclass based on source class
4. **Store in works table** — `WorkRepository.create(work_data)` with source_metadata JSONB
   - **DB failure** → raises exception ❌ DEAD END (uncaught)

**Determines ProcessingRoute:**
- PRIMARY → `FULL_ENRICHMENT`
- SECONDARY → `EMBEDDINGS_AND_GRAPH`
- CONTEXTUAL → `EMBEDDINGS_AND_LINKS`
- TERTIARY → `METADATA_ONLY`
- PERSONAL → `PERSONAL_ENRICHMENT`

**⚠️ OBSERVATION**: Classification failures are uncaught. If the LLM is down or returns garbage, the entire ingestion fails. This is probably correct — classification is the gate.

---

## Step 2b: Re-ingestion Detection + Cleanup (ingestion_pipeline.py:286-308)

```python
current_max_pass = await storage.chunks.get_max_pass_number(work_id)
pass_number = current_max_pass + 1 if current_max_pass > 0 else 1
deleted_chunks = await storage.chunks.delete_by_work(work_id)       # PG
deleted_graph_chunks = await storage.graph.delete_chunks_by_work(work_id)  # Neo4j
```

- **First ingestion** → pass_number=1, no deletes
- **Re-ingestion** → increments pass number, deletes ALL old chunks from PG + Neo4j
- **Neo4j cleanup failure** → `log.warning`, continues ✅ CONTINUE
  - **⚠️ ISSUE NOTED (prior session)**: Before our fix, Neo4j cleanup didn't exist. Now it does, but failure is non-fatal. This means a Neo4j outage during re-ingestion leaves orphaned graph nodes.

---

## Step 2c: Upsert Work + Author (ingestion_pipeline.py:310-346)

1. **Upsert Work node in Neo4j** — `graph.upsert_work_node()`
   - **Failure** → uncaught ❌ DEAD END
   - **⚠️ OBSERVATION**: This is one of the few uncaught Neo4j calls. If Neo4j is down here, the whole pipeline dies. But Step 8 (chunk upsert) has error handling. Inconsistent.

2. **Upsert Author in PG** — `INSERT INTO authors ... ON CONFLICT DO NOTHING`
   - **Failure** → uncaught ❌ DEAD END (PG down = nothing works anyway)

3. **Upsert Author node + AUTHORED edge in Neo4j** — direct Cypher via `neo4j.execute_write()`
   - **Failure** → uncaught ❌ DEAD END
   - **⚠️ OBSERVATION**: Same inconsistency. This direct neo4j call has no try/except, but later chunk upserts do.

---

## Step 3: Route by Source Class (ingestion_pipeline.py:348-360)

```
METADATA_ONLY (tertiary) → return IngestionResult immediately ⏹️ EARLY RETURN
All other routes → CONTINUE to Step 4
```

- Tertiary sources get NO content processing — just the catalog entry in PG + work/author nodes in Neo4j
- Returns with 0 chunks, 0 embeddings, 0 entities

---

## Step 4: Chunk (ingestion_pipeline.py:362-379)

```python
strategy = get_chunking_strategy(genre_tags)
chunks = strategy.chunk(document, work_id, source_class.value)
```

**Strategy selection** (`chunking/__init__.py`):
- 7 strategies checked in priority order: Poetry → Transcript → Interview → Letter → Blog → Sermon → ScholarlyProse (fallback)
- First strategy whose `supported_genres()` intersects with work's genre_tags wins
- **Empty genre_tags** → raises `IngestionError` ❌ DEAD END
- **No match** → ScholarlyProse fallback ✅ CONTINUE

**Chunking itself**:
- Each strategy produces chunks at multiple granularities (macro/meso/micro/nano)
- Chunks have: id, text, granularity, work_id, source_class, chapter, section, position, metadata
- **⚠️ OBSERVATION**: Chunking errors are uncaught. A malformed document tree could cause unexpected exceptions here.

---

## Step 4b: Section-Type Routing (ingestion_pipeline.py:381-396)

```python
chunks, skipped_sections, structural_chunks = self._filter_by_section_type(chunks, work_id)
```

**Filter logic** (`_filter_by_section_type`, line 1288):
- **KEPT** (full pipeline): `chapter`, `preface`, `back_matter`
- **REMOVED** (routed elsewhere): `index` → VocabularyManager, `bibliography` → AcquisitionManager
- **REMOVED** (discarded): `toc`, `front_matter`

**Structural routing** (`_route_structural_sections`, line 1348):
- Index chunks → `VocabularyManager.propose()` (one term per line)
- Bibliography chunks → `AcquisitionManager.flag()` (one citation per line)
- **Individual term/citation errors** → silently caught with `pass` ✅ CONTINUE
  - **⚠️ OBSERVATION**: Silent `except: pass` on line 1392 and 1424. No logging at all for individual failures. If the VocabularyManager or AcquisitionManager tables don't exist, every single term/citation fails silently and you'd never know.

---

## Step 5: Annotate (ingestion_pipeline.py:397-407)

```python
annotator = ChunkAnnotator(settings)
chunks = await annotator.annotate_chunks(chunks, annotation_ctx)
```

**ChunkAnnotator.annotate_chunks()**:
- Groups chunks by source_class (primary, secondary, contextual, other)
- 4 concurrent annotation tasks (one per group)
- Each group processes in batches of 10
- **LLM annotation** (Anthropic API): generates topic, positioning, preceding/following context
- **LLM failure per batch** → falls back to template-only for that batch ✅ CONTINUE
- **No API key** → all chunks get template-only annotations ✅ CONTINUE
- Annotations are **prepended to chunk text before embedding** — this is critical for retrieval quality

**⚠️ OBSERVATION**: Annotation failure is gracefully handled. Template fallback means chunks always get SOME annotation. Good design.

---

## Step 6: Store Chunks in PG (ingestion_pipeline.py:409-456)

```python
# Sort: macro → meso → micro → nano (FK ordering)
for chunk in sorted_chunks:
    pg_id = await storage.chunks.create(chunk_data)
    chunk_id_map[chunk.id] = pg_id
```

- Sorts by granularity before insertion to satisfy `parent_chunk_id` FK constraint
- Each chunk stored individually (not batched)
- Builds `chunk_id_map` mapping in-memory IDs → PG UUIDs
- **Individual chunk create failure** → uncaught ❌ DEAD END
  - **⚠️ ISSUE**: No try/except around `chunks.create()`. A single corrupt chunk (e.g. text too long, invalid UTF-8) kills the ENTIRE pipeline. All prior chunks for this work are already in PG but we never get to embed them.
  - **⚠️ IMPACT**: This is the most likely cause of "partial ingestion" — chunks in PG but no embeddings. The cleanup on re-ingestion (Step 2b) handles this on retry, but the first attempt loses all progress.

- Updates `engagement_passes` on the work record

---

## Step 7: Embed Chunks (ingestion_pipeline.py:458-541)

```python
all_texts = [c.annotated_text for c in chunks]    # annotation + "\n\n" + text
token_batches = build_token_aware_batches(all_texts)
for batch_idx, batch_texts in enumerate(token_batches):
    stored = await self._embed_batch_with_retry(...)
    embeddings_stored += stored
```

**Token-aware batching**:
- `estimate_tokens()`: `word_count × 1.5` heuristic
- `build_token_aware_batches()`: greedy accumulation, max 80K tokens / 128 items per batch
- Single oversized text → gets its own solo batch (API may reject)

**`_embed_batch_with_retry()`** (line 1082):
- **Retry logic**: 2 retries with exponential backoff (2s, 4s)
- **Split-on-failure**: After retries exhausted, splits batch in half and retries each half recursively (max depth 3)
- This isolates a single toxic chunk from killing the whole batch
- **Permanently failed single chunk** → `log.error`, error added to `errors` list ✅ CONTINUE
  - The chunk is stored in PG but never gets an embedding

**Embedding storage**:
- Each vector stored individually via `storage.embeddings.store(pg_id, vector, provider, model, dimensions)`
- **Storage failure per vector** → uncaught within the inner loop (only the embed API call has retry)
  - **⚠️ OBSERVATION**: If the embed API succeeds but `embeddings.store()` fails (PG write error), that vector is lost. Not retried.

**Post-embed verification** (line 505-531):
- If `embeddings_stored < len(all_chunk_ids)`, queries DB for which chunks are missing embeddings
- Records `unembedded_chunk_ids` in the result
- **Verification failure** → falls back to count-based estimate ✅ CONTINUE

---

## Step 8: Upsert Chunk Nodes in Neo4j (ingestion_pipeline.py:543-585)

```python
for chunk in chunks:
    try:
        await storage.graph.upsert_chunk_node(chunk_node)
        neo4j_chunks_synced += 1
    except Exception as exc:
        neo4j_chunk_errors += 1
        # Rate-limited logging: first 3 errors logged, then suppressed
```

- Syncs each chunk as a Neo4j `Chunk` node
- Personal chunks get `user_id` attribute for `USER_REFLECTS_ON` edges
- **Per-chunk failure** → caught, counted, rate-limited logging ✅ CONTINUE
- **All chunks fail** → error appended to errors list, but pipeline continues ✅ CONTINUE
- **⚠️ OBSERVATION (fixed this session)**: Before our fix, there was ZERO error handling here. Any Neo4j exception crashed the pipeline and all subsequent steps (entity extraction, passage linking, etc.) were lost. Now it's non-fatal.

---

## Step 9: Entity Extraction (ingestion_pipeline.py:587-666)

**Only runs for**: `FULL_ENRICHMENT` (primary) and `EMBEDDINGS_AND_GRAPH` (secondary)

```python
# Filter to configured granularities (default: macro,meso,micro)
# Also exclude structural sections (bibliography, index, toc, front_matter)
extraction_chunks = [c for c in chunks if granularity_ok and section_type_ok]

extractor = EntityExtractor(neo4j, api_keys, llm_settings, storage)
extraction_result = await extractor.extract_and_persist(extraction_chunks, work_title, author)
```

**EntityExtractor.extract_and_persist()**:
- Batches of 10 chunks, up to `entity_extraction_concurrency` parallel LLM calls
- Each batch → LLM prompt → JSON response → parse themes/arguments/concepts/persons
- **isinstance guard**: Filters bare strings from entity lists (LLM sometimes returns strings instead of dicts)
- **JSON parse failure** → split batch in half, retry (up to 2 levels)
- **Single chunk permanent failure** → logged, error added to result ✅ CONTINUE
- **Persist failure** (Neo4j write) → per-chunk catch, error added ✅ CONTINUE
- Returns `ExtractionResult(nodes_created, edges_created, errors)`

**Entire extraction block** wrapped in try/except (line 623-642):
- **Catastrophic failure** → `log.error`, error added ✅ CONTINUE

### Step 9b: Theme Deduplication (ingestion_pipeline.py:644-666)

```python
dedup_result = await deduplicate_themes(neo4j, embedding_provider)
```

- Uses embedding-based cosine similarity (threshold 0.85) to merge near-duplicate Theme nodes
- **Failure** → `log.error`, error added ✅ CONTINUE
- **⚠️ OBSERVATION**: This runs on ALL themes in Neo4j, not just the current work's. After every ingestion, it re-deduplicates the entire theme space. This could be slow for large graphs.

---

## Step 10: Passage Linking (ingestion_pipeline.py:668-676)

**Only runs for**: `FULL_ENRICHMENT` (primary) and `EMBEDDINGS_AND_LINKS` (contextual)

```python
link_edges = await self._create_passage_links(chunks, source_class)
```

**`_create_passage_links()`** (line 1178):

**For PRIMARY sources**:
1. Loads all contextual chunks from DB (all contextual works by the same author)
2. Converts DB rows to Chunk objects
3. Runs 3-tier linking:
   - **Tier 1**: `ExplicitLinkDetector` — direct citations
   - **Tier 2**: `ImplicitEngagementDetector` — indirect references (excludes already-linked pairs)
   - **Tier 3**: `ThematicParallelDetector` — embedding-based thematic similarity
4. Each tier → Neo4j edges created

**For CONTEXTUAL sources**:
1. Loads all primary chunks from DB
2. Runs only Tier 1 (explicit citations) against primary chunks
3. **⚠️ OBSERVATION**: Contextual sources only get explicit link detection, not implicit or thematic. This is asymmetric with the primary path.

**For SECONDARY sources**: No passage linking at all (route is `EMBEDDINGS_AND_GRAPH`, not in the route check)

**Failure** → caught at top level (line 670), error added ✅ CONTINUE

**⚠️ OBSERVATION**: Loading ALL contextual/primary chunks into memory for cross-linking could be expensive for large libraries. No pagination or streaming.

---

## Step 11: Personal Source Edges (ingestion_pipeline.py:678-689)

**Only runs for**: `PERSONAL_ENRICHMENT`

- Just logs that personal route was taken
- **⚠️ OBSERVATION**: The code comment says "create USER_REFLECTS_ON edges to targets" but the actual implementation just LOGS. The USER_REFLECTS_ON edges are presumably created elsewhere (maybe via the chunk node's `user_id` attribute from Step 8?). Need to verify this isn't missing functionality.

---

## Step 12: Post-Ingestion Connection Surfacing (ingestion_pipeline.py:691-708)

**Only runs for**: `FULL_ENRICHMENT` (primary) and `EMBEDDINGS_AND_LINKS` (contextual)

```python
surfacing_result = await self._surface_connections(work_id, work_title, work_author)
```

**`_surface_connections()`** (line 764):
- Creates `BatchSurfacer` → calls `surface_after_ingestion()`
- Scans newly ingested work's chunks for cross-work connections
- Filters out already-linked pairs
- Groups by confidence and target work
- Generates PR content if enough connections found
- **Failure** → caught, error added ✅ CONTINUE

---

## Step 13: Post-Ingestion Quality Checks — QG1 (ingestion_pipeline.py:710-723)

```python
quality_checks = await self._run_quality_checks(work_id, source_class, subject_author_id)
```

**`_run_quality_checks()`** (line 800) — 5 checks:

1. **Orphaned entity nodes** (Neo4j) — entities with degree ≤ 1 → DETACH DELETE
   - Records lesson via `record_lesson()` if orphans found
   - **Failure** → warning added ✅ CONTINUE

2. **Classification sanity** — author name matches subject_author but classified as contextual/tertiary
   - **Failure** → warning added ✅ CONTINUE

3. **Chunk noise** — chunks with text < 50 chars
   - **Failure** → warning added ✅ CONTINUE

4. **Embedding coverage** — compares chunk count vs embedding count in PG
   - **Failure** → warning added ✅ CONTINUE

5. **Entity coverage** — chunks in Neo4j with zero entity edges
   - Warns if > 10% of chunks lack entities
   - **Failure** → warning added ✅ CONTINUE

**⚠️ OBSERVATION**: QG1 DELETES orphaned entity nodes (Check 1). This is a destructive action during a "quality check". If the entity extraction is still in progress or was partially successful, this could delete valid entities that just haven't been fully linked yet. Since entity extraction (Step 9) runs synchronously before QG1 (Step 13), this should be safe — but if the pipeline is ever made async, this is a race condition.

**Entire QG1 block** wrapped in try/except (line 712-723):
- **Failure** → error added ✅ CONTINUE

---

## Return: IngestionResult (ingestion_pipeline.py:750-762)

Pipeline returns `IngestionResult` with:
- `work_id`, `source_class`, `processing_route`
- `chunks_by_granularity` (dict)
- `embeddings_stored` (int)
- `entity_count`, `edge_count` (ints)
- `errors` (list of non-fatal error messages)
- `total_chunks` (int)
- `unembedded_chunk_ids` (list)
- `quality_checks` (dict)

---

## Back in `handle_ingest_book()` — Post-Pipeline Steps

### Cross-Work Analysis (ingest.py:102-112)

**Only for PRIMARY sources**:

```python
cross_work_summary = await _run_cross_work_analysis(subject_author_id, settings, storage, embedding_provider)
```

**`_run_cross_work_analysis()`** (ingest.py:379-474) — 3 steps:

1. **Thematic Index Generation** — `ThematicIndexGenerator.generate()`
   - **Failure** → caught, error in summary ✅ CONTINUE

2. **Voice Profile Extraction** — `VoiceProfileExtractor.extract()` + `VoiceProfileManager.store_profile()`
   - Only primary sources contribute to voice profiles (voice contamination rule)
   - **Failure** → caught, error in summary ✅ CONTINUE

3. **Thematic Evolution Analysis** — `ThematicEvolutionAnalyzer.analyze()`
   - Only runs if themes were generated in step 1
   - Re-fetches themes from repository
   - **Failure** → caught, error in summary ✅ CONTINUE

### Cache Invalidation (ingest.py:114-116)

- If `cache_manager` provided, calls `invalidate_on_ingestion(author_id)`
- **⚠️ OBSERVATION**: Cache manager is optional. If not wired up, stale cache entries persist.

### Quality Gate Enqueue — QG2 (ingest.py:118-128)

- If `task_queue` provided, enqueues async quality gate via arq/Redis
- **Enqueue failure** → `log.warning` ✅ CONTINUE (non-blocking)
- **⚠️ OBSERVATION**: Task queue is optional. If Redis is down or not configured, QG2 never runs.

**QG2 task** (`tasks.py:330`) runs asynchronously:
1. Theme deduplication (again — duplicates QG1 Step 9b and QG1 if themes were already deduped)
2. PG-Neo4j consistency check + structural backfill
3. Cross-work passage re-linking
4. Entity extraction coverage audit

### Post-Ingestion Backup (ingest.py:131)

- Same fire-and-forget pattern as pre-ingestion backup
- **Failure** → logged, non-blocking ✅ CONTINUE

### Post-Ingestion Report (ingest.py:137-150)

```python
report = await generate_ingestion_report(work_id, storage, pipeline_result, settings, embedding_provider)
```

**`ingestion_report.py`** — Self-healing report:
- Queries live DB for actual counts (chunks, embeddings, entities, graph sync)
- **Auto-heal** (if `auto_heal=True`, which is default):
  1. **Embedding gaps** → backfills missing embeddings inline
  2. **PG/Neo4j chunk mismatch** → `backfill_work_graph()` to sync missing chunks
  3. **Missing entity extraction** → `_run_entity_extraction_for_work()` from graph/backfill
- Re-queries counts after healing
- Writes report to `/home/marty/parlour-backups/ingestion-reports/`
- **Failure** → `log.warning` ✅ CONTINUE (non-blocking)

### Final Response (ingest.py:133-152)

- Serializes `IngestionResult.to_dict()` + cross_work_summary + ingestion_report to JSON
- Returns to MCP client

---

## Error Handling Summary

| Step | Error Handling | Severity |
|------|---------------|----------|
| Input validation | IngestionError raised | Fatal — correct |
| Backup (pre/post) | Fire-and-forget | Non-fatal — correct |
| Step 0: Booklore | Graceful degradation | Non-fatal — correct |
| Step 1: Parse | Uncaught propagation | Fatal — correct |
| Step 2: Classify | Uncaught propagation | Fatal — correct |
| Step 2b: Re-ingest cleanup (Neo4j) | try/except, warning | Non-fatal — correct |
| Step 2c: Work/Author upsert (Neo4j) | **UNCAUGHT** | Fatal — ⚠️ inconsistent |
| Step 4: Chunk | Uncaught propagation | Fatal — debatable |
| Step 4b: Section routing | Silent `except: pass` | Non-fatal — ⚠️ too silent |
| Step 5: Annotate | LLM fallback to template | Non-fatal — good |
| Step 6: Store chunks (PG) | **UNCAUGHT per chunk** | Fatal — ⚠️ ISSUE |
| Step 7: Embed | Retry + split + fallback | Non-fatal — excellent |
| Step 8: Neo4j chunk sync | Per-chunk catch, rate-limited | Non-fatal — good (new) |
| Step 9: Entity extraction | Per-batch + top-level catch | Non-fatal — good |
| Step 9b: Theme dedup | Caught | Non-fatal — good |
| Step 10: Passage linking | Caught | Non-fatal — good |
| Step 12: Surfacing | Caught | Non-fatal — good |
| Step 13: QG1 | Per-check catch | Non-fatal — good |
| Cross-work analysis | Per-step catch | Non-fatal — good |
| QG2 enqueue | Caught | Non-fatal — good |
| Ingestion report | Caught | Non-fatal — good |

---

## Issues Found (DO NOT FIX — for review only)

### Critical

1. **Step 6: No per-chunk error handling for PG storage** (ingestion_pipeline.py:420-446)
   - A single corrupt chunk kills the pipeline after chunks are partially stored
   - Embeddings, entity extraction, passage linking all lost for the entire work
   - Fix: wrap `chunks.create()` in try/except, skip corrupt chunks, log error

### Moderate

2. **Step 2c: Inconsistent Neo4j error handling** (ingestion_pipeline.py:310-339)
   - Work node and Author node upserts have NO try/except
   - Step 8 (chunk upsert) has per-item try/except
   - If Neo4j is briefly down during Step 2c, entire pipeline dies
   - Fix: wrap in try/except like Step 8, or at minimum log and continue

3. **Step 4b: Silent error swallowing** (ingestion_pipeline.py:1392, 1424)
   - VocabularyManager.propose() and AcquisitionManager.flag() failures caught with bare `except: pass`
   - If tables don't exist or DB is misconfigured, every term/citation silently fails
   - Fix: at minimum `log.debug` or count failures

4. **Step 11: Misleading comment about USER_REFLECTS_ON edges** (ingestion_pipeline.py:678-689)
   - Code comment says "create USER_REFLECTS_ON edges" but Step 11 only logs
   - **Pass 2 verified**: This is BY DESIGN — USER_REFLECTS_ON edges are created by `create_personal_reflection()` in GraphRepository, called from the Socratic synthesis module when users interact (not during ingestion)
   - Step 8 sets `user_id` on personal chunk nodes, which enables later edge creation
   - The comment is misleading, not the code

5. **Step 9b: Global theme dedup** (ingestion_pipeline.py:644-666)
   - Runs on ALL themes, not scoped to current work
   - Could become slow as theme count grows
   - Also duplicated in QG2 (tasks.py:362)

### Minor

6. **Step 7: Embedding store failure not retried** (ingestion_pipeline.py:1117-1123)
   - embed API call has retry, but `embeddings.store()` PG write does not
   - If PG hiccups during store, that vector is lost permanently

7. **Cross-work analysis: voice profile on every PRIMARY ingestion** (ingest.py:417-434)
   - Re-extracts entire voice profile on every single book ingestion
   - For an author with 10 books, ingesting book #10 re-processes all 10
   - Consider: only update if voice profile is outdated or new primary chunks added

8. **QG2 duplicates QG1 work** (tasks.py:362 vs ingestion_pipeline.py:644)
   - Theme dedup runs in Step 9b (inline) AND in QG2 (async)
   - The second run is usually a no-op but still queries Neo4j

9. **Passage linking loads all chunks into memory** (ingestion_pipeline.py:1193-1220)
   - For large libraries with many contextual/primary works, this could OOM
   - No pagination or streaming

---

## Flow Diagram

```
handle_ingest_book()
│
├── Validate inputs (file_path, subject_author_id, file exists)
│   └── ❌ IngestionError if missing
│
├── auto_confirm=False?
│   └── ⏹️ _classify_only() → return preview JSON
│
├── _run_pre_ingest_backup() [fire-and-forget]
│
├── IngestionPipeline.ingest()
│   │
│   ├── Step 0: Booklore metadata [optional, graceful]
│   │
│   ├── Step 1: Parse file → ParsedDocument
│   │   └── ❌ ParsingError if unsupported/corrupt
│   │
│   ├── _ingest_from_document()
│   │   │
│   │   ├── Step 2: Classify → CatalogEntry + ProcessingRoute
│   │   │   └── ❌ Uncaught if LLM/DB fails
│   │   │
│   │   ├── Step 2b: Re-ingestion cleanup (PG + Neo4j)
│   │   │   └── Neo4j failure: ⚠️ warning, continue
│   │   │
│   │   ├── Step 2c: Upsert Work + Author (PG + Neo4j)
│   │   │   └── ❌ Uncaught if Neo4j fails
│   │   │
│   │   ├── Step 3: TERTIARY? → ⏹️ return (metadata only)
│   │   │
│   │   ├── Step 4: Chunk (genre-aware strategy)
│   │   │   └── ❌ Uncaught if chunking fails
│   │   │
│   │   ├── Step 4b: Section-type filter + structural routing
│   │   │   └── Individual routing errors: silently swallowed
│   │   │
│   │   ├── Step 5: Annotate (LLM + template fallback)
│   │   │   └── LLM failure: falls back to template ✅
│   │   │
│   │   ├── Step 6: Store chunks in PG
│   │   │   └── ❌ Uncaught per-chunk — kills pipeline
│   │   │
│   │   ├── Step 7: Embed (retry + split + fallback)
│   │   │   └── Per-chunk permanent failure: logged ✅
│   │   │
│   │   ├── Step 8: Neo4j chunk sync
│   │   │   └── Per-chunk failure: caught, rate-limited ✅
│   │   │
│   │   ├── Step 9: Entity extraction [PRIMARY/SECONDARY]
│   │   │   └── Failure: caught ✅
│   │   │
│   │   ├── Step 9b: Theme dedup [PRIMARY/SECONDARY]
│   │   │   └── Failure: caught ✅
│   │   │
│   │   ├── Step 10: Passage linking [PRIMARY/CONTEXTUAL]
│   │   │   └── Failure: caught ✅
│   │   │
│   │   ├── Step 11: Personal edges [PERSONAL]
│   │   │   └── Just logs (⚠️ edges not actually created?)
│   │   │
│   │   ├── Step 12: Connection surfacing [PRIMARY/CONTEXTUAL]
│   │   │   └── Failure: caught ✅
│   │   │
│   │   └── Step 13: QG1 quality checks
│   │       └── Per-check failure: caught ✅
│   │
│   └── return IngestionResult
│
├── Cross-work analysis [PRIMARY only]
│   ├── Thematic index generation
│   ├── Voice profile extraction
│   └── Thematic evolution analysis
│   └── Each: failure caught ✅
│
├── Cache invalidation [optional]
│
├── QG2 enqueue [optional, async via arq/Redis]
│   └── Failure: warning ✅
│
├── _run_post_ingest_backup() [fire-and-forget]
│
├── generate_ingestion_report() [auto-heal]
│   └── Failure: warning ✅
│
└── return JSON response
```

---

## Processing Routes Summary

| Route | Steps Executed | LLM Calls |
|-------|---------------|-----------|
| FULL_ENRICHMENT (primary) | 0-13 + cross-work + QG2 | Classification, Annotation, Entity Extraction, Voice Profile, Thematic Index, Evolution |
| EMBEDDINGS_AND_GRAPH (secondary) | 0-9b, 13 | Classification, Annotation, Entity Extraction |
| EMBEDDINGS_AND_LINKS (contextual) | 0-8, 10, 12-13 | Classification, Annotation |
| METADATA_ONLY (tertiary) | 0-3 | Classification |
| PERSONAL_ENRICHMENT (personal) | 0-8, 11, 13 | Classification, Annotation |

---

## Composable Pipeline Path (Alternative)

The composable pipeline (`composable_ingestion.py`) exposes the same steps as individual MCP tools:

```
classify_source → catalog_source → chunk_source → detect_passage_links → flag_acquisition
```

Each tool can be called independently with human review between steps. The internal implementation mirrors Steps 2-10 from the monolithic pipeline but with user confirmation gates between each stage. Same error handling patterns apply (including the same Step 6 vulnerability).

---

---

## Pass 2/3 Consolidated Findings

### Verified from Pass 2

- **Issue #4 reclassified**: USER_REFLECTS_ON edges are NOT missing — they're created by `create_personal_reflection()` in `storage/repositories.py:944` via the Socratic synthesis module (`synthesis/socratic.py:164`). The pipeline correctly sets `user_id` on personal chunk nodes in Step 8 (line 556) to enable later edge creation. **Issue is a misleading comment, not a bug.**

- **Issue #1 confirmed**: Step 6 (`chunks.create()` at line 445) has zero error handling in BOTH `ingestion_pipeline.py` AND `composable_ingestion.py` (line 570). Same vulnerability in both paths.

- **Step 2c inconsistency confirmed**: `upsert_work_node()` (line 311) and `neo4j.execute_write()` for author (line 328) are both uncaught. But `upsert_chunk_node()` in Step 8 (line 558) is wrapped in try/except. The inconsistency is real.

### Found in Pass 3

- **Epistolary script gap**: `scripts/ingest_epistolary.py` calls `pipeline.ingest_document()` directly, bypassing `handle_ingest_book()`. This means:
  - No pre/post-ingestion backup
  - No cross-work analysis (voice profile, thematic index, evolution)
  - No cache invalidation
  - No QG2 async quality gate
  - No post-ingestion report (auto-heal)
  - Entity extraction and QG1 still run (they're inside the pipeline)
  - **Impact**: Epistolary-ingested works need manual follow-up for cross-work analysis and backups

- **`_route_structural_sections` bare except**: Lines 1392 and 1424 use `except Exception: pass` (vocabulary) and `except Exception: already_flagged += 1` (acquisition). The vocabulary path swallows ALL errors with no logging. The acquisition path counts errors as "already_flagged" which misreports the count.

### Priority Ranking (consolidated across all 3 passes)

| # | Issue | Severity | Fix Complexity | Impact |
|---|-------|----------|----------------|--------|
| 1 | Step 6: No per-chunk PG storage error handling | **Critical** | Low (add try/except) | Partial ingestion → lost work |
| 2 | Step 2c: Uncaught Neo4j errors for Work/Author upsert | Moderate | Low | Pipeline crash if Neo4j briefly down |
| 3 | Silent error swallowing in structural routing | Moderate | Low (add logging) | Hidden failures |
| 4 | Composable pipeline has same Step 6 vulnerability | Moderate | Low | Both code paths affected |
| 5 | Global theme dedup after every ingestion | Moderate | Medium | Performance at scale |
| 6 | Epistolary script bypasses post-pipeline hooks | Moderate | Medium | Missing backups/analysis |
| 7 | Embedding store failure not retried | Minor | Low | Rare vector loss |
| 8 | Voice profile re-extraction on every PRIMARY | Minor | Medium | Unnecessary LLM cost |
| 9 | QG2 duplicates QG1 theme dedup | Minor | Low | Wasted computation |
| 10 | Passage linking loads all chunks into memory | Minor | High | OOM risk at scale |

---

*This audit was produced by reading the actual code in ingestion_pipeline.py, ingest.py, composable_ingestion.py, repositories.py, entity_extraction.py, and supporting modules across 3 passes. Issues noted are observations only — no fixes applied.*
