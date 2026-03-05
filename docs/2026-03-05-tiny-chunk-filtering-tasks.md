# Tiny Chunk Filtering — Task Breakdown

**Goal:** Eliminate noise chunks (index headwords, bibliography fragments, footnote orphans, single-char fragments) from the composable ingestion MCP path and existing database, and store `section_type` in chunk metadata for search-time filtering.

**Approach:** The `filter_min_chunk_size()` and `_filter_by_section_type()` functions already exist but are only wired into the `IngestionPipeline` class path — the composable MCP tool `handle_chunk_source()` in `composable_ingestion.py` bypasses both. Wire the filters into the composable path, enrich stored chunk metadata with `section_type`, and clean existing noisy data. TDD throughout.

**Skills:** `@skills/td-task-management`

**Tech Details:** Python 3.13, asyncpg, pytest, structlog, Pydantic. Parlour Car submodule at `parlour-car/`.

---

### Task 1: Wire section_type filter into handle_chunk_source()

The composable ingestion MCP tool `handle_chunk_source()` runs the full chunking pipeline but skips the section_type filter that `IngestionPipeline.ingest()` applies at step 4b (line 290). This means MCP-driven ingestion keeps bibliography, index, ToC, and front-matter chunks.

**Files:**
- Modify: `src/author_library/tools/composable_ingestion.py:482-495` (after `strategy.chunk()`, before annotate)
- Test: `tests/test_tools/test_composable_ingestion.py`

**Step 1: Write the failing test**

```python
# tests/test_tools/test_composable_ingestion.py — add new test class

class TestChunkSourceSectionTypeFilter:
    """Verify handle_chunk_source filters non-content section types."""

    @staticmethod
    def _make_index_chunk(work_id: str = "test--work") -> Chunk:
        from author_library.chunking.models import Chunk, ChunkGranularity
        return Chunk(
            text="imagination as active 166-7, 172",
            granularity=ChunkGranularity.MICRO,
            work_id=work_id,
            source_class="primary",
            position=0,
            section_type="index",
            chapter="Index",
        )

    @staticmethod
    def _make_content_chunk(work_id: str = "test--work") -> Chunk:
        from author_library.chunking.models import Chunk, ChunkGranularity
        return Chunk(
            text="The imagination, far from being a merely subjective realm of fantasy, is an essential instrument with which we grasp the truth.",
            granularity=ChunkGranularity.MESO,
            work_id=work_id,
            source_class="primary",
            position=0,
            section_type="chapter",
        )
```

Actual test assertions depend on the existing test harness patterns in the file. The key assertion: after `handle_chunk_source()` runs, stored chunks should NOT include any with `section_type` in `("index", "bibliography", "toc", "front_matter")`.

**Step 2: Run test to verify it fails**

Run: `cd parlour-car && python -m pytest tests/test_tools/test_composable_ingestion.py -k "section_type" -v --timeout=30`
Expected: FAIL — no section_type filter applied

**Step 3: Implement the fix**

In `src/author_library/tools/composable_ingestion.py`, add a section_type filter after the chunking step (line 483) and before annotation (line 498). Import `SectionType` from `author_library.parsing.models` and reuse the same content-section logic as `IngestionPipeline._filter_by_section_type()`:

```python
# After line 483: chunks = strategy.chunk(document, work_id, source_class_str)
# Add section-type filtering (matches IngestionPipeline._filter_by_section_type)
_CONTENT_SECTION_TYPES = {"chapter", "preface", "back_matter"}
pre_filter_count = len(chunks)
skipped_sections: dict[str, int] = {}
content_chunks: list[Chunk] = []
for chunk in chunks:
    if chunk.section_type in _CONTENT_SECTION_TYPES:
        content_chunks.append(chunk)
    else:
        skipped_sections[chunk.section_type] = skipped_sections.get(chunk.section_type, 0) + 1

if skipped_sections:
    log.info(
        "chunk_source_section_type_filter",
        work_id=work_id,
        original_chunks=pre_filter_count,
        kept_chunks=len(content_chunks),
        skipped_by_type=skipped_sections,
    )
chunks = content_chunks
```

Consider extracting the `_CONTENT_SECTION_TYPES` set and the filter logic into a shared function in a common location (e.g. `author_library/chunking/filters.py` or alongside the `SectionType` enum in `parsing/models.py`) so both `IngestionPipeline` and `handle_chunk_source` use the same code. DRY.

**Step 4: Run test to verify it passes**

Run: `cd parlour-car && python -m pytest tests/test_tools/test_composable_ingestion.py -k "section_type" -v --timeout=30`
Expected: PASS

**Step 5: Run full suite**

Run: `cd parlour-car && python -m pytest tests/ -x --timeout=30 -q`
Expected: All pass, 0 failures

---

### Task 2: Store section_type in chunk metadata jsonb

When chunks are stored in PostgreSQL, the `section_type` field from the `Chunk` model is silently dropped — it's not included in the `chunk_data` dict passed to `storage.chunks.create()`. This means search-time queries cannot filter by section_type. The fix: inject `section_type` into the `metadata` dict before storage.

**Files:**
- Modify: `src/author_library/tools/composable_ingestion.py:515-529` (the chunk_data dict construction)
- Modify: `src/author_library/tools/ingestion_pipeline.py` (same pattern in the IngestionPipeline storage loop)
- Test: `tests/test_tools/test_composable_ingestion.py`

**Step 1: Write the failing test**

```python
# A test that creates a chunk via the pipeline and then verifies
# the stored metadata jsonb contains "section_type": "chapter"
```

The assertion: after chunk storage, `SELECT metadata FROM chunks WHERE work_id = $1` should contain `{"section_type": "chapter"}` (or whatever the section_type was).

**Step 2: Run test to verify it fails**

Run: `cd parlour-car && python -m pytest tests/test_tools/test_composable_ingestion.py -k "metadata_section_type" -v --timeout=30`
Expected: FAIL — metadata does not contain section_type

**Step 3: Implement**

In `composable_ingestion.py` at line 525 (the `chunk_data` construction), inject `section_type` into the metadata dict:

```python
# Before: "metadata": chunk.metadata,
# After:
metadata = dict(chunk.metadata)
metadata["section_type"] = chunk.section_type
# ... then use metadata in chunk_data
"metadata": metadata,
```

Do the same in `ingestion_pipeline.py` wherever chunks are stored to PG (search for the chunk storage loop).

**Step 4: Run tests**

Run: `cd parlour-car && python -m pytest tests/ -x --timeout=30 -q`
Expected: All pass

---

### Task 3: Add SQL migration to backfill section_type for existing chunks

Existing chunks in the database have no `section_type` in their metadata jsonb. We can infer it from the `chapter` column:
- `chapter ILIKE '%index%'` → `"index"`
- `chapter ILIKE '%bibliograph%'` or `chapter ILIKE '%works cited%'` or `chapter ILIKE '%references%'` or `chapter ILIKE '%further reading%'` → `"bibliography"`
- `chapter ILIKE '%contents%'` → `"toc"`
- `chapter ILIKE '%copyright%'` or `chapter ILIKE '%dedication%'` or `chapter ILIKE '%acknowledgement%'` → `"front_matter"`
- `chapter ILIKE '%preface%'` or `chapter ILIKE '%foreword%'` or `chapter ILIKE '%introduction%'` → `"preface"`
- Everything else → `"chapter"`

**Files:**
- Create: `src/author_library/storage/migrations/010_backfill_section_type.sql`
- Modify: `tests/test_storage/test_migrations.py` (update expected migration count)

**Step 1: Write the migration**

```sql
-- 010_backfill_section_type.sql
-- Backfill section_type into chunk metadata for pre-existing chunks.

UPDATE chunks
SET metadata = jsonb_set(
    COALESCE(metadata, '{}'::jsonb),
    '{section_type}',
    CASE
        WHEN chapter ILIKE '%index%' THEN '"index"'
        WHEN chapter ILIKE '%bibliograph%' OR chapter ILIKE '%works cited%'
             OR chapter ILIKE '%references%' OR chapter ILIKE '%further reading%' THEN '"bibliography"'
        WHEN chapter ILIKE '%contents%' THEN '"toc"'
        WHEN chapter ILIKE '%copyright%' OR chapter ILIKE '%dedication%'
             OR chapter ILIKE '%acknowledgement%' THEN '"front_matter"'
        WHEN chapter ILIKE '%preface%' OR chapter ILIKE '%foreword%' THEN '"preface"'
        ELSE '"chapter"'
    END::jsonb
)
WHERE metadata IS NULL
   OR NOT (metadata ? 'section_type');
```

**Step 2: Run migration**

Run: `cd parlour-car && python -m pytest tests/test_storage/test_migrations.py -v --timeout=30`
Expected: PASS (migration count updated)

**Step 3: Verify backfill against real data**

```bash
cd parlour-car && uv run python -c "
import asyncio, asyncpg
async def main():
    conn = await asyncpg.connect('postgresql://author_library:author_library@localhost:5432/author_library')
    r = await conn.fetchrow('''SELECT count(*) as total,
        count(*) filter (where metadata->>\'section_type\' is not null) as has_type
        FROM chunks''')
    print(f'section_type coverage: {r[\"has_type\"]}/{r[\"total\"]}')
    await conn.close()
asyncio.run(main())
"
```

Expected: 100% coverage after migration runs

---

### Task 4: Delete existing noise chunks from database

With section_type now backfilled, delete the noise chunks that should have been filtered during ingestion. These are micro/nano chunks under 50 chars from non-content sections, plus single-character fragments.

**Files:**
- Create: `src/author_library/storage/migrations/011_delete_noise_chunks.sql`
- Modify: `tests/test_storage/test_migrations.py` (update expected migration count)

**Step 1: Write the migration**

```sql
-- 011_delete_noise_chunks.sql
-- Remove noise micro/nano chunks that predate the section_type and min_chunk_size filters.

-- Delete chunks from non-content sections (index, bibliography, toc, front_matter)
DELETE FROM chunks
WHERE granularity IN ('micro', 'nano')
  AND metadata->>'section_type' IN ('index', 'bibliography', 'toc', 'front_matter');

-- Delete remaining tiny fragments (single chars, footnote orphans, section markers)
DELETE FROM chunks
WHERE granularity IN ('micro', 'nano')
  AND length(text) < 50;
```

**Step 2: Verify counts before/after**

```bash
cd parlour-car && uv run python -c "
import asyncio, asyncpg
async def main():
    conn = await asyncpg.connect('postgresql://author_library:author_library@localhost:5432/author_library')
    before = await conn.fetchrow('SELECT count(*) as c FROM chunks WHERE granularity IN (\'micro\', \'nano\') AND length(text) < 50')
    print(f'Noise chunks to delete: {before[\"c\"]}')
    await conn.close()
asyncio.run(main())
"
```

**Step 3: Run migration and verify**

Run the migration runner, then verify noise is gone:

```bash
cd parlour-car && uv run python -c "
import asyncio, asyncpg
async def main():
    conn = await asyncpg.connect('postgresql://author_library:author_library@localhost:5432/author_library')
    r = await conn.fetchrow('SELECT count(*) as c FROM chunks WHERE granularity IN (\'micro\', \'nano\') AND length(text) < 50')
    print(f'Remaining noise chunks: {r[\"c\"]}')
    r2 = await conn.fetchrow('SELECT count(*) as c FROM chunks')
    print(f'Total chunks: {r2[\"c\"]}')
    await conn.close()
asyncio.run(main())
"
```

Expected: 0 remaining noise chunks.

**Step 4: Run full test suite**

Run: `cd parlour-car && python -m pytest tests/ -x --timeout=30 -q`
Expected: All pass

---

### Task 5: Integration test — end-to-end chunk_source with noise filtering

Write an integration test that proves the full composable path now filters noise correctly. This test exercises the complete chain: parse → chunk → filter → store.

**Files:**
- Modify: `tests/test_tools/test_composable_ingestion.py`

**Step 1: Write the test**

Create a test that:
1. Mocks a parsed document with chapters including an "Index" and a "Bibliography" section
2. Calls `handle_chunk_source()`
3. Asserts that stored chunks contain NO index/bibliography section_type
4. Asserts that all stored micro chunks are >= 50 chars
5. Asserts that stored chunk metadata contains `section_type`

**Step 2: Run test**

Run: `cd parlour-car && python -m pytest tests/test_tools/test_composable_ingestion.py -k "integration_noise_filter" -v --timeout=30`
Expected: PASS

**Step 3: Run full suite**

Run: `cd parlour-car && python -m pytest tests/ -x --timeout=30 -q`
Expected: All pass, 0 regressions

---

## Verification Checklist

After all tasks:

- [ ] `handle_chunk_source()` applies section_type filter (same as `IngestionPipeline`)
- [ ] All stored chunks have `section_type` in metadata jsonb
- [ ] Existing DB chunks backfilled with section_type
- [ ] Noise chunks (tiny fragments, index headwords, bib entries) deleted from DB
- [ ] Full test suite passes
- [ ] No mock data, no placeholder implementations
