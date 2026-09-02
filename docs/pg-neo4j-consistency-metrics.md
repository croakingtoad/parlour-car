# PostgreSQL / Neo4j consistency metrics

These metrics select different populations. Equal totals are not evidence that
the same chunks exist in both stores, and absence of an entity edge is not
evidence that a chunk is stale.

## Current shipped behavior

| Metric | Population | Meaning | Correct remedy |
|---|---|---|---|
| `audit_library.pg_neo4j.is_consistent` | Work-ID equality plus normalized chunk-ID equality for every work | `true` only when both stores contain the same works and chunk identities | Use the directional remedies reported in `chunk_delta`; there is no count-only clean result anymore |
| `audit_library.pg_neo4j.chunk_delta` | Per-work PG-only and Neo4j-only normalized chunk IDs | Actual directional store drift, with counts and samples capped at 20 IDs | PG-only: the default graph/entity backfill; Neo4j-only: reviewed dry-run and explicit approval before cleanup |
| `audit_library.works[*].neo4j_chunks_without_entity_edges` | Neo4j chunks with none of five outgoing entity-edge types | Entity-extraction coverage signal, not PG/Neo4j drift and not proof of a defect | Investigate first; only confirmed incomplete extraction warrants the default non-destructive backfill |

The legacy `orphaned_neo4j_chunks` key remains as an alias for
`neo4j_chunks_without_entity_edges` for compatibility. Its old name must not be
interpreted as “absent from PostgreSQL.”

## 1. Identity-aware consistency verdict

**Source:** `src/author_library/graph/backfill.py:381-460`, packaged and turned
into recommendations by `src/author_library/tools/meta.py:537-589`.

The function reads work IDs and counts, then fetches chunk identities with:

```sql
SELECT work_id, array_agg(id::text) AS chunk_ids FROM chunks GROUP BY work_id
```

```cypher
MATCH (c:Chunk)
RETURN c.work_id AS work_id, collect(c.chunk_id) AS chunk_ids
```

UUID-shaped IDs are normalized before comparison because PostgreSQL stores
hyphenated UUID strings while some Neo4j nodes contain 32-character hex. The
per-work `in_sync` value requires both equal counts and empty set differences.
The top-level `is_consistent` value requires equal work-ID sets and every work
to be in sync. Equal-and-opposite drift therefore cannot cancel out.

Each `chunk_delta` row includes `pg_only_chunk_count`,
`neo4j_only_chunk_count`, and deterministic samples of each store's original ID
representation. The samples are bounded by `CHUNK_ID_SAMPLE_LIMIT`; the counts
represent the complete sets.

**Remedies:**

- PG-only IDs require additive graph repair. Run the exact default command
  emitted by the audit, for example
  `uv run python scripts/backfill_graph_and_entities.py <work-id>`. Its actual
  default execution plan contains only chunk sync and entity extraction
  (`scripts/backfill_graph_and_entities.py:23-54,84-109`). Theme deduplication
  is a separate `--deduplicate-themes` opt-in and audit recommendations
  explicitly prohibit that flag.
- Neo4j-only IDs are candidates for removal only after the cleanup scope is
  corrected and its dry-run is reviewed. The cleanup predicate compares the
  normalized ID sets (`scripts/cleanup_neo4j_orphans.py:46-48,89-103`), but its
  execute path deletes graph data (`scripts/cleanup_neo4j_orphans.py:51-67`).
  No deletion is justified or authorized by the audit alone.

## 2. Neo4j chunks without entity edges

**Source:** query and grouping at `src/author_library/tools/meta.py:460-481`,
warning classification at `src/author_library/tools/meta.py:513-532`, and
recommendation construction at `src/author_library/tools/meta.py:684-699`.

```cypher
MATCH (c:Chunk)
WHERE NOT (c)-[:EXPLORES_THEME|MAKES_ARGUMENT|ATTRIBUTED_BY_CRITIC|CONCEPT_USED_IN|REFERENCES_PERSON]->()
RETURN c.work_id AS work_id, COUNT(c) AS orphan_count
```

This selects Neo4j `Chunk` nodes with no outgoing relationship of the five
listed entity types. It does not test whether the chunk exists in PostgreSQL,
whether a PostgreSQL chunk is missing from Neo4j, incoming relationships,
other outgoing relationship types, or whether an extraction correctly found
no entities.

Consequently this metric alone establishes no defect and never justifies
deletion. The audit recommends investigation. If a separate review confirms
that extraction is incomplete, its emitted graph/entity backfill command uses
the non-destructive default plan described above and explicitly omits
`--deduplicate-themes`.

### Warning-threshold decision record

The warning fires at **at least 10 uncovered chunks or at least 10% of a
work's chunks** (`src/author_library/tools/meta.py:26-38`). The absolute limb
keeps a large work's sizable uncovered population visible even below 10%; the
percentage limb keeps a smaller work's broad coverage gap visible below 10
chunks.

No empirical corpus study or product decision was recorded for either number
when these thresholds were introduced. They are provisional alerting
heuristics, not evidence that the selected chunks are defective. Changes to
the values require a measured false-positive/false-negative review or an
explicit product decision; this record must not invent a stronger rationale.

## 3. Directional identity set differences

The historical `/parlour-booklist` check compared all PostgreSQL chunk IDs with
all Neo4j chunk IDs. The shipped `chunk_delta` metric now performs the same
identity comparison per work inside `check_pg_neo4j_consistency`, so the audit
itself exposes both directions and uses them in `is_consistent`.

**Source:** normalization and indexing at
`src/author_library/graph/backfill.py:35-49`; identity queries and set
differences at `src/author_library/graph/backfill.py:413-450`; report assembly
at `src/author_library/graph/backfill.py:453-460`.

**Population:**

- `pg_only_chunk_count`: current PostgreSQL chunk IDs absent from Neo4j.
- `neo4j_only_chunk_count`: Neo4j chunk IDs absent from PostgreSQL.

This does not measure entity-edge coverage, chunk text/property equality,
relationship equality, or duplicate IDs collapsed within one store's set.
Its remedies are the directional remedies in section 1.

## Dated observations

The following observations are historical evidence, not definitions of the
current implementation:

- **Pre-fix, 2026-09-02 20:37:50–20:39:00 UTC:** the count-only audit returned
  `is_consistent: true` and `chunk_delta: []` while an independent identity
  comparison found 34 PG-only and 34 Neo4j-only IDs. This was the defect that
  motivated the identity-aware implementation.
- **Tier 2 audit, 2026-09-02 21:19 UTC:** the shipped identity-aware branch
  returned `is_consistent: false`, 34 PG-only IDs, and 35 Neo4j-only IDs. The
  additional Neo4j-only ID belonged to the suite-created
  `coleridge--biographia` work. The test fixture is now namespaced as
  `test--coleridge-biographia`; no cleanup or deletion was performed as part of
  that fix.

## Safety verdict

The entity-edge population and the Neo4j-only identity population overlap only
incidentally; neither is a substitute for the other. Routing
`neo4j_chunks_without_entity_edges` to `cleanup_neo4j_orphans.py` would be
incorrect and potentially destructive because most such chunks may still have
valid PostgreSQL rows. Cleanup may be considered only for the independently
measured Neo4j-only identity set, after a reviewed dry-run and explicit
deletion approval.
